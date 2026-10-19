#!/usr/bin/env python3
"""Shared GPT harness episode execution and summary utilities."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

import harness_utils as base
from harness_models import checked_model, model_matches

ARMS = ['without_skills', 'with_skills']
NODE = '192.168.48.153'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_public_instruction(observation):
    value = observation.get('instruction')
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Official public instruction is missing; this is an input transport error, not a scored task failure')
    return value


def inventory(sim):
    directory = sim / 'Assets/Eval_Layout/RoboDojo/arx_x5/0'
    files = {int(p.stem.rsplit('_', 1)[1]): p for p in directory.iterdir()
             if re.fullmatch(r'align_blocks_\d+\.json', p.name)}
    if set(files) != set(range(55)):
        raise ValueError('Expected exactly the official seed-0 align_blocks layouts 0..54')
    return [{'layout_id': i, 'layout_file': str(files[i]), 'layout_sha256': digest(files[i])}
            for i in range(55)]




def aggregate(cases, rows, status):
    result = {'schema': 'awomopaw.gpt6-harness-skills-ablation/v1', 'status': status, 'updated_at': base.now(),
        'task': 'align_blocks', 'seed': 0, 'requested_layouts': len(cases),
        'requested_episodes': 2*len(cases), 'finished': len(rows), 'arms': {},
        'execution': 'gpt6_direct_harness', 'frozen_controller_executed': False}
    for arm in ARMS:
        selected = [r for r in rows if r['arm'] == arm]
        valid = [r for r in selected if r.get('valid_episode')]
        successes = sum(bool(r['success']) for r in valid)
        result['arms'][arm] = {'requested': len(cases), 'finished': len(selected),
            'valid_episodes': len(valid), 'successes': successes,
            'success_rate_over_valid_episodes': successes/len(valid) if valid else None,
            'mean_official_score': sum(r['score'] for r in valid)/len(valid) if valid else None,
            'unscored_or_invalid': len(selected)-len(valid),
            'complete_scene_coverage': len(valid) == len(cases)}
    pairs = []
    for c in cases:
        pair = {r['arm']: r for r in rows if r['layout_id'] == c['layout_id'] and r.get('valid_episode')}
        if set(pair) == set(ARMS):
            pairs.append({'layout_id': c['layout_id'],
                'without_skills': pair['without_skills']['score'], 'with_skills': pair['with_skills']['score'],
                'delta': pair['with_skills']['score']-pair['without_skills']['score']})
    result['complete_pairs'] = len(pairs)
    result['paired_mean_score_delta'] = sum(p['delta'] for p in pairs)/len(pairs) if pairs else None
    result['pairs'] = pairs
    return result


def run_case(case, arm, gpu, args):
    expected_model=checked_model(json.loads((args.output/'agent-config.json').read_text())['active_model']['model'])
    parent = args.output / f'seed-{case["seed"]}' if 'case_index' in case else args.output
    dest = parent / f'layout-{case["layout_id"]:02d}' / arm
    dest.mkdir(parents=True, exist_ok=False)
    port = args.port_base + case['layout_id']*6 + (ARMS.index(arm)*3 if arm in ARMS else 0)
    if not all(base.port_available(port+i) for i in range(3)):
        raise RuntimeError('Isolated port conflict')
    row = dict(case, arm=arm, gpu=gpu, status='starting', valid_episode=False,
        started_at=base.now(), output=str(dest), actions=0, provider='gpt6-harness', model=expected_model, vla_used=False)
    base.save(dest / 'receipt.json', row)
    fixture = dict(case, task=case.get('task', 'align_blocks'), seed=case.get('seed', 0), arm=arm, output=str(dest),
        test_id=str(args.output), scene_index=case.get('case_index', case['layout_id'])+1)
    if arm == 'self_evolving':
        snapshot = args.experience_snapshot
        knowledge = json.loads(snapshot.read_text())
        from gpt6_failure_learning import source_order, scene_order
        if any(source_order(e) >= scene_order(case) for e in knowledge['entries']):
            raise RuntimeError('Experience contains current/future scene evidence')
        row.update(experience_version=knowledge['version'], experience_sha256=digest(snapshot),
                   experience_source_layouts=[e['source_layout_id'] for e in knowledge['entries']])
        fixture.update(experience_snapshot=str(snapshot), experience_sha256=digest(snapshot),
                       complete_experience_view_required=True)
        if knowledge.get('schema')=='awomopaw.task-experience/v2':
            fixture.update(task_artifact=knowledge.get('task_artifact'),
                           subtask_supervision={'max_actions':args.segment_actions})
    base.save(dest / 'fixture.json', fixture)
    env = dict(os.environ, ROBODOJO_ROOT=str(args.sim_root),
        ROBODOJO_POLICY_REGISTRY_JSON=(args.output / 'registry.snapshot.json').read_text(),
        ROBODOJO_HARNESS_STATE_DIR=str(dest / 'bridge-state'),
        ROBODOJO_ENV_GPU=str(gpu), ROBODOJO_POLICY_GPU=str(gpu),
        ROBODOJO_ACTION_TYPE='joint', AWOMOPAW_HARNESS_PORT=str(port+1),
        ROBODOJO_HARNESS_POLICY_DIR='XPolicyLab/policy/GPT6_Harness_Ablation',
        ROBODOJO_HARNESS_POLICY_NAME='GPT6_Harness_Ablation', ROBODOJO_CODE_BRIDGE_URL='http://127.0.0.1:'+str(port), PYTHONUNBUFFERED='1',
        PYTHONDONTWRITEBYTECODE='1')
    # Never inherit segment caps into frozen baseline arms.
    env.pop('ROBODOJO_SUBTASK_MAX_ACTIONS',None)
    if fixture.get('subtask_supervision'):
        env['ROBODOJO_SUBTASK_MAX_ACTIONS']=str(fixture['subtask_supervision']['max_actions'])
    endpoint = 'http://127.0.0.1:' + str(port)
    bridge = agent = None
    runs = []
    python = str(args.awomopaw / '.venv/bin/python')
    deadline = time.monotonic() + args.case_timeout + 120
    def call(path, payload=None):
        return base.api(endpoint, path, payload, timeout=30)
    try:
        if digest(Path(case['layout_file'])) != case['layout_sha256']:
            raise ValueError('Official scene changed')
        references = [] if arm == 'self_evolving' else json.loads((args.output/'skill/source-manifest.json').read_text())['references']
        for reference in references:
            if digest(Path(reference['copy'])) != reference['sha256']:
                raise ValueError('Skill reference changed')
        with (dest / 'bridge.log').open('ab', buffering=0) as log:
            bridge = subprocess.Popen([python, str(Path(__file__).with_name('gpt6_ablation_bridge.py')),
                '--fixture', str(dest/'fixture.json'), '--port', str(port), '--policy-port', str(port+2)],
                env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        row['bridge_pid'] = bridge.pid
        for _ in range(60):
            if bridge.poll() is not None:
                raise RuntimeError('Isolated bridge exited')
            try:
                health = call('/health')
                break
            except OSError:
                time.sleep(1)
        else:
            raise TimeoutError('Isolated bridge readiness timeout')
        base.save(dest / 'health.json', health)
        if health['status'] != 'ready':
            raise RuntimeError('GPT6 harness bridge preflight failed')
        with (dest / 'agent.log').open('ab', buffering=0) as log:
            command = [python, str(Path(__file__).with_name('run_gpt6_harness_agent.py')),
                '--output', str(dest), '--agent-config', str(args.output/'agent-config.json'),
                '--bridge', endpoint, '--arm', arm, '--skill', str(args.output/'skill'),
                '--timeout', str(args.case_timeout)]
            command += ['--model',expected_model]
            if 'task' in case:
                command += ['--task', case['task'], '--seed', str(case['seed'])]
            if arm == 'self_evolving':
                command += ['--experience', str(args.experience_snapshot)]
            agent = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        row.update(status='running', agent_pid=agent.pid)
        base.save(dest / 'receipt.json', row)
        while agent.poll() is None and time.monotonic() < deadline:
            try:
                state = call('/v1/evaluations/current')
                if state.get('run_id'):
                    if state.get('status') == 'failed':
                        raise RuntimeError('Native harness startup/execution failed: ' + str(state.get('error')))
                    row.update(run_id=state['run_id'], evaluator_pid=state.get('pid'))
                    if state.get('pid') and all(r.get('run_id') != state['run_id'] for r in runs):
                        runs.append(state)
                        base.save(dest/'run.json', state)
                if row.get('run_id'):
                    try:
                        obs = call('/v1/sessions/observation')
                        if not (dest/'initial-observation.json').exists():
                            base.save(dest/'initial-observation.json',obs)
                        require_public_instruction(obs)
                        row['actions'] = obs.get('action_count', row['actions'])
                    except OSError:
                        pass
                base.save(dest/'receipt.json', row)
            except OSError:
                pass
            time.sleep(3)
        if agent.poll() is None:
            raise TimeoutError('Harness case wall-clock budget exhausted')
        if agent.returncode != 0:
            raise RuntimeError('AwomoPaw agent exited: ' + str(agent.returncode))
        state = call('/v1/evaluations/current')
        if not state.get('run_id'):
            raise RuntimeError('Agent returned without starting an episode')
        if all(r.get('run_id') != state['run_id'] for r in runs):
            runs.append(state)
        row.update(run_id=state['run_id'], evaluator_pid=state.get('pid'))
        base.save(dest/'run.json', state)
        if state['status'] in {'starting', 'running'}:
            raise RuntimeError('Agent returned before its episode ended')
        official = call('/v1/evaluations/result?run_id='+state['run_id'])
        base.save(dest/'official-result.json', official)
        native = official['run']['result']
        if not native or native.get('eval_time') != 1 or len(native.get('details', {})) != 1:
            log = Path(state['log_path']).read_text(errors='replace')
            if native and native.get('eval_time') == 0 and 'All scene Unstable Error!' in log:
                row.update(status='native_unstable_before_actions', native_unstable=True)
                return row
            raise RuntimeError('Expected exactly one official scored episode')
        detail = next(iter(native['details'].values()))
        if detail['layout_id'] != case['layout_id'] or state['policy_id'] != 'gpt6-harness':
            raise RuntimeError('Official layout or action provider mismatch')
        selection = json.loads((Path(state['artifact_dir'])/'chat-eval-fixture.json').read_text())
        if selection['layout_sha256'] != case['layout_sha256']:
            raise RuntimeError('Native fixture hash mismatch')
        base.save(dest/'selected-scene.json', selection)
        videos = []
        for original in official.get('videos', []):
            target = dest/'videos'/Path(original).name
            target.parent.mkdir(exist_ok=True)
            shutil.copy2(original, target)
            probe = subprocess.check_output(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                '-show_entries','stream=width,height,nb_read_frames','-of','json',str(target)], text=True, timeout=60)
            stream = json.loads(probe)['streams'][0]
            if int(stream['nb_read_frames']) <= 0:
                raise RuntimeError('Empty camera video')
            videos.append({'path':str(target), 'sha256':digest(target), **stream})
        if len(videos) != 3 or not all(any(c in v['path'] for v in videos)
                for c in ['cam_head','cam_left_wrist','cam_right_wrist']):
            raise RuntimeError('Expected three native camera videos')
        base.save(dest/'videos.json', videos)
        if official.get('harness_trace'):
            shutil.copy2(official['harness_trace'], dest/'harness-trace.jsonl')
        generated = Path(state['artifact_dir'])/'code-policies'
        if generated.exists(): shutil.copytree(generated,dest/'code-policies')
        reads_path = dest/'skill-reads.json'
        reads = json.loads(reads_path.read_text()) if reads_path.exists() else []
        if arm!='self_evolving' and (arm == 'with_skills') != bool(reads):
            raise RuntimeError('Skill reading evidence does not match assigned arm')
        if arm == 'self_evolving':
            learned_reads = json.loads((dest/'experience-reads.json').read_text())
            if not learned_reads or any(r.get('sha256') != row['experience_sha256'] for r in learned_reads):
                raise RuntimeError('Learned experience reading evidence missing/mismatched')
            row['experience_read_count'] = len(learned_reads)
            if fixture.get('task_artifact'):
                row['task_artifact_revision']=fixture['task_artifact']['revision']
                row['task_artifact_sha256']=fixture['task_artifact']['sha256']
            if fixture.get('subtask_supervision'):
                path=dest/'subtask-reflections.jsonl'
                row['subtask_reflection_count']=len(path.read_text().splitlines()) if path.exists() else 0
        audit = [json.loads(line) for line in (dest/'model-audit.jsonl').read_text().splitlines()]
        if not audit or any(r['requested_model'] != expected_model or not model_matches(expected_model,r['returned_model']) for r in audit):
            raise RuntimeError('Missing or mismatched GPT6 model response evidence')
        trace = [json.loads(line) for line in (dest/'harness-trace.jsonl').read_text().splitlines()]
        if any(r.get('source') == 'policy' for r in trace):
            raise RuntimeError('Unexpected pretrained policy actions in GPT6-only arm')
        save_audit = {'verified':True,'responses':len(audit),'model':expected_model,'vla_used':False}
        base.save(dest/'model-audit-summary.json',save_audit)
        row.update(status='completed', valid_episode=True, score=native['score'],
            success=bool(official.get('task_succeeded')), videos=videos, skill_read_count=len(reads),
            provider='gpt6-harness', planner='QwenPawAgent', frozen_controller_executed=False)
    except Exception as error:
        row.update(status='infrastructure_or_harness_error', error=str(error))
        try:
            state = call('/v1/evaluations/current')
            if state.get('run_id'):
                base.save(dest/'failure-evaluation-state.json', state)
                if all(r.get('run_id') != state['run_id'] for r in runs):
                    runs.append(state)
                if state['status'] in {'starting','running'}:
                    call('/v1/sessions/abort', {'reason':'Owned ablation harness cleanup'})
        except Exception:
            pass
    finally:
        if agent:
            base.stop_group(agent.pid)
            agent.wait(timeout=10)
        try:
            if bridge: call('/v1/sessions/abort', {'reason':'Owned GPT6 case cleanup'})
        except Exception:
            pass
        for run in runs:
            base.stop_group(run.get('pid'))
            log = Path(run.get('log_path',''))
            if log.is_file():
                shutil.copy2(log, dest/'native-run.log')
        if bridge:
            base.stop_group(bridge.pid)
            bridge.wait(timeout=10)
        row.update(finished_at=base.now(), owned_services_cleaned=True)
        base.save(dest/'receipt.json', row)
    return row




