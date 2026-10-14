#!/usr/bin/env python3
"""Evaluate a frozen task skill through the live AwomoPaw embodiment.

No LLM generation, strategy edits, Pi0.5 inference or privileged observations.
Keep all episodes, scores and videos, including task and infrastructure failures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import compare_local_tasks as control

PROVIDER = 'physicalrsi-align'
SIM = Path('/home/ymz/robodojo-pi05/RoboDojo')
ROOT = Path(__file__).resolve().parents[1]


def archive(official, dest):
    videos = []
    for original in official.get('videos', []):
        src = Path(original)
        target = dest / 'videos' / src.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        probe = subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
            '-select_streams', 'v:0', '-show_entries', 'stream=width,height,nb_read_frames',
            '-of', 'json', str(target)], text=True)
        stream = json.loads(probe)['streams'][0]
        if int(stream['nb_read_frames']) <= 0:
            raise ValueError('Video has no decodable frames: ' + str(target))
        videos.append({'path': str(target), 'original': str(src),
                       'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), **stream})
    trace = official.get('harness_trace')
    if trace:
        shutil.copy2(trace, dest / 'harness-trace.jsonl')
    runtime = SIM / 'eval_result/frozen_skill_runs' / official['run']['run_id']
    if runtime.exists():
        shutil.copytree(runtime, dest / 'skill-runtime')
    control.save(dest / 'videos.json', videos)
    return videos


def episode(seed, index, output, *, layout=None, fixture=None):
    control.wait_gpu()
    health = control.request('/health')
    provider = next(p for p in health['available_policies'] if p['provider_id'] == PROVIDER)
    if not provider['ready'] or provider['kind'] != 'code':
        raise RuntimeError('Frozen skill provider is not ready')
    active = health.get('active_evaluation')
    if active and active['status'] in {'starting', 'running'}:
        raise RuntimeError('Another episode owns the simulation GPU')
    suffix = '' if layout is None else f'-layout-{layout}'
    dest = output / f'episode-{index:02d}-seed-{seed}{suffix}'
    if layout is not None:
        if fixture is None:
            raise ValueError('Explicit layouts require the native fixture bridge')
        case = {'task': 'align_blocks', 'seed': seed, 'layout_id': layout,
                'test_id': output.name, 'scene_index': index + 1}
        control.save(fixture, case)
        control.save(dest / 'case.json', case)
    run = control.request('/v1/sessions', {'task': 'align_blocks',
        'seed': seed, 'policy_id': PROVIDER})
    control.save(dest / 'run.json', run)
    control.log(f'START frozen-align seed={seed} run={run["run_id"]}')
    row = {'seed': seed, 'run_id': run['run_id'], 'provider': PROVIDER,
           'status': 'running', 'actions': 0, 'valid_episode': False,
           'requested_layout_id': layout}
    try:
        obs = control.wait_ready(run)
        instruction = obs.get('instruction')
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError('Official task instruction is missing')
        control.save(dest / 'initial-observation.json', obs)
        for step in range(int(obs['action_limit']) + 1):
            if obs.get('episode_end'):
                break
            if obs['instruction'] != instruction:
                raise ValueError('Official instruction changed within the episode')
            chunk = control.request('/v1/sessions/policy-chunks', {
                'observation_id': obs['observation_id'], 'provider_id': PROVIDER,
                'subtask_goal': instruction, 'optimized_task_prompt': instruction,
                'planning_summary': 'User-selected unchanged frozen PhysicalRSI task skill'})
            result = control.request('/v1/sessions/action-chunks/execute', {
                'chunk_id': chunk['chunk_id'], 'max_actions': 10, 'discard_remaining': True})
            control.save(dest / 'execution' / f'{step:03d}.json', result)
            row['actions'] += result['executed']
            if step % 20 == 0:
                control.log(f'PROGRESS frozen-align seed={seed} actions={row["actions"]}')
            if result.get('episode_end'):
                break
            obs = control.request('/v1/sessions/observation')
        else:
            raise RuntimeError('Episode exceeded its advertised action limit')
        official = control.official_result(run)
        control.save(dest / 'official-result.json', official)
        result = official['run']['result']
        if result.get('eval_time') != 1 or len(result.get('details', {})) != 1:
            raise ValueError('Expected exactly one official scored episode')
        detail = next(iter(result['details'].values()))
        if layout is not None and detail['layout_id'] != layout:
            raise ValueError('Native evaluator scored a different layout than requested')
        videos = archive(official, dest)
        if len(videos) != 3:
            raise ValueError('Expected head and both wrist camera videos')
        row.update(status='completed', valid_episode=True, score=result['score'],
                   layout_id=detail['layout_id'],
                   success=bool(official['task_succeeded']),
                   details=result['details'], videos=[v['path'] for v in videos])
    except Exception as exc:
        row.update(status='infrastructure_error', error=str(exc))
        try:
            state = control.request('/v1/evaluations/status?run_id=' + run['run_id'])
            if state['status'] in {'starting', 'running'}:
                control.request('/v1/sessions/abort', {'reason': 'Frozen skill evaluation error: ' + str(exc)[:400]})
            official = control.official_result(run)
            control.save(dest / 'official-result.json', official)
            row['videos'] = [v['path'] for v in archive(official, dest)]
        except Exception as cleanup_error:
            row['artifact_error'] = str(cleanup_error)
    control.save(dest / 'receipt.json', row)
    control.log('END ' + json.dumps(row))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--layouts', nargs='+', type=int)
    parser.add_argument('--fixture', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.layouts is not None and (len(args.seeds) != 1 or args.fixture is None):
        parser.error('--layouts requires exactly one --seeds value and --fixture')
    cases = ([(args.seeds[0], layout) for layout in args.layouts]
             if args.layouts is not None else [(seed, None) for seed in args.seeds])
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((ROOT / 'runtime/physicalrsi-align-skill/adapter/frozen-skill.json').read_text())
    control.save(args.output / 'experiment.json', {
        'task': 'align_blocks', 'seeds': args.seeds, 'layouts': args.layouts,
        'gpu': 'local RTX 4090',
        'manifest': manifest, 'strategy_frozen': True, 'llm_strategy_edits': False,
        'qualification': 'Small-sample imported task-skill evaluation; not zero-shot or leaderboard'})
    rows = []
    for index, (seed, layout) in enumerate(cases):
        rows.append(episode(seed, index, args.output, layout=layout, fixture=args.fixture))
        valid = [r for r in rows if r['valid_episode']]
        successes = sum(r['success'] for r in valid)
        summary = {'requested': len(cases), 'attempted': len(rows),
                   'valid_episodes': len(valid), 'successes': successes,
                   'success_rate': successes / len(valid) if valid else None,
                   'infrastructure_errors': len(rows) - len(valid), 'episodes': rows}
        control.save(args.output / 'summary.json', summary)
        # Continuing after an infrastructure failure could hide a contract mismatch.
        if not rows[-1]['valid_episode']:
            break
        time.sleep(2)
    control.log('SUMMARY ' + json.dumps(summary))


if __name__ == '__main__':
    main()
