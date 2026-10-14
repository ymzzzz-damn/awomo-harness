#!/usr/bin/env python3
"""Sequential development: subtask supervision, success skills and video feedback."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import gpt6_episode as control
import harness_utils as base
from gpt6_failure_learning import empty_memory, validate_memory, digest, save, learning_eligibility
from gpt6_task_learning import empty_task_memory, validate_task_memory

ARM = 'self_evolving'


def summary(cases, rows, memory, status, phase=None):
    valid = [r for r in rows if r.get('valid_episode')]
    successes = sum(bool(r['success']) for r in valid)
    return {'schema':'awomopaw.gpt6-self-evolution/v1', 'task':'align_blocks','seed':0,
            'status':status,'phase':phase,'updated_at':base.now(),'requested_episodes':55,
            'finished':len(rows),'valid_finished':len(valid),'experience_version':memory['version'],
            'arms':{ARM:{'requested':55,'finished':len(rows),'valid_episodes':len(valid),
                        'successes':successes,'success_rate_over_valid_episodes':successes/len(valid) if valid else None,
                        'mean_official_score':sum(r['score'] for r in valid)/len(valid) if valid else None,
                        'unscored_or_invalid':len(rows)-len(valid),'complete_scene_coverage':len(valid)==55}}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--awomopaw', type=Path, required=True)
    p.add_argument('--sim-root', type=Path, default=Path('/opt/robodojo'))
    p.add_argument('--gpus', type=int, nargs='+', default=[5,6])
    p.add_argument('--port-base', type=int, default=26000)
    p.add_argument('--case-timeout', type=int, default=5400)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--learning-mode', choices=['task-artifacts', 'failure-only'], default='task-artifacts')
    p.add_argument('--segment-actions', type=int, choices=range(10, 101), default=40)
    p.add_argument('--task-artifact', type=Path, help='Explicit same-task learned skill from an earlier run')
    args=p.parse_args()
    if os.environ.get('P6000_ABLATION_VERIFIED_NODE') != control.NODE or args.sim_root != Path('/opt/robodojo'):
        raise RuntimeError('Restricted to the verified P6000 node 153')
    # Generated code is part of both benchmark arms' original tool contract.
    # Reject a broken deployment before creating/starting a scored episode.
    required_runtime = ['ROBODOJO_CODE_POLICY_RUNTIME', 'ROBODOJO_CODE_POLICY_PYTHON', 'ROBODOJO_PRIMITIVES_PATH']
    missing = [key for key in required_runtime if not os.environ.get(key)]
    if missing:
        raise RuntimeError('Missing generic code-runtime deployment variables: ' + ', '.join(missing))
    runtime = Path(os.environ['ROBODOJO_CODE_POLICY_RUNTIME'])
    if not all((runtime / name).is_file() for name in ['contract.py', 'worker.py']):
        raise RuntimeError('Generic code runtime is incomplete')
    if args.learning_mode=='task-artifacts' and 'awomopaw.subtask-boundary/v1' not in (runtime/'worker.py').read_text():
        raise RuntimeError('Install the subtask-boundary worker before task-artifacts runs')
    if not Path(os.environ['ROBODOJO_PRIMITIVES_PATH']).is_dir() or not os.access(os.environ['ROBODOJO_CODE_POLICY_PYTHON'], os.X_OK):
        raise RuntimeError('Generic code runtime assets/interpreter are unavailable')
    args.output=args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    cases=control.inventory(args.sim_root)
    baseline_manifest=json.loads((args.baseline/'manifest.json').read_text())
    if cases != baseline_manifest['cases'] or len(args.gpus)!=2:
        raise RuntimeError('Inventory/GPU assignment must match the completed baseline')
    experiment=args.output.name
    task_learning=args.learning_mode=='task-artifacts'
    if args.task_artifact and not task_learning:
        raise ValueError('Imported learned skills require task-artifacts mode')
    reference=None
    if args.task_artifact:
        artifact=json.loads(args.task_artifact.read_text())
        reference=dict(path=str(args.task_artifact.resolve()), sha256=digest(args.task_artifact),revision=artifact['revision'])
    memory=empty_task_memory(experiment,'align_blocks',reference) if task_learning else empty_memory(experiment)
    validate_knowledge=validate_task_memory if task_learning else validate_memory
    if not args.resume:
        for name in ['agent-config.json','registry.snapshot.json']:
            shutil.copy2(args.baseline/name, args.output/name)
        model=json.loads((args.output/'agent-config.json').read_text())['active_model']
        if model['model']!='gpt-6-astra':raise RuntimeError('GPT6 configuration mismatch')
        save(args.output/'manifest.json', {'schema':'awomopaw.gpt6-self-evolution/v1',
            'node':control.NODE,'task':'align_blocks','seed':0,'cases':cases,'arms':[ARM],
            'requested_episodes':55,'model':model,'provider':'gpt6-harness','vla_used':False,
            'pretrained_skills':False,'fresh_harness_per_episode':True,'scene_order':list(range(55)),
            'sequential':True,'gpus':args.gpus,'gpu_assignment':'same layout parity as baseline',
            'baseline_experiment':str(args.baseline),'baseline_manifest_sha256':digest(args.baseline/'manifest.json'),
            'control_budget':{'case_timeout_seconds':args.case_timeout,'max_agent_iters':1200,
                              'official_action_budget_unchanged':True,
                              'supervision_boundary_actions':args.segment_actions if task_learning else None},
            'learning_budget':{'frames_per_camera':5,'cameras':3,'reflection_timeout_seconds':650,
                               'max_reflection_output_attempts':3,'max_working_memory_characters':10000},
            'learning_mode':args.learning_mode,
            'initial_task_artifact':reference,
            'learning_source':'earlier completed development successes and failures' if task_learning else 'only earlier officially failed episodes with executed motion and no deployment error',
            'learning_protocol':'task-artifacts-v2; timestamped success/failure rethinking; success skill reuse; subtask supervision' if task_learning else 'clean-v2; complete controller history within 28000 source characters; bounded working-memory view; lesson citation audit',
            'automatic_episode_retries':0,'created_at':base.now()})
        save(args.output/'experience.json',memory)
    elif json.loads((args.output/'manifest.json').read_text())['cases']!=cases:
        raise RuntimeError('Inventory changed; cannot resume')
    if args.resume:
        manifest=json.loads((args.output/'manifest.json').read_text())
        if manifest.get('learning_mode','failure-only')!=args.learning_mode or manifest.get('initial_task_artifact')!=reference:
            raise RuntimeError('Learning protocol/initial skill changed; use a new output directory')
        if task_learning and manifest['control_budget']['supervision_boundary_actions']!=args.segment_actions:
            raise RuntimeError('Supervision boundary changed; use a new output directory')
    rows=json.loads((args.output/'results.json').read_text()) if (args.output/'results.json').exists() else []
    if [r['layout_id'] for r in rows]!=list(range(len(rows))) or any(not r.get('valid_episode') for r in rows):
        raise RuntimeError('Cannot silently replay invalid, unordered or duplicate episodes')
    save(args.output/'process.json',{'pid':os.getpid(),'started_at':base.now(),'node':control.NODE})
    locks=base.acquire_gpu_locks(args,time.monotonic()+300)
    def publish(status,phase):
        save(args.output/'results.json',rows)
        report=summary(cases,rows,memory,status,phase)
        save(args.output/'summary.json',report)
        save(args.output/'status.json',report)
        baseline=json.loads((args.baseline/'summary.json').read_text())
        valid={r['layout_id']:r for r in rows if r.get('valid_episode')}
        paired={}
        for arm in ['without_skills','with_skills']:
            original=[r for r in json.loads((args.baseline/'results.json').read_text()) if r['arm']==arm and r.get('valid_episode')]
            pairs=[{'layout_id':r['layout_id'],'baseline_score':r['score'],
                    'self_evolving_score':valid[r['layout_id']]['score']} for r in original if r['layout_id'] in valid]
            paired[arm]={'complete_pairs':len(pairs),'pairs':pairs,
                'mean_score_delta':sum(p['self_evolving_score']-p['baseline_score'] for p in pairs)/len(pairs) if pairs else None}
        save(args.output/'comparison.json',{'baseline_experiment':str(args.baseline),
            'baseline_arms':baseline['arms'],'self_evolving_arm':report['arms'][ARM],
            'paired_over_completed_scenes':paired,
            'interpretation':'Online sequential learning adds recorded post-task GPT6 calls; action budgets match the baseline. Success skill reuse and subtask supervision belong to the task-artifacts protocol; results are not a frozen baseline.'})
    try:
        for case in cases:
            layout=case['layout_id']
            validate_knowledge(memory,experiment,layout)
            snapshot=args.output/'learning/snapshots'/f'layout-{layout:02d}.json'
            if snapshot.exists():
                if json.loads(snapshot.read_text())!=memory:raise RuntimeError('Immutable experience snapshot mismatch')
            else:save(snapshot,memory)
            args.experience_snapshot=snapshot
            existing=next((r for r in rows if r['layout_id']==layout),None)
            if existing is None:
                gpu=args.gpus[layout%2]
                publish('running',{'type':'waiting_for_gpu','layout_id':layout,'gpu':gpu})
                deadline=time.monotonic()+600
                while base.gpu_memory(gpu)['used_mib']>1024:
                    if time.monotonic()>deadline:raise RuntimeError('GPU did not become available')
                    time.sleep(5)
                publish('running',{'type':'episode','layout_id':layout,'experience_version':memory['version']})
                print(base.now(),'START',layout,ARM,'experience',memory['version'],flush=True)
                row=control.run_case(case,ARM,gpu,args)
                rows.append(row)
                publish('running',{'type':'episode_finished','layout_id':layout})
                if not row.get('valid_episode'):
                    raise RuntimeError('Episode infrastructure failure: '+str(row.get('error',row['status'])))
                print(base.now(),'END',layout,row['score'],flush=True)
            else:row=existing
            if row['experience_sha256']!=digest(snapshot):raise RuntimeError('Episode read a different experience version')
            if task_learning or not row['success']:
                folder=args.output/f'layout-{layout:02d}'/ARM
                learning=folder/('task-learning' if task_learning else 'learning')
                eligibility=learning_eligibility(folder,allow_success=task_learning)
                save(folder/'learning-eligibility.json',eligibility)
                if not eligibility['eligible']:
                    if eligibility['infrastructure_error']:
                        raise RuntimeError('Deployment failure cannot be distilled into task memory: '+eligibility['reason'])
                    save(learning/'receipt.json',{'status':'skipped','reason':eligibility['reason'],
                         'source_layout_id':layout,'input_version':memory['version'],'output_version':memory['version'],
                         'no_model_call':True,'no_episode_replay':True})
                    publish('running',{'type':'between_scenes','last_layout_id':layout,'experience_version':memory['version']})
                    continue
                receipt=learning/'receipt.json'
                publish('running',{'type':'success_distillation' if row['success'] else 'failure_reflection','layout_id':layout,'experience_version':memory['version']})
                if not receipt.exists():
                    with (folder/'learning-process.log').open('ab',buffering=0) as stream:
                        command=[str(args.awomopaw/'.venv/bin/python'),
                            str(Path(__file__).with_name('gpt6_task_learning.py' if task_learning else 'gpt6_failure_learning.py')),
                            '--case-dir',str(folder),'--snapshot',str(snapshot),
                            '--agent-config',str(args.output/'agent-config.json')]
                        if task_learning:command+=['--output',str(learning),'--split','development']
                        subprocess.run(command,
                            stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=650)
                learned=json.loads(receipt.read_text())
                after=learning/'experience-after.json'
                if learned['before_sha256']!=digest(snapshot) or learned['after_sha256']!=digest(after):
                    raise RuntimeError('Failure reflection provenance mismatch')
                memory=json.loads(after.read_text())
                validate_knowledge(memory,experiment,layout+1)
                save(args.output/'experience.json',memory)
                print(base.now(),'LEARNED',layout,'version',memory['version'],flush=True)
            publish('running',{'type':'between_scenes','last_layout_id':layout,'experience_version':memory['version']})
        publish('completed',{'type':'completed'})
    except BaseException as error:
        save(args.output/'scheduler-error.json',{'type':type(error).__name__,'error':str(error),'at':base.now()})
        publish('needs_attention',{'type':'stopped_on_error','error':str(error)})
        raise
    finally:
        for lock in locks:lock.close()


if __name__=='__main__':main()
