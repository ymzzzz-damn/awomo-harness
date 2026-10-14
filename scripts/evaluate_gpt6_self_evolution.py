#!/usr/bin/env python3
"""One sequential 55-scene pass, initially empty memory, learn only after failures."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import evaluate_gpt6_skills_seed0 as control
import run_parallel_frozen_skills as base
from gpt6_failure_learning import empty_memory, validate_memory, digest, save, learning_eligibility

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
    args=p.parse_args()
    if os.environ.get('P6000_ABLATION_VERIFIED_NODE') != control.NODE or args.sim_root != Path('/opt/robodojo'):
        raise RuntimeError('Restricted to the verified P6000 node 153')
    # Generated code is part of both benchmark arms' original tool contract.
    # Reject a broken deployment before creating/starting a scored episode.
    required_runtime = ['ROBODOJO_CODE_POLICY_RUNTIME', 'ROBODOJO_CODE_POLICY_PYTHON', 'ROBODOJO_CODE_ASSETS']
    missing = [key for key in required_runtime if not os.environ.get(key)]
    if missing:
        raise RuntimeError('Missing generic code-runtime deployment variables: ' + ', '.join(missing))
    runtime = Path(os.environ['ROBODOJO_CODE_POLICY_RUNTIME'])
    if not all((runtime / name).is_file() for name in ['contract.py', 'worker.py', 'dependencies.json']):
        raise RuntimeError('Generic code runtime is incomplete')
    if not Path(os.environ['ROBODOJO_CODE_ASSETS']).is_dir() or not os.access(os.environ['ROBODOJO_CODE_POLICY_PYTHON'], os.X_OK):
        raise RuntimeError('Generic code runtime assets/interpreter are unavailable')
    args.output=args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=args.resume)
    cases=control.inventory(args.sim_root)
    baseline_manifest=json.loads((args.baseline/'manifest.json').read_text())
    if cases != baseline_manifest['cases'] or len(args.gpus)!=2:
        raise RuntimeError('Inventory/GPU assignment must match the completed baseline')
    experiment=args.output.name
    memory=empty_memory(experiment)
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
                              'official_action_budget_unchanged':True},
            'learning_budget':{'frames_per_camera':5,'cameras':3,'reflection_timeout_seconds':650,
                               'max_reflection_output_attempts':3,'max_working_memory_characters':10000},
            'learning_source':'only earlier officially failed episodes with executed motion and no deployment error',
            'learning_protocol':'clean-v2; complete controller history within 28000 source characters; bounded working-memory view; lesson citation audit',
            'automatic_episode_retries':0,'created_at':base.now()})
        save(args.output/'experience.json',memory)
    elif json.loads((args.output/'manifest.json').read_text())['cases']!=cases:
        raise RuntimeError('Inventory changed; cannot resume')
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
            'interpretation':'Online sequential learning adds post-failure GPT6 calls; action budgets match the baseline. Scene order and learning-call cost are recorded separately.'})
    try:
        for case in cases:
            layout=case['layout_id']
            validate_memory(memory,experiment,layout)
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
            if not row['success']:
                folder=args.output/f'layout-{layout:02d}'/ARM
                eligibility=learning_eligibility(folder)
                save(folder/'learning/eligibility.json',eligibility)
                if not eligibility['eligible']:
                    if eligibility['infrastructure_error']:
                        raise RuntimeError('Deployment failure cannot be distilled into task memory: '+eligibility['reason'])
                    save(folder/'learning/receipt.json',{'status':'skipped','reason':eligibility['reason'],
                         'source_layout_id':layout,'input_version':memory['version'],'output_version':memory['version'],
                         'no_model_call':True,'no_episode_replay':True})
                    publish('running',{'type':'between_scenes','last_layout_id':layout,'experience_version':memory['version']})
                    continue
                receipt=folder/'learning/receipt.json'
                publish('running',{'type':'failure_reflection','layout_id':layout,'experience_version':memory['version']})
                if not receipt.exists():
                    with (folder/'learning-process.log').open('ab',buffering=0) as stream:
                        subprocess.run([str(args.awomopaw/'.venv/bin/python'),
                            str(Path(__file__).with_name('gpt6_failure_learning.py')),
                            '--case-dir',str(folder),'--snapshot',str(snapshot),
                            '--agent-config',str(args.output/'agent-config.json')],
                            stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=650)
                learned=json.loads(receipt.read_text())
                after=folder/'learning/experience-after.json'
                if learned['before_sha256']!=digest(snapshot) or learned['after_sha256']!=digest(after):
                    raise RuntimeError('Failure reflection provenance mismatch')
                memory=json.loads(after.read_text())
                validate_memory(memory,experiment,layout+1)
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
