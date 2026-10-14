#!/usr/bin/env python3
"""One scoped recovery of the diagnosed zero-action HTTP bind failure.

Wait for healthy episodes to drain through the original scheduler. Preserve all
scored cases and raw infrastructure evidence; resume the same inventory on ports
outside the Linux ephemeral range. No strategy, model or budget changes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from run_parallel_frozen_skills import save, now


def eligible(row, folder):
    if row.get('valid_episode') or row.get('actions', 0) != 0:
        return False
    state=folder/'run.json'
    if not state.exists():return False
    run=json.loads(state.read_text())
    log=Path(run['log_path']).read_text(errors='replace')
    if "error while attempting to bind" not in log or '34004' not in log:return False
    trace=Path(run['artifact_dir'])/'awomopaw_harness_trace.jsonl'
    if trace.exists():
        for line in trace.read_text().splitlines():
            event=json.loads(line)
            if event.get('event')=='action_chunk_execution' and event.get('executed',0)>0:return False
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--awomopaw',type=Path,required=True)
    args=parser.parse_args()
    root=args.output
    assert root==Path('/opt/robodojo/eval_result/gpt6-skills-ablation/20261013-233000')
    recovery=root.with_name(root.name+'-recovery')
    recovery.mkdir(exist_ok=True)
    deadline=time.monotonic()+6000
    while time.monotonic()<deadline:
        status=json.loads((root/'summary.json').read_text())['status']
        if status=='needs_attention':break
        if status not in ['running']:
            raise RuntimeError('Unexpected scheduler state; no automatic resume: '+status)
        save(recovery/'repair-status.json',{'status':'waiting_for_healthy_cases_to_finish','checked_at':now(),
            'no_healthy_episode_interrupted':True,'helper_pid':os.getpid()})
        time.sleep(30)
    else:raise TimeoutError('Original scheduler did not reach a safe boundary')
    time.sleep(3)
    rows=json.loads((root/'results.json').read_text())
    faults=[r for r in rows if r.get('status')=='infrastructure_or_harness_error']
    if len(faults)!=1 or faults[0]['layout_id']!=0 or faults[0]['arm']!='with_skills':
        raise RuntimeError('Unexpected faults; refuse broad automatic recovery')
    fault=faults[0];folder=root/'layout-00/with_skills'
    if not eligible(fault,folder):raise RuntimeError('Attempt is not the diagnosed zero-action bind failure')
    save(recovery/'results-before-recovery.json',rows)
    save(recovery/'summary-before-recovery.json',json.loads((root/'summary.json').read_text()))
    target=recovery/'zero-action-bind-failure/layout-00/with_skills'
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():raise RuntimeError('This recovery has already been applied')
    shutil.move(str(folder),str(target))
    kept=[r for r in rows if r is not fault]
    save(root/'results.json',kept)
    manifest=json.loads((root/'manifest.json').read_text())
    entry={'at':now(),'layout_id':0,'arm':'with_skills','old_run_id':fault.get('run_id'),
        'reason':'Native control HTTP bind failure before first action','archive':str(target),
        'old_port_base':34000,'new_port_base':24000,'already_scored_cases_retested':False,
        'preserved_valid_cases':sum(bool(r.get('valid_episode')) for r in kept)}
    manifest.setdefault('infrastructure_recoveries',[]).append(entry)
    save(root/'manifest.json',manifest)
    command=[sys.executable,str(args.source/'scripts/evaluate_gpt6_skills_seed0.py'),
        '--output',str(root),'--source','/mnt/auwomo-data/wangxin/qwenpaw-harness/physicalrsi-all-skills/20261013-195000',
        '--sim-root','/opt/robodojo','--awomopaw',str(args.awomopaw),'--gpus','5','6',
        '--port-base','24000','--resume']
    with (recovery/'resumed-scheduler.log').open('xb',buffering=0) as log:
        process=subprocess.Popen(command,env=os.environ.copy(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    save(recovery/'resume-launch.json',{'pid':process.pid,'command':command,'recovery':entry})
    save(recovery/'repair-status.json',{'status':'resumed_on_non_ephemeral_ports','at':now(),
        'scheduler_pid':process.pid,'recovery':entry})
    print(json.dumps({'resumed_pid':process.pid,'preserved_cases':len(kept),'repair':entry}),flush=True)

if __name__=='__main__':main()
