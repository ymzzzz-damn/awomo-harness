#!/usr/bin/env python3
"""Read-only mirror and native video verification for a queued Awomo open run."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from gpt6_failure_learning import digest, save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',required=True)
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--interval',type=int,default=60)
    parser.add_argument('--name',help='Exact output folder name for a selected-task/model experiment')
    args=parser.parse_args()
    if not args.experiment.replace('-','').isdigit():
        raise ValueError('Invalid experiment ID')
    root=Path(__file__).resolve().parents[1]
    state=root/'runtime'/('awomo-open-'+args.experiment)
    name=args.name or 'Awomo-open-node153-'+args.experiment
    if Path(name).name!=name or not name.startswith('Awomo-open-'):
        raise ValueError('Invalid experiment output folder')
    local=root/'eval_result'/name
    remote='/mnt/auwomo-data/wangxin/qwenpaw-harness/eval_result/'+name
    ssh=['ssh','-S','/tmp/robodojo-p6000-current.sock','-p','2288','-o','BatchMode=yes',
         '-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10']
    verified=set()
    receipt=state/('mirror-process.json' if args.watch else 'one-shot-sync.json')
    save(receipt,dict(pid=os.getpid(),remote=remote,interval=args.interval,
        mode='watch' if args.watch else 'one_shot',started_at=datetime.now(timezone.utc).isoformat()))
    while True:
        try:
            local.mkdir(parents=True,exist_ok=True)
            subprocess.run(['rsync','-az','--partial','--timeout=90','--exclude=*.tmp','-e',shlex.join(ssh),
                'root@180.76.186.140:'+remote+'/',str(local)+'/'],check=True,capture_output=True,timeout=300)
            manifest=json.loads((local/'manifest.json').read_text())
            if manifest['schema']!='awomopaw.open-online-harness/v1' or manifest['requested']!=len(manifest['cases']):
                raise ValueError('Unexpected remote inventory')
            for record in local.glob('tasks/*/seed-*/layout-*/self_evolving/videos.json'):
                for video in json.loads(record.read_text()):
                    path=record.parent/'videos'/Path(video['path']).name
                    key=(str(path),video['sha256'])
                    if key not in verified:
                        if digest(path)!=video['sha256']:
                            raise ValueError('Mirrored video differs: '+str(path))
                        verified.add(key)
            summary=json.loads((local/'summary.json').read_text())
            process=json.loads((local/'process.json').read_text())
            probe=('from pathlib import Path; import json; p=Path("/proc/'+str(int(process['pid']))+
                '/cmdline"); print(json.dumps({"alive":p.exists() and b"evaluate_awomo_open.py" in p.read_bytes()}))')
            command=['env','KUBECONFIG=/root/.kube/p6000-user.conf','kubectl','-n','dreamzero-training','exec',
                'qwenpaw-sim-harness-5b46b76889-hsv7r','--','/opt/conda/envs/RoboDojo/bin/python','-c',probe]
            checked=subprocess.run([*ssh,'root@180.76.186.140',shlex.join(command)],capture_output=True,text=True,check=True,timeout=40)
            alive=json.loads(checked.stdout)['alive']
            stop_path=local/'user-requested-stop-for-rsi-hybrid.json'
            explicit_stop=json.loads(stop_path.read_text()).get('stopped') is True if stop_path.exists() else False
            effective_status='stopped_by_user' if explicit_stop and not alive else summary['status']
            record=dict(synced_at=datetime.now(timezone.utc).isoformat(),remote=remote,node=manifest['node'],
                finished=summary['finished'],requested=manifest['requested'],status=effective_status,backend_alive=alive,
                scheduler_summary_status=summary['status'],explicit_user_stop=explicit_stop,
                verified_mirrored_videos=len(verified),local_simulation=False)
            save(local/'ui-sync.json',record)
            print(json.dumps(record),flush=True)
            if not args.watch or (local/'scheduler-finished.json').exists() or not alive:
                return
        except Exception as error:
            save(state/'mirror-error.json',dict(at=datetime.now(timezone.utc).isoformat(),error=str(error)))
            if not args.watch:
                raise
        time.sleep(max(30,args.interval))


if __name__=='__main__':
    main()
