#!/usr/bin/env python3
"""Mirror this remote evaluation's evidence; never runs simulation or inference."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = '20261013-224000'
LOCAL = ROOT/'eval_result'/('pi05-skills-align-seed0-node153-'+EXPERIMENT)
REMOTE = '/mnt/auwomo-data/wangxin/qwenpaw-harness/eval_result/pi05-skills-ablation/'+EXPERIMENT
STATE = ROOT/'runtime'/('pi05-skills-seed0-'+EXPERIMENT)


def save(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    temp.replace(path)


def sync(socket):
    LOCAL.mkdir(parents=True,exist_ok=True)
    ssh=['ssh','-S',str(socket),'-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
         '-o','ConnectTimeout=15','-p','2288']
    subprocess.run([*ssh,'-O','check','root@180.76.186.140'],check=True,capture_output=True)
    command=' '.join(ssh)
    for remote,local in [(REMOTE,LOCAL),(REMOTE+'-runner',STATE/'remote-runner'),
                          (REMOTE+'-preflight-v2',STATE/'remote-preflight')]:
        local.mkdir(parents=True,exist_ok=True)
        subprocess.run(['rsync','-az','--partial','--timeout=90','--exclude=*.tmp',
            '-e',command,'root@180.76.186.140:'+remote+'/',str(local)+'/'],check=True,timeout=300)
    manifest=json.loads((LOCAL/'manifest.json').read_text())
    summary=json.loads((LOCAL/'summary.json').read_text())
    if manifest['node']!='192.168.48.153' or manifest['requested_episodes']!=110:
        raise RuntimeError('Unexpected remote evaluation manifest')
    videos=0
    for path in LOCAL.glob('layout-*/*/videos.json'):
        for video in json.loads(path.read_text()):
            local=path.parent/'videos'/Path(video['path']).name
            if local.exists() and hashlib.sha256(local.read_bytes()).hexdigest()!=video['sha256']:
                raise RuntimeError('Mirrored video hash mismatch: '+str(local))
            if local.exists():videos+=1
    receipt={'synced_utc':datetime.now(timezone.utc).isoformat(),'remote':REMOTE,
        'local':str(LOCAL),'status':summary['status'],'finished':summary['finished'],
        'requested':110,'verified_mirrored_videos':videos,'node':manifest['node'],
        'simulation_runs_locally':False}
    save(LOCAL/'ui-sync.json',receipt)
    print(json.dumps(receipt),flush=True)
    return summary['status'] in {'completed','failed','needs_attention'}


def main():
    global EXPERIMENT, LOCAL, REMOTE, STATE
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-socket',type=Path,required=True)
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--experiment',default=EXPERIMENT)
    args=parser.parse_args()
    if not args.experiment.replace('-','').isdigit():
        parser.error('Invalid experiment identifier')
    EXPERIMENT=args.experiment
    LOCAL=ROOT/'eval_result'/('pi05-skills-align-seed0-node153-'+EXPERIMENT)
    REMOTE='/mnt/auwomo-data/wangxin/qwenpaw-harness/eval_result/pi05-skills-ablation/'+EXPERIMENT
    STATE=ROOT/'runtime'/('pi05-skills-seed0-'+EXPERIMENT)
    failures=0
    while True:
        try:
            finished=sync(args.control_socket)
            failures=0
        except Exception as error:
            failures+=1
            save(STATE/'mirror-error.json',{'type':type(error).__name__,'error':str(error),
                'at':datetime.now(timezone.utc).isoformat(),'no_simulation_or_retry':True,
                'consecutive_mirror_failures':failures})
            if not args.watch or failures>=30:raise
            time.sleep(min(60,10*failures))
            continue
        if finished or not args.watch:return
        time.sleep(60)


if __name__=='__main__':
    main()
