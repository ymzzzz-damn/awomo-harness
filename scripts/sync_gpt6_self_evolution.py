#!/usr/bin/env python3
"""Read-only mirroring of a separate online-learning experiment."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
from gpt6_failure_learning import digest, save


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment',required=True)
    p.add_argument('--watch',action='store_true')
    p.add_argument('--control-socket',default='/tmp/robodojo-p6000-current.sock')
    a=p.parse_args()
    if not a.experiment.replace('-','').isdigit():raise ValueError('Invalid experiment id')
    root=Path(__file__).resolve().parents[1]
    state=root/'runtime'/f'gpt6-self-evolution-seed0-{a.experiment}'
    local=root/'eval_result'/f'gpt6-self-evolution-align-seed0-node153-{a.experiment}'
    remote=f'/mnt/auwomo-data/wangxin/qwenpaw-harness/eval_result/gpt6-self-evolution/{a.experiment}'
    ssh=['ssh','-S',a.control_socket,'-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10','-p','2288']
    failures=0
    while True:
        try:
            local.mkdir(parents=True,exist_ok=True)
            subprocess.run(['rsync','-az','--delete-delay','--partial','--timeout=90','--exclude=*.tmp',
                '-e',' '.join(ssh),'root@180.76.186.140:'+remote+'/',str(local)+'/'],check=True,timeout=300)
            manifest=json.loads((local/'manifest.json').read_text())
            assert manifest['node']=='192.168.48.153' and manifest['requested_episodes']==55
            assert manifest['pretrained_skills'] is False and manifest['arms']==['self_evolving']
            report=json.loads((local/'summary.json').read_text())
            videos=0
            for file in local.glob('layout-*/self_evolving/videos.json'):
                for v in json.loads(file.read_text()):
                    path=file.parent/'videos'/Path(v['path']).name
                    if not path.exists() or digest(path)!=v['sha256']:raise RuntimeError('Video mirror incomplete or hash mismatch')
                    videos+=1
            r={'synced_utc':datetime.now(timezone.utc).isoformat(),'status':report['status'],
               'finished':report['finished'],'requested':55,'verified_mirrored_videos':videos,
               'experience_version':report['experience_version'],'remote':remote,'local':str(local)}
            save(local/'ui-sync.json',r);print(json.dumps(r),flush=True)
            failures=0
            if report['status']=='completed' or not a.watch:return
        except Exception as error:
            failures+=1
            record={'at':datetime.now(timezone.utc).isoformat(),'error':str(error),'failures':failures}
            try:save(state/'mirror-error.json',record)
            except OSError:
                save(Path('/dev/shm')/f'robodojo-self-evolution-{a.experiment}-mirror-error.json',record)
            if not a.watch or failures>=30:raise
        time.sleep(60)


if __name__=='__main__':main()
