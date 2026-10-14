#!/usr/bin/env python3
"""Check control, failure reflection, provenance, score coverage and UI mirroring."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
import urllib.request
from monitor_gpt6_skills_seed0 import REMOTE as BASE_REMOTE
from gpt6_failure_learning import save


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--experiment',required=True)
    p.add_argument('--heartbeat-time',required=True)
    a=p.parse_args()
    if not a.experiment.replace('-','').isdigit():raise ValueError('Invalid experiment id')
    root=Path(__file__).resolve().parents[1]
    state=root/'runtime'/f'gpt6-self-evolution-seed0-{a.experiment}'
    remote=BASE_REMOTE.replace('/gpt6-skills-ablation/20261013-233000',f'/gpt6-self-evolution/{a.experiment}')
    remote=remote.replace("!={'with_skills','without_skills'}", "!={'self_evolving'}")
    remote=remote.replace("'requested':110", "'requested':55")
    remote=remote.replace("print(json.dumps({'remote_epoch'", "extra_report={'remote_epoch'")
    # Replace only the final print; append additional causal-chain validation.
    tail="'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()}))"
    remote=remote.replace(tail, "'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()}\n")
    remote += '''
import hashlib
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
memory=read(root/'experience.json');extra_report['phase']=summary.get('phase');extra_report['experience_version']=memory.get('version')
extra_report['runtime_repair']=read(root.with_name(root.name+'-recovery')/'code-runtime-env/repair-status.json').get('status')
for case in cases:
 folder=root/f"layout-{case['layout_id']:02d}"/'self_evolving'
 before=root/'learning/snapshots'/f"layout-{case['layout_id']:02d}.json"
 knowledge=read(before)
 entries=knowledge.get('entries',[])
 if knowledge.get('experiment')!=root.name or knowledge.get('pretrained_skills') is not False or knowledge.get('version')!=len(entries) or any(e.get('source_layout_id',999)>=case['layout_id'] for e in entries):issues.append({'case':[case['layout_id'],'self_evolving'],'error':'Experience provenance/current-future scene leak'})
 for entry in entries:
  source=root/f"layout-{entry['source_layout_id']:02d}"/'self_evolving'
  receipt=read(source/'receipt.json')
  if not receipt.get('valid_episode') or receipt.get('success') is not False or receipt.get('run_id')!=entry.get('source_run_id') or sha(source/'learning/analysis.json')!=entry.get('analysis_sha256'):issues.append({'error':'Experience came from invalid/nonfailure/changed analysis'})
 if case.get('valid_episode'):
  reads=read(folder/'experience-reads.json',[])
  if not reads or any(r.get('sha256')!=sha(before) for r in reads):issues.append({'case':[case['layout_id'],'self_evolving'],'error':'Experience was not read'})
  fixture=read(folder/'fixture.json')
  if fixture.get('complete_experience_view_required') and not any(r.get('complete_working_memory') is True for r in reads):issues.append({'case':[case['layout_id'],'self_evolving'],'error':'Working memory view was truncated/partial'})
  applications=events(folder/'experience-application.jsonl')
  case['memory_application_claims']=sum(bool(a.get('cited_lesson_ids')) for a in applications)
  if any(a.get('unknown_lesson_ids') for a in applications):issues.append({'case':[case['layout_id'],'self_evolving'],'error':'Action rationale cited an unknown lesson ID'})
  learned=read(folder/'learning/receipt.json')
  if learned and learned.get('status')=='completed':
   audits=events(folder/'learning/model-audit.jsonl')
   if not audits or any(x.get('requested_model')!='gpt-6-astra' or not str(x.get('returned_model')).startswith('gpt-6-astra') for x in audits):issues.append({'error':'Reflection model mismatch'})
   expected_frames=sum(min(5,int(v.get('nb_read_frames',0))) for v in read(folder/'videos.json',[]))
   if learned.get('frame_count')!=expected_frames:issues.append({'error':'Failure frame coverage mismatch'})
phase=summary.get('phase') or {}
if phase.get('type')=='failure_reflection':
 pid=read(root/'process.json').get('pid')
 active=[]
 for proc in Path('/proc').iterdir():
  if proc.name.isdigit():
   try:
    stat=(proc/'stat').read_text().split();command=(proc/'cmdline').read_bytes().replace(b'\\x00',b' ').decode()
    if int(stat[3])==pid and 'gpt6_failure_learning.py' in command:active.append(int(proc.name))
   except (OSError,ValueError):pass
 extra_report['reflection_pids']=active
extra_report['errors']=issues
print(json.dumps(extra_report))
'''
    command=['ssh','-S','/tmp/robodojo-p6000-current.sock','-p','2288','-o','BatchMode=yes',
             '-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10','root@180.76.186.140',
             'export KUBECONFIG=/root/.kube/p6000-user.conf; kubectl -n dreamzero-training exec -i qwenpaw-sim-harness-5b46b76889-hsv7r -- /opt/conda/envs/RoboDojo/bin/python -']
    compile(remote, '<remote self-evolution monitor>', 'exec')
    result=subprocess.run(command,input=remote,text=True,capture_output=True,timeout=50)
    if result.returncode:raise RuntimeError(result.stderr[-2000:])
    report=json.loads(result.stdout)
    prior=json.loads((state/'monitor/latest.json').read_text()) if (state/'monitor/latest.json').exists() else {}
    mirror=json.loads((state/'mirror-process.json').read_text())['pid']
    lines=(state/'sync.log').read_text().splitlines();last={}
    for line in lines:
        try:
            candidate=json.loads(line)
            if candidate.get('synced_utc'):last=candidate
        except ValueError:pass
    catalog=json.load(urllib.request.urlopen('http://127.0.0.1:8791/api/catalog?refresh=1',timeout=10))
    group=f'gpt6-self-evolution-align-seed0-node153-{a.experiment}::self_evolving'
    count=sum(s['experiment']==group for s in catalog['scenes'])
    age=time.time()-datetime.fromisoformat(last['synced_utc']).timestamp() if last else None
    report['local']={'mirror_alive':Path(f'/proc/{mirror}/cmdline').exists(),'sync_age_seconds':age,
                     'verified_videos':last.get('verified_mirrored_videos'),'ui_scenes':count}
    if report['status']!='completed' and not report['local']['mirror_alive']:report['errors'].append({'error':'Mirror not live'})
    if age is None or (age>240 and report['status']!='completed'):report['errors'].append({'error':'Local synchronization stale'})
    if count!=55:report['errors'].append({'error':'UI inventory mismatch'})
    old={json.dumps(e,sort_keys=True) for e in prior.get('errors',[])};new={json.dumps(e,sort_keys=True) for e in report['errors']}
    report.update(heartbeat_time=a.heartbeat_time,checked_at=datetime.now(timezone.utc).isoformat(),
                  changes={'finished_delta':report['valid_finished']-prior.get('valid_finished',0),
                           'new_errors':[json.loads(e) for e in new-old],'recovered_errors':[json.loads(e) for e in old-new]})
    stamp=datetime.fromisoformat(a.heartbeat_time.replace('Z','+00:00')).strftime('%Y%m%d-%H%M%S')
    save(state/'monitor'/f'{stamp}.json',report);save(state/'monitor/latest.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['cases','gpu']},ensure_ascii=False))


if __name__=='__main__':main()
