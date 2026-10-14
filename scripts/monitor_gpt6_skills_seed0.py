#!/usr/bin/env python3
"""Read-only heartbeat inspection for the active GPT6 comparison, not a scheduler.

Remote ages are measured on the remote node; local sync ages on this machine.
Writes monitoring receipts only. Never restarts, kills or rescores an episode.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
import urllib.request
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
STATE=ROOT/'runtime/gpt6-skills-seed0-20261013-233000'
REMOTE=r'''
from pathlib import Path
import json,time,urllib.request,subprocess
root=Path('/opt/robodojo/eval_result/gpt6-skills-ablation/20261013-233000')
now=time.time()
def read(p, default=None):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return {} if default is None else default
def events(p):
 try:lines=p.read_text().splitlines()
 except OSError:return []
 result=[]
 for line in lines:
  try:result.append(json.loads(line))
  except ValueError:pass
 return result
def alive(pid):
 p=Path(f'/proc/{pid}/stat')
 try:return p.read_text().split()[2]!='Z'
 except OSError:return False
summary=read(root/'summary.json');manifest=read(root/'manifest.json');cases=[];issues=[]
for path in sorted(root.glob('layout-*/*/receipt.json')):
 row=read(path)
 if not row:continue
 case={k:row.get(k) for k in ['layout_id','arm','status','actions','valid_episode','score','success','run_id','error']}
 def issue(kind):issues.append({'case':[row['layout_id'],row['arm']],'error':kind})
 if row['status'] in ['running','starting']:
  state=read(path.parent/'bridge-state'/f"{row.get('run_id')}.json")
  case['native_lifecycle']=state.get('status')
  native_done=state.get('status') in ['completed','failed']
  case['processes_alive']={k:alive(row[k]) for k in ['agent_pid','bridge_pid','evaluator_pid'] if row.get(k)}
  if any(not value for key,value in case['processes_alive'].items() if not (native_done and key=='evaluator_pid')):issue('Owned process exited')
  agent=path.parent/'agent-messages.jsonl'
  case['agent_age_seconds']=round(now-agent.stat().st_mtime,1) if agent.exists() else None
  run=read(path.parent/'run.json')
  if run:
   native=Path(run['artifact_dir']);trace=native/'awomopaw_harness_trace.jsonl';traces=events(trace)
   case['actual_actions']=max((e.get('action_count',0) for e in traces),default=0)
   case['trace_age_seconds']=round(now-trace.stat().st_mtime,1) if trace.exists() else None
   log=Path(run['log_path']).read_text(errors='replace')
   if 'error while attempting to bind' in log:issue('Native control bind failure')
   health=read(path.parent/'health.json');control=health.get('control_url')
   if control and not native_done:
    try:
     with urllib.request.urlopen(control+'/v1/observations/current',timeout=2) as reply:
      obs=json.load(reply)
     case['control']={k:obs.get(k) for k in ['run_id','status','action_count','episode_end']}
     if obs.get('run_id')!=run['run_id']:issue('Control run_id mismatch')
    except Exception as error:case['control_error']=str(error)[:180]
   workers=[]
   for worker in (native/'code-policies').glob('*/receipt.json'):
    record=read(worker)
    if alive(record.get('pid')):
     stat=Path(f"/proc/{record['pid']}/stat").read_text().split()
     elapsed=float(Path('/proc/uptime').read_text().split()[0])-int(stat[21])/__import__('os').sysconf('SC_CLK_TCK')
     workers.append({'policy_id':record['policy_id'],'pid':record['pid'],'elapsed_seconds':round(elapsed,1)})
     if elapsed>930:issue('Code worker exceeds supervised runtime limit')
   case['live_workers']=workers
  if case['agent_age_seconds'] is not None and case['agent_age_seconds']>600:issue('No agent message for over 10 minutes')
 elif not row.get('valid_episode'):
  issue(row.get('error') or 'Unscored finished case')
 audits=events(path.parent/'model-audit.jsonl');case['model_responses']=len(audits)
 if any(a.get('requested_model')!='gpt-6-astra' or not str(a.get('returned_model','')).startswith('gpt-6-astra') for a in audits):issue('Model mismatch')
 if row.get('valid_episode'):
  finished_trace=events(path.parent/'harness-trace.jsonl')
  case['actual_actions']=max((e.get('action_count',0) for e in finished_trace),default=row.get('actions',0))
  official=read(path.parent/'official-result.json');native_result=official.get('run',{}).get('result',{})
  details=native_result.get('details',{})
  if native_result.get('eval_time')!=1 or len(details)!=1 or next(iter(details.values()),{}).get('layout_id')!=row['layout_id'] or native_result.get('score')!=row.get('score'):issue('Official result/fixture mismatch')
  videos=read(path.parent/'videos.json',[])
  if len(videos)!=3 or any(int(v.get('nb_read_frames',0))<=0 for v in videos):issue('Missing/empty camera videos')
 cases.append(case)
if len(manifest.get('cases',[]))!=55 or {c['layout_id'] for c in manifest.get('cases',[])}!=set(range(55)) or set(manifest.get('arms',[]))!={'with_skills','without_skills'}:issues.append({'error':'Inventory mismatch'})
valid=sum(bool(c.get('valid_episode')) for c in cases)
if valid!=sum(a.get('valid_episodes',0) for a in summary.get('arms',{}).values()):issues.append({'error':'Summary count mismatch'})
pid=read(root/'process.json').get('pid')
if summary.get('status')=='running' and not alive(pid):issues.append({'error':'Scheduler not live'})
print(json.dumps({'remote_epoch':now,'status':summary.get('status'),'finished':summary.get('finished'),'valid_finished':valid,'requested':110,'scheduler_pid':pid,'scheduler_alive':alive(pid),'arms':summary.get('arms'),'cases':cases,'errors':issues,'repair':read(root.with_name(root.name+'-recovery')/'repair-status.json').get('status'),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()}))
'''

def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--heartbeat-time',required=True)
    args=parser.parse_args()
    command=['ssh','-S','/tmp/robodojo-p6000-current.sock','-p','2288','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10','root@180.76.186.140',
      'export KUBECONFIG=/root/.kube/p6000-user.conf; kubectl -n dreamzero-training exec -i qwenpaw-sim-harness-5b46b76889-hsv7r -- /opt/conda/envs/RoboDojo/bin/python -']
    reply=subprocess.run(command,input=REMOTE,capture_output=True,text=True,timeout=45)
    reply.check_returncode()
    snapshot=json.loads(reply.stdout)
    snapshot.update(automation_id='gpt6-align-blocks',heartbeat_time=args.heartbeat_time)
    previous=json.loads((STATE/'monitor/latest.json').read_text()) if (STATE/'monitor/latest.json').exists() else {}
    old={(c['layout_id'],c['arm']):c for c in previous.get('cases',[])}
    changes=[]
    for case in snapshot['cases']:
        before=old.get((case['layout_id'],case['arm']),{})
        current=case.get('control',{}).get('action_count',case.get('actual_actions',case.get('actions',0))) or 0
        earlier=before.get('control',{}).get('action_count',before.get('actual_actions',before.get('actions',0))) or 0
        if current!=earlier or case['status']!=before.get('status'):
            changes.append({'layout_id':case['layout_id'],'arm':case['arm'],'action_delta':current-earlier,'status':case['status']})
    mirror=json.loads((STATE/'mirror-process.json').read_text())['pid']
    sync=json.loads((STATE/'sync.log').read_text().splitlines()[-1])
    catalog=json.load(urllib.request.urlopen('http://127.0.0.1:8791/api/catalog?refresh=1',timeout=5))
    prefix='gpt6-skills-align-seed0-node153-20261013-233000::'
    groups=Counter(c['experiment'] for c in catalog['scenes'] if c['experiment'].startswith(prefix))
    sync_age=time.time()-datetime.fromisoformat(sync['synced_utc']).timestamp()
    snapshot['local']={'checked_at':datetime.now(timezone.utc).isoformat(),'mirror_alive':Path(f'/proc/{mirror}/cmdline').exists(),'last_sync_utc':sync['synced_utc'],'sync_age_seconds':round(sync_age,1),'verified_mirrored_videos':sync['verified_mirrored_videos'],'target_ui_groups':dict(groups)}
    if not snapshot['local']['mirror_alive'] and snapshot['status']=='running':snapshot['errors'].append({'error':'Mirror not live'})
    if sync_age>180:snapshot['errors'].append({'error':'Local synchronization stale'})
    if len(groups)!=2 or set(groups.values())!={55}:snapshot['errors'].append({'error':'Target UI inventory mismatch'})
    prior_errors={json.dumps(e,sort_keys=True) for e in previous.get('errors',[])}
    new_errors=[e for e in snapshot['errors'] if json.dumps(e,sort_keys=True) not in prior_errors]
    snapshot['changes']={'finished_delta':snapshot['valid_finished']-previous.get('valid_finished',0),'case_changes':changes,'new_actionable_errors':new_errors,'recovered_errors':[json.loads(e) for e in prior_errors if e not in {json.dumps(x,sort_keys=True) for x in snapshot['errors']} ]}
    stamp=datetime.fromisoformat(args.heartbeat_time.replace('Z','+00:00')).astimezone(__import__('zoneinfo').ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S')
    save(STATE/'monitor'/f'{stamp}.json',snapshot);save(STATE/'monitor/latest.json',snapshot)
    print(json.dumps({'status':snapshot['status'],'finished':snapshot['valid_finished'],'requested':110,'errors':snapshot['errors'],'changes':snapshot['changes'],'running':[c for c in snapshot['cases'] if c['status'] in ['running','starting']],'local':snapshot['local']},ensure_ascii=False))

if __name__=='__main__':main()
