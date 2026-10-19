#!/usr/bin/env python3
"""Queue the actual AwomoPaw supervised learning harness on the official open suite."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import subprocess
import sys
import threading
import time

import gpt6_episode as control
import harness_utils as transport
from gpt6_failure_learning import digest, save, learning_eligibility
from gpt6_task_learning import empty_task_memory, validate_task_memory
from harness_models import SUPPORTED_MODELS, checked_model

SCHEMA = 'awomopaw.open-online-harness/v1'
TITLE = 'awomo-open-子任务监督与跨场景学习'
ARM = 'self_evolving'


def inventory(sim):
    tree = ast.parse((sim/'scripts/internal/task_inventory.py').read_text())
    groups = next(ast.literal_eval(node.value) for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Dict)
        and 'open' in ast.literal_eval(node.value))
    tasks = groups['open']
    root = sim/'Assets/Eval_Layout/RoboDojo/arx_x5'
    cases = []
    for task in tasks:
        index = 0
        for seed in sorted((p for p in root.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p:int(p.name)):
            files = sorted(seed.glob(task+'_*.json'), key=lambda p:int(p.stem.rsplit('_', 1)[1]))
            if [int(p.stem.rsplit('_', 1)[1]) for p in files] != list(range(len(files))):
                raise ValueError('Noncontiguous official layouts: '+task)
            for layout, path in enumerate(files):
                cases.append(dict(task=task, seed=int(seed.name), layout_id=layout, case_index=index,
                    layout_file=str(path), layout_sha256=digest(path)))
                index += 1
        if index == 0:
            raise ValueError('No official scenes: '+task)
    return tasks, cases


def identity(case):
    return case['task'], case['seed'], case['layout_id']


def verify_inventory(cases, baseline):
    left = {identity(c):c['layout_sha256'] for c in cases}
    right = {identity(c):c['layout_sha256'] for c in baseline}
    if len(left) != len(cases) or len(right) != len(baseline) or left != right:
        raise ValueError('Awomo and baseline must use identical unique official scenes and hashes')


def select_cases(tasks,cases,selected_tasks=None,seeds=None):
    selected_tasks=list(selected_tasks) if selected_tasks is not None else list(tasks)
    if not selected_tasks or len(set(selected_tasks))!=len(selected_tasks) or set(selected_tasks)-set(tasks):
        raise ValueError('Choose distinct official open tasks')
    available={c['seed'] for c in cases}
    seeds=sorted(available) if seeds is None else sorted(seeds)
    if not seeds or len(set(seeds))!=len(seeds) or set(seeds)-available:
        raise ValueError('Choose distinct available official seeds')
    result=[]
    for task in selected_tasks:
        selected=[c for c in cases if c['task']==task and c['seed'] in seeds]
        if {c['seed'] for c in selected}!=set(seeds):
            raise ValueError('Missing requested task/seed inventory')
        result.extend(dict(c,case_index=i) for i,c in enumerate(selected))
    return selected_tasks,result


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else ({} if default is None else default)


def aggregate(tasks, cases, rows, states, status):
    valid = [r for r in rows if r.get('valid_episode')]
    successes = sum(bool(r['success']) for r in valid)
    groups = []
    for task in tasks:
        selected = [r for r in valid if r['task']==task]
        wins = sum(bool(r['success']) for r in selected)
        requested = sum(c['task']==task for c in cases)
        groups.append(dict(task=task, requested=requested, finished=len(selected), valid_episodes=len(selected),
            successes=wins, success_rate=wins/len(selected) if selected else None,
            complete_scene_coverage=len(selected)==requested, **states.get(task, {'status':'queued'})))
    return dict(schema=SCHEMA, title=TITLE, node=control.NODE, updated_at=transport.now(), status=status,
        requested=len(cases), finished=len(valid), valid_episodes=len(valid), successes=successes,
        success_rate=successes/len(valid) if valid else None, complete_scene_coverage=len(valid)==len(cases),
        unscored_or_invalid=len(rows)-len(valid), tasks=groups,
        score_basis='one official primary episode per task/seed/layout; unscored attempts excluded',
        qualification='Online development success rate; incomplete coverage is provisional')


def comparison(tasks, awomo, baseline):
    ours = {identity(r):r for r in awomo if r.get('valid_episode')}
    theirs = {identity(r):r for r in baseline if r.get('valid_episode')}
    groups = []
    for task in tasks:
        keys = sorted(k for k in ours.keys() & theirs.keys() if k[0]==task)
        count = len(keys)
        groups.append(dict(task=task, complete_pairs=count,
            awomo_successes=sum(bool(ours[k]['success']) for k in keys),
            baseline_successes=sum(bool(theirs[k]['success']) for k in keys),
            paired_success_rate_delta=sum(int(ours[k]['success'])-int(theirs[k]['success']) for k in keys)/count if count else None,
            paired_mean_score_delta=sum(ours[k]['score']-theirs[k]['score'] for k in keys)/count if count else None,
            pairs=[dict(seed=k[1], layout_id=k[2], awomo_score=ours[k]['score'], baseline_score=theirs[k]['score'],
                awomo_success=ours[k]['success'], baseline_success=theirs[k]['success']) for k in keys]))
    return dict(tasks=groups, complete_pairs=sum(g['complete_pairs'] for g in groups),
        interpretation='Same native scenes and action budgets. Awomo uses online subtask supervision and post-task learning; '
            'RSI uses original CodeProposer/SelfHarness survivor comparisons, including a scored empty bootstrap. '
            'Additional model calls and incumbent trials differ. This compares complete harness protocols, not one isolated feature.')


def predecessor_completed(root):
    """A failed scheduler exit does not release its dependent batch."""
    marker = read(root/'scheduler-finished.json')
    summary = read(root/'summary.json')
    requested = summary.get('requested', 0)
    return (marker.get('status') == 'completed' and summary.get('status') == 'completed'
        and summary.get('complete_scene_coverage') is True and requested > 0
        and summary.get('finished') == summary.get('valid_episodes') == requested)


def task_run(task, cases, gpu, args, port):
    root = args.output/'tasks'/task
    experiment = args.output.name+':'+task
    memory = empty_task_memory(experiment, task)
    rows = read(root/'results.json', [])
    if [r['case_index'] for r in rows] != list(range(len(rows))) or any(not r.get('valid_episode') for r in rows):
        raise RuntimeError('Resume requires a contiguous prefix of valid official episodes')
    save(root/'process.json', dict(pid=os.getpid(), task=task, gpu=gpu, started_at=transport.now()))
    def status(state, **extra):
        save(root/'status.json',dict(status=state, gpu=gpu, updated_at=transport.now(), **extra))
    for case in cases:
        index = case['case_index']
        validate_task_memory(memory, experiment, index, task)
        snapshot = root/'learning/snapshots'/f'case-{index:03d}.json'
        if snapshot.exists():
            if read(snapshot) != memory:
                raise ValueError('Immutable task experience mismatch')
        else:
            save(snapshot, memory)
        folder = root/f'seed-{case["seed"]}'/f'layout-{case["layout_id"]:02d}'/ARM
        local = SimpleNamespace(**vars(args))
        local.output, local.port_base, local.experience_snapshot = root, port, snapshot
        if index < len(rows):
            row = rows[index]
            if identity(row) != identity(case) or row['layout_sha256'] != case['layout_sha256']:
                raise ValueError('Recorded native episode differs from requested scene')
        else:
            if folder.exists():
                raise RuntimeError('Unscored attempt requires review before resume: '+str(folder))
            status('waiting_for_gpu_memory', current_case=case)
            deadline = time.monotonic()+900
            while transport.gpu_memory(gpu)['used_mib'] > 1024:
                if time.monotonic() > deadline:
                    raise TimeoutError('GPU has an unrelated active process after queue lock acquisition')
                time.sleep(10)
            status('running', phase='episode', current_case=case, experience_version=memory['version'])
            print(transport.now(),'START',task,case['seed'],case['layout_id'],'GPU',gpu,flush=True)
            row = control.run_case(case, ARM, gpu, local)
            if not row.get('valid_episode'):
                status('needs_attention', current_case=case, error=row.get('error',row['status']))
                raise RuntimeError('Unscored harness/native attempt: '+str(row.get('error',row['status'])))
            rows.append(row)
            save(root/'results.json', rows)
            print(transport.now(),'END',task,case['seed'],case['layout_id'],row['score'],flush=True)
        if row['experience_sha256'] != digest(snapshot):
            raise ValueError('Episode used a different experience snapshot')
        learning = folder/'task-learning'
        eligibility = learning_eligibility(folder, allow_success=True)
        save(folder/'learning-eligibility.json', eligibility)
        if not eligibility['eligible']:
            if eligibility['infrastructure_error']:
                raise RuntimeError('Deployment failure cannot enter task learning: '+eligibility['reason'])
            save(folder/'learning-skipped.json', dict(status='skipped', reason=eligibility['reason'], no_model_call=True))
            continue
        status('running',phase='success_distillation' if row['success'] else 'failure_reflection',
               current_case=case, experience_version=memory['version'])
        receipt = learning/'receipt.json'
        if not receipt.exists():
            if learning.exists():
                # A partial post-task reflection has no score and can be retried
                # without replaying the already scored native episode.
                archive = folder/'learning-interruptions'/str(time.time_ns())
                archive.parent.mkdir(exist_ok=True)
                learning.rename(archive)
            command = [str(args.awomopaw/'.venv/bin/python'),
                str(Path(__file__).with_name('gpt6_task_learning.py')), '--case-dir',str(folder),
                '--snapshot',str(snapshot), '--agent-config',str(root/'agent-config.json'),
                '--output',str(learning), '--split','development']
            with (folder/'learning-process.log').open('ab',buffering=0) as stream:
                subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=650)
        learned = read(receipt)
        after = learning/'experience-after.json'
        if learned['before_sha256'] != digest(snapshot) or learned['after_sha256'] != digest(after):
            raise ValueError('Task learning provenance mismatch')
        memory = read(after)
        validate_task_memory(memory,experiment,index+1,task)
        save(root/'experience.json',memory)
    status('completed', experience_version=memory['version'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--agent-template',type=Path,required=True)
    p.add_argument('--awomopaw',type=Path,required=True)
    p.add_argument('--sim-root',type=Path,default=Path('/opt/robodojo'))
    p.add_argument('--gpus',type=int,nargs='+',default=list(range(8)))
    p.add_argument('--port-base',type=int,default=45000)
    p.add_argument('--case-timeout',type=int,default=5400)
    p.add_argument('--queue-timeout',type=int,default=2592000)
    p.add_argument('--segment-actions',type=int,choices=range(1,41),default=40)
    p.add_argument('--model',choices=SUPPORTED_MODELS,default='gpt-6-astra')
    p.add_argument('--tasks',nargs='+')
    p.add_argument('--seeds',type=int,nargs='+')
    p.add_argument('--title')
    p.add_argument('--predecessor',type=Path,help='Earlier batch whose scheduler must finish before this batch takes GPU locks')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if os.environ.get('P6000_ABLATION_VERIFIED_NODE') != control.NODE or args.sim_root != Path('/opt/robodojo'):
        raise RuntimeError('Restricted to the verified P6000 node 153')
    if len(set(args.gpus))!=len(args.gpus) or not 1<=len(args.gpus)<=8:
        raise ValueError('Choose one to eight distinct GPUs')
    tasks,cases=inventory(args.sim_root)
    baseline_manifest=read(args.baseline/'manifest.json')
    verify_inventory(cases,baseline_manifest['cases'])
    if len(tasks)!=8 or len(cases)!=1320 or any(sum(c['task']==t for c in cases)!=165 for t in tasks):
        raise ValueError('Expected the complete eight-task official open inventory')
    tasks,cases=select_cases(tasks,cases,args.tasks,args.seeds)
    selected_baseline=[c for c in baseline_manifest['cases'] if c['task'] in tasks and c['seed'] in {c['seed'] for c in cases}]
    verify_inventory(cases,selected_baseline)
    requested=len(cases)
    title=args.title or (TITLE if args.model=='gpt-6-astra' else 'awomo-'+args.model+'-open')
    args.output.mkdir(parents=True,exist_ok=args.resume)
    manifest_path=args.output/'manifest.json'
    if not args.resume:
        for name in ['agent-config.json','registry.snapshot.json']:
            shutil.copy2(args.agent_template/name,args.output/name)
        config=read(args.output/'agent-config.json')
        model=checked_model(args.model)
        config['active_model']['model']=model
        save(args.output/'agent-config.json',config)
        save(manifest_path,dict(schema=SCHEMA,title=title,node=control.NODE,tasks=tasks,cases=cases,requested=requested,
            model=model,provider='gpt6-harness',pretrained_skills=False,vla_used=False,initial_memory={},
            task_isolation=True,fresh_harness_per_episode=True,sequential_within_task=True,
            scene_order='seed ascending, layout ascending; task-local memory continues across seeds',
            learning_mode='task-artifacts',segment_actions=args.segment_actions,case_timeout=args.case_timeout,
            automatic_native_episode_retries=0,gpus=args.gpus,baseline=str(args.baseline),
            baseline_manifest_sha256=digest(args.baseline/'manifest.json'),created_at=transport.now(),
            official_action_budget_unchanged=True,post_task_learning_timeout=650,
            seeds=sorted({c['seed'] for c in cases}),predecessor=str(args.predecessor) if args.predecessor else None))
        save(args.output/'experiment.json',dict(schema=SCHEMA,title=title,framework='awomo',model=model))
        save(args.output/'official-case-inventory.json',cases)
        for task in tasks:
            root=args.output/'tasks'/task
            root.mkdir(parents=True)
            for name in ['agent-config.json','registry.snapshot.json']:
                shutil.copy2(args.output/name,root/name)
            save(root/'experience.json',empty_task_memory(args.output.name+':'+task,task))
            save(root/'results.json',[])
            save(root/'status.json',dict(status='queued'))
    else:
        manifest=read(manifest_path)
        if (manifest['cases']!=cases or manifest['segment_actions']!=args.segment_actions or
                manifest['gpus']!=args.gpus or manifest['case_timeout']!=args.case_timeout or
                manifest['model']!=args.model or manifest.get('predecessor')!=(str(args.predecessor) if args.predecessor else None)):
            raise ValueError('Cannot change an existing experiment protocol on resume')
    state={'status':'queued'}
    stop=threading.Event()
    def publish():
        rows,states,baseline_rows=[],{},[]
        for task in tasks:
            rows.extend(read(args.output/'tasks'/task/'results.json',[]))
            states[task]=read(args.output/'tasks'/task/'status.json')
            baseline_rows.extend(read(args.baseline/'tasks'/task/'results.json',[]))
        save(args.output/'results.json',rows)
        report=aggregate(tasks,cases,rows,states,state['status'])
        report.update(title=title,model=args.model,wait_reason=state.get('wait_reason'))
        save(args.output/'summary.json',report)
        save(args.output/'comparison.json',comparison(tasks,rows,baseline_rows))
    publish()
    if args.prepare_only:
        return
    save(args.output/'process.json',dict(pid=os.getpid(),command_marker='evaluate_awomo_open.py',
        node=control.NODE,gpus=args.gpus,started_at=transport.now()))
    def summaries():
        while not stop.wait(20):
            publish()
    thread=threading.Thread(target=summaries,daemon=True)
    thread.start()
    locks=[]
    try:
        deadline=time.monotonic()+args.queue_timeout
        if args.predecessor:
            state['wait_reason']='earlier_awomo_batch'
            while not predecessor_completed(args.predecessor):
                if time.monotonic()>deadline:
                    raise TimeoutError('Queue timeout waiting for the preceding Awomo batch')
                time.sleep(20)
        state['wait_reason']='shared_gpu_locks'
        locks=transport.acquire_gpu_locks(args,deadline)
        state['status']='running'
        state.pop('wait_reason',None)
        publish()
        def lane(index,gpu):
            for task in tasks[index::len(args.gpus)]:
                try:
                    task_run(task,[c for c in cases if c['task']==task],gpu,args,args.port_base+index*400)
                except Exception as error:
                    save(args.output/'tasks'/task/'status.json',dict(status='needs_attention',gpu=gpu,
                        error=str(error),error_type=type(error).__name__,updated_at=transport.now()))
                    print(task,'NEEDS_ATTENTION',str(error),flush=True)
        with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
            for future in as_completed([pool.submit(lane,i,g) for i,g in enumerate(args.gpus)]):
                future.result()
        state['status']='completed' if all(read(args.output/'tasks'/t/'status.json').get('status')=='completed' for t in tasks) else 'needs_attention'
    except BaseException as error:
        state['status']='needs_attention'
        save(args.output/'scheduler-error.json',dict(error=str(error),type=type(error).__name__,at=transport.now()))
        raise
    finally:
        stop.set()
        thread.join(timeout=25)
        publish()
        for lock in locks:
            lock.close()
        save(args.output/'scheduler-finished.json',dict(status=state['status'],finished_at=transport.now()))


if __name__=='__main__':
    main()
