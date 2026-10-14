#!/usr/bin/env python3
"""Evaluate every exact official scene with unchanged frozen providers.

Resumable finite queue, one simulator per GPU, immutable run manifest. Legitimate
scored failures are final; only infrastructure failures get bounded retries.
"""
from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import run_parallel_frozen_skills as base

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ports_for_case(port_base, index, scene_count, attempt):
    port=port_base+3*(index+scene_count*(attempt-1))
    if port+2>65535:raise ValueError('Suite exceeds available unique port range')
    return port


def inventory(sim_root, tasks):
    root = sim_root / 'Assets/Eval_Layout/RoboDojo/arx_x5'
    by_task = {task: [] for task in tasks}
    for directory in sorted((p for p in root.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name)):
        for task in tasks:
            pattern = re.compile(rf'{re.escape(task)}_\d+\.json')
            files = sorted((p for p in directory.iterdir() if pattern.fullmatch(p.name)), key=lambda p: int(p.stem.rsplit('_', 1)[1]))
            ids = [int(p.stem.rsplit('_', 1)[1]) for p in files]
            if ids != list(range(len(files))):
                raise ValueError(f'Noncontiguous official layouts for {task}, seed {directory.name}')
            for layout, path in enumerate(files):
                by_task[task].append(dict(task=task, seed=int(directory.name), layout=layout,
                    layout_file=str(path), layout_sha256=digest(path)))
    if any(not cases for cases in by_task.values()):
        raise ValueError('Requested task has no exact official scenes')
    # Interleave tasks so progress samples all four tasks throughout the sweep.
    cases = [dict(case) for row in itertools.zip_longest(*by_task.values()) for case in row if case]
    for index, case in enumerate(cases):
        case['index'] = index
        case['case_id'] = f'{index:04d}-{case["task"]}-seed-{case["seed"]}-layout-{case["layout"]}'
    return cases


def source_hashes(sim_root):
    catalog = json.loads((sim_root / 'XPolicyLab/policy/PhysicalRSI_Frozen_Skills/catalog.json').read_text())
    files = {}
    for entry in catalog['releases'].values():
        if entry['kind'] == 'code':
            files.update(entry['configuration']['files'])
    actual = {}
    for template, expected in files.items():
        path = Path(template.replace('{root}', catalog['asset_root']))
        actual[str(path)] = digest(path)
        if actual[str(path)] != expected:
            raise ValueError('Frozen source changed: ' + str(path))
    return actual


def aggregate(cases, results, attempts, status):
    rows = {row['index']: row for row in results}
    valid = [row for row in results if row.get('valid_episode')]
    groups = []
    for task in dict.fromkeys(case['task'] for case in cases):
        selected = [case for case in cases if case['task'] == task]
        scored = [row for row in valid if row['task'] == task]
        finished = [row for row in results if row['task'] == task]
        groups.append(dict(task=task, requested=len(selected), finished=len(finished), valid_episodes=len(scored),
            infrastructure_errors=len(finished)-len(scored), pending=len(selected)-len(finished),
            successes=sum(bool(row['success']) for row in scored),
            mean_score=sum(float(row['score']) for row in scored)/len(scored) if scored else None,
            success_rate=sum(bool(row['success']) for row in scored)/len(scored) if scored else None))
    means = [group['mean_score'] for group in groups if group['mean_score'] is not None]
    return dict(schema='awomopaw.full-frozen-scores/v1', status=status, updated_at=base.now(),
        requested=len(cases), finished=len(results), pending=len(cases)-len(results), valid_episodes=len(valid),
        successes=sum(bool(row['success']) for row in valid), infrastructure_errors=len(results)-len(valid),
        infrastructure_attempt_errors=sum(not row.get('valid_episode') for row in attempts),
        mean_score=sum(float(row['score']) for row in valid)/len(valid) if valid else None,
        macro_task_mean_score=sum(means)/len(means) if len(means)==len(groups) else None,
        success_rate=sum(bool(row['success']) for row in valid)/len(valid) if valid else None,
        tasks=groups, execution='direct_frozen_provider', llm_used=False,
        complete_scene_coverage=len(valid)==len(cases),
        score_basis='Official 0-100 episode scores. Infrastructure errors have no score and remain explicit; partial aggregates are provisional.')


def publish(output, cases, results, attempts, status):
    summary = aggregate(cases, results, attempts, status)
    base.save(output/'results.json', results)
    base.save(output/'attempts.json', attempts)
    base.save(output/'summary.json', summary)
    by_index = {row['index']: row for row in results}
    buffer = io.StringIO()
    fields = ['task','seed','layout','status','score','success','actions','attempt','run_id','output','layout_sha256']
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction='ignore')
    writer.writeheader()
    for case in cases:
        row = dict(case, status='pending')
        row.update(by_index.get(case['index'], {}))
        writer.writerow(row)
    (output/'scores.csv.tmp').write_text(buffer.getvalue())
    (output/'scores.csv.tmp').replace(output/'scores.csv')
    lines = ['# Four-task official-scene frozen skill evaluation', '',
        f'Status: {status}. Scored episodes: {summary["valid_episodes"]}/{len(cases)}. Node: 192.168.48.153.', '',
        'The unchanged RSI providers are invoked directly through AwomoPaw. No LLM planning is used. All exact official scenes from seeds 0, 1, and 2 are included. Scored failures are retained; only infrastructure errors are retried.', '',
        '| Task | Requested | Scored | Mean score /100 | Successes | Success rate | Infrastructure errors |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for group in summary['tasks']:
        mean = 'pending' if group['mean_score'] is None else f'{group["mean_score"]:.2f}'
        rate = 'pending' if group['success_rate'] is None else f'{100*group["success_rate"]:.2f}%'
        lines.append(f'| {group["task"]} | {group["requested"]} | {group["valid_episodes"]} | {mean} | {group["successes"]} | {rate} | {group["infrastructure_errors"]} |')
    if summary['mean_score'] is not None:
        lines.extend(['', f'Weighted mean over scored scenes: {summary["mean_score"]:.2f}/100.'])
    if summary['macro_task_mean_score'] is not None:
        lines.append(f'Mean of the four task averages: {summary["macro_task_mean_score"]:.2f}/100.')
    lines.extend(['', 'Any aggregate before complete_scene_coverage=true is provisional. See scores.csv for every requested scene, manifest.json for scene/source hashes, attempts.json for retries, and each attempt directory for official result, action receipts and three verified videos.', ''])
    (output/'report.md.tmp').write_text('\n'.join(lines))
    (output/'report.md.tmp').replace(output/'report.md')
    return summary


def worker(case, args):
    path = Path(case['layout_file'])
    if digest(path) != case['layout_sha256']:
        raise ValueError('Official scene changed after manifest creation')
    row = base.worker(case, args)
    if row.get('valid_episode'):
        try:
            official = json.loads((Path(case['output'])/'official-result.json').read_text())
            run = official['run']
            if run['task'] != case['task'] or run['seed'] != case['seed']:
                raise ValueError('Official scored task or seed differs from manifest')
            fixture = json.loads((Path(run['artifact_dir'])/'chat-eval-fixture.json').read_text())
            if fixture['layout_file'] != case['layout_file'] or fixture['layout_sha256'] != case['layout_sha256']:
                raise ValueError('Selected official scene differs from manifest')
            base.save(Path(case['output'])/'selected-scene.json', fixture)
            if not math.isfinite(float(row['score'])) or not 0 <= float(row['score']) <= 100:
                raise ValueError('Invalid official score')
            row['scene_identity_verified'] = True
        except Exception as error:
            row.update(valid_episode=False, status='infrastructure_error', error=str(error), error_type=type(error).__name__)
        base.save(Path(case['output'])/'receipt.json', row)
    return row


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', nargs='+', default=base.DEFAULT_TASKS)
    p.add_argument('--gpus', nargs='+', type=int, default=[1,2,3,4])
    p.add_argument('--max-parallel', type=int, default=4)
    p.add_argument('--min-free-mib', type=int, default=11000)
    p.add_argument('--case-timeout', type=int, default=3600)
    p.add_argument('--queue-timeout', type=int, default=604800)
    p.add_argument('--startup-spacing', type=int, default=10)
    p.add_argument('--infra-retries', type=int, default=2)
    p.add_argument('--port-base', type=int, default=8320)
    p.add_argument('--sim-root', type=Path, required=True)
    p.add_argument('--awomopaw', type=Path, required=True)
    p.add_argument('--registry', type=Path, required=True)
    p.add_argument('--task-providers', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--detach', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--worker-case', type=Path)
    return p


def main():
    p = parser(); args = p.parse_args(); args.output=args.output.resolve()
    if len(set(args.gpus)) != len(args.gpus) or any(gpu<0 for gpu in args.gpus) or not 1<=args.max_parallel<=min(4,len(args.gpus)):
        p.error('Use distinct GPU IDs, at most four simulators and one per GPU')
    if args.infra_retries<0 or args.min_free_mib<4000:
        p.error('Invalid retries or GPU memory bound')
    if args.worker_case:
        row=worker(json.loads(args.worker_case.read_text()),args)
        return 0 if row['valid_episode'] else 1
    if args.detach:
        args.output.mkdir(parents=True, exist_ok=args.resume)
        argv=[item for item in sys.argv[1:] if item!='--detach']
        with (args.output/'scheduler.log').open('ab',buffering=0) as log:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*argv],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        base.save(args.output/'process.json',dict(pid=process.pid,started_at=base.now(),command=argv))
        print(json.dumps(dict(pid=process.pid,output=str(args.output))),flush=True);return 0
    args.output.mkdir(parents=True,exist_ok=True)
    lock=(args.output/'scheduler.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest_path=args.output/'manifest.json'
    script_hashes={str(path.relative_to(ROOT)):digest(path) for path in sorted((ROOT/'scripts').glob('*.py'))}
    hashes=source_hashes(args.sim_root)
    registry=json.loads(args.registry.read_text());mapping=json.loads(args.task_providers.read_text())
    cases=inventory(args.sim_root,args.tasks)
    for case in cases:
        provider=mapping[case['task']]
        case.update(provider_id=provider['provider_id'],action_type=provider['action_type'])
        if not any(r['provider_id']==case['provider_id'] and r['kind']=='code' for r in registry):
            raise ValueError('Frozen provider unavailable: '+case['task'])
    identity=dict(cases=cases,registry_sha256=digest(args.registry),source_hashes=hashes,script_hashes=script_hashes)
    if manifest_path.exists():
        if not args.resume:raise FileExistsError('Existing experiment requires --resume')
        original=json.loads(manifest_path.read_text())
        for key,value in identity.items():
            if original[key]!=value:raise ValueError('Resume identity changed: '+key)
    else:
        base.save(manifest_path,dict(schema='awomopaw.full-official-scenes/v1',created_at=base.now(),
            execution='direct_frozen_provider',llm_used=False,gpus=args.gpus,max_parallel=args.max_parallel,
            infra_retries=args.infra_retries,**identity))
        base.save(args.output/'registry.snapshot.json',registry)
        base.save(args.output/'task-providers.snapshot.json',mapping)
    attempts=[];results=[];pending=[]
    for case in cases:
        history=[]
        for receipt in sorted((args.output/'cases'/case['case_id']).glob('attempt-*/receipt.json')):
            row=json.loads(receipt.read_text());history.append(row);attempts.append(row)
        valid=next((row for row in history if row.get('valid_episode')),None)
        if valid:results.append(valid)
        elif len(history)>args.infra_retries:results.append(history[-1])
        else:pending.append((case,len(history)+1))
    running={};locks=[];started=time.monotonic();last_launch=-float('inf')
    def interrupted(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        locks=base.acquire_gpu_locks(args,started+args.queue_timeout)
        publish(args.output,cases,results,attempts,'running')
        while pending or running:
            for gpu,(process,case,launched) in list(running.items()):
                if process.poll() is None and time.monotonic()-launched>args.case_timeout+60:base.stop_worker(process,case)
                if process.poll() is not None:
                    base.stop_worker(process,case)
                    path=Path(case['output'])/'receipt.json'
                    row=json.loads(path.read_text()) if path.exists() else dict(case,status='worker_crashed',valid_episode=False)
                    row['worker_return_code']=process.returncode
                    base.save(path,row);attempts.append(row);del running[gpu]
                    if not row.get('valid_episode') and case['attempt']<=args.infra_retries:
                        original=cases[case['index']];pending.append((original,case['attempt']+1))
                    else:results.append(row)
                    publish(args.output,cases,results,attempts,'running')
                    print(base.now(),'END',case['case_id'],row['status'],'score',row.get('score'),flush=True)
            owned=[]
            for process,case,_ in running.values():
                owned.append(process.pid)
                for name in ('run.json','receipt.json'):
                    path=Path(case['output'])/name
                    if path.exists():
                        info=json.loads(path.read_text());owned.extend(info.get(key) for key in ('pid','evaluator_pid','bridge_pid') if info.get(key))
            reasons=base.external_busy(owned)
            available=[gpu for gpu in args.gpus if gpu not in running and base.gpu_memory(gpu)['free_mib']>=args.min_free_mib]
            base.save(args.output/'status.json',dict(status='running',updated_at=base.now(),
                requested=len(cases),finished=len(results),pending=len(pending),attempts=len(attempts),
                running=[case for _,case,_ in running.values()],blocking=reasons,available_gpus=available))
            if time.monotonic()-started>args.queue_timeout:raise TimeoutError('Full sweep exceeded queue timeout')
            if pending and available and len(running)<args.max_parallel and not reasons and time.monotonic()-last_launch>=args.startup_spacing:
                original,attempt=pending.pop(0);gpu=available[0]
                # Fresh ports per scene and retry avoid TCP TIME_WAIT from the
                # preceding simulator. The full official suite fits below 65536.
                port=ports_for_case(args.port_base,original['index'],len(cases),attempt)
                case=dict(original,attempt=attempt,gpu=gpu,bridge_port=port,control_port=port+1,policy_port=port+2,
                    output=str(args.output/'cases'/original['case_id']/f'attempt-{attempt:02d}'))
                if not all(base.port_available(port+i) for i in range(3)):raise RuntimeError('Port conflict on GPU lane '+str(gpu))
                dest=Path(case['output']);dest.mkdir(parents=True,exist_ok=False);base.save(dest/'case.json',case)
                argv=[item for item in sys.argv[1:] if item not in {'--detach','--resume'}]
                with (dest/'driver.log').open('ab',buffering=0) as log:
                    process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),*argv,'--worker-case',str(dest/'case.json')],
                        stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                running[gpu]=(process,case,time.monotonic());last_launch=time.monotonic()
                print(base.now(),'START',case['case_id'],'attempt',attempt,'GPU',gpu,'PID',process.pid,flush=True)
            time.sleep(5)
        after=source_hashes(args.sim_root)
        if after!=hashes:raise ValueError('Frozen source hashes changed during evaluation')
        base.save(args.output/'source-hashes-after.json',dict(verified_files=len(after),unchanged=True,hashes=after))
        status='completed' if all(row.get('valid_episode') for row in results) else 'completed_with_infrastructure_errors'
        summary=publish(args.output,cases,results,attempts,status)
        base.save(args.output/'status.json',dict(status=status,updated_at=base.now(),requested=len(cases),finished=len(results),running=[]))
        print(json.dumps(summary),flush=True)
        return 0 if summary['complete_scene_coverage'] else 1
    except BaseException as error:
        base.save(args.output/'scheduler-error.json',dict(type=type(error).__name__,error=str(error),at=base.now()))
        publish(args.output,cases,results,attempts,'failed')
        base.save(args.output/'status.json',dict(status='failed',updated_at=base.now(),error=str(error)))
        raise
    finally:
        for process,case,_ in running.values():base.stop_worker(process,case)
        for stream in locks:stream.close()
        lock.close()


if __name__=='__main__':raise SystemExit(main())
