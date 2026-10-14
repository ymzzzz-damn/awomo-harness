#!/usr/bin/env python3
"""Queue isolated direct frozen-provider evaluations, across one or several GPUs.

Detachable scheduler; no Chat, LLM planning, service reconfiguration or old-process
termination. One official native layout per task, with every failure retained.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
AWOMO = Path('/home/ymz/physclaw/awomopaw')
SIM = Path('/home/ymz/robodojo-pi05/RoboDojo')
DEFAULT_TASKS = ['push_T', 'stack_blocks_by_language', 'press_by_number', 'general_pickup']
RUNNER_MARKERS = ['evaluate_frozen_align_skill.py', 'run_gpt56_skill_chats.py',
                  'compare_local_tasks.py', 'run_evolution.py', 'simulator-entry.py',
                  'run_embodiment_chats.py', 'evaluate_fresh_chat_align.py', 'run_code_policy_fallback.py', 'src/eval_client/main.py', 'run_policy_eval.sh']


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def api(base, path, payload=None, timeout=15):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base + path, data=data, headers={'Content-Type': 'application/json'})
    # Loopback requests must not traverse configured proxy servers.
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=timeout) as response:
        return json.load(response)


def gpu_memory(gpu):
    output = subprocess.check_output(['nvidia-smi', '-i', str(gpu),
        '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'], text=True)
    total, used, free = [int(value.strip()) for value in output.splitlines()[0].split(',')]
    return {'total_mib': total, 'used_mib': used, 'free_mib': free}


def external_busy(owned_groups=()):
    reasons = []
    try:
        active = api('http://127.0.0.1:8098', '/health', timeout=5).get('active_evaluation')
        if active and active['status'] in {'starting', 'running'} and active.get('pid') not in owned_groups:
            reasons.append({'kind': 'live_bridge', 'run_id': active['run_id'], 'task': active.get('task')})
    except urllib.error.URLError:
        pass
    rows = subprocess.check_output(['ps', '-eo', 'pid=,ppid=,pgid=,args='], text=True)
    entries = []
    for row in rows.splitlines():
        pid, ppid, pgid, command = row.strip().split(None, 3)
        entries.append((int(pid), int(ppid), int(pgid), command))
    owned_pids = set(owned_groups)
    while True:
        children = {pid for pid, ppid, pgid, command in entries if ppid in owned_pids}
        expanded = owned_pids | children
        if expanded == owned_pids:
            break
        owned_pids = expanded
    all_owned_groups = set(owned_groups) | {pgid for pid, ppid, pgid, command in entries if pid in owned_pids}
    for pid, ppid, pgid, command in entries:
        # Match executable invocations, not shell commands quoting source text.
        executable = Path(command.split(None, 1)[0]).name
        is_interpreter = executable == 'python' or executable.startswith('python3') or executable in {'node', 'bash', 'sh'}
        if (pid == os.getpid() or not is_interpreter or pgid in all_owned_groups
            or command.startswith(('bash -', 'sh -', '/bin/bash -', '/bin/sh -'))):
            continue
        if any(marker in command for marker in RUNNER_MARKERS):
            reasons.append({'kind': 'existing_runner', 'pid': pid, 'command': command[:500]})
    return reasons


def stop_group(pid):
    if pid is None:
        return
    # Only groups created by this launcher or the isolated bridge are accepted.
    try:
        if os.getpgid(pid) != pid:
            raise RuntimeError('Refusing cleanup of non-owned process group')
    except ProcessLookupError:
        pass  # Descendants can outlive the owned group leader.
    rows = subprocess.check_output(['ps', '-eo', 'pid=,ppid='], text=True)
    children = {}
    for row in rows.splitlines():
        child, parent = map(int, row.split())
        children.setdefault(parent, []).append(child)
    def descendants(parent):
        result = []
        for child in children.get(parent, []):
            result.extend(descendants(child))
            result.append(child)
        return result
    owned_descendants = descendants(pid)
    for sig in (signal.SIGTERM, signal.SIGKILL):
        # Frozen program workers may own nested process groups. Capture their
        # lineage before terminating the evaluator so they cannot be orphaned.
        for child in owned_descendants:
            try:
                os.kill(child, sig)
            except ProcessLookupError:
                pass
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            pass
        if sig == signal.SIGTERM:
            time.sleep(2)


def stop_worker(process, case):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=25)
        except subprocess.TimeoutExpired:
            pass
    # Worker and bridge/evaluator each own separate groups; also clean up after
    # a worker crash that bypassed its finally clause.
    for name in ('run.json', 'receipt.json'):
        path = Path(case['output']) / name
        if path.exists():
            info = json.loads(path.read_text())
            for key in ('pid', 'evaluator_pid', 'bridge_pid'):
                stop_group(info.get(key))
    stop_group(process.pid)
    process.wait(timeout=10)


def worker(case, args):
    dest = Path(case['output'])
    dest.mkdir(parents=True, exist_ok=True)
    row = dict(case, status='starting', started_at=now(), valid_episode=False,
               execution='direct_frozen_provider', llm_used=False, actions=0)
    save(dest / 'receipt.json', row)
    bridge = run = None
    base = 'http://127.0.0.1:' + str(case['bridge_port'])
    deadline = time.monotonic() + args.case_timeout
    def check_time():
        if time.monotonic() >= deadline:
            raise TimeoutError('Frozen provider case exceeded its wall-clock timeout')
    def call(path, payload=None, timeout=240):
        check_time()
        return api(base, path, payload, timeout=max(1, min(timeout, deadline - time.monotonic())))
    def interrupted(signum, frame):
        raise RuntimeError('Worker interrupted by signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        if case['action_type'] not in {'joint', 'ee'}:
            row.update(status='unsupported_action_type', error='Frozen embodiment accepts joint or ee actions only')
            return row
        fixture = dest / 'fixture.json'
        save(fixture, {'task': case['task'], 'seed': case['seed'], 'layout_id': case['layout'],
            'test_id': str(args.output), 'scene_index': case['index'] + 1})
        env = os.environ.copy()
        env.update(ROBODOJO_ROOT=str(args.sim_root), ROBODOJO_POLICY_REGISTRY_JSON=(args.output / 'registry.snapshot.json').read_text(),
            ROBODOJO_HARNESS_STATE_DIR=str(dest / 'bridge-state'),
            ROBODOJO_ENV_GPU=str(case['gpu']), ROBODOJO_POLICY_GPU=str(case['gpu']),
            ROBODOJO_ACTION_TYPE=case['action_type'], AWOMOPAW_HARNESS_PORT=str(case['control_port']),
            ROBODOJO_HARNESS_POLICY_DIR='XPolicyLab/policy/PhysicalRSI_Frozen_Skills',
            ROBODOJO_HARNESS_POLICY_NAME='PhysicalRSI_Frozen_Skills',
            PYTHONPATH=os.pathsep.join(filter(None, [str(args.awomopaw / 'src'),
                os.environ.get('ROBODOJO_COMPAT_PYTHONPATH', '')])), PYTHONUNBUFFERED='1')
        with (dest / 'bridge.log').open('ab', buffering=0) as log:
            bridge = subprocess.Popen([str(args.awomopaw / '.venv/bin/python'),
                str(ROOT / 'scripts/frozen_skill_parallel_bridge.py'), '--fixture', str(fixture),
                '--port', str(case['bridge_port']), '--policy-port', str(case['policy_port'])],
                cwd=args.awomopaw, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        row['bridge_pid'] = bridge.pid
        save(dest / 'receipt.json', row)
        for _ in range(90):
            check_time()
            if bridge.poll() is not None:
                raise RuntimeError('Isolated bridge exited: ' + str(bridge.returncode))
            try:
                health = call('/health', timeout=5)
                break
            except urllib.error.URLError:
                time.sleep(1)
        else:
            raise TimeoutError('Isolated bridge did not become ready')
        save(dest / 'health.json', health)
        providers = {p['provider_id']: p for p in health['available_policies']}
        provider = providers[case['provider_id']]
        if not provider['ready'] or provider['kind'] != 'code':
            raise RuntimeError('Frozen provider is not ready: ' + json.dumps(provider))
        run = call('/v1/sessions', {'task': case['task'], 'seed': case['seed'], 'policy_id': case['provider_id']})
        save(dest / 'run.json', run)
        row.update(status='initializing', run_id=run['run_id'], evaluator_pid=run['pid'])
        save(dest / 'receipt.json', row)
        initialized = time.monotonic() + 1200
        while time.monotonic() < initialized:
            check_time()
            state = call('/v1/evaluations/status?run_id=' + run['run_id'])
            if state['status'] not in {'starting', 'running'}:
                raise RuntimeError('Evaluator ended during initialization: ' + json.dumps(state))
            try:
                obs = call('/v1/sessions/observation', timeout=30)
                if obs.get('observation_id'):
                    break
            except urllib.error.HTTPError as error:
                if error.code != 409:
                    raise
            time.sleep(3)
        else:
            raise TimeoutError('Initial public observation did not become ready')
        instruction = obs.get('instruction')
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError('Official task instruction is missing')
        row.update(status='running', initialized_at=now())
        save(dest / 'receipt.json', row)
        save(dest / 'initial-observation.json', obs)
        for step in range(int(obs['action_limit']) + 1):
            check_time()
            if obs.get('episode_end'):
                break
            if obs['instruction'] != instruction:
                raise ValueError('Official instruction changed within the episode')
            chunk = call('/v1/sessions/policy-chunks', {
                'observation_id': obs['observation_id'], 'provider_id': case['provider_id'],
                'subtask_goal': instruction, 'optimized_task_prompt': instruction,
                'planning_summary': 'Direct invocation of an unchanged frozen PhysicalRSI provider; no LLM'})
            result = call('/v1/sessions/action-chunks/execute', {
                'chunk_id': chunk['chunk_id'], 'max_actions': 10, 'discard_remaining': True})
            save(dest / 'execution' / f'{step:04d}.json', result)
            row['actions'] += result['executed']
            save(dest / 'receipt.json', row)
            if result.get('episode_end'):
                break
            obs = call('/v1/sessions/observation')
        else:
            raise RuntimeError('Episode exceeded advertised action limit')
        finish_deadline = min(deadline, time.monotonic() + 180)
        while time.monotonic() < finish_deadline:
            state = call('/v1/evaluations/status?run_id=' + run['run_id'])
            if state['status'] not in {'starting', 'running'}:
                break
            time.sleep(2)
        official = call('/v1/evaluations/result?run_id=' + run['run_id'])
        save(dest / 'official-result.json', official)
        result = official['run']['result']
        if result.get('eval_time') != 1 or len(result.get('details', {})) != 1:
            raise ValueError('Expected exactly one official scored episode')
        detail = next(iter(result['details'].values()))
        if detail['layout_id'] != case['layout']:
            raise ValueError('Evaluator scored a different native layout')
        videos = []
        for original in official.get('videos', []):
            target = dest / 'videos' / Path(original).name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, target)
            probe = subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,nb_read_frames', '-of', 'json', str(target)], text=True, timeout=60)
            stream = json.loads(probe)['streams'][0]
            if int(stream['nb_read_frames']) <= 0:
                raise ValueError('Video has no decodable frames')
            videos.append({'path': str(target), 'original': original,
                'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), **stream})
        save(dest / 'videos.json', videos)
        if official.get('harness_trace'):
            shutil.copy2(official['harness_trace'], dest / 'harness-trace.jsonl')
        runtime = args.sim_root / 'eval_result/frozen_skill_runs' / run['run_id']
        if runtime.exists():
            shutil.copytree(runtime, dest / 'skill-runtime', dirs_exist_ok=True)
        if len(videos) != 3:
            raise ValueError('Expected head and both wrist camera videos')
        row.update(status='completed', valid_episode=True, score=result['score'],
            success=bool(official['task_succeeded']), details=result['details'], videos=videos)
    except Exception as error:
        row.update(status='infrastructure_error', error=str(error), error_type=type(error).__name__)
        if run:
            try:
                state = api(base, '/v1/evaluations/status?run_id=' + run['run_id'], timeout=5)
                save(dest / 'failure-evaluation-state.json', state)
                if state['status'] not in {'starting', 'running'}:
                    save(dest / 'failure-official-result.json', api(base, '/v1/evaluations/result?run_id=' + run['run_id'], timeout=5))
            except Exception as result_error:
                row['failure_result_error'] = str(result_error)
            try:
                api(base, '/v1/sessions/abort', {'reason': 'Direct frozen evaluation error: ' + str(error)[:400]}, timeout=10)
            except Exception as cleanup_error:
                row['abort_error'] = str(cleanup_error)
    finally:
        if run:
            stop_group(run.get('pid'))
        if bridge:
            stop_group(bridge.pid)
            bridge.wait(timeout=10)
        row['finished_at'] = now()
        row['owned_services_cleaned'] = True
        save(dest / 'receipt.json', row)
    return row


def acquire_gpu_locks(args, deadline):
    directory = Path(os.environ.get('ROBODOJO_FROZEN_LOCK_DIR', str(ROOT / 'runtime/physicalrsi-all-skills')))
    directory.mkdir(parents=True, exist_ok=True)
    # Acquire the entire requested GPU set atomically. Release partial locks
    # before retrying so overlapping batches cannot deadlock.
    while True:
        streams = []
        try:
            for gpu in sorted(args.gpus):
                stream = (directory / f'gpu-{gpu}-evaluation.lock').open('a+')
                streams.append(stream)
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for gpu, stream in zip(sorted(args.gpus), streams):
                stream.seek(0)
                stream.truncate()
                stream.write(json.dumps({'pid': os.getpid(), 'gpu': gpu, 'output': str(args.output), 'at': now()}) + '\n')
                stream.flush()
            return streams
        except BlockingIOError:
            for stream in streams:
                stream.close()
            save(args.output / 'status.json', {'status': 'waiting', 'wait_reason': 'another_frozen_batch',
                'gpus': args.gpus, 'updated_at': now()})
            if time.monotonic() > deadline:
                raise TimeoutError('Queue timeout while waiting for the frozen batch GPU locks')
            time.sleep(15)
        except BaseException:
            for stream in streams:
                stream.close()
            raise


def port_available(port):
    with socket.socket() as probe:
        try:
            probe.bind(('127.0.0.1', port))
            return True
        except OSError:
            return False


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks', nargs='+', default=DEFAULT_TASKS)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--layout', type=int, default=0)
    p.add_argument('--max-parallel', type=int, choices=[1, 2, 3, 4], default=2)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--gpus', nargs='+', type=int, help='Distinct GPU IDs, assigned round-robin to cases')
    p.add_argument('--min-free-mib', type=int, default=11000)
    p.add_argument('--case-timeout', type=int, default=3600)
    p.add_argument('--queue-timeout', type=int, default=43200)
    p.add_argument('--startup-spacing', type=int, help='Launch spacing in seconds; default 90 for one GPU, 10 for several')
    p.add_argument('--registry', type=Path, default=ROOT / 'runtime/physicalrsi-all-skills/registry.json')
    p.add_argument('--task-providers', type=Path, default=ROOT / 'runtime/physicalrsi-all-skills/task-providers.json')
    p.add_argument('--sim-root', type=Path, default=SIM)
    p.add_argument('--awomopaw', type=Path, default=AWOMO)
    p.add_argument('--port-base', type=int, default=8120)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--detach', action='store_true')
    p.add_argument('--worker-case', type=Path, help=argparse.SUPPRESS)
    return p


def main():
    p = parser()
    args = p.parse_args()
    args.output = args.output.resolve()
    args.gpus = args.gpus if args.gpus is not None else [args.gpu]
    if len(set(args.gpus)) != len(args.gpus) or any(gpu < 0 for gpu in args.gpus):
        p.error('--gpus must contain distinct nonnegative GPU IDs')
    if args.max_parallel > (2 if len(args.gpus) == 1 else len(args.gpus)):
        p.error('At most two cases may share one GPU; with several GPUs use at most one case per GPU')
    if args.startup_spacing is None:
        args.startup_spacing = 90 if len(args.gpus) == 1 else 10
    if args.worker_case:
        row = worker(json.loads(args.worker_case.read_text()), args)
        return 0 if row['valid_episode'] else 1
    if args.seed < 0 or args.layout < 0 or args.min_free_mib < 4000:
        p.error('Invalid seed/layout or unsafe minimum free GPU memory')
    registry = json.loads(args.registry.read_text())
    available = {item['provider_id'] for item in registry if item['kind'] == 'code'}
    mapping = json.loads(args.task_providers.read_text()) if args.task_providers.exists() else {}
    cases = []
    for index, task in enumerate(args.tasks):
        item = mapping.get(task, {'provider_id': 'physicalrsi-' + task.replace('_', '-').lower(), 'action_type': 'joint'})
        if isinstance(item, str):
            item = {'provider_id': item, 'action_type': 'joint'}
        if item['provider_id'] not in available:
            p.error('Task has no registered frozen provider: ' + task)
        base = args.port_base + index * 3
        cases.append(dict(index=index, task=task, provider_id=item['provider_id'], action_type=item['action_type'],
            seed=args.seed, layout=args.layout, gpu=args.gpus[index % len(args.gpus)], bridge_port=base, control_port=base + 1, policy_port=base + 2,
            output=str(args.output / f'{index:02d}-{task}-seed-{args.seed}-layout-{args.layout}')))
    if args.detach:
        args.output.mkdir(parents=True, exist_ok=False)
        argv = [argument for argument in sys.argv[1:] if argument != '--detach']
        with (args.output / 'scheduler.log').open('ab', buffering=0) as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *argv],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save(args.output / 'process.json', {'pid': process.pid, 'started_at': now(), 'command': argv})
        print(json.dumps({'pid': process.pid, 'output': str(args.output)}), flush=True)
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'manifest.json').exists():
        raise FileExistsError('Refusing reuse of an existing experiment manifest')
    save(args.output / 'manifest.json', {'schema': 'awomopaw.parallel-frozen-evaluation/v1',
        'created_at': now(), 'pid': os.getpid(), 'execution': 'direct_frozen_provider',
        'llm_used': False, 'chat_invoked': False, 'max_parallel': args.max_parallel, 'gpu': args.gpu, 'gpus': args.gpus,
        'per_gpu_concurrency_limit': 2 if len(args.gpus) == 1 else 1, 'startup_spacing': args.startup_spacing,
        'min_free_mib': args.min_free_mib, 'registry': registry,
        'registry_sha256': hashlib.sha256(args.registry.read_bytes()).hexdigest(), 'cases': cases,
        'qualification': 'Small-sample frozen provider screening, not Chat agent planning or leaderboard evidence'})
    save(args.output / 'registry.snapshot.json', registry)
    save(args.output / 'task-providers.snapshot.json', mapping)
    save(args.output / 'source-hashes.json', {name: hashlib.sha256((ROOT / 'scripts' / name).read_bytes()).hexdigest()
        for name in ['run_parallel_frozen_skills.py', 'frozen_skill_parallel_bridge.py',
                     'frozen_skill_episode_process.py', 'chat_eval_bridge.py']})
    pending = list(cases)
    running = {}
    results = []
    started = time.monotonic()
    last_launch = -float('inf')
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    gpu_locks = []
    try:
        gpu_locks = acquire_gpu_locks(args, started + args.queue_timeout)
        while pending or running:
            for index, (process, case, launched) in list(running.items()):
                if process.poll() is None and time.monotonic() - launched > args.case_timeout + 60:
                    stop_worker(process, case)
                if process.poll() is not None:
                    stop_worker(process, case)
                    receipt = Path(case['output']) / 'receipt.json'
                    row = json.loads(receipt.read_text()) if receipt.exists() else dict(case, status='worker_crashed', valid_episode=False)
                    row['worker_return_code'] = process.returncode
                    results.append(row)
                    del running[index]
                    save(args.output / 'results.json', results)
            owned_groups = []
            for process, case, _ in running.values():
                owned_groups.append(process.pid)
                for name in ('run.json', 'receipt.json'):
                    path = Path(case['output']) / name
                    if path.exists():
                        info = json.loads(path.read_text())
                        owned_groups.extend(info.get(key) for key in ('pid', 'evaluator_pid', 'bridge_pid') if info.get(key))
            reasons = external_busy(owned_groups)
            per_gpu_limit = 2 if len(args.gpus) == 1 else 1
            counts = {gpu: sum(case['gpu'] == gpu for _, case, _ in running.values()) for gpu in args.gpus}
            next_case = next((case for case in pending if counts[case['gpu']] < per_gpu_limit), None)
            next_gpu = next_case['gpu'] if next_case else args.gpus[0]
            memory = gpu_memory(next_gpu)
            startup = []
            for _, case, _ in running.values():
                receipt = Path(case['output']) / 'receipt.json'
                state = json.loads(receipt.read_text()) if receipt.exists() else {}
                if not state.get('initialized_at') and case['gpu'] == next_gpu:
                    startup.append(case['task'])
            can_launch = (next_case is not None and len(running) < args.max_parallel and not reasons and not startup
                and memory['free_mib'] >= args.min_free_mib and time.monotonic() - last_launch >= args.startup_spacing)
            wait_reason = ('existing_evaluation' if reasons else 'initializing_owned_task_on_gpu' if startup else
                           'gpu_memory' if memory['free_mib'] < args.min_free_mib else 'concurrency_or_spacing')
            save(args.output / 'status.json', {'updated_at': now(), 'status': 'running' if running else 'waiting',
                'wait_reason': wait_reason, 'blocking': reasons, 'initializing': startup,
                'gpu': dict(index=next_gpu, **memory), 'gpus': args.gpus, 'gpu_running_counts': counts,
                'pending': [case['task'] for case in pending], 'running': [case for _, case, _ in running.values()],
                'completed': len(results)})
            if time.monotonic() - started > args.queue_timeout:
                raise TimeoutError('Scheduler queue timeout exceeded')
            if can_launch:
                case = next_case
                pending.remove(case)
                ports = [case[k] for k in ('bridge_port', 'control_port', 'policy_port')]
                if not all(port_available(port) for port in ports):
                    row = dict(case, status='port_conflict', valid_episode=False)
                    results.append(row)
                    save(args.output / 'results.json', results)
                    continue
                dest = Path(case['output'])
                dest.mkdir(parents=True, exist_ok=False)
                save(dest / 'case.json', case)
                argv = [argument for argument in sys.argv[1:] if argument != '--detach']
                with (dest / 'driver.log').open('ab', buffering=0) as log:
                    process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *argv,
                        '--worker-case', str(dest / 'case.json')], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                running[case['index']] = (process, case, time.monotonic())
                last_launch = time.monotonic()
                print(now(), 'START', case['task'], 'GPU', case['gpu'], 'worker', process.pid, flush=True)
            time.sleep(5 if running else 15)
        valid = [r for r in results if r.get('valid_episode')]
        save(args.output / 'summary.json', {'status': 'completed', 'finished_at': now(),
            'requested': len(cases), 'valid_episodes': len(valid),
            'successes': sum(bool(r['success']) for r in valid),
            'infrastructure_errors': len(results) - len(valid), 'episodes': results})
        save(args.output / 'status.json', {'status': 'completed', 'updated_at': now(), 'completed': len(results)})
        return 0 if len(valid) == len(cases) else 1
    except BaseException as error:
        save(args.output / 'scheduler-error.json', {'type': type(error).__name__, 'error': str(error), 'at': now()})
        save(args.output / 'status.json', {'status': 'failed', 'updated_at': now(), 'error': str(error)})
        raise
    finally:
        for process, case, _ in running.values():
            stop_worker(process, case)
        for gpu_lock in gpu_locks:
            gpu_lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
