"""Shared utilities for AwomoPaw episode launchers."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.request
ROOT = Path(__file__).resolve().parents[1]

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


def acquire_gpu_locks(args, deadline):
    directory = Path(os.environ.get('AWOMOPAW_GPU_LOCK_DIR', str(ROOT / 'runtime/gpu-locks')))
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
            save(args.output / 'status.json', {'status': 'waiting', 'wait_reason': 'another_evaluation',
                'gpus': args.gpus, 'updated_at': now()})
            if time.monotonic() > deadline:
                raise TimeoutError('Queue timeout while waiting for the evaluation GPU locks')
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

