#!/usr/bin/env python3
"""Supervise this deployment using the already allocated node-153 pod."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path('/mnt/auwomo-data/wangxin/qwenpaw-harness/physicalrsi-all-skills/20261013-195000')
POD = 'qwenpaw-sim-harness-5b46b76889-hsv7r'
PYTHON = '/mnt/auwomo-data/wangxin/qwenpaw-harness/releases/20261012-e6e0dcd8-663991b07e95/runtime/venv/bin/python'
ASSETS = ROOT / 'runtime/physicalrsi-skills'
STATE_POD = Path('/opt/robodojo/eval_result/physicalrsi-all-skills') / ROOT.name
STATE_HOST = ROOT.parent.parent / 'eval_result/physicalrsi-all-skills' / ROOT.name


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def stage_status(stage, **extra):
    save(ROOT / 'deployment-status.json', {'stage': stage, 'updated_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **extra})
    print(time.strftime('%H:%M:%S', time.gmtime()), stage, flush=True)


def wait_archive(name, expected):
    path = ROOT / 'uploads' / name
    deadline = time.monotonic() + 14400
    while not path.exists() or path.stat().st_size != expected['size']:
        if time.monotonic() >= deadline:
            raise TimeoutError('Upload incomplete: ' + name)
        time.sleep(5)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected['sha256']:
        raise ValueError('Archive hash mismatch: ' + name)
    return path


def extract(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(['tar', '--zstd', '-xf', str(archive), '-C', str(destination)], check=True)


def pod_run(command, environment=None):
    env = dict(os.environ, KUBECONFIG='/root/.kube/p6000-user.conf')
    args = ['kubectl', '-n', 'dreamzero-training', 'exec', POD, '--', 'env',
        'PYTHONPATH=' + str(ROOT / 'awomopaw/src') + ':' + str(ROOT / 'scripts') + ':/runtime/python-compat',
        'ROBODOJO_COMPAT_PYTHONPATH=/runtime/python-compat',
        'ROBODOJO_FROZEN_LOCK_DIR=/opt/robodojo/eval_result/physicalrsi-all-skills/gpu-locks',
        'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
        'PYTHONDONTWRITEBYTECODE=1']
    args.extend(key + '=' + value for key, value in (environment or {}).items())
    return subprocess.run([*args, *command], env=env, check=True)


def main():
    spec = json.loads((ROOT / 'deployment-spec.json').read_text())
    if not (ROOT / 'source-environment-ready.json').exists():
        stage_status('waiting_source_and_inference_environment')
        core = wait_archive('awomopaw-frozen-runner-v2.tar.zst', spec['awomopaw-frozen-runner-v2.tar.zst'])
        extract(core, ROOT)
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(wait_archive, name, spec[name]) for name in
                ['physicalrsi-source-assets.tar.zst', 'physicalrsi-inference-env.tar.zst']]
            archives = [job.result() for job in jobs]
        stage_status('extracting_frozen_sources_and_environment')
        for archive in archives:
            extract(archive, ASSETS / 'implementations')
    stage_status('installing_24_source_skills')
    pod_run([PYTHON, str(ROOT / 'scripts/prepare_p6000_skills.py'), '--root', str(ROOT)])
    stage_status('launching_four_isolated_gpu_tasks')
    output = STATE_POD / 'evaluations'
    pod_run([PYTHON, str(ROOT / 'scripts/run_parallel_frozen_skills.py'), '--output', str(output),
        '--gpus', '1', '2', '3', '4', '--max-parallel', '4', '--startup-spacing', '10',
        '--sim-root', '/opt/robodojo', '--awomopaw', str(STATE_POD / 'awomopaw'),
        '--registry', str(STATE_POD / 'runtime/physicalrsi-all-skills/registry.json'),
        '--task-providers', str(ROOT / 'runtime/physicalrsi-all-skills/task-providers.json'), '--detach'])
    stage_status('four_tasks_dispatched_vla_checkpoint_sync_continues', evaluation_output=str(output))
    checkpoint = wait_archive('physicalrsi-vla-checkpoints.tar.zst', spec['physicalrsi-vla-checkpoints.tar.zst'])
    stage_status('extracting_two_vla_checkpoints', evaluation_output=str(output))
    extract(checkpoint, ASSETS / 'checkpoints')
    # New checkpoint files do not modify a running frozen code controller.
    deadline = time.monotonic() + 7200
    while True:
        try:
            pod_run([PYTHON, str(ROOT / 'scripts/prepare_p6000_skills.py'), '--root', str(ROOT), '--activate-vla'])
            break
        except subprocess.CalledProcessError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(30)
    stage_status('all_24_skills_available', evaluation_output=str(output))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        stage_status('failed', error_type=type(error).__name__, error=str(error))
        raise
