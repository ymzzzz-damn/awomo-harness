#!/usr/bin/env python3
"""Mirror the active native RSI run for the local UI, without remote mutations."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import time
import types

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / 'runtime/rsi-native-open-20261015'
SSH = ['ssh', '-p', '2288', '-oBatchMode=yes', '-oStrictHostKeyChecking=yes',
       '-oConnectTimeout=15', '-oControlPath=/tmp/robodojo-p6000-current.sock']


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        Path(name).replace(path)
    finally:
        Path(name).unlink(missing_ok=True)


def mapped(path, remote, local):
    relative = Path(path).relative_to(remote)
    result = (local / relative).resolve()
    if not result.is_relative_to(local.resolve()):
        raise ValueError('Evidence escaped the active run')
    return result


def rsync(remote, local, *, filters=None, paths=None):
    command = ['rsync', '-az', '--partial', '--timeout=90', '-e', shlex.join(SSH)]
    temporary = None
    try:
        if filters:
            for rule in filters:
                command += ['--filter', rule]
        if paths is not None:
            fd, name = tempfile.mkstemp(prefix='rsi-ui-files-')
            temporary = Path(name)
            with os.fdopen(fd, 'w') as stream:
                for path in sorted(paths):
                    if '\n' in path or path.startswith('/') or '..' in Path(path).parts:
                        raise ValueError('Invalid mirror file path')
                    stream.write(path + '\n')
            command += ['--files-from', name]
        command += ['root@180.76.186.140:' + remote + '/', str(local) + '/']
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=300)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)


def verifier(remote, local):
    path = '/mnt/auwomo-data/RoboDojo/qwenpaw-harness/eval_result/rsi-native-open-runtime/20261015/continuation-approved-v1/original_run_rsi_native_open.py'
    source = subprocess.run([*SSH, 'root@180.76.186.140', 'cat ' + shlex.quote(path)],
        capture_output=True, check=True, timeout=45).stdout
    if hashlib.sha256(source).hexdigest() != '69f7dc5119f337505fdab2fba81dc5695b5316e7c9477a6cdd8bad85b25a5af5':
        raise ValueError('The original official verifier changed')
    # The local framework was removed separately. Read the pinned verifier into
    # this mirror process only; do not restore framework/source trees locally.
    module = types.ModuleType('rsi_ui_original_verifier')
    module.__file__ = '<frozen-original-official-verifier>'
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    original_hash = module.evidence_sha256
    # Translate only file locations. Original native score checks stay intact.
    module.evidence_sha256 = lambda p: original_hash(mapped(p, remote, local))
    module.read = lambda p: read(mapped(p, remote, local))
    return module


def sync(active, local, native, verified):
    remote = active['remote_output']
    name = Path(remote).name
    if not name.startswith('RSI-hybrid-emptytasks-'):
        raise ValueError('Only the active native hybrid output can be mirrored')
    shared = '/mnt/auwomo-data/RoboDojo/qwenpaw-harness/eval_result/' + name
    filters = ['+ /official-inventory.json', '+ /summary.json', '+ /continuations/',
        '+ /continuations/approved-v1/', '+ /continuations/approved-v1/queue-status.json',
        '+ /continuations/approved-v1/queue-process.json', '+ /tasks/', '+ /tasks/*/',
        '+ /continuations/official-preaction-approved-v*/',
        '+ /continuations/official-preaction-approved-v*/*.json',
        '+ /continuations/official-preaction-approved-v*/proofs/',
        '+ /continuations/official-preaction-approved-v*/proofs/***',
        '+ /tasks/*/progress.json', '+ /tasks/*/empty-task-seed.json',
        '+ /tasks/*/continuation/', '+ /tasks/*/continuation/***',
        '+ /tasks/*/workspace/', '+ /tasks/*/workspace/state/', '+ /tasks/*/workspace/state/*.json',
        '+ /tasks/*/workspace/seed/', '+ /tasks/*/workspace/seed/*.json',
        '+ /tasks/*/workspace/rounds/', '+ /tasks/*/workspace/rounds/round-*/',
        '+ /tasks/*/workspace/rounds/round-*/*.json',
        '+ /tasks/*/official/', '+ /tasks/*/official/*/', '+ /tasks/*/official/*/*.json',
        '+ /tasks/*/official/*/cohort/', '+ /tasks/*/official/*/cohort/cases.json',
        '+ /tasks/*/official/*/cohort/initialization/',
        '+ /tasks/*/official/*/cohort/initialization/**/',
        '+ /tasks/*/official/*/cohort/initialization/**.json',
        '+ /tasks/*/official/*/cohort/initialization/**.log',
        '+ /tasks/*/official/*/cohort/official-provenance.json', '- *']
    filters[-1:-1] = [
        '+ /tasks/*/official/*/recovery-attempts/',
        '+ /tasks/*/official/*/recovery-attempts/preaction-*/',
        '+ /tasks/*/official/*/recovery-attempts/preaction-*/*.json',
        '+ /tasks/*/official/*/recovery-attempts/preaction-*/cohort/',
        '+ /tasks/*/official/*/recovery-attempts/preaction-*/cohort/cases.json',
        '+ /tasks/*/official/*/recovery-attempts/preaction-*/cohort/official-provenance.json',
        '+ /tasks/*/official/*/recovery-attempts/initialization-*/',
        '+ /tasks/*/official/*/recovery-attempts/initialization-*/*.json',
        '+ /tasks/*/official/*/recovery-attempts/initialization-*/cohort/',
        '+ /tasks/*/official/*/recovery-attempts/initialization-*/cohort/**/',
        '+ /tasks/*/official/*/recovery-attempts/initialization-*/cohort/**.json',
        '+ /tasks/*/official/*/recovery-attempts/initialization-*/cohort/**.log']
    local.mkdir(parents=True, exist_ok=True)
    rsync(shared, local, filters=filters)
    inventory = read(local / 'official-inventory.json')
    expected = {(r['task'], r['seed'], r['layout_id']): r for r in inventory}
    if len(expected) != 1320 or len(inventory) != 1320:
        raise ValueError('The full distinct official inventory is required')
    pending, paths = [], set()
    for score_path in sorted(local.glob('tasks/*/official/*/score.json')):
        relative = str(score_path.relative_to(local))
        digest = native.sha256(score_path)
        if verified.get(relative, {}).get('sha256') == digest:
            continue
        score = read(score_path)
        key = (score['task'], score['seed'], score['layout_id'])
        case = expected[key]
        if (score['layout_sha256'] != case['layout_sha256'] or
                score['official_file_sha256'] != native.official_file_sha256(case)):
            raise ValueError('Official scene identity changed')
        for path in score['native_result']['evidence']:
            target = mapped(path, remote, local)
            if not target.is_relative_to(score_path.parent):
                raise ValueError('Official evidence belongs to another scene')
            paths.add(str(target.relative_to(local)))
        pending.append((relative, digest, score, case))
    if paths:
        rsync(shared, local, paths=paths)
    for relative, digest, score, case in pending:
        native.verify_result(score['native_result'], case=case,
            harness_sha256=score['selected_harness_sha256'], expert=score['selected_expert'])
        videos = [mapped(p, remote, local) for p in score['native_result']['evidence'] if p.endswith('.mp4')]
        if len(videos) != 3 or not all(any(camera in p.name for p in videos)
                for camera in ('cam_head', 'cam_left_wrist', 'cam_right_wrist')):
            raise ValueError('Three original official camera videos required')
        verified[relative] = dict(sha256=digest, native_evidence_verified=True, camera_count=3)
    summary = read(local / 'summary.json')
    from rsi_native_ui import recovery_queue_path
    queue = read(recovery_queue_path(local) or local / 'continuations/approved-v1/queue-status.json')
    record = dict(schema='rsi.native-open.ui-sync/v1', synced_at=datetime.now(timezone.utc).isoformat(),
        remote_output=remote, local_output=str(local), status=queue.get('state'),
        remote_completed=summary['completed'], verified_completed=len(verified),
        verified_scores=verified, original_official_verifier=True, remote_mutations=False)
    save(local / 'ui-sync.json', record)
    save(STATE / 'frontend/latest-sync.json', {k:v for k,v in record.items() if k != 'verified_scores'})
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=int, default=60)
    args = parser.parse_args()
    active = read(STATE / 'active-experiment.json')
    local = (ROOT / 'eval_result' / Path(active['remote_output']).name).resolve()
    lock_path = STATE / 'frontend/.mirror.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        native = verifier(active['remote_output'], local)
        verified = {}
        # Revalidate existing scenes once per mirror process, then cache immutable evidence.
        save(local / 'experiment.json', dict(schema='physicalrsi.native-open-results/v1',
            remote_output=active['remote_output'], mode='hybrid'))
        save(STATE / 'frontend/mirror-process.json', dict(pid=os.getpid(), watch=args.watch,
            interval_seconds=args.interval, remote_output=active['remote_output'], local_output=str(local),
            starttime_ticks=int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
            command=[part.decode() for part in Path('/proc/self/cmdline').read_bytes().split(b'\0') if part],
            started_at=datetime.now(timezone.utc).isoformat()))
        while True:
            try:
                result = sync(active, local, native, verified)
                print(json.dumps({k:v for k,v in result.items() if k != 'verified_scores'}), flush=True)
                if not args.watch:
                    return
            except Exception as error:
                save(STATE / 'frontend/mirror-error.json', dict(at=datetime.now(timezone.utc).isoformat(),
                    error=type(error).__name__ + ': ' + str(error)))
                if not args.watch:
                    raise
            time.sleep(max(30, args.interval))


if __name__ == '__main__':
    main()
