#!/usr/bin/env python3
"""Read-only remote result/video mirror for the existing local trajectory UI.

Never creates a remote experiment, makes a robot action, retries a trial or
falls back to local simulation. Authentication uses an existing private SSH
socket, not an embedded password. Stops after experiment verification/failure.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
NAME = 'physicalrsi-align-all-scenes-node153-20261013-213700'
LOCAL = ROOT / 'eval_result' / NAME
REMOTE = '/mnt/auwomo-data/wangxin/qwenpaw-harness/eval_result/' + NAME
CONTROL = ROOT / 'runtime/align-all-scenes/ssh/control'
STATE = ROOT / 'runtime/align-all-scenes/node153-20261013-213700'
SSH = ['ssh', '-S', str(CONTROL), '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
       '-o', 'ConnectTimeout=15', '-p', '2288']


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def sync():
    subprocess.run([*SSH, '-O', 'check', 'root@180.76.186.140'], check=True, capture_output=True)
    LOCAL.mkdir(parents=True, exist_ok=True)
    ssh_command = ' '.join(SSH)
    subprocess.run(['rsync', '-az', '--partial', '--timeout=90', '--exclude=*.tmp', '--exclude=*.tmp.*',
                    '-e', ssh_command, 'root@180.76.186.140:' + REMOTE + '/', str(LOCAL) + '/'],
                   check=True, timeout=240)
    # Include the launch and preflight receipt in the local evidence, not keys.
    subprocess.run(['rsync', '-az', '--timeout=30', '-e', ssh_command,
                    'root@180.76.186.140:/mnt/auwomo-data/wangxin/qwenpaw-harness/eval_result/align-node153-runner-20261013-213700/',
                    str(STATE / 'remote-runner') + '/'], check=True, timeout=90)
    status = json.loads((LOCAL / 'status.json').read_text())
    summary = json.loads((LOCAL / 'summary.json').read_text())
    if summary.get('node') != '192.168.48.153':
        raise RuntimeError('Remote result is not from node 153')
    receipt = {'synced_utc': datetime.now(timezone.utc).isoformat(), 'remote': REMOTE,
               'local': str(LOCAL), 'remote_status': status['status'], 'node': summary['node'],
               'finished': summary['finished'], 'requested': summary['requested'],
               'simulation_runs_locally': False}
    if (LOCAL / 'verified-summary.json').exists():
        checked = 0
        for record in LOCAL.glob('episode-*/videos.json'):
            for video in json.loads(record.read_text()):
                path = record.parent / 'videos' / Path(video['path']).name
                if hashlib.sha256(path.read_bytes()).hexdigest() != video['sha256']:
                    raise RuntimeError('Local UI video differs from remote verified video')
                checked += 1
        expected = 3 * summary['valid_episodes']
        if checked != expected:
            raise RuntimeError('Local video mirror is incomplete')
        receipt['verified_mirrored_videos'] = checked
        receipt['all_video_hashes_match_remote'] = True
    save(LOCAL / 'ui-sync.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    return (status['status'] in ['failed', 'stopped_by_user']
            or (status['status'] == 'completed' and (LOCAL / 'verified-summary.json').exists()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--interval', type=int, default=60)
    args = parser.parse_args()
    if args.watch:
        save(STATE / 'mirror-process.json', {'pid': os.getpid(), 'interval': args.interval,
            'source': REMOTE, 'purpose': 'Read-only remote UI synchronization; no simulator/actions'})
    deadline = time.monotonic() + 43200
    while True:
        try:
            done = sync()
        except Exception as error:
            save(STATE / 'mirror-error.json', {'type': type(error).__name__, 'error': str(error),
                'at': datetime.now(timezone.utc).isoformat(), 'no_remote_test_restarted': True})
            raise
        if done or not args.watch:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError('UI mirroring exceeded the existing SSH session lifetime')
        time.sleep(max(10, min(args.interval, 60)))


if __name__ == '__main__':
    main()
