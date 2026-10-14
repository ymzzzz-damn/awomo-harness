#!/usr/bin/env python3
"""Owned server/client pair with a fixed policy port; never changes live services."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import signal
import socket
import subprocess
import time


def terminate_tree(process):
    if process is None:
        return
    # Descendants are in the evaluator's owned process group. Explicitly stop
    # this subtree for normal exit; the driver kills the whole group on timeout.
    rows = subprocess.check_output(['ps', '-eo', 'pid=,ppid='], text=True)
    children = {}
    for row in rows.splitlines():
        pid, ppid = map(int, row.split())
        children.setdefault(ppid, []).append(pid)
    def descendants(pid):
        result = []
        for child in children.get(pid, []):
            result.extend(descendants(child))
            result.append(child)
        return result
    owned = descendants(process.pid) + [process.pid]
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in owned:
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        if sig == signal.SIGTERM:
            time.sleep(1)
    process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--policy-dir', type=Path, required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--env-cfg', required=True)
    parser.add_argument('--action-type', choices=['joint', 'ee'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--policy-port', type=int, required=True)
    parser.add_argument('--eval-env', default='RoboDojo')
    args = parser.parse_args()
    server = client = None
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    common = ['RoboDojo', args.task, args.checkpoint, args.env_cfg, args.action_type, str(args.seed)]
    try:
        server = subprocess.Popen(['bash', str(args.policy_dir / 'setup_eval_policy_server.sh'),
            *common, str(args.gpu), 'uv', str(args.policy_port), '127.0.0.1'],
            cwd=args.root)
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError('Owned policy server exited: ' + str(server.returncode))
            try:
                with socket.create_connection(('127.0.0.1', args.policy_port), timeout=1):
                    break
            except OSError:
                time.sleep(1)
        else:
            raise TimeoutError('Owned policy server did not open its port')
        env = os.environ.copy()
        env['EVAL_NUM'] = '1'
        client = subprocess.Popen(['bash', str(args.policy_dir / 'setup_eval_env_client.sh'),
            *common, str(args.gpu), args.eval_env,
            f'ckpt_name={args.checkpoint},action_type={args.action_type}', str(args.policy_port), '127.0.0.1'],
            cwd=args.root, env=env)
        return client.wait()
    finally:
        terminate_tree(client)
        terminate_tree(server)


if __name__ == '__main__':
    raise SystemExit(main())
