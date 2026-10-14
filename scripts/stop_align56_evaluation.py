#!/usr/bin/env python3
"""Stop only the user-selected experiment, retaining its evidence."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

from run_gpt56_skill_chats import ROOT, api
from compare_local_tasks import save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if root.parent != ROOT / 'eval_result':
        raise RuntimeError('Refuse an experiment outside this workspace')
    context = json.loads((root / 'audit-context.json').read_text())
    process = json.loads((root / 'process.json').read_text())
    pid = process['pid']
    command_file = Path('/proc') / str(pid) / 'cmdline'
    if command_file.exists():
        command = command_file.read_bytes().decode().split('\0')
        if (not any(Path(part).name == 'run_gpt56_skill_chats.py' for part in command)
                or str(root) not in command):
            raise RuntimeError('PID no longer belongs to this evaluator; refuse signalling')
    receipt = {'status': 'stopping_by_user_request', 'chat_id': context['chat_id'],
               'experiment': str(root), 'pid': pid,
               'reason': 'User requested stopping GPT-5.6 and switching to all align_blocks scenes',
               'incomplete_episode_is_not_a_scored_model_failure': True}
    chat = api('/agents/car_controller/chats/' + context['chat_id'])
    if chat.get('status') == 'running':
        receipt['chat_stop'] = api('/agents/car_controller/console/chat/stop?chat_id=' + context['chat_id'], {}, 'POST')
    if command_file.exists():
        os.kill(pid, signal.SIGTERM)
        receipt['evaluator_sigterm_sent'] = True
    save(root / 'user-stop.json', receipt)
    deadline = time.monotonic() + 55
    while time.monotonic() < deadline:
        if (root / 'restoration.json').exists() and not command_file.exists():
            receipt['restoration'] = json.loads((root / 'restoration.json').read_text())
            receipt['status'] = 'stopped_by_user_request'
            break
        time.sleep(1)
    else:
        receipt['status'] = 'stop_cleanup_pending'
    save(root / 'user-stop.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
