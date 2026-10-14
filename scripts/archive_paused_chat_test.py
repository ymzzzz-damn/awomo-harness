#!/usr/bin/env python3
"""Preserve the explicitly interrupted Chat episode; do not resume its actions."""
import json
from pathlib import Path
import shutil

import compare_local_tasks as control
from evaluate_frozen_align_skill import archive
from evaluate_fresh_chat_align import app_json

ROOT = Path('/home/ymz/Documents/ChatGPT/robodojo/eval_result/align-five-fresh-chats-20261013-173756')
SCENE = ROOT / 'scene-02-layout-1'
RUN_ID = '20261013T095406Z-90f00970'
CHAT_ID = '670c0597-7f90-48a9-8b7f-f74033078b83'


def main():
    control.save(SCENE / 'chat-history-at-pause.json', app_json('/chats/' + CHAT_ID))
    state = control.request('/v1/evaluations/status?run_id=' + RUN_ID)
    final = control.official_result(state)
    dest = SCENE / RUN_ID
    control.save(dest / 'official-result.json', final)
    videos = archive(final, dest)
    policies = Path(final['run']['artifact_dir']) / 'code-policies'
    if policies.exists():
        shutil.copytree(policies, dest / 'code-policies', dirs_exist_ok=True)
    chat = json.loads((SCENE / 'chat.json').read_text())
    session = Path('/home/ymz/.awomopaw/workspaces/car_controller/sessions') / f'default_{chat["session_id"]}.json'
    if session.exists():
        shutil.copy2(session, SCENE / 'raw-session-at-pause.json')
    control.save(ROOT / 'pause.json', {
        'status': 'paused_by_user', 'completed_chats': 1, 'interrupted_scene': 2,
        'run_id': RUN_ID, 'chat_id': CHAT_ID,
        'interrupted_episode_not_counted_as_finished_test': True,
        'reason': 'User requested frozen-skill evaluation on seed0 layouts0-4 first',
        'videos': [v['path'] for v in videos]})
    print(json.dumps({'paused': True, 'preserved_videos': len(videos), 'run_id': RUN_ID}))


if __name__ == '__main__':
    main()
