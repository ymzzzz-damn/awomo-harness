#!/usr/bin/env python3
"""Preview one live trial's saved public RGB frames; never control the robot."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from archive_interrupted_align56 import CAMERAS, ROOT, STATE, json_lines, reconstruct, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if root.parent != ROOT / 'eval_result':
        raise ValueError('Preview must belong to one explicit workspace experiment')
    case = json.loads((root / 'audit-context.json').read_text())
    if case['test_id'] != str(root / 'chats'):
        raise ValueError('Test context provenance mismatch')
    runs = []
    for path in STATE.glob('*.json'):
        try:
            run = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        fixture = Path(run['artifact_dir']) / 'chat-eval-fixture.json'
        if fixture.exists():
            fixture_data = json.loads(fixture.read_text())
            if (fixture_data.get('test_id') == case['test_id']
                    and fixture_data.get('session_id') == case['session_id']):
                runs.append(run)
    if len(runs) != 1:
        raise RuntimeError('Expected one uniquely bound live trial')
    run = runs[0]
    artifact = Path(run['artifact_dir'])
    events, incomplete = json_lines(artifact / 'awomopaw_harness_trace.jsonl')
    now = datetime.now(timezone.utc)
    output = root / 'progress-previews' / ('layout-' + str(case['layout_id'])) / now.strftime('%Y%m%dT%H%M%SZ')
    videos = [reconstruct(events, artifact / 'awomopaw_harness_frames', output, camera,
                          status_label='IN PROGRESS') for camera in CAMERAS]
    save(output / 'preview.json', {'run_id': run['run_id'], 'layout_id': case['layout_id'],
        'seed': case['seed'], 'created_utc': now.isoformat(), 'score': None,
        'qualification': 'In-progress public RGB replay at 5fps, NOT a complete or officially scored evaluation video',
        'partial_trace_lines_skipped': incomplete,
        'last_recorded_action_count': max((e.get('action_count', 0) for e in events), default=0),
        'videos': videos})
    print(json.dumps({'layout_id': case['layout_id'], 'output': str(output),
                      'videos': [v.get('video') for v in videos]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
