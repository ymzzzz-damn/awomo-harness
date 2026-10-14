#!/usr/bin/env python3
"""Verify finished scene coverage, native scores, action provenance and videos."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from compare_local_tasks import save


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    summary = json.loads((root / 'summary.json').read_text())
    if not summary['complete']:
        raise RuntimeError('Full scene inventory is not finished; do not report a final rate')
    inventory = json.loads((root / 'official-case-inventory.json').read_text())
    expected = {(c['seed'], c['layout_id']) for c in inventory}
    actual = [(r['seed'], r['requested_layout_id']) for r in summary['episodes']]
    if set(actual) != expected or len(actual) != len(expected):
        raise RuntimeError('Coverage has missing or duplicate official scenes')
    rows = []
    count = 0
    for index, row in enumerate(summary['episodes']):
        seed, layout = row['seed'], row['requested_layout_id']
        dest = root / f'episode-{index:02d}-seed-{seed}-layout-{layout}'
        if row.get('unscored_native_scene'):
            rows.append({'seed': seed, 'layout_id': layout, 'status': 'native_scene_unstable',
                         'score': None, 'success': None, 'videos': []})
            continue
        official = json.loads((dest / 'official-result.json').read_text())
        native = official['run']['result']
        if native['eval_time'] != 1 or len(native['details']) != 1:
            raise RuntimeError('Expected one official episode per fixture')
        detail = next(iter(native['details'].values()))
        if detail['layout_id'] != layout or native['score'] != row['score']:
            raise RuntimeError('Native score/layout does not match the recorded result')
        trace = [json.loads(line) for line in (dest / 'harness-trace.jsonl').read_text().splitlines() if line]
        executed = [e for e in trace if e.get('event') == 'action_chunk_execution']
        chunks = [e for e in trace if e.get('event') == 'action_chunk_ready']
        if not executed or not all(e['source'] == 'policy' and e['provider_id'] == 'physicalrsi-align' for e in executed):
            raise RuntimeError('Scene did not exclusively execute the frozen skill')
        if sum(e['executed'] for e in executed) != row['actions']:
            raise RuntimeError('Action totals do not match')
        if not all(e['subtask_goal'] == e['official_instruction'] == e['optimized_task_prompt'] for e in chunks):
            raise RuntimeError('Official task instruction was modified')
        cameras = {'cam_head', 'cam_left_wrist', 'cam_right_wrist'}
        saved = json.loads((dest / 'videos.json').read_text())
        if len(saved) != 3 or {c for v in saved for c in cameras if c in Path(v['path']).name} != cameras:
            raise RuntimeError('Missing official camera videos')
        for video in saved:
            if not digest(video['path']) == digest(video['original']) == video['sha256']:
                raise RuntimeError('Video differs from native original')
            probe = subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                '-show_entries', 'stream=nb_read_frames', '-of', 'json', video['path']], text=True)
            if int(json.loads(probe)['streams'][0]['nb_read_frames']) <= 0:
                raise RuntimeError('Video has no decodable frames')
        count += len(saved)
        rows.append({'seed': seed, 'layout_id': layout, 'run_id': row['run_id'],
            'score': row['score'], 'success': row['success'], 'actions': row['actions'],
            'videos': saved})
    sources = json.loads((root / 'frozen-release.json').read_text())['verified_release_files']
    source_unchanged = all(digest(item['path']) == item['sha256'] for item in sources)
    dataset_unchanged = all(digest(item['file']) == item['sha256'] for item in inventory)
    restored = json.loads((root / 'restoration.json').read_text())
    if not source_unchanged or not dataset_unchanged or not restored.get('ordinary_bridge_restored'):
        raise RuntimeError('Frozen source/dataset identity or bridge restoration failed verification')
    final = dict(summary, final_success_rate=summary['successes'] / summary['valid_episodes']
        if summary['valid_episodes'] else None, verified_camera_videos=count,
        frozen_source_files_verified=len(sources), frozen_sources_unchanged=source_unchanged,
        official_dataset_unchanged=dataset_unchanged, scene_coverage_verified=True,
        qualification='Existing frozen task-specific skill; no GPT-5.6/Pi0.5/LLM-generated controller calls',
        platform='local RTX 4090', video_index=rows)
    save(root / 'verified-summary.json', final)
    save(root / 'video-index.json', rows)
    print(json.dumps({k: final[k] for k in ['requested', 'valid_episodes', 'successes',
        'final_success_rate', 'unscored_native_scenes', 'verified_camera_videos']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
