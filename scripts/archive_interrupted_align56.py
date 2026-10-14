#!/usr/bin/env python3
"""Preserve one interrupted experiment without changing native evidence.

No network, model calls, simulation controls or privileged scene data. Optional
replay videos use only archived public RGB observations in native trace order.
They are visibly labelled reconstructions, not scored official rollout videos.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from zoneinfo import ZoneInfo

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
STATE = Path('/home/ymz/robodojo-pi05/RoboDojo/eval_result/harness_runs')
CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def digest(path):
    hasher = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def copy_file(source, target, manifest):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError('Refuse to overwrite recovery evidence: ' + str(target))
    shutil.copy2(source, target)
    source_hash, target_hash = digest(source), digest(target)
    entry = {'original': str(source), 'copy': str(target), 'bytes': target.stat().st_size,
             'sha256': target_hash, 'byte_identical': target_hash == source_hash}
    manifest.append(entry)
    if not entry['byte_identical']:
        raise RuntimeError('Source changed while archiving: ' + str(source))
    return entry


def probe(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-count_frames',
        '-select_streams', 'v:0', '-show_entries',
        'stream=width,height,nb_read_frames,duration,r_frame_rate', '-of', 'json', str(path)],
        capture_output=True, text=True, timeout=60)
    try:
        details = json.loads(result.stdout) if result.stdout else {}
    except json.JSONDecodeError:
        details = {}
    streams = details.get('streams', [])
    frames = sum(int(s.get('nb_read_frames') or 0) for s in streams
                 if str(s.get('nb_read_frames', '')).isdigit())
    return {'path': str(path), 'exit_code': result.returncode,
            'decodable': result.returncode == 0 and frames > 0,
            'frames': frames, 'streams': streams, 'stderr': result.stderr.strip()}


def json_lines(path, *, sse=False):
    valid, invalid = [], []
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            if sse:
                if not line.startswith('data:'):
                    continue
                line = line[5:].strip()
            if not line.strip() or line.strip() == '[DONE]':
                continue
            try:
                valid.append(json.loads(line))
            except json.JSONDecodeError as error:
                invalid.append({'line': number, 'error': str(error)})
    return valid, invalid


def reconstruct(events, frames_dir, output, camera, *, status_label='INTERRUPTED'):
    if status_label not in {'INTERRUPTED', 'IN PROGRESS'}:
        raise ValueError('Unsupported replay qualification')
    ordered = []
    missing = []
    seen = set()
    sequence = output / 'replay-inputs' / camera
    sequence.mkdir(parents=True, exist_ok=False)
    for event in events:
        if event.get('event') != 'observation':
            continue
        observation_id = event.get('observation_id')
        if not isinstance(observation_id, str) or observation_id in seen:
            continue
        seen.add(observation_id)
        source = frames_dir / observation_id / (camera + '.jpg')
        if not source.exists():
            missing.append({'observation_id': observation_id, 'reason': 'missing public RGB'})
            continue
        try:
            with Image.open(source) as image:
                image.verify()
        except Exception as error:
            missing.append({'observation_id': observation_id,
                            'reason': 'undecodable public RGB: ' + type(error).__name__})
            continue
        target = sequence / f'{len(ordered):06d}.jpg'
        shutil.copy2(source, target)
        ordered.append({'frame': len(ordered), 'observation_id': observation_id,
                        'action_count': event.get('action_count'), 'time': event.get('time'),
                        'archived_public_rgb': str(source), 'sha256': digest(source)})
    report = {'camera': camera, 'ordered_frames': ordered, 'missing_or_bad_frames': missing,
              'qualification': 'Public RGB snapshot replay only; ' + status_label.lower() + ' and not officially scored',
              'playback_fps': 5, 'real_time_preserved': False}
    if len(ordered) < 2:
        report['created'] = False
        save(output / (camera + '-reconstruction.json'), report)
        return report
    video = output / (camera + '_PUBLIC_FRAMES_RECONSTRUCTED_' + status_label.replace(' ', '_') + '_NOT_OFFICIAL.mp4')
    result = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n',
        '-framerate', '5', '-i', str(sequence / '%06d.jpg'), '-vf',
        'drawtext=text=' + status_label + ' / PUBLIC FRAMES / NO SCORE:x=8:y=8:'
        'fontcolor=white:fontsize=14:box=1:boxcolor=black@0.8,'
        'scale=trunc(iw/2)*2:trunc(ih/2)*2',
        '-c:v', 'libx264', '-threads', '1', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', str(video)], capture_output=True, text=True, timeout=120)
    report.update(created=result.returncode == 0, ffmpeg_exit_code=result.returncode,
                  ffmpeg_stderr=result.stderr.strip(), video=str(video))
    if video.exists():
        report['probe'] = probe(video)
        report['sha256'] = digest(video)
    save(output / (camera + '-reconstruction.json'), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    experiment = args.root.resolve()
    if experiment.parent != ROOT / 'eval_result':
        raise ValueError('Recovery must target one explicit experiment under workspace eval_result')
    context = json.loads((experiment / 'audit-context.json').read_text())
    expected_test = str(experiment / 'chats')
    if context.get('test_id') != expected_test:
        raise ValueError('Experiment context provenance mismatch')
    recovery = experiment / 'recovery'
    if (recovery / 'interrupt-report.json').exists():
        raise FileExistsError('Recovery report already exists; do not overwrite')
    recovery.mkdir(exist_ok=True)
    manifest = []
    audits, invalid_audits = [], []
    for source in sorted((experiment / 'model-audit').glob('*.json')):
        copy_file(source, recovery / 'model-audit' / source.name, manifest)
        try:
            record = json.loads(source.read_text())
            if record.get('context', {}).get('session_id') == context['session_id']:
                audits.append(record)
        except json.JSONDecodeError as error:
            invalid_audits.append({'file': source.name, 'bytes': source.stat().st_size,
                                   'error': str(error)})
    scene = experiment / 'chats' / f'scene-{context["scene_index"]:02d}-layout-{context["layout_id"]}'
    sse = scene / 'stream.sse'
    sse_events, sse_errors = json_lines(sse, sse=True)
    copy_file(sse, recovery / 'chat-stream.sse', manifest)
    for name in ('audit-context.json', 'experiment.json', 'driver.log', 'audit-proxy.log'):
        source = experiment / name
        if source.exists():
            copy_file(source, recovery / name, manifest)
    calls = []
    for event in sse_events:
        if event.get('object') == 'message' and event.get('type') == 'plugin_call' and event.get('status') == 'completed':
            for content in event.get('content') or []:
                data = content.get('data') or {}
                arguments = data.get('arguments') or '{}'
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                calls.append({'name': data.get('name'), 'input': arguments,
                              'call_id': data.get('call_id')})
    save(recovery / 'completed-stream-tool-calls.json', calls)
    recovered_runs = []
    for state_path in sorted(STATE.glob('*.json')):
        run = json.loads(state_path.read_text())
        artifact = Path(run['artifact_dir'])
        fixture_file = artifact / 'chat-eval-fixture.json'
        if not fixture_file.exists():
            continue
        fixture = json.loads(fixture_file.read_text())
        if (fixture.get('test_id') != expected_test
                or fixture.get('session_id') != context['session_id']):
            continue
        dest = recovery / 'native' / run['run_id']
        copy_file(state_path, dest / 'run-state.json', manifest)
        for source, name in ((Path(run['log_path']), 'native-run.log'),
                             (artifact / 'awomopaw_harness_trace.jsonl', 'harness-trace.jsonl'),
                             (fixture_file, 'fixture.json')):
            if source.exists():
                copy_file(source, dest / name, manifest)
        for folder in ('_stream', 'awomopaw_harness_frames'):
            source_dir = artifact / folder
            if source_dir.exists():
                for source in sorted(source_dir.rglob('*')):
                    if source.is_file():
                        copy_file(source, dest / folder / source.relative_to(source_dir), manifest)
        trace_events, trace_errors = json_lines(dest / 'harness-trace.jsonl')
        tmp_probes = [probe(path) for path in sorted((dest / '_stream').glob('*.mp4'))]
        save(dest / 'temporary-video-probes.json', tmp_probes)
        reconstructed = [reconstruct(trace_events, dest / 'awomopaw_harness_frames',
                                     dest / 'public-frame-replays', camera) for camera in CAMERAS]
        executions = [event for event in trace_events if event.get('event') == 'action_chunk_execution']
        official_result = Path(run['result_path'])
        recovered_runs.append({'run_id': run['run_id'], 'native_state_status': run['status'],
            'native_result_exists': official_result.exists(), 'native_result_in_state': run.get('result'),
            'official_score': None, 'task_succeeded': None,
            'last_recorded_action_count': max((e.get('action_count', 0) for e in trace_events), default=0),
            'executed_actions': sum(e.get('executed', 0) for e in executions),
            'all_actions_from_frozen_provider': bool(executions) and all(
                e.get('provider_id') == 'physicalrsi-align' and e.get('source') == 'policy' for e in executions),
            'trace_parse_errors': trace_errors, 'temporary_videos': tmp_probes,
            'reconstructed_public_frame_videos': reconstructed})
    boot_time = next(int(line.split()[1]) for line in Path('/proc/stat').read_text().splitlines()
                     if line.startswith('btime '))
    report = {'status': 'infrastructure_interrupted_by_system_reboot',
        'archived_at': datetime.now(timezone.utc).isoformat(),
        'system_boot_time': datetime.fromtimestamp(boot_time, ZoneInfo('Asia/Shanghai')).isoformat(),
        'boot_time_source': '/proc/stat btime', 'context': context,
        'valid_model_audit_records': len(audits), 'invalid_or_empty_model_audit_records': invalid_audits,
        'all_valid_audits_gpt56': bool(audits) and all(r.get('requested_model') == 'gpt-5.6-sol'
            and r.get('returned_model') == 'gpt-5.6-sol' and r.get('status') == 200 for r in audits),
        'sse_parse_errors': sse_errors, 'completed_tool_call_count': len(calls),
        'runs': recovered_runs, 'copied_files': len(manifest),
        'score': None, 'success_rate': None, 'count_as_scored_evaluation': False,
        'scope': 'Only this interrupted experiment and its public RGB observations were read.',
        'video_qualification': 'Temporary original videos are preserved even if invalid. Rebuilt videos '
                               'are visibly labelled public-frame replays, not official complete rollouts.',
        'sources_unchanged': all(item['byte_identical'] for item in manifest)}
    save(recovery / 'copied-file-manifest.json', manifest)
    save(recovery / 'interrupt-report.json', report)
    print(json.dumps({'recovery': str(recovery), 'valid_audits': len(audits),
        'runs': [{'run_id': r['run_id'], 'actions': r['executed_actions'],
                  'temporary_video_decodable': [p['decodable'] for p in r['temporary_videos']],
                  'replay_videos': [v.get('video') for v in r['reconstructed_public_frame_videos']]}
                 for r in recovered_runs]}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
