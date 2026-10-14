#!/usr/bin/env python3
"""Preserve an explicitly stopped Chat trial; never count it as a valid test."""
import argparse
import json
from pathlib import Path
import shutil

from compare_local_tasks import save, request
from evaluate_frozen_align_skill import archive
from evaluate_fresh_chat_align import collect_model_audit, memory_tools, stream_tool_calls
from run_gpt56_skill_chats import ROOT, api


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    context = json.loads((root / 'audit-context.json').read_text())
    backup = json.loads((root / 'original-agent-settings.json').read_text())
    current = api('/agents/car_controller')
    if current['active_model'] != backup['active_model'] or current['running'] != backup['running']:
        raise RuntimeError('Original harness model/running settings have not been restored')
    if request('/health').get('active_evaluation'):
        raise RuntimeError('A simulation is still active')
    history = api('/agents/car_controller/chats/' + context['chat_id'])
    if history.get('status') == 'running':
        raise RuntimeError('This Chat is still running')
    dest = root / 'stopped-by-user'
    dest.mkdir(exist_ok=False)
    save(dest / 'chat-history.json', history)
    scene = root / 'chats' / f'scene-{context["scene_index"]:02d}-layout-{context["layout_id"]}'
    calls = stream_tool_calls(scene / 'stream.sse')
    save(dest / 'complete-tool-calls.json', calls)
    audit = collect_model_audit(root / 'model-audit', context['session_id'], dest, 'gpt-5.6-sol')
    session = Path('/home/ymz/.awomopaw/workspaces/car_controller/sessions') / f'default_{context["session_id"]}.json'
    if session.exists():
        shutil.copy2(session, dest / 'raw-session.json')
    runs = []
    for path in Path('/home/ymz/robodojo-pi05/RoboDojo/eval_result/harness_runs').glob('*.json'):
        run = json.loads(path.read_text())
        artifact = Path(run['artifact_dir'])
        fixture = artifact / 'chat-eval-fixture.json'
        if not fixture.exists():
            continue
        case = json.loads(fixture.read_text())
        if case.get('test_id') != str(root / 'chats') or case.get('session_id') != context['session_id']:
            continue
        target = dest / run['run_id']
        target.mkdir()
        shutil.copy2(path, target / 'run-state.json')
        shutil.copy2(fixture, target / 'fixture.json')
        shutil.copy2(run['log_path'], target / 'native-run.log')
        native = Path(run['result_path'])
        if native.exists():
            shutil.copy2(native, target / 'native-result.json')
        # The native scorer closes interrupted rollouts with a failure score.
        # Retain it without representing this as a completed GPT-5.6 trial.
        official = {'run': run, 'videos': [str(p) for p in artifact.glob('*.mp4')],
            'harness_trace': str(artifact / 'awomopaw_harness_trace.jsonl'),
            'task_succeeded': False, 'user_stopped': True}
        save(target / 'native-finalization.json', official)
        videos = archive(official, target)
        runs.append({'run_id': run['run_id'], 'layout_id': case['layout_id'],
            'native_finalization_result': run.get('result'),
            'valid_performance_trial': False, 'videos': [v['path'] for v in videos]})
    receipt = {'status': 'stopped_by_user_request', 'original_model_restored': True,
        'original_running_restored': True, 'simulation_inactive': True, 'chat_idle': True,
        'model_audit': audit, 'runs': runs, 'valid_completed_gpt56_trials': 0,
        'remaining_layouts_not_run': [1, 2, 3, 4],
        'qualification': 'Explicit user cancellation, not a scored five-layout GPT-5.6 success-rate result'}
    save(dest / 'stop-receipt.json', receipt)
    save(root / 'user-stop.json', receipt)
    print(json.dumps({'status': receipt['status'], 'runs': runs}, ensure_ascii=False))


if __name__ == '__main__':
    main()
