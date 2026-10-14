#!/usr/bin/env python3
"""Send only the user's exact goal to five clean AwomoPaw Chats.

The driver never selects a provider, requests action chunks or actuates Isaac.
It sets native scene IDs outside the model prompt and archives actual Chat
evidence. An unfinished episode is aborted after Chat stops, never completed
manually. Goal completion and automatic frozen-skill selection are separate.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import time
import urllib.parse
import urllib.request
import uuid

import compare_local_tasks as control
from evaluate_frozen_align_skill import archive

APP = 'http://127.0.0.1:8088/api/agents/car_controller'
WORKSPACE = Path('/home/ymz/.awomopaw/workspaces/car_controller')
STATE = Path('/home/ymz/robodojo-pi05/RoboDojo/eval_result/harness_runs')
PROMPT = '完成robodojo中的align_blocks'


def app_request(path, payload=None, timeout=30):
    request = urllib.request.Request(APP + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json', 'X-Agent-Id': 'car_controller'})
    return urllib.request.urlopen(request, timeout=timeout)


def app_json(path, payload=None):
    with app_request(path, payload) as response:
        return json.load(response)


def memory_tools(session):
    state = json.loads(session.read_text())
    tools = []
    texts = []
    for item in state.get('agent', {}).get('memory', {}).get('content', []):
        message = item[0] if isinstance(item, list) else item
        if not isinstance(message, dict):
            continue
        for block in message.get('content', []):
            if not isinstance(block, dict):
                continue
            if block.get('type') == 'tool_use':
                tools.append({'name': block['name'], 'input': block.get('input', {}),
                              'call_id': block.get('id'), 'evidence_source': 'raw_memory'})
            if block.get('type') == 'text' and message.get('role') in {'user', 'assistant'}:
                texts.append({'role': message['role'], 'text': block.get('text', '')})
    return tools, texts


def stream_tool_calls(path):
    """Retain full completed tool calls even if long-Chat memory compacted."""
    calls = []
    completed_outputs = set()

    def consume(lines):
        if not lines:
            return
        raw = '\n'.join(lines)
        if raw == '[DONE]':
            return
        event = json.loads(raw)
        if event.get('object') != 'message' or event.get('status') != 'completed':
            return
        for content in event.get('content') or []:
            data = content.get('data')
            if not isinstance(data, dict):
                continue
            if event.get('type') == 'plugin_call_output':
                if data.get('call_id'):
                    completed_outputs.add(data['call_id'])
            elif event.get('type') == 'plugin_call':
                arguments = data.get('arguments', {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise ValueError('Completed SSE tool arguments must be a JSON object')
                calls.append({'name': data['name'], 'input': arguments,
                              'call_id': data.get('call_id'),
                              'message_id': event.get('id'),
                              'evidence_source': 'completed_stream_event'})

    if not path.exists():
        return calls
    lines = []
    with path.open() as stream:
        for line in stream:
            line = line.rstrip('\r\n')
            if not line:
                consume(lines)
                lines = []
            elif line.startswith('data:'):
                lines.append(line[5:].lstrip(' '))
    consume(lines)
    for call in calls:
        call['output_completed'] = call.get('call_id') in completed_outputs
    return calls


def merge_tool_calls(stream_calls, memory_calls):
    merged = []
    seen_ids = set()
    stream_values = {json.dumps([t['name'], t['input']], sort_keys=True)
                     for t in stream_calls}
    for call in [*stream_calls, *memory_calls]:
        call_id = call.get('call_id')
        if call_id:
            if call_id in seen_ids:
                continue
            seen_ids.add(call_id)
        elif (call.get('evidence_source') == 'raw_memory'
              and json.dumps([call['name'], call['input']], sort_keys=True) in stream_values):
            continue
        merged.append(call)
    return merged


def model_matches(actual, expected):
    """Accept the exact model ID or its dated snapshot, not another variant."""
    return isinstance(actual, str) and (
        actual == expected or re.fullmatch(
            re.escape(expected) + r'-(?:\d{4}-\d{2}-\d{2}|\d{8})', actual) is not None)


def collect_model_audit(audit_dir, session_id, dest, expected_model=None):
    records = []
    for path in sorted(audit_dir.glob('*.json')):
        record = json.loads(path.read_text())
        if record.get('context', {}).get('session_id') == session_id:
            records.append(dict(record, audit_file=str(path)))
    control.save(dest / 'model-audit.json', records)
    successful = [r for r in records if isinstance(r.get('status'), int)
                  and 200 <= r['status'] < 300]
    summary = {
        'records': len(records), 'successful_responses': len(successful),
        'failed_responses': len(records) - len(successful),
        'requested_models': sorted({r['requested_model'] for r in records
                                    if isinstance(r.get('requested_model'), str)}),
        'returned_models': sorted({r['returned_model'] for r in successful
                                   if isinstance(r.get('returned_model'), str)}),
        'expected_model': expected_model,
        'verified': bool(successful) and (expected_model is None or (
            all(model_matches(r.get('requested_model'), expected_model) for r in records)
            and all(model_matches(r.get('returned_model'), expected_model)
                    for r in successful))),
    }
    control.save(dest / 'model-audit-summary.json', summary)
    return summary


def stop_own_chat(chat_id, dest):
    """Use the app's scoped stop endpoint; never stop another Chat."""
    history = app_json('/chats/' + chat_id)
    if history.get('status') == 'running':
        receipt = app_json('/console/chat/stop?chat_id=' + urllib.parse.quote(chat_id), {})
        control.save(dest / 'chat-stop.json', receipt)
        deadline = time.monotonic() + 30
        while history.get('status') == 'running' and time.monotonic() < deadline:
            time.sleep(1)
            history = app_json('/chats/' + chat_id)
        if history.get('status') == 'running':
            raise TimeoutError('This trial Chat did not stop; do not switch its bridge or model')
    control.save(dest / 'chat-history.json', history)
    return history


def own_runs(before, test_id, session_id, scene_index):
    """Require native fixture provenance in addition to a new run file."""
    runs = []
    for path in sorted(STATE.glob('*.json')):
        if path.name in before:
            continue
        run = json.loads(path.read_text())
        selection_file = Path(run['artifact_dir']) / 'chat-eval-fixture.json'
        if not selection_file.exists():
            continue
        selection = json.loads(selection_file.read_text())
        if (selection.get('test_id') == test_id
                and selection.get('session_id') == session_id
                and selection.get('scene_index') == scene_index):
            runs.append((run, selection_file, selection))
    return runs


def archive_run(run, selection_file, selection, layout, dest):
    state = control.request('/v1/evaluations/status?run_id=' + run['run_id'])
    abandoned = state['status'] in {'starting', 'running'}
    if abandoned:
        active = control.request('/health').get('active_evaluation')
        if not active or active.get('run_id') != run['run_id']:
            raise RuntimeError('Refusing to abort a session not owned by this trial run')
        # Cleanup only: do not complete the episode or execute remaining actions.
        control.request('/v1/sessions/abort', {
            'reason': 'This fresh Chat stopped before completing its own episode; cleanup only'})
    official = control.official_result(run)
    run_dir = dest / run['run_id']
    control.save(run_dir / 'official-result.json', official)
    shutil.copy2(selection_file, run_dir / 'fixture.json')
    videos = archive(official, run_dir)
    native = official['run'].get('result')
    entry = {
        'run_id': run['run_id'], 'provider': run['policy_id'],
        'score': native.get('score') if native else None,
        'success': False, 'valid_episode': False,
        'chat_ended_before_episode': abandoned,
        'videos': [v['path'] for v in videos],
        'details': native.get('details') if native else None,
    }
    errors = []
    if run.get('task') != 'align_blocks' or run.get('seed') != 0:
        errors.append('Chat selected another task or seed')
    if (not native or native.get('eval_time') != 1
            or len(native.get('details', {})) != 1):
        errors.append('Expected exactly one official scored episode')
    else:
        detail = next(iter(native['details'].values()))
        entry['actual_layout_id'] = detail.get('layout_id')
        if detail.get('layout_id') != layout or selection.get('layout_id') != layout:
            errors.append('Native evaluator scored a different layout than requested')
    cameras = {'cam_head', 'cam_left_wrist', 'cam_right_wrist'}
    observed_cameras = {camera for v in videos for camera in cameras
                        if camera in Path(v['path']).name}
    # archive() independently checks nonzero decodable frames for every video.
    if len(videos) != 3 or observed_cameras != cameras:
        errors.append('Expected exactly head and both wrist decodable camera videos')
    if errors:
        entry['validation_errors'] = errors
    else:
        entry['valid_episode'] = True
        entry['success'] = bool(official.get('task_succeeded')) and not abandoned
    control.save(run_dir / 'validation.json', entry)
    return entry


def trial(index, layout, output, fixture, *, prompt=PROMPT, audit_dir=None,
          expected_model=None):
    dest = output / f'scene-{index+1:02d}-layout-{layout}'
    dest.mkdir(parents=True, exist_ok=False)
    test_id = str(output.resolve())
    session_id = 'align-clean-' + uuid.uuid4().hex
    case = {'task': 'align_blocks', 'seed': 0, 'layout_id': layout,
            'test_id': test_id, 'scene_index': index + 1, 'session_id': session_id}
    before = {p.name for p in STATE.glob('*.json')}
    row = {'scene': index + 1, 'layout_id': layout, 'chat_id': None,
           'session_id': session_id, 'prompt': prompt, 'status': 'running', 'runs': []}
    chat = None
    tools, texts = [], []
    errors = []
    try:
        control.wait_gpu()
        health = control.request('/health')
        if health.get('active_evaluation'):
            raise RuntimeError('Another simulation is active')
        chat = app_json('/chats', {'name': prompt, 'session_id': session_id,
                                'user_id': 'default', 'channel': 'console', 'meta': {}})
        row['chat_id'] = chat['id']
        case['chat_id'] = chat['id']
        control.save(dest / 'chat.json', chat)
        control.save(fixture, case)
        control.save(dest / 'case.json', case)
        context = dict(case)
        context_path = audit_dir.parent / 'audit-context.json' if audit_dir else output / 'audit-context.json'
        control.save(context_path, context)
        control.save(dest / 'audit-context.json', context)
        initial = app_json('/chats/' + chat['id'])
        control.save(dest / 'initial-chat.json', initial)
        if initial.get('messages'):
            raise ValueError('New chat contains prior messages')
        payload = {'input': [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}],
                   'session_id': session_id, 'user_id': 'default', 'channel': 'console'}
        control.save(dest / 'submitted-request.json', payload)
        control.log(f'CHAT_START scene={index+1} layout={layout} chat={chat["id"]}')
        # The same streaming endpoint used by the Chat UI, no planner injection.
        with app_request('/console/chat', payload, timeout=3600) as response, (dest / 'stream.sse').open('wb') as stream:
            for line in response:
                stream.write(line)
                stream.flush()
        history = app_json('/chats/' + chat['id'])
        deadline = time.monotonic() + 30
        while history.get('status') == 'running' and time.monotonic() < deadline:
            time.sleep(2)
            history = app_json('/chats/' + chat['id'])
        if history.get('status') == 'running':
            raise TimeoutError('Chat stream ended but this trial Chat is still running')
    except Exception as error:
        errors.append(str(error))
    finally:
        if chat is not None:
            try:
                stop_own_chat(chat['id'], dest)
            except Exception as error:
                errors.append('Chat cleanup: ' + str(error))
        session = WORKSPACE / 'sessions' / f'default_{session_id}.json'
        try:
            if session.exists():
                shutil.copy2(session, dest / 'raw-session.json')
                tools, texts = memory_tools(session)
        except Exception as error:
            errors.append('Chat evidence: ' + str(error))
        try:
            stream_calls = stream_tool_calls(dest / 'stream.sse')
            control.save(dest / 'stream-tool-calls.json', stream_calls)
            tools = merge_tool_calls(stream_calls, tools)
        except Exception as error:
            errors.append('Completed stream tool evidence: ' + str(error))
        control.save(dest / 'model-tool-calls.json', tools)
        control.save(dest / 'visible-messages.json', texts)
        user_texts = [t['text'] for t in texts if t['role'] == 'user']
        row['exact_prompt_verified'] = user_texts == [prompt]
        try:
            owned = own_runs(before, test_id, session_id, index + 1)
        except Exception as error:
            owned = []
            errors.append('Run provenance: ' + str(error))
        for run, selection_file, selection in owned:
            try:
                entry = archive_run(run, selection_file, selection, layout, dest)
                row['runs'].append(entry)
                errors.extend(entry.get('validation_errors', []))
            except Exception as error:
                row['runs'].append({'run_id': run['run_id'], 'provider': run.get('policy_id'),
                                    'success': False, 'valid_episode': False, 'error': str(error)})
                errors.append('Run archive ' + run['run_id'] + ': ' + str(error))
        if audit_dir is not None:
            try:
                row['model_audit'] = collect_model_audit(audit_dir, session_id, dest, expected_model)
                if expected_model and not row['model_audit']['verified']:
                    errors.append('Requested and returned model audit does not verify ' + expected_model)
            except Exception as error:
                errors.append('Model audit: ' + str(error))
        row['automatic_skill_selected'] = any(
            t['name'] == 'robodojo_sim__start_session' and
            t['input'].get('policy_id') == 'physicalrsi-align' for t in tools)
        row['success'] = any(r['success'] for r in row['runs'])
        row['skill_success'] = row['automatic_skill_selected'] and any(
            r['success'] and r.get('provider') == 'physicalrsi-align' for r in row['runs'])
        row['status'] = 'completed' if row['runs'] else 'no_episode_started'
        if errors:
            row.update(status='test_error', error='; '.join(errors), errors=errors,
                       success=False, skill_success=False)
    control.save(dest / 'receipt.json', row)
    control.log('CHAT_END ' + json.dumps(row))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--layouts', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--prompt', default=PROMPT,
                        help='Exact sole user message (default preserves the bare-goal experiment)')
    parser.add_argument('--audit-dir', type=Path,
                        help='Directory of proxy JSON model records; context goes in its parent')
    parser.add_argument('--expected-model',
                        help='Require matching requested and actual returned model IDs')
    args = parser.parse_args()
    if args.expected_model and args.audit_dir is None:
        parser.error('--expected-model requires --audit-dir')
    if not args.prompt.strip():
        parser.error('--prompt must not be empty')
    deadline = time.monotonic() + 30
    while True:
        try:
            control.request('/health', timeout=5)
            break
        except (OSError, ValueError):
            if time.monotonic() >= deadline:
                raise RuntimeError('Test fixture bridge did not become ready')
            time.sleep(1)
    args.output.mkdir(parents=True, exist_ok=False)
    control.save(args.output / 'experiment.json', {
        'prompt': args.prompt, 'agent': 'car_controller', 'fresh_chat_per_scene': True,
        'scene_ids': args.layouts, 'seed': 0, 'driver_selects_provider': False,
        'driver_executes_actions': False, 'manual_assistance': False,
        'request_count': len(args.layouts), 'time': datetime.now(timezone.utc).isoformat(),
        'audit_dir': str(args.audit_dir) if args.audit_dir else None,
        'expected_model': args.expected_model,
        'note': 'Unmodified routing and tool permissions; fixture selection is outside the prompt'})
    rows = []
    for index, layout in enumerate(args.layouts):
        rows.append(trial(index, layout, args.output, args.fixture, prompt=args.prompt,
                          audit_dir=args.audit_dir, expected_model=args.expected_model))
        success = sum(r.get('success', False) for r in rows)
        skill = sum(r.get('skill_success', False) for r in rows)
        control.save(args.output / 'summary.json', {'requested': len(args.layouts),
            'finished': len(rows), 'task_successes': success,
            'task_success_rate': success / len(rows), 'automatic_skill_successes': skill,
            'automatic_skill_success_rate': skill / len(rows), 'scenes': rows})
        if rows[-1]['status'] == 'test_error':
            raise RuntimeError('Infrastructure error; preserve evidence and stop: ' + rows[-1]['error'])


if __name__ == '__main__':
    main()
