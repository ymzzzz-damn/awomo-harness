#!/usr/bin/env python3
"""Summarize preserved evidence; never perform robot or model calls."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from compare_local_tasks import save
from evaluate_fresh_chat_align import collect_model_audit, stream_tool_calls

ROOT = Path(__file__).resolve().parents[1]


def summarize(root):
    experiment = json.loads((root / 'experiment.json').read_text())
    rows = []
    for scene in sorted((root / 'chats').glob('scene-*')):
        receipt_file = scene / 'receipt.json'
        if not receipt_file.exists():
            continue
        receipt = json.loads(receipt_file.read_text())
        calls = stream_tool_calls(scene / 'stream.sse')
        save(scene / 'complete-stream-tool-calls.json', calls)
        audit = collect_model_audit(root / 'model-audit', receipt['session_id'], scene, experiment['model'])
        selected = any(c['name'] == 'robodojo_sim__start_session'
                       and c['input'].get('policy_id') == 'physicalrsi-align'
                       and c.get('output_completed') for c in calls)
        runs = []
        for run in receipt['runs']:
            dest = scene / run['run_id']
            trace = dest / 'harness-trace.jsonl'
            events = [json.loads(line) for line in trace.read_text().splitlines() if line] if trace.exists() else []
            chunks = [e for e in events if e.get('event') == 'action_chunk_ready']
            executions = [e for e in events if e.get('event') == 'action_chunk_execution']
            instruction = next((e['official_instruction'] for e in events if e.get('official_instruction')), None)
            policy_only = bool(executions) and all(e.get('source') == 'policy'
                and e.get('provider_id') == 'physicalrsi-align' for e in executions)
            original_prompts = bool(chunks) and all(e.get('optimized_task_prompt') == instruction for e in chunks)
            original_goals = bool(chunks) and all(e.get('subtask_goal') == instruction for e in chunks)
            videos = json.loads((dest / 'videos.json').read_text()) if (dest / 'videos.json').exists() else []
            verified_videos = []
            for video in videos:
                target_hash = hashlib.sha256(Path(video['path']).read_bytes()).hexdigest()
                original_hash = hashlib.sha256(Path(video['original']).read_bytes()).hexdigest()
                verified_videos.append(dict(video, original_sha256=original_hash,
                    byte_identical=(target_hash == original_hash == video['sha256'])))
            save(dest / 'verified-videos.json', verified_videos)
            success = bool(run.get('success')) and bool(run.get('valid_episode')) and policy_only
            runs.append(dict(run, success=success, actions=sum(e.get('executed', 0) for e in executions),
                policy_only=policy_only, original_task_prompts_preserved=original_prompts,
                subtask_goals_verbatim=original_goals, verified_videos=verified_videos))
        row = {'layout_id': receipt['layout_id'], 'chat_id': receipt['chat_id'],
            'session_id': receipt['session_id'], 'status': receipt['status'], 'runs': runs,
            'model_audit': audit, 'frozen_skill_selected_by_model': selected,
            'skill_success': selected and audit['verified'] and any(r['success'] for r in runs),
            'exact_submitted_prompt': json.loads((scene / 'submitted-request.json').read_text())['input'] == [
                {'role': 'user', 'content': [{'type': 'text', 'text': experiment['prompt']}]}],
            'tool_calls': len(calls)}
        rows.append(row)
    successes = sum(r['skill_success'] for r in rows)
    videos = [v for row in rows for run in row['runs'] for v in run['verified_videos']]
    checks = {'layouts_exactly_0_to_4': sorted(r['layout_id'] for r in rows) == list(range(5)),
        'exact_submitted_prompts': bool(rows) and all(r['exact_submitted_prompt'] for r in rows),
        'official_task_prompts_preserved': bool(rows) and all(
            run['original_task_prompts_preserved'] for row in rows for run in row['runs']),
        'fifteen_videos_verified': len(videos) == 15 and all(
            v['byte_identical'] and int(v.get('nb_read_frames', 0)) > 0 for v in videos)}
    summary = {'requested': 5, 'finished': len(rows), 'skill_successes': successes,
        'success_rate_of_requested': successes / 5,
        'complete': len(rows) == 5, 'model': experiment['model'], 'seed': 0,
        'reasoning_effort': experiment['reasoning_effort'], 'scenes': rows,
        'final_evidence_checks': checks,
        'source_archive_note': ('executed-source contains startup snapshots of the evaluator, driver, relay and fixture bridge; source contains the later archiving-time source. Submitted requests, complete SSE, model replies and native traces are the execution evidence.'
            if (root / 'executed-source').exists() else
            'Archived driver source includes a post-start SSE evidence-extraction fix; it is not claimed to be the byte-for-byte executed driver snapshot. Submitted requests, complete SSE, model replies and native traces are the execution evidence.'),
        'interrupted_experiment': experiment.get('interrupted_experiment'),
        'qualification': experiment['qualification'],
        'baseline': {'type': 'Direct frozen-skill execution WITHOUT LLM',
            'successes': 5, 'requested': 5,
            'directory': str(ROOT / 'eval_result/physicalrsi-skill-seed0-layouts0-4-20261013-180637')},
        'restoration': json.loads((root / 'restoration.json').read_text()) if (root / 'restoration.json').exists() else None}
    save(root / 'verified-summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--archive', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    summary = summarize(root)
    if args.archive:
        if not summary['complete']:
            raise RuntimeError('Do not mark an unfinished five-scene experiment as final')
        for name in ['run_gpt56_skill_chats.py', 'align56_audit_proxy.py',
                     'evaluate_fresh_chat_align.py', 'summarize_gpt56_skill_chats.py',
                     'chat_eval_bridge.py', 'launch_align_fixture_bridge.sh']:
            dest = root / 'source' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / 'scripts' / name, dest)
        archive = root / 'evidence.zip'
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as output:
            for path in sorted(root.rglob('*')):
                if path.is_file() and path != archive:
                    output.write(path, path.relative_to(root))
        with zipfile.ZipFile(archive) as output:
            if output.testzip() is not None:
                raise RuntimeError('Evidence archive failed integrity check')
    print(json.dumps({k: summary[k] for k in ['requested', 'finished', 'skill_successes', 'complete']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
