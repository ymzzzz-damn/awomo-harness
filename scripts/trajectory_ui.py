#!/usr/bin/env python3
"""Read-only, dependency-free explorer for saved RoboDojo evaluation evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
import mimetypes
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

WEB = Path(__file__).resolve().parents[1] / 'ui' / 'trajectory'
CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {} if default is None else default


def read_events(path):
    events = []
    try:
        for number, line in enumerate(Path(path).read_text(errors='replace').splitlines(), 1):
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    events.append(dict(item, line=number))
            except ValueError:
                pass  # A live writer may not have finished the last line yet.
    except OSError:
        pass
    return events


def identifier(value):
    return hashlib.sha256(str(value).encode()).hexdigest()[:24]


def classify(row, result):
    details = result.get('details') or {}
    detail = next(iter(details.values()), {})
    # Only a scored episode can be a task success/failure.
    valid = row.get('valid_episode')
    if valid is None:
        valid = bool(result.get('eval_time', 0)) or (row.get('score') is not None and row.get('status') == 'completed')
    if row.get('unscored_native_scene') or row.get('status') == 'native_unstable_before_actions':
        return 'unscored'
    if valid:
        success = detail.get('success', row.get('success', row.get('task_succeeded')))
        return 'success' if success is True else 'failure' if success is False else 'unscored'
    if row.get('error') or row.get('status') in ('failed', 'infrastructure_error', 'error'):
        return 'error'
    if row.get('status') in ('aborted', 'interrupted', 'stopped', 'stopped_by_user'):
        return 'unscored'
    return 'running' if row.get('status') in ('running', 'starting') else 'pending'


class EvidenceStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.scenes = {}
        self.files = {}
        self.catalog = {}
        self.scanned = 0
        self.lock = threading.RLock()

    def file_url(self, path):
        if not path:
            return None
        p = Path(path)
        if not p.is_absolute():
            p = self.root / p
        try:
            p = p.resolve(strict=True)
        except OSError:
            return None
        # Only discovered evidence files are exposed; clients cannot request paths.
        if not p.is_file() or p.suffix.lower() not in ('.mp4', '.webm', '.json', '.jsonl', '.log', '.jpg', '.png', '.md', '.py'):
            return None
        key = identifier(p)
        self.files[key] = p
        return '/media/' + key

    def scan(self, force=False):
        with self.lock:
            if not force and time.monotonic() - self.scanned < 5:
                return self.catalog
            self.scenes = {}
            for experiment in sorted(self.root.iterdir(), reverse=True):
                if not experiment.is_dir():
                    continue
                exp = read_json(experiment / 'experiment.json')
                if exp.get('schema') == 'physicalrsi.native-open-hybrid-view/v1':
                    from rsi_native_ui import add_native_hybrid
                    add_native_hybrid(self, experiment, exp)
                    continue
                if exp.get('schema') == 'physicalrsi.native-open-results/v1':
                    continue  # Mirrored raw evidence is shown through its stable UI registration.
                if exp.get('schema') == 'physicalrsi.open-online-baseline/v1':
                    self.add_rsi_baseline(experiment, exp)
                    continue
                if exp.get('schema') == 'awomopaw.open-online-harness/v1':
                    self.add_awomo_open(experiment)
                    continue
                if exp.get('schema') == 'physicalrsi.online-code-evolution/v1':
                    self.add_online_experiment(experiment, exp)
                    continue
                seen = set()
                # A receipt is the best preserved episode-level record.
                for receipt in sorted(experiment.rglob('receipt.json')):
                    row = read_json(receipt)
                    if not isinstance(row, dict) or not row.get('run_id'):
                        continue
                    if 'code-policies' in receipt.parts or row.get('source_sha256'):
                        continue  # Code worker receipts are not independent episodes.
                    if row['run_id'] in seen:
                        continue
                    official = read_json(receipt.parent / 'official-result.json')
                    run = official.get('run') or read_json(receipt.parent / 'run.json')
                    if not run:
                        run = read_json(receipt.parent / 'failure-evaluation-state.json')
                    result = run.get('result') or {}
                    case = read_json(receipt.parent / 'case.json')
                    task = row.get('task') or run.get('task') or case.get('task') or exp.get('task') or ('align_blocks' if 'align' in experiment.name else 'unknown')
                    merged = dict(run, **row)
                    detail = next(iter((result.get('details') or row.get('details') or {}).values()), {})
                    layout = row.get('layout_id', row.get('requested_layout_id', case.get('layout_id', case.get('layout', detail.get('layout_id', 0)))))
                    self.add_scene(experiment, receipt.parent, task, merged, result, layout, official, receipt)
                    seen.add(row['run_id'])
                # Chat receipts contain multiple runs rather than a top-level run_id.
                # The archived official record is authoritative for each of those runs.
                for record in sorted(experiment.rglob('official-result.json')):
                    official = read_json(record)
                    run = official.get('run') or {}
                    if not run.get('run_id') or run['run_id'] in seen:
                        continue
                    fixture = read_json(record.parent / 'fixture.json')
                    result = run.get('result') or {}
                    detail = next(iter((result.get('details') or {}).values()), {})
                    layout = detail.get('layout_id', fixture.get('layout_id', 0))
                    self.add_scene(experiment, record.parent, run.get('task', exp.get('task', 'unknown')),
                                   run, result, layout, official, record)
                    seen.add(run['run_id'])
                for record in sorted(experiment.rglob('run-state.json')):
                    run = read_json(record)
                    if not run.get('run_id') or run['run_id'] in seen:
                        continue
                    fixture = read_json(record.parent / 'fixture.json')
                    finalization = read_json(record.parent / 'native-finalization.json')
                    result = (finalization.get('run') or {}).get('result') or {}
                    interrupted = any(part in ('recovery', 'stopped-by-user') for part in record.relative_to(experiment).parts)
                    if interrupted:
                        run = dict(run, status='interrupted', valid_episode=False, score=None,
                                   qualification='本次执行已中断；不计入已评分成功率。')
                    self.add_scene(experiment, record.parent, run.get('task', exp.get('task', 'unknown')), run,
                                   result, fixture.get('layout_id', 0), finalization, record)
                    seen.add(run['run_id'])
                # Preserve in-progress snapshots, including scenes with no final receipt.
                for record in sorted(experiment.rglob('run.json')):
                    run = read_json(record)
                    if not run.get('task') or not run.get('run_id') or run['run_id'] in seen:
                        continue
                    fixture = read_json(record.parent / 'case.json')
                    match = re.search(r'layout-(\d+)', record.parent.name)
                    layout = fixture.get('layout_id', fixture.get('layout', int(match[1]) if match else 0))
                    self.add_scene(experiment, record.parent, run['task'], run, run.get('result') or {}, layout, {}, record)
                    seen.add(run['run_id'])
                # Comparison runs sometimes reference original artifacts directly.
                rows = read_json(experiment / 'results.json', [])
                if isinstance(rows, list):
                    for index, row in enumerate(rows):
                        if not isinstance(row, dict) or not row.get('run_id') or row['run_id'] in seen:
                            continue
                        result_path = row.get('official_result_path')
                        result = read_json(result_path) if result_path else {}
                        self.add_scene(experiment, experiment, row.get('task', exp.get('task', 'unknown')), row, result,
                                       row.get('layout_id', row.get('layout', 0)), {}, experiment / 'results.json', suffix=str(index))
                        seen.add(row['run_id'])
                # Native PhysicalRSI evaluations preserve official per-layout scores/videos.
                for result_path in sorted(experiment.rglob('_result.json')):
                    if result_path.parent.name != 'official':
                        continue
                    result = read_json(result_path)
                    native_root = result_path.parent.parent
                    summary = read_json(native_root / 'summary.json')
                    run = read_json(native_root / 'run.json') or read_json(experiment / 'run.json')
                    task = summary.get('task') or exp.get('task') or ('align_blocks' if 'align' in experiment.name or 'physicalrsi' in experiment.name else 'unknown')
                    for episode, detail in (result.get('details') or {}).items():
                        row = dict(run, status='completed', valid_episode=True, success=detail.get('success'), score=detail.get('score'),
                                   run_id=f'{native_root.name}:{episode}', native_episode=int(episode), seed=run.get('seed', 0))
                        self.add_scene(experiment, native_root, task, row, {'details': {episode: detail}, 'eval_time': 1},
                                       detail.get('layout_id', int(episode)), {}, result_path, suffix=episode)
                # Inventory explicitly distinguishes scenes not yet evaluated.
                inventories = [(p, read_json(p, [])) for p in experiment.rglob('official-case-inventory.json')]
                manifest_path = experiment / 'manifest.json'
                manifest = read_json(manifest_path)
                if manifest.get('schema') == 'awomopaw.full-official-scenes/v1':
                    inventories.append((manifest_path, manifest.get('cases', [])))
                active = read_json(experiment / 'status.json').get('running', [])
                for inventory, cases in inventories:
                    if not isinstance(cases, list):
                        continue
                    for case in cases:
                        task = case.get('task') or exp.get('task', 'align_blocks')
                        seed, layout = case.get('seed', 0), case.get('layout_id', case.get('layout', 0))
                        if any(s['experiment'] == experiment.name and s['task'] == task and s['seed'] == seed and s['layout'] == layout for s in self.scenes.values()):
                            continue
                        running = next((r for r in active if r.get('task') == task and r.get('seed') == seed and r.get('layout') == layout), None)
                        row = dict(case, status='running' if running else 'pending')
                        self.add_scene(experiment, inventory.parent, task, row, {}, layout, {}, inventory,
                                       suffix=f'pending-{task}-{seed}-{layout}')
                for layout in exp.get('layouts', []):
                    task, seed = exp.get('task', 'unknown'), exp.get('seed', 0)
                    if not any(s['experiment'] == experiment.name and s['task'] == task and s['seed'] == seed and s['layout'] == layout for s in self.scenes.values()):
                        self.add_scene(experiment, experiment, task, {'seed': seed, 'status': 'pending'}, {}, layout, {},
                                       experiment / 'experiment.json', suffix=f'pending-{task}-{seed}-{layout}')
                # A paired ablation is two independent groups over the same
                # official scene inventory, including their pending episodes.
                arms = manifest.get('arms', [])
                if set(arms) == {'with_skills', 'without_skills'} or arms == ['self_evolving']:
                    for case in manifest.get('cases', []):
                        task, seed = manifest['task'], manifest['seed']
                        layout = case['layout_id']
                        for arm in arms:
                            group = experiment.name + '::' + arm
                            if any(s['experiment'] == group and s['seed'] == seed and s['layout'] == layout
                                   for s in self.scenes.values()):
                                continue
                            folder = experiment / f'layout-{layout:02d}' / arm
                            row = dict(read_json(folder / 'receipt.json') or case,
                                       arm=arm, seed=seed, provider=manifest.get('provider', 'pi05'))
                            row.setdefault('status', 'pending')
                            self.add_scene(experiment, folder, task, row, {}, layout, {}, manifest_path)
            groups = read_json(self.root / 'ui-catalog.json').get('groups')
            if isinstance(groups, list):
                configured = {group['id']: group for group in groups}
                self.scenes = {key: scene for key, scene in self.scenes.items()
                               if scene['experiment'] in configured}
                for scene in self.scenes.values():
                    group = configured[scene['experiment']]
                    scene['experiment_title'] = group.get('title', scene.get('experiment_title'))
                    scene['framework'] = group.get('framework')
            public = [{k: v for k, v in s.items() if not k.startswith('_')} for s in self.scenes.values()]
            counts = {}
            experiments = {}
            for scene in public:
                counts[scene['task']] = counts.get(scene['task'], 0) + 1
                experiments[scene['experiment']] = experiments.get(scene['experiment'], 0) + 1
            if isinstance(groups, list):
                experiments = {group['id']: experiments[group['id']] for group in groups if group['id'] in experiments}
            self.catalog = {'tasks': [{'name': k, 'scenes': v} for k, v in sorted(counts.items())],
                            'experiments': [{'id': k, 'scenes': v,
                                'title': next((s.get('experiment_title') for s in public if s['experiment'] == k), None),
                                'framework': next((s.get('framework') for s in public if s['experiment'] == k), None),
                                'online_code_evolution': any(s.get('online_code_evolution') for s in public if s['experiment'] == k)}
                                for k, v in experiments.items()],
                            'scenes': public, 'root': str(self.root), 'updated': time.time()}
            self.scanned = time.monotonic()
            return self.catalog

    def add_online_experiment(self, experiment, registration):
        root = Path(registration['artifact_root']).resolve()
        summary = read_json(root / 'summary.json')
        # Native per-episode outcomes remain authoritative, including resumed runs.
        # Historical startup error files do not override completed episode scores.
        result_path = root / 'official/_result.json'
        native = read_json(result_path)
        outcomes = native.get('details') or {}
        status = read_json(root / 'status.json')
        episodes = dict(outcomes)
        for index in registration.get('cases', []):
            episodes.setdefault(str(index), {})
        for episode, outcome in sorted(episodes.items(), key=lambda item: int(item[0])):
            index = int(episode)
            completed = bool(outcome)
            active = status.get('case') == index and status.get('phase') == 'executing-case'
            candidate = read_json(root / f'ipc/assignments/case-{index:03d}.json').get('candidate', {})
            row = {'run_id': f'{root.name}:{episode}', 'seed': summary.get('seed', registration.get('seed', 0)),
                   'status': 'completed' if completed else 'running' if active else 'pending',
                   'valid_episode': completed, 'success': outcome.get('success'),
                   'score': outcome.get('score'), 'provider': registration.get('provider', 'PhysicalRSI · Code-policy'),
                   'videos': [str(p) for p in sorted((root / 'official').glob(f'episode_{index:07d}_*.mp4'))],
                   'qualification': registration.get('qualification')}
            self.add_scene(experiment, experiment, summary.get('task', registration['task']), row,
                           {'details': {episode: outcome}, 'eval_time': 1} if completed else {}, outcome.get('layout_id', index),
                           {}, result_path, suffix=episode)
            scene = self.scenes[identifier(str(result_path) + episode)]
            scene.update(experiment_title=registration.get('title'), online_code_evolution=True,
                         policy_id=candidate.get('id'), framework=registration.get('framework'),
                         paired_validation=registration.get('paired_validation', False),
                         video_rethink_enabled=registration.get('video_rethink_enabled', False),
                         evidence_revision=max((p.stat().st_mtime_ns for p in [
                             root / f'feedback/case-{index:03d}.json',
                             root / f'selections/case-{index:03d}.json',
                             root / f'rethinking/case-{index:03d}/accepted.json'] if p.is_file()), default=0),
                         _online_root=root, _episode_index=index)

    def online_detail(self, scene):
        root, index = scene['_online_root'], scene['_episode_index']
        case = f'case-{index:03d}'
        bound = read_json(root / 'ipc/assignments' / f'{case}.json').get('candidate', {})
        candidate = Path(bound['root']) if bound.get('root') else root / 'candidates' / case
        trace = root / f'execution/episode-{index:03d}/events.jsonl'
        events = read_events(trace)
        feedback = read_json(root / 'feedback' / f'{case}.json')
        accepted = read_json(root / 'updates' / case / 'accepted-proposal.json')
        proposal_root = Path(accepted.get('directory', root / 'updates' / case))
        proposal = read_json(proposal_root / 'skill-proposal.json')
        next_accepted = read_json(root / 'updates' / f'case-{index + 1:03d}/accepted-proposal.json')
        next_root = Path(next_accepted.get('directory', root / 'updates' / f'case-{index + 1:03d}'))
        next_proposal = read_json(next_root / 'skill-proposal.json')
        reflection_receipt = read_json(root / 'rethinking' / case / 'accepted.json')
        reflection_root = Path(reflection_receipt['directory']) if reflection_receipt else None
        reflection = read_json(reflection_root / 'harness-feedback.json') if reflection_root else {}
        steps, evidence = [], []
        stage = 'initial'
        for event in events:
            kind = event.get('event')
            if kind in ('approach', 'servo', 'servo_position', 'gripper'):
                stage = kind
            arm = event.get('arm')
            if kind == 'move_joints' and arm in ('left', 'right'):
                steps.append({'step': event.get('step', 0), 'stage': stage, 'line': event['line'],
                              'action': {f'{arm}_arm_joint_state': event.get('joints', [])}})
            elif kind == 'gripper' and arm in ('left', 'right'):
                steps.append({'step': event.get('step', 0), 'stage': stage, 'line': event['line'],
                              'action': {f'{arm}_ee_joint_state': [event.get('command')]}})
            elif kind not in ('initial', 'scene', 'observed_state', 'policy_returned'):
                item = dict(event, count=event.get('step', 0), stage=kind)
                if event.get('pose'):
                    item['target'] = event['pose'][:3]
                evidence.append(item)
        files = []
        def source(path, label):
            url = self.file_url(path)
            if url:
                files.append({'label': label, 'url': url, 'path': str(path)})
        for path, label in [
            (root / 'summary.json', '实验汇总 · 在线 55 场景'),
            (root / 'official/_result.json', '官方逐场评分'),
            (root / 'ipc/requests' / f'{case}.json', '本场景初始观测与任务'),
            (candidate / 'manifest.json', '本场景策略版本与来源'),
            (candidate / 'policy.py', '本场景实际执行的 policy'),
            (candidate / 'memory_rules.json', '本场景执行前的 memory'),
            (proposal_root / 'skill-proposal.json', '生成本场景策略的模型输出'),
            (trace, '本场景执行轨迹'),
            (root / 'feedback' / f'{case}.json', '本场景评分、执行反馈与运行后 memory'),
            (next_root / 'skill-proposal.json', '本场景反馈后的模型更新'),
            (root / 'updates' / case / 'input-provenance.json', '策略输入来源 · 下一场景观测隔离'),
            (root / 'experiment-plan.json', '本次重测协议'),
            (root / 'prior-protocol-audit.json', '旧流程缺项与重测原因'),
            (root / 'task-artifact/SKILL.md', '最终蒸馏的 SKILL'),
            (root / 'task-artifact/policy.py', '最终蒸馏的 policy · 未单独评测'),
            (root / 'task-artifact/memory.json', '最终蒸馏的 memory'),
            (root / 'verification.json', '覆盖与模型输出校验'),
            (root / 'reconciliation.json', '续跑与结果核对记录'),
            (root / 'protocol-source-audit.json', '原流程复用与改动边界'),
            (root / 'selections' / f'{case}.json', '原选择器 · 本场景新旧配对决定'),
            (root / 'selected-policy.py', '原选择器最后采用的 policy'),
        ]:
            source(path, label)
        if reflection_root:
            for name, label in [('video-evidence.json', 'VideoRethink · 三路视频抽帧证据'),
                                ('diagnosis.json', 'VideoRethink · 视频诊断'),
                                ('rethink.json', 'VideoRethink · 策略修改与 memory 计划'),
                                ('harness-feedback.json', 'VideoRethink · 返回策略生成器的反馈')]:
                source(reflection_root / name, label)
        request = read_json(root / 'ipc/requests' / f'{case}.json')
        for path, label in [(request.get('head_image'), '本场景初始观测图'),
                            (feedback.get('end_image'), '本场景结束观测图')]:
            if path:
                source(Path(path), label)
        previous = read_json(root / 'official/_result.json').get('details', {})
        source_layouts = [outcome.get('layout_id', int(episode)) for episode, outcome in previous.items()
                          if int(episode) < index]
        memory = read_json(candidate / 'memory_rules.json')
        rationale = proposal.get('rationale', '')
        detail = {k: v for k, v in scene.items() if not k.startswith('_')}
        detail.update(instruction=request.get('instruction', feedback.get('instruction', '')),
                      rationale=rationale, messages=[{'role': 'assistant', 'text': rationale}] if rationale else [],
                      calls=[], steps=steps, evidence=evidence, events=events,
                      total_steps=max((event.get('step', 0) for event in events), default=0),
                      files=files, error=None, official=scene['_result'],
                      trajectory_source='Code-policy 原始执行事件 · 仅展示已记录的动作目标',
                      learning={'kind': 'code-policy', 'version': index, 'source_layouts': source_layouts,
                                'task_memory': json.dumps(memory, ensure_ascii=False, indent=2),
                                'update_rationale': next_proposal.get('rationale', ''), 'failure_analysis': {}})
        if scene.get('paired_validation'):
            paired = read_json(root / 'selections' / f'{case}.json')
            old = read_json(root / 'comparisons' / f'{case}-incumbent.json')
            old_directory = Path(paired.get('incumbent_directory', old.get('directory', root / 'missing')))
            old_result = read_json(old_directory / 'official/_result.json')
            old_outcome = next(iter(old_result.get('details', {}).values()), {})
            for video in sorted((old_directory / 'official').glob('*.mp4')):
                source(video, '对照旧版视频 · ' + next((camera for camera in CAMERAS if camera in video.name), video.name))
            source(old_directory / 'official/_result.json', '对照旧版 · 官方评分')
            source(old_directory / 'execution/episode-000/events.jsonl', '对照旧版 · 执行轨迹')
            detail['comparison'] = {
                'status': 'bootstrap' if index == 0 else 'completed' if paired else 'pending',
                'incumbent': paired.get('incumbent_before', old.get('candidate', {}).get('id')),
                'candidate': paired.get('candidate', bound.get('id')),
                'incumbent_score': old_outcome.get('score'),
                'candidate_score': scene.get('score'),
                'decision': paired.get('decision', {}).get('decision'),
                'survivor': paired.get('incumbent_after'),
                'source_scene': paired.get('source_scene', index - 1 if index else None),
                'validation_scene': index}
        if scene.get('video_rethink_enabled'):
            detail['learning']['video_rethinking'] = {
                'status': 'completed' if reflection else 'bootstrap' if scene.get('paired_validation') and index == 0 else 'not_required' if scene['status'] == 'success' else 'pending',
                'diagnosis': (reflection.get('diagnosis') or {}).get('failure_reason', ''),
                'policy_changes': (reflection.get('rethinking') or {}).get('policy_changes', []),
                'cameras': len(reflection.get('video_evidence', [])),
                'frames': sum(len(v['frames']) for v in reflection.get('video_evidence', []))}
        return detail

    def add_awomo_open(self, experiment):
        seen = set()
        model=read_json(experiment/'manifest.json').get('model')
        stopped = (read_json(experiment/'user-requested-stop-for-rsi-hybrid.json').get('stopped') is True
                   and read_json(experiment/'ui-sync.json').get('backend_alive') is not True)
        for receipt in sorted((experiment/'tasks').glob('*/seed-*/layout-*/self_evolving/receipt.json')):
            row = read_json(receipt)
            row['awomo_open'] = True
            row['model']=row.get('model') or model
            key = (row.get('task'), row.get('seed'), row.get('layout_id'))
            if key in seen:
                continue
            official = read_json(receipt.parent/'official-result.json')
            if (stopped and not row.get('valid_episode') and row.get('status') in {'starting', 'running'}
                    and not ((official.get('run') or {}).get('result') or {}).get('eval_time')):
                row = dict(row, status='stopped_by_user',
                           qualification='已按用户指令停止；此中断执行未评分，不计入成功率。')
            self.add_scene(experiment, receipt.parent, key[0], row,
                (official.get('run') or {}).get('result') or {}, key[2], official, receipt)
            seen.add(key)
        for case in read_json(experiment/'official-case-inventory.json', []):
            key = (case['task'], case['seed'], case['layout_id'])
            if key in seen:
                continue
            folder=experiment/'tasks'/key[0]/f'seed-{key[1]}'/f'layout-{key[2]:02d}'/'self_evolving'
            self.add_scene(experiment, folder, key[0], dict(case, arm='self_evolving', status='pending',
                provider='gpt6-harness', awomo_open=True,model=model), {}, key[2], {}, experiment/'official-case-inventory.json')

    def add_rsi_baseline(self, experiment, registration):
        """One primary trial per task/seed/layout; paired controls stay evidence."""
        seen = set()
        backend_stopped = read_json(experiment / 'ui-sync.json').get('backend_alive') is False
        for receipt in sorted((experiment / 'tasks').rglob('receipt.json')):
            parts = receipt.relative_to(experiment).parts
            primary_bootstrap = len(parts) == 4 and parts[2] == 'bootstrap'
            primary_round = len(parts) == 6 and parts[2] == 'rounds' and parts[4] in {'candidate', 'retained'}
            # Archived attempts and paired incumbents are evidence, not the
            # current primary trial for this official scene identity.
            if not (primary_bootstrap or primary_round):
                continue
            row = read_json(receipt)
            if not row.get('run_id'):
                continue
            key = (row.get('task'), row.get('seed', 0), row.get('layout_id', 0))
            if key in seen:
                continue
            row['rsi_baseline'] = True
            if row.get('scene_variant') == 'repaired-native-fixture-v1':
                row['qualification'] = '修复场景：杂物盒沿 y 方向移动 6.5 厘米。'
            if backend_stopped and row.get('status') in {'starting', 'running'}:
                row.update(status='infrastructure_error', valid_episode=False,
                           error='远端队列已停止，等待恢复；此执行没有计入官方成功率。')
            official = read_json(receipt.parent / 'official-result.json')
            self.add_scene(experiment, receipt.parent, key[0], row,
                           (official.get('run') or {}).get('result') or {}, key[2], official, receipt)
            seen.add(key)
        for case in read_json(experiment / 'official-case-inventory.json', []):
            key = (case['task'], case['seed'], case['layout_id'])
            if key not in seen:
                self.add_scene(experiment, experiment, key[0], dict(case, status='pending', rsi_baseline=True), {}, key[2], {},
                               experiment / 'official-case-inventory.json', suffix=':'.join(map(str, key)))

    def add_scene(self, experiment, folder, task, row, result, layout, official, source, suffix=''):
        arm = row.get('arm') or (folder.name if folder.name in {'with_skills', 'without_skills', 'self_evolving'} else None)
        group = experiment.name + '::' + arm if arm else experiment.name
        task_key = ':'+task if row.get('awomo_open') else ''
        key = identifier(f'{group}{task_key}:{row.get("seed", 0)}:{layout}') if arm else identifier(str(source) + suffix)
        videos = sorted((folder / 'videos').glob('*.mp4'))
        if not videos:
            episode = row.get('native_episode')
            videos = sorted((folder / 'official').glob(f'episode_{episode:07d}_*.mp4')) if episode is not None else []
        if not videos:
            videos = [Path(v['path'] if isinstance(v, dict) else v) for v in row.get('videos', official.get('videos', []))]
        replay = False
        if not any(v.is_file() for v in videos):
            videos = sorted((folder / 'public-frame-replays').glob('*.mp4'))
            replay = bool(videos)
        cameras = {}
        for camera in CAMERAS:
            p = next((v for v in videos if camera in v.name and v.is_file()), None)
            if p:
                cameras[camera] = self.file_url(p)
        self.scenes[key] = {'id': key, 'experiment': group, 'arm': arm, 'task': task, 'seed': row.get('seed', 0),
            'rsi_baseline': row.get('rsi_baseline', False), 'candidate_id': row.get('candidate_id'),
            'layout': layout, 'status': classify(row, result), 'score': row.get('score', result.get('score')),
            'provider': ('Awomo · '+row['model']) if row.get('awomo_open') and row.get('model') else
                row.get('provider') or row.get('provider_id') or row.get('policy_id') or row.get('mode') or 'PhysicalRSI',
            'model':row.get('model'),
            'run_id': row.get('run_id'), 'actions': row.get('actions', row.get('action_count', 0)),
            'cameras': cameras, 'variant': str(folder.relative_to(experiment)),
            'qualification': ('中断执行的公共 RGB 帧回放，非完整官方评测视频。' if replay else row.get('qualification')),
            '_folder': folder, '_experiment': experiment, '_source': source, '_row': row, '_result': result, '_official': official}

    def detail(self, key):
        with self.lock:
            scene = self.scenes.get(key)
            if not scene:
                raise KeyError(key)
            if scene.get('rsi_native_hybrid'):
                from rsi_native_ui import native_detail
                return native_detail(self, scene)
            if scene.get('online_code_evolution'):
                return self.online_detail(scene)
            folder, row, official = scene['_folder'], scene['_row'], scene['_official']
            files = []
            def source(p, label):
                url = self.file_url(p)
                if url:
                    files.append({'label': label, 'url': url, 'path': str(p)})
            source(scene['_source'], '场景结果')
            source(folder / 'official-result.json', '官方评分')
            if read_json(scene['_experiment'] / 'experiment.json').get('schema') == 'physicalrsi.open-online-baseline/v1':
                source(scene['_experiment'] / 'summary.json', '全部任务 · 成功率与覆盖范围')
                source(folder.parent / 'selection.json', 'RSI · 原版选择器决定')
                source(folder.parent / 'harness-candidate/policy.py', 'RSI · 本场候选策略')
                source(folder.parent / 'harness-candidate/memory_rules.json', 'RSI · 本场候选记忆')
                source(folder.parent / 'development-feedback.json', 'RSI · 上一场反馈')
                source(folder / 'native-scene-repair-applied.json', '原生场景修复记录')
                source(folder / 'selected-scene.json', '原生场景与来源')
                for response in sorted(folder.parent.glob('generation-*/model-response.json')):
                    source(response, 'RSI · 模型生成记录')
            experience_path=scene['_experiment']/'learning/snapshots'/f'layout-{scene["layout"]:02d}.json'
            learning_path=folder/'learning'
            if row.get('awomo_open'):
                source(scene['_experiment']/'user-requested-stop-for-rsi-hybrid.json', '用户停止回执')
                experience_path=scene['_experiment']/'tasks'/scene['task']/'learning/snapshots'/f'case-{row["case_index"]:03d}.json'
                learning_path=folder/'task-learning'
            experience = read_json(experience_path)
            if scene.get('arm') == 'self_evolving':
                source(scene['_experiment'] / 'comparison.json', '相同官方场景的基线对比')
                source(experience_path, '本场景使用的经验')
                source(folder / 'experience-reads.json', '经验读取记录')
                source(learning_path/'analysis.json', '任务复盘与新经验')
                source(learning_path/'evidence.json', '复盘帧与轨迹来源')
                source(learning_path/'experience-after.json', '传给下一场景的经验')
                for frame in sorted((learning_path/'frames').glob('*/*.jpg')):
                    source(frame, f'复盘关键帧 · {frame.parent.name} · {frame.stem}')
            trace = next((p for p in [folder / 'harness-trace.jsonl', folder / 'awomopaw_harness_trace.jsonl',
                         Path(row.get('harness_trace') or official.get('harness_trace') or '/nonexistent')] if p.is_file()), None)
            events = read_events(trace) if trace else []
            if trace:
                source(trace, 'Harness 轨迹')
            log = next((p for p in [folder / 'native-run.log', folder / 'worker.log',
                       Path((official.get('run') or row).get('log_path') or '/nonexistent')] if p.is_file()), None)
            worker_events = []
            if log:
                source(log, '技能 / 仿真日志')
                for number, line in enumerate(log.read_text(errors='replace').splitlines(), 1):
                    match = re.search(r'ALIGN\w* (\{.*\})', line)
                    if match:
                        try:
                            worker_events.append(dict(json.loads(match[1]), line=number))
                        except ValueError:
                            pass
            # Chat records are scoped to this episode's scene, never another trial.
            messages, calls = [], []
            experiment = scene['_experiment']
            for parent in [folder, *folder.parents]:
                if not parent.is_relative_to(experiment):
                    break
                if (parent / 'agent-messages.jsonl').is_file():
                    for record in read_events(parent / 'agent-messages.jsonl'):
                        message = record.get('message', {})
                        blocks = message.get('content', [])
                        if not isinstance(blocks, list):
                            blocks = [{'type': 'text', 'text': blocks}] if isinstance(blocks, str) else []
                        text = '\n'.join(b.get('text', '') for b in blocks if isinstance(b, dict) and b.get('type') == 'text')
                        if text and message.get('role') in {'user', 'assistant'}:
                            messages.append({'role': message['role'], 'text': text})
                        calls.extend({'name': b['name'], 'input': b.get('input', {}), 'call_id': b.get('id')}
                                     for b in blocks if isinstance(b, dict) and b.get('type') == 'tool_use')
                    source(parent / 'agent-messages.jsonl', 'Agent 可见消息与工具调用')
                    source(parent / 'skill-reads.json', 'Skill 读取记录')
                    break
                if (parent / 'visible-messages.json').is_file():
                    messages = read_json(parent / 'visible-messages.json', [])
                    calls = read_json(parent / 'model-tool-calls.json', [])
                    source(parent / 'visible-messages.json', 'Agent 可见消息')
                    source(parent / 'model-tool-calls.json', 'Agent 工具调用')
                    break
                if (parent / 'chat-history.json').is_file():
                    history = read_json(parent / 'chat-history.json')
                    messages = [{'role': m.get('role'), 'text': '\n'.join(c.get('text', '') for c in m.get('content', []) if c.get('type') == 'text')}
                                for m in history.get('messages', []) if m.get('role') in ('assistant', 'user')]
                    messages = [m for m in messages if m['text']]
                    calls = read_json(parent / 'complete-tool-calls.json', [])
                    source(parent / 'chat-history.json', 'Agent 可见消息')
                    source(parent / 'complete-tool-calls.json', 'Agent 工具调用')
                    break
            chunks = [e for e in events if e.get('event') == 'action_chunk_ready']
            executions = [e for e in events if e.get('event') == 'action_chunk_execution']
            instruction = next((e.get('official_instruction') for e in events if e.get('official_instruction')), '')
            if not instruction:
                instruction = read_json(folder / 'initial-observation.json').get('instruction', '')
            steps = []
            for e in worker_events:
                if 'action' in e:
                    steps.append({'step': e.get('count', 0), 'stage': e.get('stage'), 'action': e['action'], 'line': e['line']})
            evidence = [e for e in worker_events if 'action' not in e and 'scene' not in e]
            if not steps:
                count = 0
                for e in executions:
                    count = e.get('action_count', count + e.get('executed', 0))
                    steps.append({'step': count, 'stage': e.get('subtask_goal') or '执行动作窗口', 'execution': e, 'line': e['line']})
            if not steps:
                for p in sorted((folder / 'execution').glob('*.json')):
                    e = read_json(p)
                    steps.append({'step': e.get('action_count', len(steps) + 1), 'stage': e.get('subtask_goal') or '执行动作窗口', 'execution': e})
            max_step = max([scene['actions'], *[s['step'] for s in steps]], default=0)
            rationale = next((c.get('planning_summary') for c in chunks if c.get('planning_summary')), '')
            result = {k: v for k, v in scene.items() if not k.startswith('_')}
            result.update(instruction=instruction, rationale=rationale, messages=messages, calls=calls,
                          steps=steps, evidence=evidence, events=events, total_steps=max_step, files=files,
                          error=row.get('error'), official=scene['_result'],
                          trajectory_source='技能日志' if worker_events else 'Harness 执行回执' if steps else '未保存逐步轨迹')
            if scene.get('arm') == 'self_evolving':
                result['learning'] = {'version':experience.get('version',0),
                    'source_layouts':[e['source_layout_id'] for e in experience.get('entries',[])],
                    'task_memory':experience.get('task_memory',''),
                    'failure_analysis':read_json(learning_path/'analysis.json')}
                if row.get('awomo_open'):
                    result['learning'].update(kind='task-artifacts',source_layouts=[
                        f"seed {e['source_seed']} / layout {e['source_layout_id']}" for e in experience.get('entries',[])])
            video_records = read_json(folder / 'videos.json', [])
            if isinstance(video_records, list):
                head = next((v for v in video_records if isinstance(v, dict) and 'cam_head' in str(v.get('path'))), {})
                frames = head.get('nb_read_frames')
                if frames and str(frames).isdigit():
                    result['video_frames'] = int(frames)
            return result


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        request = urlsplit(self.path)
        try:
            if request.path == '/api/catalog':
                self.json(self.server.store.scan(parse_qs(request.query).get('refresh') == ['1']))
            elif request.path.startswith('/api/scenes/'):
                self.server.store.scan()
                self.json(self.server.store.detail(request.path.rsplit('/', 1)[-1]))
            elif request.path.startswith('/media/'):
                path = self.server.store.files.get(request.path.rsplit('/', 1)[-1])
                if path is None:
                    raise KeyError('Unknown evidence file')
                self.file(path)
            else:
                name = {'/': 'index.html', '/app.js': 'app.js', '/style.css': 'style.css'}.get(request.path)
                if name is None:
                    raise KeyError('Unknown route')
                self.file(WEB / name)
        except (KeyError, FileNotFoundError):
            self.json({'error': '未找到该记录或文件'}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            self.json({'error': str(error)}, 500)

    def json(self, value, code=200):
        data = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def file(self, path):
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        requested = self.headers.get('Range')
        if requested:
            match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested)
            if not match or not any(match.groups()):
                return self.range_error(size)
            left, right = match.groups()
            if left:
                start = int(left)
                end = min(int(right), size - 1) if right else size - 1
            else:
                start = max(0, size - int(right))
            if start >= size or end < start:
                return self.range_error(size)
            status = 206
        self.send_response(status)
        mime = ('text/plain; charset=utf-8' if path.suffix.lower() in ('.md', '.py')
                else mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(max(0, end - start + 1)))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('X-Content-Type-Options', 'nosniff')
        if status == 206:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        with path.open('rb') as stream:
            stream.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                data = stream.read(min(256 * 1024, remaining))
                if not data:
                    break
                self.wfile.write(data)
                remaining -= len(data)

    def range_error(self, size):
        self.send_response(416)
        self.send_header('Content-Range', f'bytes */{size}')
        self.send_header('Content-Length', '0')
        self.end_headers()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1] / 'eval_result')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8790)
    args = parser.parse_args()
    if not args.root.is_dir():
        parser.error('--root must be an existing evaluation directory')
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.store = EvidenceStore(args.root)
    catalog = server.store.scan()
    print(f'Trajectory UI: http://{args.host}:{server.server_port} ({len(catalog["scenes"])} scenes)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
