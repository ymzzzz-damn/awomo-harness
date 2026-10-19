"""Read-only presentation of verified native hybrid official episodes."""
import hashlib
import json
from pathlib import Path

CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')


def recovery_queue_path(root):
    candidates = [p for p in root.glob('continuations/official-preaction-approved-v*/queue-status.json')
                  if p.parent.name.rsplit('-v', 1)[-1].isdigit()]
    return max(candidates, key=lambda p: int(p.parent.name.rsplit('-v', 1)[-1])) if candidates else None


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {} if default is None else default


def local_path(path, root, remote):
    target = (root / Path(path).relative_to(remote)).resolve()
    if not target.is_relative_to(root):
        raise ValueError('Native evidence escaped the mirrored run')
    return target


def compact_score(store, path, pin):
    if not pin.get('native_evidence_verified') or not path.is_file():
        return {}
    stat = path.stat()
    signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, pin.get('sha256'))
    cache = store.__dict__.setdefault('_native_score_cache', {})
    cached = cache.get(str(path))
    if cached and cached[0] == signature:
        return cached[1]
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != pin.get('sha256'):
        return {}
    score = json.loads(data)
    result = score.get('native_result', {})
    if (result.get('state') != 'completed' or result.get('natural_terminal') is not True
            or type(result.get('success')) is not bool):
        return {}
    value = {k: score.get(k) for k in ('task', 'seed', 'layout_id', 'layout_sha256',
        'selected_expert', 'selected_revision', 'selected_harness_sha256', 'evolution_round',
        'original_evolution_completed', 'official_preaction_recovery', 'official_initialization_recovery')}
    value.update(success=result['success'], score=result['episode_score'],
        actions=result.get('diagnostics', {}).get('action_count', 0),
        videos=[p for p in result.get('evidence', {}) if p.endswith('.mp4')])
    cache[str(path)] = (signature, value)
    return value


def add_native_hybrid(store, experiment, registration):
    root = Path(registration['artifact_root']).resolve()
    remote = registration['remote_output']
    sync = read(root / 'ui-sync.json')
    paused = read(root / 'user-pause.json').get('paused') is True
    recovery_queue = recovery_queue_path(root)
    queue = read(recovery_queue or root / 'continuations/approved-v1/queue-status.json')
    progress = {}
    for case in read(root / 'official-inventory.json', []):
        task, seed, layout = case['task'], case['seed'], case['layout_id']
        lane = root / 'tasks' / task
        if task not in progress:
            progress[task] = read(lane / 'progress.json')
        current = progress[task]
        scene = lane / 'official' / f'seed-{seed}-layout-{layout:02d}'
        score_path = scene / 'score.json'
        relative = str(score_path.relative_to(root))
        score = compact_score(store, score_path, sync.get('verified_scores', {}).get(relative, {}))
        if score and ((score['task'], score['seed'], score['layout_id']) != (task, seed, layout)
                      or score['layout_sha256'] != case['layout_sha256']):
            score = {}
        active = current.get('scene', {})
        is_current = (active.get('seed'), active.get('layout_id')) == (seed, layout)
        failure = queue.get('failed_tasks', {}).get(task) if is_current else None
        status = ('success' if score['success'] else 'failure') if score else (
            'error' if failure else 'running' if is_current and current.get('state') != 'completed' else 'pending')
        cameras = {}
        for camera in CAMERAS:
            name = next((p for p in score.get('videos', []) if camera in Path(p).name), None)
            if name:
                path = local_path(name, root, remote)
                url = store.file_url(path)
                if url:
                    cameras[camera] = url
        expert = score.get('selected_expert') or {}
        fallback = score.get('original_evolution_completed') is False
        phase = current.get('state') if is_current else None
        note = ('本场进化失败，按已授权规则使用最后已提交策略完成正式评测。' if fallback else
                '正在本场正式评测前执行原版 RSI 自进化；尚未计分。' if phase == 'evolving' and not score else
                '正在执行独立正式场景；等待原生终止与评分核验。' if is_current and not score else '')
        if failure and not score:
            note = '本场执行待核对，尚未计入正式成功率。'
        if score.get('official_initialization_recovery'):
            note = '本场原布局初始化失败后，按授权以相同布局、参数和稳定性检查重试并完成正式评测。'
        elif score.get('official_preaction_recovery'):
            note = '本场先前因接口异常在执行前中断，按授权保留证据后完成了独立重试。'
        elif is_current and current.get('official_initialization_recovery'):
            note = ('本场同布局初始化重试未完成，仍未评分。' if failure else
                    '正在按授权以相同布局、参数和稳定性检查重试初始化；尚未评分。')
        elif is_current and current.get('official_preaction_recovery'):
            note = '本场已验证尚未执行动作，正在使用同一策略重试正式评测。'
        if paused and not score:
            status = 'error' if failure else 'pending'
            phase = 'paused_by_user'
            note = ('用户已暂停本次评测。' + note if failure else
                    '用户已暂停本次评测；本场尚未评分，已有执行记录保留。')
        key = hashlib.sha256(f'{experiment.name}:{task}:{seed}:{layout}'.encode()).hexdigest()[:24]
        store.scenes[key] = dict(id=key, experiment=experiment.name, experiment_title=registration.get('title'),
            framework='RSI', arm=None, task=task, seed=seed, layout=layout,
            status=status, score=score.get('score'), rsi_native_hybrid=True,
            rsi_baseline=False, candidate_id=expert.get('name'), selected_expert=expert,
            provider='PhysicalRSI · hybrid' + (' · ' + expert['name'] if expert else ''),
            run_id=f'{Path(remote).name}:{task}:{seed}:{layout}' if score else None,
            actions=score.get('actions', 0), cameras=cameras, variant=str(scene.relative_to(root)),
            qualification=note, evaluation_phase=phase, external_parent_fallback=fallback,
            evidence_revision=max(score_path.stat().st_mtime_ns if score_path.exists() else 0,
                (lane / 'progress.json').stat().st_mtime_ns if is_current and not score and (lane / 'progress.json').exists() else 0),
            _native_root=root, _native_remote=remote, _folder=scene, _native_score=score,
            _native_registration=experiment, _source=score_path, _native_failure=failure,
            _native_progress=current if is_current else {})


def native_detail(store, scene):
    root, folder, remote = scene['_native_root'], scene['_folder'], scene['_native_remote']
    score = scene['_native_score']
    recovery = score.get('official_preaction_recovery') or {}
    pending_attempt = scene.get('_native_progress', {}).get('official_preaction_recovery')
    if not recovery and pending_attempt:
        recovery = dict(attempt=pending_attempt)
    execution = local_path(recovery['attempt'], root, remote) if recovery else folder
    receipt = read(execution / 'episode/episodes/0.json') if score else {}
    cases = read(execution / 'cohort/cases.json', [])
    instruction = cases[0].get('instruction', '') if cases else ''
    request = read(execution / 'episode/native/request.json')
    if not instruction:
        instruction = request.get('instruction', '')
    files = []
    plans = [(path, read(path)) for path in sorted((execution / 'episode/model/agent-plans').glob('*.json'))]
    if not instruction:
        instruction = next((plan.get('instruction') for _, plan in plans if plan.get('instruction')), '')
    messages = []
    for _, plan in plans:
        value = plan.get('plan', {})
        text = value.get('subgoal') if isinstance(value, dict) else None
        if text:
            messages.append(dict(role='assistant', text=text))

    def source(path, label):
        url = store.file_url(path)
        if url:
            files.append(dict(label=label, url=url, path=str(path)))

    for path, label in [(root / 'summary.json', '全部任务 · 原始正式评分汇总'),
        (root / 'ui-sync.json', '本地回传与评分证据核验'), (folder / 'score.json', '本场原始正式评分'),
        (execution / 'episode/episodes/0.json', '原生终止回执与动作步数'),
        (execution / 'cohort/official-provenance.json', '固定官方场景来源'),
        (execution / 'continuation-official-intent.json', '正式执行绑定与策略版本'),
        (recovery_queue_path(root) or root / 'continuations/approved-v1/queue-status.json', '当前恢复与原队列合并状态'),
        (root / 'continuations/approved-v1/queue-status.json', '当前全量队列状态')]:
        source(path, label)
    if recovery:
        proof = recovery.get('proof') or read(execution / 'initialization-retry-intent.json').get('proof')
        if proof:
            source(local_path(proof, root, remote), '正式重试前的零动作与进程清理证明')
        source(execution / 'score.json', '独立重试的原始正式评分')
        source(folder / 'episode/native/processes/simulator/process.log', '保留的原始中断尝试日志')
    if score.get('official_initialization_recovery') or scene.get('_native_progress', {}).get('official_initialization_recovery'):
        source(folder / 'cohort/initialization' / scene['task'] / '0/processes/simulator/process.log',
               '保留的原始初始化失败日志')
        source(execution / 'initialization-retry-intent.json', '同布局初始化重试绑定与一次限额')
        source(execution / 'initialization-retry-failure.json', '同布局初始化重试失败原始记录')
        source(execution / 'cohort/initialization' / scene['task'] / '0/processes/simulator/process.log',
               '本次同布局初始化重试原始日志')
    for path, _ in plans:
        source(path, '本场保存的执行计划与官方指令')
    journal = read(folder / 'evolution-start.json')
    round_name = score.get('evolution_round') or journal.get('round')
    round_dir = local_path(round_name, root, remote) if round_name else None
    evidence = []
    if round_dir:
        for name, label in [('develop.json', 'RSI · 开发回合'), ('propose.json', 'RSI · 候选生成'),
            ('selection.json', 'RSI · 原版筛选决定'), ('commit.json', 'RSI · 已提交版本'),
            ('retained.json', 'RSI · 原版保留决定')]:
            source(round_dir / name, label)
        ledger = root / 'tasks' / scene['task'] / 'continuation/failed-round-ledger' / (round_dir.name + '.json')
        source(ledger, '进化失败记录 · 使用已提交父策略')
        value = read(ledger)
        for path, error in value.get('original_errors', {}).items():
            evidence.append(dict(count=0, stage='原版进化失败', line=0,
                                 planning_summary=error))
    log = execution / 'episode/native/processes/simulator/process.log'
    source(log, '本场原生仿真日志')
    source(execution / 'episode/native/run/observed.json', '本场原生观测与终止结果')
    for camera, url in scene['cameras'].items():
        files.append(dict(label='原始正式视频 · ' + camera, url=url))
    diagnostics = receipt.get('diagnostics', {})
    detail = {k:v for k,v in scene.items() if not k.startswith('_')}
    detail.update(instruction=instruction, total_steps=diagnostics.get('action_count', 0),
        steps=[], evidence=evidence, events=[], messages=messages, calls=[], files=files,
        error='正式执行中断，等待核对原始记录。' if scene['_native_failure'] else None,
        official=read(folder / 'score.json') if score else {},
        trajectory_source='原生正式评测视频与终止回执 · 未保存逐步关节目标',
        rationale='\n'.join(m['text'] for m in messages), native_run=Path(remote).name)
    return detail
