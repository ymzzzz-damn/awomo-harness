"""Official native RSI results stay distinct from development and partial copies."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from trajectory_ui import EvidenceStore
from sync_rsi_native_open import mapped


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    remote = '/opt/robodojo/eval_result/native-run'
    mirror = root / 'native-run'
    view = root / 'stable-rsi-group'
    save(mirror / 'experiment.json', dict(schema='physicalrsi.native-open-results/v1'))
    save(view / 'experiment.json', dict(schema='physicalrsi.native-open-hybrid-view/v1',
        artifact_root=str(mirror), remote_output=remote, title='RSI hybrid'))
    cases = [dict(task='solve_equation', seed=0, layout_id=i, layout_sha256=str(i)) for i in range(3)]
    save(mirror / 'official-inventory.json', cases)
    lane = mirror / 'tasks/solve_equation'
    save(lane / 'progress.json', dict(state='evolving', scene=cases[1]))
    save(mirror / 'continuations/approved-v1/queue-status.json', dict(state='running', failed_tasks={}))
    scene = lane / 'official/seed-0-layout-00'
    videos = []
    for camera in ('cam_head', 'cam_left_wrist', 'cam_right_wrist'):
        p = scene / 'episode/videos' / (camera + '.mp4')
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b'video')
        videos.append(remote + '/' + str(p.relative_to(mirror)))
    score = dict(task='solve_equation', seed=0, layout_id=0, layout_sha256='0',
        selected_expert=dict(name='pi05', revision='frozen'), original_evolution_completed=False,
        native_result=dict(state='completed', natural_terminal=True, success=False,
            episode_score=0.0, evidence={v:'hash' for v in videos}))
    save(scene / 'score.json', score)
    save(scene / 'episode/episodes/0.json', dict(diagnostics=dict(action_count=300)))
    save(scene / 'cohort/cases.json', [dict(task='solve_equation')])
    save(scene / 'episode/model/agent-plans/000001.json', dict(
        instruction='saved official instruction', plan=dict(subgoal='saved original execution plan')))
    # A successful internal trial must never enter the official denominator.
    save(lane / 'workspace/rounds/round-000000/development/score.json', dict(success=True))
    return mirror, scene, score


def test_native_view_requires_verified_copy_and_keeps_ids_and_raw_scores(tmp_path):
    mirror, scene, score = fixture(tmp_path)
    store = EvidenceStore(tmp_path)
    before = (scene / 'score.json').read_bytes()
    first = store.scan()['scenes']
    assert len(first) == 3
    assert {s['layout']:s['status'] for s in first} == {0:'pending', 1:'running', 2:'pending'}
    key = next(s['id'] for s in first if s['layout'] == 0)
    relative = str((scene / 'score.json').relative_to(mirror))
    save(mirror / 'ui-sync.json', dict(verified_scores={relative:dict(
        sha256=hashlib.sha256(before).hexdigest(), native_evidence_verified=True, camera_count=3)}))
    second = store.scan(force=True)['scenes']
    completed = next(s for s in second if s['layout'] == 0)
    assert completed['id'] == key and completed['status'] == 'failure'
    assert len(completed['cameras']) == 3
    assert all(s['experiment'] == 'stable-rsi-group' for s in second)
    detail = store.detail(key)
    assert detail['total_steps'] == 300 and detail['instruction'] == 'saved official instruction'
    assert detail['external_parent_fallback'] and detail['rsi_native_hybrid']
    assert detail['messages'] == [dict(role='assistant', text='saved original execution plan')]
    assert len([f for f in detail['files'] if f['label'].startswith('原始正式视频')]) == 3
    assert (scene / 'score.json').read_bytes() == before


def test_modified_score_or_nonterminal_result_is_never_counted(tmp_path):
    mirror, scene, score = fixture(tmp_path)
    relative = str((scene / 'score.json').relative_to(mirror))
    save(mirror / 'ui-sync.json', dict(verified_scores={relative:dict(
        sha256='wrong', native_evidence_verified=True)}))
    store = EvidenceStore(tmp_path)
    assert not any(s['status'] in {'success','failure'} for s in store.scan()['scenes'])
    score['native_result']['natural_terminal'] = False
    save(scene / 'score.json', score)
    save(mirror / 'ui-sync.json', dict(verified_scores={relative:dict(
        sha256=hashlib.sha256((scene / 'score.json').read_bytes()).hexdigest(), native_evidence_verified=True)}))
    assert not any(s['status'] in {'success','failure'} for s in store.scan(force=True)['scenes'])


def test_mirror_paths_cannot_read_another_run(tmp_path):
    assert mapped('/run/official/score.json', '/run', tmp_path) == tmp_path / 'official/score.json'
    for value in ('/other-run/score.json', '/run/../secret.json'):
        try:
            mapped(value, '/run', tmp_path)
        except ValueError:
            pass
        else:
            raise AssertionError('An external evidence path was accepted')


def test_recovered_official_scene_uses_new_receipt_and_preserves_original_attempt(tmp_path):
    mirror, scene, score = fixture(tmp_path)
    remote = '/opt/robodojo/eval_result/native-run'
    attempt = scene / 'recovery-attempts/preaction-000001'
    proof = mirror / 'continuations/official-preaction-approved-v1/proofs/zero-action.json'
    save(proof, dict(passed=True, zero_environment_actions=True))
    save(attempt / 'episode/episodes/0.json', dict(diagnostics=dict(action_count=184)))
    save(attempt / 'cohort/cases.json', [dict(instruction='recovered official instruction')])
    save(attempt / 'episode/model/agent-plans/000001.json', dict(
        instruction='recovered official instruction', plan=dict(subgoal='recovered original plan')))
    score['official_preaction_recovery'] = dict(
        attempt=remote + '/' + str(attempt.relative_to(mirror)),
        proof=remote + '/' + str(proof.relative_to(mirror)))
    score['native_result']['diagnostics'] = dict(action_count=184)
    save(scene / 'score.json', score)
    save(attempt / 'score.json', score)
    relative = str((scene / 'score.json').relative_to(mirror))
    save(mirror / 'ui-sync.json', dict(verified_scores={relative:dict(
        sha256=hashlib.sha256((scene / 'score.json').read_bytes()).hexdigest(),
        native_evidence_verified=True, camera_count=3)}))
    store = EvidenceStore(tmp_path)
    completed = next(s for s in store.scan()['scenes'] if s['layout'] == 0)
    detail = store.detail(completed['id'])
    assert completed['actions'] == detail['total_steps'] == 184
    assert detail['instruction'] == 'recovered official instruction'
    assert detail['messages'] == [dict(role='assistant', text='recovered original plan')]
    assert any(f['label'] == '正式重试前的零动作与进程清理证明' for f in detail['files'])
    assert json.loads((scene / 'episode/episodes/0.json').read_text())['diagnostics']['action_count'] == 300


def test_failed_initialization_retry_remains_unscored_and_links_both_attempts(tmp_path):
    mirror, scene, _ = fixture(tmp_path)
    remote = '/opt/robodojo/eval_result/native-run'
    attempt = scene / 'recovery-attempts/initialization-000001'
    proof = mirror / 'continuations/official-preaction-approved-v4/proofs/init.json'
    save(proof, dict(passed=True, zero_environment_actions=True))
    save(attempt / 'initialization-retry-intent.json', dict(
        proof=remote + '/' + str(proof.relative_to(mirror)), max_initialization_retries=1))
    save(attempt / 'initialization-retry-failure.json', dict(error='original instability'))
    original = scene / 'cohort/initialization/solve_equation/0/processes/simulator/process.log'
    original.parent.mkdir(parents=True)
    original.write_text('Unstable envs: [0]\n')
    save(mirror / 'tasks/solve_equation/progress.json', dict(state='official_episode',
        scene=dict(seed=0, layout_id=0), official_initialization_recovery=True,
        official_preaction_recovery=remote + '/' + str(attempt.relative_to(mirror))))
    save(mirror / 'continuations/official-preaction-approved-v4/queue-status.json', dict(
        state='running', failed_tasks=dict(solve_equation=dict(returncode=1))))
    store = EvidenceStore(tmp_path)
    current = next(s for s in store.scan()['scenes'] if s['layout'] == 0)
    assert current['status'] == 'error' and current['score'] is None
    assert '同布局' in current['qualification'] and '未评分' in current['qualification']
    assert '接口异常' not in current['qualification']
    files = store.detail(current['id'])['files']
    assert any(f['label'] == '同布局初始化重试失败原始记录' for f in files)
    assert any(f['label'] == '保留的原始初始化失败日志' for f in files)
    assert original.read_text() == 'Unstable envs: [0]\n'


def test_initialization_metadata_mirror_excludes_observation_arrays(tmp_path, monkeypatch):
    import subprocess
    import sync_rsi_native_open as mirror

    class Captured(Exception):
        pass

    filters = []

    def capture(*args, **kwargs):
        filters.extend(kwargs['filters'])
        raise Captured()

    monkeypatch.setattr(mirror, 'rsync', capture)
    source, target = tmp_path / 'source', tmp_path / 'target'
    target.mkdir()
    for tail in ('cohort/initialization/solve_equation/0',
                 'recovery-attempts/initialization-000001/cohort/initialization/solve_equation/0'):
        folder = source / 'tasks/solve_equation/official/seed-0-layout-00' / tail
        (folder / 'processes/simulator').mkdir(parents=True)
        (folder / 'run').mkdir()
        (folder / 'request.json').write_text('{}')
        (folder / 'processes/simulator/process.log').write_text('Unstable envs: [0]')
        (folder / 'run/observation.npy').write_bytes(b'observation array')
    try:
        mirror.sync({'remote_output': '/run/RSI-hybrid-emptytasks-test'}, target, None, {})
    except Captured:
        pass
    command = ['rsync', '-rni', '--out-format=%n']
    for rule in filters:
        command += ['--filter', rule]
    result = subprocess.run(command + [str(source) + '/', str(target) + '/'],
                            capture_output=True, text=True, check=True)
    paths = result.stdout.splitlines()
    assert sum(path.endswith('request.json') for path in paths) == 2
    assert sum(path.endswith('process.log') for path in paths) == 2
    assert not any(path.endswith('.npy') for path in paths)


def test_user_pause_preserves_scores_and_stops_showing_active_scenes(tmp_path):
    mirror, scene, _ = fixture(tmp_path)
    before = (scene / 'score.json').read_bytes()
    store = EvidenceStore(tmp_path)
    initial = store.scan()['scenes']
    completed = {s['id']: s['score'] for s in initial if s['status'] in {'success', 'failure'}}
    assert any(s['status'] == 'running' for s in initial)
    save(mirror / 'user-pause.json', dict(paused=True, reason='user_requested_stop'))
    current = store.scan(force=True)['scenes']
    assert not any(s['status'] == 'running' for s in current)
    assert {s['id']: s['score'] for s in current if s['status'] in {'success', 'failure'}} == completed
    assert all(s['score'] is None and '用户已暂停' in s['qualification']
               for s in current if s['status'] == 'pending')
    assert (scene / 'score.json').read_bytes() == before
