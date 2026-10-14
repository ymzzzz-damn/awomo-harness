import copy
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from chunk_supervisor import CAMERAS, ChunkSupervisor
from supervised_bridge import SupervisedManager, supervised_app
from qwenpaw.app.embodiments.robodojo_adapter import AdapterConfigurationError, RoboDojoRunManager


def observation(index=0):
    return dict(run_id='20261013T000000Z-12345678', observation_id=f'observation-{index}',
                task='align_blocks', instruction='Exact native instruction', episode_end=False,
                action_count=index, action_limit=200, robot_state={}, action_spec={})


def executed_gate():
    gate = ChunkSupervisor()
    gate.require('policy', observation())
    gate.ready(dict(chunk_id='provider-chunk', source='policy'))
    gate.executed(dict(executed=10, outcome='window_complete'), observation(1))
    return gate


def review(gate, verdict='continue', index=1):
    for camera in CAMERAS:
        gate.frame(observation(index), camera, b'public RGB')
    return gate.review(dict(observation_id=f'observation-{index}', chunk_id='provider-chunk',
                            verdict=verdict, reflection='Visible comparison with the prior frames'), observation(index))


def test_every_window_requires_reflection_and_all_three_current_frames():
    gate = executed_gate()
    with pytest.raises(ValueError, match='Submit reflection'):
        gate.require('policy', observation(1))
    gate.frame(observation(0), 'cam_head', b'old frame')
    with pytest.raises(ValueError, match='all three'):
        gate.review(dict(observation_id='observation-1', chunk_id='provider-chunk',
                         verdict='continue', reflection='Progress'), observation(1))
    value = review(gate)
    assert set(value['frames']) == set(CAMERAS)
    gate.require('policy', observation(1))
    with pytest.raises(ValueError, match='verdict=correct'):
        gate.require('gpt', observation(1))


def test_first_wrong_chunk_permits_immediate_takeover_and_blocks_vla():
    gate = executed_gate()
    review(gate, 'correct')
    gate.require('gpt', observation(1))
    with pytest.raises(ValueError, match='verdict=continue'):
        gate.require('policy', observation(1))
    gate.ready(dict(chunk_id='gpt-correction', source='llm_direct'))
    gate.executed(dict(executed=1), observation(2))
    for camera in CAMERAS:
        gate.frame(observation(2), camera, b'corrected frame')
    gate.review(dict(observation_id='observation-2', chunk_id='gpt-correction',
                     verdict='continue', reflection='Recovery observed'), observation(2))
    gate.require('policy', observation(2))


def test_uncertainty_stale_evidence_and_cross_run_state_cannot_authorize_actions():
    gate = executed_gate()
    review(gate, 'uncertain')
    for source in ('policy', 'gpt'):
        with pytest.raises(ValueError):
            gate.require(source, observation(1))
    with pytest.raises(ValueError, match='stale'):
        review(gate, 'continue', index=2)
    new = {**observation(), 'run_id': 'another-run'}
    gate.require('policy', new)
    assert gate.last is None
    with pytest.raises(ValueError, match='failure evidence'):
        gate.require('gpt', new)


def test_provider_failure_requires_takeover_and_cannot_be_approved_as_progress():
    gate = ChunkSupervisor()
    gate.failed('provider-chunk', 'timeout', observation(1))
    with pytest.raises(ValueError, match='Provider failure'):
        review(gate)
    review(gate, 'correct')
    gate.require('gpt', observation(1))


@pytest.fixture
def manager(tmp_path, monkeypatch):
    state = {'obs': observation(), 'calls': []}
    value = object.__new__(SupervisedManager)
    value._lock = threading.RLock()
    value.supervisor = ChunkSupervisor()
    value._code_processes = {}
    monkeypatch.setattr(RoboDojoRunManager, 'current_observation', lambda _: copy.deepcopy(state['obs']))
    monkeypatch.setattr(RoboDojoRunManager, 'current_observation_image', lambda _, topic: (b'RGB', 'image/jpeg'))
    monkeypatch.setattr(RoboDojoRunManager, 'request_policy_chunk', lambda _, payload:
                        dict(chunk_id='provider-chunk', source='policy'))
    monkeypatch.setattr(RoboDojoRunManager, 'submit_direct_action_chunk', lambda _, payload:
                        dict(chunk_id='gpt-correction', source='llm_direct'))
    monkeypatch.setattr(RoboDojoRunManager, 'get_run', lambda _, run_id: SimpleNamespace(artifact_dir=str(tmp_path)))

    def execute(_, chunk_id, max_actions, discard_remaining):
        state['calls'].append(chunk_id)
        state['obs'] = observation(len(state['calls']))
        return dict(executed=max_actions, observation_id=state['obs']['observation_id'], episode_end=False)
    monkeypatch.setattr(RoboDojoRunManager, 'execute_action_chunk', execute)
    return value, state


def test_http_reflection_and_exclusive_controller_routing(manager):
    from fastapi.testclient import TestClient
    value, state = manager
    client = TestClient(supervised_app(value))
    value.request_policy_chunk({})
    value.execute_action_chunk('provider-chunk', 10, True)
    for camera in CAMERAS:
        value.current_observation_image(camera)
    result = client.post('/v1/sessions/reflections', json=dict(
        observation_id='observation-1', chunk_id='provider-chunk', verdict='correct',
        reflection='Wrong object moved; immediately correct it'))
    assert result.status_code == 200
    with pytest.raises(ValueError, match='verdict=continue'):
        value.request_policy_chunk({})
    value.submit_direct_action_chunk({})
    value.execute_action_chunk('gpt-correction', 1, True)
    assert state['calls'] == ['provider-chunk', 'gpt-correction']


def test_terminal_receipt_is_not_lost_when_native_server_exits(manager, monkeypatch):
    value, state = manager
    value.request_policy_chunk({})
    monkeypatch.setattr(RoboDojoRunManager, 'execute_action_chunk', lambda *args:
                        dict(episode_end=True, observation_id='terminal-observation', executed=1))
    monkeypatch.setattr(RoboDojoRunManager, 'current_observation', lambda _:
                        (_ for _ in ()).throw(AdapterConfigurationError('native server exited')))
    result = value.execute_action_chunk('provider-chunk', 10, True)
    assert result['reflection_phase'] == 'post_episode_video'
    assert value.supervisor.last['observation_id'] == 'terminal-observation'


def test_bad_window_does_not_execute(manager):
    value, state = manager
    value.request_policy_chunk({})
    for maximum, discard in ((None, False), (11, True), (True, True), (10, False)):
        with pytest.raises(AdapterConfigurationError):
            value.execute_action_chunk('provider-chunk', maximum, discard)
    assert not state['calls']


def test_validation_errors_are_not_misreported_as_provider_failure(manager, monkeypatch):
    value, state = manager
    def fail(*args):
        raise AdapterConfigurationError('stale observation')
    monkeypatch.setattr(RoboDojoRunManager, 'request_policy_chunk', fail)
    with pytest.raises(AdapterConfigurationError):
        value.request_policy_chunk({})
    assert value.supervisor.last is None


def test_native_inference_failure_is_reflected_and_allows_code_takeover(manager, monkeypatch):
    value, state = manager
    state['obs']['last_failure'] = dict(observation_id='observation-0', code='provider_inference_failed')
    def fail(*args):
        raise AdapterConfigurationError('policy server returned an invalid chunk')
    monkeypatch.setattr(RoboDojoRunManager, 'request_policy_chunk', fail)
    monkeypatch.setattr(RoboDojoRunManager, 'start_code_policy', lambda _, payload: {'status': 'running'})
    with pytest.raises(AdapterConfigurationError):
        value.request_policy_chunk({})
    for camera in CAMERAS:
        value.current_observation_image(camera)
    value.reflect(dict(observation_id='observation-0', chunk_id=value.supervisor.last['chunk']['chunk_id'],
                       verdict='correct', reflection='Provider returned an invalid chunk; take over now'))
    assert value.start_code_policy({})['status'] == 'running'


def test_runner_reflects_first_failure_then_returns_to_vla(manager, monkeypatch, tmp_path):
    import run_supervised_task as runner
    value, state = manager
    decisions = iter([
        dict(verdict='continue', reflection='Initial subtask', mode='policy',
             subtask_goal='native goal', optimized_task_prompt='Exact native instruction'),
        dict(verdict='correct', reflection='The first window moved the wrong object', mode='direct',
             subtask_goal='Correct', actions=[{'left_arm_joint_state': [0.0] * 6}]),
        dict(verdict='continue', reflection='Correction recovered useful progress', mode='policy',
             subtask_goal='native goal', optimized_task_prompt='Exact native instruction'),
    ])
    class Model:
        def __init__(self, **kwargs):
            pass
        def complete(self, messages, tools):
            return json.dumps(next(decisions)), [], None
    monkeypatch.setattr(runner, 'LLM', Model)
    def request(path, payload=None):
        if path == '/health':
            return {'chunk_supervision': {'required': True}}
        if path == '/v1/sessions':
            return {'run_id': observation()['run_id']}
        if path == '/v1/sessions/code-policy-api':
            return {'methods': {}}
        if path == '/v1/sessions/observation':
            return value.current_observation()
        if path == '/v1/sessions/reflections':
            return value.reflect(payload)
        if path == '/v1/sessions/policy-chunks':
            return value.request_policy_chunk(payload)
        if path == '/v1/sessions/direct-action-chunks':
            return value.submit_direct_action_chunk(payload)
        if path == '/v1/sessions/action-chunks/execute':
            result = value.execute_action_chunk(**payload)
            if len(state['calls']) == 3:
                result['episode_end'] = True
            return result
        raise AssertionError(path)
    def capture(dest):
        dest.mkdir(parents=True)
        files = []
        for camera in CAMERAS:
            pixels, _ = value.current_observation_image(camera)
            path = dest / f'{camera}.jpg'
            path.write_bytes(pixels)
            files.append(path)
        return files
    monkeypatch.setattr(runner.control, 'request', request)
    monkeypatch.setattr(runner.control, 'wait_ready', lambda _: value.current_observation())
    monkeypatch.setattr(runner.control, 'official_result', lambda _:
                        {'run': {'status': 'completed'}, 'task_succeeded': True})
    monkeypatch.setattr(runner, 'capture', capture)
    args = SimpleNamespace(output=tmp_path / 'run', bridge='http://localhost', task_artifact=None,
        task='align_blocks', evolve=False, split='development', provider_config=None, model='fake',
        policy_id='pi05', seed=0, max_decisions=10, horizon=10)
    runner.run(args)
    assert state['calls'] == ['provider-chunk', 'gpt-correction', 'provider-chunk']
    reviews = json.loads((args.output / 'chunk-reflections.json').read_text())
    assert [r['verdict'] for r in reviews] == ['correct', 'continue']


def test_distilled_candidate_automatically_feeds_the_next_development_round(tmp_path, monkeypatch):
    import run_supervised_task as runner
    from PhysicalRSI_core.infra.storage import digest
    supplied_parents = []
    def one_round(args):
        supplied_parents.append(args.task_artifact)
        previous = json.loads(args.task_artifact.read_text()) if args.task_artifact else None
        content = dict(skill_text=f'Learned revision {len(supplied_parents)}', source='', memory={})
        candidate = dict(schema='awomopaw.task-artifact/v1', task='align_blocks', content=content,
                         revision=digest(content), parent_revision=previous['revision'] if previous else None)
        runner.control.save(args.output / 'evolution/task-artifact.json', candidate)
        return dict(run={'run_id': str(len(supplied_parents))}, task_succeeded=False)
    monkeypatch.setattr(runner, 'run', one_round)
    args = SimpleNamespace(rounds=3, evolve=True, split='development', task_artifact=None,
                           output=tmp_path / 'development')
    runner.run_rounds(args)
    assert supplied_parents == [None, args.output / 'round-000/evolution/task-artifact.json',
                               args.output / 'round-001/evolution/task-artifact.json']
    lineage = json.loads((args.output / 'evolution-lineage.json').read_text())
    assert lineage[1]['parent_revision'] == lineage[0]['candidate_revision']
    assert all(row['status'] == 'development_candidate' for row in lineage)
