"""Both evolution drivers return failure rethinking to their next proposal."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'PhysicalRSI'))
sys.path.insert(0, str(ROOT / 'runtime/physicalrsi-evolution'))

import run_evolution as evolution
import run_code_policy_fallback as fallback


def test_self_harness_development_handoff_and_no_validation_feedback(tmp_path, monkeypatch):
    candidate = tmp_path / 'candidate'
    candidate.mkdir()
    (candidate / 'policy.py').write_text('def policy(robot, memory):\n    return None\n')
    (candidate / 'memory_rules.json').write_text('{}')
    directory = tmp_path / 'development'
    execution = directory / 'execution/episode-001'
    execution.mkdir(parents=True)
    (execution / 'events.jsonl').write_text(json.dumps({'instruction': 'Native goal'}) + '\n')
    (directory / 'evaluation-plan.json').write_text(json.dumps({'run_id': 'native-run'}))
    result = {'eval_time': 2, 'details': {'0': {'success': True}, '1': {'success': False}}}
    data = {'result': result, 'output': directory,
            'feedback': {'task': 'align_blocks', 'split': 'development'}}
    calls = []
    class Rethinker:
        def __init__(self, model):
            pass
        def reflect(self, **kwargs):
            calls.append(kwargs)
            return {'episode': kwargs['episode'], 'diagnosis': {'failure_reason': 'Visible failure'},
                    'rethinking': {'memory_delta': {'lesson': 'Reobserve'}}}
    monkeypatch.setattr(evolution, 'VideoRethinker', Rethinker)
    proposer = evolution.Proposal.__new__(evolution.Proposal)
    proposer.model = object()
    proposer.evaluator = SimpleNamespace(development=lambda parent, output: data)
    developed = proposer.develop({'root': str(candidate)}, tmp_path / 'round')
    assert len(calls) == 1 and calls[0]['episode'] == '1'
    assert calls[0]['instruction'] == 'Native goal'
    assert developed['model_feedback']['failure_rethinking'][0]['rethinking']['memory_delta']
    assert json.loads((directory / 'model-development-feedback.json').read_text()) == developed['model_feedback']
    proposer.rethink({'root': str(candidate)}, data)
    assert len(calls) == 1


@pytest.mark.parametrize('worker_status, expected_revisions', [('returned', 2), ('episode_terminal', 2), ('error', 1)])
def test_fallback_next_revision_consumes_rethinking(tmp_path, monkeypatch, worker_status, expected_revisions):
    output = tmp_path / 'experiment'
    monkeypatch.setattr(sys, 'argv', ['run_code_policy_fallback.py', '--rounds', '2', '--output', str(output)])
    state = {'revision': 0, 'boundary': 0}
    observation = {'task': 'align_blocks', 'instruction': 'Native goal', 'initial_robot_state': {'joint': 0},
                   'observation_id': 'fresh', 'action_count': 30, 'action_limit': 200}
    def request(path, payload=None, **kwargs):
        if path == '/health':
            return {'status': 'ready'}
        if path == '/v1/sessions':
            state['revision'] += 1
            return {'run_id': f"run-{state['revision']}"}
        if path == '/v1/sessions/policy-chunks':
            return {'chunk_id': 'chunk'}
        if path == '/v1/sessions/action-chunks/execute':
            return {'executed': 10, 'outcome': 'executed', 'episode_end': False}
        if path == '/v1/sessions/observation':
            return observation
        raise AssertionError(path)
    def planner(*args, **kwargs):
        state['boundary'] += 1
        if state['boundary'] > 2:
            return {'mode': 'code_policy', 'planning_summary': 'Repeated visible deviation'}
        return {'mode': 'policy', 'subtask_goal': 'Native goal',
                'optimized_task_prompt': f"Revised prompt {state['boundary']}"}
    proposals = []
    def propose(dest, obs, history, source, memory, attempts):
        if state['revision'] == 2:
            assert attempts[0]['failure_rethinking'][0]['diagnosis']['failure_reason'] == 'Visible failed push'
            assert attempts[0]['failure_rethinking'][0]['video_evidence']
        proposals.append(state['revision'])
        return {'source': 'def policy(robot, memory):\n    return None\n', 'memory': {}}
    def final(run, reason):
        success = state['revision'] == 2
        return {'task_succeeded': success, 'videos': ['official.mp4'],
                'run': {'run_id': run['run_id'], 'status': 'completed',
                        'result': {'score': 100 if success else 0, 'eval_time': 1,
                                   'details': {'0': {'success': success}}}}}
    monkeypatch.setattr(fallback.control, 'request', request)
    monkeypatch.setattr(fallback.control, 'wait_gpu', lambda: None)
    monkeypatch.setattr(fallback.control, 'wait_ready', lambda run: observation)
    monkeypatch.setattr(fallback.control, 'planner', planner)
    monkeypatch.setattr(fallback, 'propose', propose)
    monkeypatch.setattr(fallback, 'run_worker', lambda *args: {'status': worker_status})
    monkeypatch.setattr(fallback, 'close_run', final)
    monkeypatch.setattr(fallback, 'rethink_failed_run', lambda *args: [
        {'episode': '0', 'diagnosis': {'failure_reason': 'Visible failed push'},
         'video_evidence': [{'sha256': 'video-hash'}], 'rethinking': {'memory_delta': {'lesson': 'Reobserve'}}}])
    fallback.main()
    assert proposals == list(range(1, expected_revisions + 1))
    results = json.loads((output / 'results.json').read_text())
    assert len(results) == expected_revisions
    if expected_revisions == 2:
        assert results[0]['rethinking_feedback']
        assert (output / 'selected-policy.py').exists()
    else:
        assert not (output / 'selected-policy.py').exists()
