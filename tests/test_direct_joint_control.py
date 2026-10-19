"""Exercise the real simulator control plane without loading Isaac or a policy."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'runtime/gpt6-harness-ablation-adapter/interactive_harness.py'
spec = importlib.util.spec_from_file_location('direct_harness_under_test', PATH)
native = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = native
spec.loader.exec_module(native)


def observation(step):
    return dict(instruction='Move according to the visible task',
        vision={'cam_head': {'color': np.zeros((8, 8, 3), dtype=np.uint8)}},
        state={'arm_joint_state': np.zeros(6), 'ee_joint_state': np.array([0.5])})


class Env:
    def __init__(self, root):
        self.save_dir = str(root)
        self.task_name, self.run_id = 'align_blocks', 'native-test'
        self.take_action_cnt, self.end_flag, self.success = [0], [False], [False]
        self.step_lim, self.executed = 3, []
        self.robot_action_dim_info = {'arm_dim': [6], 'ee_dim': [1]}

    def validate_action_dict(self, action):
        if set(action) != {'arm_joint_state', 'ee_joint_state'}:
            raise ValueError('Invalid fields')
        if len(action['arm_joint_state']) != 6 or len(action['ee_joint_state']) != 1:
            raise ValueError('Invalid dimensions')

    def get_action_type(self, action):
        return 'joint'

    def take_action(self, action):
        self.validate_action_dict(action)
        self.executed.append(action)
        self.take_action_cnt[0] += 1

    def is_episode_end(self):
        self.end_flag[0] = self.take_action_cnt[0] >= self.step_lim
        return self.end_flag[0]

    def get_obs(self):
        return observation(self.take_action_cnt[0])

    def get_obs_batch(self, **kwargs):
        return [self.get_obs()]


class NoPolicy:
    def call(self, *args, **kwargs):
        raise AssertionError('Direct control must never call a policy model')


def payload(harness, count=4):
    return dict(observation_id=harness.observation_metadata()['observation_id'],
        subtask_goal='A short visible motion', fallback_reason='GPT direct control',
        actions=[{'arm_joint_state': [0.01*i]*6, 'ee_joint_state': [0.5]} for i in range(count)])


def test_direct_segment_only_executes_on_request_and_stops_at_native_budget(tmp_path):
    env = Env(tmp_path)
    harness = native.InteractiveEpisodeHarness(env, NoPolicy(), host='127.0.0.1', port=0)
    harness._set_observation(env.get_obs())
    request = payload(harness)
    chunk = harness._handle_direct_action(request)
    assert env.executed == []
    result = harness._handle_execute(dict(chunk_id=chunk['chunk_id'], max_actions=40, discard_remaining=True))
    assert len(env.executed) == 3
    assert result['episode_end'] and result['executed'] == 3
    assert result['observation_id'] != request['observation_id']


@pytest.mark.parametrize('change', ['stale', 'nonfinite', 'too_long'])
def test_invalid_direct_segments_cannot_execute(tmp_path, change):
    env = Env(tmp_path)
    harness = native.InteractiveEpisodeHarness(env, NoPolicy(), host='127.0.0.1', port=0)
    harness._set_observation(env.get_obs())
    request = payload(harness, 41 if change == 'too_long' else 1)
    if change == 'stale': request['observation_id'] = 'old-observation'
    if change == 'nonfinite': request['actions'][0]['arm_joint_state'][0] = float('nan')
    with pytest.raises(ValueError): harness._handle_direct_action(request)
    assert env.executed == []


def test_partial_execution_discards_tail_and_requires_a_new_observation(tmp_path):
    env = Env(tmp_path)
    harness = native.InteractiveEpisodeHarness(env, NoPolicy(), host='127.0.0.1', port=0)
    harness._set_observation(env.get_obs())
    request = payload(harness, 3)
    chunk = harness._handle_direct_action(request)
    result = harness._handle_execute(dict(chunk_id=chunk['chunk_id'], max_actions=1, discard_remaining=True))
    assert len(env.executed) == 1 and result['pending_actions_remaining'] == 0
    assert result['discarded_actions'] == 2
    with pytest.raises(ValueError): harness._handle_execute(dict(chunk_id=chunk['chunk_id']))
    with pytest.raises(ValueError): harness._handle_direct_action(request)


def test_removed_execution_endpoints_are_unavailable():
    from qwenpaw.app.embodiments.robodojo_adapter import create_app, RoboDojoRunManager
    from fastapi.testclient import TestClient
    with TestClient(create_app(manager=object.__new__(RoboDojoRunManager))) as client:
        assert client.get('/v1/sessions/code-policy-api').status_code == 404
        assert client.post('/v1/sessions/code-policies', json={}).status_code == 404


class BimanualEnv(Env):
    def __init__(self, root):
        super().__init__(root)
        self.robot_action_dim_info = {'arm_dim': [6, 6], 'ee_dim': [1, 1]}

    def get_obs(self):
        obs = observation(0)
        obs['state'] = {f'{side}_{key}': value for side in ['left', 'right']
                        for key, value in obs['state'].items()}
        return obs

    def validate_action_dict(self, action):
        # Match RoboDojo's native validator: it checks only supplied keys.
        for key, value in action.items():
            assert len(value) == (6 if key.endswith('arm_joint_state') else 1)


@pytest.mark.parametrize('invalid', ['missing_arm', 'missing_gripper', 'gripper_range'])
def test_incomplete_bimanual_chunk_is_rejected_before_any_native_step(tmp_path, invalid):
    env = BimanualEnv(tmp_path)
    harness = native.InteractiveEpisodeHarness(env, NoPolicy(), host='127.0.0.1', port=0)
    harness._set_observation(env.get_obs())
    action = {key: list(value) for key, value in env.get_obs()['state'].items()}
    bad = {key: list(value) for key, value in action.items()}
    if invalid == 'missing_arm': del bad['right_arm_joint_state']
    elif invalid == 'missing_gripper': del bad['right_ee_joint_state']
    else: bad['left_ee_joint_state'] = [1.1]
    request = dict(observation_id=harness.observation_metadata()['observation_id'],
        subtask_goal='Complete dual-arm targets', fallback_reason='GPT control', actions=[action, bad])
    with pytest.raises(ValueError, match='fields mismatch|normalized range'):
        harness._handle_direct_action(request)
    assert env.take_action_cnt[0] == 0 and not env.executed
    assert harness.public_status()['pending_chunk_id'] is None
    request['actions'] = [action]
    chunk = harness._handle_direct_action(request)
    receipt = harness._handle_execute(dict(chunk_id=chunk['chunk_id'], max_actions=1, discard_remaining=True))
    assert receipt['executed'] == 1


def test_native_execution_failure_releases_chunk_with_explicit_receipt(tmp_path):
    env = Env(tmp_path)
    take_action = env.take_action
    def fail_once(action):
        take_action(action)
        env.take_action = take_action
        raise RuntimeError('native controller failed after a partial action')
    env.take_action = fail_once
    harness = native.InteractiveEpisodeHarness(env, NoPolicy(), host='127.0.0.1', port=0)
    harness._set_observation(env.get_obs())
    request = payload(harness, 2)
    chunk = harness._handle_direct_action(request)
    result = harness._handle_execute(dict(chunk_id=chunk['chunk_id'], max_actions=2, discard_remaining=True))
    assert result['outcome'] == 'execution_failed' and result['partial_execution_unknown']
    assert result['action_count'] == 1 and result['pending_actions_remaining'] == 0
    assert result['observation_id'] != request['observation_id']
    assert harness.public_status()['pending_chunk_id'] is None
    assert harness.public_status()['pending_actions_remaining'] == 0
    next_chunk = harness._handle_direct_action(payload(harness, 1))
    result = harness._handle_execute(dict(chunk_id=next_chunk['chunk_id'], max_actions=1, discard_remaining=True))
    assert result['executed'] == 1 and result['action_count'] == 2
