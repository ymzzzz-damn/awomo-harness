"""CPU-only checks of the adapter, without loading a task strategy or simulator."""
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, '/home/ymz/robodojo-pi05/RoboDojo')
spec = importlib.util.spec_from_file_location('frozen_align_adapter',
    ROOT / 'runtime/physicalrsi-align-skill/adapter/model.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def test_public_observation_strips_privileged_inputs_and_keeps_rgb():
    color = object()
    obs = {'vision': {'cam_head': {'color': color, 'depth': 'private'}},
           'state': {'left_arm_joint_state': [0] * 6, 'object_pose': 'private'},
           'instruction': 'official goal', 'reward': 100, 'success': True,
           'object_state': 'private'}
    public = adapter.public_observation(obs)
    assert public == {'vision': {'cam_head': {'color': color}},
                      'state': {'left_arm_joint_state': [0] * 6},
                      'instruction': 'official goal'}
    assert public['vision']['cam_head']['color'] is color


def test_descriptor_tampering_rejected_before_loading(tmp_path):
    path = tmp_path / 'descriptor.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='descriptor changed'):
        adapter.load_frozen({'descriptor': str(path), 'descriptor_sha256': '0' * 64},
                            tmp_path / 'output', tmp_path)


def test_actions_pass_through_unchanged_and_trial_result_not_forwarded():
    original = [{'left_arm_joint_state': [1, 2, 3, 4, 5, 6]}]
    class Backend:
        def get_action(self):
            return original
        def on_trial_end(self, value):
            assert value is None
    model = adapter.Model.__new__(adapter.Model)
    model.backend = Backend()
    assert model.get_action() is original
    model.on_trial_end({'reward': 100, 'ground_truth': 'private'})
