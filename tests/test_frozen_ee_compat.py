"""CPU checks against the actual interactive boundary, without Isaac startup."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "/home/ymz/robodojo-pi05/RoboDojo")
from src.eval_client import interactive_harness as harness_module

spec = importlib.util.spec_from_file_location(
    "frozen_ee_compat", ROOT / "runtime/physicalrsi-all-skills/adapter/ee_compat.py")
compat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compat)


class FakeTask:
    robot_action_dim_info = {"arm_dim": [6, 6], "ee_dim": [1, 1]}

    def __init__(self):
        self.validated_types = []

    def get_action_type(self, action):
        types = set()
        if any(key.endswith("arm_joint_state") for key in action):
            types.add("joint")
        if any(key in compat._POSE_KEYS for key in action):
            types.add("ee")
        if len(types) != 1:
            raise ValueError("Cannot infer a single action type")
        return types.pop()

    def validate_action_dict(self, action):
        # Match the real native validator's allowed fields and dimensions.
        expected = {}
        for side in ("left", "right"):
            expected.update({f"{side}_arm_joint_state": 6,
                             f"{side}_ee_joint_state": 1,
                             f"{side}_ee_pose": 7, f"{side}_tcp_pose": 7,
                             f"{side}_delta_ee_pose": 7})
        if set(action) - set(expected):
            raise ValueError("Unexpected state keys")
        for key, value in action.items():
            if not isinstance(value, (np.ndarray, list, tuple)):
                raise TypeError("must be array-like")
            arr = np.asarray(value)
            if arr.ndim != 1 or arr.size != expected[key]:
                raise ValueError("dim mismatch")
        self.validated_types.append(self.get_action_type(action))


@pytest.fixture
def boundary():
    original_spec = harness_module._action_spec
    original_validation = harness_module.InteractiveEpisodeHarness._validate_action_chunk
    assert not hasattr(harness_module, compat._SAVED)
    harness = harness_module.InteractiveEpisodeHarness.__new__(
        harness_module.InteractiveEpisodeHarness)
    harness.task_env = FakeTask()
    try:
        yield harness
    finally:
        compat.install("joint")
        assert harness_module._action_spec is original_spec
        assert harness_module.InteractiveEpisodeHarness._validate_action_chunk is original_validation


def ee_action():
    return {"left_ee_pose": [0, 0, 0, 1, 0, 0, 0], "left_ee_joint_state": [0.5]}


def joint_action():
    return {"left_arm_joint_state": [0] * 6, "left_ee_joint_state": [0.5]}


def test_ee_action_passes_unchanged_and_native_validator_runs(boundary):
    compat.install("ee")
    action = ee_action()
    result = boundary._validate_action_chunk([action], max_actions=1)
    assert result[0] is action
    assert boundary.task_env.validated_types == ["ee"]


def test_ee_rejects_joint_and_mixed_chunks(boundary):
    compat.install("ee")
    with pytest.raises(ValueError, match="only accepts ee"):
        boundary._validate_action_chunk([joint_action()], max_actions=1)
    with pytest.raises(ValueError, match="only accepts ee"):
        boundary._validate_action_chunk([ee_action(), joint_action()], max_actions=2)
    with pytest.raises(ValueError, match="single action type"):
        boundary._validate_action_chunk([{**ee_action(), **joint_action()}], max_actions=1)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_ee_rejects_nonfinite(boundary, value):
    compat.install("ee")
    action = ee_action()
    action["left_ee_pose"][0] = value
    with pytest.raises(ValueError, match="NaN or infinity"):
        boundary._validate_action_chunk([action], max_actions=1)


def test_native_dimensions_and_chunk_limits_preserved(boundary):
    compat.install("ee")
    with pytest.raises(ValueError, match="dim mismatch"):
        boundary._validate_action_chunk([{"left_ee_pose": [0] * 6}], max_actions=1)
    with pytest.raises(ValueError, match="empty or invalid"):
        boundary._validate_action_chunk([], max_actions=1)
    with pytest.raises(ValueError, match="action limit"):
        boundary._validate_action_chunk([ee_action()] * 2, max_actions=1)
    with pytest.raises(ValueError, match="pose/gripper"):
        boundary._validate_action_chunk([{**ee_action(), "left_tcp_pose": [0] * 7}], max_actions=1)


def test_action_spec_uses_public_proprioception_only(boundary):
    compat.install("ee")
    observation = {"state": {**ee_action(), "right_ee_pose": [0] * 7,
                             "right_ee_joint_state": [0.5], "object_pose": [99] * 7},
                   "reward": 1, "success": True, "object_state": {"secret": 99}}
    contract = harness_module._action_spec(boundary.task_env, observation)
    assert contract["action_type"] == "absolute_ee"
    assert contract["required_fields"] == {}
    assert contract["requires_one_arm_pose"] is True
    assert set(contract["supported_fields"]) == {
        "left_ee_pose", "left_ee_joint_state", "right_ee_pose", "right_ee_joint_state"}
    assert contract["supported_fields"]["left_ee_pose"] == {"dimension": 7, "finite": True}
    assert "99" not in str(contract)


def test_joint_default_is_unchanged_and_restores_ee(boundary):
    original = harness_module.InteractiveEpisodeHarness._validate_action_chunk
    compat.install("joint")
    assert harness_module.InteractiveEpisodeHarness._validate_action_chunk is original
    assert boundary._validate_action_chunk([joint_action()], max_actions=1)
    with pytest.raises(ValueError, match="only accepts joint"):
        boundary._validate_action_chunk([ee_action()], max_actions=1)
    compat.install("ee")
    compat.install("ee")
    compat.install("joint")
    assert harness_module.InteractiveEpisodeHarness._validate_action_chunk is original
