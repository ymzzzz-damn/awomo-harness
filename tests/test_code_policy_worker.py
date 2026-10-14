"""The code worker must use the existing action boundary and episode identity."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest


RUNTIME = Path(__file__).resolve().parents[1] / "runtime/awomopaw-code-policy"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("body", ["import os", "robot.scene()", "robot.e", "getattr(robot, 'e')", "other = robot; other.hold(1)"])
def test_undocumented_capabilities_are_rejected(body):
    contract = load("test_code_contract", RUNTIME / "contract.py")
    with pytest.raises(ValueError):
        contract.validate_source("def policy(robot, memory):\n    " + body + "\n")


def test_nested_policy_helper_can_update_its_local_closure():
    contract = load("test_code_contract", RUNTIME / "contract.py")
    source = "def policy(robot, memory):\n    count = 0\n    def step():\n        nonlocal count\n        count += 1\n    step()\n    return count\n"
    contract.validate_source(source)
    namespace = {}
    exec(source, namespace)
    assert namespace["policy"](None, {}) == 1


def test_code_actions_use_fresh_observation_and_preserve_episode_origin(tmp_path, monkeypatch):
    contract = load("test_code_contract", RUNTIME / "contract.py")
    fake_bridge = ModuleType("bridge")
    fake_bridge.PrimitiveClient = object
    fake_bridge.Robot = object
    fake_bridge.Sam3TextSegmenter = object
    fake_bridge.plain = lambda x: x
    monkeypatch.setitem(sys.modules, "awomopaw_primitives", fake_bridge)
    monkeypatch.setitem(sys.modules, "contract", contract)
    monkeypatch.setitem(sys.modules, "cv2", ModuleType("cv2"))
    worker = load("test_http_code_worker", RUNTIME / "worker.py")
    source = tmp_path / "policy.py"
    source.write_text("def policy(robot, memory):\n    return None\n")
    count, calls = [30], []
    def request(path, payload):
        calls.append((path, payload))
        if path == "/v1/sessions/observation":
            return {"run_id": "run1", "instruction": "Official goal", "observation_id": str(count[0]),
                    "robot_state": {"left_arm_joint_state": [2.0] * 6, "left_ee_joint_state": [1.0]},
                    "action_count": count[0], "action_limit": 200}
        if path == "/v1/sessions/direct-action-chunks":
            assert payload["observation_id"] == str(count[0])
            assert payload["subtask_goal"] == "Official goal"
            return {"chunk_id": "chunk" + str(count[0])}
        if path == "/v1/sessions/action-chunks/execute":
            assert payload["max_actions"] == 1 and payload["discard_remaining"] is True
            count[0] += 1
            return {"executed": 1, "episode_end": False}
        raise AssertionError(path)
    origin = {"left_arm_joint_state": [0.0] * 6, "left_ee_joint_state": [1.0]}
    config = {"output": str(tmp_path / "execution"), "run_id": "run1", "instruction": "Official goal",
              "initial_state": origin, "source": str(source), "fallback_reason": "Observed VLA deviation"}
    episode = worker.HttpEpisode(config, request=request)
    assert episode.initial == origin
    assert episode.obs["state"]["left_arm_joint_state"] != origin["left_arm_joint_state"]
    action = {"left_arm_joint_state": [1.0] * 6, "left_ee_joint_state": [1.0]}
    episode.emit(action)
    episode.emit(action)
    assert count[0] == 32
    assert [p["observation_id"] for path,p in calls if path.endswith("direct-action-chunks")] == ["30", "31"]
    config["run_id"] = "other-run"
    with pytest.raises(ValueError, match="cross"):
        worker.HttpEpisode(config, request=request)
