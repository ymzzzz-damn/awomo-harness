"""Offline tests for the awomopaw RoboDojo/Pi0.5 process adapter."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import time

import pytest

from qwenpaw.app.embodiments.robodojo_adapter import (
    EvaluationConflictError,
    EvaluationRun,
    PolicyProvider,
    RoboDojoAdapterConfig,
    RoboDojoRunManager,
    create_app,
)


def _fake_install(tmp_path: Path, *, sleep_seconds: float = 0.0) -> RoboDojoAdapterConfig:
    root = tmp_path / "RoboDojo"
    policy = root / "XPolicyLab" / "policy" / "Pi_05"
    harness_policy = root / "XPolicyLab" / "policy" / "Awomopaw_Harness"
    (root / "scripts").mkdir(parents=True)
    (root / "task" / "RoboDojo" / "tasks").mkdir(parents=True)
    (root / "Assets").mkdir()
    (root / "Assets" / "test-asset.marker").write_text(
        "fixture\n", encoding="utf-8"
    )
    for layout_seed in (0, 1):
        (
            root
            / "Assets"
            / "Eval_Layout"
            / "RoboDojo"
            / "arx_x5"
            / str(layout_seed)
        ).mkdir(parents=True)
    (policy / "checkpoints" / "RoboDojo-sim-arx_x5-joint-0").mkdir(parents=True)
    (policy / "model.py").write_text("class Model: pass\n", encoding="utf-8")
    (policy / "openpi" / ".venv" / "bin").mkdir(parents=True)
    (policy / "openpi" / ".venv" / "bin" / "python").write_text(
        "#!/usr/bin/env python\n", encoding="utf-8"
    )
    harness_policy.mkdir(parents=True)
    (harness_policy / "eval.sh").write_text(
        "#!/usr/bin/env bash\n", encoding="utf-8"
    )
    (root / "task" / "RoboDojo" / "tasks" / "align_blocks.py").write_text(
        "# test task\n", encoding="utf-8"
    )
    launcher = root / "scripts" / "robodojo.sh"
    launcher.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
task=''
seed=''
while (($#)); do
  case "$1" in
    --task) task="$2"; shift 2 ;;
    --seed) seed="$2"; shift 2 ;;
    *) shift ;;
  esac
done
sleep {sleep_seconds}
out="$PWD/eval_result/RoboDojo/$task/Pi_05/arx_x5/${{seed}}_ckpt_name=RoboDojo-sim-arx_x5-joint-0,action_type=joint/$ROBODOJO_RUN_ID"
mkdir -p "$out"
printf '{{"success_rate":0.0,"eval_time":2,"score":0.0,"details":{{}}}}\n' > "$out/_result.json"
printf 'video' > "$out/episode_0000000_cam_head_fail.mp4"
""",
        encoding="utf-8",
    )
    return RoboDojoAdapterConfig(
        robodojo_root=root,
        policy_dir=harness_policy,
        providers=(
            PolicyProvider(
                provider_id="pi05",
                kind="vla",
                policy_name="Pi_05",
                checkpoint="RoboDojo-sim-arx_x5-joint-0",
                policy_uv_env_path=policy / "openpi",
                revision="test-pi05-revision",
            ),
        ),
    )


def _wait_for_completion(manager: RoboDojoRunManager, run_id: str) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if manager.get_run(run_id).status not in {"starting", "running"}:
            return
        time.sleep(0.01)
    raise AssertionError("fake evaluation did not finish")


def test_adapter_runs_official_launcher_and_persists_result(tmp_path: Path) -> None:
    manager = RoboDojoRunManager(_fake_install(tmp_path))

    assert manager.health()["status"] == "ready"
    assert manager.health()["planning_handoff"] == (
        "llm_plan_to_registered_policy_to_embodiment_action_chunk; "
        "on_provider_failure_llm_direct_to_same_embodiment_executor"
    )
    run = manager.start_evaluation("align_blocks", eval_num=2, seed=0)
    _wait_for_completion(manager, run.run_id)

    finished = manager.get_run(run.run_id)
    assert finished.status == "completed"
    assert finished.result == {
        "success_rate": 0.0,
        "eval_time": 2,
        "score": 0.0,
        "details": {},
    }
    result = manager.get_result(run.run_id)
    assert len(result["videos"]) == 1
    receipt = manager.config.state_dir / f"{run.run_id}.json"
    assert json.loads(receipt.read_text(encoding="utf-8"))["status"] == "completed"


def test_health_rejects_empty_assets_directory(tmp_path: Path) -> None:
    config = _fake_install(tmp_path)
    (config.robodojo_root / "Assets" / "test-asset.marker").unlink()
    shutil.rmtree(config.robodojo_root / "Assets" / "Eval_Layout")

    health = RoboDojoRunManager(config).health()

    assert health["status"] == "not_ready"
    assert health["problems"] == [
        f"RoboDojo Assets are empty: {config.robodojo_root / 'Assets'}"
    ]


def test_frozen_code_provider_uses_manifest_not_fake_vla_weights(tmp_path: Path) -> None:
    import hashlib
    config = _fake_install(tmp_path)
    directory = config.robodojo_root / 'XPolicyLab/policy/Frozen_Align'
    directory.mkdir()
    (directory / 'model.py').write_text('class Model: pass\n')
    descriptor = tmp_path / 'descriptor.json'
    descriptor.write_text('{"schema":"awomopaw.test-worker/v1"}\n')
    (directory / 'frozen-skill.json').write_text(json.dumps({
        'schema': 'awomopaw.frozen-skill/v1', 'release_id': 'frozen-v1',
        'descriptor': str(descriptor),
        'descriptor_sha256': hashlib.sha256(descriptor.read_bytes()).hexdigest(),
    }))
    provider = PolicyProvider('frozen-align', 'code', 'Frozen_Align', 'frozen-v1',
                              config.providers[0].policy_uv_env_path, 'frozen-code-v1')
    assert provider.public(config.robodojo_root)['ready']
    assert provider.public(config.robodojo_root)['kind'] == 'code'
    descriptor.write_text('changed')
    status = provider.public(config.robodojo_root)
    assert not status['ready']
    assert 'hash mismatch' in status['problems'][0]


def test_adapter_serializes_gpu_evaluations(tmp_path: Path) -> None:
    manager = RoboDojoRunManager(_fake_install(tmp_path, sleep_seconds=0.2))
    run = manager.start_evaluation("align_blocks", eval_num=1, seed=0)

    with pytest.raises(EvaluationConflictError):
        manager.start_evaluation("align_blocks", eval_num=1, seed=1)

    _wait_for_completion(manager, run.run_id)


def test_adapter_rejects_missing_layout_seed(tmp_path: Path) -> None:
    manager = RoboDojoRunManager(_fake_install(tmp_path))

    with pytest.raises(ValueError, match="layout seed is unavailable"):
        manager.start_session("align_blocks", policy_id="pi05", seed=3)


@pytest.mark.parametrize("task", ["../align_blocks", "align-blocks", "", "0task"])
def test_adapter_rejects_noncanonical_task_names(tmp_path: Path, task: str) -> None:
    manager = RoboDojoRunManager(_fake_install(tmp_path))
    with pytest.raises(ValueError):
        manager.start_evaluation(task, eval_num=1, seed=0)


def test_http_start_contract_accepts_json_body(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    manager = RoboDojoRunManager(_fake_install(tmp_path))
    with TestClient(create_app(manager=manager)) as client:
        response = client.post(
            "/v1/evaluations",
            json={"task": "align_blocks", "eval_num": 2, "seed": 0},
        )
    assert response.status_code == 202
    run_id = response.json()["run_id"]
    _wait_for_completion(manager, run_id)


def test_interactive_session_is_agent_selected_and_records_control_mode(
    tmp_path: Path,
) -> None:
    manager = RoboDojoRunManager(_fake_install(tmp_path))

    run = manager.start_session("align_blocks", policy_id="pi05", seed=0)
    assert run.execution_mode == "interactive"
    assert run.policy_id == "pi05"
    environment = manager._execution_environment(run)
    assert environment["AWOMOPAW_HARNESS_HOST"] == "127.0.0.1"
    assert environment["AWOMOPAW_HARNESS_PORT"] == "8099"
    assert environment["AWOMOPAW_VLA_POLICY_NAME"] == "Pi_05"
    assert environment["AWOMOPAW_POLICY_ID"] == "pi05"
    assert environment["AWOMOPAW_POLICY_KIND"] == "vla"
    assert environment["AWOMOPAW_POLICY_REVISION"] == "test-pi05-revision"
    assert environment["AWOMOPAW_POLICY_UV_ENV_PATH"].endswith("Pi_05/openpi")
    _wait_for_completion(manager, run.run_id)


def test_http_session_contract_returns_control_url(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    manager = RoboDojoRunManager(_fake_install(tmp_path))
    with TestClient(create_app(manager=manager)) as client:
        response = client.post(
            "/v1/sessions",
            json={"task": "align_blocks", "policy_id": "pi05", "seed": 0},
        )
    assert response.status_code == 202
    assert response.json()["execution_mode"] == "interactive"
    assert response.json()["control_url"] == "http://127.0.0.1:8099"
    _wait_for_completion(manager, response.json()["run_id"])


def test_http_observation_image_is_model_visible(tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    manager = RoboDojoRunManager(_fake_install(tmp_path))
    manager.current_observation_image = lambda topic: (  # type: ignore[method-assign]
        b"jpeg-bytes-" + topic.encode("ascii"),
        "image/jpeg",
    )

    with TestClient(create_app(manager=manager)) as client:
        response = client.get(
            "/v1/sessions/observation/image", params={"topic": "cam_head"}
        )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content == b"jpeg-bytes-cam_head"


def test_execution_window_survives_http_bridge(tmp_path: Path):
    from fastapi.testclient import TestClient

    manager = RoboDojoRunManager(_fake_install(tmp_path))
    captured = []
    def control(method, path, **kwargs):
        captured.append((method, path, kwargs["payload"]))
        return {"executed": 10, "discarded_actions": 40}
    manager._control_request = control
    with TestClient(create_app(manager=manager)) as client:
        response = client.post("/v1/sessions/action-chunks/execute", json={
            "chunk_id": "a" * 32, "max_actions": 10, "discard_remaining": True})
        assert response.status_code == 200
        assert response.json()["executed"] == 10
        assert captured[-1][2] == {
            "chunk_id": "a" * 32, "max_actions": 10, "discard_remaining": True}
        response = client.post("/v1/sessions/action-chunks/execute", json={"chunk_id": "b" * 32})
        assert response.status_code == 200
        assert captured[-1][2] == {"chunk_id": "b" * 32}
        response = client.post("/v1/sessions/action-chunks/execute", json={
            "chunk_id": "c" * 32, "max_actions": 0})
        assert response.status_code == 422
        assert len(captured) == 2










def test_execution_environment_recovers_conda_for_service_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager = RoboDojoRunManager(_fake_install(tmp_path))
    conda_bin = tmp_path / "miniconda" / "bin"
    conda_bin.mkdir(parents=True)
    (conda_bin / "conda").write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("ROBODOJO_CONDA_BIN_DIR", str(conda_bin))

    run = EvaluationRun(
        run_id="20261012T000000Z-1234abcd",
        task="align_blocks",
        seed=0,
        eval_num=1,
        status="starting",
        created_at="2026-10-12T00:00:00+00:00",
    )
    environment = manager._execution_environment(run)

    assert environment["PATH"].split(":")[0] == str(conda_bin)
    assert environment["OMNI_KIT_ACCEPT_EULA"] == "Y"
    assert environment["ROBODOJO_RUN_ID"] == run.run_id


def test_registry_accepts_future_wam_without_changing_session_contract(
    tmp_path: Path,
) -> None:
    config = _fake_install(tmp_path)
    wam_dir = config.robodojo_root / "XPolicyLab" / "policy" / "Future_WAM"
    (wam_dir / "checkpoints" / "wam-v1").mkdir(parents=True)
    (wam_dir / "model.py").write_text("class Model: pass\n", encoding="utf-8")
    (wam_dir / "runtime" / ".venv" / "bin").mkdir(parents=True)
    (wam_dir / "runtime" / ".venv" / "bin" / "python").write_text(
        "#!/usr/bin/env python\n", encoding="utf-8"
    )
    config = RoboDojoAdapterConfig(
        **{
            **config.__dict__,
            "providers": (
                *config.providers,
                PolicyProvider(
                    provider_id="future-wam",
                    kind="wam",
                    policy_name="Future_WAM",
                    checkpoint="wam-v1",
                    policy_uv_env_path=wam_dir / "runtime",
                    revision="test-wam-revision",
                ),
            ),
        }
    )
    manager = RoboDojoRunManager(config)

    run = manager.start_session("align_blocks", policy_id="future-wam", seed=0)

    assert run.provider_kind == "wam"
    assert run.provider_policy_name == "Future_WAM"
    assert manager._execution_environment(run)["AWOMOPAW_POLICY_KIND"] == "wam"
    _wait_for_completion(manager, run.run_id)
