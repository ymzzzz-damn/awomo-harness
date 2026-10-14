"""Supervise simulation code policies; the existing embodiment applies actions."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import threading
from typing import Any


class CodePolicyManagerMixin:
    def _code_runtime(self):
        runtime = Path(os.environ.get("ROBODOJO_CODE_POLICY_RUNTIME", "/home/ymz/Documents/ChatGPT/robodojo/runtime/awomopaw-code-policy")).resolve()
        spec = importlib.util.spec_from_file_location("awomopaw_sim_code_contract", runtime / "contract.py")
        if spec is None or spec.loader is None:
            raise ValueError("Code-policy runtime is unavailable")
        contract = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(contract)
        return runtime, contract

    def code_policy_api(self) -> dict[str, Any]:
        runtime, contract = self._code_runtime()
        return {**contract.API, "worker_available": (runtime / "worker.py").is_file(),
                "scope": "experimental simulation-only code-policy fallback"}

    def code_policy_running(self) -> bool:
        return any(p.poll() is None for p in self._code_processes.values())

    def start_code_policy(self, payload: dict[str, Any]) -> dict[str, Any]:
        runtime, contract = self._code_runtime()
        contract.validate_source(payload["source"])
        if self.config.env_cfg != "arx_x5":
            raise ValueError("The installed generic code backend is calibrated for arx_x5 only")
        if len(json.dumps(payload.get("memory", {}))) > 64000:
            raise ValueError("code-policy memory exceeds the 64000-character limit")
        with self._lock:
            if self.code_policy_running():
                raise ValueError("A generated code policy is already running")
            observation = self.current_observation()
            if observation["observation_id"] != payload["observation_id"]:
                raise ValueError("observation_id is stale")
            if observation.get("episode_end") or observation.get("pending_chunk_id"):
                raise ValueError("code takeover requires a live episode with no pending VLA chunk")
            if not observation.get("instruction") or not observation.get("initial_robot_state"):
                raise ValueError("official instruction and episode initial robot state are required")
            run = self.get_run(observation["run_id"])
            job_id = "code-" + os.urandom(8).hex()
            output = Path(run.artifact_dir) / "code-policies" / job_id
            output.mkdir(parents=True, exist_ok=False)
            source = output / "policy.py"
            source.write_text(payload["source"], encoding="utf-8")
            sha = hashlib.sha256(source.read_bytes()).hexdigest()
            config = {
                "bridge": os.environ.get("ROBODOJO_CODE_BRIDGE_URL", "http://127.0.0.1:8098"),
                "run_id": run.run_id, "instruction": observation["instruction"],
                "initial_state": observation["initial_robot_state"],
                "source": str(source), "memory": payload.get("memory", {}),
                "output": str(output / "execution"), "fallback_reason": payload["fallback_reason"],
            }
            config_path = output / "worker-config.json"
            config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
            primitives_path = Path(os.environ["ROBODOJO_PRIMITIVES_PATH"]).resolve()
            if not primitives_path.is_dir():
                raise ValueError("Configured primitive provider directory is unavailable")
            python = os.environ["ROBODOJO_CODE_POLICY_PYTHON"]
            env = os.environ.copy()
            for key in list(env):
                if "API_KEY" in key:
                    env.pop(key)
            env.update(PYTHONPATH=os.pathsep.join([str(runtime), str(primitives_path)]),
                       PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       CUDA_VISIBLE_DEVICES=str(self.config.policy_gpu))
            with (output / "worker.log").open("ab", buffering=0) as log:
                process = subprocess.Popen([python, str(runtime / "worker.py"), str(config_path)],
                                           env=env, cwd=str(runtime), stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
            job = {"policy_id": job_id, "run_id": run.run_id, "status": "running", "pid": process.pid,
                   "source_sha256": sha, "source_path": str(source), "artifact_dir": str(output),
                   "fallback_reason": payload["fallback_reason"], "instruction": observation["instruction"]}
            self._code_jobs[job_id] = job
            self._code_processes[job_id] = process
            (output / "receipt.json").write_text(json.dumps(job, ensure_ascii=False, indent=2) + "\n")
            threading.Thread(target=self._watch_code_policy, args=(job_id,), daemon=True).start()
            return dict(job)

    def _watch_code_policy(self, job_id: str) -> None:
        process = self._code_processes[job_id]
        job = self._code_jobs[job_id]
        try:
            rc = process.wait(timeout=900)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
            rc = process.returncode
            job["error"] = "code policy exceeded 900-second runtime limit"
        status_path = Path(job["artifact_dir"]) / "execution" / "worker-status.json"
        if status_path.is_file():
            job.update(json.loads(status_path.read_text()))
        else:
            job.update(status="error", error=job.get("error", "worker exited without status"))
        job["return_code"] = rc
        (Path(job["artifact_dir"]) / "receipt.json").write_text(json.dumps(job, ensure_ascii=False, indent=2) + "\n")

    def code_policy_status(self, policy_id: str) -> dict[str, Any]:
        try:
            return dict(self._code_jobs[policy_id])
        except KeyError as exc:
            raise ValueError("unknown code policy id") from exc

    def stop_code_policies(self) -> None:
        for process in self._code_processes.values():
            if process.poll() is None:
                process.terminate()
