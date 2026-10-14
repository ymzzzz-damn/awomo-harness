"""HTTP bridge from awomopaw tools to the RoboDojo simulation embodiment.

The bridge starts a provider-backed interactive episode and proxies the
observation/action contract owned by RoboDojo. Registered VLA/WAM providers
and the bounded LLM direct-action fallback all submit canonical action chunks
to the same embodiment executor; the bridge never applies an Isaac action.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import threading
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


_TASK_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,127}$")
_RUN_ID_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{8}$")


class AdapterConfigurationError(RuntimeError):
    """Raised when the fixed RoboDojo/Pi0.5 installation is incomplete."""


class EvaluationConflictError(RuntimeError):
    """Raised when a second GPU-heavy evaluation is requested concurrently."""


class EvaluationNotFoundError(KeyError):
    """Raised when a run id is unknown to this adapter process."""


class StartEvaluationRequest(BaseModel):
    """Backward-compatible request for one default VLA attempt."""

    model_config = ConfigDict(extra="forbid")

    task: str = Field(min_length=1, max_length=128)
    eval_num: int = Field(default=1, ge=1, le=8)
    seed: int = Field(default=0, ge=0, le=2_147_483_647)


class StartAttemptRequest(BaseModel):
    """One agent-selected harness attempt.

    ``llm_plan`` means the connected task LLM authored ``plan`` after a prior
    VLA task failure. The configured VLA remains the bounded action generator;
    RoboDojo remains the simulator and scoring authority.
    """

    model_config = ConfigDict(extra="forbid")

    task: str = Field(min_length=1, max_length=128)
    policy_id: str = Field(min_length=1, max_length=64)
    execution_mode: str = Field(default="vla", min_length=1, max_length=32)
    plan: list[str] | None = None
    parent_run_id: str | None = Field(default=None, max_length=64)
    eval_num: int = Field(default=1, ge=1, le=8)
    seed: int = Field(default=0, ge=0, le=2_147_483_647)


class StartSessionRequest(BaseModel):
    """Start one interactive LLM -> VLA -> harness episode."""

    model_config = ConfigDict(extra="forbid")

    task: str = Field(min_length=1, max_length=128)
    policy_id: str = Field(min_length=1, max_length=64)
    seed: int = Field(default=0, ge=0, le=2_147_483_647)


class PolicyChunkRequest(BaseModel):
    """Ask the active VLA/WAM provider for one action chunk."""

    model_config = ConfigDict(extra="forbid")

    observation_id: str = Field(min_length=8, max_length=64)
    provider_id: str = Field(min_length=1, max_length=64)
    subtask_goal: str = Field(min_length=1, max_length=1000)
    optimized_task_prompt: str = Field(min_length=1, max_length=3000)
    planning_summary: str = Field(default="", max_length=2000)


class DirectActionRequest(BaseModel):
    """Bounded planner fallback using the embodiment's canonical action spec."""

    model_config = ConfigDict(extra="forbid")

    observation_id: str = Field(min_length=8, max_length=64)
    subtask_goal: str = Field(min_length=1, max_length=1000)
    fallback_reason: str = Field(min_length=1, max_length=1000)
    rationale: str = Field(default="", max_length=2000)
    actions: list[dict[str, list[float]]] = Field(min_length=1, max_length=1)


class ExecuteChunkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=8, max_length=64)


class AbortSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(default="agent_abort", max_length=500)


@dataclass(frozen=True)
class PolicyProvider:
    """One model adapter registered behind the common action contract."""

    provider_id: str
    kind: str
    policy_name: str
    checkpoint: str
    policy_uv_env_path: Path
    revision: str

    def public(self, robodojo_root: Path) -> dict[str, Any]:
        policy_dir = robodojo_root / "XPolicyLab" / "policy" / self.policy_name
        problems: list[str] = []
        if self.kind not in {"vla", "wam"}:
            problems.append("kind must be vla or wam")
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", self.provider_id):
            problems.append("provider_id is invalid")
        if not _TASK_RE.fullmatch(self.policy_name):
            problems.append("policy_name is invalid")
        if not self.revision or len(self.revision) > 256:
            problems.append("provider revision is missing or too long")
        if not (policy_dir / "model.py").is_file():
            problems.append(f"policy module is missing: {policy_dir}")
        if not (policy_dir / "checkpoints" / self.checkpoint).is_dir():
            problems.append(
                f"checkpoint is missing: {policy_dir / 'checkpoints' / self.checkpoint}"
            )
        if not (self.policy_uv_env_path / ".venv" / "bin" / "python").is_file():
            problems.append(
                f"provider environment is missing: {self.policy_uv_env_path}"
            )
        return {
            "provider_id": self.provider_id,
            "kind": self.kind,
            "policy_name": self.policy_name,
            "revision": self.revision,
            "capabilities": ["observation_to_action_chunk", "joint_action"],
            "input_contract": "robodojo.observation/v1",
            "output_contract": "robodojo.action-chunk/v1",
            "lifecycle": ["observe", "execute", "reset", "close"],
            "effects": ["simulation-action"],
            "ready": not problems,
            "problems": problems,
        }


@dataclass(frozen=True)
class RoboDojoAdapterConfig:
    """Fixed simulation/provider configuration; callers cannot override paths."""

    robodojo_root: Path
    policy_dir: Path
    policy_id: str = "pi05"
    policy_name: str = "Pi_05"
    checkpoint: str = "RoboDojo-sim-arx_x5-joint-0"
    env_cfg: str = "arx_x5"
    action_type: str = "joint"
    policy_env: str = "uv"
    eval_env: str = "RoboDojo"
    policy_gpu: int = 0
    env_gpu: int = 0
    max_eval_num: int = 8
    control_port: int = 8099
    vla_policy_name: str = "Pi_05"
    providers: tuple[PolicyProvider, ...] = ()

    @classmethod
    def from_env(cls) -> "RoboDojoAdapterConfig":
        root = Path(
            os.environ.get(
                "ROBODOJO_ROOT", "/home/ymz/robodojo-pi05/RoboDojo"
            )
        ).expanduser()
        policy_value = os.environ.get(
            "ROBODOJO_HARNESS_POLICY_DIR",
            "XPolicyLab/policy/Awomopaw_Harness",
        )
        policy_dir = Path(policy_value).expanduser()
        if not policy_dir.is_absolute():
            policy_dir = root / policy_dir
        default_checkpoint = os.environ.get(
            "ROBODOJO_VLA_CHECKPOINT",
            os.environ.get(
                "ROBODOJO_PI05_CHECKPOINT",
                "RoboDojo-sim-arx_x5-joint-0",
            ),
        )
        default_name = os.environ.get("AWOMOPAW_VLA_POLICY_NAME", "Pi_05")
        provider_payload = os.environ.get("ROBODOJO_POLICY_REGISTRY_JSON", "").strip()
        if provider_payload:
            decoded = json.loads(provider_payload)
            if not isinstance(decoded, list) or not decoded:
                raise ValueError("ROBODOJO_POLICY_REGISTRY_JSON must be a non-empty list")
        else:
            decoded = [
                {
                    "provider_id": os.environ.get("ROBODOJO_VLA_ID", "pi05"),
                    "kind": "vla",
                    "policy_name": default_name,
                    "checkpoint": default_checkpoint,
                    "policy_uv_env_path": str(
                        root / "XPolicyLab" / "policy" / default_name / "openpi"
                    ),
                }
            ]
        providers: list[PolicyProvider] = []
        for item in decoded:
            if not isinstance(item, dict):
                raise ValueError("every provider registry entry must be an object")
            policy_name = str(item.get("policy_name") or "").strip()
            env_path = Path(str(item.get("policy_uv_env_path") or "")).expanduser()
            if not env_path.is_absolute():
                env_path = root / "XPolicyLab" / "policy" / policy_name / env_path
            providers.append(
                PolicyProvider(
                    provider_id=str(item.get("provider_id") or "").strip(),
                    kind=str(item.get("kind") or "").strip().lower(),
                    policy_name=policy_name,
                    checkpoint=str(item.get("checkpoint") or "").strip(),
                    policy_uv_env_path=env_path,
                    revision=str(
                        item.get("revision")
                        or f"checkpoint:{item.get('checkpoint') or 'unknown'}"
                    ).strip(),
                )
            )
        return cls(
            robodojo_root=root,
            policy_dir=policy_dir,
            policy_id=os.environ.get("ROBODOJO_VLA_ID", "pi05"),
            policy_name=os.environ.get(
                "ROBODOJO_HARNESS_POLICY_NAME", "Awomopaw_Harness"
            ),
            checkpoint=default_checkpoint,
            env_cfg=os.environ.get("ROBODOJO_ENV_CFG", "arx_x5"),
            action_type=os.environ.get("ROBODOJO_ACTION_TYPE", "joint"),
            policy_env=os.environ.get("ROBODOJO_POLICY_ENV", "uv"),
            eval_env=os.environ.get("ROBODOJO_EVAL_ENV", "RoboDojo"),
            policy_gpu=int(os.environ.get("ROBODOJO_POLICY_GPU", "0")),
            env_gpu=int(os.environ.get("ROBODOJO_ENV_GPU", "0")),
            max_eval_num=int(os.environ.get("ROBODOJO_MAX_EVAL_NUM", "8")),
            control_port=int(os.environ.get("AWOMOPAW_HARNESS_PORT", "8099")),
            vla_policy_name=default_name,
            providers=tuple(providers),
        )

    @property
    def launcher(self) -> Path:
        return self.robodojo_root / "scripts" / "robodojo.sh"

    @property
    def state_dir(self) -> Path:
        configured = os.environ.get("ROBODOJO_HARNESS_STATE_DIR", "").strip()
        if configured:
            return Path(configured).expanduser()
        return self.robodojo_root / "eval_result" / "harness_runs"

    @property
    def provider_registry(self) -> dict[str, PolicyProvider]:
        providers = self.providers or (
            PolicyProvider(
                provider_id=self.policy_id,
                kind="vla",
                policy_name=self.vla_policy_name,
                checkpoint=self.checkpoint,
                policy_uv_env_path=(
                    self.robodojo_root
                    / "XPolicyLab"
                    / "policy"
                    / self.vla_policy_name
                    / "openpi"
                ),
                revision=f"checkpoint:{self.checkpoint}",
            ),
        )
        registry = {provider.provider_id: provider for provider in providers}
        if len(registry) != len(providers):
            raise ValueError("policy provider ids must be unique")
        return registry

    def get_provider(self, provider_id: str) -> PolicyProvider:
        try:
            return self.provider_registry[provider_id]
        except KeyError as exc:
            available = ", ".join(sorted(self.provider_registry))
            raise ValueError(
                f"unknown provider_id: {provider_id}; available: {available}"
            ) from exc

    def validate(self) -> list[str]:
        problems: list[str] = []
        if not self.robodojo_root.is_dir():
            problems.append(f"RoboDojo root is missing: {self.robodojo_root}")
        if not self.launcher.is_file():
            problems.append(f"RoboDojo launcher is missing: {self.launcher}")
        if not (self.policy_dir / "eval.sh").is_file():
            problems.append(
                f"interactive harness adapter is missing: {self.policy_dir / 'eval.sh'}"
            )
        assets_dir = self.robodojo_root / "Assets"
        if not assets_dir.is_dir():
            problems.append(f"RoboDojo Assets are missing: {assets_dir}")
        else:
            try:
                if not any(assets_dir.iterdir()):
                    problems.append(f"RoboDojo Assets are empty: {assets_dir}")
            except OSError as exc:
                problems.append(f"RoboDojo Assets are unreadable: {assets_dir}: {exc}")
        if self.action_type != "joint":
            problems.append("This harness deployment only permits validated joint actions")
        if not _TASK_RE.fullmatch(self.policy_name):
            problems.append("ROBODOJO_HARNESS_POLICY_NAME must be canonical")
        try:
            registry = self.provider_registry
        except ValueError as exc:
            problems.append(str(exc))
            registry = {}
        for provider in registry.values():
            problems.extend(provider.public(self.robodojo_root)["problems"])
        if self.max_eval_num < 1:
            problems.append("ROBODOJO_MAX_EVAL_NUM must be positive")
        return problems


@dataclass
class EvaluationRun:
    run_id: str
    task: str
    seed: int
    eval_num: int
    status: str
    created_at: str
    policy_id: str = "pi05"
    provider_kind: str = "vla"
    provider_policy_name: str = "Pi_05"
    provider_checkpoint: str = ""
    provider_revision: str = ""
    execution_mode: str = "vla"
    plan: list[str] | None = None
    parent_run_id: str | None = None
    started_at: str | None = None
    finished_at: str | None = None
    pid: int | None = None
    return_code: int | None = None
    log_path: str | None = None
    result_path: str | None = None
    artifact_dir: str | None = None
    error: str | None = None
    result: dict[str, Any] | None = field(default=None, repr=False)

    def public(self, *, include_result: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        result = payload.pop("result", None)
        payload["result_available"] = bool(result)
        if include_result:
            payload["result"] = result
        return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RoboDojoRunManager:
    """Serialize harness attempts and retain post-episode receipts."""

    def __init__(self, config: RoboDojoAdapterConfig) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._runs: dict[str, EvaluationRun] = {}
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._latest_run_id: str | None = None

    def health(self) -> dict[str, Any]:
        problems = self.config.validate()
        with self._lock:
            active = next(
                (
                    run.public()
                    for run in self._runs.values()
                    if run.status in {"starting", "running"}
                ),
                None,
            )
        return {
            "status": "ready" if not problems else "not_ready",
            "adapter": "awomopaw-robodojo-embodiment/v3",
            "available_policies": [
                provider.public(self.config.robodojo_root)
                for provider in self.config.provider_registry.values()
            ],
            "execution_modes": ["interactive", "llm_direct_fallback"],
            "control_url": f"http://127.0.0.1:{self.config.control_port}",
            "planning_handoff": (
                "llm_plan_to_registered_policy_to_embodiment_action_chunk; "
                "on_provider_failure_llm_direct_to_same_embodiment_executor"
            ),
            "simulator": "RoboDojo/Isaac Sim",
            "problems": problems,
            "active_evaluation": active,
        }

    def _validate_request(self, task: str, eval_num: int, seed: int) -> None:
        problems = self.config.validate()
        if problems:
            raise AdapterConfigurationError("; ".join(problems))
        if not _TASK_RE.fullmatch(task):
            raise ValueError("task must be a canonical RoboDojo task name")
        task_file = (
            self.config.robodojo_root
            / "task"
            / "RoboDojo"
            / "tasks"
            / f"{task}.py"
        )
        if not task_file.is_file():
            raise ValueError(f"unknown RoboDojo task: {task}")
        layout_seed_dir = (
            self.config.robodojo_root
            / "Assets"
            / "Eval_Layout"
            / "RoboDojo"
            / self.config.env_cfg
            / str(seed)
        )
        if not layout_seed_dir.is_dir():
            raise ValueError(
                f"RoboDojo layout seed is unavailable: {layout_seed_dir}"
            )
        if isinstance(eval_num, bool) or not 1 <= int(eval_num) <= self.config.max_eval_num:
            raise ValueError(
                f"eval_num must be between 1 and {self.config.max_eval_num}"
            )
        if isinstance(seed, bool) or not 0 <= int(seed) <= 2_147_483_647:
            raise ValueError("seed must be an integer between 0 and 2147483647")

    def _validate_attempt(
        self,
        *,
        task: str,
        policy_id: str,
        execution_mode: str,
        plan: list[str] | None,
        parent_run_id: str | None,
        eval_num: int,
        seed: int,
    ) -> list[str] | None:
        self._validate_request(task, eval_num, seed)
        provider = self.config.get_provider(policy_id)
        provider_status = provider.public(self.config.robodojo_root)
        if not provider_status["ready"]:
            raise AdapterConfigurationError("; ".join(provider_status["problems"]))
        if execution_mode not in {"vla", "llm_plan", "interactive"}:
            raise ValueError(
                "execution_mode must be 'vla', 'llm_plan', or 'interactive'"
            )

        if execution_mode in {"vla", "interactive"}:
            if plan or parent_run_id:
                raise ValueError("vla mode does not accept plan or parent_run_id")
            return None

        if not parent_run_id:
            raise ValueError("llm_plan mode requires parent_run_id")
        parent = self.get_run(parent_run_id)
        if parent.task != task:
            raise ValueError("planned attempt task must match its parent run")
        if parent.policy_id != policy_id:
            raise ValueError("planned attempt policy must match its parent run")
        if parent.status != "completed" or parent.result is None:
            raise ValueError("planned attempt requires a completed parent VLA run")
        if float(parent.result.get("success_rate", 0.0)) >= 1.0:
            raise ValueError("parent VLA run already completed the task")
        if not isinstance(plan, list) or not 1 <= len(plan) <= 12:
            raise ValueError("llm_plan mode requires 1 to 12 high-level plan steps")
        clean_plan: list[str] = []
        for step in plan:
            if not isinstance(step, str):
                raise ValueError("every plan step must be text")
            cleaned = " ".join(step.split())
            if not 1 <= len(cleaned) <= 512:
                raise ValueError("every plan step must contain 1 to 512 characters")
            clean_plan.append(cleaned)
        return clean_plan

    def _new_run_id(self) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{timestamp}-{os.urandom(4).hex()}"

    def _artifact_dir(self, run: EvaluationRun) -> Path:
        additional = (
            f"{run.seed}_ckpt_name={run.provider_checkpoint},"
            f"action_type={self.config.action_type}"
        )
        return (
            self.config.robodojo_root
            / "eval_result"
            / "RoboDojo"
            / run.task
            / self.config.policy_name
            / self.config.env_cfg
            / additional
            / run.run_id
        )

    def _command(self, run: EvaluationRun) -> list[str]:
        return [
            "bash",
            str(self.config.launcher),
            "eval",
            "--policy-dir",
            str(self.config.policy_dir),
            "--task",
            run.task,
            "--ckpt",
            run.provider_checkpoint,
            "--env-cfg",
            self.config.env_cfg,
            "--action-type",
            self.config.action_type,
            "--seed",
            str(run.seed),
            "--policy-gpu",
            str(self.config.policy_gpu),
            "--env-gpu",
            str(self.config.env_gpu),
            "--policy-env",
            self.config.policy_env,
            "--eval-env",
            self.config.eval_env,
            "--eval-num",
            str(run.eval_num),
        ]

    def _execution_environment(self, run: EvaluationRun) -> dict[str, str]:
        """Build a stable launcher environment for services started by tmux/systemd.

        Interactive shells commonly add conda to PATH, while a long-lived tmux
        server does not.  RoboDojo's official policy launcher calls ``conda``
        before activating the Pi0.5 uv environment, so recover the conventional
        local installation without sourcing user shell startup files.
        """
        environment = os.environ.copy()
        path_parts = environment.get("PATH", "").split(os.pathsep)
        configured = environment.get("ROBODOJO_CONDA_BIN_DIR", "").strip()
        candidates = [
            Path(configured).expanduser() if configured else None,
            Path.home() / "miniconda3" / "bin",
            Path.home() / "anaconda3" / "bin",
            Path("/opt/conda/bin"),
        ]
        if not any((Path(part) / "conda").is_file() for part in path_parts if part):
            for candidate in candidates:
                if candidate is not None and (candidate / "conda").is_file():
                    path_parts.insert(0, str(candidate))
                    break
        environment["PATH"] = os.pathsep.join(part for part in path_parts if part)
        # The installed Isaac Sim kit checks OMNI_KIT_ACCEPT_EULA before it
        # prompts on stdin, which is unavailable to a loopback service.
        environment.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")
        environment["ROBODOJO_RUN_ID"] = run.run_id
        environment["ROBODOJO_HARNESS_EXECUTION_MODE"] = run.execution_mode
        environment["AWOMOPAW_HARNESS_HOST"] = "127.0.0.1"
        environment["AWOMOPAW_HARNESS_PORT"] = str(self.config.control_port)
        provider = self.config.get_provider(run.policy_id)
        environment["AWOMOPAW_POLICY_ID"] = provider.provider_id
        environment["AWOMOPAW_POLICY_KIND"] = provider.kind
        environment["AWOMOPAW_POLICY_REVISION"] = provider.revision
        environment["AWOMOPAW_VLA_POLICY_NAME"] = provider.policy_name
        environment["AWOMOPAW_POLICY_UV_ENV_PATH"] = str(
            provider.policy_uv_env_path
        )
        if run.plan:
            environment["ROBODOJO_HARNESS_PLAN_JSON"] = json.dumps(
                run.plan, ensure_ascii=False
            )
        else:
            environment.pop("ROBODOJO_HARNESS_PLAN_JSON", None)
        environment["PYTHONUNBUFFERED"] = "1"
        return environment

    def start_attempt(
        self,
        *,
        task: str,
        policy_id: str,
        execution_mode: str,
        plan: list[str] | None,
        parent_run_id: str | None,
        eval_num: int,
        seed: int,
    ) -> EvaluationRun:
        clean_plan = self._validate_attempt(
            task=task,
            policy_id=policy_id,
            execution_mode=execution_mode,
            plan=plan,
            parent_run_id=parent_run_id,
            eval_num=eval_num,
            seed=seed,
        )
        provider = self.config.get_provider(policy_id)
        with self._lock:
            if any(
                run.status in {"starting", "running"}
                for run in self._runs.values()
            ):
                raise EvaluationConflictError(
                    "A RoboDojo evaluation is already using the configured GPU"
                )

            run = EvaluationRun(
                run_id=self._new_run_id(),
                task=task,
                seed=int(seed),
                eval_num=int(eval_num),
                status="starting",
                created_at=_utc_now(),
                policy_id=policy_id,
                provider_kind=provider.kind,
                provider_policy_name=provider.policy_name,
                provider_checkpoint=provider.checkpoint,
                provider_revision=provider.revision,
                execution_mode=execution_mode,
                plan=clean_plan,
                parent_run_id=parent_run_id,
            )
            artifact_dir = self._artifact_dir(run)
            run.artifact_dir = str(artifact_dir)
            run.result_path = str(artifact_dir / "_result.json")

            self.config.state_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.config.state_dir / f"{run.run_id}.log"
            run.log_path = str(log_path)
            self._runs[run.run_id] = run
            self._latest_run_id = run.run_id
            self._persist(run)

            environment = self._execution_environment(run)
            with log_path.open("ab", buffering=0) as log_stream:
                process = subprocess.Popen(
                    self._command(run),
                    cwd=str(self.config.robodojo_root),
                    env=environment,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            run.pid = process.pid
            run.status = "running"
            run.started_at = _utc_now()
            self._processes[run.run_id] = process
            self._persist(run)

            watcher = threading.Thread(
                target=self._watch,
                args=(run.run_id,),
                name=f"robodojo-eval-{run.run_id}",
                daemon=True,
            )
            watcher.start()
            return run

    def start_evaluation(self, task: str, eval_num: int, seed: int) -> EvaluationRun:
        """Backward-compatible alias for one default end-to-end VLA attempt."""
        return self.start_attempt(
            task=task,
            policy_id=self.config.policy_id,
            execution_mode="vla",
            plan=None,
            parent_run_id=None,
            eval_num=eval_num,
            seed=seed,
        )

    def start_session(self, task: str, policy_id: str, seed: int) -> EvaluationRun:
        """Start one interactive episode; agent decisions arrive over control API."""
        return self.start_attempt(
            task=task,
            policy_id=policy_id,
            execution_mode="interactive",
            plan=None,
            parent_run_id=None,
            eval_num=1,
            seed=seed,
        )

    def _active_session(self) -> EvaluationRun:
        run = self.current_run()
        if run is None or run.execution_mode != "interactive":
            raise AdapterConfigurationError("No interactive session has been started")
        if run.status not in {"starting", "running"}:
            raise AdapterConfigurationError(
                f"Interactive session is not active: {run.status}"
            )
        return run

    def _control_request(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout: float = 190.0,
    ) -> dict[str, Any]:
        """Proxy one tool call to the in-episode embodiment endpoint."""
        self._active_session()
        import httpx

        url = f"http://127.0.0.1:{self.config.control_port}{path}"
        try:
            response = httpx.request(
                method,
                url,
                json=payload,
                timeout=timeout,
                trust_env=False,
            )
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPStatusError as exc:
            try:
                detail = exc.response.json().get("detail", exc.response.text)
            except (ValueError, AttributeError):
                detail = exc.response.text
            raise AdapterConfigurationError(
                f"simulation embodiment returned HTTP {exc.response.status_code}: {detail}"
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise AdapterConfigurationError(
                f"simulation embodiment is not ready at {url}: {exc}"
            ) from exc
        if not isinstance(body, dict):
            raise AdapterConfigurationError("simulation embodiment returned non-object JSON")
        return body

    def session_status(self) -> dict[str, Any]:
        return self._control_request("GET", "/v1/session", timeout=10.0)

    def current_observation(self) -> dict[str, Any]:
        return self._control_request(
            "GET", "/v1/observations/current", timeout=15.0
        )

    def current_observation_image(self, topic: str) -> tuple[bytes, str]:
        """Return one model-visible RGB snapshot from the active embodiment."""
        if topic not in {"cam_head", "cam_left_wrist", "cam_right_wrist"}:
            raise ValueError(f"unsupported observation topic: {topic}")
        self._active_session()
        import httpx

        url = (
            f"http://127.0.0.1:{self.config.control_port}"
            f"/v1/observations/current/image"
        )
        try:
            response = httpx.get(
                url,
                params={"topic": topic},
                timeout=15.0,
                trust_env=False,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise AdapterConfigurationError(
                "simulation embodiment returned "
                f"HTTP {exc.response.status_code} for {topic}"
            ) from exc
        except httpx.HTTPError as exc:
            raise AdapterConfigurationError(
                f"simulation snapshot is not ready at {url}: {exc}"
            ) from exc
        return response.content, response.headers.get("content-type", "image/jpeg")

    def request_policy_chunk(self, payload: dict[str, Any]) -> dict[str, Any]:
        run = self._active_session()
        if payload.get("provider_id") != run.policy_id:
            raise ValueError(
                f"session provider is {run.policy_id!r}; received "
                f"{payload.get('provider_id')!r}"
            )
        return self._control_request(
            "POST", "/v1/policies/chunks", payload=payload
        )

    def submit_direct_action_chunk(
        self, payload: dict[str, Any]
    ) -> dict[str, Any]:
        return self._control_request(
            "POST", "/v1/actions/direct", payload=payload, timeout=30.0
        )

    def execute_action_chunk(self, chunk_id: str) -> dict[str, Any]:
        return self._control_request(
            "POST",
            "/v1/action-chunks/execute",
            payload={"chunk_id": chunk_id},
        )

    def abort_session(self, reason: str) -> dict[str, Any]:
        return self._control_request(
            "POST",
            "/v1/session/abort",
            payload={"reason": reason},
            timeout=20.0,
        )

    def _watch(self, run_id: str) -> None:
        with self._lock:
            process = self._processes[run_id]
        return_code = process.wait()
        with self._lock:
            run = self._runs[run_id]
            run.return_code = return_code
            run.finished_at = _utc_now()
            result_file = Path(run.result_path or "")
            if return_code == 0 and result_file.is_file():
                try:
                    payload = json.loads(result_file.read_text(encoding="utf-8"))
                    if not isinstance(payload, dict):
                        raise ValueError("result is not a JSON object")
                    run.result = payload
                    run.status = "completed"
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    run.status = "failed"
                    run.error = f"RoboDojo result could not be read: {exc}"
            else:
                run.status = "failed"
                if return_code != 0:
                    run.error = f"RoboDojo evaluator exited with code {return_code}"
                else:
                    run.error = "RoboDojo evaluator produced no _result.json"
            self._processes.pop(run_id, None)
            self._persist(run)

    def _persist(self, run: EvaluationRun) -> None:
        receipt = self.config.state_dir / f"{run.run_id}.json"
        temporary = receipt.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(run.public(include_result=True), ensure_ascii=False, indent=2)
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(receipt)

    def get_run(self, run_id: str) -> EvaluationRun:
        if not _RUN_ID_RE.fullmatch(run_id):
            raise EvaluationNotFoundError(run_id)
        with self._lock:
            try:
                return self._runs[run_id]
            except KeyError as exc:
                raise EvaluationNotFoundError(run_id) from exc

    def current_run(self) -> EvaluationRun | None:
        with self._lock:
            if self._latest_run_id is None:
                return None
            return self._runs[self._latest_run_id]

    def get_result(self, run_id: str) -> dict[str, Any]:
        run = self.get_run(run_id)
        if run.status in {"starting", "running"}:
            raise EvaluationConflictError(
                "Evaluation is still running; results are only exposed post-episode"
            )
        if run.status != "completed" or run.result is None:
            raise AdapterConfigurationError(run.error or "Evaluation failed")
        artifact_dir = Path(run.artifact_dir or "")
        videos = sorted(str(path) for path in artifact_dir.glob("*.mp4"))
        trace_path = artifact_dir / "awomopaw_harness_trace.jsonl"
        return {
            "run": run.public(include_result=True),
            "task_succeeded": float(run.result.get("success_rate", 0.0)) >= 1.0,
            "harness_trace": str(trace_path) if trace_path.is_file() else None,
            "videos": videos,
        }


def create_app(
    config: RoboDojoAdapterConfig | None = None,
    manager: RoboDojoRunManager | None = None,
):
    """Create the loopback HTTP bridge used by the robodojo_sim embodiment."""
    from fastapi import FastAPI, HTTPException, Response
    active_manager = manager or RoboDojoRunManager(
        config or RoboDojoAdapterConfig.from_env()
    )
    app = FastAPI(title="awomopaw RoboDojo Harness Adapter", version="3.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return active_manager.health()

    @app.post("/v1/evaluations", status_code=202)
    def start_evaluation(request: StartEvaluationRequest) -> dict[str, Any]:
        try:
            run = active_manager.start_evaluation(
                request.task, request.eval_num, request.seed
            )
            return run.public()
        except EvaluationConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (AdapterConfigurationError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/attempts", status_code=202)
    def start_attempt(request: StartAttemptRequest) -> dict[str, Any]:
        try:
            run = active_manager.start_attempt(
                task=request.task,
                policy_id=request.policy_id,
                execution_mode=request.execution_mode,
                plan=request.plan,
                parent_run_id=request.parent_run_id,
                eval_num=request.eval_num,
                seed=request.seed,
            )
            return run.public()
        except EvaluationConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except EvaluationNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Unknown parent run") from exc
        except (AdapterConfigurationError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/v1/sessions", status_code=202)
    def start_session(request: StartSessionRequest) -> dict[str, Any]:
        try:
            run = active_manager.start_session(
                request.task, request.policy_id, request.seed
            )
            payload = run.public()
            payload["control_url"] = (
                f"http://127.0.0.1:{active_manager.config.control_port}"
            )
            return payload
        except EvaluationConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except (AdapterConfigurationError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/v1/sessions/status")
    def session_status() -> dict[str, Any]:
        try:
            return active_manager.session_status()
        except AdapterConfigurationError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/v1/sessions/observation")
    def current_observation() -> dict[str, Any]:
        try:
            return active_manager.current_observation()
        except AdapterConfigurationError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/v1/sessions/observation/image")
    def current_observation_image(topic: str = "cam_head") -> Response:
        try:
            content, media_type = active_manager.current_observation_image(topic)
            return Response(content=content, media_type=media_type)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except AdapterConfigurationError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/sessions/policy-chunks")
    def request_policy_chunk(request: PolicyChunkRequest) -> dict[str, Any]:
        try:
            return active_manager.request_policy_chunk(request.model_dump())
        except (AdapterConfigurationError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/sessions/direct-action-chunks")
    def submit_direct_action_chunk(request: DirectActionRequest) -> dict[str, Any]:
        try:
            return active_manager.submit_direct_action_chunk(request.model_dump())
        except (AdapterConfigurationError, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/sessions/action-chunks/execute")
    def execute_action_chunk(request: ExecuteChunkRequest) -> dict[str, Any]:
        try:
            return active_manager.execute_action_chunk(request.chunk_id)
        except AdapterConfigurationError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/v1/sessions/abort")
    def abort_session(request: AbortSessionRequest) -> dict[str, Any]:
        try:
            return active_manager.abort_session(request.reason)
        except AdapterConfigurationError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/v1/evaluations/current")
    def current_evaluation() -> dict[str, Any]:
        run = active_manager.current_run()
        if run is None:
            raise HTTPException(status_code=404, detail="No evaluation has been started")
        return run.public()

    @app.get("/v1/evaluations/status")
    def evaluation_status(run_id: str) -> dict[str, Any]:
        try:
            return active_manager.get_run(run_id).public()
        except EvaluationNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Unknown evaluation run") from exc

    @app.get("/v1/evaluations/result")
    def evaluation_result(run_id: str) -> dict[str, Any]:
        try:
            return active_manager.get_result(run_id)
        except EvaluationNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Unknown evaluation run") from exc
        except EvaluationConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except AdapterConfigurationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    app.state.robodojo_manager = active_manager
    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8098)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("The simulation bridge must remain loopback-only")

    import uvicorn

    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
