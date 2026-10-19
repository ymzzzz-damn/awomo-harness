"""Interactive awomopaw embodiment for one RoboDojo/Isaac episode.

The embodiment is the single action boundary for the simulator. A registered
VLA/WAM normally receives the current observation and a planner-authored
subtask, then returns a canonical action chunk. If that provider reports that
it cannot continue, the connected LLM may submit a short direct-action chunk
through the same validation and execution path. No provider talks to Isaac
directly and no in-episode reward or success signal is exposed to planning.
"""

from copy import deepcopy
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import queue
import threading
import time
from typing import Any
import uuid

import numpy as np
from PIL import Image


_CAMERA_NAMES = ("cam_head", "cam_left_wrist", "cam_right_wrist")
_MAX_POLICY_ACTIONS = 256
_MAX_DIRECT_ACTIONS = 40
_ROBOT_STATE_SUFFIXES = (
    "arm_joint_state",
    "ee_joint_state",
    "ee_pose",
    "tcp_pose",
    "delta_ee_pose",
)


@dataclass
class _Command:
    kind: str
    payload: dict[str, Any]
    done: threading.Event = field(default_factory=threading.Event)
    result: dict[str, Any] | None = None
    error: str | None = None


def _normalise_rgb(value: Any) -> np.ndarray:
    """Return an HWC uint8 RGB image suitable for a model-visible snapshot."""
    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"camera frame must have 3 dimensions, got {array.shape}")
    if array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] not in (3, 4):
        raise ValueError(f"camera frame must have 3 or 4 channels, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(array)) <= 1.0 else 1.0
        array = np.clip(array * scale, 0.0, 255.0).astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array[..., :3])


def _action_signature(action: Any) -> dict[str, Any]:
    if not isinstance(action, dict):
        return {"type": type(action).__name__}
    signature: dict[str, Any] = {}
    for key, value in action.items():
        try:
            signature[str(key)] = list(np.asarray(value).shape)
        except Exception:
            signature[str(key)] = []
    return signature


def _json_vector(value: Any, *, max_items: int = 32) -> list[float] | None:
    """Return a small finite 1-D vector for the model-visible contract."""
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.ndim != 1 or array.size > max_items or not np.isfinite(array).all():
        return None
    return [float(item) for item in array.tolist()]


def _public_robot_state(observation: dict[str, Any]) -> dict[str, list[float]]:
    """Expose proprioception only; simulator reward/state stays private."""
    state = observation.get("state")
    if not isinstance(state, dict):
        return {}
    public: dict[str, list[float]] = {}
    for key, value in state.items():
        if not str(key).endswith(_ROBOT_STATE_SUFFIXES):
            continue
        vector = _json_vector(value)
        if vector is not None:
            public[str(key)] = vector
    return public


def _action_spec(task_env: Any, observation: dict[str, Any]) -> dict[str, Any]:
    """Describe the canonical absolute-joint action accepted by the embodiment."""
    required: dict[str, dict[str, Any]] = {}
    info = getattr(task_env, "robot_action_dim_info", None)
    if isinstance(info, dict):
        arm_dims = list(info.get("arm_dim") or [])
        ee_dims = list(info.get("ee_dim") or [])
        if len(arm_dims) == len(ee_dims) == 1:
            required = {
                "arm_joint_state": {"dimension": int(arm_dims[0]), "finite": True},
                "ee_joint_state": {
                    "dimension": int(ee_dims[0]),
                    "finite": True,
                    "normalized_range": [0.0, 1.0],
                },
            }
        elif len(arm_dims) == len(ee_dims) == 2:
            required = {
                "left_arm_joint_state": {
                    "dimension": int(arm_dims[0]),
                    "finite": True,
                },
                "left_ee_joint_state": {
                    "dimension": int(ee_dims[0]),
                    "finite": True,
                    "normalized_range": [0.0, 1.0],
                },
                "right_arm_joint_state": {
                    "dimension": int(arm_dims[1]),
                    "finite": True,
                },
                "right_ee_joint_state": {
                    "dimension": int(ee_dims[1]),
                    "finite": True,
                    "normalized_range": [0.0, 1.0],
                },
            }
    if not required:
        for key, value in _public_robot_state(observation).items():
            if key.endswith(("arm_joint_state", "ee_joint_state")):
                required[key] = {"dimension": len(value), "finite": True}
    return {
        "schema": "robodojo.action-chunk/v1",
        "action_type": "absolute_joint",
        "required_fields": required,
        "max_policy_actions": _MAX_POLICY_ACTIONS,
        "max_direct_actions": _MAX_DIRECT_ACTIONS,
        "supports_execution_window": True,
        "max_execution_actions": _MAX_POLICY_ACTIONS,
        "execution_owner": "robodojo_sim_embodiment",
    }


class InteractiveEpisodeHarness:
    """Thread-safe control plane used by HTTP handlers and Isaac's main thread."""

    def __init__(self, task_env: Any, model_client: Any, *, host: str, port: int) -> None:
        self.task_env = task_env
        self.model_client = model_client
        self.host = host
        self.port = int(port)
        self._commands: queue.Queue[_Command] = queue.Queue(maxsize=8)
        self._lock = threading.RLock()
        self._server = None
        self._server_thread: threading.Thread | None = None
        self._status = "starting"
        self._channel_id = uuid.uuid4().hex
        self._observation_sequence = 0
        self._observation_received_at: float | None = None
        self._observation: dict[str, Any] | None = None
        self._initial_robot_state: dict[str, Any] | None = None
        self._observation_id: str | None = None
        self._observation_paths: dict[str, str] = {}
        self._pending_actions: list[dict[str, Any]] = []
        self._pending_chunk_id: str | None = None
        self._pending_source: str | None = None
        self._pending_provider_id: str | None = None
        self._pending_created_at: float | None = None
        self._action_cursor = 0
        self._last_error: str | None = None
        self._last_failure: dict[str, Any] | None = None
        self._last_subtask_goal: str | None = None
        self._last_task_prompt: str | None = None
        self._last_planning_summary: str | None = None
        self._started_at = time.time()
        self._trace_path = Path(task_env.save_dir) / "awomopaw_harness_trace.jsonl"
        self._snapshot_dir = Path(task_env.save_dir) / "awomopaw_harness_frames"
        self._provider_id = os.environ.get("AWOMOPAW_POLICY_ID", "pi05").strip()
        self._provider_kind = os.environ.get("AWOMOPAW_POLICY_KIND", "vla").strip()
        self._provider_revision = os.environ.get(
            "AWOMOPAW_POLICY_REVISION", "unversioned"
        ).strip()
        self._max_observation_age_s = float(
            os.environ.get("AWOMOPAW_OBSERVATION_MAX_AGE_S", "300")
        )
        self._max_chunk_age_s = float(
            os.environ.get("AWOMOPAW_ACTION_CHUNK_MAX_AGE_S", "300")
        )
        if self._max_observation_age_s <= 0 or self._max_chunk_age_s <= 0:
            raise ValueError("observation and action chunk age limits must be positive")

    def _trace(self, event: str, **payload: Any) -> None:
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"time": time.time(), "event": event, **payload}
        with self._trace_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _set_observation(self, observation: dict[str, Any]) -> None:
        observation_id = uuid.uuid4().hex
        snapshot_dir = self._snapshot_dir / observation_id
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, str] = {}
        vision = observation.get("vision") or {}
        for camera_name, camera_value in vision.items():
            color = camera_value.get("color") if isinstance(camera_value, dict) else camera_value
            if color is None:
                continue
            try:
                rgb = _normalise_rgb(color)
            except (TypeError, ValueError):
                continue
            path = snapshot_dir / f"{camera_name}.jpg"
            Image.fromarray(rgb).save(path, format="JPEG", quality=90)
            paths[str(camera_name)] = str(path)

        with self._lock:
            if self._initial_robot_state is None:
                self._initial_robot_state = _public_robot_state(observation)
            self._observation_sequence += 1
            self._observation = observation
            self._observation_id = observation_id
            self._observation_received_at = time.monotonic()
            self._observation_paths = paths
            self._status = "ready"
        self._trace(
            "observation",
            observation_id=observation_id,
            official_instruction=observation.get("instruction"),
            cameras=sorted(paths),
            action_count=int(self.task_env.take_action_cnt[0]),
        )

    def public_status(self) -> dict[str, Any]:
        with self._lock:
            episode_end = bool(self.task_env.end_flag[0])
            task_success = bool(self.task_env.success[0]) if episode_end else None
            return {
                "status": self._status,
                "task": self.task_env.task_name,
                "run_id": self.task_env.run_id,
                "observation_id": self._observation_id,
                "observation_sequence": self._observation_sequence,
                "cameras": sorted(self._observation_paths),
                "action_count": int(self.task_env.take_action_cnt[0]),
                "action_limit": int(self.task_env.step_lim),
                "episode_end": episode_end,
                "task_success": task_success,
                "pending_chunk_id": self._pending_chunk_id,
                "pending_source": self._pending_source,
                "pending_provider_id": self._pending_provider_id,
                "pending_actions_remaining": max(
                    0, len(self._pending_actions) - self._action_cursor
                ),
                "last_subtask_goal": self._last_subtask_goal,
                "last_task_prompt": self._last_task_prompt,
                "last_error": self._last_error,
                "last_failure": self._last_failure,
                "active_provider": {
                    "provider_id": self._provider_id,
                    "kind": self._provider_kind,
                    "revision": self._provider_revision,
                },
                "uptime_seconds": round(time.time() - self._started_at, 3),
            }

    def observation_metadata(self) -> dict[str, Any]:
        status = self.public_status()
        if status["observation_id"] is None:
            raise RuntimeError("initial observation is not ready")
        with self._lock:
            observation = deepcopy(self._observation) or {}
            status["image_paths"] = dict(self._observation_paths)
        # Task language is already part of the policy's public observation.
        # Expose it to the planner too, including dynamic color/category goals,
        # so it never has to infer the requested task from privileged layouts.
        status["instruction"] = observation.get("instruction", "")
        status["robot_state"] = _public_robot_state(observation)
        status["initial_robot_state"] = deepcopy(self._initial_robot_state)
        status["action_spec"] = _action_spec(self.task_env, observation)
        return status

    def _require_current_observation(self, observation_id: Any) -> str:
        with self._lock:
            expected_id = self._observation_id
            received_at = self._observation_received_at
        if observation_id != expected_id or expected_id is None:
            raise ValueError("observation_id is stale; inspect the current frame first")
        if received_at is None or time.monotonic() - received_at > self._max_observation_age_s:
            raise ValueError("observation exceeded the bounded freshness window")
        return expected_id

    @staticmethod
    def _require_official_instruction(observation: dict[str, Any]) -> str:
        instruction = observation.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(
                "official task instruction is unavailable; do not infer the goal "
                "from a task name or substitute a planner-authored task"
            )
        return instruction

    def snapshot_path(self, topic: str) -> Path:
        with self._lock:
            requested = str(topic or "cam_head")
            path = self._observation_paths.get(requested)
            if path is None and requested == "head":
                path = self._observation_paths.get("cam_head")
            if path is None:
                raise KeyError(
                    f"camera {requested!r} is unavailable; available={sorted(self._observation_paths)}"
                )
            return Path(path)

    def submit(self, kind: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
        command = _Command(kind=kind, payload=payload)
        try:
            self._commands.put(command, timeout=1.0)
        except queue.Full as exc:
            raise RuntimeError("harness command queue is busy") from exc
        if not command.done.wait(timeout=timeout):
            raise TimeoutError(f"harness {kind} timed out")
        if command.error:
            raise RuntimeError(command.error)
        return command.result or {}

    def _record_failure(self, provider_id: str, code: str, message: str) -> None:
        failure = {
            "provider_id": provider_id,
            "code": code,
            "message": " ".join(str(message).split())[:1000],
            "observation_id": self._observation_id,
            "time": time.time(),
        }
        with self._lock:
            self._last_failure = failure
            self._last_error = failure["message"]
        self._trace("provider_failure", **failure)

    def _validate_action_chunk(
        self, actions: Any, *, max_actions: int
    ) -> list[dict[str, Any]]:
        if not isinstance(actions, (list, tuple)) or not actions:
            raise ValueError("provider returned an empty or invalid action chunk")
        if len(actions) > max_actions:
            raise ValueError(f"action chunk exceeds the {max_actions}-action limit")
        validated: list[dict[str, Any]] = []
        for action in actions:
            if not isinstance(action, dict) or not action:
                raise ValueError("every action must be a non-empty object")
            self.task_env.validate_action_dict(action)
            if hasattr(self.task_env, "get_action_type"):
                action_type = self.task_env.get_action_type(action)
                if action_type != "joint":
                    raise ValueError("this embodiment only accepts joint actions")
            for key, value in action.items():
                try:
                    numeric = np.asarray(value, dtype=np.float64)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"action field {key!r} is not numeric") from exc
                if not np.isfinite(numeric).all():
                    raise ValueError(f"action field {key!r} contains NaN or infinity")
            validated.append(action)
        return validated

    def _store_chunk(
        self,
        *,
        actions: list[dict[str, Any]],
        source: str,
        provider_id: str,
        subtask_goal: str,
        task_prompt: str | None,
        planning_summary: str | None,
        observation_id: str,
    ) -> dict[str, Any]:
        chunk_id = uuid.uuid4().hex
        with self._lock:
            if self._pending_chunk_id is not None:
                raise ValueError(
                    "an action chunk is already outstanding; execute it before requesting another"
                )
            self._pending_actions = actions
            self._pending_chunk_id = chunk_id
            self._pending_source = source
            self._pending_provider_id = provider_id
            self._pending_created_at = time.monotonic()
            self._action_cursor = 0
            self._last_subtask_goal = subtask_goal
            self._last_task_prompt = task_prompt
            self._last_planning_summary = planning_summary or None
            self._last_error = None
        signatures = [_action_signature(action) for action in actions]
        self._trace(
            "action_chunk_ready",
            observation_id=observation_id,
            official_instruction=(self._observation or {}).get("instruction"),
            chunk_id=chunk_id,
            source=source,
            provider_id=provider_id,
            subtask_goal=subtask_goal,
            optimized_task_prompt=task_prompt,
            planning_summary=planning_summary,
            action_signatures=signatures,
        )
        return {
            "chunk_id": chunk_id,
            "observation_id": observation_id,
            "source": source,
            "provider_id": provider_id,
            "subtask_goal": subtask_goal,
            "action_count": len(actions),
            "action_signatures": signatures,
            "note": "The validated action chunk remains inside the embodiment until execute_action_chunk is called.",
        }

    def _handle_infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            observation = deepcopy(self._observation)
        expected_id = self._require_current_observation(payload.get("observation_id"))
        if observation is None:
            raise ValueError("current observation is unavailable")
        self._require_official_instruction(observation)
        if self.task_env.end_flag[0]:
            raise ValueError("episode has already ended")
        provider_id = " ".join(str(payload.get("provider_id") or "").split())
        if provider_id != self._provider_id:
            raise ValueError(
                f"provider {provider_id!r} is not loaded for this session; "
                f"active provider is {self._provider_id!r}"
            )
        subtask_goal = " ".join(str(payload.get("subtask_goal") or "").split())
        task_prompt = " ".join(
            str(payload.get("optimized_task_prompt") or "").split()
        )
        planning_summary = " ".join(
            str(payload.get("planning_summary") or "").split()
        )
        if not 1 <= len(subtask_goal) <= 1000:
            raise ValueError("subtask_goal must contain 1 to 1000 characters")
        if not 1 <= len(task_prompt) <= 3000:
            raise ValueError(
                "optimized_task_prompt must contain 1 to 3000 characters"
            )
        if len(planning_summary) > 2000:
            raise ValueError("planning_summary exceeds 2000 characters")

        # The VLA contract is prompt + current image/observation. Do not wrap the
        # optimized prompt in adapter-specific instructions or convert it to joints.
        observation["instruction"] = task_prompt
        try:
            self.model_client.call(func_name="update_obs", obs=observation)
            actions = self.model_client.call(func_name="get_action")
            complete_actions = self._validate_action_chunk(
                actions, max_actions=_MAX_POLICY_ACTIONS
            )
        except Exception as exc:
            self._record_failure(provider_id, "provider_inference_failed", str(exc))
            raise
        return self._store_chunk(
            actions=complete_actions,
            source="policy",
            provider_id=provider_id,
            subtask_goal=subtask_goal,
            task_prompt=task_prompt,
            planning_summary=planning_summary,
            observation_id=expected_id,
        )

    def _handle_direct_action(self, payload: dict[str, Any]) -> dict[str, Any]:
        expected_id = self._require_current_observation(payload.get("observation_id"))
        with self._lock:
            self._require_official_instruction(self._observation or {})
        if self.task_env.end_flag[0]:
            raise ValueError("episode has already ended")
        subtask_goal = " ".join(str(payload.get("subtask_goal") or "").split())
        fallback_reason = " ".join(
            str(payload.get("fallback_reason") or "").split()
        )
        rationale = " ".join(str(payload.get("rationale") or "").split())
        if not 1 <= len(subtask_goal) <= 1000:
            raise ValueError("subtask_goal must contain 1 to 1000 characters")
        if not 1 <= len(fallback_reason) <= 1000:
            raise ValueError("fallback_reason must contain 1 to 1000 characters")
        if len(rationale) > 2000:
            raise ValueError("rationale exceeds 2000 characters")
        actions = self._validate_action_chunk(
            payload.get("actions"), max_actions=_MAX_DIRECT_ACTIONS
        )
        return self._store_chunk(
            actions=actions,
            source="llm_direct",
            provider_id="gpt-direct-action",
            subtask_goal=subtask_goal,
            task_prompt=None,
            planning_summary=f"{fallback_reason}: {rationale}".strip(": "),
            observation_id=expected_id or "",
        )

    def _handle_execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        max_actions = payload.get("max_actions")
        discard_remaining = payload.get("discard_remaining", False)
        if max_actions is not None and (
            isinstance(max_actions, bool) or not isinstance(max_actions, int)
            or not 1 <= max_actions <= _MAX_POLICY_ACTIONS
        ):
            raise ValueError("max_actions must be an integer from 1 to 256")
        if not isinstance(discard_remaining, bool):
            raise ValueError("discard_remaining must be a boolean")
        with self._lock:
            chunk_id = self._pending_chunk_id
            cursor = self._action_cursor
            actions = self._pending_actions
            task_prompt = self._last_task_prompt
            source = self._pending_source
            provider_id = self._pending_provider_id
            pending_created_at = self._pending_created_at
        if not chunk_id or payload.get("chunk_id") != chunk_id:
            raise ValueError("chunk_id is stale or unknown")
        if cursor >= len(actions):
            raise ValueError("the pending VLA action chunk is exhausted")
        if (
            pending_created_at is None
            or time.monotonic() - pending_created_at > self._max_chunk_age_s
        ):
            with self._lock:
                self._pending_chunk_id = None
                self._pending_source = None
                self._pending_provider_id = None
                self._pending_created_at = None
            raise ValueError("action chunk exceeded the bounded freshness window")

        executed = 0
        try:
            while cursor < len(actions) and (max_actions is None or executed < max_actions):
                action = actions[cursor]
                self.task_env.take_action(action)
                executed += 1
                cursor += 1
                with self._lock:
                    self._action_cursor = cursor
                if self.task_env.is_episode_end():
                    break
                # Keep the provider's observation current during execution.
                # The planner is called only after this execution window ends.
                if cursor < len(actions) and source == "policy":
                    intermediate_observation = self.task_env.get_obs()
                    intermediate_observation["instruction"] = task_prompt
                    self.model_client.call(
                        func_name="update_obs", obs=intermediate_observation
                    )
        except Exception:
            observation = (
                self.task_env.get_obs_batch(env_idx_list=[0], last_frame=True)[0]
                if self.task_env.is_episode_end()
                else self.task_env.get_obs()
            )
            self._set_observation(observation)
            with self._lock:
                self._pending_chunk_id = None
                self._pending_source = None
                self._pending_provider_id = None
                self._pending_created_at = None
            raise

        if self.task_env.is_episode_end():
            observation = self.task_env.get_obs_batch(env_idx_list=[0], last_frame=True)[0]
        else:
            observation = self.task_env.get_obs()
        self._set_observation(observation)

        episode_end = bool(self.task_env.end_flag[0])
        task_success = bool(self.task_env.success[0]) if episode_end else None
        outcome = (
            "episode_succeeded"
            if task_success is True
            else "episode_failed"
            if episode_end
            else "executed"
        )
        with self._lock:
            remaining = max(0, len(self._pending_actions) - self._action_cursor)
            chunk_complete = remaining == 0
            discarded = remaining if discard_remaining and not episode_end else 0
            released = chunk_complete or episode_end or discard_remaining
            if released:
                self._pending_chunk_id = None
                self._pending_source = None
                self._pending_provider_id = None
                self._pending_created_at = None
                self._pending_actions = []
                self._action_cursor = 0
        self._trace(
            "action_chunk_execution",
            chunk_id=chunk_id,
            source=source,
            provider_id=provider_id,
            executed=executed,
            chunk_complete=chunk_complete,
            max_actions=max_actions,
            discarded_actions=discarded,
            chunk_released=released,
            outcome=outcome,
            observation_id=self._observation_id,
            action_count=int(self.task_env.take_action_cnt[0]),
        )
        return {
            "outcome": outcome,
            "executed": executed,
            "chunk_complete": chunk_complete,
            "remaining_in_chunk": remaining,
            "pending_actions_remaining": 0 if released else remaining,
            "max_actions": max_actions,
            "discarded_actions": discarded,
            "chunk_released": released,
            "action_count": int(self.task_env.take_action_cnt[0]),
            "observation_id": self._observation_id,
            "episode_end": episode_end,
            "task_success": task_success,
            "subtask_goal": self._last_subtask_goal,
            "source": source,
            "provider_id": provider_id,
            "instruction": (
                "This is an execution-window boundary. Inspect the fresh camera frame, "
                "judge whether the current subtask is complete, then continue, "
                "revise, or replace the subtask before requesting the next chunk."
            ),
        }

    def _handle_abort(self, payload: dict[str, Any]) -> dict[str, Any]:
        reason = " ".join(str(payload.get("reason") or "agent_abort").split())[:500]
        self.task_env.success[0] = False
        self.task_env.end_flag[0] = True
        with self._lock:
            self._status = "aborted"
            self._last_error = reason
        self._trace("abort", reason=reason)
        return {"status": "aborted", "reason": reason}

    def _handle_command(self, command: _Command) -> None:
        try:
            if command.kind == "infer":
                command.result = self._handle_infer(command.payload)
            elif command.kind == "direct_action":
                command.result = self._handle_direct_action(command.payload)
            elif command.kind == "execute":
                command.result = self._handle_execute(command.payload)
            elif command.kind == "abort":
                command.result = self._handle_abort(command.payload)
            else:
                raise ValueError(f"unknown harness command: {command.kind}")
        except Exception as exc:
            command.error = str(exc)
            with self._lock:
                self._last_error = command.error
            self._trace("command_error", command=command.kind, error=command.error)
        finally:
            command.done.set()

    def _create_app(self):
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import FileResponse
        from pydantic import BaseModel, ConfigDict, Field

        harness = self
        app = FastAPI(title="awomopaw RoboDojo Interactive Harness", version="1.0")

        class InferRequest(BaseModel):
            model_config = ConfigDict(extra="forbid")
            observation_id: str = Field(min_length=8, max_length=64)
            provider_id: str = Field(min_length=1, max_length=64)
            subtask_goal: str = Field(min_length=1, max_length=1000)
            optimized_task_prompt: str = Field(min_length=1, max_length=3000)
            planning_summary: str = Field(default="", max_length=2000)

        class ExecuteRequest(BaseModel):
            model_config = ConfigDict(extra="forbid")
            chunk_id: str = Field(min_length=8, max_length=64)
            max_actions: int | None = Field(default=None, ge=1, le=_MAX_POLICY_ACTIONS, strict=True)
            discard_remaining: bool = False

        class DirectActionRequest(BaseModel):
            model_config = ConfigDict(extra="forbid")
            observation_id: str = Field(min_length=8, max_length=64)
            subtask_goal: str = Field(min_length=1, max_length=1000)
            fallback_reason: str = Field(min_length=1, max_length=1000)
            rationale: str = Field(default="", max_length=2000)
            actions: list[dict[str, list[float]]] = Field(
                min_length=1, max_length=_MAX_DIRECT_ACTIONS
            )

        class AbortRequest(BaseModel):
            model_config = ConfigDict(extra="forbid")
            reason: str = Field(default="agent_abort", max_length=500)

        @app.get("/health")
        def health() -> dict[str, Any]:
            return harness.public_status()

        @app.get("/v1/session")
        def session_status() -> dict[str, Any]:
            return harness.public_status()

        @app.get("/v1/observations/current")
        def current_observation() -> dict[str, Any]:
            try:
                return harness.observation_metadata()
            except RuntimeError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

        @app.get("/v1/observations/current/image")
        def current_image(topic: str = "cam_head"):
            try:
                return FileResponse(harness.snapshot_path(topic), media_type="image/jpeg")
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc

        @app.post("/v1/policies/chunks")
        @app.post("/v1/vla/chunks", include_in_schema=False)
        def request_policy_chunk(request: InferRequest) -> dict[str, Any]:
            try:
                return harness.submit("infer", request.model_dump(), timeout=180.0)
            except TimeoutError as exc:
                raise HTTPException(status_code=504, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

        @app.post("/v1/actions/direct")
        def request_direct_action(request: DirectActionRequest) -> dict[str, Any]:
            try:
                return harness.submit(
                    "direct_action", request.model_dump(), timeout=15.0
                )
            except TimeoutError as exc:
                raise HTTPException(status_code=504, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

        @app.post("/v1/action-chunks/execute")
        @app.post("/v1/vla/chunks/execute", include_in_schema=False)
        def execute_action_chunk(request: ExecuteRequest) -> dict[str, Any]:
            try:
                return harness.submit("execute", request.model_dump(), timeout=180.0)
            except TimeoutError as exc:
                raise HTTPException(status_code=504, detail=str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

        @app.post("/v1/session/abort")
        def abort_session(request: AbortRequest) -> dict[str, Any]:
            try:
                return harness.submit("abort", request.model_dump(), timeout=15.0)
            except (TimeoutError, RuntimeError) as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

        return app

    def start_server(self) -> None:
        import uvicorn

        config = uvicorn.Config(
            self._create_app(), host=self.host, port=self.port, log_level="info"
        )
        self._server = uvicorn.Server(config)
        self._server_thread = threading.Thread(
            target=self._server.run,
            name="awomopaw-robodojo-control",
            daemon=True,
        )
        self._server_thread.start()

    def stop_server(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._server_thread is not None:
            self._server_thread.join(timeout=5.0)

    def run(self) -> None:
        self.start_server()
        try:
            self.model_client.call(func_name="reset")
            self._set_observation(self.task_env.get_obs())
            self._trace(
                "session_started",
                task=self.task_env.task_name,
                run_id=self.task_env.run_id,
                control_url=f"http://{self.host}:{self.port}",
            )
            while not self.task_env.is_episode_end():
                try:
                    command = self._commands.get(timeout=0.25)
                except queue.Empty:
                    continue
                self._handle_command(command)
            with self._lock:
                if self._status != "aborted":
                    self._status = "completed"
            self._trace(
                "session_finished",
                task_success=bool(self.task_env.success[0]),
                action_count=int(self.task_env.take_action_cnt[0]),
            )
        finally:
            self.stop_server()


def run_interactive_episode(task_env: Any, model_client: Any) -> None:
    """Entry point used by ``Awomopaw_Harness.deploy``."""
    host = os.environ.get("AWOMOPAW_HARNESS_HOST", "127.0.0.1")
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("interactive RoboDojo control must remain loopback-only")
    port = int(os.environ.get("AWOMOPAW_HARNESS_PORT", "8099"))
    InteractiveEpisodeHarness(task_env, model_client, host=host, port=port).run()

