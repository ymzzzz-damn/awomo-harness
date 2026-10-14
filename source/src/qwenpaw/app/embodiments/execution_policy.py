"""Workspace test/maintenance interlock, outside immutable device atomics."""
from __future__ import annotations

import json
import math
import fcntl
import re
import time
from pathlib import Path


_FRANKA_ALWAYS_AVAILABLE = {
    "health_check",
    "grasp_status",
    "stop",
    "cancel_grasp",
    "disconnect",
}


def execution_block_reason(workspace_dir: Path | None, tool: str) -> str | None:
    name = tool.replace(".", "__")
    # Stops and read-only feedback must remain callable during an interlock.
    operation = name.split("__")[-1]
    if operation.startswith(("stop", "emergency_stop", "get_", "read_", "capture_")):
        return None
    if name.startswith("franka_") and operation in _FRANKA_ALWAYS_AVAILABLE:
        return None
    if workspace_dir is None:
        if name.startswith("franka_"):
            return "FRANKA_EXECUTION_BLOCKED: no workspace execution policy"
        return None
    path = Path(workspace_dir) / "robot_execution_policy.json"
    if not path.exists():
        if name.startswith("franka_"):
            return "FRANKA_EXECUTION_BLOCKED: robot_execution_policy.json is missing"
        return None
    try:
        policy = json.loads(path.read_text())
    except (OSError, ValueError):
        return "ROBOT_EXECUTION_BLOCKED: execution policy is unreadable"
    if not isinstance(policy, dict):
        return "ROBOT_EXECUTION_BLOCKED: execution policy must be a JSON object"
    if name.startswith(("jetrover_arm__", "jetrover_gripper__")) and not policy.get("manipulation_enabled", True):
        return "MANIPULATION_DISABLED: arm is under maintenance; planning/offline validation only"
    if name.startswith("jetrover_base__") and not policy.get("base_motion_enabled", True):
        return "BASE_MOTION_DISABLED: read-only chat acceptance mode; do not report motion as executed"
    if name.startswith("franka_"):
        if policy.get("franka_enabled") is not True:
            return "FRANKA_DISABLED: explicit franka_enabled=true is required"
        if policy.get("manipulation_enabled") is not True:
            return "MANIPULATION_DISABLED: explicit manipulation_enabled=true is required"
        mode = policy.get("franka_execution_mode")
        if mode not in {"committed_grasp_only", "atomic_supervised"}:
            return (
                "FRANKA_EXECUTION_BLOCKED: franka_execution_mode must be "
                "committed_grasp_only or atomic_supervised"
            )
        allowed = policy.get("franka_allowed_operations")
        if not isinstance(allowed, list) or operation not in allowed:
            if mode == "atomic_supervised":
                return (
                    "FRANKA_ATOMIC_ACTION_BLOCKED: operation is not in the "
                    "explicit atomic allowlist"
                )
            return (
                "RAW_FRANKA_ACTION_BLOCKED: only a policy-whitelisted, "
                "bridge-committed high-level operation may actuate Franka"
            )
    return None


def execution_argument_block_reason(tool: str, arguments: dict) -> str | None:
    """Validate the opaque bridge grant before any high-level dispatch."""
    name = tool.replace(".", "__")
    if name != "franka_fr3_controller__execute_committed_grasp":
        return None
    request_id = str(arguments.get("request_id") or "")
    plan_id = str(arguments.get("plan_id") or "")
    token = str(arguments.get("commit_token") or "")
    authorization = arguments.get("authorization")
    authorization = authorization if isinstance(authorization, dict) else {}
    authorization_hash = str(authorization.get("prompt_hash") or "")
    if not re.fullmatch(r"[A-Za-z0-9._:-]{8,128}", request_id):
        return "FRANKA_EXECUTION_BLOCKED: invalid idempotent request_id"
    if not re.fullmatch(r"[A-Za-z0-9._:-]{8,128}", plan_id):
        return "FRANKA_EXECUTION_BLOCKED: invalid committed plan_id"
    if not re.fullmatch(r"[A-Za-z0-9_-]{24,256}", token):
        return "FRANKA_EXECUTION_BLOCKED: invalid single-use commit token"
    if authorization.get("action") != "execute_grasp":
        return "FRANKA_EXECUTION_BLOCKED: invalid authorization action"
    if authorization.get("plan_id") != plan_id:
        return "FRANKA_EXECUTION_BLOCKED: authorization plan_id mismatch"
    if not re.fullmatch(r"[0-9a-f]{64}", authorization_hash):
        return "FRANKA_EXECUTION_BLOCKED: invalid user-command authorization hash"
    issued_at_ms = authorization.get("issued_at_ms")
    expires_at_ms = authorization.get("expires_at_ms")
    if (
        isinstance(issued_at_ms, bool)
        or not isinstance(issued_at_ms, int)
        or isinstance(expires_at_ms, bool)
        or not isinstance(expires_at_ms, int)
    ):
        return "FRANKA_EXECUTION_BLOCKED: invalid authorization timestamps"
    if not 0 < expires_at_ms - issued_at_ms <= 5_000:
        return "FRANKA_EXECUTION_BLOCKED: authorization TTL exceeds 5 seconds"
    now_ms = int(time.time() * 1000)
    if issued_at_ms > now_ms or now_ms > expires_at_ms:
        return "FRANKA_EXECUTION_BLOCKED: authorization is not currently valid"
    if not re.fullmatch(r"[0-9a-f]{32}", str(authorization.get("nonce") or "")):
        return "FRANKA_EXECUTION_BLOCKED: invalid authorization nonce"
    if not re.fullmatch(
        r"[0-9a-f]{64}", str(authorization.get("signature") or "")
    ):
        return "FRANKA_EXECUTION_BLOCKED: invalid authorization signature"
    if set(authorization) != {
        "action",
        "plan_id",
        "prompt_hash",
        "issued_at_ms",
        "expires_at_ms",
        "nonce",
        "signature",
    }:
        return "FRANKA_EXECUTION_BLOCKED: unexpected authorization fields"
    return None


def reserve_test_motion(workspace_dir: Path | None, tool: str, arguments: dict) -> str | None:
    """Optional finite hardware-smoke budget; absent in normal deployment."""
    if workspace_dir is None or tool.replace(".", "__") != "jetrover_base__set_base_twist":
        return None
    path = Path(workspace_dir) / "robot_execution_policy.json"
    if not path.exists():
        return None
    with path.with_suffix(".lock").open("w") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        try:
            policy = json.loads(path.read_text())
            budget = policy.get("test_twist_calls_remaining")
            if budget is None:
                return None
            x = float(arguments.get("linear_x_mps", 0))
            y = float(arguments.get("linear_y_mps", 0))
            omega = float(arguments.get("angular_z_rps", 0))
            duration = float(arguments.get("duration_seconds", 0))
            if not all(math.isfinite(v) for v in (x,y,omega,duration)):
                return "TEST_MOTION_LIMIT: nonfinite command"
            if budget <= 0:
                return "BASE_MOTION_DISABLED: finite hardware test budget exhausted"
            if x != 0 or y != 0 or abs(omega) > .30 or not 0 < duration <= 2.0:
                return "TEST_MOTION_LIMIT: this smoke test permits only one short in-place turn"
            policy["test_twist_calls_remaining"] = int(budget) - 1
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(policy, ensure_ascii=False, indent=2))
            temporary.replace(path)
        except (OSError, ValueError, TypeError):
            return "TEST_MOTION_LIMIT: invalid test policy"
    return None
