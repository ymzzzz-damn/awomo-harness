"""Bridge embodiment tool definitions to executable AgentScope tools."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict

from agentscope.message import TextBlock
from agentscope.tool import ToolResponse

from .approval import is_physical_action_approved
from .executor import EmbodimentExecutor
from .execution_policy import (
    execution_argument_block_reason,
    execution_block_reason,
    reserve_test_motion,
)

if TYPE_CHECKING:
    from embodiments.runtime.loader import EmbodimentLoader

logger = logging.getLogger(__name__)

_TOOL_SAFETY: dict[str, str] = {}

_LOCAL_NAVIGATION_TASK_TOOLS = {
    "jetrover_navigation_create_mapped_task",
    "jetrover_navigation_create_online_vln_task",
}
_LOCAL_NAVIGATION_EXECUTE_TOOL = "jetrover_navigation_execute_planned_segment"
_FRANKA_TOOL_PREFIX = "franka_fr3_controller__"


def _api_safe_tool_name(tool_name: str) -> str:
    """Convert an embodiment tool name to the OpenAI function-name grammar.

    EmbodimentLoader names tools as ``node.operation`` so that dispatch can
    split the name into its node and operation. OpenAI-compatible APIs reject
    the dot, however, and only allow letters, digits, ``_`` and ``-``.
    Double underscore keeps the node/operation boundary visible while leaving
    the loader's canonical name unchanged.
    """
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", tool_name.replace(".", "__"))
    return safe_name or "embodiment_tool"


def get_embodiment_tool_safety(tool_name: str) -> str | None:
    """Return the registered safety class for an embodiment tool."""
    if _api_safe_tool_name(tool_name) in {
        *_LOCAL_NAVIGATION_TASK_TOOLS,
        _LOCAL_NAVIGATION_EXECUTE_TOOL,
    }:
        return "supervisor"
    return _TOOL_SAFETY.get(tool_name)


def embodiment_tool_requires_approval(
    tool_name: str,
    arguments: Dict[str, Any] | None = None,
) -> bool:
    """Return whether this invocation needs explicit human approval.

    Existing JetRover operations keep their unattended local-experiment
    behavior. FR3 operations are handled more conservatively: every operation
    marked ``confirmation`` or ``supervisor`` must pass the tool-guard approval
    flow. Read-only checks, stop, and disconnect remain immediately available.
    """
    api_name = _api_safe_tool_name(tool_name)
    safety_class = get_embodiment_tool_safety(api_name)
    if api_name.startswith(_FRANKA_TOOL_PREFIX):
        return safety_class in {"confirmation", "supervisor"}
    # Keep the safety class metadata visible to diagnostics, but auto-authorize
    # existing non-FR3 invocations in this explicitly unattended deployment.
    # Malformed parameters are still rejected before transport.
    del arguments, api_name, safety_class
    return False


def make_tool_function(
    loader: "EmbodimentLoader",
    tool_name: str,
    description: str,
    parameters: Dict[str, Any],
    safety_class: str = "safe",
    workspace_dir: Path | None = None,
) -> Callable:
    """Return an async callable that safely dispatches and executes a tool."""
    executor = EmbodimentExecutor(workspace_dir)
    api_tool_name = _api_safe_tool_name(tool_name)
    _TOOL_SAFETY[api_tool_name] = safety_class

    async def _tool(**kwargs: Any) -> ToolResponse:
        blocked = execution_block_reason(workspace_dir, tool_name)
        blocked = blocked or execution_argument_block_reason(tool_name, kwargs)
        blocked = blocked or reserve_test_motion(workspace_dir, tool_name, kwargs)
        if blocked:
            return ToolResponse(content=[TextBlock(type="text", text=blocked)],
                                metadata={"success": False, "executed": False, "error": blocked})
        # ToolGuard sees the OpenAI-safe name, while the loader dispatches the
        # canonical dotted name. Accept either spelling for this one scoped
        # approval so a confirmed FR3 action reaches the loader exactly once.
        approved = (
            is_physical_action_approved(tool_name)
            or is_physical_action_approved(api_tool_name)
            or not embodiment_tool_requires_approval(tool_name, kwargs)
        )
        dispatch = loader.dispatch_tool(tool_name, kwargs, approved=approved)
        if not dispatch.get("success"):
            if dispatch.get("requires_approval"):
                text = (
                    "Physical action blocked: explicit user approval is required "
                    "before this operation can execute."
                )
            else:
                text = f"Device operation rejected: {dispatch.get('error', 'unknown error')}"
            return ToolResponse(
                content=[TextBlock(type="text", text=text)],
                metadata=dispatch,
            )

        if api_tool_name == "jetrover_base__set_base_twist":
            # LLM latency between enable and twist exceeds the vehicle's 1 s
            # watchdog. Bind the existing enable primitive to the same socket
            # as this finite command; keep the vehicle watchdog unchanged.
            enable = loader.dispatch_tool("jetrover_base.enable_motion", {}, approved=approved)
            if not enable.get("success"):
                return ToolResponse(content=[TextBlock(type="text", text="Motion enable dispatch rejected")],
                                    metadata={"success": False, "executed": False})
            dispatch = {**dispatch, "motion_enable": enable}
        return await executor.execute(dispatch)

    # Keep the canonical loader name for local dispatch and backwards
    # compatibility. The OpenAI-facing schema below uses the sanitized name.
    _tool.__name__ = tool_name
    _tool.__doc__ = description or tool_name
    _tool.__annotations__ = {}
    _tool.__embodiment_parameters__ = parameters  # type: ignore[attr-defined]
    _tool.__embodiment_api_name__ = api_tool_name  # type: ignore[attr-defined]
    _tool.__embodiment_safety_class__ = safety_class  # type: ignore[attr-defined]
    _tool.__embodiment_schema__ = {  # type: ignore[attr-defined]
        "type": "function",
        "function": {
            "name": api_tool_name,
            "description": description or tool_name,
            "parameters": parameters,
        },
    }
    return _tool


def build_tool_functions_from_loader(
    loader: "EmbodimentLoader",
    workspace_dir: Path | None = None,
) -> list[Callable]:
    """Build executable callable tools from all active packages."""
    result = []
    for tool_def in loader.get_tools():
        fn = tool_def.get("function", {})
        metadata = tool_def.get("metadata", {})
        name = fn.get("name", "")
        if not name:
            continue
        result.append(
            make_tool_function(
                loader=loader,
                tool_name=name,
                description=fn.get("description", ""),
                parameters=fn.get(
                    "parameters", {"type": "object", "properties": {}}
                ),
                safety_class=metadata.get("safety_class", "safe"),
                workspace_dir=workspace_dir,
            )
        )
    return result
