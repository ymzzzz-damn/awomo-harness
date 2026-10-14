"""AgentScope tool bridge for the generic embodied orchestrator."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Callable

from agentscope.message import TextBlock
from agentscope.tool import ToolResponse

from ..embodiments.tool_adapter import make_tool_function
from .engine import OrchestrationEngine
from .registry import CapabilityRegistry
from .store import TaskStore


def _json_response(payload: dict[str, Any], *, success: bool = True) -> ToolResponse:
    return ToolResponse(
        content=[
            TextBlock(
                type="text",
                text=json.dumps(payload, ensure_ascii=False, indent=2),
            )
        ],
        metadata={"success": success, **payload},
    )


def _content_text(response: ToolResponse) -> str:
    parts: list[str] = []
    for block in response.content:
        text = getattr(block, "text", None)
        if text is None and isinstance(block, dict):
            text = block.get("text")
        if text:
            parts.append(str(text))
    return "\n".join(parts)


def _register_schema(
    fn: Callable,
    name: str,
    description: str,
    parameters: dict[str, Any],
) -> Callable:
    fn.__name__ = name
    fn.__doc__ = description
    fn.__annotations__ = {}
    fn.__orchestration_api_name__ = name  # type: ignore[attr-defined]
    fn.__orchestration_schema__ = {  # type: ignore[attr-defined]
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }
    return fn


_NODE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["id", "tool"],
    "properties": {
        "id": {"type": "string", "description": "Unique node id in this task"},
        "description": {"type": "string"},
        "tool": {
            "type": "string",
            "description": "Canonical atomic tool name, for example node.operation",
        },
        "arguments": {"type": "object", "additionalProperties": True},
        "depends_on": {"type": "array", "items": {"type": "string"}},
        "execution_mode": {
            "type": "string",
            "enum": ["real", "simulated"],
            "default": "real",
        },
        "assigned_to": {"type": "string"},
    },
}

_GOAL_CONTRACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["objective", "required_outcomes"],
    "properties": {
        "objective": {
            "type": "string",
            "description": "Immutable statement of the requested end state",
        },
        "required_outcomes": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "description"],
                "properties": {
                    "id": {"type": "string"},
                    "description": {"type": "string"},
                },
            },
        },
        "constraints": {"type": "array", "items": {"type": "string"}},
    },
}

_TERMINAL_PREDICATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "id",
        "outcome_id",
        "description",
        "tool",
        "evidence_path",
        "operator",
    ],
    "properties": {
        "id": {"type": "string"},
        "outcome_id": {
            "type": "string",
            "description": "Required outcome id verified by this predicate",
        },
        "description": {"type": "string"},
        "tool": {
            "type": "string",
            "description": "Real read-only domain MCP state/evidence tool",
        },
        "arguments": {"type": "object", "additionalProperties": True},
        "evidence_path": {
            "type": "string",
            "description": (
                "Dotted path in the domain tool runtime result, for example "
                "data.pose.at_home"
            ),
        },
        "operator": {
            "type": "string",
            "enum": [
                "eq",
                "not_eq",
                "truthy",
                "falsy",
                "gte",
                "lte",
                "contains",
                "in",
                "exists",
            ],
        },
        "expected": {
            "description": "Required for comparison operators; any JSON value"
        },
    },
}


def build_orchestration_tool_functions(
    loader: Any,
    workspace_dir: Path | None = None,
    mcp_clients: list[Any] | None = None,
) -> list[Callable]:
    """Build persistent planning/execution tools for one active loader."""
    registry = CapabilityRegistry(loader)
    runtime_functions: dict[str, Callable] = {}
    mcp_runtime_functions: dict[str, Callable] = {}
    for tool_def in loader.get_tools():
        function = tool_def.get("function") or {}
        metadata = tool_def.get("metadata") or {}
        name = str(function.get("name") or "")
        if not name:
            continue
        runtime_functions[name] = make_tool_function(
            loader=loader,
            tool_name=name,
            description=str(function.get("description") or name),
            parameters=function.get("parameters")
            or {"type": "object", "properties": {}},
            safety_class=str(metadata.get("safety_class") or "safe"),
            workspace_dir=workspace_dir,
        )

    async def refresh_mcp_capabilities() -> list[str]:
        """Refresh live MCP tools and their raw callable functions."""
        external: list[dict[str, Any]] = []
        refreshed_functions: dict[str, Callable] = {}
        errors: list[str] = []
        embodiment_names = set(runtime_functions)
        duplicate_names: set[str] = set()
        for client in mcp_clients or []:
            client_name = str(getattr(client, "name", "mcp"))
            if not bool(getattr(client, "is_connected", False)):
                errors.append(f"{client_name}: disconnected")
                continue
            try:
                tools = await client.list_tools()
                for tool in tools:
                    name = str(getattr(tool, "name", "") or "")
                    if not name or name in embodiment_names:
                        continue
                    if name in refreshed_functions:
                        duplicate_names.add(name)
                        continue
                    function = await client.get_callable_function(
                        func_name=name,
                        wrap_tool_result=False,
                        execution_timeout=getattr(
                            client, "read_timeout_seconds", None
                        ),
                    )
                    refreshed_functions[name] = function
                    external.append(
                        {
                            "tool": name,
                            "api_name": name,
                            "assignee": client_name,
                            "operation": name,
                            "description": str(
                                getattr(tool, "description", "") or ""
                            ),
                            "safety_class": "domain_mcp",
                            "protocol": "mcp",
                            "endpoint_bound": True,
                            "availability": "AVAILABLE_REAL",
                            "health": "CONNECTED",
                            "source": "mcp",
                            "parameters": copy.deepcopy(
                                getattr(tool, "inputSchema", None)
                                or {"type": "object", "properties": {}}
                            ),
                        }
                    )
            except Exception as exc:  # client health is reflected, not hidden
                errors.append(f"{client_name}: {exc}")
        if duplicate_names:
            external = [
                item for item in external if item["tool"] not in duplicate_names
            ]
            for name in duplicate_names:
                refreshed_functions.pop(name, None)
                errors.append(f"duplicate MCP tool name: {name}")
        mcp_runtime_functions.clear()
        mcp_runtime_functions.update(refreshed_functions)
        registry.replace_external_capabilities(external)
        return errors

    def parse_json_output(output: str) -> Any:
        try:
            return json.loads(output)
        except (TypeError, ValueError):
            return None

    def raw_mcp_result(raw: Any, tool_name: str) -> dict[str, Any]:
        content = list(getattr(raw, "content", None) or [])
        text_parts = [
            str(getattr(block, "text", ""))
            for block in content
            if getattr(block, "text", None) is not None
        ]
        output = "\n".join(text_parts)
        structured = getattr(raw, "structuredContent", None)
        if structured is None:
            structured = parse_json_output(output)
        is_error = bool(getattr(raw, "isError", False))
        result: dict[str, Any] = {
            "success": not is_error,
            "tool": tool_name,
            "source": "mcp",
            "output": output,
            "data": copy.deepcopy(structured),
            "metadata": copy.deepcopy(getattr(raw, "meta", None) or {}),
        }
        if is_error:
            result["error"] = output or "domain MCP tool returned an error"
        return result

    async def runtime_execute(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        function = runtime_functions.get(tool_name)
        if function is None:
            function = mcp_runtime_functions.get(tool_name)
            if function is None:
                await refresh_mcp_capabilities()
                function = mcp_runtime_functions.get(tool_name)
            if function is not None:
                try:
                    raw = await function(**arguments)
                    return raw_mcp_result(raw, tool_name)
                except Exception as exc:
                    return {
                        "success": False,
                        "tool": tool_name,
                        "source": "mcp",
                        "error": str(exc),
                    }
        if function is None:
            return {"success": False, "error": f"atomic tool unavailable: {tool_name}"}
        response = await function(**arguments)
        metadata = dict(getattr(response, "metadata", None) or {})
        success = bool(metadata.get("success", False))
        result: dict[str, Any] = {
            "success": success,
            "tool": tool_name,
            "output": _content_text(response),
            "metadata": metadata,
        }
        if not success:
            result["error"] = str(
                metadata.get("error")
                or metadata.get("message")
                or result["output"]
                or "atomic tool execution failed"
            )
        return result

    engine = OrchestrationEngine(
        registry=registry,
        store=TaskStore(workspace_dir),
        runtime_executor=runtime_execute,
    )

    async def inspect_capabilities(
        assignee: str = "",
        include_schemas: bool = False,
    ) -> ToolResponse:
        try:
            mcp_errors = await refresh_mcp_capabilities()
            snapshot = registry.snapshot(
                    assignee.strip() or None,
                    include_schemas=bool(include_schemas),
                )
            snapshot["mcp_refresh_errors"] = mcp_errors
            return _json_response(snapshot)
        except Exception as exc:
            return _json_response({"error": str(exc)}, success=False)

    _register_schema(
        inspect_capabilities,
        "embodied_orchestrator__inspect_capabilities",
        "List currently active atomic embodiment capabilities and their schemas. "
        "Use this before creating a multi-agent task graph.",
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "assignee": {
                    "type": "string",
                    "description": "Optional embodiment node/assignee filter",
                },
                "include_schemas": {
                    "type": "boolean",
                    "default": False,
                    "description": (
                        "Return full parameter schemas only when needed; atomic "
                        "tools already expose these schemas to Chat"
                    ),
                },
            },
        },
    )

    async def create_task(
        goal: str,
        nodes: list[dict[str, Any]],
        goal_contract: dict[str, Any],
        terminal_predicates: list[dict[str, Any]],
        request_id: str = "",
    ) -> ToolResponse:
        try:
            await refresh_mcp_capabilities()
            return _json_response(
                engine.create_task(
                    goal,
                    nodes,
                    goal_contract,
                    terminal_predicates,
                    request_id or None,
                )
            )
        except Exception as exc:
            return _json_response({"error": str(exc)}, success=False)

    _register_schema(
        create_task,
        "embodied_orchestrator__create_task",
        "Validate and persist a cross-agent embodied task DAG with an immutable "
        "goal contract and domain-backed terminal predicates. The engine creates "
        "protected final verification nodes automatically. request_id makes "
        "creation idempotent.",
        {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "goal",
                "goal_contract",
                "terminal_predicates",
                "nodes",
            ],
            "properties": {
                "goal": {"type": "string"},
                "goal_contract": _GOAL_CONTRACT_SCHEMA,
                "terminal_predicates": {
                    "type": "array",
                    "minItems": 1,
                    "items": _TERMINAL_PREDICATE_SCHEMA,
                },
                "request_id": {"type": "string"},
                "nodes": {"type": "array", "minItems": 1, "items": _NODE_SCHEMA},
            },
        },
    )

    async def get_task(task_id: str) -> ToolResponse:
        try:
            return _json_response({"task": engine.get_task(task_id)})
        except Exception as exc:
            return _json_response({"error": str(exc)}, success=False)

    _register_schema(
        get_task,
        "embodied_orchestrator__get_task",
        "Read the durable state, evidence and node history of an embodied task.",
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["task_id"],
            "properties": {"task_id": {"type": "string"}},
        },
    )

    async def execute_next(task_id: str, node_id: str = "") -> ToolResponse:
        try:
            result = await engine.execute_next(task_id, node_id or None)
            return _json_response(result, success=bool(result.get("success", True)))
        except Exception as exc:
            return _json_response({"error": str(exc)}, success=False)

    _register_schema(
        execute_next,
        "embodied_orchestrator__execute_next",
        "Execute one ready real node through its existing atomic embodiment tool. "
        "Never replays succeeded nodes. Simulated nodes are not sent to hardware.",
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["task_id"],
            "properties": {
                "task_id": {"type": "string"},
                "node_id": {"type": "string", "description": "Optional ready node"},
            },
        },
    )

    async def record_simulated_result(
        task_id: str,
        node_id: str,
        success: bool,
        evidence: str,
    ) -> ToolResponse:
        try:
            task = engine.record_simulated_result(
                task_id, node_id, bool(success), evidence
            )
            return _json_response({"task": task})
        except Exception as exc:
            return _json_response({"error": str(exc)}, success=False)

    _register_schema(
        record_simulated_result,
        "embodied_orchestrator__record_simulated_result",
        "Record explicit evidence for one ready simulated node. This never calls hardware.",
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["task_id", "node_id", "success", "evidence"],
            "properties": {
                "task_id": {"type": "string"},
                "node_id": {"type": "string"},
                "success": {"type": "boolean"},
                "evidence": {"type": "string"},
            },
        },
    )

    async def replan_pending(
        task_id: str,
        nodes: list[dict[str, Any]],
        reason: str,
    ) -> ToolResponse:
        try:
            await refresh_mcp_capabilities()
            task = engine.replan_pending(task_id, nodes, reason)
            return _json_response({"task": task})
        except Exception as exc:
            return _json_response({"error": str(exc)}, success=False)

    _register_schema(
        replan_pending,
        "embodied_orchestrator__replan_pending",
        "Replace unfinished work after a failure or world-state change. Successful "
        "action history, the goal contract, terminal predicates, and final "
        "verification nodes cannot be deleted. All terminal evidence is rerun.",
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["task_id", "nodes", "reason"],
            "properties": {
                "task_id": {"type": "string"},
                "reason": {"type": "string"},
                "nodes": {"type": "array", "items": _NODE_SCHEMA},
            },
        },
    )

    async def resolve_uncertain_node(
        task_id: str,
        node_id: str,
        succeeded: bool,
        evidence: str,
        confirmed_not_running: bool,
    ) -> ToolResponse:
        try:
            task = engine.resolve_uncertain_node(
                task_id,
                node_id,
                bool(succeeded),
                evidence,
                bool(confirmed_not_running),
            )
            return _json_response({"task": task})
        except Exception as exc:
            return _json_response({"error": str(exc)}, success=False)

    _register_schema(
        resolve_uncertain_node,
        "embodied_orchestrator__resolve_uncertain_node",
        "Resolve a persisted running node after a service/process crash, using "
        "external evidence and without replaying the physical action.",
        {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "task_id",
                "node_id",
                "succeeded",
                "evidence",
                "confirmed_not_running",
            ],
            "properties": {
                "task_id": {"type": "string"},
                "node_id": {"type": "string"},
                "succeeded": {"type": "boolean"},
                "evidence": {"type": "string"},
                "confirmed_not_running": {
                    "type": "boolean",
                    "description": (
                        "Must be true only after checking that no physical command "
                        "is still in flight"
                    ),
                },
            },
        },
    )

    async def abort_task(task_id: str, reason: str = "") -> ToolResponse:
        try:
            return _json_response({"task": engine.abort(task_id, reason)})
        except Exception as exc:
            return _json_response({"error": str(exc)}, success=False)

    _register_schema(
        abort_task,
        "embodied_orchestrator__abort_task",
        "Abort unfinished task nodes. If hardware is running, call its atomic stop "
        "tool first and then abort the task.",
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["task_id"],
            "properties": {
                "task_id": {"type": "string"},
                "reason": {"type": "string"},
            },
        },
    )

    return [
        inspect_capabilities,
        create_task,
        get_task,
        execute_next,
        record_simulated_result,
        replan_pending,
        resolve_uncertain_node,
        abort_task,
    ]
