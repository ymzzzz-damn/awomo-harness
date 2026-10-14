"""Offline tests for the generic embodied task orchestrator."""

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from qwenpaw.app.orchestration.engine import (
    OrchestrationEngine,
    OrchestrationError,
)
from qwenpaw.app.orchestration.models import (
    TERMINAL_NODE_PREFIX,
    PlanValidationError,
)
from qwenpaw.app.orchestration.prompt import ORCHESTRATION_CONTEXT
from qwenpaw.app.orchestration.registry import CapabilityRegistry
from qwenpaw.app.orchestration.store import TaskStore


class FakeLoader:
    def get_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "arm.move",
                    "description": "Move an arm",
                    "parameters": {"type": "object", "properties": {}},
                },
                "metadata": {
                    "node_id": "arm",
                    "operation_id": "move",
                    "endpoint": "http://arm",
                    "protocol": "http",
                    "safety_class": "safe",
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "rover.drive",
                    "description": "Drive a rover",
                    "parameters": {"type": "object", "properties": {}},
                },
                "metadata": {
                    "node_id": "rover",
                    "operation_id": "drive",
                    "endpoint": "ws://rover",
                    "protocol": "rosbridge_publish",
                    "safety_class": "safe",
                },
            },
        ]

    @staticmethod
    def status():
        return {"active": ["arm_pkg", "rover_pkg"], "total_tools": 2}


def make_engine(tmp_path: Path, outcomes=None):
    calls = []
    outcomes = list(outcomes or [])

    async def execute(tool, arguments):
        calls.append((tool, arguments))
        if outcomes:
            return outcomes.pop(0)
        return {
            "success": True,
            "value": arguments,
            "state": {"goal_reached": True},
        }

    registry = CapabilityRegistry(FakeLoader())
    registry.replace_external_capabilities(
        [
            {
                "tool": "domain.get_state",
                "api_name": "domain_get_state",
                "assignee": "domain",
                "operation": "get_state",
                "description": "Read domain state",
                "safety_class": "domain_mcp",
                "protocol": "mcp",
                "endpoint_bound": True,
                "availability": "AVAILABLE_REAL",
                "health": "CONNECTED",
                "source": "mcp",
            }
        ]
    )
    engine = OrchestrationEngine(registry, TaskStore(tmp_path), execute)
    return engine, calls


def goal_contract():
    return {
        "objective": "Reach and verify the requested physical end state",
        "required_outcomes": [
            {"id": "goal_reached", "description": "Requested end state is true"}
        ],
    }


def terminal_predicates(*, expected=True):
    return [
        {
            "id": "verify_goal_reached",
            "outcome_id": "goal_reached",
            "description": "Read the domain state after all actions",
            "tool": "domain.get_state",
            "arguments": {"query": "goal_state"},
            "evidence_path": "state.goal_reached",
            "operator": "eq",
            "expected": expected,
        }
    ]


def create_task(engine, goal, nodes, *, request_id=None, expected=True):
    return engine.create_task(
        goal,
        nodes,
        goal_contract(),
        terminal_predicates(expected=expected),
        request_id=request_id,
    )


def test_registry_is_derived_from_active_atomic_tools():
    snapshot = CapabilityRegistry(FakeLoader()).snapshot()
    assert snapshot["capability_count"] == 2
    assert snapshot["by_assignee"] == {"arm": 1, "rover": 1}
    assert snapshot["capabilities"][0]["availability"] == "AVAILABLE_REAL"
    assert snapshot["capabilities"][0]["health"] == "UNKNOWN"
    assert "parameters" not in snapshot["capabilities"][0]
    detailed = CapabilityRegistry(FakeLoader()).snapshot(include_schemas=True)
    assert detailed["capabilities"][0]["parameters"]["type"] == "object"

    registry = CapabilityRegistry(FakeLoader())
    assert registry.terminal_verifier_names() == set()


def test_plan_validation_rejects_unknown_real_tool_and_cycles(tmp_path):
    engine, _ = make_engine(tmp_path)
    with pytest.raises(PlanValidationError, match="unavailable real tool"):
        create_task(engine, "bad", [{"id": "x", "tool": "ghost.move"}])
    with pytest.raises(PlanValidationError, match="unavailable real tool"):
        create_task(
            engine,
            "do not flatten MCP workflow calls into the DAG",
            [{"id": "x", "tool": "domain.get_state"}],
        )
    with pytest.raises(PlanValidationError, match="dependency cycle"):
        create_task(
            engine,
            "cycle",
            [
                {"id": "a", "tool": "arm.move", "depends_on": ["b"]},
                {"id": "b", "tool": "rover.drive", "depends_on": ["a"]},
            ],
        )


@pytest.mark.asyncio
async def test_real_dag_executes_in_order_and_never_replays(tmp_path):
    engine, calls = make_engine(tmp_path)
    created = create_task(
        engine,
        "arm then rover",
        [
            {"id": "pick", "tool": "arm.move", "arguments": {"x": 1}},
            {
                "id": "deliver",
                "tool": "rover.drive",
                "depends_on": ["pick"],
            },
        ],
        request_id="same-chat-request",
    )
    task_id = created["task"]["id"]
    duplicate = create_task(
        engine,
        "ignored duplicate",
        [{"id": "other", "tool": "arm.move"}],
        request_id="same-chat-request",
    )
    assert not duplicate["created"]
    assert duplicate["task"]["id"] == task_id

    first = await engine.execute_next(task_id)
    assert first["node_id"] == "pick"
    with pytest.raises(OrchestrationError, match="will not be replayed"):
        await engine.execute_next(task_id, "pick")
    second = await engine.execute_next(task_id)
    assert second["task"]["status"] == "pending"
    assert second["task"]["nodes"][-1]["node_kind"] == "terminal_verification"
    verified = await engine.execute_next(task_id)
    assert verified["task"]["status"] == "completed"
    assert calls == [
        ("arm.move", {"x": 1}),
        ("rover.drive", {}),
        ("domain.get_state", {"query": "goal_state"}),
    ]

    # A new engine instance sees durable state and still cannot replay hardware.
    restarted, restarted_calls = make_engine(tmp_path)
    assert restarted.get_task(task_id)["status"] == "completed"
    with pytest.raises(OrchestrationError, match="task is completed"):
        await restarted.execute_next(task_id)
    assert restarted_calls == []


@pytest.mark.asyncio
async def test_failure_blocks_dependents_and_replan_preserves_success(tmp_path):
    engine, calls = make_engine(
        tmp_path,
        outcomes=[
            {"success": True},
            {"success": False, "error": "object not visible"},
            {"success": True},
        ],
    )
    task_id = create_task(
        engine,
        "coordinate",
        [
            {"id": "observe", "tool": "arm.move"},
            {"id": "handoff", "tool": "rover.drive", "depends_on": ["observe"]},
            {"id": "finish", "tool": "arm.move", "depends_on": ["handoff"]},
        ],
    )["task"]["id"]
    original_contract = engine.get_task(task_id)["goal_contract"]
    original_predicates = engine.get_task(task_id)["terminal_predicates"]
    await engine.execute_next(task_id)
    failed = await engine.execute_next(task_id)
    assert failed["task"]["status"] == "blocked"
    assert {node["id"]: node["status"] for node in failed["task"]["nodes"]} == {
        "observe": "succeeded",
        "handoff": "failed",
        "finish": "blocked",
        f"{TERMINAL_NODE_PREFIX}verify_goal_reached": "blocked",
    }

    replanned = engine.replan_pending(
        task_id,
        [
            {
                "id": "alternate_finish",
                "tool": "arm.move",
                "depends_on": ["observe"],
            }
        ],
        "rover cannot see the object",
    )
    assert replanned["status"] == "pending"
    assert replanned["goal_contract"] == original_contract
    assert replanned["terminal_predicates"] == original_predicates
    assert replanned["nodes"][-1]["protected"] is True
    assert replanned["nodes"][-1]["depends_on"] == ["observe", "alternate_finish"]
    done = await engine.execute_next(task_id)
    assert done["task"]["status"] == "pending"
    verified = await engine.execute_next(task_id)
    assert verified["task"]["status"] == "completed"
    assert len(calls) == 4


@pytest.mark.asyncio
async def test_simulated_nodes_are_labelled_and_never_call_hardware(tmp_path):
    engine, calls = make_engine(tmp_path)
    task_id = create_task(
        engine,
        "simulate unavailable gripper",
        [
            {
                "id": "sim_grasp",
                "tool": "unavailable.grasp",
                "execution_mode": "simulated",
            }
        ],
    )["task"]["id"]
    result = await engine.execute_next(task_id)
    assert result["requires_simulated_result"]
    assert calls == []
    task = engine.record_simulated_result(
        task_id, "sim_grasp", True, "planner collision check passed"
    )
    assert task["status"] == "pending"
    assert task["nodes"][0]["result"]["simulated"] is True
    verified = await engine.execute_next(task_id)
    assert verified["task"]["status"] == "completed"
    assert calls == [("domain.get_state", {"query": "goal_state"})]


def test_prompt_keeps_safe_shutdown_distinct_from_goal_completion():
    normalized = " ".join(ORCHESTRATION_CONTEXT.split())
    assert "abort_task" in normalized
    assert "safe shutdown completion" in normalized
    assert "absolute base-to-tool" in normalized
    assert "cross-robot or cross-embodiment" in normalized
    assert "JetRover high-level MCP workflow" in normalized
    assert "terminal_predicate" in normalized


@pytest.mark.asyncio
async def test_uncertain_running_node_requires_evidence_and_is_never_replayed(
    tmp_path,
):
    engine, calls = make_engine(tmp_path)
    task_id = create_task(
        engine,
        "recover a crash boundary", [{"id": "move", "tool": "arm.move"}]
    )["task"]["id"]
    task = engine.get_task(task_id)
    old_revision = task["revision"]
    task["status"] = "running"
    task["nodes"][0]["status"] = "running"
    task["nodes"][0]["attempts"] = 1
    engine._touch(task, old_revision)  # pylint: disable=protected-access
    engine.store.save(task, old_revision)

    with pytest.raises(OrchestrationError, match="confirm the physical operation"):
        engine.resolve_uncertain_node(
            task_id, "move", True, "camera shows target reached", False
        )
    resolved = engine.resolve_uncertain_node(
        task_id, "move", True, "camera shows target reached", True
    )
    assert resolved["status"] == "pending"
    assert resolved["nodes"][0]["result"]["externally_resolved"] is True
    verified = await engine.execute_next(task_id)
    assert verified["task"]["status"] == "completed"
    assert calls == [("domain.get_state", {"query": "goal_state"})]


@pytest.mark.asyncio
async def test_actions_cannot_complete_without_satisfying_terminal_evidence(tmp_path):
    engine, calls = make_engine(
        tmp_path,
        outcomes=[
            {"success": True, "accepted": True},
            {"success": True, "state": {"goal_reached": False}},
        ],
    )
    task_id = create_task(
        engine,
        "return rover home",
        [{"id": "return", "tool": "rover.drive"}],
    )["task"]["id"]

    action = await engine.execute_next(task_id)
    assert action["task"]["status"] == "pending"
    verification = await engine.execute_next(task_id)
    assert verification["success"] is False
    assert verification["task"]["status"] == "blocked"
    final = verification["task"]["nodes"][-1]
    assert final["result"]["terminal_validation"]["satisfied"] is False
    assert final["result"]["terminal_validation"]["actual"] is False
    assert calls[-1] == ("domain.get_state", {"query": "goal_state"})


@pytest.mark.asyncio
async def test_replan_cannot_delete_contract_or_complete_with_cleanup_only(tmp_path):
    engine, _ = make_engine(
        tmp_path,
        outcomes=[
            {"success": False, "error": "route unavailable"},
            {"success": True, "stopped": True},
            {"success": True, "state": {"goal_reached": False}},
        ],
    )
    task_id = create_task(
        engine,
        "return rover home",
        [{"id": "return", "tool": "rover.drive"}],
    )["task"]["id"]
    failed = await engine.execute_next(task_id)
    assert failed["task"]["status"] == "blocked"

    replanned = engine.replan_pending(
        task_id,
        [{"id": "safe_stop", "tool": "arm.move"}],
        "navigation failed; stop safely",
    )
    protected_id = f"{TERMINAL_NODE_PREFIX}verify_goal_reached"
    assert any(node["id"] == protected_id for node in replanned["nodes"])
    await engine.execute_next(task_id)
    final = await engine.execute_next(task_id)
    assert final["task"]["status"] == "blocked"


def test_contract_requires_terminal_coverage_and_detects_mutation(tmp_path):
    engine, _ = make_engine(tmp_path)
    with pytest.raises(PlanValidationError, match="every required goal outcome"):
        engine.create_task(
            "two outcomes",
            [{"id": "move", "tool": "arm.move"}],
            {
                "objective": "finish",
                "required_outcomes": [
                    {"id": "first", "description": "first done"},
                    {"id": "second", "description": "second done"},
                ],
            },
            [
                {
                    **terminal_predicates()[0],
                    "outcome_id": "first",
                }
            ],
        )

    task_id = create_task(
        engine,
        "protected",
        [{"id": "move", "tool": "arm.move"}],
    )["task"]["id"]
    task = engine.get_task(task_id)
    revision = task["revision"]
    task["goal_contract"]["objective"] = "silently changed"
    engine._touch(task, revision)  # pylint: disable=protected-access
    engine.store.save(task, revision)
    with pytest.raises(OrchestrationError, match="were modified"):
        engine.replan_pending(task_id, [], "try to erase goal")


def test_legacy_completed_task_is_invalidated_on_store_startup(tmp_path):
    store = TaskStore(tmp_path)
    legacy = {
        "id": "task_legacy",
        "request_id": None,
        "goal": "return rover home",
        "status": "completed",
        "revision": 1,
        "created_at": 1.0,
        "updated_at": 1.0,
        "abort_reason": None,
        "replan_history": [],
        "nodes": [{"id": "drive", "status": "succeeded"}],
    }
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            """
            INSERT INTO orchestration_tasks
                (id, request_id, status, revision, updated_at, payload)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                legacy["id"],
                None,
                legacy["status"],
                legacy["revision"],
                legacy["updated_at"],
                json.dumps(legacy),
            ),
        )

    migrated = TaskStore(tmp_path).get("task_legacy")
    assert migrated["status"] == "blocked"
    assert migrated["revision"] == 2
    assert "legacy completion invalidated" in migrated["blocked_reason"]


@pytest.mark.asyncio
async def test_live_mcp_result_can_satisfy_terminal_predicate(tmp_path):
    from qwenpaw.app.orchestration.tool_adapter import (
        build_orchestration_tool_functions,
    )

    class FakeMCPClient:
        name = "domain"
        is_connected = True
        read_timeout_seconds = 5

        @staticmethod
        async def list_tools():
            return [
                SimpleNamespace(
                    name="domain_goal_state",
                    description="Read final goal state",
                    inputSchema={"type": "object", "properties": {}},
                )
            ]

        @staticmethod
        async def get_callable_function(**_kwargs):
            async def call(**_arguments):
                return SimpleNamespace(
                    content=[SimpleNamespace(text='{"goal_reached": true}')],
                    structuredContent={"goal_reached": True},
                    isError=False,
                    meta=None,
                )

            return call

    functions = build_orchestration_tool_functions(
        FakeLoader(),
        workspace_dir=tmp_path,
        mcp_clients=[FakeMCPClient()],
    )
    by_name = {fn.__orchestration_api_name__: fn for fn in functions}
    created = await by_name["embodied_orchestrator__create_task"](
        goal="verify a simulated stage",
        goal_contract=goal_contract(),
        terminal_predicates=[
            {
                "id": "verify_goal_reached",
                "outcome_id": "goal_reached",
                "description": "Read the domain MCP state",
                "tool": "domain_goal_state",
                "arguments": {},
                "evidence_path": "data.goal_reached",
                "operator": "eq",
                "expected": True,
            }
        ],
        nodes=[
            {
                "id": "simulated_stage",
                "tool": "unavailable.stage",
                "execution_mode": "simulated",
            }
        ],
    )
    task_id = created.metadata["task"]["id"]
    recorded = await by_name["embodied_orchestrator__record_simulated_result"](
        task_id=task_id,
        node_id="simulated_stage",
        success=True,
        evidence="offline stage fixture",
    )
    assert recorded.metadata["task"]["status"] == "pending"
    verified = await by_name["embodied_orchestrator__execute_next"](
        task_id=task_id
    )
    assert verified.metadata["task"]["status"] == "completed"
    assert (
        verified.metadata["task"]["nodes"][-1]["result"][
            "terminal_validation"
        ]["satisfied"]
        is True
    )
