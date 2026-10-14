"""Data helpers and validation for generic embodied task graphs."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
import uuid
from typing import Any, Iterable

NODE_TERMINAL = {"succeeded", "failed", "blocked", "skipped"}
TASK_TERMINAL = {"completed", "aborted"}
CONTRACT_VERSION = 1
TERMINAL_NODE_PREFIX = "__terminal_verify__"
TERMINAL_OPERATORS = {
    "eq",
    "not_eq",
    "truthy",
    "falsy",
    "gte",
    "lte",
    "contains",
    "in",
    "exists",
}
_CONTRACT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class PlanValidationError(ValueError):
    """Raised when a proposed orchestration graph is invalid."""


def utc_timestamp() -> float:
    return time.time()


def new_task_id() -> str:
    return f"task_{uuid.uuid4().hex[:16]}"


def validate_goal_contract(raw_contract: dict[str, Any]) -> dict[str, Any]:
    """Normalize the immutable, user-visible definition of task success."""
    if not isinstance(raw_contract, dict):
        raise PlanValidationError("goal_contract must be an object")
    objective = str(raw_contract.get("objective") or "").strip()
    if not objective:
        raise PlanValidationError("goal_contract.objective is required")
    raw_outcomes = raw_contract.get("required_outcomes")
    if not isinstance(raw_outcomes, list) or not raw_outcomes:
        raise PlanValidationError(
            "goal_contract.required_outcomes must be a non-empty array"
        )
    outcomes: list[dict[str, str]] = []
    outcome_ids: set[str] = set()
    for index, item in enumerate(raw_outcomes):
        if not isinstance(item, dict):
            raise PlanValidationError(
                f"goal_contract.required_outcomes[{index}] must be an object"
            )
        outcome_id = str(item.get("id") or "").strip()
        description = str(item.get("description") or "").strip()
        if not _CONTRACT_ID_PATTERN.fullmatch(outcome_id):
            raise PlanValidationError(
                f"invalid goal outcome id: {outcome_id!r}; use letters, digits, _ or -"
            )
        if outcome_id in outcome_ids:
            raise PlanValidationError(f"duplicate goal outcome id: {outcome_id}")
        if not description:
            raise PlanValidationError(
                f"goal outcome '{outcome_id}' requires a description"
            )
        outcome_ids.add(outcome_id)
        outcomes.append({"id": outcome_id, "description": description})

    raw_constraints = raw_contract.get("constraints") or []
    if not isinstance(raw_constraints, list) or not all(
        isinstance(item, str) and item.strip() for item in raw_constraints
    ):
        raise PlanValidationError("goal_contract.constraints must be a string array")
    return {
        "objective": objective,
        "required_outcomes": outcomes,
        "constraints": [item.strip() for item in raw_constraints],
    }


def validate_terminal_predicates(
    raw_predicates: Iterable[dict[str, Any]],
    available_tools: set[str],
    goal_contract: dict[str, Any],
) -> list[dict[str, Any]]:
    """Validate read-only domain evidence checks that gate task completion."""
    if not isinstance(raw_predicates, list) or not raw_predicates:
        raise PlanValidationError("terminal_predicates must be a non-empty array")
    outcome_ids = {
        item["id"] for item in goal_contract.get("required_outcomes", [])
    }
    covered_outcomes: set[str] = set()
    predicate_ids: set[str] = set()
    predicates: list[dict[str, Any]] = []
    operators_requiring_expected = {"eq", "not_eq", "gte", "lte", "contains", "in"}
    for index, item in enumerate(raw_predicates):
        if not isinstance(item, dict):
            raise PlanValidationError(
                f"terminal_predicates[{index}] must be an object"
            )
        predicate_id = str(item.get("id") or "").strip()
        if not _CONTRACT_ID_PATTERN.fullmatch(predicate_id):
            raise PlanValidationError(
                f"invalid terminal predicate id: {predicate_id!r}; "
                "use letters, digits, _ or -"
            )
        if predicate_id in predicate_ids:
            raise PlanValidationError(
                f"duplicate terminal predicate id: {predicate_id}"
            )
        outcome_id = str(item.get("outcome_id") or "").strip()
        if outcome_id not in outcome_ids:
            raise PlanValidationError(
                f"terminal predicate '{predicate_id}' references unknown "
                f"goal outcome '{outcome_id}'"
            )
        description = str(item.get("description") or "").strip()
        if not description:
            raise PlanValidationError(
                f"terminal predicate '{predicate_id}' requires a description"
            )
        tool = str(item.get("tool") or "").strip()
        if tool not in available_tools:
            raise PlanValidationError(
                f"terminal predicate '{predicate_id}' references unavailable "
                f"domain MCP verifier tool '{tool}'"
            )
        arguments = item.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise PlanValidationError(
                f"terminal predicate '{predicate_id}'.arguments must be an object"
            )
        evidence_path = str(item.get("evidence_path") or "").strip()
        if not evidence_path:
            raise PlanValidationError(
                f"terminal predicate '{predicate_id}'.evidence_path is required"
            )
        operator = str(item.get("operator") or "").strip().lower()
        if operator not in TERMINAL_OPERATORS:
            raise PlanValidationError(
                f"terminal predicate '{predicate_id}' has unsupported operator "
                f"'{operator}'"
            )
        if operator in operators_requiring_expected and "expected" not in item:
            raise PlanValidationError(
                f"terminal predicate '{predicate_id}' operator '{operator}' "
                "requires expected"
            )
        predicate = {
            "id": predicate_id,
            "outcome_id": outcome_id,
            "description": description,
            "tool": tool,
            "arguments": copy.deepcopy(arguments),
            "evidence_path": evidence_path,
            "operator": operator,
            "expected": copy.deepcopy(item.get("expected")),
            "verification_node_id": f"{TERMINAL_NODE_PREFIX}{predicate_id}",
        }
        try:
            json.dumps(predicate, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise PlanValidationError(
                f"terminal predicate '{predicate_id}' is not JSON serializable"
            ) from exc
        predicate_ids.add(predicate_id)
        covered_outcomes.add(outcome_id)
        predicates.append(predicate)

    uncovered = outcome_ids - covered_outcomes
    if uncovered:
        raise PlanValidationError(
            "every required goal outcome needs terminal evidence; missing: "
            f"{sorted(uncovered)}"
        )
    return predicates


def contract_fingerprint(
    goal: str,
    goal_contract: dict[str, Any],
    terminal_predicates: list[dict[str, Any]],
) -> str:
    payload = {
        "goal": goal.strip(),
        "goal_contract": goal_contract,
        "terminal_predicates": terminal_predicates,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def make_terminal_verification_nodes(
    predicates: list[dict[str, Any]],
    action_node_ids: list[str],
) -> list[dict[str, Any]]:
    """Create protected final evidence nodes from immutable predicates."""
    return [
        {
            "id": predicate["verification_node_id"],
            "description": f"Final verification: {predicate['description']}",
            "tool": predicate["tool"],
            "arguments": copy.deepcopy(predicate["arguments"]),
            "depends_on": list(action_node_ids),
            "execution_mode": "real",
            "assigned_to": predicate["tool"].split(".", 1)[0],
            "node_kind": "terminal_verification",
            "protected": True,
            "terminal_predicate_id": predicate["id"],
            "status": "pending",
            "attempts": 0,
            "result": None,
            "verification_history": [],
            "error": None,
            "started_at": None,
            "finished_at": None,
        }
        for predicate in predicates
    ]


def validate_plan_nodes(
    raw_nodes: Iterable[dict[str, Any]],
    available_tools: set[str],
    *,
    external_dependencies: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Normalize and validate a directed acyclic task graph.

    Real nodes must reference a currently registered atomic tool. Simulated
    nodes may name a capability which is not present, but are never executed as
    hardware operations.
    """
    external_dependencies = external_dependencies or set()
    nodes: list[dict[str, Any]] = []
    ids: set[str] = set()
    for index, item in enumerate(raw_nodes):
        if not isinstance(item, dict):
            raise PlanValidationError(f"nodes[{index}] must be an object")
        node_id = str(item.get("id") or "").strip()
        if not node_id:
            raise PlanValidationError(f"nodes[{index}].id is required")
        if node_id.startswith(TERMINAL_NODE_PREFIX):
            raise PlanValidationError(
                f"node id '{node_id}' uses the reserved terminal verification prefix"
            )
        if node_id in ids or node_id in external_dependencies:
            raise PlanValidationError(f"duplicate node id: {node_id}")
        ids.add(node_id)
        tool = str(item.get("tool") or "").strip()
        if not tool:
            raise PlanValidationError(f"node '{node_id}' requires a tool/capability")
        mode = str(item.get("execution_mode") or "real").lower()
        if mode not in {"real", "simulated"}:
            raise PlanValidationError(
                f"node '{node_id}' execution_mode must be real or simulated"
            )
        if mode == "real" and tool not in available_tools:
            raise PlanValidationError(
                f"node '{node_id}' references unavailable real tool '{tool}'"
            )
        arguments = item.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise PlanValidationError(f"node '{node_id}'.arguments must be an object")
        depends_on = item.get("depends_on") or []
        if not isinstance(depends_on, list) or not all(
            isinstance(dep, str) and dep for dep in depends_on
        ):
            raise PlanValidationError(
                f"node '{node_id}'.depends_on must be a string array"
            )
        if len(set(depends_on)) != len(depends_on):
            raise PlanValidationError(f"node '{node_id}' has duplicate dependencies")
        nodes.append(
            {
                "id": node_id,
                "description": str(item.get("description") or "").strip(),
                "tool": tool,
                "arguments": copy.deepcopy(arguments),
                "depends_on": list(depends_on),
                "execution_mode": mode,
                "assigned_to": str(item.get("assigned_to") or tool.split(".", 1)[0]),
                "status": "pending",
                "attempts": 0,
                "result": None,
                "error": None,
                "started_at": None,
                "finished_at": None,
            }
        )

    allowed_dependencies = ids | external_dependencies
    for node in nodes:
        unknown = set(node["depends_on"]) - allowed_dependencies
        if unknown:
            raise PlanValidationError(
                f"node '{node['id']}' has unknown dependencies: {sorted(unknown)}"
            )
        if node["id"] in node["depends_on"]:
            raise PlanValidationError(f"node '{node['id']}' depends on itself")

    # Kahn validation, counting persisted successful dependencies as already met.
    indegree = {
        node["id"]: len(set(node["depends_on"]) - external_dependencies)
        for node in nodes
    }
    children: dict[str, list[str]] = {node_id: [] for node_id in ids}
    for node in nodes:
        for dep in node["depends_on"]:
            if dep in ids:
                children[dep].append(node["id"])
    ready = [node_id for node_id, degree in indegree.items() if degree == 0]
    visited = 0
    while ready:
        current = ready.pop()
        visited += 1
        for child in children[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if visited != len(nodes):
        raise PlanValidationError("task graph contains a dependency cycle")
    return nodes


def make_task(
    goal: str,
    nodes: list[dict[str, Any]],
    goal_contract: dict[str, Any],
    terminal_predicates: list[dict[str, Any]],
    *,
    request_id: str | None = None,
) -> dict[str, Any]:
    now = utc_timestamp()
    final_nodes = make_terminal_verification_nodes(
        terminal_predicates,
        [node["id"] for node in nodes],
    )
    return {
        "id": new_task_id(),
        "request_id": request_id,
        "goal": goal.strip(),
        "contract_version": CONTRACT_VERSION,
        "goal_contract": copy.deepcopy(goal_contract),
        "terminal_predicates": copy.deepcopy(terminal_predicates),
        "contract_fingerprint": contract_fingerprint(
            goal,
            goal_contract,
            terminal_predicates,
        ),
        "status": "pending",
        "revision": 1,
        "created_at": now,
        "updated_at": now,
        "abort_reason": None,
        "replan_history": [],
        "nodes": nodes + final_nodes,
    }
