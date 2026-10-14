"""Generic DAG executor for active embodiment capabilities."""

from __future__ import annotations

import copy
from typing import Any, Awaitable, Callable

from .models import (
    CONTRACT_VERSION,
    TASK_TERMINAL,
    PlanValidationError,
    contract_fingerprint,
    make_task,
    utc_timestamp,
    validate_goal_contract,
    validate_plan_nodes,
    validate_terminal_predicates,
)
from .registry import CapabilityRegistry
from .store import TaskStore

RuntimeExecutor = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


class OrchestrationError(RuntimeError):
    """Raised for invalid task state transitions or execution requests."""


class OrchestrationEngine:
    """Validate, persist and execute generic multi-embodiment task graphs."""

    def __init__(
        self,
        registry: CapabilityRegistry,
        store: TaskStore,
        runtime_executor: RuntimeExecutor,
    ) -> None:
        self.registry = registry
        self.store = store
        self._runtime_executor = runtime_executor

    def create_task(
        self,
        goal: str,
        nodes: list[dict[str, Any]],
        goal_contract: dict[str, Any],
        terminal_predicates: list[dict[str, Any]],
        request_id: str | None = None,
    ) -> dict[str, Any]:
        if not isinstance(goal, str) or not goal.strip():
            raise PlanValidationError("goal is required")
        if not nodes:
            raise PlanValidationError("at least one task node is required")
        normalized = validate_plan_nodes(nodes, self.registry.action_tool_names())
        normalized_contract = validate_goal_contract(goal_contract)
        normalized_predicates = validate_terminal_predicates(
            terminal_predicates,
            self.registry.terminal_verifier_names(),
            normalized_contract,
        )
        task, created = self.store.create(
            make_task(
                goal,
                normalized,
                normalized_contract,
                normalized_predicates,
                request_id=request_id or None,
            )
        )
        return {"created": created, "task": task}

    def get_task(self, task_id: str) -> dict[str, Any]:
        task = self.store.get(task_id)
        if task is None:
            raise OrchestrationError(f"task not found: {task_id}")
        return task

    @staticmethod
    def _node_map(task: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {node["id"]: node for node in task["nodes"]}

    @staticmethod
    def _is_terminal_verification(node: dict[str, Any]) -> bool:
        return node.get("node_kind") == "terminal_verification"

    @classmethod
    def _action_nodes(cls, task: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            node for node in task["nodes"] if not cls._is_terminal_verification(node)
        ]

    @classmethod
    def _terminal_nodes(cls, task: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            node for node in task["nodes"] if cls._is_terminal_verification(node)
        ]

    @classmethod
    def _require_contract_integrity(cls, task: dict[str, Any]) -> None:
        if task.get("contract_version") != CONTRACT_VERSION:
            raise OrchestrationError(
                "task has no supported immutable goal contract; abort it and "
                "create a new contracted task"
            )
        goal_contract = task.get("goal_contract")
        terminal_predicates = task.get("terminal_predicates")
        if not isinstance(goal_contract, dict) or not isinstance(
            terminal_predicates, list
        ):
            raise OrchestrationError("task goal contract is missing or corrupted")
        expected_fingerprint = contract_fingerprint(
            str(task.get("goal") or ""),
            goal_contract,
            terminal_predicates,
        )
        if task.get("contract_fingerprint") != expected_fingerprint:
            raise OrchestrationError(
                "task goal contract or terminal predicates were modified"
            )

        expected = {
            predicate["verification_node_id"]: predicate
            for predicate in terminal_predicates
        }
        actual = {node["id"]: node for node in cls._terminal_nodes(task)}
        if set(actual) != set(expected):
            raise OrchestrationError(
                "protected terminal verification nodes are missing or replaced"
            )
        for node_id, predicate in expected.items():
            node = actual[node_id]
            immutable_fields = {
                "tool": predicate["tool"],
                "arguments": predicate["arguments"],
                "terminal_predicate_id": predicate["id"],
                "execution_mode": "real",
                "protected": True,
            }
            if any(node.get(key) != value for key, value in immutable_fields.items()):
                raise OrchestrationError(
                    f"protected terminal verification node '{node_id}' was modified"
                )

    @classmethod
    def _ready_nodes(cls, task: dict[str, Any]) -> list[dict[str, Any]]:
        nodes = cls._node_map(task)
        return [
            node
            for node in task["nodes"]
            if node["status"] == "pending"
            and all(nodes[dep]["status"] == "succeeded" for dep in node["depends_on"])
        ]

    @classmethod
    def _block_descendants(cls, task: dict[str, Any], failed_id: str) -> None:
        blocked = {failed_id}
        changed = True
        while changed:
            changed = False
            for node in task["nodes"]:
                if node["status"] == "pending" and blocked.intersection(
                    node["depends_on"]
                ):
                    node["status"] = "blocked"
                    node["error"] = "blocked by failed dependency"
                    blocked.add(node["id"])
                    changed = True

    @classmethod
    def _refresh_task_status(cls, task: dict[str, Any]) -> None:
        statuses = {node["status"] for node in task["nodes"]}
        if statuses == {"succeeded"}:
            terminal_nodes = cls._terminal_nodes(task)
            verified = bool(terminal_nodes) and all(
                node.get("result", {})
                .get("terminal_validation", {})
                .get("satisfied")
                is True
                for node in terminal_nodes
            )
            task["status"] = "completed" if verified else "blocked"
            if not verified:
                task["blocked_reason"] = (
                    "all action nodes ended, but terminal predicates lack "
                    "satisfying domain evidence"
                )
        elif "failed" in statuses or "blocked" in statuses:
            task["status"] = "blocked"
        elif "running" in statuses:
            task["status"] = "running"
        else:
            task["status"] = "pending"

    @staticmethod
    def _read_evidence_path(
        payload: dict[str, Any], path: str
    ) -> tuple[bool, Any]:
        normalized = path.strip()
        if normalized in {"$", ""}:
            return True, payload
        if normalized.startswith("$."):
            normalized = normalized[2:]
        current: Any = payload
        for part in normalized.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
                continue
            if isinstance(current, list) and part.isdigit():
                index = int(part)
                if 0 <= index < len(current):
                    current = current[index]
                    continue
            return False, None
        return True, current

    @classmethod
    def _evaluate_terminal_predicate(
        cls,
        runtime_result: dict[str, Any],
        predicate: dict[str, Any],
    ) -> dict[str, Any]:
        found, actual = cls._read_evidence_path(
            runtime_result,
            predicate["evidence_path"],
        )
        operator = predicate["operator"]
        expected = predicate.get("expected")
        satisfied = False
        try:
            if operator == "exists":
                satisfied = found
            elif not found:
                satisfied = False
            elif operator == "eq":
                satisfied = actual == expected
            elif operator == "not_eq":
                satisfied = actual != expected
            elif operator == "truthy":
                satisfied = bool(actual)
            elif operator == "falsy":
                satisfied = not bool(actual)
            elif operator == "gte":
                satisfied = actual >= expected
            elif operator == "lte":
                satisfied = actual <= expected
            elif operator == "contains":
                satisfied = expected in actual
            elif operator == "in":
                satisfied = actual in expected
        except (TypeError, ValueError):
            satisfied = False
        return {
            "predicate_id": predicate["id"],
            "outcome_id": predicate["outcome_id"],
            "description": predicate["description"],
            "evidence_path": predicate["evidence_path"],
            "operator": operator,
            "expected": copy.deepcopy(expected),
            "evidence_found": found,
            "actual": copy.deepcopy(actual),
            "satisfied": satisfied,
        }

    @staticmethod
    def _predicate_for_node(
        task: dict[str, Any], node: dict[str, Any]
    ) -> dict[str, Any]:
        predicate_id = node.get("terminal_predicate_id")
        for predicate in task["terminal_predicates"]:
            if predicate["id"] == predicate_id:
                return predicate
        raise OrchestrationError(
            f"terminal predicate missing for protected node '{node['id']}'"
        )

    @staticmethod
    def _touch(task: dict[str, Any], old_revision: int) -> None:
        task["revision"] = old_revision + 1
        task["updated_at"] = utc_timestamp()

    def _select_ready_node(
        self, task: dict[str, Any], node_id: str | None
    ) -> dict[str, Any]:
        node_map = self._node_map(task)
        if node_id:
            if node_id not in node_map:
                raise OrchestrationError(f"node not found: {node_id}")
            node = node_map[node_id]
            if node["status"] == "succeeded":
                raise OrchestrationError(
                    f"node '{node_id}' already succeeded and will not be replayed"
                )
            ready_ids = {item["id"] for item in self._ready_nodes(task)}
            if node_id not in ready_ids:
                raise OrchestrationError(
                    f"node '{node_id}' is not ready (status={node['status']})"
                )
            return node
        ready = self._ready_nodes(task)
        if not ready:
            raise OrchestrationError("task has no ready nodes")
        return ready[0]

    async def execute_next(
        self, task_id: str, node_id: str | None = None
    ) -> dict[str, Any]:
        task = self.get_task(task_id)
        self._require_contract_integrity(task)
        if task["status"] in TASK_TERMINAL:
            raise OrchestrationError(f"task is {task['status']}")
        if task["status"] == "blocked":
            raise OrchestrationError("task is blocked; replan pending work first")
        node = self._select_ready_node(task, node_id)
        if node["execution_mode"] == "simulated":
            return {
                "executed": False,
                "requires_simulated_result": True,
                "node": copy.deepcopy(node),
                "message": (
                    "Simulated nodes are never sent to hardware. Evaluate the "
                    "simulation, then call record_simulated_result with evidence."
                ),
            }

        old_revision = task["revision"]
        node["status"] = "running"
        node["attempts"] += 1
        node["started_at"] = utc_timestamp()
        task["status"] = "running"
        self._touch(task, old_revision)
        self.store.save(task, old_revision)

        try:
            runtime_result = await self._runtime_executor(
                node["tool"], copy.deepcopy(node["arguments"])
            )
        except Exception as exc:  # defensive boundary around transports
            runtime_result = {"success": False, "error": str(exc)}

        current = self.get_task(task_id)
        self._require_contract_integrity(current)
        current_node = self._node_map(current)[node["id"]]
        if current_node["status"] != "running":
            raise OrchestrationError(
                "node state changed while its physical operation was running"
            )
        old_revision = current["revision"]
        current_node["finished_at"] = utc_timestamp()
        current_node["result"] = copy.deepcopy(runtime_result)
        if runtime_result.get("success"):
            if self._is_terminal_verification(current_node):
                predicate = self._predicate_for_node(current, current_node)
                validation = self._evaluate_terminal_predicate(
                    runtime_result,
                    predicate,
                )
                current_node["result"] = {
                    **copy.deepcopy(runtime_result),
                    "terminal_validation": validation,
                }
                if validation["satisfied"]:
                    current_node["status"] = "succeeded"
                    current_node["error"] = None
                else:
                    current_node["status"] = "failed"
                    current_node["error"] = (
                        f"terminal predicate '{predicate['id']}' was not satisfied"
                    )
            else:
                current_node["status"] = "succeeded"
                current_node["error"] = None
        else:
            current_node["status"] = "failed"
            current_node["error"] = str(
                runtime_result.get("error") or "atomic tool execution failed"
            )
            self._block_descendants(current, current_node["id"])
        self._refresh_task_status(current)
        self._touch(current, old_revision)
        self.store.save(current, old_revision)
        return {
            "executed": True,
            "node_id": current_node["id"],
            "execution_mode": "real",
            "success": current_node["status"] == "succeeded",
            "runtime_result": runtime_result,
            "task": current,
        }

    def record_simulated_result(
        self,
        task_id: str,
        node_id: str,
        success: bool,
        evidence: str,
    ) -> dict[str, Any]:
        if not evidence.strip():
            raise OrchestrationError("simulated result requires explicit evidence")
        task = self.get_task(task_id)
        self._require_contract_integrity(task)
        if task["status"] in TASK_TERMINAL:
            raise OrchestrationError(f"task is {task['status']}")
        node = self._select_ready_node(task, node_id)
        if node["execution_mode"] != "simulated":
            raise OrchestrationError("real nodes must be executed by execute_next")
        old_revision = task["revision"]
        node["attempts"] += 1
        node["started_at"] = node["started_at"] or utc_timestamp()
        node["finished_at"] = utc_timestamp()
        node["result"] = {
            "success": bool(success),
            "simulated": True,
            "evidence": evidence,
        }
        node["status"] = "succeeded" if success else "failed"
        node["error"] = None if success else "simulated step failed"
        if not success:
            self._block_descendants(task, node_id)
        self._refresh_task_status(task)
        self._touch(task, old_revision)
        self.store.save(task, old_revision)
        return task

    def replan_pending(
        self,
        task_id: str,
        nodes: list[dict[str, Any]],
        reason: str,
    ) -> dict[str, Any]:
        if not reason.strip():
            raise OrchestrationError("replan reason is required")
        task = self.get_task(task_id)
        self._require_contract_integrity(task)
        if task["status"] in TASK_TERMINAL:
            raise OrchestrationError(f"task is {task['status']}")
        if any(node["status"] == "running" for node in task["nodes"]):
            raise OrchestrationError("cannot replan while a physical node is running")
        terminal_nodes = self._terminal_nodes(task)
        succeeded = [
            node
            for node in self._action_nodes(task)
            if node["status"] == "succeeded"
        ]
        succeeded_ids = {node["id"] for node in succeeded}
        replacement = validate_plan_nodes(
            nodes,
            self.registry.action_tool_names(),
            external_dependencies=succeeded_ids,
        )
        old_revision = task["revision"]
        replaced_nodes = [
            copy.deepcopy(node)
            for node in self._action_nodes(task)
            if node["status"] != "succeeded"
        ]
        task["replan_history"].append(
            {
                "at": utc_timestamp(),
                "reason": reason,
                "replaced_node_ids": [node["id"] for node in replaced_nodes],
                "replaced_nodes": replaced_nodes,
                "terminal_verification_reset": True,
            }
        )
        action_nodes = succeeded + replacement
        action_node_ids = [node["id"] for node in action_nodes]
        for node in terminal_nodes:
            if node["status"] != "pending" or node.get("result") is not None:
                node.setdefault("verification_history", []).append(
                    {
                        "status": node["status"],
                        "attempts": node["attempts"],
                        "result": copy.deepcopy(node.get("result")),
                        "error": node.get("error"),
                        "started_at": node.get("started_at"),
                        "finished_at": node.get("finished_at"),
                    }
                )
            node["depends_on"] = list(action_node_ids)
            node["status"] = "pending"
            node["result"] = None
            node["error"] = None
            node["started_at"] = None
            node["finished_at"] = None
        task["nodes"] = action_nodes + terminal_nodes
        task.pop("blocked_reason", None)
        self._refresh_task_status(task)
        self._touch(task, old_revision)
        self.store.save(task, old_revision)
        return task

    def resolve_uncertain_node(
        self,
        task_id: str,
        node_id: str,
        succeeded: bool,
        evidence: str,
        confirmed_not_running: bool,
    ) -> dict[str, Any]:
        """Settle a persisted running node after a crash without replaying it."""
        if not confirmed_not_running:
            raise OrchestrationError(
                "confirm the physical operation is no longer running before resolution"
            )
        if not evidence.strip():
            raise OrchestrationError("uncertain node resolution requires evidence")
        task = self.get_task(task_id)
        self._require_contract_integrity(task)
        if task["status"] in TASK_TERMINAL:
            raise OrchestrationError(f"task is {task['status']}")
        node = self._node_map(task).get(node_id)
        if node is None:
            raise OrchestrationError(f"node not found: {node_id}")
        if node["status"] != "running":
            raise OrchestrationError(
                f"node '{node_id}' is not uncertain/running (status={node['status']})"
            )
        if self._is_terminal_verification(node) and succeeded:
            raise OrchestrationError(
                "terminal verification can only succeed from fresh structured "
                "domain MCP evidence; resolve it as failed, then replan and rerun"
            )
        old_revision = task["revision"]
        node["finished_at"] = utc_timestamp()
        node["result"] = {
            "success": bool(succeeded),
            "externally_resolved": True,
            "evidence": evidence,
        }
        node["status"] = "succeeded" if succeeded else "failed"
        node["error"] = None if succeeded else "uncertain physical step resolved as failed"
        if not succeeded:
            self._block_descendants(task, node_id)
        self._refresh_task_status(task)
        self._touch(task, old_revision)
        self.store.save(task, old_revision)
        return task

    def abort(self, task_id: str, reason: str) -> dict[str, Any]:
        task = self.get_task(task_id)
        if task["status"] in TASK_TERMINAL:
            return task
        if any(node["status"] == "running" for node in task["nodes"]):
            raise OrchestrationError(
                "a physical node is running; call the embodiment stop tool before aborting"
            )
        old_revision = task["revision"]
        for node in task["nodes"]:
            if node["status"] in {"pending", "failed", "blocked"}:
                node["status"] = "skipped"
                node["error"] = "task aborted"
        task["status"] = "aborted"
        task["abort_reason"] = reason.strip() or "aborted by user/agent"
        self._touch(task, old_revision)
        self.store.save(task, old_revision)
        return task
