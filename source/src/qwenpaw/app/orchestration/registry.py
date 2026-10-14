"""Capability registry derived from active embodiment packages."""

from __future__ import annotations

import copy
from typing import Any


class CapabilityRegistry:
    """Expose a normalized, queryable view of current atomic tools."""

    def __init__(self, loader: Any) -> None:
        self._loader = loader
        self._external_capabilities: list[dict[str, Any]] = []

    def replace_external_capabilities(
        self, capabilities: list[dict[str, Any]]
    ) -> None:
        """Replace the live non-embodiment capability snapshot (currently MCP)."""
        self._external_capabilities = copy.deepcopy(capabilities)

    def capabilities(
        self,
        assignee: str | None = None,
        *,
        include_schemas: bool = False,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for tool in self._loader.get_tools():
            function = tool.get("function") or {}
            metadata = tool.get("metadata") or {}
            canonical = str(function.get("name") or "")
            if not canonical:
                continue
            node_id = str(metadata.get("node_id") or canonical.split(".", 1)[0])
            if assignee and assignee not in {node_id, canonical.split(".", 1)[0]}:
                continue
            endpoint = metadata.get("endpoint")
            bound = bool(endpoint) and not str(endpoint).startswith("@role:")
            capability = {
                    "tool": canonical,
                    "api_name": canonical.replace(".", "__"),
                    "assignee": node_id,
                    "operation": metadata.get("operation_id"),
                    "description": function.get("description") or "",
                    "safety_class": metadata.get("safety_class", "safe"),
                    "protocol": metadata.get("protocol"),
                    "endpoint_bound": bound,
                    "availability": "AVAILABLE_REAL" if bound else "UNAVAILABLE",
                    "health": "UNKNOWN",
                    "source": "embodiment",
                }
            if include_schemas:
                capability["parameters"] = copy.deepcopy(
                    function.get("parameters")
                    or {"type": "object", "properties": {}}
                )
            result.append(capability)
        for capability in self._external_capabilities:
            if assignee and assignee not in {
                capability.get("assignee"),
                str(capability.get("tool") or "").split(".", 1)[0],
            }:
                continue
            item = copy.deepcopy(capability)
            if not include_schemas:
                item.pop("parameters", None)
            result.append(item)
        return result

    def tool_names(self) -> set[str]:
        return {item["tool"] for item in self.capabilities()}

    def action_tool_names(self) -> set[str]:
        """DAG actions stay on embodiment tools, not internal MCP workflow calls."""
        return {
            item["tool"]
            for item in self.capabilities()
            if item.get("source") == "embodiment"
        }

    def terminal_verifier_names(self) -> set[str]:
        """Only live domain MCP tools may provide completion evidence."""
        return {
            item["tool"]
            for item in self.capabilities()
            if item.get("source") == "mcp"
            and item.get("availability") == "AVAILABLE_REAL"
        }

    def snapshot(
        self,
        assignee: str | None = None,
        *,
        include_schemas: bool = False,
    ) -> dict[str, Any]:
        capabilities = self.capabilities(
            assignee,
            include_schemas=include_schemas,
        )
        by_assignee: dict[str, int] = {}
        for capability in capabilities:
            key = capability["assignee"]
            by_assignee[key] = by_assignee.get(key, 0) + 1
        status = self._loader.status() if hasattr(self._loader, "status") else {}
        return {
            "active_packages": status.get("active", []),
            "capability_count": len(capabilities),
            "by_assignee": by_assignee,
            "capabilities": capabilities,
            "note": (
                "AVAILABLE_REAL means configured and bound; health is UNKNOWN until "
                "a read-only health/state tool is executed. Simulated task nodes are "
                "always labelled and never sent to hardware."
            ),
        }
