# -*- coding: utf-8 -*-
"""Memory backend that disables persistent memory services."""

from collections.abc import Callable

from agentscope.tool import ToolResponse

from .base_memory_manager import BaseMemoryManager, memory_registry


@memory_registry.register("none")
class NullMemoryManager(BaseMemoryManager):
    """Keep only the agent's in-session context and expose no memory tools."""

    async def start(self) -> None:
        return None

    async def close(self) -> bool:
        return True

    def get_memory_prompt(self, language: str = "zh") -> str:
        return ""

    def list_memory_tools(self) -> list[Callable[..., ToolResponse]]:
        return []
