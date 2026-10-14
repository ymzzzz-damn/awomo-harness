"""Per-task authorization state for physical embodiment actions."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator


_approved_physical_tool: ContextVar[str | None] = ContextVar(
    "approved_physical_tool", default=None
)


def is_physical_action_approved(tool_name: str) -> bool:
    """Return whether the current tool task received explicit approval."""
    return _approved_physical_tool.get() == tool_name


@contextmanager
def physical_action_approval(tool_name: str) -> Iterator[None]:
    """Authorize physical execution only for the current context."""
    token = _approved_physical_tool.set(tool_name)
    try:
        yield
    finally:
        _approved_physical_tool.reset(token)
