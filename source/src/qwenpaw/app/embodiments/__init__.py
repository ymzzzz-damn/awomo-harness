"""Embodiments integration for awomopaw agent runtime."""

from .manager import EmbodimentManager
from .tool_adapter import build_tool_functions_from_loader

__all__ = ["EmbodimentManager", "build_tool_functions_from_loader"]
