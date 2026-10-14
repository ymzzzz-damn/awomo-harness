"""EmbodimentManager: loads and activates embodiment packages for an agent."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from ...config.config import EmbodimentsConfig

logger = logging.getLogger(__name__)


class EmbodimentManager:
    """Manages embodiment packages for a single agent workspace.

    Mirrors MCPClientManager: loaded at workspace startup (priority 20),
    provides activated packages to the runner for injection into the agent.
    """

    def __init__(self) -> None:
        self._loader: Optional[object] = None  # EmbodimentLoader when available
        self._active: bool = False
        self._workspace_dir: Optional[Path] = None

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def init_from_config(
        self,
        config: "EmbodimentsConfig",
        workspace_dir: Optional[Path] = None,
    ) -> None:
        """Discover and activate all enabled packages from config."""
        self._workspace_dir = workspace_dir
        try:
            from embodiments.runtime.loader import EmbodimentLoader
        except ImportError:
            logger.info(
                "embodiments package not installed; skipping embodiment loading"
            )
            return

        loader = EmbodimentLoader()

        for key, pkg_cfg in config.packages.items():
            if not pkg_cfg.enabled:
                logger.debug("Embodiment package '%s' is disabled, skipping", key)
                continue

            pkg_path = Path(pkg_cfg.path).expanduser()
            if not pkg_path.exists():
                logger.warning(
                    "Embodiment package path does not exist: %s (key=%s)",
                    pkg_path,
                    key,
                )
                continue

            from embodiments.runtime.loader import EmbodimentPackage

            pkg = EmbodimentPackage(pkg_path)
            loader._available[pkg.package_id] = pkg  # pylint: disable=protected-access

            try:
                loader.activate(pkg.package_id, bindings=pkg_cfg.bindings or {})
                logger.info(
                    "Activated embodiment package '%s' (v%s) from %s",
                    pkg.package_id,
                    pkg.version,
                    pkg_path,
                )
            except ValueError as exc:
                logger.warning(
                    "Cannot activate embodiment '%s': %s", pkg.package_id, exc
                )

        self._loader = loader
        self._active = True

    async def close(self) -> None:
        self._loader = None
        self._active = False

    # ── Agent System Interface ───────────────────────────────────────────────

    def get_loader(self):
        """Return the underlying EmbodimentLoader, or None if not active."""
        return self._loader

    @property
    def workspace_dir(self) -> Optional[Path]:
        """Directory used for device artifacts such as camera captures."""
        return self._workspace_dir

    def get_combined_context(self) -> str:
        """Return concatenated EMBODIMENT.md from all active packages."""
        if self._loader is None:
            return ""
        return self._loader.get_context()  # type: ignore[union-attr]

    def get_active_tool_count(self) -> int:
        if self._loader is None:
            return 0
        return len(self._loader.get_tools())  # type: ignore[union-attr]

    @property
    def is_active(self) -> bool:
        return self._active and self._loader is not None
