# -*- coding: utf-8 -*-
"""API routes for Embodiment package management."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, Path as FPath, Request
from pydantic import BaseModel, Field

from ...constant import WORKING_DIR
from ..utils import schedule_agent_reload

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/embodiments", tags=["embodiments"])

# ── Pydantic models ────────────────────────────────────────────────────────


class EmbodimentPackageInfo(BaseModel):
    key: str
    package_id: str
    path: str
    name: str = Field(default="")
    version: str = Field(default="")
    description: str = Field(default="")
    enabled: bool
    bindings: Dict[str, str] = Field(default_factory=dict)
    kind: str = Field(default="")
    tool_count: int = Field(default=0)
    path_exists: bool = Field(default=True)
    safety_profile: str = Field(default="")
    hardware_roles: List[str] = Field(default_factory=list)


class EmbodimentCreateRequest(BaseModel):
    package_id: str
    path: str
    description: str = ""
    enabled: bool = True
    bindings: Dict[str, str] = Field(default_factory=dict)


class EmbodimentUpdateRequest(BaseModel):
    package_id: Optional[str] = None
    path: Optional[str] = None
    description: Optional[str] = None
    enabled: Optional[bool] = None
    bindings: Optional[Dict[str, str]] = None


class EmbodimentToolInfo(BaseModel):
    name: str
    description: str
    method: str
    path: str
    safety_class: str
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    node_id: str
    node_name: str
    latency_estimate_ms: Optional[int] = None


class HardwareRequirement(BaseModel):
    role: str
    description: str
    device_type: str
    hint: str
    bound_endpoint: str = ""
    default_port: Optional[int] = None


class EmbodimentDiscoveryEntry(BaseModel):
    package_id: str
    name: str
    path: str
    version: str
    kind: str
    description: str
    hardware_roles: List[str]
    already_configured: bool
    config_key: Optional[str] = None
    hardware_detected: bool = False
    probe_info: Dict[str, Any] = Field(default_factory=dict)


# ── Helpers ────────────────────────────────────────────────────────────────


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _build_package_info(key: str, pkg_cfg) -> EmbodimentPackageInfo:
    expanded = Path(pkg_cfg.path).expanduser()
    path_exists = expanded.exists()
    version = name = kind = safety_profile = ""
    tool_count = 0
    hardware_roles: List[str] = []

    if path_exists:
        yaml_path = expanded / "embodiment.yaml"
        if yaml_path.exists():
            data = _read_yaml(yaml_path)
            version = data.get("version", "")
            name = data.get("name", "")
            kind = data.get("kind", "")
            safety_profile = data.get("active_safety_profile", "")
            hardware_roles = [
                r.get("role", "") for r in (data.get("hardware_requirements") or [])
            ]
            registry = data.get("registry") or {}
            nodes = registry.get("nodes") or []
            for node in nodes:
                ti = node.get("tool_interface")
                if not ti or not ti.get("operations"):
                    continue
                for op in ti["operations"]:
                    if op.get("safety_class") != "forbidden":
                        tool_count += 1

    return EmbodimentPackageInfo(
        key=key,
        package_id=pkg_cfg.package_id,
        path=pkg_cfg.path,
        name=name,
        version=version,
        description=pkg_cfg.description,
        enabled=pkg_cfg.enabled,
        bindings=pkg_cfg.bindings or {},
        kind=kind,
        tool_count=tool_count,
        path_exists=path_exists,
        safety_profile=safety_profile,
        hardware_roles=hardware_roles,
    )


def _run_passive_probe(package_dir: Path) -> Dict[str, Any]:
    """Run probes/passive_probe.py for a package and return its JSON output."""
    probe = package_dir / "probes" / "passive_probe.py"
    if not probe.exists():
        return {}
    try:
        import subprocess
        import json as _json
        import ast
        result = subprocess.run(
            ["python3", str(probe)],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=str(package_dir),
        )
        if result.returncode == 0 and result.stdout.strip():
            raw = result.stdout.strip()
            try:
                return _json.loads(raw)
            except Exception:
                return ast.literal_eval(raw)
    except Exception:
        pass
    return {}


def _get_scan_dirs() -> List[Path]:
    """Return directories to scan for embodiment packages."""
    dirs = []
    # User-installed
    user_dir = WORKING_DIR / "embodiments"
    if user_dir.exists():
        dirs.append(user_dir)

    # Bundled packages alongside the awomopaw repo
    try:
        import qwenpaw
        pkg_root = Path(qwenpaw.__file__).parent.parent.parent
        bundled = pkg_root / "embodiments" / "packages"
        if bundled.exists():
            dirs.append(bundled)
    except Exception:
        pass

    # Also check common sibling paths
    for candidate in [
        Path("/home/sun/wangxin/physclaw_code/embodiments/packages"),
        Path.cwd() / "embodiments" / "packages",
        Path.cwd().parent / "embodiments" / "packages",
    ]:
        if candidate.exists() and candidate not in dirs:
            dirs.append(candidate)
    return dirs


# ── Routes ─────────────────────────────────────────────────────────────────


@router.get("", response_model=List[EmbodimentPackageInfo])
async def list_embodiments(request: Request) -> List[EmbodimentPackageInfo]:
    from ..agent_context import get_agent_for_request
    agent = await get_agent_for_request(request)
    emb = agent.config.embodiments
    if not emb or not emb.packages:
        return []
    return [_build_package_info(k, v) for k, v in emb.packages.items()]


@router.get("/discover", response_model=List[EmbodimentDiscoveryEntry])
async def discover_embodiments(request: Request) -> List[EmbodimentDiscoveryEntry]:
    """Use EmbodimentLoader.discover() to scan filesystem for available packages."""
    from ..agent_context import get_agent_for_request
    agent = await get_agent_for_request(request)
    emb = agent.config.embodiments
    configured: Dict[str, str] = {}
    if emb and emb.packages:
        for key, pkg in emb.packages.items():
            configured[pkg.package_id] = key

    try:
        from embodiments.runtime.loader import EmbodimentLoader
    except ImportError:
        logger.warning("embodiments package not installed; returning empty discover results")
        return []

    scan_dirs = [str(d) for d in _get_scan_dirs()]
    if not scan_dirs:
        return []

    loader = EmbodimentLoader(config={"scan_dirs": scan_dirs})
    found_ids = loader.discover()

    results: List[EmbodimentDiscoveryEntry] = []
    for pid in found_ids:
        pkg = loader._available.get(pid)  # type: ignore[attr-defined]
        if not pkg:
            continue
        data = pkg.embodiment_yaml
        hw_roles = [r.get("role", "") for r in (data.get("hardware_requirements") or [])]
        probe_info = _run_passive_probe(Path(str(pkg.package_dir)))
        hardware_detected = bool(probe_info.get("ok") or probe_info.get("detected"))
        results.append(EmbodimentDiscoveryEntry(
            package_id=pid,
            name=data.get("name", pid),
            path=str(pkg.package_dir),
            version=data.get("version", ""),
            kind=data.get("kind", ""),
            description=data.get("summary", data.get("description", "")),
            hardware_roles=hw_roles,
            already_configured=pid in configured,
            config_key=configured.get(pid),
            hardware_detected=hardware_detected,
            probe_info=probe_info,
        ))
    return results


@router.get("/{package_key}/tools", response_model=List[EmbodimentToolInfo])
async def get_embodiment_tools(
    request: Request,
    package_key: str = FPath(...),
) -> List[EmbodimentToolInfo]:
    """Return all tool operations for the package, including forbidden ones."""
    from ..agent_context import get_agent_for_request
    agent = await get_agent_for_request(request)
    emb = agent.config.embodiments
    if not emb or package_key not in (emb.packages or {}):
        raise HTTPException(404, detail=f"Embodiment '{package_key}' not found")

    pkg_cfg = emb.packages[package_key]
    expanded = Path(pkg_cfg.path).expanduser()
    yaml_path = expanded / "embodiment.yaml"
    if not yaml_path.exists():
        return []

    data = _read_yaml(yaml_path)
    registry = data.get("registry") or {}
    nodes = registry.get("nodes") or []

    tools: List[EmbodimentToolInfo] = []
    for node in nodes:
        ti = node.get("tool_interface")
        if not ti or not ti.get("operations"):
            continue
        node_id = node.get("id", "")
        node_name = node.get("name", node_id)
        for op in ti["operations"]:
            tools.append(EmbodimentToolInfo(
                name=f"{node_id}.{op['operation_id']}",
                description=op.get("description", ""),
                method=op.get("method", ""),
                path=op.get("path", ""),
                safety_class=op.get("safety_class", "safe"),
                input_schema=op.get("input_schema") or {},
                node_id=node_id,
                node_name=node_name,
                latency_estimate_ms=op.get("latency_estimate_ms"),
            ))
    return tools


@router.get("/{package_key}/context")
async def get_embodiment_context(
    request: Request,
    package_key: str = FPath(...),
) -> Dict[str, str]:
    """Return the EMBODIMENT.md content for the package."""
    from ..agent_context import get_agent_for_request
    agent = await get_agent_for_request(request)
    emb = agent.config.embodiments
    if not emb or package_key not in (emb.packages or {}):
        raise HTTPException(404, detail=f"Embodiment '{package_key}' not found")

    pkg_cfg = emb.packages[package_key]
    md_path = Path(pkg_cfg.path).expanduser() / "EMBODIMENT.md"
    if not md_path.exists():
        return {"content": "", "package_id": pkg_cfg.package_id}
    return {"content": md_path.read_text(encoding="utf-8"), "package_id": pkg_cfg.package_id}


@router.get("/{package_key}/hardware", response_model=List[HardwareRequirement])
async def get_embodiment_hardware(
    request: Request,
    package_key: str = FPath(...),
) -> List[HardwareRequirement]:
    """Return hardware requirements with current binding status."""
    from ..agent_context import get_agent_for_request
    agent = await get_agent_for_request(request)
    emb = agent.config.embodiments
    if not emb or package_key not in (emb.packages or {}):
        raise HTTPException(404, detail=f"Embodiment '{package_key}' not found")

    pkg_cfg = emb.packages[package_key]
    expanded = Path(pkg_cfg.path).expanduser()
    yaml_path = expanded / "embodiment.yaml"
    if not yaml_path.exists():
        return []

    data = _read_yaml(yaml_path)
    reqs = data.get("hardware_requirements") or []
    bindings = pkg_cfg.bindings or {}

    return [
        HardwareRequirement(
            role=r.get("role", ""),
            description=r.get("description", ""),
            device_type=r.get("device_type", ""),
            hint=r.get("hint", ""),
            bound_endpoint=bindings.get(r.get("role", ""), ""),
            default_port=r.get("connection", {}).get("default_port"),
        )
        for r in reqs
    ]


@router.get("/{package_key}", response_model=EmbodimentPackageInfo)
async def get_embodiment(
    request: Request,
    package_key: str = FPath(...),
) -> EmbodimentPackageInfo:
    from ..agent_context import get_agent_for_request
    agent = await get_agent_for_request(request)
    emb = agent.config.embodiments
    if not emb or package_key not in (emb.packages or {}):
        raise HTTPException(404, detail=f"Embodiment '{package_key}' not found")
    return _build_package_info(package_key, emb.packages[package_key])


@router.post("", response_model=EmbodimentPackageInfo, status_code=201)
async def create_embodiment(
    request: Request,
    package_key: str = Body(..., embed=True),
    package: EmbodimentCreateRequest = Body(..., embed=True),
) -> EmbodimentPackageInfo:
    from ..agent_context import get_agent_for_request
    from ...config.config import save_agent_config, EmbodimentsConfig, EmbodimentPackageConfig

    agent = await get_agent_for_request(request)
    if agent.config.embodiments is None:
        agent.config.embodiments = EmbodimentsConfig(packages={})
    if package_key in agent.config.embodiments.packages:
        raise HTTPException(400, detail=f"Embodiment '{package_key}' already exists")

    # Auto-read package_id from yaml if path exists and package_id not specified
    pid = package.package_id
    if not pid:
        yaml_path = Path(package.path).expanduser() / "embodiment.yaml"
        if yaml_path.exists():
            data = _read_yaml(yaml_path)
            pid = data.get("package_id", Path(package.path).name)

    new_pkg = EmbodimentPackageConfig(
        package_id=pid,
        path=package.path,
        description=package.description,
        enabled=package.enabled,
        bindings=package.bindings,
    )
    agent.config.embodiments.packages[package_key] = new_pkg
    save_agent_config(agent.agent_id, agent.config)
    schedule_agent_reload(request, agent.agent_id)
    return _build_package_info(package_key, new_pkg)


@router.put("/{package_key}", response_model=EmbodimentPackageInfo)
async def update_embodiment(
    request: Request,
    package_key: str = FPath(...),
    updates: EmbodimentUpdateRequest = Body(...),
) -> EmbodimentPackageInfo:
    from ..agent_context import get_agent_for_request
    from ...config.config import save_agent_config, EmbodimentPackageConfig

    agent = await get_agent_for_request(request)
    emb = agent.config.embodiments
    if not emb or package_key not in (emb.packages or {}):
        raise HTTPException(404, detail=f"Embodiment '{package_key}' not found")

    existing = emb.packages[package_key]
    merged = existing.model_dump(mode="json")
    merged.update(updates.model_dump(exclude_unset=True))
    updated = EmbodimentPackageConfig.model_validate(merged)
    emb.packages[package_key] = updated
    save_agent_config(agent.agent_id, agent.config)
    schedule_agent_reload(request, agent.agent_id)
    return _build_package_info(package_key, updated)


@router.patch("/{package_key}/toggle", response_model=EmbodimentPackageInfo)
async def toggle_embodiment(
    request: Request,
    package_key: str = FPath(...),
) -> EmbodimentPackageInfo:
    from ..agent_context import get_agent_for_request
    from ...config.config import save_agent_config

    agent = await get_agent_for_request(request)
    emb = agent.config.embodiments
    if not emb or package_key not in (emb.packages or {}):
        raise HTTPException(404, detail=f"Embodiment '{package_key}' not found")

    pkg = emb.packages[package_key]
    pkg.enabled = not pkg.enabled
    save_agent_config(agent.agent_id, agent.config)
    schedule_agent_reload(request, agent.agent_id)
    return _build_package_info(package_key, pkg)


@router.delete("/{package_key}", response_model=Dict[str, str])
async def delete_embodiment(
    request: Request,
    package_key: str = FPath(...),
) -> Dict[str, str]:
    from ..agent_context import get_agent_for_request
    from ...config.config import save_agent_config

    agent = await get_agent_for_request(request)
    emb = agent.config.embodiments
    if not emb or package_key not in (emb.packages or {}):
        raise HTTPException(404, detail=f"Embodiment '{package_key}' not found")

    del emb.packages[package_key]
    save_agent_config(agent.agent_id, agent.config)
    schedule_agent_reload(request, agent.agent_id)
    return {"message": f"Embodiment '{package_key}' removed"}
