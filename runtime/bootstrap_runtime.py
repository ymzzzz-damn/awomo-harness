#!/usr/bin/env python3
"""Initialize and validate an isolated, simulation-only QwenPaw runtime."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any


EXPECTED_TOOLS = {
    "robodojo_sim.health",
    "robodojo_sim.start_session",
    "robodojo_sim.session_status",
    "robodojo_sim.current_observation",
    "robodojo_sim.capture_head_frame",
    "robodojo_sim.capture_left_wrist_frame",
    "robodojo_sim.capture_right_wrist_frame",
    "robodojo_sim.request_policy_chunk",
    "robodojo_sim.submit_direct_action_chunk",
    "robodojo_sim.execute_action_chunk",
    "robodojo_sim.abort_session",
    "robodojo_sim.start_evaluation",
    "robodojo_sim.current_evaluation",
    "robodojo_sim.evaluation_status",
    "robodojo_sim.evaluation_result",
}
FORBIDDEN_PACKAGE_NAMES = {
    "franka_fr3",
    "g1_usb_dual_view",
    "jetrover_base",
    "jetrover_camera_arm",
    "unitree_g1",
    "unitree_g1_sensor_only",
    "usb_1080p_camera",
}


def require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"required environment variable is missing: {name}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_simulation_only(release_root: Path) -> Path:
    packages_dir = release_root / "source" / "embodiments" / "packages"
    package_dir = packages_dir / "robodojo_sim"
    if not package_dir.is_dir():
        raise RuntimeError(f"simulation embodiment is missing: {package_dir}")

    present = {path.name for path in packages_dir.iterdir() if path.is_dir()}
    forbidden = sorted(present & FORBIDDEN_PACKAGE_NAMES)
    if forbidden:
        raise RuntimeError(f"real-device packages present in release: {forbidden}")
    if present != {"robodojo_sim"}:
        raise RuntimeError(f"unexpected embodiment package set: {sorted(present)}")

    for path in (release_root / "source" / "embodiments").rglob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        if "real_device: true" in text:
            raise RuntimeError(f"real-device declaration found in {path}")
    return package_dir


async def configure_runtime(release_root: Path, bridge_url: str) -> dict[str, Any]:
    api_key = require_env("QWENPAW_API_KEY")
    base_url = require_env("QWENPAW_API_BASE_URL")
    model = os.environ.get("QWENPAW_MODEL", "gpt-6-astra").strip()
    provider_id = "robodojo-model-api"

    from embodiments.runtime.loader import EmbodimentLoader, EmbodimentPackage
    from qwenpaw.config.config import (
        AgentProfileConfig,
        AgentProfileRef,
        AgentsRunningConfig,
        ChannelConfig,
        EmbodimentPackageConfig,
        EmbodimentsConfig,
        ModelSlotConfig,
        Config,
        save_agent_config,
    )
    from qwenpaw.config.utils import save_config
    from qwenpaw.constant import BUILTIN_QA_AGENT_ID, WORKING_DIR, SECRET_DIR
    from qwenpaw.agents.skills_manager import (
        SkillService,
        reconcile_workspace_manifest,
    )
    from qwenpaw.providers.openai_provider import OpenAIProvider
    from qwenpaw.providers.provider import ModelInfo
    from qwenpaw.providers.provider_manager import ProviderManager

    package_dir = assert_simulation_only(release_root)
    package = EmbodimentPackage(package_dir)
    loader = EmbodimentLoader()
    loader._available[package.package_id] = package  # pylint: disable=protected-access
    loader.activate(package.package_id, bindings={"robodojo_sim_api": bridge_url})
    tools = loader.get_tools()
    tool_names = {tool["function"]["name"] for tool in tools}
    if tool_names != EXPECTED_TOOLS:
        raise RuntimeError(f"unexpected simulation tool set: {sorted(tool_names)}")
    for tool in tools:
        metadata = tool.get("metadata") or {}
        if metadata.get("protocol") not in {"http", "http_snapshot"}:
            raise RuntimeError(f"non-HTTP protocol exposed: {metadata}")
        if metadata.get("endpoint") != bridge_url.rstrip("/"):
            raise RuntimeError(f"simulation endpoint is not bound: {metadata}")

    root_config = Config()
    workspace_dir = WORKING_DIR / "workspaces" / "default"
    root_config.agents.profiles["default"].workspace_dir = str(workspace_dir)
    # The generic QA profile is unrelated to RoboDojo evaluation. Keep its
    # reserved slot disabled so first-start migration does not create and load
    # a second agent with a persistent memory index.
    root_config.agents.profiles[BUILTIN_QA_AGENT_ID] = AgentProfileRef(
        id=BUILTIN_QA_AGENT_ID,
        workspace_dir=str(WORKING_DIR / "workspaces" / BUILTIN_QA_AGENT_ID),
        enabled=False,
    )
    save_config(root_config)

    agent_config = AgentProfileConfig(
        id="default",
        name="RoboDojo Simulation Harness",
        description="QwenPaw harness isolated to RoboDojo simulation.",
        workspace_dir=str(workspace_dir),
        channels=ChannelConfig(),
        running=AgentsRunningConfig(memory_manager_backend="none"),
        active_model=ModelSlotConfig(provider_id=provider_id, model=model),
        embodiments=EmbodimentsConfig(
            packages={
                "robodojo_sim": EmbodimentPackageConfig(
                    package_id="robodojo_sim",
                    path=str(package_dir),
                    enabled=True,
                    bindings={"robodojo_sim_api": bridge_url},
                    description="Simulation-only RoboDojo embodiment",
                )
            }
        ),
        language="zh",
    )
    save_agent_config("default", agent_config)

    skill_name = "robodojo-vla-control"
    skill_source = release_root / "skills" / skill_name
    if not (skill_source / "SKILL.md").is_file():
        raise RuntimeError(f"required VLA control skill is missing: {skill_source}")
    skill_target = workspace_dir / "skills" / skill_name
    skill_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(skill_source, skill_target, dirs_exist_ok=True)
    reconcile_workspace_manifest(workspace_dir)
    skill_result = SkillService(workspace_dir).enable_skill(skill_name)
    if not skill_result.get("success"):
        raise RuntimeError(f"failed to enable VLA control skill: {skill_result}")

    manager = ProviderManager()
    provider = manager.get_provider(provider_id)
    provider_config = {
        "base_url": base_url,
        "api_key": api_key,
        "extra_models": [{"id": model, "name": model}],
        "generate_kwargs": {"api_mode": "responses"},
    }
    if provider is None:
        await manager.add_custom_provider(
            OpenAIProvider(
                id=provider_id,
                name="RoboDojo model API",
                base_url=base_url,
                api_key=api_key,
                models=[ModelInfo(id=model, name=model)],
                support_model_discovery=False,
                support_connection_check=False,
                generate_kwargs={"api_mode": "responses"},
            )
        )
    else:
        if not manager.update_provider(provider_id, provider_config):
            raise RuntimeError("failed to update model provider")
    manager.active_model = ModelSlotConfig(provider_id=provider_id, model=model)
    manager.save_active_model(manager.active_model)

    plaintext_found = False
    for path in SECRET_DIR.rglob("*.json"):
        if api_key in path.read_text(encoding="utf-8"):
            plaintext_found = True
            break
    if plaintext_found:
        raise RuntimeError("provider secret was persisted as plaintext")

    return {
        "status": "ready",
        "mode": "simulation-only",
        "model": model,
        "provider_id": provider_id,
        "base_url": base_url,
        "secret_persisted_plaintext": False,
        "embodiment_packages": ["robodojo_sim"],
        "tools": sorted(tool_names),
        "bridge_url": bridge_url,
        "memory_backend": "none",
        "builtin_qa_agent_enabled": False,
        "enabled_skills": [skill_name],
        "working_dir": str(WORKING_DIR),
        "source_manifest_sha256": sha256_file(release_root / "SOURCE_MANIFEST.json"),
    }


async def check_api() -> None:
    import httpx

    base_url = require_env("QWENPAW_API_BASE_URL").rstrip("/")
    api_key = require_env("QWENPAW_API_KEY")
    model = os.environ.get("QWENPAW_MODEL", "gpt-6-astra").strip()
    payload = {"model": model, "input": "Reply with OK."}
    async with httpx.AsyncClient(timeout=45, trust_env=False) as client:
        response = await client.post(
            f"{base_url}/responses",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        )
    if response.status_code != 200:
        raise RuntimeError(f"model API check failed with HTTP {response.status_code}")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--bridge-url", default="http://127.0.0.1:8098")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--check-api", action="store_true")
    args = parser.parse_args()

    release_root = args.release_root.resolve()
    if args.check_api:
        await check_api()
    receipt = await configure_runtime(release_root, args.bridge_url.rstrip("/"))
    receipt["model_api_check"] = "passed" if args.check_api else "not-requested"

    serialized = json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main()))
    except Exception as exc:  # keep output compact and secret-free
        print(f"QwenPaw simulation bootstrap failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

