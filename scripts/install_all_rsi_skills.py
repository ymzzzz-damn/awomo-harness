#!/usr/bin/env python3
"""Stage and install every released PhysicalRSI skill without replacing active code."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "runtime/physicalrsi-all-skills"
ORIGINAL = Path("/data/ymz/physicalrsi-reproduction/skill-assets/implementations/code-skills")
RELEASED_FRAMEWORK = Path("/data/ymz/physicalrsi-reproduction/RoboDojo/XPolicyLab/policy/physicalRSI/runtime")
INSTALL = Path("/home/ymz/physclaw/awomopaw/runtime/physicalrsi-skills")
SIM = Path("/home/ymz/robodojo-pi05/RoboDojo")
WORKSPACE = Path("/home/ymz/.awomopaw/workspaces/car_controller")
ENVIRONMENT = Path("/data/ymz/physicalrsi-reproduction/skill-assets/implementations/openpi")


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def prepare():
    programs = json.loads((ORIGINAL / "programs.json").read_text())["programs"]
    releases, mappings, providers, receipts = {}, {}, [], []
    skills = STAGE / "skills"
    skills.mkdir(exist_ok=True)
    for program in programs:
        name = program["name"]
        command = program["configuration"]["command"]
        template = command[command.index("--descriptor") + 1]
        descriptor_path = Path(template.replace("{root}", str(ORIGINAL)))
        descriptor = json.loads(descriptor_path.read_text())
        task = descriptor["name"]
        revision = descriptor["implementation"]["implementation_revision"]
        release_id = "rsi-" + name + "-" + revision[:16]
        # Verify the complete declared executable dependency set, not only its entrypoint.
        for template, expected in program["configuration"]["files"].items():
            path = Path(template.replace("{root}", str(ORIGINAL)))
            if not path.resolve().is_relative_to(ORIGINAL.resolve()):
                raise ValueError("Dependency escapes frozen asset root")
            if digest(path) != expected:
                raise ValueError("Frozen dependency changed: " + str(path))
        entry = {
            "kind": "code", "name": name, "task": task,
            "action_type": program["action_type"], "revision": revision,
            "configuration": program["configuration"],
            "descriptor_sha256": digest(descriptor_path),
        }
        releases[release_id] = entry
        provider = {
            "provider_id": "physicalrsi-" + name, "kind": "code",
            "policy_name": "PhysicalRSI_Frozen_Skills", "checkpoint": release_id,
            "policy_uv_env_path": str(ENVIRONMENT), "revision": "frozen-code:" + revision,
        }
        providers.append(provider)
        mappings[task] = {"provider_id": provider["provider_id"], "action_type": program["action_type"]}
        receipts.append({"name": name, "verified_files": len(program["configuration"]["files"]),
                         "source_revision": revision, "action_type": program["action_type"]})
        write_skill(skills, name, program["description"], provider["provider_id"],
                    task, program["action_type"])
    example = json.loads((RELEASED_FRAMEWORK / "PhysicalRSI_baselines/robodojo/configs/skill_library.example.json").read_text())
    for name, implementation in [("pi05", "pi05"), ("pi05-sparse-memory", "sparse")]:
        configuration = example["skills"][name]["configuration"]
        configuration["model_path"] = str(Path("/data/ymz/physicalrsi-reproduction/skill-assets/checkpoints") / name)
        checkpoint = Path(configuration["model_path"])
        if not (checkpoint / "params").exists():
            raise FileNotFoundError(str(checkpoint))
        backend = RELEASED_FRAMEWORK / ("PhysicalRSI_baselines/robodojo/skills/" + implementation + ".py")
        revision = digest(backend)
        release_id = "rsi-" + name + "-" + revision[:16]
        releases[release_id] = {
            "kind": "vla", "name": name, "action_type": "joint", "revision": revision,
            "implementation": "PhysicalRSI_baselines.robodojo.skills." + implementation,
            "configuration": configuration,
        }
        providers.append({
            "provider_id": "physicalrsi-" + name, "kind": "vla",
            "policy_name": "PhysicalRSI_Frozen_Skills", "checkpoint": release_id,
            "policy_uv_env_path": str(ENVIRONMENT), "revision": "frozen-source:" + revision,
        })
        write_skill(skills, name, "Released PhysicalRSI " + name + " inference skill with a separate checkpoint.",
                    "physicalrsi-" + name, "a supported official RoboDojo task", "joint")
    save(STAGE / "adapter/catalog.json", {
        "schema": "awomopaw.physicalrsi-frozen-catalog/v1",
        "asset_root": str(INSTALL / "code-skills"),
        "framework_source": str(INSTALL / "framework"), "releases": releases,
    })
    existing = json.loads((ROOT / "runtime/physicalrsi-align-skill/registry.json").read_text())
    save(STAGE / "registry.json", existing + providers)
    save(STAGE / "task-providers.json", mappings)
    save(STAGE / "preflight.json", {"code_skills": len(programs), "vla_skills": 2,
        "source_dependencies_verified": True, "skills": receipts})
    with zipfile.ZipFile(STAGE / "skills.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(skills.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(skills))
    print(json.dumps({"staged": True, "code_skills": len(programs), "vla_skills": 2}))


def write_skill(root, name, description, provider, task, action_type):
    skill_name = "physicalrsi-frozen-" + name
    directory = root / skill_name
    directory.mkdir(exist_ok=True)
    text = f'''---
name: {skill_name}
description: {json.dumps("Explicitly reuse the released PhysicalRSI " + name + " skill in RoboDojo simulation. " + description)}
metadata:
  qwenpaw:
    emoji: "🧩"
    requires: {{}}
---

# Frozen PhysicalRSI {name}

Use for an explicit request to reuse or evaluate PhysicalRSI skills. This skill
is a supplied frozen implementation, not new code authored by the current LLM.

1. Read robodojo_sim.health. Require provider `{provider}` with ready=true.
   This provider uses `{action_type}` actions and supports `{task}`. Match the
   bridge action contract; stop on mismatch rather than substituting a provider.
2. Start the specified task/seed using policy_id=`{provider}`. Run only one
   episode per bridge. Parallel tasks require separate managed bridge endpoints.
3. Read current_observation and preserve its full official instruction. Request
   the registered provider's chunk with that instruction as the current goal.
4. Execute the unchanged actions through execute_action_chunk. Use a 10-action
   window and discard_remaining=true, then obtain a fresh observation; a shorter
   frozen chunk must not be padded, repeated or rewritten.
5. Read evaluation_result only after episode completion. Archive official score,
   trace, source identity and all three native camera videos by unique run id.
   Report infrastructure errors separately from valid task outcomes.

Source: `/home/ymz/physclaw/awomopaw/runtime/physicalrsi-skills`.
Registry: `{STAGE / "registry.json"}`. Actions are accepted only by the
AwomoPaw embodiment; the skill never actuates a simulator directly.
'''
    (directory / "SKILL.md").write_text(text)


def install():
    if not (STAGE / "adapter/ee_compat.py").exists():
        raise FileNotFoundError("Complete the ee compatibility adapter before installation")
    INSTALL.mkdir(parents=True, exist_ok=True)
    copy_receipt = INSTALL / "source-copy-complete.json"
    if not copy_receipt.exists():
        target = INSTALL / "code-skills"
        if target.exists():
            raise FileExistsError("Unverified partial source install: " + str(target))
        # Asset versions share hardlinked weights and contain sparse tensors.
        # GNU cp preserves their allocation and identity instead of duplicating
        # each logical checkpoint as copytree would.
        subprocess.run(["cp", "-a", "--reflink=auto", "--sparse=always",
                        str(ORIGINAL), str(target)], check=True)
        shutil.copytree(RELEASED_FRAMEWORK, INSTALL / "framework", symlinks=True,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        # Check copied dependencies using the same official hashes.
        unique = {}
        for entry in json.loads((STAGE / "adapter/catalog.json").read_text())["releases"].values():
            if entry["kind"] == "code":
                unique.update(entry["configuration"]["files"])
        for template, expected in unique.items():
            path = Path(template.replace("{root}", str(target)))
            if digest(path) != expected:
                raise ValueError("Copied source hash mismatch: " + str(path))
        save(copy_receipt, {"copied_source_root": str(target), "verified_unique_files": len(unique),
                            "original_controller_unchanged": True})
    adapter = SIM / "XPolicyLab/policy/PhysicalRSI_Frozen_Skills"
    if not adapter.exists():
        shutil.copytree(STAGE / "adapter", adapter, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy2(SIM / "XPolicyLab/policy/Awomopaw_Harness/setup_eval_policy_server.sh",
                     adapter / "setup_eval_policy_server.sh")
        shutil.copy2(SIM / "XPolicyLab/policy/Awomopaw_Harness/setup_eval_env_client.sh",
                     adapter / "setup_eval_env_client.sh")
    else:
        for path in (STAGE / "adapter").iterdir():
            if path.is_file() and (not (adapter / path.name).exists() or digest(path) != digest(adapter / path.name)):
                raise ValueError("Refuse to overwrite existing provider: " + str(adapter / path.name))
    from qwenpaw.agents.skills_manager import SkillService
    service = SkillService(WORKSPACE)
    names = [path.name for path in sorted((STAGE / "skills").iterdir())]
    present = [name for name in names if (WORKSPACE / "skills" / name).exists()]
    if not present:
        result = service.import_from_zip((STAGE / "skills.zip").read_bytes(), enable=False)
        if result.get("count") != 24:
            raise RuntimeError("Skill import incomplete: " + json.dumps(result))
    elif len(present) != len(names):
        raise RuntimeError("Partial existing skill import; inspect before repair")
    for name in names:
        if digest(STAGE / "skills" / name / "SKILL.md") != digest(WORKSPACE / "skills" / name / "SKILL.md"):
            raise ValueError("Installed skill source changed: " + name)
    save(STAGE / "install-receipt.json", {
        "schema": "awomopaw.physicalrsi-all-skills-install/v1",
        "installed_at": datetime.now(timezone.utc).isoformat(),
        "code_skills": 22, "vla_skills": 2, "workspace_skills": names,
        "source": str(INSTALL), "adapter": str(adapter), "source_copy": json.loads(copy_receipt.read_text()),
        "enabled": False, "scope": "Local simulation; existing episode and services unchanged",
    })
    print(json.dumps({"installed": True, "skills": len(names), "adapter": str(adapter)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    install() if args.install else prepare()
