"""Explicit frozen PhysicalRSI provider bindings, with owned worker isolation."""
from __future__ import annotations

import atexit
import importlib
import json
import os
from pathlib import Path
import sys

from XPolicyLab.model_template import ModelTemplate


PUBLIC_STATE = (
    "left_arm_joint_state", "right_arm_joint_state", "left_ee_joint_state",
    "right_ee_joint_state", "left_ee_pose", "right_ee_pose",
)


def public_observation(obs):
    clean = {
        "vision": {name: {"color": camera["color"]}
                   for name, camera in obs["vision"].items()
                   if name in {"cam_head", "cam_left_wrist", "cam_right_wrist"}
                   and "color" in camera},
        "state": {key: obs["state"][key] for key in PUBLIC_STATE if key in obs["state"]},
    }
    for key in ("instruction", "env_idx"):
        if key in obs:
            clean[key] = obs[key]
    return clean


def expand(value, variables):
    if isinstance(value, str):
        for key, replacement in variables.items():
            value = value.replace("{" + key + "}", str(replacement))
        return value
    if isinstance(value, list):
        return [expand(item, variables) for item in value]
    if isinstance(value, dict):
        return {expand(key, variables): expand(item, variables) for key, item in value.items()}
    return value


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        super().__init__()
        self.backend = None
        manifest = json.loads(Path(__file__).with_name("catalog.json").read_text())
        entry = manifest["releases"].get(model_cfg.get("ckpt_name"))
        if entry is None:
            selected = os.environ.get("AWOMOPAW_VLA_POLICY_NAME", "")
            if selected not in {"Pi_05", "PhysicalRSI_Align_Skill"}:
                raise ValueError("Explicit registered frozen release required")
            self.backend = importlib.import_module("XPolicyLab.policy." + selected + ".model").Model(model_cfg)
            self._indices = [0]
            atexit.register(self.close)
            return
        if entry["kind"] == "code" and model_cfg.get("task_name") != entry["task"]:
            raise ValueError("Selected frozen skill does not support this task")
        if model_cfg.get("env_cfg_type") != "arx_x5" or model_cfg.get("action_type") != entry["action_type"]:
            raise ValueError("Frozen observation/action contract mismatch")
        run_id = os.environ.get("ROBODOJO_RUN_ID")
        if not run_id:
            raise ValueError("Unique RoboDojo run id required")
        framework = Path(__file__).resolve().parents[2]
        output = framework.parent / "eval_result/frozen_skill_runs" / run_id
        output.mkdir(parents=True, exist_ok=False)
        source = Path(manifest["asset_root"])
        if entry["kind"] == "code":
            from .program_service import Model as ProgramModel
            configuration = expand(entry["configuration"], {
                "root": source, "python": sys.executable, "framework": framework,
                "evidence": output, "port": "0",
            })
            configuration.update(output=str(output), request_timeout_s=1800)
            self.backend = ProgramModel(configuration)
        else:
            # VLA weights remain separate from the code package, as in RSI.
            sys.path.insert(0, manifest["framework_source"])
            from importlib import import_module
            module = import_module(entry["implementation"])
            cfg = dict(model_cfg, **entry["configuration"])
            self.backend = module.Model(cfg)
        (output / "binding.json").write_text(json.dumps({
            "entry": entry, "asset_root": str(source),
            "original_controller_unchanged": True,
            "qualification": "Registered frozen skill via AwomoPaw embodiment; no LLM selection claim",
        }, indent=2) + "\n")
        atexit.register(self.close)
        self._indices = [0]

    def update_obs(self, obs):
        return self.update_obs_batch([obs])

    def update_obs_batch(self, observations):
        self._indices = [obs.get("env_idx", i) for i, obs in enumerate(observations)]
        return self.backend.update_obs_batch([public_observation(obs) for obs in observations])

    def get_action(self):
        return self.get_action_batch(self._indices[:1])[0]

    def get_action_batch(self, env_idx_list=None):
        return self.backend.get_action_batch(env_idx_list or self._indices)

    def reset(self):
        return self.backend.reset()

    def prepare_case(self, case_meta=None):
        return None

    def on_trial_end(self, result=None):
        return None

    def close(self):
        if self.backend is not None:
            close = getattr(self.backend, "close", None)
            if callable(close):
                close()
            self.backend = None
