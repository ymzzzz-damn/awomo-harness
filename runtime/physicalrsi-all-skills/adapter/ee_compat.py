"""Process-local EE contract for frozen skills; action values remain unchanged.

Install after importing the simulator's interactive harness. Joint runs retain
the original harness methods. Each simulator process selects one action type.
"""
import sys

import numpy as np


_MODULE = "src.eval_client.interactive_harness"
_SAVED = "_physicalrsi_ee_original_methods"
_POSE_KEYS = frozenset(("ee_pose", "left_ee_pose", "right_ee_pose"))
_GRIPPER_KEYS = frozenset(("ee_joint_state", "left_ee_joint_state", "right_ee_joint_state"))


def install(action_type, module_name=_MODULE):
    """Enable the selected contract and return a callable restoring the original.

    Only an already loaded harness is patched, so this does not import Isaac or
    replace an external source file. Repeated EE installs are idempotent; joint
    explicitly restores a previous EE install in this process.
    """
    if action_type not in ("joint", "ee"):
        raise ValueError("frozen action type must be joint or ee")
    module = sys.modules.get(module_name)
    if module is None:
        raise RuntimeError("import the interactive harness before installing the action contract")
    harness = module.InteractiveEpisodeHarness
    saved = getattr(module, _SAVED, None)

    def restore():
        originals = getattr(module, _SAVED, None)
        if originals is not None:
            module._action_spec, harness._validate_action_chunk = originals
            delattr(module, _SAVED)

    if action_type == "joint":
        restore()
        return restore
    if saved is not None:
        return restore
    original_spec = module._action_spec
    setattr(module, _SAVED, (original_spec, harness._validate_action_chunk))

    def ee_action_spec(task_env, observation):
        spec = dict(original_spec(task_env, observation))
        supported = {}
        for key, value in module._public_robot_state(observation).items():
            if key in _POSE_KEYS and len(value) == 7:
                supported[key] = {"dimension": 7, "finite": True}
            elif key in _GRIPPER_KEYS:
                supported[key] = {"dimension": len(value), "finite": True,
                                  "normalized_range": [0.0, 1.0]}
        spec.update(action_type="absolute_ee", required_fields={},
                    supported_fields=supported, requires_one_arm_pose=True)
        return spec

    def validate_ee_chunk(self, actions, *, max_actions):
        # Keep the native shape/key checks and the original chunk/numeric checks.
        # Partial arm actions are valid in the native contract and stay valid.
        if not isinstance(actions, (list, tuple)) or not actions:
            raise ValueError("provider returned an empty or invalid action chunk")
        if len(actions) > max_actions:
            raise ValueError(f"action chunk exceeds the {max_actions}-action limit")
        validated = []
        for action in actions:
            if not isinstance(action, dict) or not action:
                raise ValueError("every action must be a non-empty object")
            self.task_env.validate_action_dict(action)
            if self.task_env.get_action_type(action) != "ee":
                raise ValueError("this frozen skill only accepts ee actions")
            if not _POSE_KEYS.intersection(action) or set(action) - (_POSE_KEYS | _GRIPPER_KEYS):
                raise ValueError("absolute ee actions require an ee_pose and only pose/gripper fields")
            for key, value in action.items():
                try:
                    numeric = np.asarray(value, dtype=np.float64)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"action field {key!r} is not numeric") from exc
                if not np.isfinite(numeric).all():
                    raise ValueError(f"action field {key!r} contains NaN or infinity")
            validated.append(action)
        return validated

    module._action_spec = ee_action_spec
    harness._validate_action_chunk = validate_ee_chunk
    return restore
