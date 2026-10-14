"""Load the VLA selected by deployment configuration.

The interactive harness does not contain a Pi0.5 call. The concrete VLA module
is a deployment binding and can be changed without changing the Skill or the
harness control loop.
"""

from __future__ import annotations

import importlib
import os


def _model_class():
    policy_name = os.environ.get("AWOMOPAW_VLA_POLICY_NAME", "Pi_05").strip()
    if not policy_name or not policy_name.replace("_", "").isalnum():
        raise ValueError("AWOMOPAW_VLA_POLICY_NAME is not a valid policy module")
    module = importlib.import_module(f"XPolicyLab.policy.{policy_name}.model")
    return getattr(module, "Model")


Model = _model_class()
