from __future__ import annotations

from pathlib import Path

from qwenpaw.app.embodiments.tool_adapter import (
    build_tool_functions_from_loader,
    embodiment_tool_requires_approval,
)

from embodiments.runtime.loader import EmbodimentLoader, EmbodimentPackage


PACKAGE_DIR = (
    Path(__file__).resolve().parents[3]
    / "embodiments"
    / "packages"
    / "franka_fr3"
)


def _loader() -> EmbodimentLoader:
    loader = EmbodimentLoader()
    package = EmbodimentPackage(PACKAGE_DIR)
    loader._available[package.package_id] = package
    loader.activate(
        package.package_id,
        bindings={"franka_gateway": "http://192.168.31.159:8092"},
    )
    return loader


def test_franka_package_exposes_atomic_http_tools() -> None:
    loader = _loader()
    tools = loader.get_tools()
    by_name = {tool["function"]["name"]: tool for tool in tools}

    assert "franka_fr3_controller.health_check" in by_name
    assert "franka_fr3_controller.move_joints" in by_name
    assert "franka_fr3_controller.move_joint_velocity" in by_name
    assert "franka_fr3_controller.move_cartesian_pose" in by_name
    assert "franka_fr3_controller.move_cartesian_velocity" in by_name
    assert "franka_fr3_controller.move_joints_small" not in by_name
    assert "franka_fr3_controller.execute_joint_trajectory" not in by_name
    assert "franka_fr3_controller.move_vertical_small" not in by_name
    assert "franka_fr3_controller.move_vertical_bounded" not in by_name
    move = by_name["franka_fr3_controller.move_joints"]
    assert move["metadata"]["protocol"] == "http"
    assert move["metadata"]["endpoint"] == "http://192.168.31.159:8092"
    assert move["metadata"]["safety_class"] == "safe"


def test_supported_franka_tools_do_not_require_extra_approval() -> None:
    loader = _loader()
    build_tool_functions_from_loader(loader)

    assert not embodiment_tool_requires_approval(
        "franka_fr3_controller__health_check", {}
    )
    assert not embodiment_tool_requires_approval("franka_fr3_controller__stop", {})
    assert not embodiment_tool_requires_approval(
        "franka_fr3_controller__connect", {}
    )
    assert not embodiment_tool_requires_approval(
        "franka_fr3_controller__move_joints", {"positions": [0.0] * 7}
    )
    assert not embodiment_tool_requires_approval(
        "franka_fr3_controller__move_joint_velocity",
        {"velocities_rad_s": [0.0] * 7, "duration_seconds": 1.0},
    )
    assert not embodiment_tool_requires_approval(
        "franka_fr3_controller__move_cartesian_pose",
        {
            "pose": [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.01],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "relative": True,
        },
    )
    assert not embodiment_tool_requires_approval(
        "franka_fr3_controller__move_cartesian_velocity",
        {"twist": [0.0] * 6, "duration_seconds": 1.0},
    )
    assert not embodiment_tool_requires_approval(
        "franka_fr3_controller__set_gripper", {"width_m": 0.08}
    )
    assert not embodiment_tool_requires_approval(
        "franka_fr3_controller__recover", {}
    )


def test_loader_accepts_bounded_franka_motion_without_extra_approval() -> None:
    loader = _loader()
    result = loader.dispatch_tool(
        "franka_fr3_controller.move_joints",
        {"positions": [0.0] * 7},
        approved=False,
    )
    assert result["success"]
    assert not result.get("requires_approval", False)
