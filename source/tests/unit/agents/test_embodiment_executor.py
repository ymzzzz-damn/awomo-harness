"""Offline safety tests for physical embodiment execution."""

import json
from pathlib import Path

import httpx
import pytest

from qwenpaw.app.embodiments.executor import (
    EmbodimentExecutionError,
    EmbodimentExecutor,
)


@pytest.mark.asyncio
async def test_http_json_honors_long_robot_tool_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qwenpaw.app.embodiments import executor as executor_module

    captured: dict[str, float] = {}

    class FakeResponse:
        content = b'{"ok": true}'
        headers = {"content-type": "application/json"}
        status_code = 200
        text = ""

        @staticmethod
        def raise_for_status() -> None:
            return None

        @staticmethod
        def json() -> dict[str, bool]:
            return {"ok": True}

    class FakeClient:
        def __init__(self, *, timeout: float, trust_env: bool) -> None:
            captured["timeout"] = timeout
            assert trust_env is False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def post(self, _url: str, *, json: dict):
            assert json == {"waypoints": []}
            return FakeResponse()

    monkeypatch.setattr(executor_module.httpx, "AsyncClient", FakeClient)
    result = await EmbodimentExecutor(tmp_path)._http_json(  # pylint: disable=protected-access
        {
            "endpoint": "http://robot:8092",
            "path": "/command/execute-joint-trajectory",
            "method": "POST",
            "execution": {"timeout_seconds": 45},
            "arguments": {"waypoints": []},
        }
    )

    assert result == {"ok": True}
    assert captured["timeout"] == 45.0


@pytest.mark.asyncio
async def test_http_error_preserves_gateway_recovery_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qwenpaw.app.embodiments import executor as executor_module

    class FakeResponse:
        content = b'{"ok":false,"error":"current mode (Reflex)"}'
        headers = {"content-type": "application/json"}
        status_code = 500
        text = content.decode()

        @staticmethod
        def json() -> dict[str, object]:
            return {"ok": False, "error": 'current mode ("Reflex")'}

        @staticmethod
        def raise_for_status() -> None:
            request = httpx.Request("POST", "http://robot/command")
            response = httpx.Response(500, request=request)
            raise httpx.HTTPStatusError(
                "server error", request=request, response=response
            )

    class FakeClient:
        def __init__(self, **_kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args) -> None:
            return None

        async def post(self, _url: str, *, json: dict):
            return FakeResponse()

    monkeypatch.setattr(executor_module.httpx, "AsyncClient", FakeClient)
    result = await EmbodimentExecutor(tmp_path).execute(
        {
            "protocol": "http",
            "endpoint": "http://robot",
            "path": "/command",
            "method": "POST",
            "arguments": {},
        }
    )

    assert result.metadata["success"] is False
    assert 'current mode ("Reflex")' in result.metadata["error"]


@pytest.mark.asyncio
async def test_bounded_motion_rejects_reverse_before_transport(tmp_path: Path) -> None:
    executor = EmbodimentExecutor(tmp_path)
    with pytest.raises(EmbodimentExecutionError, match="distance_m"):
        await executor._bounded_motion(  # pylint: disable=protected-access
            {
                "endpoint": "ws://127.0.0.1:1",
                "execution": {"target_kind": "linear", "max_distance_m": 0.5},
                "arguments": {"distance_m": -0.2},
            }
        )


@pytest.mark.asyncio
async def test_bounded_motion_rejects_excess_rotation_before_transport(
    tmp_path: Path,
) -> None:
    executor = EmbodimentExecutor(tmp_path)
    with pytest.raises(EmbodimentExecutionError, match="yaw_rad"):
        await executor._bounded_motion(  # pylint: disable=protected-access
            {
                "endpoint": "ws://127.0.0.1:1",
                "execution": {"target_kind": "rotate", "max_yaw_rad": 1.57},
                "arguments": {"yaw_rad": 2.0},
            }
        )


@pytest.mark.asyncio
async def test_timed_twist_rejects_reverse_before_transport(tmp_path: Path) -> None:
    executor = EmbodimentExecutor(tmp_path)
    with pytest.raises(EmbodimentExecutionError, match="negative linear_x_mps"):
        await executor._timed_twist(  # pylint: disable=protected-access
            {
                "endpoint": "ws://127.0.0.1:1",
                "execution": {
                    "topic": "/qwenpaw/base/proposed_cmd_vel",
                    "max_linear_speed_mps": 0.18,
                    "max_angular_speed_rps": 0.30,
                    "max_duration_seconds": 10.0,
                    "allow_reverse": False,
                },
                "arguments": {
                    "linear_x_mps": -0.1,
                    "angular_z_rps": 0.0,
                    "duration_seconds": 1.0,
                },
            }
        )


@pytest.mark.asyncio
async def test_timed_motor_rps_rejects_excess_speed_before_transport(
    tmp_path: Path,
) -> None:
    executor = EmbodimentExecutor(tmp_path)
    with pytest.raises(EmbodimentExecutionError, match="motor_rps exceeds"):
        await executor._timed_motor_rps(  # pylint: disable=protected-access
            {
                "endpoint": "ws://127.0.0.1:1",
                "execution": {"max_abs_rps": 1.5},
                "arguments": {
                    "motor_rps": [1.51, 0.0, 0.0, 0.0],
                    "duration_seconds": 1.0,
                },
            }
        )


@pytest.mark.asyncio
async def test_servo_pulse_rejects_out_of_range_before_transport(tmp_path: Path) -> None:
    executor = EmbodimentExecutor(tmp_path)
    with pytest.raises(EmbodimentExecutionError, match="outside"):
        await executor._send_servo_pulses(  # pylint: disable=protected-access
            {
                "endpoint": "ws://127.0.0.1:1",
                "execution": {
                    "servo_ids": [1, 2],
                    "pulse_limits": [[0, 1000], [100, 900]],
                },
                "arguments": {"pulses": [500, 901]},
            }
        )


@pytest.mark.asyncio
async def test_jetrover_joint3_floor_is_rejected_before_transport(tmp_path: Path) -> None:
    executor = EmbodimentExecutor(tmp_path)
    with pytest.raises(EmbodimentExecutionError, match=r"servo 3 pulse 199.*\[200, 1000\]"):
        await executor._send_servo_pulses(  # pylint: disable=protected-access
            {
                "endpoint": "ws://127.0.0.1:1",
                "execution": {
                    "servo_ids": [1, 2, 3, 4, 5],
                    "pulse_limits": [[0, 1000], [0, 1000], [200, 1000], [0, 1000], [0, 1000]],
                },
                "arguments": {"pulses": [500, 500, 199, 500, 500]},
            }
        )


@pytest.mark.asyncio
async def test_native_servo_execution_rejects_offline_hardware_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qwenpaw.app.embodiments import executor as executor_module

    class FakeSocket:
        def __init__(self) -> None:
            self.sent = []

        async def send(self, raw: str) -> None:
            self.sent.append(json.loads(raw))

        async def recv(self) -> str:
            request = self.sent[-1]
            return json.dumps({
                "op": "service_response",
                "id": request["id"],
                "service": "/rosapi/nodes",
                "result": True,
                "values": {"nodes": ["/controller_manager"]},
            })

    class FakeConnection:
        async def __aenter__(self) -> FakeSocket:
            self.socket = FakeSocket()
            return self.socket

        async def __aexit__(self, *_args) -> None:
            return None

    monkeypatch.setattr(
        executor_module.websockets,
        "connect",
        lambda *_args, **_kwargs: FakeConnection(),
    )
    with pytest.raises(EmbodimentExecutionError, match="native hardware driver.*offline"):
        await EmbodimentExecutor(tmp_path)._send_servo_pulses(  # pylint: disable=protected-access
            {
                "endpoint": "ws://robot:9090",
                "execution": {
                    "servo_ids": [1, 2, 3, 4, 5],
                    "pulse_limits": [[0, 1000], [0, 1000], [200, 1000], [0, 1000], [0, 1000]],
                    "require_native_driver": True,
                },
                "arguments": {"pulses": [500, 500, 500, 500, 500]},
            }
        )


@pytest.mark.asyncio
async def test_servo_pulse_uses_native_unit_and_verifies_feedback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from qwenpaw.app.embodiments import executor as executor_module

    class FakeSocket:
        def __init__(self) -> None:
            self.sent = []
            self.feedback_count = 0

        async def send(self, raw: str) -> None:
            self.sent.append(json.loads(raw))

        async def recv(self) -> str:
            self.feedback_count += 1
            positions = [500, 500] if self.feedback_count == 1 else [620, 430]
            return json.dumps(
                {
                    "op": "publish",
                    "topic": "/controller_manager/servo_states",
                    "msg": {
                        "servo_state": [
                            {"id": 1, "position": positions[0]},
                            {"id": 2, "position": positions[1]},
                        ]
                    },
                }
            )

    class FakeConnection:
        def __init__(self, socket: FakeSocket) -> None:
            self.socket = socket

        async def __aenter__(self) -> FakeSocket:
            return self.socket

        async def __aexit__(self, *_args) -> None:
            return None

    socket = FakeSocket()
    monkeypatch.setattr(
        executor_module.websockets,
        "connect",
        lambda *_args, **_kwargs: FakeConnection(socket),
    )
    result = await EmbodimentExecutor(tmp_path)._send_servo_pulses(  # pylint: disable=protected-access
        {
            "endpoint": "ws://robot:9090",
            "execution": {
                "servo_ids": [1, 2],
                "pulse_limits": [[0, 1000], [0, 1000]],
                "max_pulse_delta": 200,
                "feedback_tolerance_pulse": 2,
            },
            "arguments": {"pulses": [620, 430], "duration_seconds": 2},
        }
    )

    assert result["status"] == "SUCCEEDED"
    publish = next(item for item in socket.sent if item["op"] == "publish")
    assert publish["msg"]["position_unit"] == "pulse"
    assert publish["msg"]["position"] == [
        {"id": 1, "position": 620.0},
        {"id": 2, "position": 430.0},
    ]


def test_jetrover_package_exposes_atomic_twist_tool() -> None:
    from embodiments.runtime.loader import EmbodimentLoader, EmbodimentPackage

    package_path = (
        Path(__file__).parents[3]
        / "embodiments"
        / "packages"
        / "jetrover_base"
    )
    package = EmbodimentPackage(package_path)
    loader = EmbodimentLoader()
    loader._available[package.package_id] = package  # pylint: disable=protected-access
    loader.activate(
        package.package_id,
        bindings={
            "camera_http": "http://127.0.0.1:8080",
            "robot_rosbridge": "ws://127.0.0.1:9090",
        },
    )
    tools = {tool["function"]["name"]: tool for tool in loader.get_tools()}

    assert "jetrover_base.move_relative" not in tools
    assert "jetrover_base.rotate_relative" not in tools
    assert "jetrover_base.follow_route" not in tools
    twist = tools["jetrover_base.set_base_twist"]
    assert twist["metadata"]["safety_class"] == "supervisor"
    assert twist["metadata"]["protocol"] == "rosbridge_timed_twist"
    assert twist["metadata"]["execution"]["max_duration_seconds"] == 10.0
    motors = tools["jetrover_base.set_motor_rps"]
    assert motors["metadata"]["protocol"] == "rosbridge_timed_motor_rps"
    assert motors["metadata"]["execution"]["max_abs_rps"] == 1.5
    assert tools["jetrover_base.stop_motors"]["metadata"]["safety_class"] == "safe"
    assert tools["jetrover_base.enable_motion"]["metadata"]["safety_class"] == "supervisor"
    assert tools["jetrover_base.stop_base"]["metadata"]["safety_class"] == "safe"


def test_jetrover_arm_package_exposes_native_pulse_tools() -> None:
    from embodiments.runtime.loader import EmbodimentLoader, EmbodimentPackage

    package_path = (
        Path(__file__).parents[3]
        / "embodiments"
        / "packages"
        / "jetrover_camera_arm"
    )
    package = EmbodimentPackage(package_path)
    loader = EmbodimentLoader()
    loader._available[package.package_id] = package  # pylint: disable=protected-access
    loader.activate(
        package.package_id,
        bindings={
            "camera_http": "http://127.0.0.1:8080",
            "robot_rosbridge": "ws://127.0.0.1:9090",
        },
    )
    tools = {tool["function"]["name"]: tool for tool in loader.get_tools()}

    assert tools["jetrover_arm.move_pulses"]["metadata"]["protocol"] == "rosbridge_servo_pulse"
    assert tools["jetrover_arm.move_pulses"]["metadata"]["execution"]["pulse_limits"][2] == [200, 1000]
    assert tools["jetrover_gripper.set_pulse"]["metadata"]["protocol"] == "rosbridge_servo_pulse"
    assert tools["jetrover_depth_camera.capture_depth"]["metadata"]["protocol"] == "rosbridge_compressed_image_once"
    assert tools["jetrover_kinematics.solve_pose_ik"]["metadata"]["protocol"] == "rosbridge_service"
    assert tools["jetrover_kinematics.solve_pose_ik"]["metadata"]["execution"]["pass_arguments"] is True
