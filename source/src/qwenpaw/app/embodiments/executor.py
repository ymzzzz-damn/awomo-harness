"""awomopaw transport executor for embodiment dispatch metadata."""

from __future__ import annotations

import asyncio
import base64
import json
import math
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urljoin

import httpx
import websockets
from agentscope.message import ImageBlock, TextBlock
from agentscope.tool import ToolResponse

from ...constant import WORKING_DIR


class EmbodimentExecutionError(RuntimeError):
    """Raised when a physical device operation cannot be completed safely."""


class EmbodimentExecutor:
    """Execute the small transport vocabulary supported by awomopaw."""

    def __init__(self, workspace_dir: Path | None = None) -> None:
        root = workspace_dir or (WORKING_DIR / "working")
        self.capture_dir = Path(root) / "embodiments" / "captures"

    async def execute(self, dispatch: dict[str, Any]) -> ToolResponse:
        protocol = dispatch.get("protocol")
        try:
            if protocol == "http_snapshot":
                return await self._http_snapshot(dispatch)
            if protocol == "rosbridge_compressed_image_once":
                return await self._rosbridge_compressed_image_once(dispatch)
            if protocol == "http":
                value = await self._http_json(dispatch)
            elif protocol == "rosbridge_subscribe_once":
                value = await self._subscribe_once(dispatch)
            elif protocol == "rosbridge_service":
                value = await self._call_service(dispatch)
            elif protocol == "rosbridge_action":
                value = await self._send_action(dispatch)
            elif protocol == "rosbridge_servo_pulse":
                value = await self._send_servo_pulses(dispatch)
            elif protocol == "rosbridge_publish":
                value = await self._publish(dispatch)
            elif protocol == "rosbridge_publish_arguments":
                value = await self._publish_arguments(dispatch)
            elif protocol == "rosbridge_timed_twist":
                value = await self._timed_twist(dispatch)
            elif protocol == "rosbridge_timed_motor_rps":
                value = await self._timed_motor_rps(dispatch)
            elif protocol == "rosbridge_bounded_motion":
                value = await self._bounded_motion(dispatch)
            else:
                raise EmbodimentExecutionError(
                    f"Unsupported embodiment protocol: {protocol!r}",
                )
            return self._json_response(value)
        except Exception as exc:
            return ToolResponse(
                content=[TextBlock(type="text", text=f"Device operation failed: {exc}")],
                metadata={"success": False, "error": str(exc)},
            )

    @staticmethod
    def _json_response(value: Any) -> ToolResponse:
        return ToolResponse(
            content=[
                TextBlock(
                    type="text",
                    text=json.dumps(value, ensure_ascii=False, indent=2),
                ),
            ],
            metadata={"success": True, "result": value},
        )

    @staticmethod
    def _endpoint(dispatch: dict[str, Any]) -> str:
        endpoint = str(dispatch.get("endpoint") or "").rstrip("/")
        if not endpoint or endpoint.startswith("@role:"):
            raise EmbodimentExecutionError("Device endpoint is not bound")
        return endpoint

    async def _http_snapshot(self, dispatch: dict[str, Any]) -> ToolResponse:
        endpoint = self._endpoint(dispatch)
        execution = dispatch.get("execution") or {}
        topic = execution.get("topic")
        if not topic:
            raise EmbodimentExecutionError("Camera topic is missing")
        url = urljoin(endpoint + "/", str(dispatch.get("path") or "snapshot").lstrip("/"))
        url = f"{url}?{urlencode({'topic': str(topic)}, safe='/')}"
        timeout = min(float(execution.get("timeout_seconds", 8)), 15.0)
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            response = await client.get(url)
            response.raise_for_status()
        content_type = response.headers.get("content-type", "").split(";", 1)[0]
        if content_type not in {"image/jpeg", "image/png"}:
            raise EmbodimentExecutionError(
                f"Unexpected camera content type: {content_type or 'missing'}",
            )
        if not response.content or len(response.content) > 12_000_000:
            raise EmbodimentExecutionError("Camera image is empty or exceeds 12 MB")

        self.capture_dir.mkdir(parents=True, exist_ok=True)
        suffix = ".png" if content_type == "image/png" else ".jpg"
        path = self.capture_dir / f"snapshot-{int(time.time())}-{uuid.uuid4().hex[:8]}{suffix}"
        path.write_bytes(response.content)
        return ToolResponse(
            content=[
                ImageBlock(type="image", source={"type": "url", "url": str(path)}),
                TextBlock(type="text", text=f"Camera frame saved to {path}"),
            ],
            metadata={"success": True, "path": str(path), "topic": topic},
        )

    async def _rosbridge_compressed_image_once(
        self, dispatch: dict[str, Any]
    ) -> ToolResponse:
        """Capture one ROS CompressedImage, including compressedDepth PNG data."""
        execution = dispatch.get("execution") or {}
        topic = str(execution.get("topic") or dispatch.get("path") or "")
        if not topic:
            raise EmbodimentExecutionError("Compressed image topic is missing")
        timeout = min(float(execution.get("timeout_seconds", 8)), 15.0)
        request_id = f"qwenpaw-image-{uuid.uuid4().hex}"
        request = {
            "op": "subscribe",
            "id": request_id,
            "topic": topic,
            "type": "sensor_msgs/msg/CompressedImage",
            "queue_length": 1,
            "throttle_rate": 0,
        }
        payload: dict[str, Any] | None = None
        async with websockets.connect(
            self._endpoint(dispatch),
            open_timeout=timeout,
            close_timeout=2,
            max_size=16_000_000,
        ) as socket:
            await socket.send(json.dumps(request))
            try:
                while True:
                    raw = await asyncio.wait_for(socket.recv(), timeout)
                    message = json.loads(raw)
                    if message.get("op") == "publish" and message.get("topic") == topic:
                        payload = message.get("msg") or {}
                        break
            finally:
                await socket.send(
                    json.dumps(
                        {"op": "unsubscribe", "id": request_id, "topic": topic}
                    )
                )
        encoded = payload.get("data") if payload else None
        if not isinstance(encoded, str):
            raise EmbodimentExecutionError("Compressed image data is missing")
        try:
            raw_image = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError) as exc:
            raise EmbodimentExecutionError("Compressed image data is invalid base64") from exc
        if not raw_image or len(raw_image) > 12_000_000:
            raise EmbodimentExecutionError("Compressed image is empty or exceeds 12 MB")

        # compressed_depth_image_transport prepends a 12-byte header before its
        # PNG stream. Searching for the signature also accepts ordinary PNG data.
        png_signature = b"\x89PNG\r\n\x1a\n"
        png_offset = raw_image.find(png_signature)
        jpeg_offset = raw_image.find(b"\xff\xd8\xff")
        if png_offset >= 0:
            image_bytes, suffix = raw_image[png_offset:], ".png"
        elif jpeg_offset >= 0:
            image_bytes, suffix = raw_image[jpeg_offset:], ".jpg"
        else:
            raise EmbodimentExecutionError("Compressed image is neither PNG nor JPEG")

        self.capture_dir.mkdir(parents=True, exist_ok=True)
        path = self.capture_dir / (
            f"jetrover-{int(time.time())}-{uuid.uuid4().hex[:8]}{suffix}"
        )
        path.write_bytes(image_bytes)
        return ToolResponse(
            content=[
                ImageBlock(type="image", source={"type": "url", "url": str(path)}),
                TextBlock(type="text", text=f"JetRover compressed frame saved to {path}"),
            ],
            metadata={
                "success": True,
                "path": str(path),
                "topic": topic,
                "format": payload.get("format") if payload else None,
            },
        )

    async def _http_json(self, dispatch: dict[str, Any]) -> Any:
        endpoint = self._endpoint(dispatch)
        execution = dispatch.get("execution") or {}
        url = urljoin(
            endpoint + "/", str(dispatch.get("path") or "").lstrip("/")
        )
        method = str(dispatch.get("method") or "GET").upper()
        if method not in {"GET", "POST"}:
            raise EmbodimentExecutionError(f"Unsupported HTTP method: {method}")
        # Long-running robot trajectory endpoints legitimately block until the
        # motion finishes. Honour their declared timeout while retaining a hard
        # upper bound for generic HTTP tools.
        timeout = min(float(execution.get("timeout_seconds", 8)), 60.0)
        arguments = dispatch.get("arguments") or {}
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            if method == "GET":
                response = await client.get(url, params=arguments)
            else:
                response = await client.post(url, json=arguments)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                # Preserve the device/gateway explanation (for example Franka
                # mode="Reflex") so the planning agent can select recovery or
                # replan instead of seeing only an opaque HTTP 500.
                try:
                    payload = response.json()
                except Exception:  # malformed or non-JSON device response
                    payload = None
                if isinstance(payload, dict):
                    detail = payload.get("error") or payload.get("message")
                else:
                    detail = None
                detail = str(detail or response.text or "no response body")[:1000]
                raise EmbodimentExecutionError(
                    f"HTTP {response.status_code} from device gateway: {detail}"
                ) from exc
        if len(response.content) > 2_000_000:
            raise EmbodimentExecutionError("HTTP response exceeds 2 MB")
        content_type = response.headers.get("content-type", "")
        if "json" in content_type:
            return response.json()
        return {"status_code": response.status_code, "text": response.text}

    async def _subscribe_once(self, dispatch: dict[str, Any]) -> Any:
        execution = dispatch.get("execution") or {}
        topic = execution.get("topic") or dispatch.get("path")
        message_type = execution.get("message_type")
        timeout = min(float(execution.get("timeout_seconds", 5)), 10.0)
        request_id = f"qwenpaw-sub-{uuid.uuid4().hex}"
        request = {
            "op": "subscribe",
            "id": request_id,
            "topic": topic,
            "queue_length": 1,
            "throttle_rate": 0,
        }
        if message_type:
            request["type"] = message_type

        async with websockets.connect(
            self._endpoint(dispatch),
            open_timeout=timeout,
            close_timeout=2,
            max_size=16_000_000,
        ) as socket:
            await socket.send(json.dumps(request))
            try:
                while True:
                    raw = await asyncio.wait_for(socket.recv(), timeout)
                    message = json.loads(raw)
                    if message.get("op") == "publish" and message.get("topic") == topic:
                        return message.get("msg")
            finally:
                await socket.send(
                    json.dumps({"op": "unsubscribe", "id": request_id, "topic": topic}),
                )

    async def _call_service(self, dispatch: dict[str, Any]) -> Any:
        execution = dispatch.get("execution") or {}
        service = execution.get("service") or dispatch.get("path")
        timeout = min(float(execution.get("timeout_seconds", 5)), 10.0)
        request_id = f"qwenpaw-svc-{uuid.uuid4().hex}"
        args = dict(execution.get("args") or {})
        if execution.get("pass_arguments", False):
            args.update(dispatch.get("arguments") or {})
        request = {
            "op": "call_service",
            "id": request_id,
            "service": service,
            "args": args,
        }
        async with websockets.connect(
            self._endpoint(dispatch), open_timeout=timeout, close_timeout=2
        ) as socket:
            await socket.send(json.dumps(request))
            while True:
                raw = await asyncio.wait_for(socket.recv(), timeout)
                message = json.loads(raw)
                if message.get("id") == request_id:
                    if not message.get("result", False):
                        raise EmbodimentExecutionError(
                            f"ROS service {service} failed: {message.get('values')}",
                        )
                    return message.get("values")

    @staticmethod
    def _bounded_positions(
        arguments: dict[str, Any], execution: dict[str, Any]
    ) -> tuple[list[str], list[float], float]:
        names = list(execution.get("joint_names") or [])
        positions = arguments.get("positions")
        if not isinstance(positions, list) or len(positions) != len(names):
            raise EmbodimentExecutionError(
                f"positions must contain exactly {len(names)} values",
            )
        values = [float(value) for value in positions]
        if not all(math.isfinite(value) for value in values):
            raise EmbodimentExecutionError("All joint positions must be finite")
        limits = execution.get("joint_limits") or []
        if len(limits) != len(values):
            raise EmbodimentExecutionError("Joint limit metadata is incomplete")
        for name, value, limit in zip(names, values, limits):
            low, high = map(float, limit)
            if value < low or value > high:
                raise EmbodimentExecutionError(
                    f"{name}={value} is outside [{low}, {high}]",
                )
        duration = float(arguments.get("duration_seconds", 2.0))
        max_duration = min(float(execution.get("max_duration_seconds", 10)), 30.0)
        if not math.isfinite(duration) or duration < 0.5 or duration > max_duration:
            raise EmbodimentExecutionError(
                f"duration_seconds must be between 0.5 and {max_duration}",
            )
        return names, values, duration

    async def _send_action(self, dispatch: dict[str, Any]) -> Any:
        execution = dispatch.get("execution") or {}
        names, positions, duration = self._bounded_positions(
            dispatch.get("arguments") or {}, execution
        )
        seconds = int(duration)
        nanoseconds = int(round((duration - seconds) * 1_000_000_000))
        action = execution.get("action") or dispatch.get("path")
        request_id = f"qwenpaw-action-{uuid.uuid4().hex}"
        request = {
            "op": "send_action_goal",
            "id": request_id,
            "action": action,
            "action_type": execution.get(
                "action_type", "control_msgs/action/FollowJointTrajectory"
            ),
            "feedback": True,
            "args": {
                "trajectory": {
                    "joint_names": names,
                    "points": [
                        {
                            "positions": positions,
                            "time_from_start": {"sec": seconds, "nanosec": nanoseconds},
                        },
                    ],
                },
            },
        }
        timeout = duration + min(float(execution.get("timeout_margin_seconds", 3)), 5.0)
        endpoint = self._endpoint(dispatch)
        action_sent = False
        try:
            async with websockets.connect(
                endpoint, open_timeout=5, close_timeout=2
            ) as socket:
                try:
                    await socket.send(json.dumps(request))
                    action_sent = True
                    deadline = asyncio.get_running_loop().time() + timeout
                    while True:
                        remaining = deadline - asyncio.get_running_loop().time()
                        if remaining <= 0:
                            raise asyncio.TimeoutError
                        raw = await asyncio.wait_for(socket.recv(), remaining)
                        message = json.loads(raw)
                        if (
                            message.get("op") == "action_result"
                            and message.get("id") == request_id
                        ):
                            if not message.get("result", False):
                                raise EmbodimentExecutionError(
                                    str(message.get("values")),
                                )
                            return {
                                "action": action,
                                "status": message.get("status"),
                                "result": message.get("values"),
                            }
                except BaseException:
                    if action_sent:
                        await asyncio.shield(
                            self._cancel_action(socket, action, request_id),
                        )
                    raise
        except asyncio.TimeoutError as exc:
            await self._publish_stop(endpoint, execution.get("stop_servo_ids") or [])
            raise EmbodimentExecutionError(
                f"Action timed out after {timeout:.1f}s; stop command sent",
            ) from exc
        except asyncio.CancelledError:
            if action_sent:
                await asyncio.shield(
                    self._publish_stop(
                        endpoint,
                        execution.get("stop_servo_ids") or [],
                    )
                )
            raise
        except Exception:
            if action_sent:
                await self._publish_stop(
                    endpoint, execution.get("stop_servo_ids") or []
                )
            raise

    @staticmethod
    async def _cancel_action(socket: Any, action: str, request_id: str) -> None:
        """Best-effort cancellation on the same rosbridge connection."""
        try:
            await socket.send(
                json.dumps(
                    {
                        "op": "cancel_action_goal",
                        "id": request_id,
                        "action": action,
                    },
                ),
            )
        except Exception:
            pass

    async def _send_servo_pulses(self, dispatch: dict[str, Any]) -> Any:
        """Send one bounded JetRover native-pulse command and verify feedback.

        The vehicle kinematics node returns 0..1000 bus-servo pulses.  Sending
        those values through a guessed radian calibration can reject a valid IK
        solution or command the wrong pose, so this protocol keeps the unit
        explicit all the way to the vendor ``servo_controller`` topic.
        """
        execution = dispatch.get("execution") or {}
        arguments = dispatch.get("arguments") or {}
        servo_ids = [int(value) for value in execution.get("servo_ids") or []]
        pulses = arguments.get("pulses")
        if not servo_ids:
            raise EmbodimentExecutionError("servo_ids metadata is missing")
        if not isinstance(pulses, list) or len(pulses) != len(servo_ids):
            raise EmbodimentExecutionError(
                f"pulses must contain exactly {len(servo_ids)} values",
            )
        normalized: list[int] = []
        pulse_limits = execution.get("pulse_limits") or [[0, 1000]] * len(servo_ids)
        if len(pulse_limits) != len(servo_ids):
            raise EmbodimentExecutionError("pulse limit metadata is incomplete")
        for servo_id, raw, limit in zip(servo_ids, pulses, pulse_limits):
            value = float(raw)
            rounded = int(round(value))
            low, high = map(int, limit)
            if not math.isfinite(value) or abs(value - rounded) > 1e-6:
                raise EmbodimentExecutionError(
                    f"servo {servo_id} pulse must be a finite integer",
                )
            if rounded < low or rounded > high:
                raise EmbodimentExecutionError(
                    f"servo {servo_id} pulse {rounded} is outside [{low}, {high}]",
                )
            normalized.append(rounded)

        duration = float(arguments.get("duration_seconds", 2.0))
        max_duration = min(float(execution.get("max_duration_seconds", 10.0)), 30.0)
        if not math.isfinite(duration) or duration < 0.5 or duration > max_duration:
            raise EmbodimentExecutionError(
                f"duration_seconds must be between 0.5 and {max_duration}",
            )

        endpoint = self._endpoint(dispatch)
        topic = str(execution.get("topic") or "/servo_controller")
        message_type = str(
            execution.get("message_type")
            or "servo_controller_msgs/msg/ServosPosition"
        )
        feedback_topic = str(
            execution.get("feedback_topic")
            or "/controller_manager/servo_states"
        )
        feedback_type = str(
            execution.get("feedback_type")
            or "servo_controller_msgs/msg/ServoStateList"
        )
        max_delta = int(execution.get("max_pulse_delta", 450))
        tolerance = int(execution.get("feedback_tolerance_pulse", 18))
        request_id = f"qwenpaw-pulse-{uuid.uuid4().hex}"
        command_sent = False
        require_native_driver = bool(execution.get("require_native_driver", False))
        native_driver_node = str(
            execution.get("native_driver_node") or "/ros_robot_controller"
        )
        native_position_service = str(
            execution.get("native_position_service")
            or "/ros_robot_controller/bus_servo/get_state"
        )

        try:
            async with websockets.connect(
                endpoint, open_timeout=5, close_timeout=2
            ) as socket:
                async def call_service_values(
                    service: str, args: dict[str, Any], timeout: float = 5.0
                ) -> Any:
                    service_id = f"qwenpaw-native-{uuid.uuid4().hex}"
                    await socket.send(json.dumps({
                        "op": "call_service",
                        "id": service_id,
                        "service": service,
                        "args": args,
                    }))
                    while True:
                        raw = await asyncio.wait_for(socket.recv(), timeout)
                        response = json.loads(raw)
                        if response.get("id") != service_id:
                            continue
                        if not response.get("result", False):
                            raise EmbodimentExecutionError(
                                f"native ROS service {service} failed: "
                                f"{response.get('values')}",
                            )
                        return response.get("values") or {}

                async def native_positions() -> dict[int, int]:
                    commands = [
                        {
                            "id": servo_id,
                            "get_id": 0,
                            "get_position": 1,
                            "get_offset": 0,
                            "get_voltage": 0,
                            "get_temperature": 0,
                            "get_position_limit": 0,
                            "get_voltage_limit": 0,
                            "get_max_temperature_limit": 0,
                            "get_torque_state": 0,
                        }
                        for servo_id in servo_ids
                    ]
                    values = await call_service_values(
                        native_position_service, {"cmd": commands}
                    )
                    states = values.get("state") or []
                    if not bool(values.get("success", False)) or len(states) != len(servo_ids):
                        raise EmbodimentExecutionError(
                            "JetRover native bus-servo state query returned no complete state",
                        )
                    positions: dict[int, int] = {}
                    for servo_id, state in zip(servo_ids, states):
                        raw_position = state.get("position") if isinstance(state, dict) else None
                        if not isinstance(raw_position, list) or not raw_position:
                            raise EmbodimentExecutionError(
                                f"JetRover native servo {servo_id} did not return a position; "
                                "the servo power/driver may be offline",
                            )
                        positions[servo_id] = int(raw_position[0])
                    return positions

                native_before: dict[int, int] | None = None
                if require_native_driver:
                    nodes = await call_service_values("/rosapi/nodes", {})
                    if native_driver_node not in (nodes.get("nodes") or []):
                        raise EmbodimentExecutionError(
                            f"JetRover native hardware driver {native_driver_node} is offline; "
                            "no servo command was sent",
                        )
                    native_before = await native_positions()

                await socket.send(
                    json.dumps(
                        {
                            "op": "subscribe",
                            "id": request_id + "-feedback",
                            "topic": feedback_topic,
                            "type": feedback_type,
                            "queue_length": 1,
                            "throttle_rate": 0,
                        },
                    ),
                )

                async def feedback_positions(timeout: float) -> dict[int, int]:
                    while True:
                        raw = await asyncio.wait_for(socket.recv(), timeout)
                        message = json.loads(raw)
                        if (
                            message.get("op") != "publish"
                            or message.get("topic") != feedback_topic
                        ):
                            continue
                        states = (message.get("msg") or {}).get("servo_state") or []
                        return {
                            int(item["id"]): int(item["position"])
                            for item in states
                            if isinstance(item, dict)
                            and "id" in item
                            and "position" in item
                        }

                before = native_before or await feedback_positions(3.0)
                missing = [servo_id for servo_id in servo_ids if servo_id not in before]
                if missing:
                    raise EmbodimentExecutionError(
                        f"fresh servo feedback is missing ids {missing}",
                    )
                for servo_id, target in zip(servo_ids, normalized):
                    delta = abs(target - before[servo_id])
                    if delta > max_delta:
                        raise EmbodimentExecutionError(
                            f"servo {servo_id} pulse delta {delta} exceeds {max_delta}",
                        )

                await socket.send(
                    json.dumps(
                        {
                            "op": "publish",
                            "topic": topic,
                            "type": message_type,
                            "msg": {
                                "duration": duration,
                                "position_unit": "pulse",
                                "position": [
                                    {"id": servo_id, "position": float(pulse)}
                                    for servo_id, pulse in zip(servo_ids, normalized)
                                ],
                            },
                        },
                    ),
                )
                command_sent = True
                deadline = asyncio.get_running_loop().time() + duration + min(
                    float(execution.get("timeout_margin_seconds", 3.0)), 5.0
                )
                if require_native_driver:
                    await asyncio.sleep(duration)
                    while asyncio.get_running_loop().time() < deadline:
                        observed = await native_positions()
                        if all(
                            servo_id in observed
                            and abs(observed[servo_id] - target) <= tolerance
                            for servo_id, target in zip(servo_ids, normalized)
                        ):
                            return {
                                "status": "SUCCEEDED_NATIVE_HARDWARE_FEEDBACK",
                                "topic": topic,
                                "joint_units": "servo_pulse",
                                "servo_ids": servo_ids,
                                "targets": normalized,
                                "observed": [observed[servo_id] for servo_id in servo_ids],
                                "feedback_source": native_position_service,
                            }
                        await asyncio.sleep(0.1)
                    raise asyncio.TimeoutError
                observed: dict[int, int] = {}
                while asyncio.get_running_loop().time() < deadline:
                    remaining = deadline - asyncio.get_running_loop().time()
                    observed = await feedback_positions(max(0.05, remaining))
                    if all(
                        servo_id in observed
                        and abs(observed[servo_id] - target) <= tolerance
                        for servo_id, target in zip(servo_ids, normalized)
                    ):
                        return {
                            "status": "SUCCEEDED",
                            "topic": topic,
                            "joint_units": "servo_pulse",
                            "servo_ids": servo_ids,
                            "targets": normalized,
                            "observed": [observed[servo_id] for servo_id in servo_ids],
                        }
                raise asyncio.TimeoutError
        except asyncio.TimeoutError as exc:
            if command_sent:
                await self._publish_stop(endpoint, servo_ids)
            raise EmbodimentExecutionError(
                "Servo pulse command timed out; stop command sent",
            ) from exc
        except asyncio.CancelledError:
            if command_sent:
                await asyncio.shield(self._publish_stop(endpoint, servo_ids))
            raise
        except Exception:
            if command_sent:
                await self._publish_stop(endpoint, servo_ids)
            raise

    async def _publish(self, dispatch: dict[str, Any]) -> Any:
        execution = dispatch.get("execution") or {}
        topic = execution.get("topic") or dispatch.get("path")
        message = execution.get("message")
        if not isinstance(message, dict):
            raise EmbodimentExecutionError("Fixed ROS publish message is missing")
        await self._rosbridge_publish(
            self._endpoint(dispatch),
            topic,
            execution.get("message_type"),
            message,
        )
        return {"published": True, "topic": topic}

    async def _publish_arguments(self, dispatch: dict[str, Any]) -> Any:
        """Publish validated tool arguments as one native ROS message."""
        execution = dispatch.get("execution") or {}
        topic = execution.get("topic") or dispatch.get("path")
        arguments = dispatch.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise EmbodimentExecutionError("ROS publish arguments must be an object")
        message = dict(execution.get("message_defaults") or {})
        message.update(arguments)
        await self._rosbridge_publish(
            self._endpoint(dispatch),
            topic,
            execution.get("message_type"),
            message,
        )
        return {"published": True, "topic": topic, "message": message}

    async def _timed_twist(self, dispatch: dict[str, Any]) -> Any:
        """Publish one bounded base velocity setpoint for a finite duration.

        This is the lowest useful mobile-base primitive exposed to the model:
        the model selects linear/angular velocity and duration, while this
        transport only maintains the ROS watchdog cadence and guarantees a
        zero Twist on completion, cancellation, timeout, or socket failure.
        Distance and angle interpretation deliberately stay outside this
        executor so the agent may close the loop with odometry or vision.
        """
        execution = dispatch.get("execution") or {}
        arguments = dispatch.get("arguments") or {}
        endpoint = self._endpoint(dispatch)
        topic = str(execution.get("topic") or dispatch.get("path") or "")
        if not topic:
            raise EmbodimentExecutionError("Twist topic is missing")

        try:
            linear_x = float(arguments.get("linear_x_mps", 0.0))
            angular_z = float(arguments.get("angular_z_rps", 0.0))
            duration = float(arguments["duration_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbodimentExecutionError(
                "linear_x_mps, angular_z_rps, and duration_seconds must be numeric",
            ) from exc
        if not all(math.isfinite(value) for value in (linear_x, angular_z, duration)):
            raise EmbodimentExecutionError("Twist arguments must be finite")

        max_linear = min(float(execution.get("max_linear_speed_mps", 0.18)), 0.30)
        max_angular = min(float(execution.get("max_angular_speed_rps", 0.30)), 0.60)
        max_duration = min(float(execution.get("max_duration_seconds", 10.0)), 30.0)
        allow_reverse = bool(execution.get("allow_reverse", False))
        if linear_x < 0.0 and not allow_reverse:
            raise EmbodimentExecutionError("negative linear_x_mps is disabled")
        if abs(linear_x) > max_linear:
            raise EmbodimentExecutionError(
                f"linear_x_mps exceeds {max_linear:.3f}",
            )
        if abs(angular_z) > max_angular:
            raise EmbodimentExecutionError(
                f"angular_z_rps exceeds {max_angular:.3f}",
            )
        if duration < 0.05 or duration > max_duration:
            raise EmbodimentExecutionError(
                f"duration_seconds must be between 0.05 and {max_duration:.2f}",
            )

        publish_hz = min(max(float(execution.get("publish_hz", 10.0)), 5.0), 30.0)
        period = 1.0 / publish_hz
        message_type = str(execution.get("message_type") or "geometry_msgs/msg/Twist")
        moving_message = {
            "linear": {"x": linear_x, "y": 0.0, "z": 0.0},
            "angular": {"x": 0.0, "y": 0.0, "z": angular_z},
        }
        zero_message = {
            "linear": {"x": 0.0, "y": 0.0, "z": 0.0},
            "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
        }
        request_base = {"op": "publish", "topic": topic, "type": message_type}
        enable = dispatch.get("motion_enable")
        if enable and (enable.get("protocol") != "rosbridge_service"
                       or self._endpoint(enable) != endpoint):
            raise EmbodimentExecutionError("Motion enable must use the same ROS bridge")
        sent = 0
        async with websockets.connect(
            endpoint,
            open_timeout=5,
            close_timeout=2,
        ) as socket:
            try:
                if enable:
                    config = enable.get("execution") or {}
                    service = config.get("service") or enable.get("path")
                    args = dict(config.get("args") or {})
                    if config.get("pass_arguments"):
                        args.update(enable.get("arguments") or {})
                    request_id = "qwenpaw-motion-enable-" + uuid.uuid4().hex
                    await socket.send(json.dumps({"op": "call_service", "id": request_id,
                                                  "service": service, "args": args}))
                    enable_deadline = asyncio.get_running_loop().time() + 5.0
                    while True:
                        remaining = enable_deadline - asyncio.get_running_loop().time()
                        if remaining <= 0:
                            raise EmbodimentExecutionError("Motion enable timed out")
                        reply = json.loads(await asyncio.wait_for(socket.recv(), remaining))
                        if reply.get("id") != request_id:
                            continue
                        if not reply.get("result") or not (reply.get("values") or {}).get("success"):
                            raise EmbodimentExecutionError("Vehicle rejected motion enable")
                        break
                deadline = asyncio.get_running_loop().time() + duration
                while True:
                    now = asyncio.get_running_loop().time()
                    if now >= deadline:
                        break
                    await socket.send(json.dumps({**request_base, "msg": moving_message}))
                    sent += 1
                    await asyncio.sleep(min(period, max(0.0, deadline - now)))
            finally:
                # Repeating zero makes the stop robust against a single dropped
                # rosbridge packet; the vehicle-side watchdog remains active.
                for _ in range(3):
                    try:
                        await asyncio.shield(
                            socket.send(json.dumps({**request_base, "msg": zero_message})),
                        )
                        await asyncio.sleep(0.02)
                    except Exception:
                        break
        return {
            "status": "COMPLETED_ZERO_SENT",
            "topic": topic,
            "linear_x_mps": linear_x,
            "angular_z_rps": angular_z,
            "duration_seconds": duration,
            "publish_count": sent,
            "motion_confirmed": False,
            "next_action": "verify fresh odometry with navigation_record_segment_result",
        }

    async def _timed_motor_rps(self, dispatch: dict[str, Any]) -> Any:
        """Publish four native motor RPS values for a finite duration, then zero."""
        execution = dispatch.get("execution") or {}
        arguments = dispatch.get("arguments") or {}
        endpoint = self._endpoint(dispatch)
        topic = str(
            execution.get("topic")
            or dispatch.get("path")
            or "/ros_robot_controller/set_motor"
        )
        raw_rps = arguments.get("motor_rps")
        if not isinstance(raw_rps, list) or len(raw_rps) != 4:
            raise EmbodimentExecutionError("motor_rps must contain exactly four values")
        try:
            values = [float(value) for value in raw_rps]
            duration = float(arguments["duration_seconds"])
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbodimentExecutionError(
                "motor_rps and duration_seconds must be numeric",
            ) from exc
        if not all(math.isfinite(value) for value in [*values, duration]):
            raise EmbodimentExecutionError("Motor arguments must be finite")
        max_rps = min(float(execution.get("max_abs_rps", 1.5)), 2.0)
        max_duration = min(float(execution.get("max_duration_seconds", 10.0)), 30.0)
        if any(abs(value) > max_rps for value in values):
            raise EmbodimentExecutionError(f"motor_rps exceeds ±{max_rps:.3f}")
        if duration < 0.05 or duration > max_duration:
            raise EmbodimentExecutionError(
                f"duration_seconds must be between 0.05 and {max_duration:.2f}",
            )

        message_type = str(
            execution.get("message_type")
            or "ros_robot_controller_msgs/msg/MotorsState"
        )
        request_base = {"op": "publish", "topic": topic, "type": message_type}
        moving = {
            "data": [
                {"id": motor_id, "rps": value}
                for motor_id, value in enumerate(values, start=1)
            ]
        }
        zero = {
            "data": [
                {"id": motor_id, "rps": 0.0}
                for motor_id in range(1, 5)
            ]
        }
        publish_hz = min(max(float(execution.get("publish_hz", 10.0)), 5.0), 30.0)
        period = 1.0 / publish_hz
        sent = 0
        async with websockets.connect(endpoint, open_timeout=5, close_timeout=2) as socket:
            deadline = asyncio.get_running_loop().time() + duration
            try:
                while asyncio.get_running_loop().time() < deadline:
                    await socket.send(json.dumps({**request_base, "msg": moving}))
                    sent += 1
                    await asyncio.sleep(period)
            finally:
                for _ in range(3):
                    try:
                        await asyncio.shield(
                            socket.send(json.dumps({**request_base, "msg": zero})),
                        )
                        await asyncio.sleep(0.02)
                    except Exception:
                        break
        return {
            "status": "COMPLETED_ZERO_SENT",
            "topic": topic,
            "motor_rps": values,
            "duration_seconds": duration,
            "publish_count": sent,
        }

    async def _bounded_motion(self, dispatch: dict[str, Any]) -> Any:
        """Execute one short, odometry-closed-loop base motion.

        This deliberately does not expose a streaming velocity tool to the
        model.  The executor owns the publish loop, watchdog, laser stop, and
        final zero command.  It is suitable for supervised embodiment tools;
        route planners should call it once per short segment.
        """
        execution = dispatch.get("execution") or {}
        args = dispatch.get("arguments") or {}
        endpoint = self._endpoint(dispatch)
        topic = str(
            execution.get("proposal_topic", "/qwenpaw/base/proposed_cmd_vel")
        )
        odom_topic = str(execution.get("odom_topic", "/odom"))
        scan_topic = str(execution.get("scan_topic", "/scan_raw"))
        safety_topic = str(
            execution.get("safety_status_topic", "/qwenpaw/base/safety/status")
        )
        enable_service = str(
            execution.get("enable_service", "/vln_cmd_vel_safety/enable")
        )
        stop_service = str(
            execution.get("stop_service", "/vln_cmd_vel_safety/stop")
        )
        target_kind = str(execution.get("target_kind", "linear"))
        route_points: list[tuple[float, float]] = []
        route_goal_tolerance = 0.20
        if target_kind == "route":
            raw_waypoints = args.get("waypoints")
            if not isinstance(raw_waypoints, list) or len(raw_waypoints) < 2:
                raise EmbodimentExecutionError(
                    "waypoints must contain at least two points",
                )
            for waypoint in raw_waypoints:
                if not isinstance(waypoint, dict):
                    raise EmbodimentExecutionError("each waypoint must be an object")
                try:
                    x = float(waypoint["x"])
                    y = float(waypoint["y"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise EmbodimentExecutionError(
                        "each waypoint requires numeric x and y",
                    ) from exc
                if not math.isfinite(x) or not math.isfinite(y):
                    raise EmbodimentExecutionError("waypoint coordinates must be finite")
                route_points.append((x, y))
            route_length = sum(
                math.hypot(x2 - x1, y2 - y1)
                for (x1, y1), (x2, y2) in zip(route_points, route_points[1:])
            )
            max_route_length = min(
                float(execution.get("max_route_length_m", 100.0)), 100.0
            )
            if not math.isfinite(route_length) or route_length < 0.01:
                raise EmbodimentExecutionError("route length must be positive")
            if route_length > max_route_length:
                raise EmbodimentExecutionError(
                    f"route length must not exceed {max_route_length:.2f} m",
                )
            speed = float(args.get("max_speed_mps", 0.14))
            max_speed = min(float(execution.get("max_linear_speed_mps", 0.12)), 0.30)
            if not math.isfinite(speed) or speed <= 0.0 or speed > max_speed:
                raise EmbodimentExecutionError(
                    f"max_speed_mps must be between 0 and {max_speed}",
                )
            route_goal_tolerance = float(args.get("goal_tolerance_m", 0.20))
            if not math.isfinite(route_goal_tolerance) or not 0.10 <= route_goal_tolerance <= 0.50:
                raise EmbodimentExecutionError(
                    "goal_tolerance_m must be between 0.10 and 0.50",
                )
            target = route_length
        elif target_kind == "linear":
            target = float(args.get("distance_m", 0.0))
            speed = float(args.get("max_speed_mps", 0.08))
            max_target = min(float(execution.get("max_distance_m", 100.0)), 100.0)
            max_speed = min(float(execution.get("max_linear_speed_mps", 0.12)), 0.30)
            if not math.isfinite(target) or target < 0.01 or target > max_target:
                raise EmbodimentExecutionError(
                    f"distance_m must be between 0.01 and {max_target}",
                )
            if not math.isfinite(speed) or speed <= 0.0 or speed > max_speed:
                raise EmbodimentExecutionError(
                    f"max_speed_mps must be between 0 and {max_speed}",
                )
        elif target_kind == "rotate":
            target = float(args.get("yaw_rad", 0.0))
            speed = float(args.get("max_speed_rps", 0.20))
            max_target = min(float(execution.get("max_yaw_rad", 6.283185307179586)), 6.283185307179586)
            max_speed = min(float(execution.get("max_angular_speed_rps", 0.30)), 1.0)
            if not math.isfinite(target) or abs(target) < 0.02 or abs(target) > max_target:
                raise EmbodimentExecutionError(
                    f"abs(yaw_rad) must be between 0.02 and {max_target}",
                )
            if not math.isfinite(speed) or speed <= 0.0 or speed > max_speed:
                raise EmbodimentExecutionError(
                    f"max_speed_rps must be between 0 and {max_speed}",
                )
        else:
            raise EmbodimentExecutionError(f"Unsupported bounded motion target: {target_kind}")

        timeout = min(float(execution.get("timeout_seconds", 900.0)), 900.0)
        stop_distance = float(execution.get("obstacle_stop_distance_m", 0.24))
        scan_timeout = min(float(execution.get("scan_timeout_seconds", 0.75)), 2.0)
        half_angle = math.radians(
            min(float(execution.get("forward_half_angle_deg", 35.0)), 90.0),
        )
        if not math.isfinite(timeout) or timeout <= 0.0:
            raise EmbodimentExecutionError("timeout_seconds must be positive")
        if not math.isfinite(stop_distance) or stop_distance < 0.1:
            raise EmbodimentExecutionError(
                "obstacle_stop_distance_m must be at least 0.1",
            )
        request_id = f"qwenpaw-motion-{uuid.uuid4().hex}"
        start_pose: dict[str, float] | None = None
        latest_scan: dict[str, Any] | None = None
        latest_scan_at = 0.0
        safety_enabled = False
        safety_seen = False
        safety_confirmed = False
        done = False
        status = "TIMEOUT"
        waypoint_index = 1

        def yaw_from_quaternion(q: dict[str, Any]) -> float:
            return math.atan2(2.0 * (float(q.get("w", 1.0)) * float(q.get("z", 0.0)) + float(q.get("x", 0.0)) * float(q.get("y", 0.0))), 1.0 - 2.0 * (float(q.get("y", 0.0)) ** 2 + float(q.get("z", 0.0)) ** 2))

        def angle_delta(a: float, b: float) -> float:
            return math.atan2(math.sin(a - b), math.cos(a - b))

        def front_obstacle(scan: dict[str, Any]) -> tuple[bool, float | None]:
            ranges = scan.get("ranges")
            if not isinstance(ranges, list):
                return True, None
            angle_min = float(scan.get("angle_min", 0.0))
            angle_increment = float(scan.get("angle_increment", 0.0))
            range_min = max(float(scan.get("range_min", 0.0)), 0.02)
            range_max = float(scan.get("range_max", 0.0))
            nearest: float | None = None
            for index, raw in enumerate(ranges):
                angle = angle_min + index * angle_increment
                if abs(angle) > half_angle:
                    continue
                value = float(raw)
                if not math.isfinite(value) or value < range_min:
                    continue
                if range_max > 0.0:
                    value = min(value, range_max)
                nearest = value if nearest is None else min(nearest, value)
            return nearest is None or nearest <= stop_distance, nearest

        zero = {
            "linear": {"x": 0.0, "y": 0.0, "z": 0.0},
            "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
        }
        async with websockets.connect(
            endpoint,
            open_timeout=5,
            close_timeout=2,
            max_size=16_000_000,
        ) as socket:
            # The bridge is fail-closed. Enable it only for this approved,
            # bounded invocation, and always disable it in the finalizer.
            # A proposal sent before enable closes the vehicle-side watchdog
            # race: the bridge may publish its first status asynchronously and
            # otherwise time out before the first post-enable proposal arrives.
            await socket.send(
                json.dumps(
                    {
                        "op": "publish",
                        "topic": topic,
                        "type": "geometry_msgs/msg/Twist",
                        "msg": zero,
                    },
                ),
            )
            enable_id = request_id + "-enable"
            await socket.send(
                json.dumps(
                    {
                        "op": "call_service",
                        "id": enable_id,
                        "service": enable_service,
                        "args": {"data": True},
                    },
                ),
            )
            enable_deadline = asyncio.get_running_loop().time() + 5.0
            while True:
                remaining = enable_deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise EmbodimentExecutionError("Safety bridge enable timed out")
                raw = await asyncio.wait_for(socket.recv(), remaining)
                message = json.loads(raw)
                if message.get("id") != enable_id:
                    continue
                if not message.get("result", False):
                    raise EmbodimentExecutionError(
                        f"Safety bridge enable failed: {message.get('values')}"
                    )
                break
            # Refresh the vehicle-side proposal watchdog immediately after
            # enabling the bridge.  This is intentionally a zero command; it
            # cannot move the base and keeps the bridge alive while the
            # subscriptions and first fresh sensor samples are established.
            await socket.send(
                json.dumps(
                    {
                        "op": "publish",
                        "topic": topic,
                        "type": "geometry_msgs/msg/Twist",
                        "msg": zero,
                    },
                ),
            )
            subscriptions = (
                ("odom", odom_topic, "nav_msgs/msg/Odometry"),
                ("scan", scan_topic, "sensor_msgs/msg/LaserScan"),
                ("safety", safety_topic, "std_msgs/msg/String"),
            )
            for suffix, subscription_topic, message_type in subscriptions:
                await socket.send(
                    json.dumps(
                        {
                            "op": "subscribe",
                            "id": request_id + "-" + suffix,
                            "topic": subscription_topic,
                            "queue_length": 1,
                            "throttle_rate": 0,
                            "type": message_type,
                        },
                    ),
                )
            # The vehicle-side bridge disables itself when no proposal is
            # received for proposal_timeout_s (currently one second). Send a
            # zero-velocity proposal immediately after enabling it so that the
            # bridge remains enabled while the first fresh odometry, lidar,
            # and safety samples arrive. This cannot move the base.
            await socket.send(
                json.dumps(
                    {
                        "op": "publish",
                        "topic": topic,
                        "type": "geometry_msgs/msg/Twist",
                        "msg": zero,
                    },
                ),
            )
            deadline = asyncio.get_running_loop().time() + timeout
            safety_confirmation_deadline = (
                asyncio.get_running_loop().time() + 3.0
            )
            sensor_start_deadline = asyncio.get_running_loop().time() + 3.0
            last_send = 0.0
            try:
                while asyncio.get_running_loop().time() < deadline:
                    remaining = min(0.25, deadline - asyncio.get_running_loop().time())
                    try:
                        raw = await asyncio.wait_for(socket.recv(), remaining)
                        message = json.loads(raw)
                        if message.get("op") != "publish":
                            continue
                        if message.get("topic") == scan_topic:
                            latest_scan = message.get("msg") or {}
                            latest_scan_at = asyncio.get_running_loop().time()
                            continue
                        if message.get("topic") == safety_topic:
                            safety_seen = True
                            raw_status = (message.get("msg") or {}).get("data", "{}")
                            try:
                                safety_enabled = bool(json.loads(raw_status).get("enabled"))
                            except (AttributeError, TypeError, ValueError):
                                safety_enabled = False
                            if safety_enabled:
                                safety_confirmed = True
                            continue
                        if message.get("topic") != odom_topic:
                            continue
                        now = asyncio.get_running_loop().time()
                        # A status message already queued before the enable
                        # service response may still say DISABLED. Do not
                        # treat that stale sample as an immediate failure. The
                        # first bounded proposal is sent below, which prevents
                        # the vehicle bridge from timing out while it waits for
                        # its enabled status publication.
                        if not safety_confirmed and now >= safety_confirmation_deadline:
                            status = "SAFETY_ENABLE_UNCONFIRMED"
                            break
                        # Ignore a queued disabled status until this
                        # invocation has observed a fresh enabled status.
                        # A disabled sample can belong to the previous
                        # invocation because the status topic is latched/
                        # queued independently of the enable service reply.
                        if safety_confirmed and not safety_enabled:
                            status = "SAFETY_DISABLED"
                            break
                        msg = message.get("msg") or {}
                        pose = ((msg.get("pose") or {}).get("pose") or {})
                        position = pose.get("position") or {}
                        orientation = pose.get("orientation") or {}
                        current = {"x": float(position.get("x", 0.0)), "y": float(position.get("y", 0.0)), "yaw": yaw_from_quaternion(orientation)}
                        if start_pose is None:
                            start_pose = current
                        dx = current["x"] - start_pose["x"]
                        dy = current["y"] - start_pose["y"]
                        progress = math.hypot(dx, dy) if target_kind == "linear" else angle_delta(current["yaw"], start_pose["yaw"])
                        if target_kind != "route" and abs(progress) >= abs(target) - (0.015 if target_kind == "linear" else 0.03):
                            done = True
                            status = "SUCCEEDED"
                            break
                        if latest_scan is None and now < sensor_start_deadline:
                            continue
                        if latest_scan is None or now - latest_scan_at > scan_timeout:
                            status = "SCAN_STALE"
                            break
                        blocked, nearest = front_obstacle(latest_scan)
                        if target_kind in {"linear", "route"} and blocked:
                            status = "OBSTACLE_STOP" if nearest is not None else "SCAN_INVALID"
                            break
                        if now - last_send >= 0.05:
                            if target_kind == "route":
                                current_point = (current["x"], current["y"])
                                while waypoint_index < len(route_points) - 1:
                                    wx, wy = route_points[waypoint_index]
                                    if math.hypot(wx - current_point[0], wy - current_point[1]) <= max(0.10, route_goal_tolerance * 0.6):
                                        waypoint_index += 1
                                    else:
                                        break
                                tx, ty = route_points[-1] if waypoint_index >= len(route_points) else route_points[waypoint_index]
                                dx, dy = tx - current["x"], ty - current["y"]
                                waypoint_distance = math.hypot(dx, dy)
                                if waypoint_index >= len(route_points) - 1 and waypoint_distance <= route_goal_tolerance:
                                    done = True
                                    status = "SUCCEEDED"
                                    break
                                bearing = math.atan2(dy, dx)
                                error = angle_delta(bearing, current["yaw"])
                                max_angular = min(float(execution.get("max_angular_speed_rps", 0.30)), 1.0)
                                if abs(error) > math.radians(75.0):
                                    linear = 0.0
                                    angular = max_angular if error > 0.0 else -max_angular
                                else:
                                    linear = speed * max(0.25, math.cos(error))
                                    angular = max(-max_angular, min(max_angular, 2.0 * error))
                            else:
                                linear = speed if target_kind == "linear" else 0.0
                                angular = 0.0 if target_kind == "linear" else (
                                    speed if target > 0 else -speed
                                )
                            await socket.send(
                                json.dumps(
                                    {
                                        "op": "publish",
                                        "topic": topic,
                                        "type": "geometry_msgs/msg/Twist",
                                        "msg": {
                                            "linear": {"x": linear, "y": 0.0, "z": 0.0},
                                            "angular": {"x": 0.0, "y": 0.0, "z": angular},
                                        },
                                    },
                                ),
                            )
                            last_send = now
                    except asyncio.TimeoutError:
                        continue
            finally:
                await socket.send(
                    json.dumps(
                        {
                            "op": "publish",
                            "topic": topic,
                            "type": "geometry_msgs/msg/Twist",
                            "msg": zero,
                        },
                    ),
                )
                stop_id = request_id + "-stop"
                try:
                    await socket.send(
                        json.dumps(
                            {
                                "op": "call_service",
                                "id": stop_id,
                                "service": stop_service,
                                "args": {},
                            },
                        ),
                    )
                    stop_deadline = asyncio.get_running_loop().time() + 2.0
                    while asyncio.get_running_loop().time() < stop_deadline:
                        raw = await asyncio.wait_for(
                            socket.recv(),
                            max(0.05, stop_deadline - asyncio.get_running_loop().time()),
                        )
                        if json.loads(raw).get("id") == stop_id:
                            break
                except Exception:
                    # The zero proposal is already sent; socket teardown is
                    # still the safest fallback if the stop service is down.
                    pass
        if not done and status == "TIMEOUT":
            raise EmbodimentExecutionError(
                f"bounded motion timed out after {timeout:.1f}s; zero proposal sent",
            )
        if status in {
            "SAFETY_DISABLED",
            "SAFETY_ENABLE_UNCONFIRMED",
            "SCAN_INVALID",
            "SCAN_STALE",
            "OBSTACLE_STOP",
        }:
            raise EmbodimentExecutionError(
                f"bounded motion rejected: {status}; zero proposal sent",
            )
        return {
            "status": status,
            "target_kind": target_kind,
            "target": target,
            "proposal_topic": topic,
            "odom_topic": odom_topic,
            "scan_topic": scan_topic,
        }

    async def _publish_stop(self, endpoint: str, servo_ids: list[int]) -> None:
        if not servo_ids:
            return
        states = [
            {"present_id": [1, int(servo_id)], "stop": [1, 1]}
            for servo_id in servo_ids
        ]
        await self._rosbridge_publish(
            endpoint,
            "/ros_robot_controller/bus_servo/set_state",
            "ros_robot_controller_msgs/msg/SetBusServoState",
            {"state": states, "duration": 0.0},
        )

    @staticmethod
    async def _rosbridge_publish(
        endpoint: str, topic: str, message_type: str | None, message: dict[str, Any]
    ) -> None:
        request: dict[str, Any] = {"op": "publish", "topic": topic, "msg": message}
        if message_type:
            request["type"] = message_type
        async with websockets.connect(endpoint, open_timeout=5, close_timeout=2) as socket:
            await socket.send(json.dumps(request))
            await asyncio.sleep(0.1)
