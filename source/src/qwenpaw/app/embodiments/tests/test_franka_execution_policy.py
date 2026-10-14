import asyncio
import importlib.util
import json
import sys
import time
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

_POLICY_PATH = Path(__file__).resolve().parents[1] / "execution_policy.py"
_SPEC = importlib.util.spec_from_file_location("review_execution_policy", _POLICY_PATH)
assert _SPEC and _SPEC.loader
_POLICY = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_POLICY)
execution_argument_block_reason = _POLICY.execution_argument_block_reason
execution_block_reason = _POLICY.execution_block_reason


def valid_authorization(plan_id="pick-plan-123456"):
    now_ms = int(time.time() * 1000)
    return {
        "action": "execute_grasp",
        "plan_id": plan_id,
        "prompt_hash": "a" * 64,
        "issued_at_ms": now_ms,
        "expires_at_ms": now_ms + 5_000,
        "nonce": "b" * 32,
        "signature": "c" * 64,
    }


def _load_isolated_tool_adapter():
    """Load the adapter with tiny AgentScope/transport stubs for guard tests."""
    package_name = "review_qwenpaw_embodiments"
    package_root = _POLICY_PATH.parent
    package = types.ModuleType(package_name)
    package.__path__ = [str(package_root)]
    sys.modules[package_name] = package

    agentscope = types.ModuleType("agentscope")
    message = types.ModuleType("agentscope.message")
    tool = types.ModuleType("agentscope.tool")

    class TextBlock:
        def __init__(self, **values):
            self.__dict__.update(values)

    class ToolResponse:
        def __init__(self, *, content, metadata):
            self.content = content
            self.metadata = metadata

    message.TextBlock = TextBlock
    tool.ToolResponse = ToolResponse
    sys.modules["agentscope"] = agentscope
    sys.modules["agentscope.message"] = message
    sys.modules["agentscope.tool"] = tool

    approval_name = f"{package_name}.approval"
    approval_spec = importlib.util.spec_from_file_location(
        approval_name, package_root / "approval.py"
    )
    assert approval_spec and approval_spec.loader
    approval = importlib.util.module_from_spec(approval_spec)
    sys.modules[approval_name] = approval
    approval_spec.loader.exec_module(approval)

    policy_name = f"{package_name}.execution_policy"
    sys.modules[policy_name] = _POLICY

    executor_name = f"{package_name}.executor"
    executor = types.ModuleType(executor_name)

    class EmbodimentExecutor:
        def __init__(self, _workspace_dir):
            pass

        async def execute(self, _dispatch):
            raise AssertionError("unapproved Franka call reached transport")

    executor.EmbodimentExecutor = EmbodimentExecutor
    sys.modules[executor_name] = executor

    adapter_name = f"{package_name}.tool_adapter"
    adapter_spec = importlib.util.spec_from_file_location(
        adapter_name, package_root / "tool_adapter.py"
    )
    assert adapter_spec and adapter_spec.loader
    adapter = importlib.util.module_from_spec(adapter_spec)
    sys.modules[adapter_name] = adapter
    adapter_spec.loader.exec_module(adapter)
    return adapter


class FrankaExecutionPolicyTest(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.workspace = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write_policy(self, **overrides):
        value = {
            "franka_enabled": True,
            "manipulation_enabled": True,
            "franka_execution_mode": "committed_grasp_only",
            "franka_allowed_operations": ["execute_committed_grasp"],
        }
        value.update(overrides)
        (self.workspace / "robot_execution_policy.json").write_text(
            json.dumps(value)
        )

    def test_missing_policy_blocks_actuation_but_not_stop(self):
        self.assertIn(
            "missing",
            execution_block_reason(
                self.workspace,
                "franka_fr3_controller.execute_committed_grasp",
            ),
        )
        self.assertIsNone(
            execution_block_reason(self.workspace, "franka_fr3_controller.stop")
        )

    def test_invalid_or_non_object_policy_is_fail_closed(self):
        path = self.workspace / "robot_execution_policy.json"
        path.write_text("not-json")
        self.assertIn(
            "unreadable",
            execution_block_reason(
                self.workspace,
                "franka_fr3_controller.execute_committed_grasp",
            ),
        )
        path.write_text("[]")
        self.assertIn(
            "JSON object",
            execution_block_reason(
                self.workspace,
                "franka_fr3_controller.execute_committed_grasp",
            ),
        )

    def test_policy_allows_committed_grasp_and_blocks_every_raw_action(self):
        self.write_policy()
        self.assertIsNone(
            execution_block_reason(
                self.workspace,
                "franka_fr3_controller.execute_committed_grasp",
            )
        )
        for operation in (
            "move_joints",
            "move_cartesian_pose",
            "move_cartesian_velocity",
            "set_gripper",
            "recover",
            "connect",
        ):
            with self.subTest(operation=operation):
                reason = execution_block_reason(
                    self.workspace, f"franka_fr3_controller.{operation}"
                )
                self.assertIn("RAW_FRANKA_ACTION_BLOCKED", reason)

    def test_franka_and_manipulation_gates_must_both_be_explicitly_enabled(self):
        self.write_policy(manipulation_enabled=False)
        reason = execution_block_reason(
            self.workspace,
            "franka_fr3_controller.execute_committed_grasp",
        )
        self.assertIn("MANIPULATION_DISABLED", reason)

    def test_atomic_mode_allows_only_explicitly_listed_operations(self):
        self.write_policy(
            franka_execution_mode="atomic_supervised",
            franka_allowed_operations=[
                "connect",
                "move_joints",
                "move_joint_velocity",
                "set_gripper",
                "recover",
            ],
        )
        for operation in (
            "connect",
            "move_joints",
            "move_joint_velocity",
            "set_gripper",
            "recover",
        ):
            with self.subTest(operation=operation):
                self.assertIsNone(
                    execution_block_reason(
                        self.workspace, f"franka_fr3_controller.{operation}"
                    )
                )
        self.assertIn(
            "FRANKA_ATOMIC_ACTION_BLOCKED",
            execution_block_reason(
                self.workspace,
                "franka_fr3_controller.move_cartesian_velocity",
            ),
        )

    def test_committed_grasp_arguments_require_all_opaque_grants(self):
        tool = "franka_fr3_controller.execute_committed_grasp"
        valid = {
            "request_id": "qwenpaw-execute-123456",
            "plan_id": "pick-plan-123456",
            "commit_token": "n" * 32,
            "authorization": valid_authorization(),
        }
        self.assertIsNone(execution_argument_block_reason(tool, valid))
        for key in valid:
            with self.subTest(key=key):
                malformed = dict(valid)
                malformed[key] = "x"
                self.assertIsNotNone(
                    execution_argument_block_reason(tool, malformed)
                )

    def test_authorization_is_plan_bound_short_lived_and_strict(self):
        tool = "franka_fr3_controller.execute_committed_grasp"
        base = {
            "request_id": "qwenpaw-execute-123456",
            "plan_id": "pick-plan-123456",
            "commit_token": "n" * 32,
        }
        now_ms = int(time.time() * 1000)
        invalid_authorizations = (
            {**valid_authorization("different-plan")},
            {
                **valid_authorization(),
                "issued_at_ms": now_ms - 6_000,
                "expires_at_ms": now_ms + 1,
            },
            {
                **valid_authorization(),
                "issued_at_ms": now_ms - 5_001,
                "expires_at_ms": now_ms - 1,
            },
            {**valid_authorization(), "signature": "not-a-signature"},
            {**valid_authorization(), "unexpected": True},
        )
        for authorization in invalid_authorizations:
            with self.subTest(authorization=authorization):
                self.assertIsNotNone(
                    execution_argument_block_reason(
                        tool, {**base, "authorization": authorization}
                    )
                )

    def test_committed_grasp_without_toolguard_approval_never_executes(self):
        self.write_policy()
        adapter = _load_isolated_tool_adapter()

        class Loader:
            def __init__(self):
                self.dispatches = []

            def dispatch_tool(self, name, arguments, *, approved):
                self.dispatches.append((name, arguments, approved))
                return {
                    "success": False,
                    "executed": False,
                    "requires_approval": not approved,
                }

        loader = Loader()
        execute = adapter.make_tool_function(
            loader,
            "franka_fr3_controller.execute_committed_grasp",
            "Execute one bridge-committed grasp",
            {"type": "object", "properties": {}},
            safety_class="supervisor",
            workspace_dir=self.workspace,
        )
        response = asyncio.run(
            execute(
                request_id="qwenpaw-execute-123456",
                plan_id="pick-plan-123456",
                commit_token="n" * 32,
                authorization=valid_authorization(),
            )
        )

        self.assertEqual(len(loader.dispatches), 1)
        self.assertIs(loader.dispatches[0][2], False)
        self.assertIs(response.metadata["executed"], False)
        self.assertIs(response.metadata["requires_approval"], True)


class FrankaEmbodimentSurfaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        test_file = Path(__file__).resolve()
        cls.root = test_file.parents[2]
        repository_root = test_file.parents[5]
        package_candidates = (
            cls.root / "franka_fr3" / "embodiment.yaml",
            repository_root / "embodiments" / "packages" / "franka_fr3" / "embodiment.yaml",
        )
        workspace_candidates = (
            cls.root / "workspace",
            Path.home() / ".qwenpaw" / "workspaces" / "car_controller",
        )
        cls.package_path = next(path for path in package_candidates if path.is_file())
        cls.workspace = next(
            path
            for path in workspace_candidates
            if (path / "agent.json").is_file()
            and (path / "robot_execution_policy.json").is_file()
        )
        unit_candidates = (
            cls.root
            / "franka_mtc_bridge"
            / "deploy"
            / "franka-grasp-bridge.service",
            Path.home()
            / ".config"
            / "systemd"
            / "user"
            / "franka-grasp-bridge.service",
        )
        cls.bridge_unit_path = next(
            path for path in unit_candidates if path.is_file()
        )
        cls.adapter_source_path = _POLICY_PATH.parent / "tool_adapter.py"
        cls.package = yaml.safe_load(cls.package_path.read_text())

    def test_atomic_surface_is_bounded_and_supervisor_gated(self):
        operations = self.package["registry"]["nodes"][0]["tool_interface"][
            "operations"
        ]
        by_id = {item["operation_id"]: item for item in operations}
        self.assertEqual(
            set(by_id),
            {
                "health_check",
                "get_state",
                "connect",
                "move_joints",
                "move_joint_velocity",
                "set_gripper",
                "stop",
                "recover",
                "disconnect",
            },
        )
        for operation in (
            "move_joints",
            "move_joint_velocity",
            "set_gripper",
            "recover",
        ):
            self.assertEqual(by_id[operation]["safety_class"], "supervisor")
        for operation in ("health_check", "get_state", "connect", "stop", "disconnect"):
            self.assertEqual(by_id[operation]["safety_class"], "safe")
        positions = by_id["move_joints"]["input_schema"]["properties"]["positions"]
        self.assertEqual((positions["minItems"], positions["maxItems"]), (7, 7))
        velocities = by_id["move_joint_velocity"]["input_schema"]["properties"]
        self.assertEqual(velocities["duration_seconds"]["maximum"], 10.0)
        self.assertEqual(
            velocities["velocities_rad_s"]["items"]["maximum"], 0.25
        )

    def test_workspace_routes_atomic_embodiment_to_control_host(self):
        config = json.loads(
            (self.workspace / "agent.json").read_text()
        )
        binding = config["embodiments"]["packages"]["franka_fr3"]["bindings"]
        self.assertEqual(binding["franka_gateway"], "http://127.0.0.1:8092")
        self.assertEqual(config["approval_level"], "AUTO")

    def test_staging_is_disabled_and_atomic_motion_remains_blocked(self):
        policy_path = self.workspace / "robot_execution_policy.json"
        policy = json.loads(policy_path.read_text())
        self.assertIs(policy["franka_enabled"], False)
        reason = execution_block_reason(
            policy_path.parent,
            "franka_fr3_controller.move_joints",
        )
        self.assertIn("FRANKA_DISABLED", reason)

        operations = self.package["registry"]["nodes"][0]["tool_interface"][
            "operations"
        ]
        execute = next(
            item for item in operations
            if item["operation_id"] == "move_joints"
        )
        self.assertEqual(execute["safety_class"], "supervisor")
        adapter_source = self.adapter_source_path.read_text()
        self.assertNotIn("_FRANKA_COMMITTED_GRASP_TOOL", adapter_source)


if __name__ == "__main__":
    unittest.main()
