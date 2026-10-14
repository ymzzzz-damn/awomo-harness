import json
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from qwenpaw.app.embodiments.execution_policy import execution_block_reason
from qwenpaw.app.embodiments.tool_adapter import make_tool_function


def configure(tmp_path, base=False):
    (tmp_path / 'robot_execution_policy.json').write_text(json.dumps({
        'manipulation_enabled': False, 'base_motion_enabled': base, 'franka_enabled': False}))


@pytest.mark.parametrize('tool', ['jetrover_arm.move_pulses', 'jetrover_gripper__set_pulse', 'franka_fr3_controller__move_joints', 'jetrover_base__set_base_twist'])
def test_maintenance_blocks_actuators(tmp_path, tool):
    configure(tmp_path)
    assert execution_block_reason(tmp_path, tool)


@pytest.mark.parametrize('tool', ['jetrover_base__stop_base', 'jetrover_arm__emergency_stop', 'jetrover_camera__capture_rgb', 'jetrover_arm__get_joint_state'])
def test_stops_and_feedback_remain_available(tmp_path, tool):
    configure(tmp_path)
    assert execution_block_reason(tmp_path, tool) is None


def test_read_only_acceptance_can_restore_navigation_without_enabling_arm(tmp_path):
    configure(tmp_path, base=True)
    assert execution_block_reason(tmp_path, 'jetrover_base__set_base_twist') is None
    assert execution_block_reason(tmp_path, 'jetrover_arm__move_pulses')


@pytest.mark.asyncio
async def test_block_happens_before_loader_or_network_dispatch(tmp_path):
    configure(tmp_path)
    loader = MagicMock()
    with patch('qwenpaw.app.embodiments.tool_adapter.EmbodimentExecutor') as executor:
        fn = make_tool_function(loader, 'jetrover_arm.move_pulses', 'test', {}, workspace_dir=tmp_path)
        result = await fn(pulses=[500]*5)
        assert result.metadata['executed'] is False
        loader.dispatch_tool.assert_not_called()
        executor.return_value.execute.assert_not_called()


def test_finite_smoke_budget_cannot_translate_or_repeat(tmp_path):
    from qwenpaw.app.embodiments.execution_policy import reserve_test_motion
    path=tmp_path/'robot_execution_policy.json'
    path.write_text(json.dumps({'base_motion_enabled':True,'test_twist_calls_remaining':1}))
    tool='jetrover_base__set_base_twist'
    assert reserve_test_motion(tmp_path,tool,{'linear_x_mps':.1,'angular_z_rps':0.,'duration_seconds':1})
    assert json.loads(path.read_text())['test_twist_calls_remaining']==1
    command={'linear_x_mps':0.,'angular_z_rps':.3,'duration_seconds':1.75}
    assert reserve_test_motion(tmp_path,tool,command) is None
    assert reserve_test_motion(tmp_path,tool,command)
