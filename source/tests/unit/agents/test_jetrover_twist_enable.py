"""The enable handshake must not span separate LLM reasoning calls."""
import json
import pytest

from qwenpaw.app.embodiments.executor import EmbodimentExecutor, EmbodimentExecutionError


@pytest.mark.asyncio
@pytest.mark.parametrize('success', [True, False])
async def test_enable_on_same_socket_before_nonzero_twist(tmp_path, monkeypatch, success):
    from qwenpaw.app.embodiments import executor as module
    sent = []
    class Socket:
        async def send(self, data): sent.append(json.loads(data))
        async def recv(self):
            return json.dumps({'id': sent[-1]['id'], 'result': True,
                               'values': {'success': success}})
    class Connection:
        async def __aenter__(self): return Socket()
        async def __aexit__(self, *_): return None
    monkeypatch.setattr(module.websockets, 'connect', lambda *a, **k: Connection())
    dispatch = {'endpoint': 'ws://robot:9090',
                'execution': {'topic': '/qwenpaw/base/proposed_cmd_vel'},
                'arguments': {'linear_x_mps': 0, 'angular_z_rps': .3, 'duration_seconds': .05},
                'motion_enable': {'protocol': 'rosbridge_service', 'endpoint': 'ws://robot:9090',
                                  'execution': {'service': '/vln_cmd_vel_safety/enable', 'args': {'data': True}}}}
    if success:
        result = await EmbodimentExecutor(tmp_path)._timed_twist(dispatch)
        assert result['motion_confirmed'] is False
    else:
        with pytest.raises(EmbodimentExecutionError, match='rejected motion enable'):
            await EmbodimentExecutor(tmp_path)._timed_twist(dispatch)
    assert sent[0]['op'] == 'call_service'
    assert sent[0]['args'] == {'data': True}
    assert any(item.get('msg', {}).get('angular', {}).get('z') == .3 for item in sent) == success
    assert all(item['msg']['angular']['z'] == 0 for item in sent[-3:])
