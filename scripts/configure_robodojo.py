#!/usr/bin/env python3
"""Install the direct-control overlay into an explicit RoboDojo checkout."""
import argparse
import json
from pathlib import Path
import shutil
import shlex
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--robodojo-root', type=Path, required=True)
    parser.add_argument('--transport-python', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', default='align_blocks')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--layout', type=int, default=0)
    parser.add_argument('--bridge-port', type=int, default=8098)
    parser.add_argument('--control-port', type=int, default=8099)
    parser.add_argument('--policy-port', type=int, default=8100)
    args = parser.parse_args()
    root = args.robodojo_root.resolve()
    repo = Path(__file__).resolve().parents[1]
    python = args.transport_python.absolute()
    required = ['scripts/robodojo.sh', 'XPolicyLab/setup_policy_server.py',
                'XPolicyLab/utils/setup_env_client.sh', 'src/eval_client/main.py']
    if not python.is_file() or any(not (root / path).is_file() for path in required):
        raise SystemExit('RoboDojo/XPolicyLab checkout or transport Python is incomplete')
    check = subprocess.run([str(python), '-c',
        'import h5py, numpy, cv2, msgpack, msgpack_numpy; import websockets.asyncio.server'],
        capture_output=True, text=True)
    if check.returncode:
        raise SystemExit('Transport Python dependencies are incomplete. Install source[transport] '
                         'in that interpreter before wiring:\n'+check.stderr[-2500:])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    adapter = root / 'XPolicyLab/policy/GPT6_Harness_Ablation'
    native = root / 'src/eval_client/interactive_harness.py'
    if native.exists(): shutil.copy2(native, output / 'interactive_harness.before.py')
    shutil.copytree(repo / 'runtime/gpt6-harness-ablation-adapter', adapter, dirs_exist_ok=True)
    shutil.copy2(adapter / 'interactive_harness.py', native)
    transport = output / 'transport/.venv/bin'
    transport.mkdir(parents=True)
    wrapper = transport / 'python'
    wrapper.write_text('#!/usr/bin/env bash\nexec ' + shlex.quote(str(python)) + ' "$@"\n')
    wrapper.chmod(0o755)
    registry = [dict(provider_id='gpt6-harness', kind='direct', policy_name=adapter.name,
        checkpoint='direct-joint', policy_uv_env_path=str(output / 'transport'),
        revision='awomopaw-direct-joint-v1')]
    (output / 'registry.json').write_text(json.dumps(registry, indent=2) + '\n')
    (output / 'fixture.json').write_text(json.dumps(dict(task=args.task, seed=args.seed,
        layout_id=args.layout, arm='without_skills', output=str(output)), indent=2) + '\n')
    values = dict(ROBODOJO_ROOT=str(root), ROBODOJO_HARNESS_POLICY_DIR=str(adapter),
        ROBODOJO_HARNESS_POLICY_NAME=adapter.name, ROBODOJO_POLICY_REGISTRY_JSON=json.dumps(registry),
        AWOMOPAW_HARNESS_PORT=str(args.control_port), ROBODOJO_ACTION_TYPE='joint',
        ROBODOJO_HARNESS_STATE_DIR=str(output / 'bridge-state'))
    env_file = output / 'environment.sh'
    env_file.write_text(''.join('export ' + key + '=' + shlex.quote(value) + '\n'
                               for key, value in values.items()))
    command = [sys.executable, str(repo / 'scripts/gpt6_ablation_bridge.py'),
        '--fixture', str(output / 'fixture.json'), '--port', str(args.bridge_port),
        '--policy-port', str(args.policy_port)]
    print('source ' + shlex.quote(str(env_file)))
    print(shlex.join(command))
    print('Connect the agent to http://127.0.0.1:' + str(args.bridge_port))


if __name__ == '__main__':
    main()
