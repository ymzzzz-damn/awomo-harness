#!/usr/bin/env python3
"""Install this immutable RSI catalog inside the already allocated node-153 pod."""
import argparse
import copy
import hashlib
import json
import os
import signal
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def api(path, body=None, method=None):
    request = urllib.request.Request('http://127.0.0.1:8088/api' + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'}, method=method)
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=30) as response:
        return json.load(response)


def link(target, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        if path.resolve() != target.resolve():
            raise ValueError('Existing compatibility alias points elsewhere: ' + str(path))
    elif path.exists():
        if path.resolve() != target.resolve():
            raise FileExistsError('Refuse to replace existing path: ' + str(path))
    else:
        path.symlink_to(target, target_is_directory=target.is_dir())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--activate-vla', action='store_true')
    args = parser.parse_args()
    source_root = args.root
    sim = Path('/opt/robodojo')
    root = sim / 'eval_result/physicalrsi-all-skills' / source_root.name
    root.mkdir(parents=True, exist_ok=True)
    assets = source_root / 'runtime/physicalrsi-skills'
    code = assets / 'implementations/code-skills'
    framework = assets / 'implementations/framework'
    openpi = Path('/opt/physicalrsi-runtime') / source_root.name / 'openpi'
    compatible = openpi.with_name('openpi-gpu128')
    if compatible.is_dir():
        openpi = compatible
    awomo = root / 'awomopaw'
    stage = source_root / 'runtime/physicalrsi-all-skills'
    state_stage = root / 'runtime/physicalrsi-all-skills'
    link(source_root / 'awomopaw/src', awomo / 'src')
    link(source_root / 'scripts', root / 'scripts')
    original_registry = json.loads(os.environ['ROBODOJO_POLICY_REGISTRY_JSON'])
    for path in [code / 'programs.json', framework / 'PhysicalRSI_baselines/robodojo/skills/pi05.py',
                 openpi / '.venv/lib/python3.11/site-packages']:
        if not path.exists():
            raise FileNotFoundError('Complete source/environment upload first: ' + str(path))
    link(Path('/opt/conda/envs/RoboDojo'), Path('/home/ymz/miniconda3/envs/RoboDojo'))
    view = Path('/opt/physicalrsi-runtime') / source_root.name / 'skill-assets'
    link(code, view / 'implementations/code-skills')
    # The editable source alias stays stable; the runtime may use a separate
    # compatible virtualenv whose src links to these same frozen source bytes.
    link(Path('/opt/physicalrsi-runtime') / source_root.name / 'openpi', view / 'implementations/openpi')
    link(assets / 'checkpoints', view / 'checkpoints')
    link(view, Path('/data/ymz/physicalrsi-reproduction/skill-assets'))
    link(Path(os.environ['QWENPAW_VENV_ROOT']), awomo / '.venv')
    link(assets, awomo / 'runtime/physicalrsi-skills')
    catalog = json.loads((stage / 'adapter/catalog.json').read_text())
    catalog.update(asset_root=str(code), framework_source=str(framework))
    for entry in catalog['releases'].values():
        if entry['kind'] == 'vla':
            entry['configuration']['model_path'] = str(assets / 'checkpoints' / entry['name'])
    target = sim / 'XPolicyLab/policy/PhysicalRSI_Frozen_Skills'
    if not target.exists():
        shutil.copytree(stage / 'adapter', target)
        for name in ['setup_eval_policy_server.sh', 'setup_eval_env_client.sh']:
            shutil.copy2(sim / 'XPolicyLab/policy/Awomopaw_Harness' / name, target / name)
        save(target / 'catalog.json', catalog)
    elif json.loads((target / 'catalog.json').read_text()) != catalog:
        raise ValueError('Refuse to overwrite an already deployed frozen catalog')
    registry = [item for item in original_registry if item['provider_id'] == 'pi05']
    for item in json.loads((stage / 'registry.json').read_text()):
        if item['policy_name'] != 'PhysicalRSI_Frozen_Skills':
            continue
        item['policy_uv_env_path'] = str(openpi)
        registry.append(item)
    save(state_stage / 'registry.all.json', registry)
    available = registry if args.activate_vla else [item for item in registry
        if item['kind'] == 'code' or item['provider_id'] == 'pi05']
    save(state_stage / 'registry.json', available)
    pyenv = dict(os.environ, PYTHONPATH=os.pathsep.join([
        str(awomo / 'src'), str(root / 'scripts'), '/runtime/python-compat']),
        ROBODOJO_ROOT=str(sim), ROBODOJO_HARNESS_POLICY_DIR=str(target),
        ROBODOJO_HARNESS_POLICY_NAME='PhysicalRSI_Frozen_Skills',
        ROBODOJO_POLICY_REGISTRY_JSON=json.dumps(available),
        ROBODOJO_HARNESS_STATE_DIR=str(root / 'ordinary-bridge-state'),
        ROBODOJO_COMPAT_PYTHONPATH='/runtime/python-compat',
        ROBODOJO_ACTION_TYPE='joint', AWOMOPAW_HARNESS_PORT='8199',
        PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    check = subprocess.run([str(awomo / '.venv/bin/python'), '-c',
        'import json;from frozen_skill_parallel_bridge import ParallelAdapterConfig;'
        'c=ParallelAdapterConfig.from_env();p=c.validate();'
        'print(json.dumps({"problems":p,"providers":len(c.providers)}));assert not p'],
        env=pyenv, check=True, capture_output=True, text=True)
    save(root / 'static-preflight.json', json.loads(check.stdout))
    # Start a catalog-aware endpoint alongside the original service, first.
    record = root / 'ordinary-bridge-process.json'
    if args.activate_vla and record.exists():
        with urllib.request.urlopen('http://127.0.0.1:8180/health', timeout=10) as response:
            prior = json.load(response)
        if prior.get('active_evaluation'):
            raise RuntimeError('Catalog bridge is active; defer VLA activation')
        process_info = json.loads(record.read_text())
        cmdline = Path('/proc') / str(process_info['pid']) / 'cmdline'
        if cmdline.exists():
            if b'all_rsi_skills_bridge.py' not in cmdline.read_bytes():
                raise RuntimeError('Bridge PID no longer belongs to this deployment')
            os.kill(process_info['pid'], signal.SIGTERM)
            for _ in range(20):
                if not cmdline.exists():
                    break
                time.sleep(.5)
        record.unlink()
    if not record.exists():
        with (root / 'ordinary-bridge.log').open('ab', buffering=0) as log:
            process = subprocess.Popen([str(awomo / '.venv/bin/python'),
                str(root / 'scripts/all_rsi_skills_bridge.py'), '--port', '8180', '--policy-port', '8210'],
                env=pyenv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        save(record, {'pid': process.pid, 'port': 8180, 'policy_port': 8210})
    health = None
    for _ in range(90):
        try:
            with urllib.request.urlopen('http://127.0.0.1:8180/health', timeout=10) as response:
                health = json.load(response)
            if health['status'] == 'ready':
                break
        except OSError:
            pass
        time.sleep(1)
    if not health or health['status'] != 'ready':
        raise RuntimeError('New catalog endpoint failed readiness')
    save(root / 'catalog-health.json', health)
    # The old simulation and any user Chat must be idle before refreshing the
    # agent's binding. GPU benchmark workers use separate endpoints.
    with urllib.request.urlopen('http://127.0.0.1:8098/health', timeout=10) as response:
        old = json.load(response)
    if old.get('active_evaluation'):
        raise RuntimeError('Original bridge has an active evaluation; leave binding unchanged')
    chats = api('/agents/default/chats')
    chats = chats.get('chats', []) if isinstance(chats, dict) else chats
    if any(chat.get('status') == 'running' for chat in chats):
        raise RuntimeError('Default agent has a running Chat; leave binding unchanged')
    agent = api('/agents/default')
    workspace = Path(agent['workspace_dir'])
    # Runtime wrapper prose is relocatable; controller/source bytes stay pinned.
    payload = root / 'skills-relocated.zip'
    with zipfile.ZipFile(payload, 'w', zipfile.ZIP_DEFLATED) as archive:
        for skill in sorted((stage / 'skills').iterdir()):
            content = (skill / 'SKILL.md').read_text().replace(
                '/home/ymz/physclaw/awomopaw/runtime/physicalrsi-skills', str(assets))
            content = content.replace('/home/ymz/Documents/ChatGPT/robodojo/runtime/physicalrsi-all-skills', str(state_stage))
            archive.writestr(skill.name + '/SKILL.md', content)
    from qwenpaw.agents.skills_manager import SkillService
    service = SkillService(workspace)
    names = [skill.name for skill in sorted((stage / 'skills').iterdir())]
    if not any((workspace / 'skills' / name).exists() for name in names):
        imported = service.import_from_zip(payload.read_bytes(), enable=False)
        if imported.get('count') != 24:
            raise RuntimeError('Incomplete RSI skill import')
    for name in names:
        result = service.enable_skill(name)
        if not result.get('success'):
            raise RuntimeError('Could not enable ' + name)
    backup = root / 'original-default-embodiment.json'
    if not backup.exists():
        save(backup, agent['embodiments'])
    changed = copy.deepcopy(agent)
    changed['embodiments']['packages']['robodojo_sim']['bindings']['robodojo_sim_api'] = 'http://127.0.0.1:8180'
    api('/agents/default', changed, 'PUT')
    save(root / 'install-receipt.json', {
        'node': '192.168.48.153', 'existing_pod': 'qwenpaw-sim-harness-5b46b76889-hsv7r',
        'source_skills': 24, 'code_skills': 22, 'vla_skills': 2,
        'workspace_skills_enabled': names, 'source_root': str(assets),
        'bridge_url': 'http://127.0.0.1:8180', 'static_ready_providers': len(available),
        'vla_checkpoints_pending': not args.activate_vla,
        'original_controller_unchanged': True,
        'benchmark_execution': 'Direct frozen providers through AwomoPaw embodiment; no LLM Chat claim',
    })
    print(json.dumps({'installed': True, 'source_skills': 24, 'ready_providers': len(available),
                      'vla_checkpoints_pending': not args.activate_vla}), flush=True)


if __name__ == '__main__':
    main()
