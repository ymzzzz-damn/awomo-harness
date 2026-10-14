#!/usr/bin/env python3
"""Run five real GPT-5.6 harness Chats with a frozen simulation skill.

Creates a temporary loopback audit provider and layout fixture bridge; restores
the original agent settings and ordinary bridge even when evaluation fails.
No benchmark actions are selected or executed by this orchestration process.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import urllib.request

from compare_local_tasks import request, save, log, wait_gpu

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/home/ymz/physclaw/awomopaw/.venv/bin/python'
APP = 'http://127.0.0.1:8088/api'
MODEL = 'gpt-5.6-sol'
PROVIDER = 'align56-eval'
PROMPT = ('使用已安装的 PhysicalRSI physicalrsi-align-blocks skill，在 seed 0 完成 '
          'RoboDojo 中的 align_blocks。保持官方 instruction 和冻结技能代码不变，'
          '通过 robodojo_sim embodiment 执行，任务结束后保存官方评分和三路视频。')


def api(path, body=None, method=None):
    req = urllib.request.Request(APP + path,
        data=None if body is None else json.dumps(body).encode(),
        method=method, headers={'Content-Type': 'application/json', 'X-Agent-Id': 'car_controller'})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def wait_url(url, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                return json.load(response)
        except OSError:
            time.sleep(1)
    raise TimeoutError('Local evaluation service did not start')


def bridge_launch(path):
    subprocess.run(['tmux', 'respawn-pane', '-k', '-t', 'awomopaw-robodojo-bridge:0.0',
                    '/bin/bash ' + str(path)], check=True)
    time.sleep(2)
    wait_url('http://127.0.0.1:8098/health')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--interrupted-experiment', type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'process.json', {'pid': os.getpid(), 'started': datetime.now().astimezone().isoformat()})
    for name in ['run_gpt56_skill_chats.py', 'evaluate_fresh_chat_align.py',
                 'align56_audit_proxy.py', 'chat_eval_bridge.py', 'launch_align_fixture_bridge.sh']:
        snapshot = output / 'executed-source' / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / 'scripts' / name, snapshot)
    wait_gpu()
    health = request('/health')
    if health.get('active_evaluation'):
        raise RuntimeError('Another simulation is active; refuse temporary bridge replacement')
    chats = api('/agents/car_controller/chats')
    chat_list = chats if isinstance(chats, list) else chats.get('chats', [])
    if any(c.get('status') == 'running' for c in chat_list):
        raise RuntimeError('Another harness Chat is running; refuse model hot reload')
    providers = api('/models')
    if any(p['id'] == PROVIDER for p in providers):
        raise RuntimeError('Temporary evaluation provider already exists; do not overwrite')
    original = api('/agents/car_controller')
    backup = {k: original[k] for k in ['id', 'name', 'running', 'active_model']}
    save(output / 'original-agent-settings.json', backup)
    release = json.loads((ROOT / 'runtime/physicalrsi-align-skill/install-receipt.json').read_text())
    for item in release['verified_release_files']:
        if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() != item['sha256']:
            raise RuntimeError('Frozen skill release differs from installed manifest')
    save(output / 'frozen-skill-install-receipt.json', release)
    shutil.copy2(ROOT / 'runtime/physicalrsi-align-skill/gpt56-probe.json', output / 'model-probe.json')
    save(output / 'experiment.json', {'task': 'align_blocks', 'seed': 0, 'layouts': list(range(5)),
        'model': MODEL, 'upstream': 'https://ai.zhimiaonengzhi.com/v1', 'prompt': PROMPT,
        'fresh_chat_per_scene': True, 'max_iters': 400, 'llm_max_qpm': 60,
        'reasoning_effort': 'none', 'controller': 'unchanged frozen physicalrsi-align',
        'qualification': 'Explicit skill invocation through actual harness Chat, not bare-goal routing or LLM numeric policy',
        'started': datetime.now().astimezone().isoformat(),
        'interrupted_experiment': str(args.interrupted_experiment.resolve()) if args.interrupted_experiment else None,
        'prior_system_interruption_is_not_a_scored_task_failure': True})
    fixture = ROOT / 'runtime/physicalrsi-align-skill/chat-eval-fixture.json'
    launch = output / 'launch-fixture-bridge.sh'
    # This executable source file is copied from a tracked, reviewed template.
    shutil.copy2(ROOT / 'scripts/launch_align_fixture_bridge.sh', launch)
    proxy = None
    child = None
    created = False
    switched = False
    bridge_changed = False
    restore = {}
    log('EXPERIMENT_ROOT ' + str(output))
    try:
        with (output / 'audit-proxy.log').open('wb') as log_file:
            proxy = subprocess.Popen([PYTHON, str(ROOT / 'scripts/align56_audit_proxy.py'),
                '--output', str(output / 'model-audit')], stdout=log_file, stderr=subprocess.STDOUT)
        wait_url('http://127.0.0.1:8101/health')
        api('/models/custom-providers', {'id': PROVIDER, 'name': 'Temporary GPT-5.6 simulation audit',
            'default_base_url': 'http://127.0.0.1:8101/v1', 'models': [{
                'id': MODEL, 'name': MODEL, 'supports_image': True, 'supports_multimodal': True,
                'supports_video': False, 'probe_source': 'probed'}]})
        created = True
        api('/models/' + PROVIDER + '/config', {'api_key': 'local-align56-audit',
             'base_url': 'http://127.0.0.1:8101/v1'}, 'PUT')
        api('/models/' + PROVIDER + '/models/' + MODEL + '/config', {'generate_kwargs': {
            'api_mode': 'responses', 'reasoning': {'effort': 'none'}, 'max_output_tokens': 2048}}, 'PUT')
        changed = copy.deepcopy(backup)
        changed['running']['max_iters'] = 400
        changed['running']['llm_max_qpm'] = 60
        changed['active_model'] = {'provider_id': PROVIDER, 'model': MODEL}
        api('/agents/car_controller', changed, 'PUT')
        switched = True
        time.sleep(5)
        save(output / 'effective-model.json', api('/models/active?scope=effective&agent_id=car_controller'))
        bridge_changed = True
        bridge_launch(launch)
        with (output / 'driver.log').open('wb') as log_file:
            child = subprocess.Popen([PYTHON, str(ROOT / 'scripts/evaluate_fresh_chat_align.py'),
                '--output', str(output / 'chats'), '--fixture', str(fixture), '--prompt', PROMPT,
                '--audit-dir', str(output / 'model-audit'), '--expected-model', MODEL,
                '--layouts', '0', '1', '2', '3', '4'], stdout=log_file, stderr=subprocess.STDOUT)
        code = child.wait()
        if code:
            raise RuntimeError('Chat evaluator failed; exit=' + str(code))
        log('ALL_FIVE_CHAT_TESTS_FINISHED')
    except BaseException as error:
        save(output / 'orchestration-error.json', {'type': type(error).__name__, 'error': str(error)})
        raise
    finally:
        if child and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
        try:
            context_file = output / 'audit-context.json'
            context = json.loads(context_file.read_text()) if context_file.exists() else {}
            if context.get('chat_id'):
                current = api('/agents/car_controller/chats/' + context['chat_id'])
                if current.get('status') == 'running':
                    api('/agents/car_controller/console/chat/stop?chat_id=' + context['chat_id'], {}, 'POST')
                    deadline = time.monotonic() + 30
                    while time.monotonic() < deadline:
                        if api('/agents/car_controller/chats/' + context['chat_id']).get('status') != 'running':
                            break
                        time.sleep(1)
                    else:
                        raise RuntimeError('Own Chat did not stop; refuse bridge reload')
            active = request('/health').get('active_evaluation')
            if active:
                artifact = Path(active['artifact_dir']) / 'chat-eval-fixture.json'
                if artifact.exists() and json.loads(artifact.read_text()).get('test_id') == str(output / 'chats'):
                    request('/v1/sessions/abort', {'reason': 'Cleanup of this GPT-5.6 evaluation only'})
                else:
                    raise RuntimeError('Another evaluation is active; refuse bridge reload')
            deadline = time.monotonic() + 30
            while request('/health').get('active_evaluation'):
                if time.monotonic() >= deadline:
                    raise RuntimeError('Episode did not stop; refuse bridge reload')
                time.sleep(1)
            if bridge_changed:
                bridge_launch(ROOT / 'runtime/physicalrsi-align-skill/launch-bridge.sh')
            restore['ordinary_bridge_restored'] = True
        except Exception as error:
            restore['bridge_restore_error'] = str(error)
        try:
            if switched:
                api('/agents/car_controller', backup, 'PUT')
                time.sleep(3)
                restore['active_model'] = api('/models/active?scope=effective&agent_id=car_controller')
                restore['original_running_restored'] = api('/agents/car_controller')['running'] == backup['running']
            if created:
                api('/models/custom-providers/' + PROVIDER, method='DELETE')
                restore['temporary_provider_removed'] = True
        except Exception as error:
            restore['agent_restore_error'] = str(error)
        if proxy:
            proxy.terminate()
            proxy.wait(timeout=10)
        save(output / 'restoration.json', restore)
        save(output / 'frozen-source-check-after.json', [dict(item, unchanged=
            hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256'])
            for item in release['verified_release_files']])
        log('RESTORATION ' + json.dumps(restore))


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    main()
