#!/usr/bin/env python3
"""Scoped recovery of this interrupted experiment; preserve its original data."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import time

from compare_local_tasks import save
from run_gpt56_skill_chats import api, wait_url, ROOT, PYTHON, PROVIDER


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interrupted-experiment', type=Path, required=True)
    parser.add_argument('--new-output', type=Path, required=True)
    args = parser.parse_args()
    old = args.interrupted_experiment.resolve()
    new = args.new_output.resolve()
    if new.exists() or new.with_name(new.name + '.launcher.log').exists():
        raise RuntimeError('Refuse to overwrite a previous recovery attempt')
    original = json.loads((old / 'original-agent-settings.json').read_text())
    current = api('/agents/car_controller')
    chats = api('/agents/car_controller/chats')
    items = chats if isinstance(chats, list) else chats.get('chats', [])
    if any(c.get('status') == 'running' for c in items):
        raise RuntimeError('A Chat is running; do not change its model')
    expected_running = copy.deepcopy(original['running'])
    expected_running.update(max_iters=400, llm_max_qpm=60)
    if current['active_model'] == {'provider_id': PROVIDER, 'model': 'gpt-5.6-sol'}:
        if current['running'] != expected_running:
            raise RuntimeError('Agent settings changed outside this experiment; refuse overwrite')
        api('/agents/car_controller', original, 'PUT')
        time.sleep(3)
    elif (current['active_model'] != original['active_model']
          or current['running'] != original['running']):
        raise RuntimeError('Agent settings are not owned by this interrupted experiment')
    providers = api('/models')
    if any(p['id'] == PROVIDER for p in providers):
        profiles = api('/agents')['agents']
        if any((p.get('active_model') or {}).get('provider_id') == PROVIDER for p in profiles):
            raise RuntimeError('Another agent uses the temporary provider; refuse deletion')
        global_model = api('/models/active?scope=global').get('active_llm') or {}
        if global_model.get('provider_id') == PROVIDER:
            raise RuntimeError('Temporary provider is globally active; refuse deletion')
        api('/models/custom-providers/' + PROVIDER, method='DELETE')
    ordinary = ROOT / 'runtime/physicalrsi-align-skill/launch-bridge.sh'
    session = 'awomopaw-robodojo-bridge'
    if subprocess.run(['tmux', 'has-session', '-t', session], capture_output=True).returncode:
        subprocess.run(['tmux', 'new-session', '-d', '-s', session, '/bin/bash ' + str(ordinary)], check=True)
    health = wait_url('http://127.0.0.1:8098/health')
    if health.get('active_evaluation'):
        raise RuntimeError('Another simulation is active; do not start this recovery')
    if not any(p['provider_id'] == 'physicalrsi-align' and p['ready'] for p in health['available_policies']):
        raise RuntimeError('Frozen provider is not ready after bridge recovery')
    restored = api('/agents/car_controller')
    receipt = {'original_active_model_restored': restored['active_model'] == original['active_model'],
        'original_running_restored': restored['running'] == original['running'],
        'ordinary_bridge_ready': True, 'temporary_provider_removed': True,
        'resumed_experiment': str(new), 'system_boot_local': subprocess.check_output(['uptime', '-s'], text=True).strip()}
    save(old / 'recovery/restored-before-resume.json', receipt)
    detached = 'awomopaw-gpt56-eval'
    if subprocess.run(['tmux', 'has-session', '-t', detached], capture_output=True).returncode == 0:
        raise RuntimeError('An evaluator session already exists; refuse another launch')
    # No shell credential or output redirection: a fixed Python launcher owns logs.
    command = [PYTHON, str(ROOT / 'scripts/launch_gpt56_detached.py'), '--output', str(new),
               '--interrupted-experiment', str(old)]
    subprocess.run(['tmux', 'new-session', '-d', '-s', detached, *command], check=True)
    save(old / 'recovery/resume-request.json', {'new_output': str(new), 'tmux_session': detached,
        'user_authorization': 'Human user explicitly said 继续 after the interruption',
        'no_previous_scored_episodes': True, 'same_seed_layouts_model_prompt_and_frozen_strategy': True})
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
