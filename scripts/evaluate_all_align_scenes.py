#!/usr/bin/env python3
"""Evaluate every available official align_blocks seed/layout with a frozen skill.

No LLM calls or controller modifications. The ordinary embodiment and native
scorer retain ownership of all Isaac actions and evaluation results.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time

import compare_local_tasks as control
from evaluate_frozen_align_skill import episode, ROOT, SIM

PYTHON = '/home/ymz/physclaw/awomopaw/.venv/bin/python'


def inventory():
    source = SIM / 'Assets/Eval_Layout/RoboDojo/arx_x5'
    cases = []
    for directory in sorted(source.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 999999):
        if not directory.is_dir() or not directory.name.isdigit():
            continue
        files = sorted(directory.glob('align_blocks_*.json'), key=lambda p: int(p.stem.rsplit('_', 1)[1]))
        ids = [int(p.stem.rsplit('_', 1)[1]) for p in files]
        if ids != list(range(len(files))):
            raise RuntimeError('Native fixture bridge requires dense official layout IDs')
        for layout, path in zip(ids, files):
            cases.append({'seed': int(directory.name), 'layout_id': layout, 'file': str(path),
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    if not cases:
        raise RuntimeError('No official align_blocks layouts found')
    return cases


def bridge(path):
    subprocess.run(['tmux', 'respawn-pane', '-k', '-t', 'awomopaw-robodojo-bridge:0.0',
                    '/bin/bash ' + str(path)], check=True)
    time.sleep(2)
    deadline = time.monotonic() + 30
    while True:
        try:
            control.request('/health', timeout=3)
            return
        except OSError:
            if time.monotonic() > deadline:
                raise RuntimeError('Embodiment bridge did not become ready')
            time.sleep(1)


def native_unstable_before_actions(row, output, index, seed, layout):
    """Only native, pre-action scene rejection may be recorded as unscored."""
    if row.get('valid_episode') or row.get('actions') != 0:
        return False
    dest = output / f'episode-{index:02d}-seed-{seed}-layout-{layout}'
    record = dest / 'official-result.json'
    if not record.exists():
        return False
    official = json.loads(record.read_text())
    run = official['run']
    if run.get('status') != 'completed' or run.get('return_code') != 0:
        return False
    if (run.get('result') or {}).get('eval_time') != 0:
        return False
    log = Path(run['log_path']).read_text()
    if 'All scene Unstable Error!' not in log:
        return False
    row.update(status='native_unstable_before_actions', unscored_native_scene=True,
               classification='Official simulator rejected scene before any policy action; no score')
    shutil.copy2(run['log_path'], dest / 'native-scene-rejection.log')
    control.save(dest / 'receipt.json', row)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    control.save(output / 'process.json', {'pid': os.getpid(), 'started': datetime.now(timezone.utc).isoformat()})
    cases = inventory()
    control.save(output / 'official-case-inventory.json', cases)
    receipt = json.loads((ROOT / 'runtime/physicalrsi-align-skill/install-receipt.json').read_text())
    for item in receipt['verified_release_files']:
        if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() != item['sha256']:
            raise RuntimeError('Frozen release identity changed; refuse evaluation')
    control.save(output / 'frozen-release.json', receipt)
    control.save(output / 'experiment.json', {'task': 'align_blocks', 'platform': 'local RTX 4090',
        'provider': 'physicalrsi-align', 'requested_scenes': len(cases),
        'seeds': sorted({c['seed'] for c in cases}), 'all_available_official_layouts': True,
        'llm_calls': False, 'controller_modified': False,
        'remote_attempt': 'P6000 SSH authentication rejected; local fallback under user request',
        'qualification': 'Frozen task-specific skill through harness embodiment; not GPT-5.6 or zero-shot planning'})
    for name in ['evaluate_all_align_scenes.py', 'evaluate_frozen_align_skill.py',
                 'chat_eval_bridge.py', 'launch_align_fixture_bridge.sh']:
        target = output / 'executed-source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / 'scripts' / name, target)
    control.wait_gpu()
    if control.request('/health').get('active_evaluation'):
        raise RuntimeError('Another simulation owns the embodiment')
    fixture = ROOT / 'runtime/physicalrsi-align-skill/chat-eval-fixture.json'
    rows = []
    changed = False
    try:
        changed = True
        bridge(ROOT / 'scripts/launch_align_fixture_bridge.sh')
        for index, case in enumerate(cases):
            row = episode(case['seed'], index, output, layout=case['layout_id'], fixture=fixture)
            native_unstable_before_actions(row, output, index, case['seed'], case['layout_id'])
            rows.append(row)
            valid = [r for r in rows if r['valid_episode']]
            unstable = [r for r in rows if r.get('unscored_native_scene')]
            infrastructure = len(rows) - len(valid) - len(unstable)
            successes = sum(r['success'] for r in valid)
            groups = {}
            for seed in sorted({c['seed'] for c in cases}):
                sample = [r for r in valid if r['seed'] == seed]
                groups[str(seed)] = {'valid_episodes': len(sample), 'successes': sum(r['success'] for r in sample),
                    'requested': sum(c['seed'] == seed for c in cases)}
            control.save(output / 'summary.json', {'requested': len(cases), 'attempted': len(rows),
                'valid_episodes': len(valid), 'successes': successes,
                'success_rate_so_far': successes / len(valid) if valid else None,
                'complete': len(rows) == len(cases) and infrastructure == 0,
                'unscored_native_scenes': len(unstable), 'infrastructure_errors': infrastructure,
                'per_seed': groups, 'episodes': rows})
            if not row['valid_episode'] and not row.get('unscored_native_scene'):
                raise RuntimeError('Infrastructure error; preserve failed attempt and stop instead of skipping scene')
    except BaseException as error:
        control.save(output / 'orchestration-error.json', {'type': type(error).__name__, 'error': str(error)})
        raise
    finally:
        restore = {}
        active = control.request('/health').get('active_evaluation')
        if active:
            case = Path(active['artifact_dir']) / 'chat-eval-fixture.json'
            if case.exists() and json.loads(case.read_text()).get('test_id') == output.name:
                control.request('/v1/sessions/abort', {'reason': 'Cleanup of this full align_blocks evaluation only'})
            else:
                raise RuntimeError('Another episode is active; refuse bridge replacement')
        deadline = time.monotonic() + 30
        while control.request('/health').get('active_evaluation'):
            if time.monotonic() > deadline:
                raise RuntimeError('Own episode did not stop; refuse bridge replacement')
            time.sleep(1)
        if changed:
            bridge(ROOT / 'runtime/physicalrsi-align-skill/launch-bridge.sh')
        restore['ordinary_bridge_restored'] = True
        restore['frozen_sources_unchanged'] = all(hashlib.sha256(Path(i['path']).read_bytes()).hexdigest() == i['sha256']
                                                 for i in receipt['verified_release_files'])
        control.save(output / 'restoration.json', restore)
        control.log('RESTORATION ' + json.dumps(restore))


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    main()
