#!/usr/bin/env python3
"""Finite node-153-only sweep, frozen align skill, isolated GPU and ports.

No local fallback, LLM call, controller mutation, infrastructure retry or skipped
scored failure. Uses the already deployed unchanged provider and embodiment.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import run_full_frozen_skills as full
import run_parallel_frozen_skills as base

NODE = '192.168.48.153'
REVISION = 'frozen-code:70a3d5ce6af076eaad5bf22ffc81acf6057485fe8340cb33faaa320da9651feb'


def summary(cases, rows, status):
    result = full.aggregate(cases, rows, rows, status)
    unscored = [r for r in rows if r.get('unscored_native_scene')]
    result['infrastructure_errors'] -= len(unscored)
    result['infrastructure_attempt_errors'] -= len(unscored)
    result['unscored_native_scenes'] = len(unscored)
    result['complete'] = len(rows) == len(cases) and result['infrastructure_errors'] == 0
    result['platform'] = 'P6000 / node ' + NODE
    result['node'] = NODE
    result['attempted'] = len(rows)
    result['success_rate_so_far'] = result['success_rate']
    result['per_seed'] = {}
    for seed in sorted({c['seed'] for c in cases}):
        scored = [r for r in rows if r['seed'] == seed and r.get('valid_episode')]
        result['per_seed'][str(seed)] = {
            'requested': sum(c['seed'] == seed for c in cases), 'valid_episodes': len(scored),
            'successes': sum(bool(r['success']) for r in scored),
            'success_rate': sum(bool(r['success']) for r in scored) / len(scored) if scored else None}
    result['episodes'] = rows
    return result


def publish(output, cases, rows, status):
    base.save(output / 'results.json', rows)
    base.save(output / 'attempts.json', rows)
    base.save(output / 'summary.json', summary(cases, rows, status))
    base.save(output / 'video-index.json', [
        {'seed': r['seed'], 'layout_id': r['layout'], 'run_id': r.get('run_id'),
         'status': r['status'], 'score': r.get('score'), 'success': r.get('success'),
         'videos': r.get('videos', [])} for r in rows])


def unchanged(files):
    return all(full.digest(Path(path)) == expected for path, expected in files.items())


def rejected_scene(row):
    if row.get('valid_episode') or row.get('actions') != 0:
        return False
    dest = Path(row['output'])
    for name in ['official-result.json', 'failure-official-result.json']:
        record = dest / name
        if not record.is_file():
            continue
        official = json.loads(record.read_text())
        run = official['run']
        log = Path(run['log_path'])
        if (run.get('status') == 'completed' and run.get('return_code') == 0
                and (run.get('result') or {}).get('eval_time') == 0 and log.is_file()
                and 'All scene Unstable Error!' in log.read_text()):
            row.update(status='native_unstable_before_actions', unscored_native_scene=True,
                       classification='Native simulator rejected scene before any policy action; no score')
            shutil.copy2(log, dest / 'native-scene-rejection.log')
            base.save(dest / 'receipt.json', row)
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sim-root', type=Path, default=Path('/opt/robodojo'))
    parser.add_argument('--awomopaw', type=Path, required=True)
    parser.add_argument('--registry', type=Path, required=True)
    parser.add_argument('--expected-release', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=5)
    parser.add_argument('--port-base', type=int, default=24000)
    parser.add_argument('--case-timeout', type=int, default=3600)
    args = parser.parse_args()
    if os.environ.get('P6000_ALIGN_VERIFIED_NODE') != NODE or args.sim_root != Path('/opt/robodojo'):
        raise RuntimeError('Refuse any platform except the verified P6000 node 153 pod')
    if args.gpu not in [5, 6, 7]:
        raise RuntimeError('Reserved benchmark GPU must be 5, 6 or 7; leave existing jobs untouched')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    base.save(args.output / 'process.json', {'pid': os.getpid(), 'started_at': base.now(), 'node': NODE})
    cases = full.inventory(args.sim_root, ['align_blocks'])
    expected = {(seed, layout) for seed in range(3) for layout in range(55)}
    if {(c['seed'], c['layout']) for c in cases} != expected or len(cases) != 165:
        raise RuntimeError('Expected all 165 official align_blocks scenes; refuse partial sweep')
    original_registry = json.loads(args.registry.read_text())
    provider = next(p for p in original_registry if p['provider_id'] == 'physicalrsi-align-blocks')
    if provider['kind'] != 'code' or provider['revision'] != REVISION:
        raise RuntimeError('Remote provider is not the original frozen align skill')
    catalog = json.loads((args.sim_root / 'XPolicyLab/policy/PhysicalRSI_Frozen_Skills/catalog.json').read_text())
    release = catalog['releases'][provider['checkpoint']]
    expected_files = json.loads(args.expected_release.read_text())
    if release['configuration']['files'] != expected_files or len(expected_files) != 22:
        raise RuntimeError('Remote release does not match all 22 previously frozen files')
    hashes = {template.replace('{root}', catalog['asset_root']): value for template, value in expected_files.items()}
    if not unchanged(hashes):
        raise RuntimeError('Frozen align source hash mismatch')
    for c in cases:
        c.update(provider_id=provider['provider_id'], action_type='joint')
    scripts = Path(__file__).resolve().parent
    script_hashes = {str(p.relative_to(scripts.parent)): full.digest(p) for p in scripts.glob('*.py')}
    base.save(args.output / 'manifest.json', {'schema': 'awomopaw.full-official-scenes/v1',
        'node': NODE, 'execution': 'direct_frozen_provider', 'llm_used': False,
        'cases': cases, 'source_hashes': hashes, 'script_hashes': script_hashes,
        'registry_sha256': full.digest(args.registry), 'gpus': [args.gpu], 'infra_retries': 0})
    base.save(args.output / 'official-case-inventory.json', [dict(c, layout_id=c['layout']) for c in cases])
    base.save(args.output / 'registry.snapshot.json', [provider])
    base.save(args.output / 'experiment.json', {'task': 'align_blocks', 'platform': 'P6000', 'node': NODE,
        'provider': provider['provider_id'], 'revision': REVISION, 'requested_scenes': 165,
        'seeds': [0, 1, 2], 'layouts': list(range(55)), 'gpu': args.gpu,
        'llm_calls': False, 'controller_modified': False, 'local_fallback_allowed': False})
    args.gpus = [args.gpu]
    locks = base.acquire_gpu_locks(args, time.monotonic() + 3600)
    rows = []
    status = 'running'
    def interrupted(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        publish(args.output, cases, rows, status)
        for case in cases:
            if not unchanged(hashes) or full.digest(Path(case['layout_file'])) != case['layout_sha256']:
                raise RuntimeError('Frozen source or official scene changed')
            deadline = time.monotonic() + 900
            while base.gpu_memory(args.gpu)['used_mib'] > 1024:
                base.save(args.output / 'status.json', {'status': 'waiting_for_reserved_gpu', 'gpu': args.gpu})
                if time.monotonic() >= deadline:
                    raise RuntimeError('Reserved GPU occupied by another job; do not run elsewhere')
                time.sleep(5)
            port = args.port_base + case['index'] * 3
            if not all(base.port_available(port + i) for i in range(3)):
                raise RuntimeError('Isolated evaluation port conflict')
            case = dict(case, attempt=1, gpu=args.gpu, bridge_port=port, control_port=port + 1,
                policy_port=port + 2, layout_id=case['layout'], provider=provider['provider_id'],
                output=str(args.output / f'episode-{case["index"]:03d}-seed-{case["seed"]}-layout-{case["layout"]}'))
            base.save(Path(case['output']) / 'case.json', case)
            base.save(args.output / 'status.json', {'status': 'running', 'node': NODE,
                'finished': len(rows), 'requested': 165, 'active': case})
            print(base.now(), 'START', case['case_id'], 'GPU', args.gpu, flush=True)
            row = full.worker(case, args)
            state_file = Path(case['output']) / 'run.json'
            if state_file.exists():
                native_run = json.loads(state_file.read_text())
                native_log = Path(native_run['log_path'])
                if native_log.exists():
                    shutil.copy2(native_log, Path(case['output']) / 'native-run.log')
            rejected_scene(row)
            rows.append(row)
            publish(args.output, cases, rows, status)
            print(base.now(), 'END', case['case_id'], row['status'], row.get('score'), flush=True)
            if not row.get('valid_episode') and not row.get('unscored_native_scene'):
                raise RuntimeError('Infrastructure error; retain evidence and stop without rerunning or skipping')
        status = 'completed'
    except BaseException as error:
        status = 'stopped_by_user' if isinstance(error, KeyboardInterrupt) else 'failed'
        base.save(args.output / 'orchestration-error.json', {'type': type(error).__name__, 'error': str(error)})
        raise
    finally:
        restore = {'frozen_sources_unchanged': unchanged(hashes), 'owned_services_cleaned': all(
            r.get('owned_services_cleaned') for r in rows), 'ordinary_bridge_modified': False,
            'ordinary_bridge_restored': True, 'node': NODE}
        base.save(args.output / 'restoration.json', restore)
        base.save(args.output / 'status.json', {'status': status, 'node': NODE,
            'finished': len(rows), 'requested': 165, 'active': None})
        publish(args.output, cases, rows, status)
        for lock in locks:
            lock.close()
    if not unchanged(hashes):
        raise RuntimeError('Frozen source changed during evaluation')
    subprocess.run([sys.executable, str(scripts / 'verify_full_frozen_results.py'),
                    str(args.output), '--runner-scripts', str(scripts)], check=True)
    verified_path = args.output / 'verified-summary.json'
    verified = json.loads(verified_path.read_text())
    verified.update(summary(cases, rows, status))
    verified['final_success_rate'] = verified['success_rate']
    verified['frozen_source_files_verified'] = len(hashes)
    verified['scene_coverage_verified'] = len(rows) == len(cases)
    base.save(verified_path, verified)


if __name__ == '__main__':
    main()
