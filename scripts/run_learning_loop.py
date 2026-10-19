#!/usr/bin/env python3
"""Portable GPT direct control, three-camera review and same-task skill reuse."""
import argparse
import asyncio
import fcntl
import json
from pathlib import Path
import re
import subprocess
import sys
import time

import gpt6_episode as control
import harness_utils as base
from gpt6_failure_learning import digest, save
from gpt6_task_learning import empty_task_memory, learn, validate_task_memory
from harness_models import SUPPORTED_MODELS, checked_model

ROOT = Path(__file__).resolve().parents[1]
ARM = 'self_evolving'


def inventory(root, task, seeds, layouts):
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,127}', task):
        raise ValueError('Invalid task name')
    if len(set(seeds)) != len(seeds) or len(set(layouts)) != len(layouts):
        raise ValueError('Duplicate seed/layout would replay a scene')
    cases = []
    for seed in seeds:
        for layout in layouts:
            if seed < 0 or layout < 0:
                raise ValueError('Seed/layout must be nonnegative')
            path = root/'Assets/Eval_Layout/RoboDojo/arx_x5'/str(seed)/f'{task}_{layout}.json'
            cases.append(dict(task=task, seed=seed, layout_id=layout, case_index=len(cases),
                              layout_file=str(path), layout_sha256=digest(path)))
    return cases


def case_directory(output, case):
    return output/f'seed-{case["seed"]}'/f'layout-{case["layout_id"]:02d}'/ARM


async def review_case(folder, snapshot, config, review=learn):
    """Resume post-task work without rerunning an official episode.

    Incomplete attempts stay available for diagnosis. A committed receipt can
    be recovered after an interruption between distillation and publication.
    """
    root = folder/'task-learning'
    root.mkdir(exist_ok=True)
    committed = root/'receipt.json'
    attempts = sorted(root.glob('attempt-*'))
    for attempt in attempts:
        receipt = attempt/'receipt.json'
        if receipt.is_file() and not committed.exists():
            saved = json.loads(receipt.read_text())
            if saved.get('status') == 'completed':
                save(root/'experience-after.json', json.loads((attempt/'experience-after.json').read_text()))
                save(committed, dict(saved, analysis_output=str(attempt)))
                break
    if not committed.exists():
        attempt = root/f'attempt-{len(attempts)+1:03d}'
        await review(folder, snapshot, config, attempt)
        saved = json.loads((attempt/'receipt.json').read_text())
        save(root/'experience-after.json', json.loads((attempt/'experience-after.json').read_text()))
        save(committed, dict(saved, analysis_output=str(attempt)))
    saved = json.loads(committed.read_text())
    row = json.loads((folder/'receipt.json').read_text())
    after = root/'experience-after.json'
    if (saved.get('status') != 'completed' or saved['before_sha256'] != digest(snapshot) or
            saved['after_sha256'] != digest(after) or saved['source_run_id'] != row['run_id']):
        raise ValueError('Post-task review provenance mismatch')
    return json.loads(after.read_text())


def execute(args, cases, initial_reference=None, run_case=control.run_case, review=learn):
    experiment = args.output.name
    memory = empty_task_memory(experiment, args.task, initial_reference)
    results = args.output/'results.json'
    rows = json.loads(results.read_text()) if results.exists() else []
    if len(rows) > len(cases) or any(r.get('case_index') != i for i, r in enumerate(rows)):
        raise ValueError('Results contain unordered/duplicate cases')

    def publish(status, phase):
        save(results, rows)
        save(args.output/'status.json', dict(schema='awomopaw.learning-loop/v1',
            status=status, phase=phase, task=args.task, requested=len(cases), finished=len(rows),
            reviewed=memory['version'], successes=sum(bool(r.get('success')) for r in rows),
            task_artifact=memory.get('task_artifact'), updated_at=base.now()))

    try:
        for case in cases:
            index = case['case_index']
            validate_task_memory(memory, experiment, index, args.task)
            snapshot = args.output/'learning/snapshots'/f'case-{index:04d}.json'
            if snapshot.exists() and json.loads(snapshot.read_text()) != memory:
                raise ValueError('Immutable experience snapshot mismatch')
            if not snapshot.exists():
                save(snapshot, memory)
            args.experience_snapshot = snapshot
            folder = case_directory(args.output, case)
            publish('running', dict(type='episode', case_index=index))
            if index < len(rows):
                row = rows[index]
            elif (folder/'receipt.json').is_file():
                row = json.loads((folder/'receipt.json').read_text())
                rows.append(row)
            else:
                if folder.exists():
                    raise RuntimeError('Interrupted episode directory exists; refusing automatic replay: '+str(folder))
                row = run_case(case, ARM, args.gpu, args)
                rows.append(row)
            publish('running', dict(type='video_review', case_index=index))
            if (not row.get('valid_episode') or any(row.get(k) != case[k] for k in
                    ['task', 'seed', 'layout_id', 'case_index', 'layout_sha256'])):
                raise RuntimeError('Episode is unscored, incomplete or mismatched; inspect '+str(folder))
            if json.loads((folder/'receipt.json').read_text()) != row:
                raise ValueError('Saved episode receipt differs from scheduler results')
            if row.get('experience_sha256') != digest(snapshot):
                raise ValueError('Episode used another experience snapshot')
            memory = asyncio.run(review_case(folder, snapshot, args.output/'agent-config.json', review))
            validate_task_memory(memory, experiment, index+1, args.task)
            if memory['version'] != index+1:
                raise ValueError('Every completed episode must have one video review')
            save(args.output/'experience.json', memory)
            publish('running', dict(type='between_scenes', case_index=index))
        publish('completed', dict(type='completed'))
    except BaseException as error:
        publish('needs_attention', dict(type=type(error).__name__, error=str(error)))
        raise
    return memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sim-root', type=Path, required=True)
    parser.add_argument('--registry', type=Path, required=True, help='registry.json from configure_robodojo.py')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--layouts', type=int, nargs='+', required=True)
    slot = parser.add_mutually_exclusive_group(required=True)
    slot.add_argument('--agent-config', type=Path, help='Read only active_model; use locally stored provider credentials')
    slot.add_argument('--provider', help='An already configured OpenAI-compatible provider ID')
    parser.add_argument('--model', choices=SUPPORTED_MODELS, default='gpt-6-astra')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--port-base', type=int, default=26000)
    parser.add_argument('--segment-actions', type=int, choices=range(1, 41), default=40)
    parser.add_argument('--case-timeout', type=int, default=5400)
    parser.add_argument('--task-artifact', type=Path, help='Import a verified successful skill for this same task')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Check assets, registry, provider and agent tools without starting Isaac')
    parser.add_argument('--probe-model', action='store_true', help='With --dry-run, also make one image API call')
    args = parser.parse_args()
    if args.probe_model and not args.dry_run:
        parser.error('--probe-model requires --dry-run')
    args.sim_root = args.sim_root.resolve()
    args.output = args.output.resolve()
    args.python = Path(sys.executable)
    args.awomopaw = ROOT
    args.gpus = [args.gpu]
    cases = inventory(args.sim_root, args.task, args.seeds, args.layouts)
    if args.gpu < 0 or not 1024 <= args.port_base <= 65532-max(c['layout_id'] for c in cases)*6:
        parser.error('Invalid GPU or port range')
    registry = json.loads(args.registry.read_text())
    if (not isinstance(registry, list) or len(registry) != 1 or
            registry[0].get('provider_id') != 'gpt6-harness' or registry[0].get('kind') != 'direct'):
        parser.error('Use the direct-control registry from configure_robodojo.py')
    for path in ['scripts/robodojo.sh', 'XPolicyLab/setup_policy_server.py',
                 'src/eval_client/interactive_harness.py', 'XPolicyLab/policy/GPT6_Harness_Ablation/model.py']:
        if not (args.sim_root/path).is_file():
            parser.error('Missing configured simulator file: '+path)
    if not (Path(registry[0]['policy_uv_env_path'])/'.venv/bin/python').is_file():
        parser.error('Transport Python is missing')
    selected = (json.loads(args.agent_config.read_text())['active_model'] if args.agent_config else
                dict(provider_id=args.provider, model=args.model))
    checked_model(selected['model'])
    from qwenpaw.providers import ProviderManager
    from qwenpaw.config.config import AgentProfileConfig
    provider = ProviderManager.get_instance().get_provider(selected['provider_id'])
    if provider is None or not callable(getattr(provider, '_client', None)):
        parser.error('Configure an OpenAI-compatible provider with awomopaw models first')
    try:
        from importlib import import_module
        for module in ['numpy', 'cv2', 'h5py', 'msgpack_numpy', 'websockets.asyncio.server']:
            import_module(module)
    except ImportError as error:
        parser.error('Install source[simulation,transport] in this Python: '+str(error))
    profile = AgentProfileConfig.model_validate(dict(id='default', name='AwomoPaw learning loop',
        active_model=selected, embodiments=dict(packages=dict(robodojo_sim=dict(
            package_id='robodojo_sim', path=str(ROOT/'embodiments/robodojo_sim'), enabled=True)))))
    reference = None
    if args.task_artifact:
        path = args.task_artifact.resolve()
        reference = dict(path=str(path), sha256=digest(path), revision=json.loads(path.read_text())['revision'])
    initial = empty_task_memory(args.output.name, args.task, reference)
    manifest = dict(schema='awomopaw.learning-loop/v1', task=args.task, cases=cases,
        model=selected, registry=registry, sim_root=str(args.sim_root), gpu=args.gpu,
        port_base=args.port_base, segment_actions=args.segment_actions,
        case_timeout=args.case_timeout, initial_task_artifact=reference,
        official_action_budget_unchanged=True, automatic_episode_retries=0)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    with (args.output/'.loop.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.resume:
            if json.loads((args.output/'manifest.json').read_text()) != manifest:
                raise ValueError('Run configuration or official assets changed; use another output directory')
            if (json.loads((args.output/'agent-config.json').read_text())['active_model'] != selected or
                    json.loads((args.output/'registry.snapshot.json').read_text()) != registry):
                raise ValueError('Pinned agent/transport configuration changed')
        else:
            save(args.output/'manifest.json', manifest)
            save(args.output/'agent-config.json', profile.model_dump(mode='json'))
            save(args.output/'registry.snapshot.json', registry)
            save(args.output/'experience.json', initial)
        if args.dry_run:
            snapshot = args.output/'preflight/experience.json'
            save(snapshot, initial)
            command = [sys.executable, str(ROOT/'scripts/run_gpt6_harness_agent.py'),
                '--output', str(args.output/'preflight'), '--agent-config', str(args.output/'agent-config.json'),
                '--bridge', 'http://127.0.0.1:'+str(args.port_base), '--arm', ARM,
                '--skill', str(args.output/'skill'), '--experience', str(snapshot), '--task', args.task,
                '--seed', str(args.seeds[0]), '--model', selected['model'], '--dry-run']
            if args.probe_model:
                command.append('--probe-model')
            subprocess.run(command, check=True)
            save(args.output/'status.json', dict(status='preflight_passed', no_episode_started=True))
            return
        locks = base.acquire_gpu_locks(args, time.monotonic()+300)
        try:
            execute(args, cases, reference)
        finally:
            for lock in locks:
                lock.close()


if __name__ == '__main__':
    main()
