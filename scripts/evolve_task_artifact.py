#!/usr/bin/env python3
"""Post-task video reflection and task skill/policy distillation, without Isaac."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'PhysicalRSI'))
sys.path.insert(0, str(ROOT / 'runtime/physicalrsi-evolution'))
from run_evolution import LLM
from PhysicalRSI_baselines.robodojo.task_evolution import evolve_task
from PhysicalRSI_core.infra.storage import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-result', type=Path, required=True)
    parser.add_argument('--initial-observation', type=Path, required=True)
    parser.add_argument('--task-artifact', type=Path)
    parser.add_argument('--skill', type=Path, help='Existing task SKILL.md')
    parser.add_argument('--source', type=Path, help='Existing task policy source')
    parser.add_argument('--memory', type=Path)
    parser.add_argument('--primitive-api', type=Path)
    parser.add_argument('--reflections', type=Path)
    parser.add_argument('--provider-config', type=Path)
    parser.add_argument('--model', default='gpt-6-astra')
    parser.add_argument('--split', choices=['development', 'validation', 'final'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.split != 'development':
        parser.error('Only completed development outcomes may enter distillation')
    if args.task_artifact and any((args.skill, args.source, args.memory)):
        parser.error('Supply a parent artifact or initial skill/source/memory files')
    final = json.loads(args.official_result.read_text())
    initial = json.loads(args.initial_observation.read_text())
    if initial['run_id'] != final['run']['run_id'] or initial['task'] != final['run']['task']:
        parser.error('Initial observation and result must refer to the same run and task')
    if final['run']['status'] != 'completed':
        parser.error('A completed native result is required')
    parent = json.loads(args.task_artifact.read_text()) if args.task_artifact else None
    if not parent and any((args.skill, args.source, args.memory)):
        content = dict(skill_text=args.skill.read_text() if args.skill else '',
                       source=args.source.read_text() if args.source else '',
                       memory=json.loads(args.memory.read_text()) if args.memory else {})
        parent = dict(schema='awomopaw.task-artifact/v1', task=initial['task'],
                      revision=digest(content), content=content)
    candidate = evolve_task(model=LLM(provider_config=args.provider_config, model=args.model),
        final=final, instruction=initial['instruction'], parent=parent, output=args.output,
        primitive_api=json.loads(args.primitive_api.read_text()) if args.primitive_api else None,
        reflections=json.loads(args.reflections.read_text()) if args.reflections else [], split=args.split)
    print(json.dumps({'status': candidate['status'], 'revision': candidate['revision'],
                      'artifact': str((args.output / 'task-artifact.json').resolve())}))


if __name__ == '__main__':
    main()
