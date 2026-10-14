#!/usr/bin/env python3
"""Return failed development video evidence, diagnosis and rethinking to harness.

Accepts a completed single-scene bridge result. No simulator or GPU is started.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'PhysicalRSI'))
sys.path.insert(0, str(ROOT / 'runtime/physicalrsi-evolution'))

from PhysicalRSI_baselines.robodojo.rethinking import VideoRethinker, failed_episodes
from PhysicalRSI_core.infra.storage import atomic_json
from run_evolution import LLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official-result', type=Path, required=True)
    parser.add_argument('--instruction', required=True, help='Unmodified native task instruction')
    parser.add_argument('--source', type=Path, help='Current generated policy, if any')
    parser.add_argument('--memory', type=Path)
    parser.add_argument('--provider-config', type=Path)
    parser.add_argument('--model')
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--split', choices=['development', 'validation', 'final'], required=True,
                        help='Explicit evidence classification; only development is accepted')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.split != 'development':
        parser.error('Validation and final outcomes must not enter rethinking')
    final = json.loads(args.official_result.read_text())
    run = final['run']
    if run['status'] != 'completed':
        parser.error('Completed native development results are required')
    failed = failed_episodes(run['result'])
    if not failed:
        print(json.dumps({'status': 'skipped', 'reason': 'No failed native episode'}))
        return
    model = LLM(provider_config=args.provider_config, model=args.model)
    rethinking = VideoRethinker(model, samples=args.samples)
    feedback = [rethinking.reflect(
        task=run['task'], instruction=args.instruction, run_id=run['run_id'], episode=episode,
        result=run['result'], videos=final['videos'], status=run['status'],
        source=args.source.read_text() if args.source else '',
        memory=json.loads(args.memory.read_text()) if args.memory else {},
        output=args.output / f'episode-{int(episode):03d}') for episode in failed]
    atomic_json(args.output / 'development-feedback.json', {
        'task': run['task'], 'split': 'development', 'failure_rethinking': feedback})
    print(json.dumps({'status': 'completed', 'failed_episodes': len(feedback),
                      'harness_feedback': str((args.output / 'development-feedback.json').resolve())}))


if __name__ == '__main__':
    main()
