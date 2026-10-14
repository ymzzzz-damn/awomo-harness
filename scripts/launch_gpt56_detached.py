#!/usr/bin/env python3
"""Keep the finite evaluator in its own tmux session, with durable launch logs."""
import argparse
from pathlib import Path
import subprocess

from run_gpt56_skill_chats import ROOT, PYTHON


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--interrupted-experiment', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    log = output.parent / (output.name + '.launcher.log')
    with log.open('xb') as stream:
        result = subprocess.run([PYTHON, str(ROOT / 'scripts/run_gpt56_skill_chats.py'),
            '--output', str(output), '--interrupted-experiment', str(args.interrupted_experiment.resolve())],
            cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
