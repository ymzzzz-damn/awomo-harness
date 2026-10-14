#!/usr/bin/env python3
"""Finite detached launcher for the complete align_blocks scene inventory."""
import argparse
from pathlib import Path
import subprocess

from evaluate_all_align_scenes import ROOT, PYTHON


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    log = output.parent / (output.name + '.launcher.log')
    with log.open('xb') as stream:
        result = subprocess.run([PYTHON, str(ROOT / 'scripts/evaluate_all_align_scenes.py'),
            '--output', str(output)], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
