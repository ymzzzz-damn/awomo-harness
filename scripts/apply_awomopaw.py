#!/usr/bin/env python3
"""Apply our patch and added files to the pinned external base checkout."""
import argparse
from pathlib import Path
import shutil
import subprocess

BASE = 'e6e0dcd85b11212d3bf945a724adc295baa03902'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkout', type=Path)
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    root = Path(__file__).resolve().parents[1]
    def git(*parts):
        return subprocess.check_output(['git', '-C', str(checkout), *parts], text=True).strip()
    if git('rev-parse', 'HEAD') != BASE:
        raise SystemExit('Expected external base commit ' + BASE)
    if git('status', '--porcelain'):
        raise SystemExit('Use a clean external checkout; preserve existing changes first')
    additions = [p for p in (root/'source').rglob('*') if p.is_file()]
    for path in additions:
        target = checkout/path.relative_to(root/'source')
        if target.exists():
            raise SystemExit('Added file would overwrite an existing path: ' + str(target))
    patch = str(root/'patches/awomopaw.patch')
    subprocess.run(['git', '-C', str(checkout), 'apply', '--check', patch], check=True)
    subprocess.run(['git', '-C', str(checkout), 'apply', patch], check=True)
    for path in additions:
        target = checkout/path.relative_to(root/'source')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    print('AwomoPaw changes applied to ' + str(checkout))

if __name__ == '__main__':
    main()
