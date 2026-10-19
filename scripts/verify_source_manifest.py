#!/usr/bin/env python3
"""Verify every published file's length and SHA-256 without runtime dependencies."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'SOURCE_MANIFEST.json').read_text())
    errors = []
    for entry in manifest['files']:
        path = (root / entry['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            errors.append(entry['path'] + ': missing or invalid path')
            continue
        content = path.read_bytes()
        if len(content) != entry['bytes'] or hashlib.sha256(content).hexdigest() != entry['sha256']:
            errors.append(entry['path'] + ': content differs')
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f"Verified {len(manifest['files'])} published files")


if __name__ == '__main__':
    main()
