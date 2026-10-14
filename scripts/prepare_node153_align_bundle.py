#!/usr/bin/env python3
"""Build the private finite runner bundle, without model keys or skill edits."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / 'runtime/align-all-scenes/node153-20261013-213700'


def main():
    bundle = STAGE / 'bundle'
    bundle.mkdir(parents=True, exist_ok=False)
    scripts = bundle / 'scripts'
    scripts.mkdir()
    for name in ['evaluate_align_node153.py', 'run_full_frozen_skills.py',
                 'run_parallel_frozen_skills.py', 'frozen_skill_parallel_bridge.py',
                 'chat_eval_bridge.py', 'frozen_skill_episode_process.py',
                 'verify_full_frozen_results.py']:
        shutil.copy2(ROOT / 'scripts' / name, scripts / name)
    catalog = json.loads((ROOT / 'runtime/physicalrsi-all-skills/adapter/catalog.json').read_text())
    entry = next(e for e in catalog['releases'].values() if e.get('task') == 'align_blocks')
    expected = entry['configuration']['files']
    original = json.loads((ROOT / 'runtime/physicalrsi-align-skill/install-receipt.json').read_text())
    original_hashes = {r['sha256'] for r in original['verified_release_files']}
    if set(expected.values()) != original_hashes or len(expected) != 22:
        raise ValueError('Generic provider must use the exact original 22-file frozen align release')
    (bundle / 'expected-release.json').write_text(json.dumps(expected, indent=2) + '\n')
    archive = STAGE / 'align-node153-20261013-213700.tar.gz'
    with tarfile.open(archive, 'w:gz') as stream:
        for path in sorted(bundle.rglob('*')):
            if path.is_file():
                stream.add(path, arcname=str(path.relative_to(bundle)))
    receipt = {'archive': str(archive), 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
               'size': archive.stat().st_size, 'original_skill_files': 22, 'credentials_included': False}
    (STAGE / 'bundle-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
