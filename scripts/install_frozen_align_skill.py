#!/usr/bin/env python3
"""Install an unchanged pinned PhysicalRSI task skill into the local harness."""
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
SIM = Path('/home/ymz/robodojo-pi05/RoboDojo')
SOURCE = ROOT / 'runtime/physicalrsi-align-skill/adapter'
DEST = SIM / 'XPolicyLab/policy/PhysicalRSI_Align_Skill'


def main():
    with urllib.request.urlopen('http://127.0.0.1:8098/health', timeout=10) as r:
        active = json.load(r).get('active_evaluation')
    if active and active['status'] in {'starting', 'running'}:
        raise RuntimeError('Do not change deployment during an active episode')
    manifest = json.loads((SOURCE / 'frozen-skill.json').read_text())
    descriptor = Path(manifest['descriptor'])
    assert hashlib.sha256(descriptor.read_bytes()).hexdigest() == manifest['descriptor_sha256']
    assets = Path(manifest['asset_root'])
    library = json.loads((assets / 'programs.json').read_text())
    program = next(p for p in library['programs'] if p['name'] == 'align-blocks')
    verified = []
    for template, expected in program['configuration']['files'].items():
        path = Path(template.replace('{root}', str(assets)))
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Pinned release file changed: ' + str(path))
        verified.append({'path': str(path), 'sha256': actual})
    if DEST.exists():
        raise FileExistsError('Refusing to overwrite an existing adapter: ' + str(DEST))
    skill_src = ROOT / 'skills/physicalrsi-align-blocks'
    skill_dst = Path('/home/ymz/.awomopaw/workspaces/car_controller/skills/physicalrsi-align-blocks')
    if skill_dst.exists():
        raise FileExistsError('Refusing to overwrite an existing skill: ' + str(skill_dst))
    shutil.copytree(SOURCE, DEST)
    # Reuse the current common provider launcher; no controller code is changed.
    shutil.copy2(SIM / 'XPolicyLab/policy/Awomopaw_Harness/setup_eval_policy_server.sh',
                 DEST / 'setup_eval_policy_server.sh')
    shutil.copytree(skill_src, skill_dst)
    receipt = {'schema': 'awomopaw.frozen-skill-install/v1', 'manifest': manifest,
               'adapter': str(DEST), 'agent_skill': str(skill_dst),
               'verified_release_files': verified,
               'scope': 'Local simulation only; remote node 153 is unchanged'}
    output = ROOT / 'runtime/physicalrsi-align-skill/install-receipt.json'
    output.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'installed': True, 'adapter': str(DEST), 'skill': str(skill_dst),
                      'verified_files': len(verified)}, indent=2))


if __name__ == '__main__':
    main()
