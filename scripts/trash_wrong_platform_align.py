#!/usr/bin/env python3
"""Recoverably remove only this user-cancelled, wrong-platform experiment."""
import json
from pathlib import Path
import subprocess
import urllib.request

ROOT = Path('/home/ymz/Documents/ChatGPT/robodojo')
EXPERIMENT = ROOT / 'eval_result/physicalrsi-align-all-scenes-local-20261013-195300'
SIM = Path('/home/ymz/robodojo-pi05/RoboDojo')


def main():
    with urllib.request.urlopen('http://127.0.0.1:8098/health', timeout=5) as response:
        if json.load(response).get('active_evaluation'):
            raise RuntimeError('Local simulation is still active; refuse deletion')
    process = json.loads((EXPERIMENT / 'process.json').read_text())
    if (Path('/proc') / str(process['pid']) / 'cmdline').exists():
        raise RuntimeError('Experiment driver still exists; refuse deletion')
    targets = [EXPERIMENT, EXPERIMENT.with_name(EXPERIMENT.name + '.launcher.log')]
    runs = []
    for record in sorted((SIM / 'eval_result/harness_runs').glob('*.json')):
        run = json.loads(record.read_text())
        artifact = Path(run.get('artifact_dir', '/nonexistent'))
        fixture = artifact / 'chat-eval-fixture.json'
        if not fixture.is_file() or json.loads(fixture.read_text()).get('test_id') != EXPERIMENT.name:
            continue
        if not artifact.is_relative_to(SIM / 'eval_result/RoboDojo/align_blocks'):
            raise RuntimeError('Invalid native artifact target')
        runs.append(run['run_id'])
        targets.extend([artifact, record, record.with_suffix('.log'),
                        artifact.parent / ('_resume_' + run['run_id'] + '.json'),
                        SIM / 'eval_result/frozen_skill_runs' / run['run_id']])
    targets = [p for p in targets if p.exists()]
    if not runs:
        raise RuntimeError('Could not establish experiment ownership')
    summary = json.loads((EXPERIMENT / 'summary.json').read_text())
    receipt = {'operation': 'user_authorized_gio_trash', 'reason': 'Wrong platform; replace with node 153 evaluation',
               'recoverable': True, 'local_only': True, 'targets': [str(p) for p in targets],
               'run_ids': runs, 'completed_local_trials_not_used_in_remote_score': summary.get('attempted')}
    dest = ROOT / 'runtime/align-all-scenes/local-cleanup-receipt.json'
    dest.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    subprocess.run(['gio', 'trash', *map(str, targets)], check=True)
    print(json.dumps({'trashed_targets': len(targets), 'owned_runs': len(runs), 'recoverable': True}))


if __name__ == '__main__':
    main()
