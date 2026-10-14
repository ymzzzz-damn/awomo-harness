#!/usr/bin/env python3
"""Test fixture bridge: select a native layout without changing Chat's prompt.

Uses RoboDojo's existing abandoned-layout resume mechanism, as its official
layout-range launcher does. Provider routing, action validation and scoring
remain unchanged. The test fixture contains IDs, not object poses or hints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import uvicorn

from qwenpaw.app.embodiments.robodojo_adapter import (
    RoboDojoAdapterConfig, RoboDojoRunManager, create_app,
)


class FixtureManager(RoboDojoRunManager):
    def __init__(self, config, fixture):
        super().__init__(config)
        self.fixture = fixture

    def _execution_environment(self, run):
        env = super()._execution_environment(run)
        if not self.fixture.exists():
            return env
        fixture = json.loads(self.fixture.read_text())
        if fixture['task'] != run.task or fixture['seed'] != run.seed:
            raise ValueError('Agent selected a task/seed outside the registered test case')
        if run.eval_num != 1:
            raise ValueError('Chat test fixture requires one episode per request')
        layout = int(fixture['layout_id'])
        directory = self.config.robodojo_root / 'Assets/Eval_Layout/RoboDojo' / self.config.env_cfg / str(run.seed)
        pattern = re.compile(rf'{re.escape(run.task)}_\d+\.json')
        files = sorted((p for p in directory.iterdir() if pattern.fullmatch(p.name)),
                       key=lambda p: int(p.stem.rsplit('_', 1)[-1]))
        if not 0 <= layout < len(files):
            raise ValueError('Requested native layout is unavailable')
        artifact = self._artifact_dir(run)
        artifact.mkdir(parents=True, exist_ok=False)
        additional = f'ckpt_name={run.provider_checkpoint},action_type={self.config.action_type}'
        manifest = {
            'run_id': run.run_id, 'save_dir': str(artifact), 'task_name': run.task,
            'policy_name': self.config.policy_name, 'config_name': self.config.env_cfg,
            'eval_seed': run.seed, 'additional_info': additional,
            'success_nums': 0, 'fail_nums': 0, 'unstable_nums': 0, 'total_score': 0.0,
            'completed_layout_ids': [],
            'abandoned_layout_ids': [i for i in range(len(files)) if i != layout],
            'details': {}, 'restart_count': 0,
        }
        (artifact.parent / f'_resume_{run.run_id}.json').write_text(json.dumps(manifest, indent=2) + '\n')
        selected = dict(fixture, layout_file=str(files[layout]),
                        layout_sha256=hashlib.sha256(files[layout].read_bytes()).hexdigest(),
                        scope='Native layout selection only; not passed to the model')
        (artifact / 'chat-eval-fixture.json').write_text(json.dumps(selected, indent=2) + '\n')
        return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    args = parser.parse_args()
    manager = FixtureManager(RoboDojoAdapterConfig.from_env(), args.fixture)
    uvicorn.run(create_app(manager=manager), host='127.0.0.1', port=8098)


if __name__ == '__main__':
    main()
