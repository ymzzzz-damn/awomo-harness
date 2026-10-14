#!/usr/bin/env python3
"""Isolated embodiment bridge selecting one native layout and owned policy port."""
from __future__ import annotations
import argparse
import json
import hashlib
from dataclasses import asdict, replace
from pathlib import Path
import sys
import uvicorn
from chat_eval_bridge import FixtureManager
from qwenpaw.app.embodiments.robodojo_adapter import RoboDojoAdapterConfig, PolicyProvider, create_app


class CatalogProvider(PolicyProvider):
    def public(self, robodojo_root):
        result = super().public(robodojo_root)
        if self.policy_name != 'PhysicalRSI_Frozen_Skills':
            return result
        # The generic adapter stores one explicit release per checkpoint in its
        # catalog. Retain all base installation checks except the legacy single
        # frozen-skill.json lookup, and verify each pinned file in this release.
        legacy_checkpoint = str(robodojo_root / 'XPolicyLab/policy' / self.policy_name / 'checkpoints')
        problems = [p for p in result['problems'] if not p.startswith('frozen skill is unavailable:')
                    and not p.startswith('checkpoint is missing: ' + legacy_checkpoint)]
        try:
            catalog_path = robodojo_root / 'XPolicyLab/policy' / self.policy_name / 'catalog.json'
            catalog = json.loads(catalog_path.read_text())
            if catalog['schema'] != 'awomopaw.physicalrsi-frozen-catalog/v1':
                raise ValueError('invalid frozen catalog schema')
            entry = catalog['releases'][self.checkpoint]
            if entry['kind'] != self.kind or self.provider_id != 'physicalrsi-' + entry['name']:
                raise ValueError('catalog release does not match registered provider')
            prefix = 'frozen-code:' if self.kind == 'code' else 'frozen-source:'
            if self.revision != prefix + entry['revision']:
                raise ValueError('catalog revision differs from registered revision')
            if self.kind == 'code':
                for template, expected in entry['configuration']['files'].items():
                    path = Path(template.replace('{root}', catalog['asset_root']))
                    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                        raise ValueError('Frozen release hash mismatch: ' + str(path))
            elif self.kind == 'vla':
                implementation = Path(catalog['framework_source']).joinpath(*entry['implementation'].split('.')).with_suffix('.py')
                if hashlib.sha256(implementation.read_bytes()).hexdigest() != entry['revision']:
                    raise ValueError('Frozen VLA source hash mismatch: ' + str(implementation))
                checkpoint = Path(entry['configuration']['model_path'])
                if not checkpoint.is_dir():
                    raise ValueError('Registered VLA checkpoint root is missing: ' + str(checkpoint))
                step = str(entry['configuration']['checkpoint_num'])
                if not any((path / 'params').is_dir() for path in [checkpoint, checkpoint / step,
                    *[child for child in checkpoint.iterdir() if child.is_dir() and child.name.isdigit()
                      and int(child.name) == int(step)]]):
                    raise ValueError('Registered VLA checkpoint params are missing: ' + str(checkpoint))
                result['readiness_scope'] = 'Static source hash and checkpoint params existence; weights not loaded'
            else:
                raise ValueError('Unsupported generic frozen provider kind')
            result['capabilities'] = ['observation_to_action_chunk', entry['action_type'] + '_action']
        except (OSError, ValueError, KeyError, TypeError) as error:
            problems.append('Frozen catalog unavailable: ' + str(error))
        result['problems'] = problems
        result['ready'] = not problems
        return result


class ParallelAdapterConfig(RoboDojoAdapterConfig):
    @classmethod
    def from_env(cls):
        config = super().from_env()
        return replace(config, providers=tuple(CatalogProvider(**asdict(provider)) for provider in config.providers))

    def validate(self):
        problems = super().validate()
        if self.action_type == 'ee':
            problems = [p for p in problems if p != 'This harness deployment only permits validated joint actions']
        return problems


class ParallelFixtureManager(FixtureManager):
    def __init__(self, config, fixture, policy_port):
        super().__init__(config, fixture)
        self.policy_port = policy_port

    def health(self):
        health = super().health()
        health['frozen_execution_action_type'] = self.config.action_type
        health['execution_qualification'] = 'Direct frozen provider, no Chat or LLM planning'
        return health

    def _command(self, run):
        return [sys.executable, str(Path(__file__).with_name('frozen_skill_episode_process.py')),
            '--root', str(self.config.robodojo_root), '--policy-dir', str(self.config.policy_dir),
            '--task', run.task, '--checkpoint', run.provider_checkpoint,
            '--env-cfg', self.config.env_cfg, '--action-type', self.config.action_type, '--seed', str(run.seed),
            '--gpu', str(self.config.env_gpu), '--policy-port', str(self.policy_port),
            '--eval-env', self.config.eval_env]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--policy-port', type=int, required=True)
    args = parser.parse_args()
    manager = ParallelFixtureManager(ParallelAdapterConfig.from_env(), args.fixture, args.policy_port)
    uvicorn.run(create_app(manager=manager), host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
