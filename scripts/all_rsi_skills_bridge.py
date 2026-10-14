#!/usr/bin/env python3
"""Persistent catalog-aware RSI bridge using the existing embodiment API.

Switch each idle episode to its explicit frozen provider's original action
contract. Never changes the contract while an episode is active. No fixture or
private layout override is used in this ordinary bridge.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import uvicorn
from frozen_skill_parallel_bridge import ParallelAdapterConfig, ParallelFixtureManager
from qwenpaw.app.embodiments.robodojo_adapter import (
    RoboDojoRunManager, EvaluationConflictError, create_app,
)


class AllRSISkillsManager(RoboDojoRunManager):
    def __init__(self, config, policy_port):
        super().__init__(config)
        self.policy_port = policy_port

    def _command(self, run):
        return ParallelFixtureManager._command(self, run)

    def _execution_environment(self, run):
        environment = super()._execution_environment(run)
        environment['ROBODOJO_ACTION_TYPE'] = self.config.action_type
        return environment

    def start_attempt(self, **request):
        with self._lock:
            if any(run.status in {'starting', 'running'} for run in self._runs.values()):
                raise EvaluationConflictError('A RoboDojo evaluation is already using the configured GPU')
            provider = self.config.get_provider(request['policy_id'])
            action_type = 'joint'
            if provider.policy_name == 'PhysicalRSI_Frozen_Skills':
                catalog_path = self.config.robodojo_root / 'XPolicyLab/policy' / provider.policy_name / 'catalog.json'
                catalog = json.loads(catalog_path.read_text())
                action_type = catalog['releases'][provider.checkpoint]['action_type']
            if action_type not in {'joint', 'ee'}:
                raise ValueError('Frozen release has an unsupported action contract')
            previous = self.config
            self.config = replace(self.config, action_type=action_type)
            try:
                return super().start_attempt(**request)
            except BaseException:
                self.config = previous
                raise

    def health(self):
        with self._lock:
            health = super().health()
            health['adapter'] = 'awomopaw-physicalrsi-frozen-catalog/v1'
            health['action_contract_selection'] = 'Explicit registered provider; frozen joint or ee actions per idle episode'
            health['readiness_scope'] = 'Code source hashes validated; VLA source hash and checkpoint paths validated without loading weights'
            return health


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8098)
    parser.add_argument('--policy-port', type=int, default=8210)
    args = parser.parse_args()
    manager = AllRSISkillsManager(ParallelAdapterConfig.from_env(), args.policy_port)
    uvicorn.run(create_app(manager=manager), host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
