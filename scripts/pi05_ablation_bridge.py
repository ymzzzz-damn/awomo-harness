#!/usr/bin/env python3
"""One native scene, Pi0.5 actions only, fixed ten-action execution windows."""
import argparse
import json
from pathlib import Path
import sys
import time
import uvicorn
from chat_eval_bridge import FixtureManager
from qwenpaw.app.embodiments.robodojo_adapter import RoboDojoAdapterConfig, AdapterConfigurationError, create_app


class Pi05AblationManager(FixtureManager):
    def __init__(self, config, fixture, policy_port):
        super().__init__(config, fixture)
        self.policy_port = policy_port
        self.started = False
        self.started_at = None

    def start_attempt(self, **kwargs):
        if self.started:
            raise ValueError('Exactly one episode per case; no silent retries')
        if kwargs.get('policy_id') != 'pi05' or kwargs.get('execution_mode') != 'interactive':
            raise ValueError('This ablation accepts interactive Pi0.5 only')
        result = super().start_attempt(**kwargs)
        self.started = True
        self.started_at = time.monotonic()
        return result

    def session_status(self):
        run = self.current_run()
        if run and run.status not in {'starting', 'running'}:
            return {'status': run.status, 'run_id': run.run_id, 'error': run.error,
                    'control_ready': False, 'episode_end': True}
        try:
            return super().session_status()
        except AdapterConfigurationError:
            # The policy server starts before Isaac; refusal is expected during
            # startup. Keep lifecycle state explicit rather than exposing it as
            # a device failure to the planner.
            if run and self.started_at and time.monotonic()-self.started_at < 1200:
                time.sleep(3)
                return {'status': 'initializing', 'run_id': run.run_id,
                    'control_ready': False, 'episode_end': False,
                    'startup_elapsed_seconds': round(time.monotonic()-self.started_at),
                    'instruction': 'Pi0.5/Isaac is initializing. Poll session_status; do not abort or start another episode.'}
            raise

    def _command(self, run):
        return [sys.executable, str(Path(__file__).with_name('frozen_skill_episode_process.py')),
            '--root', str(self.config.robodojo_root), '--policy-dir', str(self.config.policy_dir),
            '--task', run.task, '--checkpoint', run.provider_checkpoint,
            '--env-cfg', self.config.env_cfg, '--action-type', 'joint', '--seed', str(run.seed),
            '--gpu', str(self.config.env_gpu), '--policy-port', str(self.policy_port),
            '--eval-env', self.config.eval_env]

    def submit_direct_action_chunk(self, payload):
        raise ValueError('Direct actions disabled in Pi0.5 skills ablation')

    def start_code_policy(self, payload):
        raise ValueError('Code policies disabled in Pi0.5 skills ablation')

    def request_policy_chunk(self, payload):
        fixture = json.loads(self.fixture.read_text())
        if fixture.get('arm') == 'with_skills':
            path = Path(fixture['output']) / 'skill-reads.json'
            reads = json.loads(path.read_text()) if path.exists() else []
            names = {Path(r['path']).name for r in reads}
            if not {'SKILL.md', 'straightedge.py'}.issubset(names):
                raise ValueError('Read the registered SKILL.md and straightedge.py reference before Pi0.5 inference')
        return super().request_policy_chunk(payload)

    def execute_action_chunk(self, chunk_id, max_actions=None, discard_remaining=False):
        if max_actions != 10 or discard_remaining is not True:
            raise ValueError('Both arms require max_actions=10, discard_remaining=true')
        return super().execute_action_chunk(chunk_id, max_actions, discard_remaining)

    def get_result(self, run_id):
        if self.get_run(run_id).status in {'starting', 'running'}:
            raise ValueError('Official score is available only after episode completion')
        return super().get_result(run_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--policy-port', type=int, required=True)
    args = parser.parse_args()
    config = RoboDojoAdapterConfig.from_env()
    if len(config.providers) != 1 or config.providers[0].provider_id != 'pi05':
        raise ValueError('Registry must contain only Pi0.5')
    manager = Pi05AblationManager(config, args.fixture, args.policy_port)
    uvicorn.run(create_app(manager=manager), host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
