#!/usr/bin/env python3
"""Check native RGB -> validated hold targets -> official result, without an LLM."""
import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bridge', default='http://127.0.0.1:8098')
    p.add_argument('--task', default='align_blocks')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--startup-timeout', type=int, default=600)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path, payload=None, raw=False):
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(args.bridge.rstrip('/') + path, data=data,
                                     headers={'Content-Type': 'application/json'})
        with opener.open(req, timeout=190) as response:
            return response.read() if raw else json.load(response)

    def save(name, value):
        (args.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

    run = None
    try:
        run = request('/v1/sessions', dict(task=args.task, seed=args.seed, policy_id='gpt6-harness'))
        save('run.json', run)
        deadline = time.monotonic() + args.startup_timeout
        while time.monotonic() < deadline:
            status = request('/v1/sessions/status')
            if status.get('control_ready') or status.get('status') == 'ready': break
            if status.get('episode_end'): raise RuntimeError('Native startup failed: ' + str(status))
            time.sleep(2)
        else: raise TimeoutError('Native simulation did not become ready')
        obs = request('/v1/sessions/observation')
        save('initial-observation.json', obs)
        for camera in ('cam_head', 'cam_left_wrist', 'cam_right_wrist'):
            pixels = request('/v1/sessions/observation/image?topic=' + camera, raw=True)
            if not pixels: raise RuntimeError('Empty native camera: ' + camera)
            (args.output / (camera + '.jpg')).write_bytes(pixels)
        executions = []
        while not obs.get('episode_end'):
            count = min(40, obs['action_limit'] - obs['action_count'])
            if count <= 0: raise RuntimeError('No action budget but episode remains open')
            action = {name: obs['robot_state'][name]
                      for name in obs['action_spec']['required_fields']}
            chunk = request('/v1/sessions/direct-action-chunks', dict(
                observation_id=obs['observation_id'], subtask_goal='Deployment smoke: hold observed joints',
                fallback_reason='Hold test only; no model call or task-success claim', actions=[action]*count))
            result = request('/v1/sessions/action-chunks/execute', dict(
                chunk_id=chunk['chunk_id'], max_actions=count, discard_remaining=True))
            executions.append(result)
            save('executions.json', executions)
            if result.get('episode_end'): break
            obs = request('/v1/sessions/observation')
        deadline = time.monotonic() + 60
        while True:
            try:
                official = request('/v1/evaluations/result?run_id=' + run['run_id'])
                break
            except urllib.error.HTTPError as error:
                if error.code != 409 or time.monotonic() >= deadline: raise
                time.sleep(2)
        save('official-result.json', official)
        report = dict(native_direct_control=True, cameras=3, gpt_model_called=False,
                      executed_actions=sum(e['executed'] for e in executions),
                      task_success_claim=False, run_id=run['run_id'])
        save('report.json', report)
        print(json.dumps(report))
    finally:
        if run:
            try:
                current = request('/v1/sessions/status')
                if current.get('run_id') == run['run_id'] and not current.get('episode_end'):
                    request('/v1/sessions/abort', dict(reason='owned_deployment_smoke_cleanup'))
            except (OSError, RuntimeError): pass


if __name__ == '__main__':
    main()
