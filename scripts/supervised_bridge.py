#!/usr/bin/env python3
"""Opt-in RoboDojo bridge: each VLA/WAM window requires LLM reflection."""
import argparse
import json
import os
from pathlib import Path
import uuid

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal
import uvicorn

from qwenpaw.app.embodiments.robodojo_adapter import (
    AdapterConfigurationError, RoboDojoAdapterConfig, RoboDojoRunManager, create_app,
)
from chunk_supervisor import CAMERAS, ChunkSupervisor


class SupervisedManager(RoboDojoRunManager):
    def __init__(self, config):
        super().__init__(config)
        self.supervisor = ChunkSupervisor()

    def health(self):
        return {**super().health(), 'chunk_supervision': {
            'required': True, 'phase': 'after_execution', 'max_actions': 10,
            'discard_remaining': True, 'reflection_endpoint': '/v1/sessions/reflections',
            'takeover': 'immediate_on_correct', 'cameras': list(CAMERAS)}}

    def current_observation(self):
        with self._lock:
            obs = super().current_observation()
            self.supervisor.bind(obs)
            return {**obs, 'supervision': self.supervisor.public()}

    def current_observation_image(self, topic):
        with self._lock:
            obs = super().current_observation()
            pixels, kind = super().current_observation_image(topic)
            self.supervisor.frame(obs, topic, pixels)
            return pixels, kind

    def request_policy_chunk(self, payload):
        with self._lock:
            obs = super().current_observation()
            self.supervisor.require('policy', obs)
            try:
                chunk = super().request_policy_chunk(payload)
            except AdapterConfigurationError as error:
                latest = super().current_observation()
                failure = latest.get('last_failure') or {}
                if (failure.get('observation_id') == latest['observation_id']
                        and failure.get('code') == 'provider_inference_failed'):
                    self.supervisor.failed('failure-' + uuid.uuid4().hex, error, latest)
                raise
            self.supervisor.ready(chunk)
            return chunk

    def submit_direct_action_chunk(self, payload):
        with self._lock:
            self.supervisor.require('gpt', super().current_observation())
            chunk = super().submit_direct_action_chunk(payload)
            self.supervisor.ready(chunk)
            return chunk


    def execute_action_chunk(self, chunk_id, max_actions=None, discard_remaining=False):
        with self._lock:
            pending = self.supervisor.pending
            if not pending or pending['chunk_id'] != chunk_id:
                raise AdapterConfigurationError('Unknown supervised chunk')
            if type(max_actions) is not int or not 1 <= max_actions <= 10 or discard_remaining is not True:
                raise AdapterConfigurationError('Supervision requires max_actions=1..10, discard_remaining=true')
            result = super().execute_action_chunk(chunk_id, max_actions, discard_remaining)
            # Native Isaac closes its control server at the terminal boundary.
            # Do not lose a valid final receipt by requesting a dead endpoint.
            obs = (dict(run_id=self.supervisor.run_id, observation_id=result['observation_id'])
                   if result.get('episode_end') else super().current_observation())
            self.supervisor.executed(result, obs)
            return {**result, 'reflection_required': not result.get('episode_end'),
                    'reflection_phase': 'post_episode_video' if result.get('episode_end') else 'online',
                    'chunk_id': chunk_id}

    def reflect(self, payload):
        with self._lock:
            value = self.supervisor.review(payload, super().current_observation())
            output = Path(self.get_run(value['run_id']).artifact_dir) / 'chunk-reflections.jsonl'
            with output.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(value, ensure_ascii=False) + '\n')
            return value


class ReflectionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    observation_id: str = Field(min_length=8, max_length=64)
    chunk_id: str = Field(min_length=8, max_length=64)
    verdict: Literal['continue', 'correct', 'uncertain']
    reflection: str = Field(min_length=1, max_length=4000)


def supervised_app(manager):
    app = create_app(manager=manager)

    @app.post('/v1/sessions/reflections')
    def reflect(request: ReflectionRequest):
        try:
            return manager.reflect(request.model_dump())
        except (AdapterConfigurationError, ValueError) as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8098)
    args = parser.parse_args()
    uvicorn.run(supervised_app(SupervisedManager(RoboDojoAdapterConfig.from_env())),
                host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
