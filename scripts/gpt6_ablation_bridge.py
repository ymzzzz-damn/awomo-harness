#!/usr/bin/env python3
"""One native scene with GPT6-authored direct joint actions; VLA inference disabled."""
import argparse
from dataclasses import replace
import json
import hashlib
import os
from pathlib import Path
import sys
import time
import re
from datetime import datetime, timezone
import uvicorn
from fastapi import HTTPException
from subtask_supervisor import SubtaskSupervisor
from supervised_bridge import ReflectionRequest
from chat_eval_bridge import FixtureManager
from qwenpaw.app.embodiments.robodojo_adapter import (
    RoboDojoAdapterConfig, PolicyProvider, AdapterConfigurationError, create_app,
)

class HarnessProvider(PolicyProvider):
    def public(self, root):
        problems=[]
        if not (root/'XPolicyLab/policy'/self.policy_name/'model.py').is_file():
            problems.append('Lifecycle transport stub is missing')
        if not (self.policy_uv_env_path/'.venv/bin/python').is_file():
            problems.append('Transport Python environment is missing')
        return {'provider_id':self.provider_id,'kind':'direct','policy_name':self.policy_name,
            'revision':self.revision,'capabilities':['direct_joint_action'],
            'input_contract':'robodojo.observation/v1','output_contract':'robodojo.action-chunk/v1',
            'lifecycle':['observe','execute','reset','close'],'effects':['simulation-action'],
            'ready':not problems,'problems':problems,'vla_loaded':False,'frozen_controller':False}

class GPT6AblationManager(FixtureManager):
    def __init__(self, config, fixture, policy_port):
        super().__init__(config, fixture)
        self.policy_port=policy_port
        self.started=False
        self.started_at=None
        settings=json.loads(self.fixture.read_text()).get('subtask_supervision')
        self.subtask_gate=SubtaskSupervisor() if settings else None
        self.segment_actions=settings['max_actions'] if settings else None
        if settings:
            if type(self.segment_actions) is not int or not 1 <= self.segment_actions <= 40:
                raise ValueError('Subtask action segment must be 1..40')

    def health(self):
        result=super().health()
        result.update(planning_handoff='GPT6 authors every action directly through the harness',
                      execution_modes=['interactive','gpt6_direct_joint'],vla_used=False)
        if self.subtask_gate:
            result['subtask_supervision']={'required':True,'max_actions':self.segment_actions,
                'phase':'subtask_return_or_action_segment','model_calls_per_joint_action':0}
        return result

    def _sync_subtask(self, obs):
        gate=getattr(self, 'subtask_gate', None)
        if not gate:return
        gate.bind(obs)

    def current_observation(self):
        obs=super().current_observation()
        self._sync_subtask(obs)
        if getattr(self,'subtask_gate',None):
            return {**obs,'subtask_supervision':{
                **self.subtask_gate.public(), 'max_actions': self.segment_actions}}
        return obs

    def current_observation_image(self, topic):
        with self._lock:
            obs=self.current_observation()
            pixels,kind=super().current_observation_image(topic)
            if getattr(self,'subtask_gate',None):self.subtask_gate.frame(obs,topic,pixels)
            return pixels,kind

    def reflect_subtask(self, payload):
        with self._lock:
            value=self.subtask_gate.review(payload,self.current_observation())
            output=Path(json.loads(self.fixture.read_text())['output'])/'subtask-reflections.jsonl'
            with output.open('a') as stream:stream.write(json.dumps(value,ensure_ascii=False)+'\n')
            return value

    def _require_subtask_review(self):
        if getattr(self,'subtask_gate',None):self.subtask_gate.require(self.current_observation())


    def start_attempt(self, **kwargs):
        if self.started:
            raise ValueError('Exactly one episode per case; no silent retries')
        if kwargs.get('policy_id')!='gpt6-harness' or kwargs.get('execution_mode')!='interactive':
            raise ValueError('This ablation accepts interactive GPT6 harness control only')
        result=super().start_attempt(**kwargs)
        self.started=True
        self.started_at=time.monotonic()
        return result

    def _control_request(self, *args, **kwargs):
        result=super()._control_request(*args, **kwargs)
        run=self.current_run()
        if run and result.get('run_id') and result['run_id'] != run.run_id:
            raise AdapterConfigurationError('Control endpoint belongs to a different episode')
        return result

    def session_status(self):
        run=self.current_run()
        if run and run.status not in {'starting','running'}:
            return {'status':run.status,'run_id':run.run_id,'error':run.error,
                'control_ready':False,'episode_end':True}
        try:
            return super().session_status()
        except AdapterConfigurationError:
            if run and self.started_at and time.monotonic()-self.started_at<1200:
                time.sleep(3)
                return {'status':'initializing','run_id':run.run_id,'control_ready':False,
                    'episode_end':False,'startup_elapsed_seconds':round(time.monotonic()-self.started_at),
                    'instruction':'Isaac is initializing. Poll session_status; do not abort or restart.'}
            raise

    def _command(self, run):
        return [sys.executable,str(Path(__file__).with_name('harness_episode_process.py')),
            '--root',str(self.config.robodojo_root),'--policy-dir',str(self.config.policy_dir),
            '--task',run.task,'--checkpoint',run.provider_checkpoint,'--env-cfg',self.config.env_cfg,
            '--action-type','joint','--seed',str(run.seed),'--gpu',str(self.config.env_gpu),
            '--policy-port',str(self.policy_port),'--eval-env',self.config.eval_env]

    def request_policy_chunk(self, payload):
        raise ValueError('VLA/prewritten policy inference disabled in GPT6-only evaluation')

    def require_skill_reads(self):
        fixture=json.loads(self.fixture.read_text())
        if fixture['arm']=='with_skills':
            p=Path(fixture['output'])/'skill-reads.json'
            names={Path(r['path']).name for r in json.loads(p.read_text())} if p.exists() else set()
            if not {'SKILL.md','straightedge.py','straightedge.memory.json'}.issubset(names):
                raise ValueError('Read original task skill and memory before first motion')
        elif fixture['arm']=='self_evolving':
            snapshot=Path(fixture['experience_snapshot'])
            expected=fixture['experience_sha256']
            if hashlib.sha256(snapshot.read_bytes()).hexdigest()!=expected:
                raise ValueError('Learned experience snapshot changed')
            path=Path(fixture['output'])/'experience-reads.json'
            reads=json.loads(path.read_text()) if path.exists() else []
            if not any(r.get('sha256')==expected and Path(r['path']).resolve()==snapshot.resolve() and
                       (not fixture.get('complete_experience_view_required') or r.get('complete_working_memory') is True) for r in reads):
                raise ValueError('Read learned experience before first motion')
            if fixture.get('task_artifact'):
                from gpt6_task_learning import load_task_artifact
                knowledge=json.loads(snapshot.read_text())
                if knowledge.get('task_artifact')!=fixture['task_artifact']:
                    raise ValueError('Skill fixture differs from the frozen experience')
                artifact=load_task_artifact(fixture['task_artifact'],fixture['task'])
                path=Path(fixture['output'])/'skill-reads.json'
                reads=json.loads(path.read_text()) if path.exists() else []
                for name in ['SKILL.md','memory.json','task-artifact.json']:
                    target=(artifact['directory']/name).resolve()
                    if not any(Path(r['path']).resolve()==target and r.get('sha256')==hashlib.sha256(target.read_bytes()).hexdigest()
                               and r.get('complete_file') is True for r in reads):
                        raise ValueError('Read complete learned skill, memory and artifact before first motion')

    def record_experience_application(self, payload, result, kind):
        fixture=json.loads(self.fixture.read_text())
        if fixture['arm']!='self_evolving':return
        memory=json.loads(Path(fixture['experience_snapshot']).read_text())
        allowed={l['id'] for l in memory.get('active_lessons',[])}
        rationale=' '.join(str(payload.get(k,'')) for k in ['rationale','fallback_reason','subtask_goal'])
        cited=set(re.findall(r'E\d{2,}-L\d{2}',rationale))
        record={'at':datetime.now(timezone.utc).isoformat(),'version':memory['version'],
                'kind':kind,'accepted':True,'policy_id':result.get('policy_id'),
                'cited_lesson_ids':sorted(cited & allowed),'unknown_lesson_ids':sorted(cited-allowed),
                'rationale':rationale,'source_sha256':result.get('source_sha256'),
                'interpretation':'Planner claimed applicability in an accepted action plan; this is not proof of causal improvement.'}
        with (Path(fixture['output'])/'experience-application.jsonl').open('a') as stream:
            stream.write(json.dumps(record,ensure_ascii=False)+'\n')

    def submit_direct_action_chunk(self, payload):
        self.require_skill_reads()
        self._require_subtask_review()
        if len(payload.get('actions', [])) > (getattr(self, 'segment_actions', None) or 40):
            raise ValueError('Direct action segment exceeds its configured action cap')
        result=super().submit_direct_action_chunk(payload)
        if getattr(self,'subtask_gate',None):
            self.subtask_gate.start(result['chunk_id'],self.current_observation())
        self.record_experience_application(payload,result,'direct_action')
        return result

    def execute_action_chunk(self, chunk_id, max_actions=None, discard_remaining=False):
        if discard_remaining is not True or type(max_actions) is not int or not 1 <= max_actions <= (getattr(self, 'segment_actions', None) or 40):
            raise ValueError('Execute 1..segment_actions and discard the unexecuted tail')
        result=super().execute_action_chunk(chunk_id,max_actions,discard_remaining)
        if getattr(self,'subtask_gate',None) and not result.get('episode_end'):
            self.subtask_gate.completed(chunk_id,result,self.current_observation())
        return result


    def get_result(self, run_id):
        if self.get_run(run_id).status in {'starting','running'}:
            raise ValueError('Official score is available only after episode completion')
        return super().get_result(run_id)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture',type=Path,required=True)
    parser.add_argument('--port',type=int,required=True)
    parser.add_argument('--policy-port',type=int,required=True)
    args=parser.parse_args()
    config=RoboDojoAdapterConfig.from_env()
    providers=config.providers
    if len(providers)!=1 or providers[0].provider_id!='gpt6-harness' or providers[0].kind not in {'code','direct'}:
        raise ValueError('Registry must contain only GPT6 harness control')
    p=providers[0]
    provider=HarnessProvider(p.provider_id,'direct',p.policy_name,p.checkpoint,p.policy_uv_env_path,p.revision)
    config=replace(config,providers=(provider,))
    manager=GPT6AblationManager(config,args.fixture,args.policy_port)
    app=create_app(manager=manager)
    if getattr(manager,'subtask_gate',None):
        @app.post('/v1/sessions/reflections')
        def reflect(request: ReflectionRequest):
            try:return manager.reflect_subtask(request.model_dump())
            except (ValueError,AdapterConfigurationError) as error:
                raise HTTPException(status_code=409,detail=str(error)) from error
    uvicorn.run(app,host='127.0.0.1',port=args.port)

if __name__=='__main__':main()
