"""Behavioral regression for bounded supervision and outcome-based skill reuse."""
import asyncio
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from gpt6_failure_learning import save, digest
from gpt6_task_learning import (empty_task_memory, learn, load_task_artifact,
    validate_rethinking, validate_task_memory)
from subtask_supervisor import SubtaskSupervisor, CAMERAS
from gpt6_ablation_bridge import GPT6AblationManager
from qwenpaw.app.embodiments.robodojo_adapter import RoboDojoRunManager


def analysis(success):
    return dict(outcome_summary='Observed task outcome',
        observations=[dict(frame_ids=['cam_head:0'], observation='Visible initial objects')],
        hypotheses=[dict(cause='Controlled contact may explain motion', confidence='low')],
        lessons=['Verify tool contact before further motion'], updated_task_memory='Budget return motion; reobserve contact.',
        success_mechanisms=['Reobserving supported a corrective motion'] if success else [],
        policy_improvements=['Check contact at each subtask boundary'],
        skill_text='# Learned task\nObserve objects, plan contact, push, verify and return.' if success else '')


@pytest.fixture
def case_factory(tmp_path):
    source=tmp_path/'source.mp4'
    subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i','color=c=red:s=64x64:r=10',
                    '-t','0.3','-pix_fmt','yuv420p',str(source)],check=True)
    def create(layout, success):
        case=tmp_path/f'case-{layout}';(case/'videos').mkdir(parents=True)
        videos=[]
        for camera in CAMERAS:
            p=case/'videos'/f'{camera}.mp4';p.write_bytes(source.read_bytes())
            videos.append(dict(path=str(p),sha256=digest(p),nb_read_frames='3'))
        save(case/'videos.json',videos)
        row=dict(layout_id=layout,run_id=f'run-{layout}',valid_episode=True,success=success,score=100 if success else 0)
        save(case/'receipt.json',row)
        save(case/'initial-observation.json',dict(run_id=row['run_id'],task='align_blocks',instruction='Exact native instruction'))
        save(case/'official-result.json',dict(task_succeeded=success,run=dict(run_id=row['run_id'],task='align_blocks',status='completed',
            result=dict(eval_time=1,score=row['score'],details={'0':dict(layout_id=layout,success=success)}))))
        (case/'harness-trace.jsonl').write_text(json.dumps(dict(event='execution',action_count=20,hidden_pose='private'))+'\n')
        return case
    return create


def test_failure_success_failure_rethinking_retains_skill_and_feeds_next_task(tmp_path,case_factory):
    calls=[]
    async def model(evidence, before, config, output):
        calls.append((evidence,before))
        assert 'hidden_pose' not in str(evidence['trajectory_samples'])
        assert len(evidence['frames'])==9
        assert all('timestamp_seconds' in f for f in evidence['frames'])
        return analysis(evidence['terminal_result']['success'])
    memory=empty_task_memory('development','align_blocks')
    skill=None
    for layout,success in enumerate([False,True,False]):
        snap=tmp_path/f'snapshot-{layout}.json';save(snap,memory)
        case=case_factory(layout,success)
        memory=asyncio.run(learn(case,snap,tmp_path/'unused-model.json',tmp_path/f'learning-{layout}',model_analysis=model))
        assert memory['version']==layout+1
        assert memory['entries'][-1]['official_success'] is success
        if layout==0:
            assert memory['task_artifact'] is None
        else:
            artifact=load_task_artifact(memory['task_artifact'],'align_blocks')
            assert set(artifact['manifest']['files'])=={'SKILL.md','memory.json'}
            assert artifact['manifest']['provenance']['successful_run_ids']==['run-1']
            if layout==1:
                skill=artifact['skill_text']
                assert artifact['memory']['success_mechanisms'][0]['source_run_id']=='run-1'
            else:
                assert artifact['skill_text']==skill
                assert artifact['manifest']['status']=='failure_feedback_update'
                assert artifact['manifest']['parent_revision']==calls[-1][1]['task_artifact']['revision']
                assert artifact['memory']['source_outcome']=='failure'
    validate_task_memory(memory,'development',3,'align_blocks')
    with pytest.raises(ValueError,match='another task'):
        load_task_artifact(memory['task_artifact'],'general_pickup')
    (artifact['directory']/'SKILL.md').write_text('changed')
    with pytest.raises(ValueError,match='content changed'):
        load_task_artifact(memory['task_artifact'],'align_blocks')


def test_unscored_wrong_task_and_final_outcomes_never_call_model(tmp_path,case_factory):
    case=case_factory(0,True);snap=tmp_path/'snapshot.json';save(snap,empty_task_memory('run','align_blocks'))
    async def forbidden(*args):raise AssertionError('Model must not be called')
    with pytest.raises(ValueError,match='development'):
        asyncio.run(learn(case,snap,None,tmp_path/'final',split='final',model_analysis=forbidden))
    official=json.loads((case/'official-result.json').read_text());official['run']['task']='general_pickup';save(case/'official-result.json',official)
    with pytest.raises(ValueError,match='matching completed'):
        asyncio.run(learn(case,snap,None,tmp_path/'wrong-task',model_analysis=forbidden))
    official['run']['task']='align_blocks';official['run']['status']='failed';save(case/'official-result.json',official)
    with pytest.raises(ValueError,match='matching completed'):
        asyncio.run(learn(case,snap,None,tmp_path/'failed-transport',model_analysis=forbidden))


def test_failure_analysis_cannot_publish_success_mechanisms_or_skill():
    value=analysis(False);validate_rethinking(value,{'cam_head:0'},False)
    value['skill_text']='# Invented success'
    with pytest.raises(ValueError,match='invent'):
        validate_rethinking(value,{'cam_head:0'},False)


def obs(index=0):
    return dict(run_id='same-run',observation_id=f'observation-{index}',episode_end=False,action_count=index)


def test_subtask_gate_reviews_once_after_many_actions_and_rejects_stale_frames():
    gate=SubtaskSupervisor();gate.require(obs());gate.start('code-subtask',obs())
    for i in range(1,41):
        gate.bind(obs(i))
        assert gate.last is None
    gate.completed('code-subtask',dict(status='supervision_boundary',executed_actions=40),obs(40))
    with pytest.raises(ValueError,match='Reflect'):gate.require(obs(40))
    payload=dict(chunk_id='code-subtask',observation_id='observation-40',verdict='correct',reflection='Tool shifted; revise remaining motion')
    for camera in CAMERAS:gate.frame(obs(39),camera,b'old')
    with pytest.raises(ValueError,match='all three'):gate.review(payload,obs(40))
    for camera in CAMERAS:gate.frame(obs(40),camera,b'current')
    gate.review(payload,obs(40));gate.require(obs(40))
    gate.review(dict(payload,verdict='uncertain'),obs(40))
    with pytest.raises(ValueError,match='uncertainty'):gate.require(obs(40))


def test_bridge_synchronizes_worker_return_then_requires_review(tmp_path,monkeypatch):
    manager=object.__new__(GPT6AblationManager)
    manager._lock=threading.RLock();manager.subtask_gate=SubtaskSupervisor()
    manager.subtask_gate.start('code-subtask',obs())
    current=[obs(0)];running=[True]
    monkeypatch.setattr(RoboDojoRunManager,'current_observation',lambda _:current[0])
    monkeypatch.setattr(RoboDojoRunManager,'current_observation_image',lambda _,topic:(b'RGB','image/jpeg'))
    monkeypatch.setattr(RoboDojoRunManager,'code_policy_running',lambda _:running[0])
    monkeypatch.setattr(RoboDojoRunManager,'code_policy_status',lambda _,identifier:dict(status='supervision_boundary',executed_actions=40))
    fixture=tmp_path/'fixture.json';save(fixture,dict(output=str(tmp_path)));manager.fixture=fixture
    current[0]=obs(40)
    assert manager.current_observation()['subtask_supervision']['active_id']=='code-subtask'
    running[0]=False
    assert manager.current_observation()['subtask_supervision']['reflection_required']
    with pytest.raises(ValueError,match='Reflect'):manager._require_subtask_review()
    for camera in CAMERAS:manager.current_observation_image(camera)
    manager.reflect_subtask(dict(chunk_id='code-subtask',observation_id='observation-40',verdict='continue',reflection='Expected contact maintained'))
    manager._require_subtask_review()
    assert len((tmp_path/'subtask-reflections.jsonl').read_text().splitlines())==1


def test_learned_skill_requires_complete_hash_verified_reads(tmp_path,case_factory):
    async def model(evidence,*args):return analysis(True)
    snap=tmp_path/'before.json';save(snap,empty_task_memory('run','align_blocks'))
    memory=asyncio.run(learn(case_factory(0,True),snap,None,tmp_path/'learned',model_analysis=model))
    save(snap,memory)
    fixture=tmp_path/'fixture.json';save(fixture,dict(arm='self_evolving',task='align_blocks',output=str(tmp_path),
        experience_snapshot=str(snap),experience_sha256=digest(snap),complete_experience_view_required=True,
        task_artifact=memory['task_artifact']))
    manager=object.__new__(GPT6AblationManager);manager.fixture=fixture
    save(tmp_path/'experience-reads.json',[dict(path=str(snap),sha256=digest(snap),complete_working_memory=True)])
    with pytest.raises(ValueError,match='complete learned skill'):manager.require_skill_reads()
    directory=Path(memory['task_artifact']['path']).parent
    reads=[dict(path=str(directory/n),sha256=digest(directory/n),complete_file=True)
           for n in ['SKILL.md','memory.json','task-artifact.json']]
    save(tmp_path/'skill-reads.json',reads);manager.require_skill_reads()
    reads[-1]['complete_file']=False;save(tmp_path/'skill-reads.json',reads)
    with pytest.raises(ValueError,match='complete learned skill'):manager.require_skill_reads()


def test_worker_boundary_preserves_memory_and_never_exceeds_segment_budget(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('contract',ROOT/'runtime/awomopaw-code-policy/contract.py')
    contract=importlib.util.module_from_spec(spec);spec.loader.exec_module(contract)
    bridge=ModuleType('bridge');bridge.PrimitiveClient=object;bridge.Robot=object;bridge.Sam3TextSegmenter=object;bridge.plain=lambda v:v
    monkeypatch.setitem(sys.modules,'contract',contract);monkeypatch.setitem(sys.modules,'awomopaw_primitives',bridge)
    # These control-boundary tests do not exercise OpenCV perception.
    monkeypatch.setitem(sys.modules,'cv2',ModuleType('cv2'))
    spec=importlib.util.spec_from_file_location('test_segment_worker',ROOT/'runtime/awomopaw-code-policy/worker.py')
    worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    monkeypatch.setenv('ROBODOJO_SUBTASK_MAX_ACTIONS','40')
    source=tmp_path/'policy.py';source.write_text('def policy(robot, memory):\n    return None\n')
    count=[150]
    def request(path,payload):
        if path=='/v1/sessions/observation':return dict(run_id='run',instruction='Native goal',
            observation_id=str(count[0]),robot_state={'left_arm_joint_state':[0.0]*6},action_count=count[0],action_limit=200)
        if path=='/v1/sessions/direct-action-chunks':return dict(chunk_id='chunk')
        if path=='/v1/sessions/action-chunks/execute':
            count[0]+=1;return dict(executed=1,episode_end=False)
        raise AssertionError(path)
    episode=worker.HttpEpisode(dict(output=str(tmp_path/'actual'),source=str(source),run_id='run',
        instruction='Native goal',initial_state={},fallback_reason='Subtask'),request=request)
    episode.image=lambda:None
    for i in range(40):episode.emit({})
    with pytest.raises(worker.SupervisionBoundary):episode.emit({})
    assert episode.steps==40 and count[0]==190
    count[0]=195
    near_terminal=worker.HttpEpisode(dict(output=str(tmp_path/'near-terminal'),source=str(source),run_id='run',
        instruction='Native goal',initial_state={},fallback_reason='Subtask'),request=request)
    assert near_terminal.maximum_actions==5
    config=tmp_path/'config.json';save(config,dict(output=str(tmp_path/'out'),source=str(tmp_path/'policy.py'),memory={'next_subtask':'push'}))
    (tmp_path/'policy.py').write_text('def policy(robot, memory):\n    memory["completed"] = 40\n    robot.hold(80)\n')
    count=[0]
    class Episode:
        def __init__(self,config):self.steps=0
        def image(self):pass
        def record(self,*args,**kwargs):pass
    class Robot:
        def __init__(self,episode):self.episode=episode
        def hold(self,steps):
            for i in range(steps):
                if self.episode.steps>=40:raise worker.SupervisionBoundary()
                self.episode.steps+=1;count[0]+=1
    monkeypatch.setattr(worker,'HttpEpisode',Episode);monkeypatch.setattr(worker,'HttpRobot',Robot);monkeypatch.setattr(worker,'Backend',lambda:None)
    monkeypatch.setattr(sys,'argv',['worker',str(config)]);worker.main()
    status=json.loads((tmp_path/'out/worker-status.json').read_text())
    assert status['status']=='supervision_boundary' and status['executed_actions']==count[0]==40
    assert status['memory']==dict(next_subtask='push',completed=40)
