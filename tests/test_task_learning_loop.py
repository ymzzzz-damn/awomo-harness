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


def test_learning_crosses_seed_boundary_without_relabeling_native_layout(tmp_path,case_factory):
    memory=empty_task_memory('open','align_blocks')
    async def model(evidence,before,config,output):
        return analysis(False)
    for order,seed,layout in [(54,0,54),(55,1,0)]:
        case=case_factory(layout,False)
        row=json.loads((case/'receipt.json').read_text())
        row.update(task='align_blocks',seed=seed,case_index=order)
        save(case/'receipt.json',row)
        snap=tmp_path/f'open-snapshot-{order}.json';save(snap,memory)
        memory=asyncio.run(learn(case,snap,None,tmp_path/f'open-learning-{order}',model_analysis=model))
    validate_task_memory(memory,'open',56,'align_blocks')
    assert [e['source_layout_id'] for e in memory['entries']]==[54,0]
    assert [e['source_case_index'] for e in memory['entries']]==[54,55]
    assert [e['source_seed'] for e in memory['entries']]==[0,1]
    assert memory['active_lessons'][-1]['id']=='E55-L01'


def test_failure_analysis_cannot_publish_success_mechanisms_or_skill():
    value=analysis(False);validate_rethinking(value,{'cam_head:0'},False)
    value['skill_text']='# Invented success'
    with pytest.raises(ValueError,match='invent'):
        validate_rethinking(value,{'cam_head:0'},False)


def test_official_zero_motion_outcome_still_gets_three_camera_review(tmp_path, case_factory):
    case = case_factory(0, False)
    (case/'harness-trace.jsonl').write_text(json.dumps(dict(event='terminal', action_count=0))+'\n')
    snap = tmp_path/'snapshot.json'
    save(snap, empty_task_memory('zero-motion', 'align_blocks'))
    async def model(evidence, *args):
        assert len(evidence['frames']) == 9
        return analysis(False)
    memory = asyncio.run(learn(case, snap, None, tmp_path/'review', model_analysis=model))
    assert memory['version'] == 1 and memory['task_artifact'] is None


def test_video_analyzer_sends_all_cameras_and_checks_returned_model(tmp_path, case_factory, monkeypatch):
    from qwenpaw.providers import ProviderManager
    import gpt6_task_learning as learning
    calls = []
    class Response:
        model = 'gpt-6-astra'
        id = 'test-response'
        output_text = json.dumps(analysis(True))
        def model_dump(self, **kwargs):
            return dict(model=self.model, id=self.id, output_text=self.output_text)
    class Client:
        def __init__(self):
            self.responses = self
        def with_options(self, **kwargs):
            return self
        async def create(self, **kwargs):
            calls.append(kwargs)
            return Response()
        async def close(self):
            pass
    provider = SimpleNamespace(_client=lambda **kwargs: Client())
    monkeypatch.setattr(ProviderManager, 'get_instance', lambda: SimpleNamespace(get_provider=lambda _: provider))
    config = tmp_path/'agent.json'
    save(config, dict(active_model=dict(model='gpt-6-astra', provider_id='test')))
    snapshot = tmp_path/'snapshot.json'
    save(snapshot, empty_task_memory('api-contract', 'align_blocks'))
    case = case_factory(0, True)
    memory = asyncio.run(learn(case, snapshot, config, tmp_path/'learned-api'))
    assert memory['task_artifact']
    request = calls[0]
    images = [c for c in request['input'][0]['content'] if c['type'] == 'input_image']
    assert len(images) == 9 and all(c['image_url'].startswith('data:image/jpeg;base64,') for c in images)
    assert request['text']['format']['strict'] is True
    Response.model = 'another-model'
    with pytest.raises(RuntimeError, match='model mismatch'):
        asyncio.run(learn(case, snapshot, config, tmp_path/'wrong-model'))


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


@pytest.mark.parametrize('released', [True, False])
def test_bridge_execution_error_reconciles_only_released_native_chunks(monkeypatch, released):
    from qwenpaw.app.embodiments.robodojo_adapter import AdapterConfigurationError
    manager = object.__new__(GPT6AblationManager)
    manager.subtask_gate = SubtaskSupervisor()
    manager.segment_actions = 40
    gate = manager.subtask_gate
    gate.start('chunk-one', obs())
    current = dict(obs(1), pending_chunk_id=None if released else 'chunk-one')
    manager.current_observation = lambda: current
    def failed(*args, **kwargs):
        raise AdapterConfigurationError('native chunk expired or transport interrupted')
    monkeypatch.setattr(RoboDojoRunManager, 'execute_action_chunk', failed)
    with pytest.raises(AdapterConfigurationError):
        manager.execute_action_chunk('chunk-one', 1, True)
    if released:
        assert gate.active is None and gate.last['receipt']['chunk_released']
        with pytest.raises(ValueError, match='Reflect'):
            gate.require(current)
        for camera in CAMERAS:
            gate.frame(current, camera, b'current')
        gate.review(dict(chunk_id='chunk-one', observation_id=current['observation_id'],
                         verdict='correct', reflection='The released chunk failed; replan from current RGB.'), current)
        gate.require(current)
    else:
        assert gate.active == 'chunk-one' and gate.last is None




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


