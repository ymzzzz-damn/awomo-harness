import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from gpt6_failure_learning import empty_memory, validate_memory, validate_analysis, failure_evidence, digest
from gpt6_ablation_bridge import GPT6AblationManager


def test_memory_rejects_other_experiments_future_scenes_and_duplicates():
    memory = empty_memory('new-run')
    validate_memory(memory, 'new-run', 0)
    memory.update(version=1, entries=[{'source_layout_id':0}])
    validate_memory(memory, 'new-run', 1)
    with pytest.raises(ValueError, match='Current/future'):
        validate_memory(memory, 'new-run', 0)
    with pytest.raises(ValueError, match='exclusively'):
        validate_memory(memory, 'old-run', 1)
    memory.update(version=2, entries=[{'source_layout_id':0}, {'source_layout_id':0}])
    with pytest.raises(ValueError, match='duplicate'):
        validate_memory(memory, 'new-run', 1)


def test_success_and_unscored_episodes_cannot_create_failure_experience(tmp_path):
    for row in [{'valid_episode':True,'success':True}, {'valid_episode':False,'success':False}]:
        (tmp_path/'receipt.json').write_text(json.dumps(row))
        with pytest.raises(ValueError, match='scored failure'):
            failure_evidence(tmp_path)


def test_reflection_requires_real_frame_citations_and_bounded_memory():
    result={'failure_summary':'Unaligned row',
            'observations':[{'frame_ids':['cam_head:199'], 'observation':'Block remains apart'}],
            'hypotheses':[{'cause':'Contact may be lost','confidence':'low'}],
            'lessons':['Reobserve contact before further motion'], 'updated_task_memory':'Check contact'}
    validate_analysis(result, {'cam_head:199'})
    with pytest.raises(ValueError, match='nonexistent'):
        validate_analysis(result, {'cam_head:0'})
    result['updated_task_memory']='x'*10001
    with pytest.raises(ValueError, match='budget'):
        validate_analysis(result, {'cam_head:199'})


def test_first_motion_requires_exact_frozen_experience_read(tmp_path):
    snapshot=tmp_path/'before.json';snapshot.write_text(json.dumps(empty_memory('run')))
    fixture=tmp_path/'fixture.json';fixture.write_text(json.dumps({
        'arm':'self_evolving','output':str(tmp_path),'experience_snapshot':str(snapshot),
        'experience_sha256':digest(snapshot)}))
    manager=object.__new__(GPT6AblationManager);manager.fixture=fixture
    with pytest.raises(ValueError, match='before first motion'):manager.require_skill_reads()
    (tmp_path/'experience-reads.json').write_text(json.dumps([{'path':str(snapshot),'sha256':digest(snapshot)}]))
    manager.require_skill_reads()
    snapshot.write_text('{}')
    with pytest.raises(ValueError, match='snapshot changed'):manager.require_skill_reads()


def test_failure_frames_and_trace_keep_public_evidence_only(tmp_path):
    import subprocess
    (tmp_path/'videos').mkdir()
    source=tmp_path/'videos/source.mp4'
    subprocess.run(['ffmpeg','-nostdin','-v','error','-f','lavfi','-i','color=c=red:s=64x64:r=10',
                    '-t','1','-pix_fmt','yuv420p',str(source)],check=True)
    videos=[]
    for camera in ['cam_head','cam_left_wrist','cam_right_wrist']:
        path=tmp_path/'videos'/f'{camera}.mp4';path.write_bytes(source.read_bytes())
        videos.append({'path':str(path),'sha256':digest(path),'nb_read_frames':'10'})
    (tmp_path/'videos.json').write_text(json.dumps(videos))
    (tmp_path/'receipt.json').write_text(json.dumps({'valid_episode':True,'success':False,'score':0,
                                                   'layout_id':1,'run_id':'failed-new-run'}))
    (tmp_path/'initial-observation.json').write_text(json.dumps({'instruction':'public instruction'}))
    (tmp_path/'harness-trace.jsonl').write_text(json.dumps({'event':'observation','action_count':1,
                                                          'hidden_object_pose':[1,2,3]})+'\n')
    for i in range(9):
        folder=tmp_path/'code-policies'/f'worker-{i:02d}';folder.mkdir(parents=True)
        (folder/'receipt.json').write_text(json.dumps({'policy_id':f'worker-{i}', 'status':'returned', 'started_at':i,
                                                     'memory':{'public_rgb_estimate':[i,0,0]}}))
        (folder/'policy.py').write_text(f'def policy(robot, memory):\n    return {i}\n')
    evidence=failure_evidence(tmp_path)
    assert len(evidence['workers'])==9
    assert 'return 0' in evidence['workers'][0]['submitted_code']
    assert 'return 8' in evidence['workers'][-1]['submitted_code']
    assert sum(len(w['submitted_code']) for w in evidence['workers'])<=28000
    assert sum(len(w['public_execution_feedback']) for w in evidence['workers'])<=8000
    assert len(evidence['frames'])==15
    assert all(Path(f['path']).is_file() for f in evidence['frames'])
    assert all('hidden_object_pose' not in e for e in evidence['trajectory_samples'])
    assert evidence['official_instruction']=='public instruction'


def test_ui_preserves_baselines_and_shows_evolving_inventory(tmp_path):
    from trajectory_ui import EvidenceStore
    baseline=tmp_path/'gpt6-skills-baseline';baseline.mkdir()
    evolving=tmp_path/'gpt6-self-evolution-new';evolving.mkdir()
    cases=[{'layout_id':i} for i in range(55)]
    for path,arms in [(baseline,['with_skills','without_skills']), (evolving,['self_evolving'])]:
        (path/'manifest.json').write_text(json.dumps({'task':'align_blocks','seed':0,'cases':cases,
                                                     'arms':arms,'provider':'gpt6-harness'}))
    scenes=EvidenceStore(tmp_path).scan()['scenes']
    assert len(scenes)==165
    assert len([s for s in scenes if s['arm']=='self_evolving'])==55
    assert all(s['status']=='pending' for s in scenes)


def test_trace_only_observations_do_not_invent_video_citations():
    from gpt6_failure_learning import normalize_visual_observations
    value = {'failure_summary':'No motion',
             'observations':[{'frame_ids':['cam_head:0'],'observation':'Initial blocks'},
                             {'frame_ids':[],'observation':'Trace records zero actions'}],
             'hypotheses':[], 'lessons':['Check interface'], 'updated_task_memory':'Check interface'}
    result, omitted = normalize_visual_observations(value)
    validate_analysis(result, {'cam_head:0'})
    assert result['updated_task_memory'] == value['updated_task_memory']
    assert len(value['observations']) == 2 and len(result['observations']) == 1
    assert omitted == [value['observations'][1]]
    result['observations'][0] = {'frame_ids':['cam_head:99'],'observation':'Invalid citation'}
    with pytest.raises(ValueError, match='nonexistent'):
        validate_analysis(result, {'cam_head:0'})


def test_sparse_video_sampling_uses_only_real_unique_frames():
    from gpt6_failure_learning import select_indices
    assert select_indices(1) == [0]
    assert select_indices(3) == [0, 1, 2]
    assert len(select_indices(50)) == 5


def test_learning_filters_deployment_faults_but_not_ik_or_model_claims(tmp_path):
    from gpt6_failure_learning import learning_eligibility
    (tmp_path/'receipt.json').write_text(json.dumps({'valid_episode':True,'success':False}))
    (tmp_path/'harness-trace.jsonl').write_text(json.dumps({'action_count':20})+'\n')
    (tmp_path/'agent-messages.jsonl').write_text(json.dumps({'message':{'role':'assistant','content':[{'type':'text','text':'Maybe contract.py is missing: No such file or directory'}]}})+'\n')
    assert learning_eligibility(tmp_path)['eligible']
    (tmp_path/'agent-messages.jsonl').write_text(json.dumps({'message':{'role':'system','content':[{'type':'tool_result','output':'RuntimeError: generic IK could not reach requested Cartesian pose'}]}})+'\n')
    assert learning_eligibility(tmp_path)['eligible']
    (tmp_path/'agent-messages.jsonl').write_text(json.dumps({'message':{'role':'system','content':[{'type':'tool_result','output':"No such file or directory: /runtime/contract.py"}]}})+'\n')
    assert learning_eligibility(tmp_path)['infrastructure_error']
    assert not learning_eligibility(tmp_path)['eligible']
    (tmp_path/'agent-messages.jsonl').write_text('')
    (tmp_path/'harness-trace.jsonl').write_text(json.dumps({'action_count':0})+'\n')
    assert not learning_eligibility(tmp_path)['eligible']


def test_working_memory_remains_complete_when_provenance_grows():
    from gpt6_failure_learning import working_memory_view
    m=empty_memory('clean')
    m.update(version=55,entries=[{'source_layout_id':i,'diagnostic_metadata':'x'*1000} for i in range(55)],
             task_memory='useful current memory '+'x'*9900,
             active_lessons=[{'id':'E54-L01','source_layout_id':54,'text':'verify actual contact'}])
    view=json.loads(working_memory_view(m))
    assert view['task_memory']==m['task_memory']
    assert view['source_layouts']==list(range(55))
    assert len(working_memory_view(m))<14000
    assert 'diagnostic_metadata' not in working_memory_view(m)


def test_application_audit_is_a_claim_without_inventing_success(tmp_path):
    memory=empty_memory('new');memory.update(version=1,entries=[{'source_layout_id':0}],
        active_lessons=[{'id':'E00-L01','source_layout_id':0,'text':'check contact'}])
    snap=tmp_path/'snap.json';snap.write_text(json.dumps(memory))
    fixture=tmp_path/'fixture.json';fixture.write_text(json.dumps({'arm':'self_evolving','experience_snapshot':str(snap),'output':str(tmp_path)}))
    m=object.__new__(GPT6AblationManager);m.fixture=fixture
    m.record_experience_application({'fallback_reason':'Apply E00-L01 after checking current wrist view'},
        {'policy_id':'code-generated','source_sha256':'model-authored-sha'},'generated_code')
    record=json.loads((tmp_path/'experience-application.jsonl').read_text())
    assert record['cited_lesson_ids']==['E00-L01']
    assert record['unknown_lesson_ids']==[]
    assert 'not proof' in record['interpretation']
