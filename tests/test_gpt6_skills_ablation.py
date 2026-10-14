import importlib.util
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import evaluate_gpt6_skills_seed0 as evaluation
from gpt6_ablation_bridge import GPT6AblationManager


def test_gpt6_does_not_route_to_vla_or_frozen_provider():
    manager=object.__new__(GPT6AblationManager)
    manager.started=False
    for provider in ['pi05','physicalrsi-align-blocks']:
        with pytest.raises(ValueError,match='GPT6 harness'):
            manager.start_attempt(policy_id=provider,execution_mode='interactive')
    with pytest.raises(ValueError,match='inference disabled'):
        manager.request_policy_chunk({})
    path=Path(__file__).resolve().parents[1]/'runtime/gpt6-harness-ablation-adapter/model.py'
    spec=importlib.util.spec_from_file_location('stub',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    model=module.Model({})
    model.reset()
    with pytest.raises(RuntimeError,match='No VLA'):
        model.update_obs({})
    with pytest.raises(RuntimeError,match='GPT6'):
        model.get_action()


def test_with_skills_requires_original_reference_and_memory_before_motion(tmp_path):
    fixture=tmp_path/'fixture.json'
    fixture.write_text(json.dumps({'arm':'with_skills','output':str(tmp_path)}))
    manager=object.__new__(GPT6AblationManager);manager.fixture=fixture
    with pytest.raises(ValueError,match='before first motion'):
        manager.require_skill_reads()
    (tmp_path/'skill-reads.json').write_text(json.dumps([{'path':'/skill/'+p} for p in
        ['SKILL.md','straightedge.py','straightedge.memory.json']]))
    manager.require_skill_reads()
    fixture.write_text(json.dumps({'arm':'without_skills','output':str(tmp_path)}))
    (tmp_path/'skill-reads.json').unlink()
    manager.require_skill_reads()


def test_pair_comparison_keeps_infrastructure_errors_unscored():
    cases=[{'layout_id':0},{'layout_id':1}]
    rows=[dict(layout_id=0,arm='without_skills',valid_episode=True,score=0,success=False),
          dict(layout_id=0,arm='with_skills',valid_episode=True,score=100,success=True),
          dict(layout_id=1,arm='with_skills',valid_episode=False)]
    report=evaluation.aggregate(cases,rows,'running')
    assert report['execution']=='gpt6_direct_harness'
    assert report['complete_pairs']==1
    assert report['paired_mean_score_delta']==100
    assert report['arms']['with_skills']['unscored_or_invalid']==1


def test_bridge_cleanup_does_not_require_removed_fastapi_lifecycle_method(monkeypatch,tmp_path):
    import gpt6_ablation_bridge as bridge
    from qwenpaw.app.embodiments.robodojo_adapter import RoboDojoAdapterConfig,PolicyProvider
    cfg=RoboDojoAdapterConfig(robodojo_root=tmp_path,policy_dir=tmp_path,
        providers=(PolicyProvider('gpt6-harness','code','GPT6_Harness_Ablation','gpt6-live',tmp_path,'test'),))
    monkeypatch.setattr(bridge.RoboDojoAdapterConfig,'from_env',lambda:cfg)
    cleaned=[]
    class Manager:
        def __init__(self,*args):pass
        def stop_code_policies(self):cleaned.append(True)
    monkeypatch.setattr(bridge,'GPT6AblationManager',Manager)
    monkeypatch.setattr(bridge,'create_app',lambda **kwargs:object())
    monkeypatch.setattr(bridge.uvicorn,'run',lambda *args,**kwargs:None)
    monkeypatch.setattr(sys,'argv',['bridge','--fixture',str(tmp_path/'fixture.json'),'--port','33997','--policy-port','33999'])
    bridge.main()
    assert cleaned==[True]


def test_failed_control_server_startup_does_not_enter_endless_episode_loop(monkeypatch):
    import types
    path=Path(__file__).resolve().parents[1]/'runtime/gpt6-harness-ablation-adapter/interactive_harness.py'
    spec=importlib.util.spec_from_file_location('gpt6_transport_test',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    class DeadThread:
        def __init__(self,**kwargs):pass
        def start(self):pass
        def is_alive(self):return False
    import uvicorn
    monkeypatch.setattr(uvicorn,'Config',lambda *args,**kwargs:None)
    monkeypatch.setattr(uvicorn,'Server',lambda *args:types.SimpleNamespace(started=False,run=lambda:None))
    monkeypatch.setattr(module.threading,'Thread',DeadThread)
    harness=object.__new__(module.InteractiveEpisodeHarness)
    harness.host='127.0.0.1';harness.port=24004;harness._create_app=lambda:None
    with pytest.raises(RuntimeError,match='failed to bind/start'):
        harness.start_server()


def test_recovery_never_replays_scored_or_actuated_attempts(tmp_path):
    from recover_gpt6_zero_action_startup import eligible
    log=tmp_path/'native.log';log.write_text("error while attempting to bind on address ('127.0.0.1', 34004): address already in use")
    artifact=tmp_path/'artifact';artifact.mkdir()
    (tmp_path/'run.json').write_text(json.dumps({'log_path':str(log),'artifact_dir':str(artifact)}))
    assert eligible({'valid_episode':False,'actions':0},tmp_path)
    assert not eligible({'valid_episode':True,'actions':0},tmp_path)
    assert not eligible({'valid_episode':False,'actions':1},tmp_path)
    (artifact/'awomopaw_harness_trace.jsonl').write_text(json.dumps({'event':'action_chunk_execution','executed':1})+'\n')
    assert not eligible({'valid_episode':False,'actions':0},tmp_path)
