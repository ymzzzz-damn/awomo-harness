"""Selected models must own control, context creation and post-task learning."""
import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_awomo_open import select_cases, verify_inventory
from harness_models import checked_model, model_matches, bind_experiment_profile


def test_selected_seed_zero_tasks_have_independent_contiguous_memory_order():
    tasks=['align_blocks','general_pickup','solve_equation']
    cases=[dict(task=t,seed=s,layout_id=i,case_index=s*55+i,layout_sha256=f'{t}-{s}-{i}')
           for t in tasks for s in range(3) for i in range(55)]
    selected,filtered=select_cases(tasks,cases,tasks[:2],[0])
    assert selected==tasks[:2] and len(filtered)==110
    assert {c['seed'] for c in filtered}=={0}
    for task in selected:
        assert [c['case_index'] for c in filtered if c['task']==task]==list(range(55))
    verify_inventory(filtered,[c for c in cases if c['task'] in selected and c['seed']==0])
    with pytest.raises(ValueError):select_cases(tasks,cases,['closed_task'],[0])
    with pytest.raises(ValueError):select_cases(tasks,cases,tasks[:2],[3])


@pytest.mark.parametrize('model',['gpt-5.5-2026-04-23','gpt-5.6-sol'])
def test_model_identity_rejects_silent_fallback(model):
    assert checked_model(model)==model
    assert model_matches(model,model)
    assert not model_matches(model,'gpt-6-astra')
    assert not model_matches(model,model+'other')


@pytest.mark.parametrize('model',['gpt-5.5-2026-04-23','gpt-5.6-sol'])
def test_agent_factory_and_vision_resolver_use_process_profile(monkeypatch,model):
    import qwenpaw.config.config as configuration
    from qwenpaw.config.config import AgentProfileConfig
    from qwenpaw.providers import ProviderManager
    from agentscope.model import ChatModelBase
    from qwenpaw.agents.model_factory import create_model_and_formatter
    from qwenpaw.agents.prompt import get_active_model_supports_multimodal
    class FakeModel(ChatModelBase):
        async def __call__(self,*args,**kwargs):
            raise AssertionError('No model calls in factory regression')
    class FakeProvider:
        models=[]
        def __init__(self):self.extra_models=[]
        def get_chat_model_instance(self,identity):return FakeModel(model_name=identity,stream=False)
    provider=FakeProvider()
    manager=SimpleNamespace(get_provider=lambda identity:provider)
    monkeypatch.setattr(ProviderManager,'get_instance',classmethod(lambda cls:manager))
    global_config=AgentProfileConfig(id='default',name='Global',active_model=dict(provider_id='global',model='gpt-6-astra'))
    monkeypatch.setattr(configuration,'load_agent_config',lambda identity:global_config)
    profile=AgentProfileConfig(id='default',name='Experiment',active_model=dict(provider_id='selected',model=model))
    bind_experiment_profile(profile)
    for explicit in ['default',None]:
        runtime,_=create_model_and_formatter(explicit)
        assert runtime.model_name==model
    assert get_active_model_supports_multimodal()
    assert configuration.load_agent_config('other') is global_config
    assert global_config.active_model.model=='gpt-6-astra'


def test_rethinking_rejects_model_mismatch_before_any_request(tmp_path):
    from gpt6_failure_learning import save
    from gpt6_task_learning import analyze
    profile=tmp_path/'config.json'
    save(profile,dict(active_model=dict(provider_id='not-called',model='gpt-5.6-sol')))
    with pytest.raises(ValueError,match='same model'):
        asyncio.run(analyze(dict(controller_model='gpt-5.5-2026-04-23'),{},profile,tmp_path))
