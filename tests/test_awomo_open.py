"""Open-suite scene identity, task isolation, and paired score denominators."""
import asyncio
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_awomo_open import aggregate, comparison, verify_inventory
from gpt6_failure_learning import save, validate_memory, working_memory_view
from gpt6_task_learning import empty_task_memory, learn, validate_task_memory
from trajectory_ui import EvidenceStore


def test_task_starts_native_control_with_episode_phase(tmp_path, monkeypatch):
    import evaluate_awomo_open as runner
    from types import SimpleNamespace
    task = 'align_blocks'
    output = tmp_path/'batch'
    case = dict(task=task, seed=0, layout_id=0, case_index=0, layout_sha256='official')
    observed = []
    monkeypatch.setattr(runner.transport, 'gpu_memory', lambda gpu: {'used_mib': 0})
    def control(case, arm, args_gpu, args):
        status = json.loads((output/'tasks'/task/'status.json').read_text())
        observed.append((status['status'], status['phase'], status['current_case']))
        return dict(case, valid_episode=True, success=False, score=0,
                    experience_sha256=runner.digest(args.experience_snapshot))
    monkeypatch.setattr(runner.control, 'run_case', control)
    monkeypatch.setattr(runner, 'learning_eligibility', lambda *a, **k: dict(
        eligible=False, infrastructure_error=False, reason='no learning artifact'))
    runner.task_run(task, [case], 0, SimpleNamespace(output=output), 50000)
    assert observed == [('running', 'episode', case)]
    assert json.loads((output/'tasks'/task/'results.json').read_text())[0]['valid_episode']


def test_predecessor_failure_or_partial_coverage_does_not_release_queue(tmp_path):
    from evaluate_awomo_open import predecessor_completed
    save(tmp_path/'scheduler-finished.json', dict(status='needs_attention'))
    save(tmp_path/'summary.json', dict(status='needs_attention', requested=1320,
        finished=0, valid_episodes=0, complete_scene_coverage=False))
    assert not predecessor_completed(tmp_path)
    save(tmp_path/'scheduler-finished.json', dict(status='completed'))
    save(tmp_path/'summary.json', dict(status='completed', requested=1320,
        finished=1, valid_episodes=1, complete_scene_coverage=False))
    assert not predecessor_completed(tmp_path)
    save(tmp_path/'summary.json', dict(status='completed', requested=1320,
        finished=1320, valid_episodes=1320, complete_scene_coverage=True))
    assert predecessor_completed(tmp_path)


def test_paired_comparison_never_merges_same_layout_across_task_or_seed():
    cases=[dict(task=t,seed=s,layout_id=0,layout_sha256=f'{t}-{s}')
           for t in ['align_blocks','general_pickup'] for s in [0,1]]
    verify_inventory(cases,list(reversed(cases)))
    with pytest.raises(ValueError):verify_inventory(cases,cases[:-1])
    with pytest.raises(ValueError):verify_inventory(cases,cases+[cases[0]])
    ours=[dict(cases[0],valid_episode=True,success=True,score=100),
          dict(cases[1],valid_episode=False,success=False,score=0)]
    baseline=[dict(cases[0],valid_episode=True,success=False,score=0),
              dict(cases[2],valid_episode=True,success=True,score=100)]
    report=comparison(['align_blocks','general_pickup'],ours,baseline)
    assert report['complete_pairs']==1
    assert report['tasks'][0]['paired_success_rate_delta']==1
    assert report['tasks'][1]['complete_pairs']==0
    summary=aggregate(['align_blocks','general_pickup'],cases,ours,{},'running')
    assert summary['finished']==1 and summary['success_rate']==1
    assert not summary['complete_scene_coverage']


def test_multi_seed_order_rejects_future_and_cross_task_experience():
    memory=empty_task_memory('open:pickup','general_pickup')
    memory.update(version=2,entries=[dict(source_layout_id=54,source_case_index=54,source_seed=0,
        source_task='general_pickup',official_success=False),dict(source_layout_id=0,source_case_index=55,
        source_seed=1,source_task='general_pickup',official_success=False)],active_lessons=[dict(
        id='E55-L01',source_layout_id=0,source_case_index=55,source_seed=1,source_task='general_pickup',text='Reobserve')])
    validate_task_memory(memory,'open:pickup',56,'general_pickup')
    with pytest.raises(ValueError,match='future'):validate_task_memory(memory,'open:pickup',55,'general_pickup')
    memory['entries'][1]['source_task']='align_blocks'
    with pytest.raises(ValueError,match='another task'):validate_task_memory(memory,'open:pickup',56,'general_pickup')


def test_full_task_working_view_remains_bounded():
    memory=empty_task_memory('open:pickup','general_pickup')
    memory.update(version=165,task_memory='x'*10000,entries=[dict(source_layout_id=i%55,source_case_index=i,
        source_seed=i//55,source_task='general_pickup',official_success=False) for i in range(165)])
    validate_memory(memory,'open:pickup',165)
    view=json.loads(working_memory_view(memory))
    assert len(view['source_case_orders'])==165
    assert len(view['recent_source_scenes'])==8


def test_ui_keeps_all_task_seed_layout_identities(tmp_path):
    batch=tmp_path/'awomo-open'
    save(batch/'experiment.json',dict(schema='awomopaw.open-online-harness/v1'))
    cases=[dict(task=t,seed=s,layout_id=0,case_index=s*55) for t in ['align_blocks','general_pickup'] for s in [0,1]]
    save(batch/'official-case-inventory.json',cases)
    folder=batch/'tasks/general_pickup/seed-1/layout-00/self_evolving'
    save(folder/'receipt.json',dict(cases[3],arm='self_evolving',status='completed',valid_episode=True,
        run_id='native',success=True,score=100))
    scenes=EvidenceStore(tmp_path).scan()['scenes']
    assert len(scenes)==4
    assert len({s['id'] for s in scenes})==4
    assert sum(s['status']=='success' for s in scenes)==1
    assert {(s['task'],s['seed']) for s in scenes}=={(c['task'],c['seed']) for c in cases}
    save(batch/'tasks/general_pickup/learning/snapshots/case-055.json',dict(version=3,entries=[],task_memory='Task-specific memory'))
    store=EvidenceStore(tmp_path)
    completed=next(s for s in store.scan()['scenes'] if s['status']=='success')
    detail=store.detail(completed['id'])
    assert detail['learning']['version']==3
    assert detail['learning']['kind']=='task-artifacts'
    assert detail['learning']['task_memory']=='Task-specific memory'


def test_lesson_citations_work_for_scene_orders_above_99(tmp_path):
    from gpt6_ablation_bridge import GPT6AblationManager
    memory=tmp_path/'experience.json'
    save(memory,dict(version=1,active_lessons=[dict(id='E100-L01')]))
    fixture=tmp_path/'fixture.json'
    save(fixture,dict(arm='self_evolving',experience_snapshot=str(memory),output=str(tmp_path)))
    manager=object.__new__(GPT6AblationManager);manager.fixture=fixture
    manager.record_experience_application(dict(rationale='Use E100-L01'),{},'generated_code')
    result=json.loads((tmp_path/'experience-application.jsonl').read_text())
    assert result['cited_lesson_ids']==['E100-L01']
    assert result['unknown_lesson_ids']==[]
