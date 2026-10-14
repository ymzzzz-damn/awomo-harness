import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_full_frozen_skills as module


def test_exact_task_inventory_covers_all_seeds_without_prefix_collision(tmp_path):
    for seed in range(3):
        directory=tmp_path/'Assets/Eval_Layout/RoboDojo/arx_x5'/str(seed)
        directory.mkdir(parents=True)
        for task in ['push_T','press_by_number']:
            for index in range(2):
                (directory/f'{task}_{index}.json').write_text('{}')
        (directory/'push_T_random_0.json').write_text('{"private":true}')
    cases=module.inventory(tmp_path,['push_T','press_by_number'])
    assert len(cases)==12
    assert len({(r['task'],r['seed'],r['layout']) for r in cases})==12
    assert all('random' not in r['layout_file'] for r in cases)
    assert [r['task'] for r in cases[:4]]==['push_T','press_by_number']*2


def test_inventory_rejects_missing_native_ids(tmp_path):
    directory=tmp_path/'Assets/Eval_Layout/RoboDojo/arx_x5/0'
    directory.mkdir(parents=True)
    (directory/'push_T_1.json').write_text('{}')
    with pytest.raises(ValueError,match='Noncontiguous'):
        module.inventory(tmp_path,['push_T'])


def test_scores_preserve_failures_and_distinguish_task_from_scene_average():
    cases=[{'index':0,'task':'a'},{'index':1,'task':'a'},{'index':2,'task':'b'}]
    results=[dict(cases[0],valid_episode=True,score=100,success=True),
             dict(cases[1],valid_episode=True,score=0,success=False),
             dict(cases[2],valid_episode=True,score=20,success=False)]
    summary=module.aggregate(cases,results,results,'completed')
    assert summary['mean_score']==40
    assert summary['macro_task_mean_score']==35
    assert summary['successes']==1
    assert summary['success_rate']==pytest.approx(1/3)
    assert summary['complete_scene_coverage']


def test_infrastructure_errors_are_explicit_and_make_coverage_incomplete():
    cases=[{'index':0,'task':'a'},{'index':1,'task':'a'}]
    results=[dict(cases[0],valid_episode=True,score=20,success=False),
             dict(cases[1],valid_episode=False,status='infrastructure_error')]
    summary=module.aggregate(cases,results,results,'completed_with_infrastructure_errors')
    assert summary['mean_score']==20
    assert summary['valid_episodes']==1
    assert summary['infrastructure_errors']==1
    assert summary['requested']==2
    assert not summary['complete_scene_coverage']


def test_every_scene_and_retry_has_distinct_ports():
    ports=[port+i for attempt in range(1,4) for index in range(690)
           for port in [module.ports_for_case(16000,index,690,attempt)] for i in range(3)]
    assert len(set(ports))==6210
    assert max(ports)<65536
