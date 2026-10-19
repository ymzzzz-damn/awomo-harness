"""Exercise the deployed scheduler through real video review/distillation code.

Only native task outcomes and model analysis are fixtures. Native execution and
live model connectivity have separate deployment checks.
"""
import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_learning_loop as loop
from gpt6_failure_learning import digest, save
from gpt6_task_learning import learn, load_task_artifact
from test_task_learning_loop import case_factory, analysis


def setup_run(tmp_path):
    output = tmp_path/'run'
    output.mkdir()
    save(output/'agent-config.json', dict(active_model=dict(model='gpt-6-astra')))
    args = SimpleNamespace(output=output, task='align_blocks', gpu=0)
    cases = [dict(task=args.task, seed=0 if i<2 else 1, layout_id=i if i<2 else 0,
                  case_index=i, layout_sha256=f'fixture-{i}') for i in range(3)]
    return args, cases


def fixture_controller(args, case_factory, calls):
    def run(case, arm, gpu, options):
        calls.append(case['case_index'])
        memory = json.loads(options.experience_snapshot.read_text())
        assert memory['version'] == case['case_index']
        if case['case_index'] == 2:
            artifact = load_task_artifact(memory['task_artifact'], 'align_blocks')
            assert artifact['skill_text'].startswith('---')
            assert artifact['manifest']['provenance']['successful_run_ids'] == ['run-1']
        source = case_factory(case['case_index'], case['case_index'] == 1)
        row = json.loads((source/'receipt.json').read_text())
        row.update(case, experience_sha256=digest(options.experience_snapshot), output=str(source))
        save(source/'receipt.json', row)
        official = json.loads((source/'official-result.json').read_text())
        official['run']['result']['details']['0']['layout_id'] = case['layout_id']
        save(source/'official-result.json', official)
        # The scheduler follows the same seed/layout output paths as run_case.
        target = loop.case_directory(args.output, case)
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        row['output'] = str(target)
        save(target/'receipt.json', row)
        return row
    return run


async def review(folder, snapshot, config, output):
    async def model(evidence, before, config, dest):
        assert {f['camera'] for f in evidence['frames']} == {
            'cam_head', 'cam_left_wrist', 'cam_right_wrist'}
        assert all('timestamp_seconds' in f for f in evidence['frames'])
        return analysis(evidence['terminal_result']['success'])
    return await learn(folder, snapshot, config, output, model_analysis=model)


def test_portable_loop_reviews_both_outcomes_and_reuses_success_on_next_seed(tmp_path, case_factory):
    args, cases = setup_run(tmp_path)
    calls = []
    run = fixture_controller(args, case_factory, calls)
    memory = loop.execute(args, cases, run_case=run, review=review)
    assert calls == [0, 1, 2]
    assert memory['version'] == 3
    assert [e['official_success'] for e in memory['entries']] == [False, True, False]
    assert load_task_artifact(memory['task_artifact'], args.task)['memory']['source_outcome'] == 'failure'
    assert json.loads((args.output/'status.json').read_text())['status'] == 'completed'
    # Restart reconstructs the same chain without a model call or task replay.
    async def no_review(*args):
        raise AssertionError('Completed review must not be repeated')
    again = loop.execute(args, cases, run_case=run, review=no_review)
    assert again == memory and calls == [0, 1, 2]


def test_review_failure_resume_never_replays_native_task(tmp_path, case_factory):
    args, cases = setup_run(tmp_path)
    calls = []
    run = fixture_controller(args, case_factory, calls)
    async def unavailable(folder, snapshot, config, output):
        output.mkdir()
        save(output/'error.json', dict(error='model transport interrupted'))
        raise ConnectionError('reflection unavailable')
    with pytest.raises(ConnectionError):
        loop.execute(args, cases, run_case=run, review=unavailable)
    assert calls == [0]
    memory = loop.execute(args, cases, run_case=run, review=review)
    assert memory['version'] == 3 and calls == [0, 1, 2]
    assert (loop.case_directory(args.output, cases[0])/'task-learning/attempt-001/error.json').exists()
    assert (loop.case_directory(args.output, cases[0])/'task-learning/attempt-002/receipt.json').exists()


def test_partial_native_episode_is_not_automatically_replayed(tmp_path):
    args, cases = setup_run(tmp_path)
    loop.case_directory(args.output, cases[0]).mkdir(parents=True)
    def forbidden(*args):
        raise AssertionError('Do not replay an interrupted native episode')
    with pytest.raises(RuntimeError, match='refusing automatic replay'):
        loop.execute(args, cases, run_case=forbidden)


def test_committed_distillation_is_recovered_after_scheduler_interruption(tmp_path, case_factory):
    args, cases = setup_run(tmp_path)
    cases = cases[:1]
    calls = []
    run = fixture_controller(args, case_factory, calls)
    async def interrupted(folder, snapshot, config, output):
        await review(folder, snapshot, config, output)
        raise ConnectionError('scheduler interrupted after distillation committed')
    with pytest.raises(ConnectionError):
        loop.execute(args, cases, run_case=run, review=interrupted)
    async def forbidden(*args):
        raise AssertionError('Committed model analysis must be recovered')
    memory = loop.execute(args, cases, run_case=run, review=forbidden)
    assert memory['version'] == 1 and calls == [0]


def test_inventory_rejects_duplicate_scene_and_preserves_cross_seed_identity(tmp_path):
    for seed in [0, 1]:
        path = tmp_path/f'Assets/Eval_Layout/RoboDojo/arx_x5/{seed}/align_blocks_7.json'
        save(path, dict(fixture=True))
    cases = loop.inventory(tmp_path, 'align_blocks', [0, 1], [7])
    assert [(r['seed'], r['layout_id'], r['case_index']) for r in cases] == [(0, 7, 0), (1, 7, 1)]
    with pytest.raises(ValueError, match='Duplicate'):
        loop.inventory(tmp_path, 'align_blocks', [0, 0], [7])
