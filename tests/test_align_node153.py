import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import evaluate_align_node153 as module


def test_remote_results_preserve_scored_failures():
    cases = [{'index': i, 'task': 'align_blocks', 'seed': s} for i, s in enumerate([0, 0, 1])]
    rows = [dict(cases[0], valid_episode=True, score=100, success=True),
            dict(cases[1], valid_episode=True, score=0, success=False)]
    result = module.summary(cases, rows, 'running')
    assert result['requested'] == 3
    assert result['success_rate'] == .5
    assert result['per_seed']['0']['successes'] == 1
    assert not result['complete']
    assert result['platform'] == 'P6000 / node 192.168.48.153'


def test_infrastructure_failure_never_counts_as_scored_failure():
    cases = [{'index': 0, 'task': 'align_blocks', 'seed': 0}]
    rows = [dict(cases[0], valid_episode=False, status='infrastructure_error')]
    result = module.summary(cases, rows, 'failed')
    assert result['valid_episodes'] == 0
    assert result['success_rate'] is None
    assert result['infrastructure_errors'] == 1
    assert not result['complete']


def test_native_pre_action_rejection_is_separate():
    cases = [{'index': 0, 'task': 'align_blocks', 'seed': 0}]
    rows = [dict(cases[0], valid_episode=False, unscored_native_scene=True)]
    result = module.summary(cases, rows, 'completed')
    assert result['infrastructure_errors'] == 0
    assert result['unscored_native_scenes'] == 1
    assert result['success_rate'] is None
