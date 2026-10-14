import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import evaluate_pi05_skills_seed0 as ablation


def test_inventory_exact_seed0_excludes_similar_task_and_pins_content(tmp_path):
    directory = tmp_path/'Assets/Eval_Layout/RoboDojo/arx_x5/0'
    directory.mkdir(parents=True)
    for i in range(55):
        (directory/f'align_blocks_{i}.json').write_text(json.dumps({'layout': i}))
    (directory/'align_blocks_random_0.json').write_text('{}')
    cases = ablation.inventory(tmp_path)
    assert [c['layout_id'] for c in cases] == list(range(55))
    assert len({c['layout_sha256'] for c in cases}) == 55
    (directory/'align_blocks_17.json').unlink()
    with pytest.raises(ValueError, match='exactly'):
        ablation.inventory(tmp_path)


def test_pairing_uses_identical_scene_and_never_scores_an_infrastructure_error():
    cases = [{'layout_id': 0}, {'layout_id': 1}]
    rows = [dict(layout_id=0, arm='without_skills', valid_episode=True, score=0, success=False),
            dict(layout_id=0, arm='with_skills', valid_episode=True, score=100, success=True),
            dict(layout_id=1, arm='with_skills', valid_episode=False, status='infrastructure_or_harness_error')]
    summary = ablation.aggregate(cases, rows, 'needs_attention')
    assert summary['complete_pairs'] == 1
    assert summary['paired_mean_score_delta'] == 100
    assert summary['arms']['with_skills']['mean_official_score'] == 100
    assert summary['arms']['with_skills']['unscored_or_invalid'] == 1
    assert not summary['arms']['with_skills']['complete_scene_coverage']


def test_skill_references_are_literal_source_bytes_and_hashes(tmp_path):
    assets = tmp_path/'runtime/physicalrsi-skills/implementations/code-skills'
    impl = assets/'impl'
    impl.mkdir(parents=True)
    shim = impl/'model.py'
    shim.write_text('from .straightedge import Model\n')
    (impl/'straightedge.py').write_text('original_controller = 42\n')
    (impl/'straightedge.memory.json').write_text('{"original_memory":true}\n')
    catalog = tmp_path/'runtime/physicalrsi-all-skills/adapter/catalog.json'
    catalog.parent.mkdir(parents=True)
    catalog.write_text(json.dumps({'releases': {'one': {'task':'align_blocks', 'revision':'original',
        'configuration': {'files': {'{root}/impl/model.py':ablation.digest(shim)}}}}}))
    skill = tmp_path/'planning-skill'
    records = ablation.prepare_skill(tmp_path, skill)
    assert len(records) == 2
    for record in records:
        assert Path(record['source']).read_bytes() == Path(record['copy']).read_bytes()
        assert record['sha256'] == ablation.digest(Path(record['copy']))
    shim.write_text('changed\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        ablation.prepare_skill(tmp_path, tmp_path/'other-skill')


def test_bridge_rejects_other_actions_and_mismatched_execution_windows():
    from pi05_ablation_bridge import Pi05AblationManager
    manager = object.__new__(Pi05AblationManager)
    manager.started = False
    with pytest.raises(ValueError, match='Pi0.5 only'):
        manager.start_attempt(policy_id='physicalrsi-align-blocks', execution_mode='interactive')
    with pytest.raises(ValueError, match='Direct actions'):
        manager.submit_direct_action_chunk({})
    with pytest.raises(ValueError, match='Code policies'):
        manager.start_code_policy({})
    for window, discard in [(None,False),(5,True),(10,False)]:
        with pytest.raises(ValueError, match='max_actions=10'):
            manager.execute_action_chunk('chunk', window, discard)


def test_bridge_reports_initialization_without_false_failure(monkeypatch):
    import pi05_ablation_bridge as bridge
    from types import SimpleNamespace
    from qwenpaw.app.embodiments.robodojo_adapter import AdapterConfigurationError
    manager = object.__new__(bridge.Pi05AblationManager)
    manager.started_at = bridge.time.monotonic()
    manager.current_run = lambda: SimpleNamespace(status='running',run_id='test')
    monkeypatch.setattr(bridge.time,'sleep',lambda _:None)
    def unavailable(self):
        raise AdapterConfigurationError('connection refused')
    monkeypatch.setattr(bridge.FixtureManager,'session_status',unavailable)
    assert manager.session_status()['status'] == 'initializing'
    manager.started_at -= 1201
    with pytest.raises(AdapterConfigurationError):
        manager.session_status()


def test_missing_public_goal_is_a_transport_error_not_a_task_failure():
    for obs in [{}, {'instruction':''}, {'instruction':None}]:
        with pytest.raises(ValueError,match='transport error'):
            ablation.require_public_instruction(obs)
    assert ablation.require_public_instruction({'instruction':'exact official instruction'}) == 'exact official instruction'
