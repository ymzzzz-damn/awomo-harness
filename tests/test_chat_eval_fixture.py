"""Case selection never changes model input, benchmark layout or prior scores."""
import importlib.util
import json
from pathlib import Path
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('chat_eval_fixture', ROOT / 'scripts/chat_eval_bridge.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def build(tmp_path, monkeypatch, layout=3):
    monkeypatch.setattr(module.RoboDojoRunManager, '_execution_environment',
                        lambda self, run: {'ORIGINAL_PROVIDER': 'unchanged'})
    files = tmp_path / 'Assets/Eval_Layout/RoboDojo/arx_x5/0'
    files.mkdir(parents=True)
    for index in range(5):
        (files / f'align_blocks_{index}.json').write_text(json.dumps({'private_layout': index}))
    fixture = tmp_path / 'fixture.json'
    fixture.write_text(json.dumps({'task': 'align_blocks', 'seed': 0, 'layout_id': layout}))
    manager = module.FixtureManager.__new__(module.FixtureManager)
    manager.fixture = fixture
    manager.config = types.SimpleNamespace(robodojo_root=tmp_path, env_cfg='arx_x5',
                                           action_type='joint', policy_name='Awomopaw_Harness')
    run = types.SimpleNamespace(task='align_blocks', seed=0, eval_num=1,
                               provider_checkpoint='frozen', run_id='unique-run')
    artifact = tmp_path / 'outputs' / 'unique-run'
    manager._artifact_dir = lambda run: artifact
    return manager, run, artifact


def test_selects_only_requested_layout_without_model_hints(tmp_path, monkeypatch):
    manager, run, artifact = build(tmp_path, monkeypatch)
    assert manager._execution_environment(run) == {'ORIGINAL_PROVIDER': 'unchanged'}
    manifest = json.loads((artifact.parent / '_resume_unique-run.json').read_text())
    assert manifest['abandoned_layout_ids'] == [0, 1, 2, 4]
    assert manifest['details'] == {}
    assert manifest['completed_layout_ids'] == []
    assert json.loads((artifact / 'chat-eval-fixture.json').read_text())['layout_id'] == 3


def test_refuses_missing_layout_and_overwriting_unique_run(tmp_path, monkeypatch):
    manager, run, artifact = build(tmp_path, monkeypatch, layout=6)
    with pytest.raises(ValueError, match='unavailable'):
        manager._execution_environment(run)
    assert not artifact.exists()


def test_prefix_task_does_not_shift_native_layout_selection(tmp_path, monkeypatch):
    manager, run, artifact = build(tmp_path, monkeypatch, layout=3)
    directory = tmp_path / 'Assets/Eval_Layout/RoboDojo/arx_x5/0'
    (directory / 'align_blocks_random_0.json').write_text('{}')
    manager._execution_environment(run)
    selected = json.loads((artifact / 'chat-eval-fixture.json').read_text())
    assert Path(selected['layout_file']).name == 'align_blocks_3.json'
    resume = json.loads((artifact.parent / '_resume_unique-run.json').read_text())
    assert resume['abandoned_layout_ids'] == [0, 1, 2, 4]
