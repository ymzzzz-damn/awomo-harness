import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('all_align', ROOT / 'scripts/evaluate_all_align_scenes.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_inventory_covers_each_seed_and_layout_without_pose_hints(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'SIM', tmp_path)
    base = tmp_path / 'Assets/Eval_Layout/RoboDojo/arx_x5'
    for seed in [0, 1, 2]:
        directory = base / str(seed)
        directory.mkdir(parents=True)
        for layout in [0, 1, 2]:
            (directory / f'align_blocks_{layout}.json').write_text('{"unexposed_private_scene": true}')
        (directory / 'another_task_0.json').write_text('{}')
    cases = module.inventory()
    assert [(c['seed'], c['layout_id']) for c in cases] == [(s, l) for s in range(3) for l in range(3)]
    assert all(set(c) == {'seed', 'layout_id', 'file', 'sha256'} for c in cases)


def test_inventory_refuses_missing_scene_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'SIM', tmp_path)
    directory = tmp_path / 'Assets/Eval_Layout/RoboDojo/arx_x5/0'
    directory.mkdir(parents=True)
    (directory / 'align_blocks_1.json').write_text('{}')
    with pytest.raises(RuntimeError, match='dense'):
        module.inventory()


def test_does_not_treat_policy_failure_as_unscored_instability(tmp_path):
    row = {'valid_episode': True, 'success': False, 'actions': 200}
    assert module.native_unstable_before_actions(row, tmp_path, 0, 0, 0) is False
