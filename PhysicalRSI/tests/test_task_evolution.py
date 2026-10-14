import json
from types import SimpleNamespace

import pytest

from PhysicalRSI_baselines.robodojo.task_evolution import evolve_task, verify_artifact
from PhysicalRSI_baselines.robodojo import task_evolution
from PhysicalRSI_core.infra.storage import digest


def parent():
    content = dict(skill_text='Original skill', source='', memory={'keep': 'Successful mechanism'})
    return dict(schema='awomopaw.task-artifact/v1', task='align_blocks',
                revision=digest(content), content=content)


class Model:
    def complete(self, messages, tools):
        request = json.loads(messages[-1]['content'])
        assert len(request['feedback']['video_feedback']) == 2
        return json.dumps(dict(skill_text='Preserve useful progress and reobserve contact.', source='',
                               memory={'contact': 'Check contact'}, rationale='Evidence from both outcomes',
                               validation_plan=['Compare with parent on separate scenes.'])), [], None


def test_distills_success_and_failure_and_binds_immutable_parent(tmp_path, monkeypatch):
    episodes = []
    class Reflector:
        def __init__(self, model, **kwargs):
            assert kwargs['include_success'] is True
        def reflect(self, **kwargs):
            episodes.append(kwargs['episode'])
            return {'episode': kwargs['episode']}
    monkeypatch.setattr(task_evolution, 'VideoRethinker', Reflector)
    final = dict(run=dict(status='completed', task='align_blocks', run_id='r1', result=dict(
        eval_time=2, details={'0': {'success': True}, '1': {'success': False}})), videos=[])
    previous = parent()
    result = evolve_task(model=Model(), final=final, instruction='Original instruction',
                         parent=previous, output=tmp_path / 'candidate')
    assert episodes == ['0', '1']
    assert result['parent_revision'] == previous['revision']
    assert result['status'] == 'candidate'
    assert verify_artifact(result, 'align_blocks')['memory']['contact'] == 'Check contact'
    assert previous['content']['skill_text'] == 'Original skill'
    with pytest.raises(FileExistsError):
        evolve_task(model=Model(), final=final, instruction='Original instruction',
                    parent=previous, output=tmp_path / 'candidate')


@pytest.mark.parametrize('split,status', [('validation', 'completed'), ('final', 'completed'),
                                         ('development', 'failed'), ('development', 'running')])
def test_inappropriate_outcomes_never_reach_the_model_or_videos(tmp_path, split, status):
    with pytest.raises(ValueError, match='completed development'):
        evolve_task(model=None, final={'run': {'status': status}}, instruction='Goal', parent=None,
                    output=tmp_path / 'candidate', split=split)
    assert not (tmp_path / 'candidate').exists()


def test_artifact_cannot_be_used_for_another_task_or_with_mutated_content():
    original = parent()
    with pytest.raises(ValueError, match='identity'):
        verify_artifact(original, 'general_pickup')
    original['content']['skill_text'] = 'changed'
    with pytest.raises(ValueError, match='identity'):
        verify_artifact(original, 'align_blocks')


def test_wrong_run_reflections_are_rejected_before_io(tmp_path):
    final = dict(run=dict(status='completed', task='align_blocks', run_id='r1', result=dict(
        eval_time=1, details={'0': {'success': True}})), videos=[])
    with pytest.raises(ValueError, match='this completed run'):
        evolve_task(model=None, final=final, instruction='Native goal', parent=None,
                    output=tmp_path / 'candidate', reflections=[{'run_id': 'another-run'}])
    assert not (tmp_path / 'candidate').exists()


def test_policy_artifact_requires_the_executable_signature():
    artifact = parent()
    artifact['content']['source'] = 'def wrong(robot):\n    return None\n'
    artifact['revision'] = digest(artifact['content'])
    with pytest.raises(ValueError, match='define policy'):
        verify_artifact(artifact, 'align_blocks')
