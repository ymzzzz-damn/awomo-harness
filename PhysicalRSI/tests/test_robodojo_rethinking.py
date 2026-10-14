"""Temporal evidence, feedback handoff and development-only rethinking gates."""

import copy
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from PhysicalRSI_baselines.robodojo.proposal import CodeProposer
from PhysicalRSI_baselines.robodojo.rethinking import (
    CAMERAS, VideoRethinker, episode_videos, failed_episodes, sample_video,
)
from PhysicalRSI_core.infra.storage import read_json


class Model:
    def __init__(self):
        self.config = SimpleNamespace(public=lambda: {'model': 'test-vision-model'})
        self.output = None
        self.images = ['previous-episode-image']
        self.calls = []
        self.bad_timestamp = False
        self.fail_rethink = False

    def complete(self, messages, tools):
        self.calls.append(copy.deepcopy(messages))
        if len(self.calls) == 1:
            assert self.images == []
            content = messages[-1]['content']
            videos = json.loads(content[0]['text'])['video_evidence']
            assert len([x for x in content if x['type'] == 'image_url']) == 24
            assert all(len(v['frames']) == 8 for v in videos)
            timestamps = [json.loads(x['text'])['timestamp_s'] for x in content[1:] if x['type'] == 'text']
            assert timestamps == sorted(timestamps)
            return json.dumps(dict(
                failure_reason='The target is still displaced after the push.',
                observations=[dict(video_id='video-0', timestamp_s=999 if self.bad_timestamp else
                                   videos[0]['frames'][-1]['timestamp_s'],
                                   evidence='The final visible target remains displaced.')],
                hypotheses=[dict(cause='Contact may have been lost.', confidence='low')],
                uncertainty='Sampled frames cannot establish the exact contact instant.')), [], None
        if len(self.calls) == 2:
            assert (self.output.parent / 'diagnostic-feedback.json').is_file()
            request = json.loads(messages[-1]['content'])
            assert request['harness_feedback']['diagnosis']['failure_reason']
            assert len(request['harness_feedback']['video_evidence']) == 3
            if self.fail_rethink:
                raise TimeoutError('model unavailable')
            return json.dumps(dict(policy_changes=['Reobserve contact before pushing further.'],
                                   memory_delta={'contact': 'Reobserve before continuing.'},
                                   validation_plan=['Compare candidates on independent validation scenes.'])), [], None
        request = json.loads(messages[-1]['content'])
        reflection = request['development_feedback'][0]['failure_rethinking'][0]
        return json.dumps(dict(source='def policy(robot, memory):\n    return None\n',
                               memory=reflection['rethinking']['memory_delta'],
                               rationale=reflection['diagnosis']['failure_reason'])), [], None


@pytest.fixture
def videos(tmp_path):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg and ffprobe required for video integration')
    original = tmp_path / 'original.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                    'testsrc2=size=64x64:rate=10:duration=3', '-c:v', 'mpeg4',
                    '-threads', '1', str(original)], check=True, capture_output=True)
    paths = []
    for camera in CAMERAS:
        path = tmp_path / f'episode_0000000_{camera}_fail.mp4'
        shutil.copyfile(original, path)
        paths.append(str(path))
    return paths


def inputs(tmp_path, videos):
    return dict(task='align_blocks', instruction='Original native goal', run_id='failed-run',
                episode='0', result={'eval_time': 1, 'details': {'0': {'success': False, 'score': 0}}},
                videos=videos, source='def policy(robot, memory):\n    return None\n',
                memory={}, output=tmp_path / 'reflection')


def test_decodes_video_endpoints_and_temporal_interior(tmp_path, videos):
    evidence = sample_video(videos[0], tmp_path / 'frames')
    assert evidence['frame_count'] == 30
    indices = [f['frame_index'] for f in evidence['frames']]
    assert indices[0] == 0 and indices[-1] == 29
    assert len(indices) == 8 and indices == sorted(indices)
    assert evidence['frames'][-1]['timestamp_s'] == pytest.approx(2.9)
    assert all(Path(f['path']).stat().st_size > 0 for f in evidence['frames'])


def test_returns_video_and_cause_before_rethink_and_next_proposal(tmp_path, videos):
    model = Model()
    args = inputs(tmp_path, videos)
    feedback = VideoRethinker(model).reflect(**args)
    assert len(model.calls) == 2
    assert model.images == ['previous-episode-image'] and model.output is None
    assert read_json(args['output'] / 'harness-feedback.json') == feedback
    generation = tmp_path / 'generation'
    generation.mkdir()
    candidate = CodeProposer(model).propose(task=args['task'], source=args['source'], memory={},
        feedback=[dict(task=args['task'], split='development', failure_rethinking=[feedback])],
        skill_api={'task': args['task'], 'methods': {}}, output=generation)
    assert candidate['memory']['contact'] == 'Reobserve before continuing.'
    assert candidate['rationale'] == feedback['diagnosis']['failure_reason']
    # Idempotence: no additional model calls for the same frozen evidence.
    assert VideoRethinker(model).reflect(**args) == feedback
    assert len(model.calls) == 3
    args['memory'] = {'changed': True}
    with pytest.raises(ValueError, match='Frozen'):
        VideoRethinker(model).reflect(**args)


@pytest.mark.parametrize('change', [dict(split='validation'), dict(split='final'),
                                    dict(status='running'), dict(status='failed'),
                                    dict(result={'eval_time': 1, 'details': {'0': {'success': True}}})])
def test_blocks_success_and_non_development_before_io(tmp_path, change):
    model = Model()
    args = inputs(tmp_path, ['/missing/video.mp4'])
    args.update(change)
    with pytest.raises(ValueError):
        VideoRethinker(model).reflect(**args)
    assert not model.calls and not args['output'].exists()


def test_missing_camera_is_not_silently_ignored(tmp_path):
    with pytest.raises(ValueError, match='cam_left_wrist'):
        episode_videos([tmp_path / 'episode_0000000_cam_head_fail.mp4'], 0)


def test_failure_is_native_per_episode_not_aggregate_or_video_name():
    result = dict(eval_time=2, score=0, details={'0': {'success': True}, '1': {'success': False}})
    assert failed_episodes(result) == ['1']
    with pytest.raises(ValueError, match='per-episode'):
        failed_episodes({'score': 0})


def test_invalid_visual_citation_never_reaches_rethink(tmp_path, videos):
    model = Model()
    model.bad_timestamp = True
    args = inputs(tmp_path, videos)
    with pytest.raises(ValueError, match='timestamp'):
        VideoRethinker(model).reflect(**args)
    assert len(model.calls) == 1
    assert (args['output'] / 'error.json').is_file()
    assert not (args['output'] / 'harness-feedback.json').exists()


def test_uncertain_model_call_cannot_be_silently_replayed(tmp_path, videos):
    model = Model()
    model.fail_rethink = True
    args = inputs(tmp_path, videos)
    with pytest.raises(TimeoutError):
        VideoRethinker(model).reflect(**args)
    with pytest.raises(ValueError, match='reconciliation'):
        VideoRethinker(model).reflect(**args)
    assert len(model.calls) == 2
    assert (args['output'] / 'diagnosis.json').exists()


def test_modified_sample_evidence_cannot_be_reused(tmp_path, videos):
    model = Model()
    args = inputs(tmp_path, videos)
    feedback = VideoRethinker(model).reflect(**args)
    Path(feedback['video_evidence'][0]['frames'][0]['path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='evidence changed'):
        VideoRethinker(model).reflect(**args)


def test_success_video_can_distill_reusable_mechanisms_when_explicitly_enabled(tmp_path, videos):
    class SuccessModel(Model):
        def complete(self, messages, tools):
            if not self.calls:
                assert json.loads(messages[-1]['content'][0]['text'])['terminal_outcome'] == 'success'
                answer, calls, usage = super().complete(messages, tools)
                value = json.loads(answer)
                value['outcome_reason'] = value.pop('failure_reason')
                return json.dumps(value), calls, usage
            self.calls.append(copy.deepcopy(messages))
            return json.dumps(dict(policy_changes=['Preserve observed contact checking.'],
                                   memory_delta={'contact': 'Reobserve before continuing.'},
                                   validation_plan=['Test on separate development and validation scenes.'])), [], None
    args = inputs(tmp_path, videos)
    args['result']['details']['0']['success'] = True
    model = SuccessModel()
    result = VideoRethinker(model, include_success=True).reflect(**args)
    assert result['terminal_outcome'] == 'success'
    assert result['diagnosis']['outcome_reason']
    assert len(model.calls) == 2
