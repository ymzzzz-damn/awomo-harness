"""One visual review per completed subtask/action segment, not per joint action."""
from copy import deepcopy
import hashlib

CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')


class SubtaskSupervisor:
    def __init__(self):
        self.run_id = None
        self.active = None
        self.last = None
        self.reflection = None
        self.frames = {}

    def bind(self, obs):
        if self.run_id != obs['run_id']:
            self.__init__()
            self.run_id = obs['run_id']

    def start(self, identifier, obs):
        self.bind(obs)
        self.active = identifier
        self.last = None
        self.reflection = None
        self.frames = {}

    def completed(self, identifier, receipt, obs):
        self.bind(obs)
        if self.active != identifier:
            raise ValueError('Subtask identity mismatch')
        self.last = dict(chunk_id=identifier, receipt=deepcopy(receipt),
                         observation_id=obs['observation_id'])
        self.active = None
        self.frames = {}
        self.reflection = None

    def frame(self, obs, camera, pixels):
        self.bind(obs)
        if camera not in CAMERAS:
            raise ValueError('Unsupported public camera')
        self.frames[camera] = dict(observation_id=obs['observation_id'],
                                  sha256=hashlib.sha256(pixels).hexdigest())

    def review(self, payload, obs):
        self.bind(obs)
        if self.active or not self.last:
            raise ValueError('Wait until the subtask/action segment completes')
        if (payload['chunk_id'] != self.last['chunk_id'] or
                payload['observation_id'] != obs['observation_id'] or
                self.last['observation_id'] != obs['observation_id']):
            raise ValueError('Stale subtask reflection')
        if any(self.frames.get(c, {}).get('observation_id') != obs['observation_id'] for c in CAMERAS):
            raise ValueError('Read all three current camera frames before subtask reflection')
        if payload['verdict'] not in {'continue', 'correct', 'uncertain'} or not payload['reflection'].strip():
            raise ValueError('Subtask reflection requires a verdict and visible evidence')
        self.reflection = dict(payload, run_id=self.run_id, frames=deepcopy(self.frames),
                               execution=deepcopy(self.last))
        return deepcopy(self.reflection)

    def require(self, obs):
        self.bind(obs)
        if obs.get('episode_end'):
            raise ValueError('Episode has ended')
        if self.active:
            raise ValueError('Wait until the current subtask completes')
        if self.last and (not self.reflection or self.reflection['observation_id'] != obs['observation_id']):
            raise ValueError('Reflect on the completed subtask before the next controller')
        if self.reflection and self.reflection['verdict'] == 'uncertain':
            raise ValueError('Reassess uncertainty before continuing motion')

    def public(self):
        return dict(phase='subtask_or_action_segment', active_id=self.active,
                    chunk_id=self.last['chunk_id'] if self.last else None,
                    reflection_required=bool(self.last and not self.reflection),
                    last_execution=deepcopy(self.last),
                    verdict=self.reflection['verdict'] if self.reflection else None)
