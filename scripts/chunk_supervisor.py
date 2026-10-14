"""Public-evidence reflection gate for the supervised interactive bridge."""
from copy import deepcopy
import hashlib

CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')


class ChunkSupervisor:
    def __init__(self):
        self.run_id = None
        self.pending = None
        self.last = None
        self.reflection = None
        self.frames = {}

    def bind(self, observation):
        if observation['run_id'] != self.run_id:
            self.__init__()
            self.run_id = observation['run_id']

    def ready(self, chunk):
        self.pending = deepcopy(chunk)

    def executed(self, result, observation):
        self.bind(observation)
        if self.pending is None:
            raise ValueError('No supervised chunk is pending')
        self.last = dict(chunk=self.pending, execution=deepcopy(result),
                         observation_id=observation['observation_id'])
        self.pending = None
        self.reflection = None
        self.frames = {}

    def failed(self, failure_id, error, observation):
        self.bind(observation)
        self.last = dict(chunk={'chunk_id': failure_id, 'source': 'policy'},
                         execution={'provider_error': str(error)[:2000]},
                         observation_id=observation['observation_id'])
        self.reflection = None
        self.frames = {}

    def frame(self, observation, camera, pixels):
        self.bind(observation)
        if camera not in CAMERAS:
            raise ValueError('Unsupported public camera')
        self.frames[camera] = dict(observation_id=observation['observation_id'],
                                  sha256=hashlib.sha256(pixels).hexdigest())

    def review(self, payload, observation):
        self.bind(observation)
        if self.pending or not self.last:
            raise ValueError('Reflection requires an executed chunk or provider failure')
        if (payload['observation_id'] != observation['observation_id']
                or self.last['observation_id'] != observation['observation_id']
                or payload['chunk_id'] != self.last['chunk']['chunk_id']):
            raise ValueError('Reflection observation or chunk identity is stale')
        if any(self.frames.get(c, {}).get('observation_id') != observation['observation_id']
               for c in CAMERAS):
            raise ValueError('Read all three current camera frames before reflection')
        if payload['verdict'] not in {'continue', 'correct', 'uncertain'}:
            raise ValueError('Invalid reflection verdict')
        if not isinstance(payload['reflection'], str) or not payload['reflection'].strip():
            raise ValueError('Reflection must explain visible progress, deviation or uncertainty')
        if self.last['execution'].get('provider_error') and payload['verdict'] == 'continue':
            raise ValueError('Provider failure requires correction or uncertainty')
        self.reflection = dict(payload, run_id=self.run_id, frames=deepcopy(self.frames),
                               execution=deepcopy(self.last))
        return deepcopy(self.reflection)

    def require(self, source, observation):
        self.bind(observation)
        if observation.get('episode_end'):
            raise ValueError('Episode has ended')
        if self.pending:
            raise ValueError('Execute the pending chunk before choosing another controller')
        if not self.last:
            if source != 'policy':
                raise ValueError('GPT takeover requires public failure evidence')
            return
        if not self.reflection or self.reflection['observation_id'] != observation['observation_id']:
            raise ValueError('Submit reflection for the last chunk before choosing a controller')
        expected = 'continue' if source == 'policy' else 'correct'
        if self.reflection['verdict'] != expected:
            raise ValueError(f'{source} requires verdict={expected}; inspect or correct the last chunk')

    def public(self):
        return dict(required=self.last is not None and self.reflection is None,
                    chunk_id=self.last['chunk']['chunk_id'] if self.last else None,
                    last_execution=deepcopy(self.last),
                    verdict=self.reflection['verdict'] if self.reflection else None)
