"""Generic primitive execution bridge for model-written policies.

Only RGB, static camera/robot calibration and observed robot state enter policy
execution. No shipped task policy, hidden geometry, reward or layout is imported.
"""
import asyncio
import builtins
import copy
import json
import os
from pathlib import Path
import queue
import threading
import traceback

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from generic_tools.robot import Robot
from generic_tools.sam3_segment_text import Sam3TextSegmenter


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


class Episode:
    def __init__(self, model, observation, index):
        self.model = model
        self.obs = observation
        self.root = model.output / f'episode-{index:03d}'
        self.root.mkdir(parents=True)
        self.actions = queue.Queue()
        self.condition = threading.Condition()
        self.version = 0
        self.cancelled = False
        self.complete = False
        self.error = None
        self.steps = 0
        self.initial = copy.deepcopy(observation['state'])
        self.target = {k: np.asarray(v, dtype=float).copy() for k, v in self.initial.items()
                       if k.endswith('_arm_joint_state') or k.endswith('_ee_joint_state')}
        self.record('initial', state=self.initial,
                    instruction=observation.get('instruction'))
        self.save_frame()
        self.thread = threading.Thread(target=self.execute, daemon=True)
        self.thread.start()

    def record(self, event, **values):
        with (self.root / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(plain(dict(event=event, step=self.steps, **values))) + '\n')

    def save_frame(self):
        Image.fromarray(np.asarray(self.obs['vision']['cam_head']['color'], dtype=np.uint8)).save(
            self.root / f'head-{self.steps:03d}.png')

    def update(self, observation):
        with self.condition:
            self.obs = observation
            self.version += 1
            if self.steps % 40 == 0:
                self.save_frame()
            if self.steps % 20 == 0:
                self.record('observed_state', state=observation['state'])
            self.condition.notify_all()

    def emit(self, action):
        with self.condition:
            if self.cancelled:
                raise RuntimeError('episode reset')
            if self.steps >= 175:
                raise RuntimeError('generated policy primitive action budget exhausted')
            version = self.version
            self.steps += 1
            self.actions.put(plain(action))
            if not self.condition.wait_for(lambda: self.version > version or self.cancelled, timeout=120):
                raise TimeoutError('simulator did not acknowledge action')
            if self.cancelled:
                raise RuntimeError('episode reset')

    def execute(self):
        try:
            from PhysicalRSI_baselines.robodojo.skill_proposal import validate_policy_source
            source = self.model.source.read_text()
            validate_policy_source(source)
            allowed_builtins = {k: getattr(builtins, k) for k in (
                'abs', 'min', 'max', 'sum', 'len', 'range', 'enumerate', 'zip',
                'sorted', 'list', 'dict', 'tuple', 'float', 'int', 'str', 'bool',
                'round', 'all', 'any', 'Exception', 'ValueError')}
            namespace = {'__builtins__': allowed_builtins}
            exec(compile(source, str(self.model.source), 'exec'), namespace)
            memory = json.loads(self.model.memory.read_text())
            namespace['policy'](PrimitiveClient(self), memory)
            self.record('policy_returned', memory=memory)
        except BaseException as error:
            self.error = f'{type(error).__name__}: {error}'
            self.record('policy_error', error=self.error, traceback=traceback.format_exc())
        finally:
            self.save_frame()
            self.complete = True
            self.actions.put(None)


class PrimitiveClient:
    """General perception and movement; task sequences are model-written."""
    def __init__(self, episode):
        self.e = episode

    def reobserve(self):
        return plain({'state': self.e.obs['state'], 'initial_state': self.e.initial})

    def scene(self):
        e = self.e
        image = np.asarray(e.obs['vision']['cam_head']['color'], dtype=np.uint8)
        result = {row.prompt: row for row in e.model.segmenter.segment_many(
            image, [('cube', 0.15), ('ruler', 0.15)])}
        objects = {}
        for name, prompt, wanted in [('block', 'cube', 3), ('set_square', 'ruler', 1)]:
            item = result[prompt]
            instances = sorted(zip(item.masks, item.scores), key=lambda row: float(row[1]), reverse=True)
            detected = []
            for mask, score in instances:
                if float(score) < 0.15 or int(np.sum(mask)) < 15:
                    continue
                ys, xs = np.nonzero(mask)
                pixels = np.column_stack([xs, ys]).astype(float)
                # Generic camera-ray intersection with a declared approximate
                # tabletop object plane. This is a perception estimate.
                stride = max(1, len(pixels) // 2000)
                xyz = e.model.ground(pixels[::stride], z=0.785)
                rectangle = cv2.minAreaRect(xyz[:, :2].astype(np.float32))
                corners = cv2.boxPoints(rectangle)
                edges = np.roll(corners, -1, axis=0) - corners
                edge = edges[np.argmax(np.linalg.norm(edges, axis=1))]
                yaw = float(np.arctan2(edge[1], edge[0]))
                center = np.median(xyz, axis=0)
                detected.append(dict(position_m=center.tolist(), yaw_rad=yaw,
                    extent_xy_m=list(map(float, rectangle[1])), confidence=float(score)))
                if len(detected) == wanted:
                    break
            detected.sort(key=lambda obj: obj['position_m'][0])
            for index, obj in enumerate(detected):
                objects[f'{name}{index}' if wanted > 1 else name] = obj
        scene = {'objects': objects, 'frame': 'world', 'unit': 'm',
                 'table_height_m': 0.765, 'estimated_object_plane_m': 0.785,
                 'source': 'SAM3 RGB masks and calibrated camera rays'}
        e.record('scene', scene=scene)
        return scene

    def hold(self, steps=3):
        for _ in range(max(1, min(int(steps), 40))):
            self.e.emit(self.e.target)

    def gripper(self, arm, command, steps=3):
        if arm not in ['left', 'right'] or not 0 <= command <= 1:
            raise ValueError('invalid gripper command')
        self.e.record('gripper', arm=arm, command=command)
        self.e.target[arm + '_ee_joint_state'][:] = command
        self.hold(steps)

    def move_joints(self, arm, joints, steps=12):
        if arm not in ['left', 'right']:
            raise ValueError('invalid arm')
        start = np.asarray(self.e.obs['state'][arm + '_arm_joint_state'], dtype=float)
        end = np.asarray(joints, dtype=float)
        if end.shape != start.shape or not np.all(np.isfinite(end)):
            raise ValueError('invalid joint target')
        count = max(int(steps), int(np.ceil(np.max(np.abs(end - start)) / 0.18)), 2)
        if count > 45:
            raise ValueError('joint motion exceeds primitive budget')
        self.e.record('move_joints', arm=arm, joints=end, steps=count)
        for alpha in np.linspace(1 / count, 1, count):
            self.e.target[arm + '_arm_joint_state'] = start * (1 - alpha) + end * alpha
            self.e.emit(self.e.target)

    def approach(self, arm, position_m, quaternion_wxyz, steps=12):
        e = self.e
        pose = self._pose(position_m, quaternion_wxyz)
        e.record('approach', arm=arm, pose=pose)
        e.model.robot.update(e.obs['state'])
        trajectory = e.model.robot.path(arm, e.obs['state'][arm + '_arm_joint_state'], pose, None)
        count = min(max(2, int(steps)), 35, len(trajectory))
        previous = np.asarray(e.obs['state'][arm + '_arm_joint_state'])
        for index in np.linspace(0, len(trajectory) - 1, count, dtype=int):
            joint = trajectory[index]
            if np.max(np.abs(joint - previous)) > 0.4:
                self.move_joints(arm, joint.tolist(), 3)
            else:
                e.target[arm + '_arm_joint_state'] = joint
                e.emit(e.target)
            previous = joint

    def servo(self, arm, position_m, quaternion_wxyz=None, steps=12):
        e = self.e
        if quaternion_wxyz is None:
            quaternion_wxyz = e.obs['state'][arm + '_ee_pose'][3:]
        pose = self._pose(position_m, quaternion_wxyz)
        e.record('servo', arm=arm, pose=pose)
        e.model.robot.update(e.obs['state'])
        solution = e.model.robot.solve_ik(arm, pose)
        if solution['status'] != 'Success':
            raise RuntimeError('generic IK could not reach requested Cartesian pose')
        self.move_joints(arm, solution['joint_value'].tolist(), steps)
        self.hold(2)

    def home(self, arm='left', steps=15):
        self.move_joints(arm, self.e.initial[arm + '_arm_joint_state'], steps)
        self.hold(3)

    def contact_to_wrist(self, position_m, quaternion_wxyz):
        q = np.asarray(quaternion_wxyz, dtype=float)
        rotation = Rotation.from_quat(np.roll(q / np.linalg.norm(q), -1))
        return (np.asarray(position_m) - rotation.apply([0.135, 0, 0])).tolist()

    def quaternion_from_euler(self, xyz_radians):
        return np.roll(Rotation.from_euler('xyz', xyz_radians).as_quat(), 1).tolist()

    @staticmethod
    def _pose(position_m, quaternion_wxyz):
        p = np.asarray(position_m, dtype=float)
        q = np.asarray(quaternion_wxyz, dtype=float)
        if p.shape != (3,) or q.shape != (4,) or not np.all(np.isfinite(np.r_[p, q])):
            raise ValueError('Cartesian pose requires xyz and wxyz')
        if np.linalg.norm(q) < 0.5:
            raise ValueError('invalid quaternion')
        return np.r_[p, q / np.linalg.norm(q)].tolist()


class Model:
    def __init__(self, config):
        self.source = Path(config['source'])
        self.memory = Path(config['memory'])
        self.output = Path(config['output'])
        self.output.mkdir(parents=True)
        self.robot = Robot()
        self.segmenter = Sam3TextSegmenter()
        calibration = json.loads((Path(__file__).parent / 'generic_tools/head_camera.json').read_text())
        self.K = np.asarray(calibration['K'])
        self.T = np.asarray(calibration['T'])
        self.e = None
        self.episode_index = 0
        print('GENERATED_POLICY_READY: generic RGB and motion tools; no shipped alignment policy', flush=True)

    def ground(self, pixels, z):
        homogeneous = np.column_stack([pixels, np.ones(len(pixels))])
        camera_rays = homogeneous @ np.linalg.inv(self.K).T
        camera_rays[:, 1:] *= -1
        rays = camera_rays @ self.T[:3, :3].T
        origin = self.T[:3, 3]
        scale = (z - origin[2]) / rays[:, 2]
        return origin + rays * scale[:, None]

    def reset(self):
        if self.e is not None:
            with self.e.condition:
                self.e.cancelled = True
                self.e.condition.notify_all()
            self.e.thread.join(timeout=5)
        self.e = None

    def update_obs_batch(self, obs_list):
        if len(obs_list) != 1:
            raise ValueError('isolated generated policy requires one environment')
        self.update_obs(obs_list[0])

    def update_obs(self, obs):
        if self.e is None:
            self.e = Episode(self, obs, self.episode_index)
            self.episode_index += 1
        else:
            self.e.update(obs)

    def get_action(self):
        if self.e is None:
            raise ValueError('observation required')
        if self.e.complete and self.e.actions.empty():
            action = plain(self.e.target)
        else:
            action = self.e.actions.get(timeout=180)
            if action is None:
                action = plain(self.e.target)
        return [action]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        return [self.get_action()]


if __name__ == '__main__':
    from client_server.ws.model_server import PolicyServer, PolicyServerConfig
    model = Model(json.loads(Path(os.environ['GENERATED_POLICY_CONFIG']).read_text()))
    asyncio.run(PolicyServer(model, PolicyServerConfig(host='127.0.0.1',
        port=int(os.environ['GENERATED_POLICY_PORT']))).serve_forever())
