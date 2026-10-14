"""Model-written Python policy -> generic primitives -> existing HTTP embodiment."""

import builtins
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time
import traceback
import urllib.request

import cv2
import numpy as np
from PIL import Image

from contract import validate_source
from bridge import PrimitiveClient, Robot, Sam3TextSegmenter, plain


class EpisodeFinished(BaseException):
    pass


class HttpEpisode:
    def __init__(self, config, *, backend=None, request=None):
        self.config = config
        self.root = Path(config["output"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_id = config["run_id"]
        self.instruction = config["instruction"]
        self.initial = config["initial_state"]
        self.steps = 0
        self.target = {}
        self.request_override = request
        self.model = backend
        self.code_sha256 = hashlib.sha256(Path(config["source"]).read_bytes()).hexdigest()
        self.refresh()
        self.target = {k: np.asarray(v, dtype=float) for k, v in self.obs["state"].items()
                       if k.endswith(("_arm_joint_state", "_ee_joint_state"))}
        self.maximum_actions = min(175, self.public["action_limit"] - self.public["action_count"])
        if self.maximum_actions <= 0:
            raise ValueError("episode has no remaining action budget")
        self.record("code_policy_started", instruction=self.instruction, initial_state=self.initial,
                    takeover_state=self.obs["state"], source_sha256=self.code_sha256,
                    maximum_actions=self.maximum_actions)

    def request(self, path, payload=None):
        if self.request_override:
            return self.request_override(path, payload)
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(self.config["bridge"] + path, data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=240) as r:
            return json.load(r)

    def record(self, event, **values):
        with (self.root / "events.jsonl").open("a") as f:
            f.write(json.dumps(plain({"event": event, "step": self.steps, **values})) + "\n")

    def refresh(self):
        public = self.request("/v1/sessions/observation")
        if public["run_id"] != self.run_id:
            raise ValueError("code policy cannot cross a benchmark episode")
        if public.get("instruction") != self.instruction:
            raise ValueError("benchmark instruction changed during code execution")
        if public.get("episode_end"):
            raise EpisodeFinished()
        if public.get("pending_chunk_id"):
            raise ValueError("a previous chunk must be executed/discarded before code takeover")
        self.public = public
        # Only public proprioception enters generated code, never score/reward.
        self.obs = {"instruction": self.instruction, "state": public["robot_state"]}

    def image(self):
        with urllib.request.urlopen(self.config["bridge"] + "/v1/sessions/observation/image?topic=cam_head", timeout=30) as r:
            rgb = np.asarray(Image.open(io.BytesIO(r.read())).convert("RGB"))
        Image.fromarray(rgb).save(self.root / f"head-{self.steps:03d}.png")
        return rgb

    def emit(self, action):
        if self.steps >= self.maximum_actions:
            raise RuntimeError("code policy action budget exhausted")
        action = plain(action)
        chunk = self.request("/v1/sessions/direct-action-chunks", {
            "observation_id": self.public["observation_id"],
            "subtask_goal": self.instruction,
            "fallback_reason": self.config["fallback_reason"][:1000],
            "rationale": "Generated code policy " + self.code_sha256,
            "actions": [action],
        })
        execution = self.request("/v1/sessions/action-chunks/execute", {
            "chunk_id": chunk["chunk_id"], "max_actions": 1, "discard_remaining": True,
        })
        self.steps += execution["executed"]
        self.record("code_action_executed", action=action, execution=execution)
        if execution.get("episode_end"):
            self.record("episode_terminal")
            raise EpisodeFinished()
        self.refresh()
        if self.steps % 10 == 0:
            self.record("observed_state", state=self.obs["state"])
            self.image()


class HttpRobot(PrimitiveClient):
    def reobserve(self):
        self.e.refresh()
        self.e.record("reobserve", state=self.e.obs["state"])
        return plain({"instruction": self.e.instruction, "state": self.e.obs["state"], "initial_state": self.e.initial})

    def scene(self):
        raise ValueError("Use the general detect(text, count) API; no task-specific scene is supplied")

    def detect(self, text, count=1, plane_z=None):
        if not isinstance(text, str) or not 1 <= len(text) <= 200:
            raise ValueError("detection text must contain 1 to 200 characters")
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 20:
            raise ValueError("count must be an integer from 1 to 20")
        z = 0.785 if plane_z is None else float(plane_z)
        if not math.isfinite(z) or not 0.70 <= z <= 1.20:
            raise ValueError("perception plane is outside the calibrated workspace")
        item = self.e.model.segmenter.segment(self.e.image(), text, confidence_threshold=0.15)
        detected = []
        for mask, score in sorted(zip(item.masks, item.scores), key=lambda r: float(r[1]), reverse=True):
            if float(score) < 0.15 or int(np.sum(mask)) < 15:
                continue
            ys, xs = np.nonzero(mask)
            pixels = np.column_stack([xs, ys]).astype(float)
            xyz = self.e.model.ground(pixels[::max(1, len(pixels) // 2000)], z)
            rectangle = cv2.minAreaRect(xyz[:, :2].astype(np.float32))
            corners = cv2.boxPoints(rectangle)
            edges = np.roll(corners, -1, axis=0) - corners
            edge = edges[np.argmax(np.linalg.norm(edges, axis=1))]
            axis = edge / max(float(np.linalg.norm(edge)), 1e-8)
            detected.append({"position_m": np.median(xyz, axis=0).tolist(),
                             "yaw_rad": float(np.arctan2(edge[1], edge[0])),
                             "long_axis_xy": axis.tolist(), "extent_xy_m": list(map(float, rectangle[1])),
                             "confidence": float(score)})
            if len(detected) == count:
                break
        detected.sort(key=lambda o: o["position_m"][0])
        self.e.record("detect", text=text, plane_z=z, objects=detected,
                      source="SAM3 public RGB and static camera calibration")
        return detected


class Backend:
    def __init__(self):
        self.robot = Robot()
        self.segmenter = Sam3TextSegmenter()
        calibration = json.loads((Path(__file__).resolve().parents[1] / "physicalrsi-evolution/generic_tools/head_camera.json").read_text())
        self.K = np.asarray(calibration["K"])
        self.T = np.asarray(calibration["T"])

    def ground(self, pixels, z):
        camera_rays = np.column_stack([pixels, np.ones(len(pixels))]) @ np.linalg.inv(self.K).T
        camera_rays[:, 1:] *= -1
        rays = camera_rays @ self.T[:3, :3].T
        origin = self.T[:3, 3]
        return origin + rays * ((z - origin[2]) / rays[:, 2])[:, None]


def main():
    config = json.loads(Path(sys.argv[1]).read_text())
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    status = {"status": "running", "started_at": time.time()}
    source = Path(config["source"]).read_text()
    validate_source(source)
    episode = None
    try:
        episode = HttpEpisode(config)
        episode.image()
        episode.model = Backend()
        allowed = ("abs", "min", "max", "sum", "len", "range", "enumerate", "zip", "sorted", "list", "dict", "tuple", "float", "int", "str", "bool", "round", "all", "any", "Exception", "ValueError")
        namespace = {"__builtins__": {n: getattr(builtins, n) for n in allowed}}
        exec(compile(source, config["source"], "exec"), namespace)
        memory = config.get("memory", {})
        namespace["policy"](HttpRobot(episode), memory)
        episode.record("policy_returned", memory=memory)
        status["memory"] = memory
        # Briefly settle the last joint target so the independent evaluator can
        # observe it. This supplies no manipulation strategy or success claim.
        HttpRobot(episode).hold(3)
        status["status"] = "returned"
    except EpisodeFinished:
        status["status"] = "episode_terminal"
    except BaseException as error:
        status.update(status="error", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        if episode:
            episode.record("policy_error", **status)
    finally:
        status.update(finished_at=time.time(), executed_actions=episode.steps if episode else 0,
                      source_sha256=hashlib.sha256(source.encode()).hexdigest())
        (output / "worker-status.json").write_text(json.dumps(status, indent=2) + "\n")
        print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
