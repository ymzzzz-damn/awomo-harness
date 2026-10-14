"""Frozen providers act only through the interactive AwomoPaw executor."""
import os
from .interactive_harness import run_interactive_episode
from .ee_compat import install


def eval_one_episode(TASK_ENV, model_client):
    install(os.environ.get("ROBODOJO_ACTION_TYPE", "joint"), __package__ + ".interactive_harness")
    run_interactive_episode(TASK_ENV, model_client)


def eval_one_episode_batch(TASK_ENV, model_client):
    raise RuntimeError("Frozen skill evaluation requires one interactive environment per worker")
