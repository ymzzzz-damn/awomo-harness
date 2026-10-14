"""Expose official public instruction and short execution windows to the planner."""
from .interactive_harness import run_interactive_episode


def eval_one_episode(TASK_ENV, model_client):
    run_interactive_episode(TASK_ENV, model_client)


def eval_one_episode_batch(TASK_ENV, model_client):
    raise RuntimeError('Paired skills ablation requires one native scene per episode')
