"""RoboDojo episode loop delegated to the awomopaw interactive harness."""

from src.eval_client.interactive_harness import run_interactive_episode


def eval_one_episode(TASK_ENV, model_client):
    run_interactive_episode(TASK_ENV, model_client)


def eval_one_episode_batch(TASK_ENV, model_client):
    raise RuntimeError("Awomopaw_Harness requires eval_batch=false and one environment")
