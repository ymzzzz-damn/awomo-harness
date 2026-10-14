---
name: robodojo_sim
description: Simulation-only embodiment that applies planner, VLA/WAM, and GPT fallback actions in RoboDojo/Isaac Sim.
version: 0.5.0
kind: embodiment_context
environment: simulation
---

# RoboDojo Isaac Simulation Embodiment

This embodiment is the only component allowed to apply actions to the Isaac
simulation. The awomopaw model plans semantic subtasks. A registered VLA or WAM
normally converts a subtask plus the current observation into an action chunk.
When that provider cannot continue, awomopaw may produce a short direct-action
chunk. Both paths use the same action schema, validation, queue, simulator step,
trace, and episode limits.

## Control loop

1. Call `health` and select a ready provider from `available_policies`.
2. Call `start_session` with a canonical task and the selected `policy_id`.
3. Wait for `session_status` to report an observation.
4. Call `current_observation`; inspect public robot state and `action_spec`, then
   load the required model-visible RGB views with `capture_head_frame`,
   `capture_left_wrist_frame`, or `capture_right_wrist_frame`. These operations
   never expose reward or privileged simulator state.
5. Plan one semantic subtask and call `request_policy_chunk`.
6. Call `execute_action_chunk` with the returned chunk id.
7. At the next observation boundary, judge progress and continue or replan.
8. Only after an attributable provider failure or bounded no-progress decision,
   use `submit_direct_action_chunk`, then execute it through the same
   `execute_action_chunk` operation.
9. Abort when safe progress cannot be established. Read official results only
   after the episode process exits.

## Provider boundary

`policy_id` resolves through the server-side registry. Pi0.5 is the initial
provider; later VLA/WAM adapters register the same observation-to-action-chunk
capability without changing these embodiment tools. A session pins one provider
so a stale or mismatched id cannot redirect an active episode.

Every provider returns `robodojo.action-chunk/v1`. The embodiment validates
shape, numeric finiteness, action type, chunk length, stale observation ids, and
episode state before creating a chunk id. Only a subsequent execution call can
move the simulated arm.

## GPT direct-action fallback

Direct action is a recovery path, not the default policy. It requires the
current observation id, a subtask, an explicit fallback reason, and exactly one
action matching the dynamic `action_spec`. Use absolute joint targets and the
normalized gripper range reported by that spec. Never infer missing dimensions,
reuse an old observation id, or submit NaN/Inf values.

## Failure attribution

Treat provider timeout, transport failure, invalid or empty action chunk,
adapter-declared inability, and bounded visual no-progress as policy failures.
Malformed actions, stale ids, simulator errors, and exhausted episode limits
belong to the embodiment/session and must not be hidden by retrying. One revised
provider prompt is reasonable after ambiguous progress; repeated blind retries
are not.

## Safety and scoring boundaries

- Simulation only: no ROS, FCI, USB, or physical-device interfaces are present.
- Do not read reward, success, ground-truth object poses, or leaderboard-only
  metadata during an episode.
- Do not bypass RoboDojo validation, alter seeds mid-episode, or overwrite an
  official result.
- `task_success` is visible only after the episode has ended.
- Preserve the harness trace and post-episode artifacts for evaluation review.

## Legacy evaluation mode

`start_evaluation`, `current_evaluation`, `evaluation_status`, and
`evaluation_result` remain available for a provider-only end-to-end baseline.
Use the interactive session tools for the planner/router/fallback workflow.
