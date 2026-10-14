---
name: robodojo_sim
description: Simulation-only embodiment that applies planner, VLA/WAM, and GPT fallback actions in RoboDojo/Isaac Sim.
version: 0.8.0
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
   Supervised runs require `chunk_supervision.required=true`; use
   `scripts/supervised_bridge.py` for this protocol.
2. Call `start_session` with a canonical task and the selected `policy_id`.
3. Wait for `session_status` to report an observation.
4. Call `current_observation`; read the benchmark's full `instruction` as the
   authoritative task goal. If it is missing, stop and report an input-contract
   error; do not infer or replace the goal from a task name or image. Inspect
   public robot state and `action_spec`, then
   load the required model-visible RGB views with `capture_head_frame`,
   `capture_left_wrist_frame`, or `capture_right_wrist_frame`. These operations
   never expose reward or privileged simulator state.
5. Plan one semantic subtask from that instruction and the public views, keeping
   its required operation, tool, objects, ordering, and finish conditions in the
   provider prompt. Call `request_policy_chunk`; no task-specific solution is
   supplied by this embodiment.
6. Call `execute_action_chunk` with the returned chunk id, `max_actions=10`
   (or 5 for finer feedback), and `discard_remaining=true`. This executes only
   the original provider prefix and discards its unexecuted tail.
7. At each fresh observation boundary, read all three cameras and compare the
   previous/current views and execution receipt. Call `reflect_action_chunk` with
   the last chunk id, current observation id, visible evidence and verdict.
   `continue` permits VLA/WAM; `correct` blocks VLA/WAM and permits immediate GPT
   direct/code takeover; `uncertain` blocks motion until reassessment or abort.
   Never resume an old tail. A GPT correction also requires fresh reflection.
8. On the first clearly wrong execution or provider failure, use
   `submit_direct_action_chunk` or `run_code_policy` immediately. No failed
   retry or prompt revision is required. The terminal window is reflected using
   saved videos after the native control server closes.
9. Abort when safe progress cannot be established. Read official results only
   after the episode process exits.

## Provider boundary

Only explicitly registered external providers are available. Learned AwomoPaw skills are loaded through their versioned task artifacts.

`policy_id` resolves through the server-side registry. Pi0.5 is the initial
provider; later VLA/WAM adapters register the same observation-to-action-chunk
capability without changing these embodiment tools. A session pins one provider
so a stale or mismatched id cannot redirect an active episode.

Every provider returns `robodojo.action-chunk/v1`. The embodiment validates
shape, numeric finiteness, action type, chunk length, stale observation ids, and
episode state before creating a chunk id. Only a subsequent execution call can
move the simulated arm.

## GPT direct-action fallback

A `correct` reflection may immediately trigger a generated code policy. Read `code_policy_api`, then write a new `policy(robot, memory)`
using that API and call `run_code_policy` with the current observation id, source,
memory and visible failure reason. The benchmark instruction and episode origin
come from the environment, never from a replacement goal in the request.

The code worker uses public RGB segmentation, calibrated estimates, generic IK
and movement. It submits one original joint action at a time to the same
embodiment and reobserves after each. Do not request VLA actions or issue manual
actions concurrently. Poll `code_policy_status`; a worker return is not success.
If a code revision fails, preserve its source and public trace before revising.
An ended episode requires an explicit new recorded attempt. This backend is
currently calibrated for the installed arx_x5 robot only.

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
