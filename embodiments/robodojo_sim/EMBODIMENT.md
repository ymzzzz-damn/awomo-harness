# RoboDojo direct joint control

This package exposes public RGB cameras, robot state and validated joint actions.
GPT decides the next targets; RoboDojo executes them and owns task scoring.

1. Start a session with `start_session`, provider `gpt6-harness`, the official task
   and seed. Poll `session_status` until control is ready.
2. Read `current_observation`. Preserve its official `instruction`; inspect all
   three camera frames and the supplied `action_spec` and `robot_state`.
3. Author 1..40 absolute joint/gripper action dictionaries in that exact schema.
   Call `submit_direct_action_chunk` with the current `observation_id`,
   `subtask_goal`, `fallback_reason` and `actions`. Submission does not move robots.
4. Call `execute_action_chunk` with the returned `chunk_id`, `max_actions` in
   1..40 and `discard_remaining=true`. Use shorter chunks when uncertain.
5. Reobserve and adjust targets. If subtask supervision is enabled, capture the
   current camera frames and call `reflect_action_chunk` after each nonterminal
   segment before submitting the next segment.
6. Stop at the native episode boundary and read the official evaluation result.
   Do not treat a completed chunk as task success or query a closed camera server.

The current deployment has no Python policy worker, segmentation model, Cartesian
IK wrapper or motion planner exposed to GPT. GPT must reason from public RGB and
proprioception; it cannot call `detect`, `approach` or `servo`. Prefer small observed
changes over long blind motion. All actions pass RoboDojo validation and its native
step budget. Do not read layout files, privileged object state or intermediate
scores to select actions. Historical logs may contain old worker records; they
are evidence files, not executable tools.

The registered VLA path remains available to separately configured providers.
The GPT-only runner disables VLA inference and uses the lifecycle transport stub
solely to start and stop an official episode.
