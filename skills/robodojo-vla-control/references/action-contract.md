# Action contract

## ObservationEnvelope

`current_observation` returns one chunk-boundary snapshot:

- `observation_id`: freshness token required by the next inference/action request;
- `image_paths`: current model-visible camera frames;
- `robot_state`: finite public proprioception only;
- `action_spec`: dynamic field names, dimensions, gripper convention, and limits;
- episode/action counters and active provider metadata.
- `supervision`: last execution receipt, chunk id and reflection verdict, when
  the supervised bridge is active.

It excludes reward, success before episode end, ground-truth object state, and
scoring metadata. Never reuse an observation id after a chunk executes.

## ActionChunk

Both provider and GPT paths create the same server-side chunk:

```json
{
  "chunk_id": "opaque-id",
  "observation_id": "source-observation-id",
  "source": "policy | llm_direct",
  "provider_id": "pi05 | another-provider | gpt-direct-action",
  "action_count": 1,
  "action_signatures": [{"left_arm_joint_state": [6]}]
}
```

Raw provider actions stay inside the embodiment. GPT direct-action is accepted
only as JSON arrays described by the current `action_spec` and is immediately
converted to the same internal chunk representation.

## Validation before execution

The embodiment rejects a chunk unless all invariants hold:

1. source observation id is current and the episode is active;
2. chunk is non-empty and within its source limit (policy: 256, GPT: exactly 1);
3. every action is a non-empty object with the required joint/gripper fields;
4. each field is one-dimensional, has the advertised dimension, and is numeric;
5. every value is finite; NaN and infinity are forbidden;
6. action type is the session's fixed joint contract;
7. RoboDojo's own `validate_action_dict` accepts every action.

Validation never pads, clips, repeats, or repairs malformed arm values. RoboDojo
may map a normalized gripper scalar according to its existing robot config.

## Execution semantics

Creating a chunk never actuates Isaac. `execute_action_chunk` is the only
actuating tool. With `max_actions=5..10`, it executes at most that many original
actions, then publishes a fresh observation. With `discard_remaining=true`,
the unused tail is invalidated and the chunk id is released: the planner must
observe, replan, and infer a new chunk. Trace and receipts report `executed`,
`discarded_actions`, `chunk_released`, and the new observation id.

Omitting `max_actions` preserves full-chunk execution for provider-only baselines.
With `discard_remaining=false`, a caller may explicitly resume the pending tail;
this is not the normal replanning workflow. A stale/unknown/released chunk id is
rejected. Neither execution mode changes the provider's numeric action values.

Provider transport may replay a cached response using the same protocol request
id, but must not redo a stateful inference with a new id. Policy server restart
or lost temporal state is a provider failure and triggers replan/fallback, not a
silent reconnect to a fresh model state.

## Reflection handoff

The supervised bridge requires a current three-camera read and
`reflect_action_chunk(observation_id, chunk_id, verdict, reflection)` after each
execution window. `continue` permits policy inference; `correct` blocks policy
inference and permits immediate direct/code takeover; `uncertain` blocks motion.
An unused tail can never bypass reflection. Native terminal execution closes the
control server, so its reflection uses saved videos after episode completion.
