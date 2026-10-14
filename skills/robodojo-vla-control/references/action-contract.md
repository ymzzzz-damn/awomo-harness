# Action contract

## ObservationEnvelope

`current_observation` returns one chunk-boundary snapshot:

- `observation_id`: freshness token required by the next inference/action request;
- `image_paths`: current model-visible camera frames;
- `robot_state`: finite public proprioception only;
- `action_spec`: dynamic field names, dimensions, gripper convention, and limits;
- episode/action counters and active provider metadata.

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
actuating tool. It drains that validated chunk until completion or episode end,
then publishes a fresh observation. A stale/unknown chunk id is rejected.

Provider transport may replay a cached response using the same protocol request
id, but must not redo a stateful inference with a new id. Policy server restart
or lost temporal state is a provider failure and triggers replan/fallback, not a
silent reconnect to a fresh model state.
