# Failure and fallback

## Attributable provider failures

Route away from the active VLA/WAM when one of these is observed:

- provider unavailable, timeout, disconnect, or server instance change;
- adapter-declared unsupported/OOD;
- empty chunk, invalid shape/type, NaN/Inf, or action rejection;
- repeated actions or unchanged public visual state across a bounded subtask
  budget after one meaningful prompt revision.

Do not use hidden reward or in-episode success to detect failure. A difficult
scene, low confidence alone, or one ambiguous frame is not enough to take over.

## Recovery order

1. Re-read the latest observation and confirm the failure belongs to the provider,
   not a stale id or ended episode.
2. If the subtask/prompt was ambiguous, revise it once and request a fresh policy
   chunk from the current observation.
3. If the provider still cannot continue, replan the subtask and use one bounded
   GPT direct-action chunk matching the current action spec.
4. Observe again. Return to the registered provider when the blockage is removed;
   do not leave GPT in an unbounded low-level loop.
5. Abort if the action cannot be grounded safely, validation keeps failing, the
   simulator is unhealthy, or the episode budget is nearly exhausted.

The `fallback_reason` must name an observable signal, such as
`pi05 timeout at observation <id>` or `no visual progress after two validated
chunks and one prompt revision`. Avoid vague reasons such as “model seems bad.”

## GPT direct-action constraints

- Use exactly one absolute-joint action so every GPT step receives a fresh observation.
- Copy field names and vector dimensions from the current action spec.
- Ground target values in public robot state plus current images; do not invent
  joint limits or hidden geometry.
- Keep gripper values within the advertised normalized range.
- Prefer the smallest corrective horizon that creates a new observation boundary.
- Never encode shell commands, simulator calls, reward probes, or arbitrary
  metadata in action fields.
