# Failure and fallback

## Attributable provider failures

Route away from the active VLA/WAM when one of these is observed:

- provider unavailable, timeout, disconnect, or server instance change;
- adapter-declared unsupported/OOD;
- empty chunk, invalid shape/type, NaN/Inf, or action rejection;
- clearly wrong object, operation or trajectory in the first executed window;
- repeated actions or unchanged public visual state across a bounded subtask budget.

Do not use hidden reward or in-episode success to detect failure. A difficult
scene, low confidence alone, or one ambiguous frame is not enough to take over.

## Recovery order

1. Read fresh public state and all three cameras after every executed window.
2. Submit `reflect_action_chunk` with before/after evidence and a verdict:
   `continue` for useful progress, `correct` for a clear deviation/provider error,
   `uncertain` for insufficient evidence.
3. On `correct`, immediately use one grounded GPT direct action or generate a
   bounded direct joint action chunk. Do not require a retry/prompt revision.
4. Reobserve after correction and submit another reflection. Return to VLA/WAM
   only on `continue`. Execute the validated chunk, then inspect the fresh observation.
5. On `uncertain`, inspect again or abort; neither controller may move until
   reassessment. Abort on simulator failure, stale identities or exhausted budget.

The `fallback_reason` must name an observable signal, such as
`pi05 timeout at observation <id>` or `the first window moved a different object than the requested target`. Avoid vague reasons such as “model seems bad.”

## GPT direct-action constraints

- Use exactly one absolute-joint action so every GPT step receives a fresh observation.
- Copy field names and vector dimensions from the current action spec.
- Ground target values in public robot state plus current images; do not invent
  joint limits or hidden geometry.
- Keep gripper values within the advertised normalized range.
- Prefer the smallest corrective horizon that creates a new observation boundary.
- Never encode shell commands, simulator calls, reward probes, or arbitrary
  metadata in action fields.
