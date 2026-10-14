# RoboDojo failure video rethinking

`PhysicalRSI_baselines.robodojo.rethinking.VideoRethinker` adds an explicit
System 2 step after a failed, completed **development** episode. The model's
diagnosis does not replace RoboDojo's score or claim that a candidate works.

The original drivers retain failure-only behavior by default. The new task
evolution path explicitly uses `include_success=True`, so completed successful
episodes also yield timestamped diagnosis and reusable mechanisms. Its diagnosis
uses `outcome_reason` and includes `terminal_outcome` in feedback. See
[the supervised execution and task evolution workflow](../../docs/awomopaw-supervision.md)
for the two loops, automatic development rounds and remaining promotion boundary.

The sequence is:

1. Read the native per-episode outcome and select that episode's head and both
   wrist camera videos. A successful episode does not trigger reflection.
2. Hash the original videos and decode up to eight evenly spaced frames per camera,
   including the first and last frame. The model receives up to 24 images, ordered
   by progression through the episode, with camera IDs and exact timestamps.
3. Ask the connected vision model for a failure description, timestamped
   observations, causal hypotheses with confidence, and uncertainty. Missing
   frames, occlusion and gaps must remain explicit. This is sampled video
   analysis; the model does not view every frame continuously.
4. Return **both video evidence and the structured failure diagnosis** to the
   harness as `diagnostic-feedback.json` before making the second model call.
5. Ask the model to rethink the current policy and memory from this feedback.
   It returns `policy_changes`, `memory_delta` and `validation_plan`.
6. Save `harness-feedback.json`. The next CodeProposer receives this artifact
   through `development_feedback.failure_rethinking`, generates a new code and
   memory candidate, and executes it in a new recorded episode.

This evolves policy code, prompts and reusable lessons. It does not train or
change model weights. Lessons are proposals: the existing candidate selection
and lineage process determines whether to promote a new revision.

## Integrated workflows

From the RoboDojo deployment workspace (the parent of this repository):

- `runtime/physicalrsi-evolution/run_evolution.py`: development failures now
  invoke video diagnosis and rethinking automatically. Both parent and newly
  generated candidate development evidence carry the resulting feedback. The
  existing Self-Harness still compares candidates on separate selection cases,
  records lineage and freezes the survivor before the held-out final test.
- `scripts/run_code_policy_fallback.py`: completed failed VLA/code attempts
  receive video diagnosis and rethinking before the next code revision. The
  next proposal sees previous video evidence, diagnosis and the improvement
  plan. This driver is single-scene development; its `selected-policy.py`
  means development success, not held-out qualification.
- `scripts/run_rethinking.py`: analyze an archived bridge result without
  starting a simulator. It produces a development-feedback artifact for a
  harness to consume. It does not itself execute or promote a candidate.

Provider-only frozen baseline sweeps are unchanged. The cluster's existing
immutable release must be rebuilt and deployed separately to run new code.
This change does not add an HTTP operation to a running bridge or automatically
start development retries in unrelated Chat evaluations.

## Analyze an archived development failure

The input is the bridge's `official-result.json` shape: `run` contains `run_id`,
`task`, `status: completed` and native `result`; `videos` contains local paths
to the official camera MP4s. Pass the unmodified native instruction and the
policy/memory that produced this result. Use a new output directory.

```bash
/home/ymz/physclaw/awomopaw/.venv/bin/python scripts/run_rethinking.py \
  --official-result /absolute/path/to/official-result.json \
  --instruction 'The original native task instruction' \
  --source /absolute/path/to/policy.py \
  --memory /absolute/path/to/memory_rules.json \
  --split development \
  --output /absolute/path/to/new-rethinking-artifacts
```

The default model adapter uses the existing local provider configuration.
`--provider-config` and `--model` select another connected, vision-capable
Chat Completions provider. No credentials are saved to the evidence artifacts.
The generic `VideoRethinker` accepts any model implementing
`complete(messages, tools)` and `config.public()`; image messages use the same
multimodal contract as CodeProposer's existing model adapter. `ffmpeg` and
`ffprobe` must be available on `PATH`.

## Evidence and failure behavior

Each episode directory preserves the frozen inputs, source/video hashes,
sampled frames, `video-evidence.json`, both model requests and responses,
`diagnosis.json`, `diagnostic-feedback.json`, `rethink.json` and
`harness-feedback.json`. The latter contains hashes for its underlying evidence.
Exact completed inputs can be reused without calling the model again. Modified
inputs or evidence require a new revision. An incomplete model call requires
explicit reconciliation or a new directory; it is never silently replayed.

Validation/final splits, unfinished runs, missing camera videos and malformed
diagnoses are rejected. Infrastructure failures stop code-policy development
instead of becoming a strategy lesson. A model failure preserves the original
official task outcome and diagnostics, and prevents the next revision from
claiming to have consumed a successful reflection. Neither held-out outcomes
nor hidden simulator state enter the model's diagnostic feedback.
