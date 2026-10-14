# AwomoPaw chunk supervision and task evolution

The existing framework was partially compatible: it already had short provider
execution windows, fresh public camera observations, direct-action/code takeover,
failure-video rethinking and an independent Self-Harness. Its normal planner did
not enforce a reflection after each chunk, required repeated failures before code
takeover, and its video reflector rejected successful episodes.

Two separate loops now have executable entry points. These are local source
changes, not a deployed release or evidence of better task success rates.

## Execution loop

`scripts/supervised_bridge.py` extends the existing RoboDojo run manager. The
simulator and policy adapters remain responsible for observations, provider
inference, action validation/execution and native scoring. The bridge enforces:

1. Infer with the selected VLA/WAM and execute at most ten unchanged actions;
   discard the old chunk tail.
2. Fetch the current public state and all three camera images. The supervising
   LLM compares before/after images and the last execution receipt.
3. Submit `reflect_action_chunk` with the current observation id, last chunk id,
   explicit reflection and `continue`, `correct` or `uncertain`.
4. `continue` permits the next VLA/WAM chunk. `correct` immediately permits GPT
   direct action or generated code and blocks VLA/WAM. `uncertain` blocks both
   controllers until reassessment or abort. Correction receives fresh reflection
   before returning to VLA/WAM.

The model writes the semantic judgment. The bridge validates identities, camera
retrieval and controller routing; it cannot certify the truth of a visual
judgment. Reflections record camera hashes and the exact receipt in
`chunk-reflections.jsonl` inside the native run artifacts. The explicit runner
also saves the before/after images and model requests/responses.

Transport/validation failures must not masquerade as task failures. Only the
native `provider_inference_failed` signal enables failure takeover without an
executed chunk. A delegated code worker owns its primitive loop; it still uses
the same executor and fresh observations, and VLA inference is excluded while
it runs. Code takeover requires the installed arx_x5 perception/IK dependencies.
One grounded direct action remains available without that backend.

RoboDojo closes its in-episode server at termination. The terminal chunk's
reflection therefore uses saved videos in the post-task loop. No controller is
resumed after termination, and official results stay outside online decisions.

## Task evolution loop

`scripts/run_supervised_task.py --evolve` automatically runs post-task evolution
after a completed development episode. `scripts/evolve_task_artifact.py` can run
the same process against archived evidence without launching Isaac:

1. Decode bounded temporal samples from each native camera video, including
   endpoints, and preserve timestamps and file/frame hashes.
2. Ask the connected LLM to diagnose successful mechanisms or failure causes,
   citing supplied timestamps and distinguishing hypotheses from observations.
3. Give that diagnosis, native result, chunk reflections, existing task skill and
   the actually executed source/memory back to the LLM.
4. Distill a revised `SKILL.md`, generic `policy.py` when a primitive API is
   supplied, and `memory.json`. A `task-artifact.json` binds content revision,
   parent revision, feedback hash, rationale and validation plan.
5. Use `--rounds N --evolve` to automatically pass each candidate into the next
   recorded development run, or explicitly resume with `--task-artifact`.
   `evolution-lineage.json` records the parent/child/run links. Skill text informs planning; existing source/memory informs
   code revision at takeover. Candidates are never silently promoted.

Only completed development outcomes enter this loop. Failed infrastructure,
validation and final-test outcomes are rejected before video/model calls.
Independent comparison and promotion remain the existing Self-Harness's job.
This runner does not yet automate registering skill-text candidates with its
selection evaluator, and it does not update VLA/WAM or LLM weights. A candidate
and a successful development episode establish no general success-rate gain.

## Running the new loop

Use the existing simulation configuration/registry and an available GPU. Do not
start this alongside another episode on the same GPU. These example control
ports and state directory separate the new bridge from the ordinary local bridge;
the state directory does not isolate GPU capacity or benchmark output ownership.

```bash
cd /home/ymz/Documents/ChatGPT/robodojo
ROBODOJO_HARNESS_STATE_DIR=/tmp/robodojo-supervised-state \
AWOMOPAW_HARNESS_PORT=8119 \
/home/ymz/physclaw/awomopaw/.venv/bin/python \
  scripts/supervised_bridge.py --port 8118
```

In a second terminal with matching simulation/provider settings:

```bash
cd /home/ymz/Documents/ChatGPT/robodojo
/home/ymz/physclaw/awomopaw/.venv/bin/python scripts/run_supervised_task.py \
  --bridge http://127.0.0.1:8118 --task align_blocks --policy-id pi05 \
  --seed 0 --horizon 10 --split development --evolve \
  --output /absolute/path/to/new-supervised-run
```

Append `--rounds 3` for a bounded three-round development evolution loop. Each
round starts an explicitly recorded new episode and uses the previous candidate;
it does not turn development results into independent qualification evidence.

The default supervising model is `gpt-6-astra`; `--provider-config` and `--model`
can select the connected endpoint without editing provider settings. Credentials
are not written to evidence. The runner reads its authoritative instruction from
the native initial observation and aborts on missing instructions.

For a task with an existing artifact, append
`--task-artifact /absolute/path/to/parent/task-artifact.json` to a development
run. To begin from ordinary task skill/source/memory files instead, distill an
archived development run:

```bash
/home/ymz/physclaw/awomopaw/.venv/bin/python scripts/evolve_task_artifact.py \
  --official-result /absolute/path/to/run/official-result.json \
  --initial-observation /absolute/path/to/run/initial-observation.json \
  --skill /absolute/path/to/task/SKILL.md \
  --split development --output /absolute/path/to/new-task-candidate
```

Optional `--source`, `--memory`, `--primitive-api` and `--reflections` add existing
generic code, memory, the documented primitive API and the runner's reflection
JSON. Generic code must define `policy(robot, memory)` and pass code admission;
frozen external providers remain separate registered providers.

The embodiment package exposes `reflect_action_chunk` to the actual AwomoPaw
agent as well. Its control skill describes the protocol. The release builder
includes the supervised bridge, the bootstrap checks all nineteen tools, and the
Kubernetes manifest starts that bridge for future releases. Existing immutable
deployments and frozen Pi0.5 ablation scripts are not changed by these source edits.
