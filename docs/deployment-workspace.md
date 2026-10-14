# QwenPaw on RoboDojo (simulation-only)

远端框架、评测脚本、环境和结果的统一目录是 `/mnt/auwomo-data/wangxin/qwenpaw-harness`。
旧目录 `/mnt/auwomo-data/RoboDojo/qwenpaw-harness` 保留为兼容软链接，供已有进程和历史回执使用。
本地开发源码对应的远端工作副本位于 `workspace/awomopaw` 和 `workspace/robodojo`；已有实验代码快照保持原有内容。

## Local supervised harness update

The new local source supports two loops: mandatory LLM reflection after each
VLA/WAM execution window with immediate GPT takeover on a clear deviation, and
post-task video/result distillation of successful and failed development episodes
into versioned task skill/code/memory candidates. Use
`scripts/supervised_bridge.py` with `scripts/run_supervised_task.py --evolve`.
See [the protocol, commands and qualification limits](docs/awomopaw-supervision.md).
The existing deployed release and active frozen skill ablation remain unchanged.

This directory contains the deployment boundary for running the local QwenPaw
working tree in the RoboDojo Kubernetes environment.

The captured release intentionally contains:

- a snapshot of the local QwenPaw working tree (including its uncommitted
  source changes at packaging time),
- the generic embodiment loader/runtime,
- one simulation-only `robodojo_sim` embodiment,
- an isolated Python virtual environment built on the shared cluster mount.

It intentionally excludes all real-device embodiment packages and adapters,
including Franka, JetRover, Unitree G1, USB camera, ROS bridge deployments, and
all nested Git metadata. Simulation and real-device configuration do not share
a workspace or an embodiment search path.

## Cluster layout

- Namespace: `dreamzero-training`
- Shared release root: `/mnt/auwomo-data/RoboDojo/qwenpaw-harness/releases/<release>`
- Harness service: `qwenpaw-sim-harness:8088` (ClusterIP only)
- Persistent evaluation artifacts:
  `/mnt/auwomo-data/RoboDojo/qwenpaw-harness/eval_result`
- Pinned Isaac extension cache:
  `/mnt/auwomo-data/RoboDojo/qwenpaw-harness/isaac-cache`
- Model: `gpt-6-astra`, configured at runtime from a Kubernetes Secret
- Simulation bridge role: `robodojo_sim_api`
- Default bridge endpoint: `http://127.0.0.1:8098`

## Current deployment

- Release: `20261013-e6e0dcd8-ed28c37dd22b`
- Reused dependency layer: `20261012-e6e0dcd8-663991b07e95/runtime/venv`
- Bootstrap validation: passed with 15 `robodojo_sim` tools, one enabled skill,
  and no plaintext model secret persisted
- Deployment: `qwenpaw-sim-harness` (`1/1 Ready`)
- Service: `qwenpaw-sim-harness:8088` (`ClusterIP`)
- Pinned node: `192.168.48.153` (5090 node class)
- Enabled embodiment packages: `robodojo_sim` only
- Enabled QwenPaw skill: `robodojo-vla-control`
- Interactive bridge: `awomopaw-robodojo-embodiment/v3` on loopback port `8098`
- Harness memory backend: `none`; the unrelated builtin QA agent is disabled
- Image pull credential: existing namespace Secret `registry-secret`

The harness loads QwenPaw and embodiment source from the current immutable
release while reusing the already validated Python dependency layer. The
explicit `PYTHONPATH` prevents the dependency layer's older editable source
pointer from taking precedence.

The harness service is scheduled on the 5090 node. The P6000 admission policy
allocates that node only in 8-GPU units, so the pod reserves the full node while
`CUDA_VISIBLE_DEVICES=0` keeps the Pi0.5 policy server and headless Isaac process
on one logical device. RoboDojo/Isaac and a provider are started only when an
interactive session or formal evaluation is requested.

The simulation bridge is a narrow control-plane contract. RoboDojo/XPolicyLab
continues to own reset, Isaac stepping, action validation, scoring, and result
artifacts. The QwenPaw harness must never read reward, success, ground-truth
state, or leaderboard-only metadata during an episode.

The hybrid action path is documented in `skills/robodojo-vla-control`. GPT plans
one semantic subtask, then a registered VLA/WAM normally converts the current
observation into an action chunk. If that provider explicitly fails, GPT may
submit exactly one action against a fresh observation. Both sources use the
same embodiment queue and executor before Isaac is stepped. Model-specific
input mapping stays inside each adapter, so another VLA or WAM can be added
without changing the harness lifecycle.

The pod mounts the cluster's canonical RoboDojo Assets and Pi0.5 checkpoint
trees from `/mnt/auwomo-data/auwomo-datasets/raw-data/manipulation/RoboDojo`.
The Pi0.5 OpenPI source and virtual environment are kept separately under the
harness provider runtime root so model dependencies do not modify benchmark
data. Official evaluation JSON, harness traces, and camera videos are written
to the separate read-write `eval-result` volume; all benchmark inputs remain
read-only. The image-missing Isaac `omni.kit.pip_archive` extension is mounted
read-only from a pinned cache copied from the already validated local Isaac
runtime, so episode startup does not depend on downloading code from a public
registry.

## Deployment flow

1. `scripts/build_release.sh` packages the current local QwenPaw working tree.
2. Upload the tarball to a new immutable directory under the shared release
   root and extract it there.
3. Create/update the `qwenpaw-harness-api` Kubernetes Secret out-of-band.
4. Render the release path into the manifests.
5. Run `qwenpaw-sim-bootstrap`; it builds the shared venv, verifies the 5090,
   activates only `robodojo_sim`, and checks the model endpoint from the pod.
6. After the Job succeeds, apply the Deployment and ClusterIP Service.

No API key is stored in this repository, the release tarball, or the shared
filesystem.

## End-to-end validation

Both targets have been exercised by the configured GPT-6 planner, not only by
direct HTTP calls:

- Local RTX 4090 run `20261012T161530Z-4662f49e` captured the public camera
  observation, planned a semantic subtask, requested a 50-action Pi0.5 chunk,
  applied 41 actions through the embodiment until the episode terminal
  condition, and produced the official result, harness trace, and three camera
  videos.
- Node `192.168.48.153` run `20261012T171711Z-9d11bdff` captured all three
  public camera observations, planned a semantic subtask, requested one
  50-action Pi0.5 chunk, applied 40 actions without rewriting them, reached the
  official terminal state, and persisted `_result.json`, the harness trace,
  and three 42-frame videos on the shared evaluation volume. The pod was
  `qwenpaw-sim-harness-5b46b76889-hsv7r`, scheduled on node
  `192.168.48.153`, with zero restarts during validation.

Both validation episodes completed normally and received official score
`0.0`. This proves that the planning, provider, action execution, simulation,
evaluation, and artifact paths work on both targets; it does not claim that
the current Pi0.5 strategy solved `align_blocks`.

## Local short-window execution

The local embodiment now supports `max_actions=5..10` with
`discard_remaining=true`. It executes an unchanged prefix of the provider's
actions, discards the unused tail, publishes a fresh observation, and releases
the old chunk id. GPT replans from the new RGB frames before fresh Pi0.5 inference.
The low-level HTTP API preserves full-chunk execution when no limit is passed
for provider-only baselines. This local update has not been deployed to the
cluster release documented above.

Local seed-0 screening with a 10-action window was completed for
`general_pickup`, `stack_blocks_by_language`, `classify_objects_by_language`, and
`press_by_number`. All four hybrid scores remained 0; the pure Pi0.5 baseline
scored 100 for `general_pickup` and 0 for the other three tasks. The short-window
receipts and videos were verified, but this screening showed no score gain.
See `eval_result/local-task-comparison-20261013-horizon10/comparison.md`.

## Local imported PhysicalRSI task skill

The explicitly requested frozen PhysicalRSI `align-blocks` program is now
available as provider `physicalrsi-align`, kind `code`. Its evaluation-only
adapter is installed at the local XPolicyLab `PhysicalRSI_Align_Skill` directory;
the car_controller workspace has a separate `physicalrsi-align-blocks` skill.
The original released RGB perception / straightedge controller is unchanged.
It returns actions through the same interactive embodiment, without Pi0.5
inference or GPT-authored strategy edits. This supplied task-specific skill is
not the generic VLA-to-new-code fallback and does not replace its default route.

Local RTX 4090 screening on October 13 used seed 0, 1 and 2, one official layout
per seed: **3/3 success, score 100 for each**, using 95, 96 and 95 actions. All
nine native camera videos were copied, hashed and decoded successfully. This is
a small-sample result, not a full task-suite or leaderboard success rate. See
`eval_result/physicalrsi-imported-align-skill-20261013-171803/report.md`.

Reproduction (use a new output directory):

```bash
/home/ymz/physclaw/awomopaw/.venv/bin/python scripts/evaluate_frozen_align_skill.py \
  --seeds 0 1 2 --output /absolute/path/to/new-evidence-directory
```

The local bridge launch configuration is
`runtime/physicalrsi-align-skill/launch-bridge.sh`, and its provider registry is
`runtime/physicalrsi-align-skill/registry.json`. The adapter requires the already
installed pinned PhysicalRSI asset package and its inference Python environment.
This new code provider and task skill have **not** been deployed to node 153.

## Local inference trajectory UI

Run `python3 scripts/trajectory_ui.py --port 8790` and open
[the local viewer](http://127.0.0.1:8790). It reads saved evaluation artifacts
and provides task/run/seed filters, per-scene outcomes, three synchronized camera
videos, action-step navigation, saved Agent messages/tool calls, and controller
targets/IK evidence where recorded. Pending, interrupted and infrastructure
attempts remain distinct from scored task failures. See
[the viewer documentation](ui/trajectory/README.md) for supported artifact formats
and the limits of frame-to-action alignment.

## Failure video rethinking and self-evolution

The local development evolution drivers now automatically diagnose failed
episodes from timestamped frames covering all three official camera videos.
The harness saves the video evidence and LLM failure diagnosis, then asks the
LLM to rethink the policy and propose reusable lessons. The next CodeProposer
receives this feedback and generates a revised policy/memory candidate.

This is connected to `runtime/physicalrsi-evolution/run_evolution.py` and
`scripts/run_code_policy_fallback.py`. Self-Harness keeps independent candidate
selection and held-out testing separate from development reflection. Frozen
baseline sweeps retain their original behavior. These are local source changes;
the deployed immutable cluster release has not been updated.

Archived development failures can also be analyzed with
`scripts/run_rethinking.py`, without launching Isaac. See
[the rethinking workflow and example command](PhysicalRSI/docs/robodojo-rethinking.md)
for the model contract, artifacts and reproduction steps. This evolves code
and memory; it does not update LLM weights or imply a measured success-rate gain.

## AwomoPaw / Pi0.5 skill-text ablation on P6000

`scripts/evaluate_pi05_skills_seed0.py` evaluates all 55 official `align_blocks`
seed-0 layouts, with two fresh `QwenPawAgent` instances per layout (110 episodes).
Both conditions use GPT-6 Astra, the existing Pi0.5 checkpoint, and unchanged
ten-action prefixes with the old tail discarded. The two arms of each layout
use the same GPU; their order alternates. The task-specific difference is whether
the planner can read the original released PhysicalRSI straightedge controller
and memory as references. No code controller, numerical GPT correction or
benchmark-private state is available to either arm.

The evaluation package pins its own embodiment tool schema because the ordinary
deployed package predates the ten-action interface. The existing default agent,
bridge bindings and source controllers are not reconfigured. A separate
`Pi05_Skills_Ablation` policy adapter forwards the official public instruction
and short execution windows; it delegates inference to the original `Pi_05`.
The standard LightContextManager handles each fresh episode's active context. Each episode saves
the actual harness messages, source-file reads, native scene hash, action trace,
official score and three hashed/decoded native camera videos. Unscored native
instability and infrastructure/harness failures retain explicit records. Source
reading does not reproduce the original controller's deterministic perception,
IK and contact execution; this experiment measures transfer into Pi0.5 planning.

Current immutable runner source:
`/mnt/auwomo-data/RoboDojo/qwenpaw-harness/pi05-skills-ablation/20261013-224000`.
Results in the pod:
`/opt/robodojo/eval_result/pi05-skills-ablation/20261013-224000`.
The bastion's corresponding shared path starts with
`/mnt/auwomo-data/RoboDojo/qwenpaw-harness/eval_result/pi05-skills-ablation/`.
Launch and progress scripts are in `runtime/pi05-skills-seed0-20261013-224000`.
Startup and instruction-transport attempts are preserved separately under
`eval_result/pi05-skills-align-seed0-node153-20261013-221300` and
`eval_result/pi05-skills-align-seed0-node153-20261013-222100`. Their input errors
are excluded from the new experiment's success-rate denominator; the original
receipts and native zero scores from aborts remain archived.

Read-only local evidence synchronization:

```bash
python3 scripts/sync_pi05_skills_seed0.py \
  --experiment 20261013-224000 \
  --control-socket /tmp/robodojo-p6000-current.sock --watch
```

The local mirror is
`eval_result/pi05-skills-align-seed0-node153-20261013-224000`.
`summary.json` records progress and paired score differences; partial results
are provisional until every scene has a valid result in both arms.

