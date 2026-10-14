# PhysicalRSI skills in AwomoPaw

This catalog imports all 22 released code programs and both released VLA
inference skills (`pi05`, `pi05-sparse-memory`). Controller sources, immutable
memory files, descriptors and implementation revisions remain frozen. The VLA
checkpoints are separate inputs; static readiness is not an inference outcome.

`install_all_rsi_skills.py` verifies the declared source dependencies, stages
24 workspace skills, and installs a shared adapter. Each code skill owns an
isolated worker and its perception services. RGB and public robot state are
forwarded; depth, object poses, reward and score are stripped.

The six native `ee` programs preserve their original pose/gripper actions.
The other programs use `joint`. The copied interactive executor validates
the selected action contract and executes the original action values; it
does not translate a frozen program into newly generated numeric actions.

The user-selected remote target is the existing `qwenpaw-sim-harness` pod on
P6000 node `192.168.48.153`. This task uses its already assigned GPUs, creates
no new cluster instance, and keeps the canonical benchmark assets unchanged.
The remote deployment root is:

```
/mnt/auwomo-data/RoboDojo/qwenpaw-harness/physicalrsi-all-skills/20261013-195000
```

The initial screening runs `push_T`, `stack_blocks_by_language`,
`press_by_number`, and `general_pickup`, each on official seed 0 / layout 0.
GPU IDs 1–4, bridge/control/policy ports, fixtures, logs and run IDs are separate.
Only the official native layout ID is selected outside the policy input.

The screening uses direct frozen-provider invocation through the AwomoPaw
embodiment. It is not evidence of an LLM Chat selecting or planning a policy.
Infrastructure errors and valid task outcomes are recorded separately. Each
valid episode must preserve its official result, trace and three decodable
native camera videos before it is reported as completed.

Deployment state lives in `deployment-status.json` under the source root.
The writable evaluation root is:

```
/mnt/auwomo-data/RoboDojo/qwenpaw-harness/eval_result/physicalrsi-all-skills/20261013-195000
```

It holds `install-receipt.json` and each batch's
`{manifest,status,results,summary}.json`. The supervisor validates
uploaded archives, starts the four GPU tasks, then activates the VLA entries
after their checkpoint upload finishes. It will not change an active ordinary
bridge episode to activate those entries.

The completed screening used one official seed-0 / layout-0 episode per task:
`press_by_number=100`, `push_T=100`, `general_pickup=100`,
`stack_blocks_by_language=20`. Three of four valid episodes succeeded. Three
initial provider initialization errors were preserved, then corrected using
the node's Torch 2.7.0 / CUDA 12.8 and a separate compatible virtualenv. Runtime
memory uses a unique writable run directory. No controller source was edited.
All 12 native camera videos were copied, hashed and decoded. All evaluation
processes finished and the assigned GPUs returned to idle.

The final local evidence and video links are in
`eval_result/physicalrsi-all-skills-node153-20261013/report.md`.
Source, checkpoints and evidence are persisted on the shared mount. The
inference virtualenv cache is local to the existing pod; a replacement pod
must recreate that cache before rerunning `prepare_p6000_skills.py`.

Relevant local validation: 31 CPU tests passed across the EE compatibility
boundary, original frozen adapter, layout fixture and interactive executor.
All 26 local registered providers (24 new and two existing) passed static
source/path validation; this does not claim that all 24 skills were scored.
