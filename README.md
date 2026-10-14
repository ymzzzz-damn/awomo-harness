# Awomo Harness

Current AwomoPaw / RoboDojo simulation harness source snapshot (2026-10-14), including chunk supervision, full-episode video diagnosis, policy/memory rethinking, task artifact evolution, and cross-scene failure experience.

The earlier version remains available at commit `f38c755510d7d0f42dc4e8530aec4ecacb305a7f`. Its original release manifest is saved in `docs/historical-release-manifest.json`. `SOURCE_MANIFEST.json` records the current imported source hashes and scope.

## Layout

- `source/`: current AwomoPaw agent package, frontend source, deployment files and tests.
- `robodojo-overlay/`: native interactive episode harness and `Awomopaw_Harness` adapter.
- `scripts/`, `runtime/`: execution, reflection, code-policy takeover, development evolution and evaluation drivers.
- `PhysicalRSI/`: included PhysicalRSI runtime source and pending rethinking/task-evolution additions.
- `skills/`, `embodiments/`: simulation tools and control contracts.
- `docs/`, `tests/`, `k8s/`, `ui/`: workflow documentation, regression tests, deployment manifests and trajectory viewer.

## Workflows

See [chunk supervision and task evolution](docs/awomopaw-supervision.md), [failure-video rethinking](PhysicalRSI/docs/robodojo-rethinking.md), and [the deployment workspace guide](docs/deployment-workspace.md) for the existing commands and configuration requirements. Install the agent package from `source/` using its `pyproject.toml`; the scripts import the included PhysicalRSI runtime from the repository layout.

This snapshot requires external RoboDojo/Isaac simulator assets, provider checkpoints, model credentials, and the relevant perception/IK dependencies. Existing machine paths in the imported drivers are deployment configuration and must match the execution environment. Source publication does not deploy the code, launch evaluations, or establish improved task success rates. PhysicalRSI demonstration videos, installed environments and episode outputs are excluded.
