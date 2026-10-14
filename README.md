# Awomo Harness

Historical AwomoPaw / RoboDojo simulation harness source snapshot, before the later full-episode failure-video diagnosis and policy/memory rethinking workflow.

## Provenance

- Release: `20261013-e6e0dcd8-ed28c37dd22b` (2026-10-13).
- Upstream AwomoPaw/QwenPaw Git HEAD: `e6e0dcd85b11212d3bf945a724adc295baa03902`.
- Archive SHA256: `3b60870348b7c672c36d3e79e199fb9c6c60721e1bb870d93b81fdd07b178f2e`.
- The local archive and the P6000 release archive have identical SHA256 hashes.
- All 948 original regular files and the original checkpoint symlink are preserved.
- `SOURCE_MANIFEST.json` is the original release manifest.

## Layout

- `source/`: AwomoPaw Python package and the RoboDojo simulation embodiment.
- `robodojo-overlay/`: interactive episode harness and `Awomopaw_Harness` policy adapter for an external RoboDojo checkout.
- `skills/`: the original `robodojo-vla-control` skill.
- `runtime/`: the original runtime bootstrap and dependency constraints.

This is a historical deployment source snapshot. RoboDojo/Isaac simulator assets, VLA checkpoints, credentials, and installed environments are external. The adapter's checkpoint symlink points to `../Pi_05/checkpoints` in the intended RoboDojo deployment layout. This import preserves source; it does not establish a new simulator evaluation result.
