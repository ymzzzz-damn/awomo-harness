#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
QWENPAW_SOURCE="${QWENPAW_SOURCE:-/home/ymz/physclaw/QwenPaw}"
ROBODOJO_SOURCE="${ROBODOJO_SOURCE:-/home/ymz/robodojo-pi05/RoboDojo}"
OUTPUT_DIR="${1:-${DEPLOY_ROOT}/dist}"

if [[ ! -f "${QWENPAW_SOURCE}/pyproject.toml" || ! -d "${QWENPAW_SOURCE}/src/qwenpaw" ]]; then
  echo "QwenPaw source tree not found: ${QWENPAW_SOURCE}" >&2
  exit 1
fi
if [[ ! -f "${ROBODOJO_SOURCE}/src/eval_client/interactive_harness.py" \
   || ! -d "${ROBODOJO_SOURCE}/XPolicyLab/policy/Awomopaw_Harness" ]]; then
  echo "RoboDojo awomopaw overlay not found: ${ROBODOJO_SOURCE}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
STAGE_ROOT="$(mktemp -d)"
trap 'rm -rf -- "${STAGE_ROOT}"' EXIT
RELEASE_ROOT="${STAGE_ROOT}/release"
SOURCE_ROOT="${RELEASE_ROOT}/source"
mkdir -p \
  "${SOURCE_ROOT}/embodiments/packages" \
  "${RELEASE_ROOT}/robodojo-overlay/src/eval_client" \
  "${RELEASE_ROOT}/robodojo-overlay/XPolicyLab/policy" \
  "${RELEASE_ROOT}/runtime" \
  "${RELEASE_ROOT}/runtime/supervision" \
  "${RELEASE_ROOT}/skills"

rsync -a \
  --exclude='.git/' \
  --exclude='.venv/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='*.bak-*' \
  --exclude='.env' \
  --exclude='logs/' \
  --exclude='build/' \
  --exclude='dist/' \
  --exclude='console/node_modules/' \
  "${QWENPAW_SOURCE}/src" \
  "${SOURCE_ROOT}/"

for file in pyproject.toml README.md LICENSE; do
  cp -a "${QWENPAW_SOURCE}/${file}" "${SOURCE_ROOT}/${file}"
done

cp -a "${QWENPAW_SOURCE}/embodiments/__init__.py" "${SOURCE_ROOT}/embodiments/__init__.py"
cp -a "${QWENPAW_SOURCE}/embodiments/runtime" "${SOURCE_ROOT}/embodiments/runtime"
find "${SOURCE_ROOT}/embodiments/runtime" -type d -name __pycache__ -prune -exec rm -rf -- {} +
find "${SOURCE_ROOT}/embodiments/runtime" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
cp -a "${DEPLOY_ROOT}/runtime/minimal-embodiments-pyproject.toml" "${SOURCE_ROOT}/embodiments/pyproject.toml"
cp -a "${DEPLOY_ROOT}/embodiments/robodojo_sim" "${SOURCE_ROOT}/embodiments/packages/robodojo_sim"
cp -a "${DEPLOY_ROOT}/skills/robodojo-vla-control" "${RELEASE_ROOT}/skills/robodojo-vla-control"
cp -a "${DEPLOY_ROOT}/runtime/bootstrap_runtime.py" "${RELEASE_ROOT}/runtime/bootstrap_runtime.py"
cp -a "${DEPLOY_ROOT}/runtime/constraints.txt" "${RELEASE_ROOT}/runtime/constraints.txt"
cp -a "${DEPLOY_ROOT}/scripts/supervised_bridge.py" \
  "${DEPLOY_ROOT}/scripts/chunk_supervisor.py" "${RELEASE_ROOT}/runtime/supervision/"
cp -a \
  "${ROBODOJO_SOURCE}/src/eval_client/interactive_harness.py" \
  "${RELEASE_ROOT}/robodojo-overlay/src/eval_client/interactive_harness.py"
rsync -a \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  "${ROBODOJO_SOURCE}/XPolicyLab/policy/Awomopaw_Harness" \
  "${RELEASE_ROOT}/robodojo-overlay/XPolicyLab/policy/"

HEAD_SHA="$(git -C "${QWENPAW_SOURCE}" rev-parse HEAD)"
STATUS_SHA="$(git -C "${QWENPAW_SOURCE}" status --porcelain=v1 -uall | sha256sum | awk '{print $1}')"
SOURCE_TREE_SHA="$(find "${SOURCE_ROOT}" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')"
RELEASE_TREE_SHA="$(find "${RELEASE_ROOT}" -type f -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')"
export HEAD_SHA STATUS_SHA SOURCE_TREE_SHA RELEASE_TREE_SHA
python3 - "${RELEASE_ROOT}/SOURCE_MANIFEST.json" <<'PY'
import json
import os
import sys
from pathlib import Path

manifest = {
    "source": "local QwenPaw working tree",
    "git_head": os.environ["HEAD_SHA"],
    "git_status_digest": os.environ["STATUS_SHA"],
    "source_tree_sha256": os.environ["SOURCE_TREE_SHA"],
    "release_tree_sha256": os.environ["RELEASE_TREE_SHA"],
    "deployment_mode": "simulation-only",
    "included_embodiments": ["robodojo_sim"],
    "excluded_interfaces": [
        "franka_fr3",
        "jetrover_base",
        "jetrover_camera_arm",
        "unitree_g1",
        "unitree_g1_sensor_only",
        "g1_usb_dual_view",
        "usb_1080p_camera",
    ],
    "contains_api_key": False,
}
Path(sys.argv[1]).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

RELEASE_DATE="$(date +%Y%m%d)"
RELEASE_ID="${RELEASE_DATE}-${HEAD_SHA:0:8}-${RELEASE_TREE_SHA:0:12}"
ARCHIVE="${OUTPUT_DIR}/${RELEASE_ID}.tar.gz"
tar -C "${RELEASE_ROOT}" -czf "${ARCHIVE}" .
(cd "${OUTPUT_DIR}" && sha256sum "$(basename "${ARCHIVE}")" > "$(basename "${ARCHIVE}").sha256")
printf '%s\n' "${RELEASE_ID}" > "${OUTPUT_DIR}/LATEST_RELEASE"
printf '%s\n' "${ARCHIVE}"

