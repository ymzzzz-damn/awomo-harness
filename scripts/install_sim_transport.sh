#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${SIM_PYTHON:?Set SIM_PYTHON to the existing RoboDojo Python executable}"
TARGET="${TRANSPORT_LIB_DIR:-${ROOT}/runtime-state/sim-transport}"
# Keep Isaac's installed package set unchanged; use this path only for the client.
"${SIM_PYTHON}" -m pip install --upgrade --no-deps --target "${TARGET}" \
    websockets==15.0.1 msgpack-numpy==0.4.8
echo "export ROBODOJO_CLIENT_PYTHONPATH='${TARGET}'"
