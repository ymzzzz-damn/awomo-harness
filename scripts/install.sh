#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-${ROOT}/.venv}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/python" -m pip install -e "${ROOT}/source[simulation]"
"${VENV_DIR}/bin/python" -m pip check
"${VENV_DIR}/bin/awomopaw" --help
echo "Installed. Activate with: source ${VENV_DIR}/bin/activate"
echo "RoboDojo/Isaac and GPT provider credentials are configured separately; see docs/runtime.md."
