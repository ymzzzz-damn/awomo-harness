#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-${ROOT}/.venv}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
INSTALL_ARGS=()
INSTALL_CONSTRAINTS="${INSTALL_CONSTRAINTS:-}"
RUNTIME_TARGET="$("${VENV_DIR}/bin/python" -c 'import sys,platform; print(sys.platform, platform.machine(), "%d.%d" % sys.version_info[:2])')"
if [[ -z "${INSTALL_CONSTRAINTS}" && "${RUNTIME_TARGET}" == "linux x86_64 3.10" ]]; then
    INSTALL_CONSTRAINTS="${ROOT}/requirements/linux-py310-tested.txt"
fi
if [[ -n "${INSTALL_CONSTRAINTS}" ]]; then
    INSTALL_ARGS+=(--constraint "${INSTALL_CONSTRAINTS}")
fi
"${VENV_DIR}/bin/python" -m pip install "${INSTALL_ARGS[@]}" -e "${ROOT}/source[simulation,transport]"
"${VENV_DIR}/bin/python" -m pip check
"${VENV_DIR}/bin/awomopaw" --help
echo "Installed. Activate with: source ${VENV_DIR}/bin/activate"
echo "RoboDojo/Isaac and GPT provider credentials are configured separately; see docs/runtime.md."
