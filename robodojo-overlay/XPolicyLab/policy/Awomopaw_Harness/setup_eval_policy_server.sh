#!/usr/bin/env bash
set -euo pipefail
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.3

bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_uv_env=${8:-uv}
policy_server_port=$9
policy_server_host=${10:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${SCRIPT_DIR}/deploy.yml"
action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${BENCH_ROOT}" "${env_cfg_type}")

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
YAML_PYTHON="${CONDA_BASE}/bin/python"
policy_uv_env_path="${AWOMOPAW_POLICY_UV_ENV_PATH:-}"
if [[ -z "${policy_uv_env_path}" ]]; then
policy_uv_env_path="$(${YAML_PYTHON} - <<PYENV
import yaml
from pathlib import Path
script_dir = Path("${SCRIPT_DIR}")
cfg = yaml.safe_load(open("${yaml_file}", encoding="utf-8"))
path = Path(cfg["policy_uv_env_path"]).expanduser()
if not path.is_absolute():
    path = (script_dir / path).resolve()
print(path)
PYENV
)"
fi
policy_uv_env_path="$(realpath "${policy_uv_env_path}")"
if [[ ! -x "${policy_uv_env_path}/.venv/bin/python" ]]; then
    echo "Configured provider environment is missing: ${policy_uv_env_path}/.venv/bin/python" >&2
    exit 1
fi
source "${policy_uv_env_path}/.venv/bin/activate"
PYTHON_BIN="$(command -v python)"

exec env \
    PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=ignore::UserWarning \
    PYTHONPATH="${BENCH_ROOT}:${policy_uv_env_path}/src" \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    "${PYTHON_BIN}" "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            bench_name="${bench_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            env_cfg_type="${env_cfg_type}" \
            seed="${seed}" \
            policy_name="${policy_name}" \
            action_type="${action_type}" \
            action_dim="${action_dim}"
