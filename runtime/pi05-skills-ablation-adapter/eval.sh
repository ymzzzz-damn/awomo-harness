#!/usr/bin/env bash
# Marker used by scripts/robodojo.sh; orchestration is provided by the shared
# setup_eval_policy_server.sh and setup_eval_env_client.sh entry points.
set -euo pipefail
exec bash "$(dirname "$0")/../../../scripts/robodojo.sh" eval --policy-dir "$(dirname "$0")" "$@"
