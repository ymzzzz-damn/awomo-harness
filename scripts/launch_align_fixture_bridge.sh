#!/usr/bin/env bash
set -euo pipefail
ROBODOJO_SKILL_REGISTRY=/home/ymz/Documents/ChatGPT/robodojo/runtime/physicalrsi-align-skill/registry.json
export ROBODOJO_POLICY_REGISTRY_JSON="$(/home/ymz/physclaw/awomopaw/.venv/bin/python -m json.tool "$ROBODOJO_SKILL_REGISTRY")"
cd /home/ymz/physclaw/awomopaw
exec .venv/bin/python /home/ymz/Documents/ChatGPT/robodojo/scripts/chat_eval_bridge.py --fixture /home/ymz/Documents/ChatGPT/robodojo/runtime/physicalrsi-align-skill/chat-eval-fixture.json
