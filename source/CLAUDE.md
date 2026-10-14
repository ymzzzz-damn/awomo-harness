# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

QwenPaw is a personal AI assistant platform (Python 3.10–3.13) accessible via multiple chat channels (DingTalk, Feishu, WeChat, Discord, Telegram, etc.) with a React/TypeScript web console, extensible skills, scheduled tasks, and multi-agent collaboration. The backend uses FastAPI + AgentScope; the frontend is Vite + React 18 + Ant Design.

## Common Commands

### Frontend (console/)
```bash
cd console && npm ci && npm run build     # Build frontend
cd console && npm run dev                 # Dev server
cd console && npm run lint                # ESLint
cd console && npm run format              # Prettier
```

### Python
```bash
pip install -e ".[dev,full]"             # Install (editable)
pre-commit run --all-files               # Format + lint (black, flake8, pylint, mypy, prettier)
```

### Testing
```bash
pytest tests/unit -v --tb=short          # All unit tests
pytest tests/unit -x -q --tb=line        # Fast feedback
pytest tests/unit/agents/hooks tests/unit/agents/memory tests/unit/agents/utils -v --tb=long -s  # Gate tests (must pass for PRs)
pytest tests/contract -v                 # Contract/interface tests
pytest tests/unit --cov=src/qwenpaw --cov-report=html  # Coverage (min 30%)
```

### Run the app
```bash
# First-time setup (after building console)
mkdir -p src/qwenpaw/console && cp -R console/dist/. src/qwenpaw/console/
qwenpaw init --defaults
qwenpaw app                              # Starts on http://127.0.0.1:8088
```

### Docker
```bash
bash scripts/docker_build.sh qwenpaw:latest
docker-compose up -d
```

## Architecture

### Source layout
```
src/qwenpaw/
├── agents/         # Core agent logic
│   ├── react_agent.py      # QwenPawAgent (extends ReActAgent from AgentScope)
│   ├── skills/             # Built-in skills (each a subdirectory)
│   ├── memory/             # Long-term memory, reflection, proactive delivery
│   └── tools/              # 20+ built-in tools (browser, file, shell, delegation)
├── app/
│   ├── _app.py             # FastAPI entry point + WebSocket
│   ├── channels/           # One subdirectory per channel (DingTalk, Feishu, WeChat, …)
│   ├── crons/              # APScheduler-based scheduled tasks
│   └── agent_config_watcher.py  # Hot-reload on config change
├── cli/            # Lazy-loading CLI commands (qwenpaw init/app/agents/skills/…)
├── providers/      # Pluggable LLM provider adapters (DashScope, OpenAI, Claude, …)
├── security/
│   ├── tool_guard/         # Blocks dangerous shell commands before execution
│   ├── skill_scanner/      # Pre-install scan for injection/exfiltration
│   └── file_access_guard/  # Restricts access to ~/.ssh and system paths
└── console/        # Compiled frontend output (NOT committed; copied from console/dist)
```

### Key design patterns

**ReAct loop**: `QwenPawAgent` in `react_agent.py` follows Think → Tool Use → Observe → Reflect, extended with memory and proactive delivery hooks.

**Multi-agent isolation**: Each agent has its own YAML config, memory store, and tool set. `MultiAgentManager` in `app/_app.py` manages independent agent instances.

**Hot-reload**: `agent_config_watcher.py` watches YAML config files and restarts individual agents without restarting the full app.

**Channel abstraction**: Every channel (`app/channels/<name>/`) inherits from `BaseChannel` and implements `async send_message()` / `async receive_message()`. Registered in `channels/registry.py`.

**Skill auto-discovery**: Skills in `agents/skills/<name>/` (or `~/.qwenpaw/working/skills/`) are auto-loaded. Each skill is a directory with `__init__.py` (tool functions decorated `@tool`) and `skill.yaml`.

**Security layers**: Tool Guard (blocklist patterns) → File Access Guard → Skill Scanner (pre-install) → optional web auth (`QWENPAW_AUTH_ENABLED=true`).

### Runtime working directory
```
~/.qwenpaw/working/
├── config/agents/     # Per-agent YAML configs
├── skills/            # User-installed custom skills
├── memory/            # Vectorized long-term memory
└── chats/             # Chat history

~/.qwenpaw/working.secret/
└── secrets.yaml       # API keys (kept separate from config)
```

### Key environment variables
```
DASHSCOPE_API_KEY      # Qianwen/DashScope (primary LLM)
OPENAI_API_KEY
ANTHROPIC_API_KEY
TAVILY_API_KEY         # Web search
QWENPAW_PORT=8088
QWENPAW_AUTH_ENABLED=false
QWENPAW_WORKING_DIR    # Override default working dir
```
Legacy `COPAW_*` vars are auto-converted to `QWENPAW_*`.

## Commit Style

Conventional Commits: `<type>(<scope>): <subject>`
Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`
Example: `feat(channels): add Slack integration`

## CI Gate

The following test directories must pass for every PR:
- `tests/unit/agents/hooks`
- `tests/unit/agents/memory`
- `tests/unit/agents/utils`

CI matrix: Python 3.10 and 3.13.
