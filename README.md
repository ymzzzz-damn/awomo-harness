# AwomoPaw 启动指南

## 1. 安装框架

推荐 Linux、Python 3.10。运行仿真还需安装 RoboDojo/Isaac、官方场景资产、匹配的 NVIDIA 驱动、Conda 和 FFmpeg（包含 `ffprobe`）；仿真器安装见 [环境配置](docs/runtime.md#2-准备独立的-robodojoisaac-环境)。

```bash
git clone https://github.com/ymzzzz-damn/awomo-harness.git
cd awomo-harness
PYTHON_BIN=python3.10 bash scripts/install.sh
source .venv/bin/activate
```

后续命令均在仓库根目录、激活 `.venv` 后执行。

## 2. 配置 GPT

初始化配置，注册支持图片和 Responses API 的模型服务，并按提示输入 API Key。已有配置时可跳过对应步骤。

```bash
awomopaw init --defaults
awomopaw models add-provider gpt6 --name 'GPT Gateway' \
  --base-url https://ai.zhimiaonengzhi.com/v1
awomopaw models config-key gpt6
awomopaw models add-model gpt6 --model-id gpt-6-astra --model-name gpt-6-astra
```

使用其他服务时替换 `--base-url`；运行时的 `--provider` 和 `--model` 须与实际配置、服务支持的模型一致。API Key 只在本机交互输入。

## 3. 接入 RoboDojo

将下面两个路径替换为已安装的 RoboDojo 目录和其独立仿真环境的 Python：

```bash
export ROBODOJO_ROOT=/absolute/path/RoboDojo
export SIM_PYTHON=/absolute/path/RoboDojo-env/bin/python

bash scripts/install_sim_transport.sh
export ROBODOJO_CLIENT_PYTHONPATH="$PWD/runtime-state/sim-transport"

python scripts/configure_robodojo.py \
  --robodojo-root "$ROBODOJO_ROOT" \
  --transport-python "$PWD/.venv/bin/python" \
  --output "$PWD/runtime-state/demo" \
  --task align_blocks --seed 0 --layout 0
```

接入命令只需执行一次；`--output` 须使用尚不存在的目录。若 Conda 不在 `PATH`，先设置 `ROBODOJO_CONDA_BIN_DIR=/absolute/path/miniconda3/bin`；仿真 Conda 环境名称默认是 `RoboDojo`，其他名称用 `ROBODOJO_EVAL_ENV` 指定。

## 4. 启动 GPT 直控与学习循环

先检查配置和模型图片接口（会调用一次模型 API）：

```bash
python scripts/run_learning_loop.py \
  --sim-root "$ROBODOJO_ROOT" \
  --registry "$PWD/runtime-state/demo/registry.json" \
  --output "$PWD/runtime-state/align-run" \
  --task align_blocks --seeds 0 --layouts 0 1 \
  --provider gpt6 --model gpt-6-astra --gpu 0 \
  --dry-run --probe-model
```

检查通过后，正式启动：

```bash
python scripts/run_learning_loop.py \
  --sim-root "$ROBODOJO_ROOT" \
  --registry "$PWD/runtime-state/demo/registry.json" \
  --output "$PWD/runtime-state/align-run" \
  --task align_blocks --seeds 0 --layouts 0 1 \
  --provider gpt6 --model gpt-6-astra --gpu 0 --resume
```

入口会自动启动仿真桥接和 agent。成功与失败任务均进行视频复盘；成功任务生成 `SKILL.md`、`memory.json`、`task-artifact.json`，供后续同任务场景使用。

按需修改 `--task`、`--seeds`、`--layouts` 和 `--gpu`。运行状态、视频与学习产物保存在 `runtime-state/align-run/`。同一轮续跑使用原命令和 `--resume`；新一轮更换 `--output`。新开终端时须重新激活 `.venv` 并设置第 3 步的环境变量。

## 5. 可选：启动 Web 控制台

安装 Node.js/npm 后，首次构建前端：

```bash
cd source/console
npm ci --include=dev
npm run build
mkdir -p ../src/qwenpaw/console
cp -R dist/. ../src/qwenpaw/console/
cd ../..
```

选择默认模型并启动服务：

```bash
awomopaw models set-llm
awomopaw app --host 127.0.0.1 --port 8088
```

浏览器打开 <http://127.0.0.1:8088>。命令行学习循环可独立运行，无需启动 Web 控制台。
