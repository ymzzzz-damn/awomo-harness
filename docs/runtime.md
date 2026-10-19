# 安装与 RoboDojo 直接关节控制

当前控制路径：三路 RGB + 关节状态 → GPT 生成绝对关节目标 → 有界动作块 → RoboDojo 执行与评分。已移除代码策略 worker，以及它的 SAM3/CuRobo primitive 接口。不需要为 GPT 路径安装分割模型、额外 IK 库或 VLA 权重。RoboDojo 自身仍可能使用其原生运动学依赖。

## 1. 安装 AwomoPaw

建议 Linux、Python 3.11。框架也支持 Python 3.10–3.13，仿真环境按 RoboDojo 要求使用 Python 3.11。

```bash
git clone https://github.com/ymzzzz-damn/awomo-harness.git
cd awomo-harness
PYTHON_BIN=python3.11 bash scripts/install.sh
source .venv/bin/activate
python -m pip install -e './source[transport]'
awomopaw --help
python scripts/verify_source_manifest.py
```

主安装包已包含 `qwenpaw`、`embodiments` 加载器和 RoboDojo 描述文件，不再要求另装本地 `file:./embodiments` 依赖。`embodiments` extra 保留为兼容名称。独立的 `source/embodiments` 包仍可单独构建，其 wheel 也包含描述文件；通常无需再安装一次。

控制台需要 Node.js/npm：

```bash
cd source/console
npm ci --include=dev
npm run build
mkdir -p ../src/qwenpaw/console
cp -R dist/. ../src/qwenpaw/console/
cd ../..
```

也可用 `source/deploy/Dockerfile` 构建控制台。前端构建产物不是源码清单的一部分。

## 2. 准备独立的 RoboDojo/Isaac 环境

使用 RoboDojo 官方安装与资产下载流程。其 GPU 驱动、Isaac Sim 和 CUDA 版本必须匹配；这些二进制环境与官方场景资产不随本仓库 clone。原生环境通常需数百 GB 磁盘空间。

本次接线使用的源码版本：

| 组件 | 版本 |
| --- | --- |
| RoboDojo | `09bd5f5e762b949f5858ece31347ce62057b3a5e` |
| XPolicyLab | `53c5da5a01bc720d1cf0e6ab6ed212e8cef0eda2` |
| IsaacLab 子模块 | `afca7b09d60d8beb9c1cb28b43066499940b969b` |
| RoboDojo 原生 CuRobo 子模块 | `d17b54ce32cba095c0b000c4c58777075d11de0e` |

```bash
git clone https://github.com/RoboDojo-Benchmark/RoboDojo.git /absolute/path/RoboDojo
git -C /absolute/path/RoboDojo checkout 09bd5f5e762b949f5858ece31347ce62057b3a5e
git -C /absolute/path/RoboDojo submodule update --init third_party/IsaacLab third_party/curobo XPolicyLab
```

然后遵循该版本的 `scripts/install.sh`、`scripts/init_assets.sh` 和 `docker/README.md`。相关系统包包括 FFmpeg/ffprobe、编译工具以及 GPU/Vulkan 运行库。首次使用 Isaac 前应按其官方流程完成许可证与环境配置。

RoboDojo 的 Isaac 依赖可能固定 `websockets==12`，而当前 XPolicyLab 传输需要新 asyncio 接口。不要直接升级现有仿真环境；可建立只供该次仿真客户端使用的依赖目录：

```bash
SIM_PYTHON=/absolute/path/RoboDojo-env/bin/python bash scripts/install_sim_transport.sh
export ROBODOJO_CLIENT_PYTHONPATH="$PWD/runtime-state/sim-transport"
```

该脚本安装 `websockets==15.0.1`、`msgpack-numpy==0.4.8` 到独立目录；NumPy/msgpack 等依赖由原生仿真环境提供。生命周期传输服务器则使用安装了 `[transport]` 的 AwomoPaw Python。

## 3. 将直接控制接到仿真器

选择单独的 RoboDojo checkout 进行接线。此操作会复制当前交互 harness 与 GPT 生命周期 adapter；旧的交互 harness 会备份到输出目录。不会启动任务或修改模型默认设置。

```bash
python scripts/configure_robodojo.py \
  --robodojo-root /absolute/path/RoboDojo \
  --transport-python "$PWD/.venv/bin/python" \
  --output "$PWD/runtime-state/demo" \
  --task align_blocks --seed 0 --layout 0
```

脚本输出 `source .../environment.sh` 和启动桥接的完整命令。依次执行；默认桥接端口 8098、原生动作服务端口 8099、生命周期 WebSocket 端口 8100。可通过 `--bridge-port`、`--control-port`、`--policy-port` 选择空闲端口。若 Conda 不在 PATH，设置 `ROBODOJO_CONDA_BIN_DIR`；若原生环境名称不是 `RoboDojo`，设置 `ROBODOJO_EVAL_ENV`。

当前 fixture 为一次官方 episode 选择 task/seed/layout，控制模型只能读取官方公开指令、画面和状态。查看桥接 `/health` 的 `problems` 后，再启动 session。

## 4. 配置 GPT 并执行

用 `awomopaw init` 或已有 provider 配置设置模型与凭据。凭据只保存在运行机器，不能提交到仓库。把 `robodojo_sim` embodiment 绑定到 `http://127.0.0.1:8098`；模型可使用的直接控制顺序是：

1. `start_session(task, policy_id='gpt6-harness', seed)`，轮询 `session_status`。
2. `current_observation`，并获取三路当前画面。读取返回的 `instruction`、`robot_state`、`action_spec`。
3. 生成符合 `action_spec` 的绝对关节/夹爪目标，提交 `submit_direct_action_chunk`。每块 1–40 步；提交本身不执行动作。
4. `execute_action_chunk(chunk_id, max_actions=1..40, discard_remaining=true)`。
5. 读取新画面与关节反馈，再规划下一块。启用子任务监督时先完成 `reflect_action_chunk`。
6. 原生 episode 结束后，从 `evaluation_result` 读取结果。

项目内的 `run_gpt6_harness_agent.py` 可用于记录配置好的 agent 的运行；它要求一个有效的 `AgentProfileConfig` JSON（包含 active_model 和启用的 robodojo_sim embodiment），可用 `--dry-run` 检查工具加载。评测批处理脚本仍保留原 P6000 节点、库存和基线约束，不应直接作为通用安装入口。

移除 IK/分割后，GPT 需要自行从图像和关节反馈生成动作。部署链路可用不代表操作任务能够成功；不要把链路 smoke test 的结果记成模型任务成功率。
