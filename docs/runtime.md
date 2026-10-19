# 安装与 RoboDojo 直接关节控制

当前控制路径：三路 RGB + 关节状态 → GPT 生成绝对关节目标 → 有界动作块 → RoboDojo 执行与评分。已移除代码策略 worker，以及它的 SAM3/CuRobo primitive 接口。不需要为 GPT 路径安装分割模型、额外 IK 库或 VLA 权重。RoboDojo 自身仍可能使用其原生运动学依赖。

## 1. 安装 AwomoPaw

建议 Linux；本次框架洁净安装使用 Python 3.10，独立仿真环境使用 Python 3.11。框架声明支持 Python 3.10–3.13，其余版本未逐一做洁净环境验证。

```bash
git clone https://github.com/ymzzzz-damn/awomo-harness.git
cd awomo-harness
PYTHON_BIN=python3.10 bash scripts/install.sh
source .venv/bin/activate
awomopaw --help
python scripts/verify_source_manifest.py
```

安装脚本同时安装 `[simulation,transport]`，包括 h5py、OpenCV、WebSocket 和消息序列化依赖。ReMe 要求 NumPy >=2.2.6，因此框架传输环境使用 NumPy 2 和支持它的 OpenCV，不能套用旧的 numpy<2 限制。Isaac 的 Python 与此环境独立。当前 AgentScope 固定版本使用 MCP 1 接口，因此还限制 mcp<2、fastmcp<3，避免新主版本导致 agent 启动失败；ACP 客户端限制在 0.9 系列，保留当前 agent 需要的 SetSessionModelResponse 接口。

Linux x86_64 / Python 3.10 安装会自动使用 `requirements/linux-py310-tested.txt` 中实际验证的依赖版本。其他环境按 pyproject 约束解析。需要指定另一份约束时可设置 `INSTALL_CONSTRAINTS=/absolute/path/constraints.txt`。

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

## 5. 运行完整学习循环

使用通用入口 `scripts/run_learning_loop.py`。它自行启动并清理每场任务的桥接和 agent；无需旧 baseline、固定机器 IP 或两块 GPU。第 3 节只需完成接线，运行此入口时不用另外启动手动桥接。

先配置支持图片与 Responses API 的 GPT provider，例如 `awomopaw models config-key openai`（也可执行 `awomopaw models config` 交互配置）。凭据由本机 AwomoPaw 管理。若使用自定义配置目录，执行以下命令前设置 `AWOMOPAW_WORKING_DIR` 和 `AWOMOPAW_SECRET_DIR`，与配置 provider 时一致。

使用自定义兼容接口时，可按下面的方式注册；已有同名 provider 时复用现有配置即可。`config-key` 交互输入的密钥保存在本机，不写入运行产物或 Git。

```bash
awomopaw models add-provider gpt6 --name 'GPT Gateway' \
  --base-url https://ai.zhimiaonengzhi.com/v1
awomopaw models config-key gpt6
awomopaw models add-model gpt6 --model-id gpt-6-astra --model-name gpt-6-astra
```

随后把下面运行命令的 `--provider openai` 改为 `--provider gpt6`。图片预检使用 384×384 PNG，只检查最终文字回答；16×16 缩略图在该兼容接口上曾出现错误识色，不能用于判定模型的仿真视觉能力。

```bash
# 检查场景、Python 依赖、实际 agent 工具，以及一次图片 API 调用。
# 不启动 Isaac，不消耗官方任务动作。
python scripts/run_learning_loop.py \
  --sim-root /absolute/path/RoboDojo \
  --registry "$PWD/runtime-state/demo/registry.json" \
  --output "$PWD/runtime-state/align-development" \
  --task align_blocks --seeds 0 --layouts 0 1 \
  --provider openai --model gpt-6-astra --gpu 0 \
  --dry-run --probe-model

# 使用同一配置正式运行；--resume 复用已通过的预检目录。
python scripts/run_learning_loop.py \
  --sim-root /absolute/path/RoboDojo \
  --registry "$PWD/runtime-state/demo/registry.json" \
  --output "$PWD/runtime-state/align-development" \
  --task align_blocks --seeds 0 --layouts 0 1 \
  --provider openai --model gpt-6-astra --gpu 0 --resume
```

`--provider` 是本机配置好的 provider ID。已有 agent 配置时可用 `--agent-config /absolute/path/agent.json` 代替 `--provider`；入口只读取其中的 `active_model`，建立隔离的运行配置。`--model` 的支持范围可由 `--help` 查看，控制与视频复盘使用同一模型并校验返回身份。模型服务需实际提供所选模型。

每场任务依次完成：

1. 冻结此前同任务经验，GPT 从三路当前 RGB、关节状态和官方指令生成直接动作。
2. 执行最多 `--segment-actions` 步（默认 40，可设 1–40）。非终止片段后必须看三路当前图片并提交子任务反思，才能继续运动。
3. 等待官方评分和三路完整 MP4；校验 task、run ID、layout、结果与视频哈希。
4. **成功与失败均复盘**：默认从每路视频抽取最多 5 帧，保留真实时间戳，并结合公开动作历史、执行记录生成分析、修正建议与任务记忆。
5. **成功时蒸馏** `SKILL.md`、`memory.json`、`task-artifact.json`。失败更新经验；若已有成功技能，则保留技能并更新失败反馈。下一场同任务必须读取完整经验及这三个产物，随后根据当前场景生成策略。

动作提交前会检查整个块的必填字段、维度、有限数值及夹爪范围。双臂机器人每个动作必须包含两臂与两夹爪的目标，未运动的手臂也须明确保持当前值。缺字段的块会在任何动作执行前被拒绝。若原生执行发生异常，返回标明部分执行不确定性的失败回执，释放剩余动作；模型须重新观察、反思后规划，不会自动重放失败动作。已释放或过期的块也不会令子任务监督永久卡住。

这些是上下文技能和记忆，不是模型权重训练。顺序为 seed 列表内的 layout 列表；可用 `--seeds 0 1 --layouts 0 1` 跨 seed 复用，同一个输出目录只对应一个 task。

输出中的 `status.json` 记录阶段和已复盘数量，`results.json` 保存官方结果。每场在 `seed-N/layout-NN/self_evolving/` 下保存三路视频、动作/模型审计、子任务反思；视频分析与成功技能位于 `task-learning/attempt-NNN/`，`task-learning/receipt.json` 指向完成的分析，根目录 `experience.json` 指向最新同任务技能。

再次运行相同正式命令并带 `--resume`，会校验配置、场景、经验和产物哈希。视频复盘接口中断后，只补做未完成的复盘，不重新执行已经评分的任务。未完成或未评分的仿真任务会停止并报告错误，须先检查日志；入口不会静默重跑。成功/失败视频缺失或复盘失败时状态为 `needs_attention`，不会冒充完成或跳到下一场。

复用另一轮已有成功技能时，在新输出目录的命令中增加 `--task-artifact /absolute/path/task-artifact.json`，保留它同目录下的 `SKILL.md` 和 `memory.json`。task 与内容哈希必须匹配。历史 `evaluate_gpt6_self_evolution.py`、open 批处理脚本仍用于原实验约束；通用部署使用本节入口。

移除 IK/分割后，GPT 需要自行从图像和关节反馈生成动作。部署链路可用不代表操作任务能够成功；不要把链路 smoke test 的结果记成模型任务成功率。
