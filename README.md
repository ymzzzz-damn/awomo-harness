# AwomoPaw Harness

本仓库包含完整的 **QwenPaw 基础源码 + AwomoPaw 扩展和修改 + RoboDojo 接入与学习流程**。直接使用 `source/` 即可，无需再下载 QwenPaw 或先应用补丁。

AwomoPaw 的发行包和主命令叫 `awomopaw`，内部 Python 包仍叫 `qwenpaw`，兼容命令 `qwenpaw`、`copaw` 也保留。QwenPaw 部分位于 [`source/src/qwenpaw/`](source/src/qwenpaw/)，不是仅包含新增文件的空壳。

## 代码位置

| 路径 | 内容 |
| --- | --- |
| `source/src/qwenpaw/agents/` | 基础 agent、记忆、工具、技能和提示模板，以及 AwomoPaw 修改 |
| `source/src/qwenpaw/providers/` | 模型供应商、模型路由和 Responses 兼容实现 |
| `source/src/qwenpaw/app/` | API、工作区、渠道、具身执行和编排 |
| `source/src/qwenpaw/cli/` | 命令行入口 |
| `source/console/` | 完整控制台前端源码和依赖锁文件 |
| `source/website/` | QwenPaw 上游网站源码和文档 |
| `source/embodiments/` | 仿真 embodiment 加载器和 RoboDojo 描述文件 |
| `robodojo-overlay/` | RoboDojo 交互 harness 与 Awomopaw_Harness 策略接入 |
| `runtime/gpt6-harness-ablation-adapter/` | GPT 直接关节控制、原生仿真交互与生命周期传输 |
| `scripts/` | 子任务监督、视频复盘、技能蒸馏、跨场景记忆、open 任务评测与结果同步 |
| `ui/trajectory/` | 视频、轨迹、评分和学习过程的查看界面 |
| `tests/`、`source/tests/` | 集成与单元回归测试 |

## 安装框架

使用 Python 3.10–3.13，在仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e "./source[simulation,transport]"
awomopaw --help
```

控制台构建需要 Node.js/npm，可在 `source/console/` 执行 `npm ci --include=dev` 和 `npm run build`，再将 `dist/` 的内容复制到 `source/src/qwenpaw/console/`。也可以使用 `source/deploy/Dockerfile`，其中包含前端构建步骤。生成后的前端产物不提交到仓库。

## 监督与学习

任务执行在 subtask 返回或动作片段达到上限时进行监督，默认每段最多 40 步。成功与失败任务都会进行视频复盘；成功任务蒸馏 `SKILL.md`、`memory.json` 和 `task-artifact.json`，供后续同任务场景生成策略；失败反馈进入记忆，并保留已有成功技能。open 评测脚本保留任务、seed、layout 的独立身份、跨 seed 的记忆顺序及模型身份校验。

部署配置见 [`docs/runtime.md`](docs/runtime.md)。完整性核对见 [`docs/source-completeness.md`](docs/source-completeness.md)，逐文件 SHA-256 清单见 [`SOURCE_MANIFEST.json`](SOURCE_MANIFEST.json)。

通用运行入口是 `python scripts/run_learning_loop.py --help`。配置仿真器与 GPT provider 后，可指定 task、seed、layout 和 GPU；入口自动执行三路视频复盘、成功技能蒸馏和下一场同任务复用，并支持 `--resume`。完整命令见 [运行完整学习循环](docs/runtime.md#5-运行完整学习循环)。

## 源码范围与依赖

本次完整源码以 QwenPaw `e6e0dcd85b11212d3bf945a724adc295baa03902` 为基础，包含已发布的 AwomoPaw 修改，并同步本地近期的评测、学习和查看器代码。上游许可证与声明保存在 `source/LICENSE`、根目录 `LICENSE` 和 `NOTICE`。`patches/awomopaw.patch` 保留用于追溯原有修改；日常安装直接使用 `source/`。

当前 GPT 路径采用“画面和关节状态 → GPT 生成关节动作块 → 仿真执行 → 反馈”的闭环，已移除原代码策略 worker 及其 SAM3/CuRobo primitive 依赖。每块最多 40 步，支持更短执行窗口与执行后反思。任务成功率需要单独评测。

RoboDojo/Isaac 仿真器、官方场景资产、GPT 凭据和独立真机驱动不在本仓库内。GPT 直接控制不需要 SAM3 或 VLA 权重。完整安装与接线步骤见 [docs/runtime.md](docs/runtime.md)。框架 wheel 已包含 embodiment 加载器和描述文件，`embodiments` extra 作为兼容名称保留。
