# AwomoPaw

AwomoPaw 是我们开发的具身智能 harness：通过统一的工具与动作执行边界连接 RoboDojo，让模型根据当前画面规划、执行、反思，并将任务经验用于下一次策略生成。

本仓库保存 AwomoPaw 新增代码，以及我们对基础框架的修改补丁。上游基础框架作为外部依赖，不在仓库中复制完整源码。Python 中的 `qwenpaw` 命名空间用于兼容现有运行环境；项目与命令名称为 `awomopaw`。

## 我们的实现

- 具身工具加载、执行队列、审批与 orchestration，以及前端 Embodiments 页面。
- RoboDojo simulation adapter、交互式 episode 接口与模型生成的 Python policy 执行边界。
- 按 subtask 或动作片段进行画面检查与 reflection，默认片段上限为 40 个动作。
- 根据成功和失败任务的视频、结果与日志进行复盘；成功任务生成版本化 `SKILL.md`、`memory.json`、`task-artifact.json`，供同 task 后续策略生成使用。

这些是源码功能；实际评测需要另行部署，不能据此宣称任务成功率提升。

## 目录

| 路径 | 内容 |
| --- | --- |
| `source/` | 我们新增的 Python 模块、前端页面、具身运行时和测试；是覆盖层 |
| `patches/awomopaw.patch` | 对上游已有文件的修改 |
| `robodojo-overlay/` | RoboDojo 的交互接口与 AwomoPaw policy 接入 |
| `scripts/` | 执行、片段监督、视频复盘、经验蒸馏和开发评测 |
| `runtime/` | 生成代码的执行契约、worker 与 episode transport |
| `embodiments/`、`skills/` | 仿真接口定义与控制流程 |
| `docs/ownership.json` | 上游版本与新增、修改文件清单 |

## 组装开发环境

外部基础依赖为 [QwenPaw](https://github.com/agentscope-ai/QwenPaw)，固定版本 `e6e0dcd85b11212d3bf945a724adc295baa03902`。先在独立目录取得这一版本，再运行：

```bash
python scripts/apply_awomopaw.py /absolute/path/to/upstream-checkout
cd /absolute/path/to/upstream-checkout
python -m pip install -e '.[dev,embodiments]'
```

组装脚本只接受该版本的干净 checkout，先检查补丁，再应用修改并复制新增文件。不要直接对 `source/` 执行 `pip install`。

RoboDojo、Isaac、VLA/WAM、分割与 IK 服务、相机标定和模型凭据均由部署环境提供。生成代码 worker 的外部 primitive 接口见 [运行配置](docs/runtime.md)。普通 agent 与具身工具模块不要求启用该 worker。

## 视频反馈与技能

```bash
python scripts/gpt6_task_learning.py --help
python scripts/evaluate_gpt6_self_evolution.py --help
```

`gpt6_task_learning.py` 处理归档的开发任务；评测入口需要已有 baseline 清单、注册表、RoboDojo 安装与 GPU 环境，目前限定原开发节点及 `align_blocks` seed 0。它不是通用的一键部署入口。成功技能必须具有同 task 的成功来源记录；失败反馈不会覆盖已有成功技能。

仓库不包含第三方完整项目、预训练任务技能、模型权重、模拟器资产、运行环境或实验结果。保留 Apache 许可证与上游归属说明。
