# 源码完整性核对（2026-10-19）

本次发布的父提交是 `8442593eca942af5a0f610dd482f4dcbcbbdc705`。该版本的 `source/` 仅包含 AwomoPaw 新增文件；QwenPaw 原有文件和修改后的文件需要从固定上游版本和补丁组装，README 为空。

现在 `source/` 保存组装后的完整框架：

- 固定 QwenPaw 上游 `e6e0dcd85b11212d3bf945a724adc295baa03902` 的 1,472 个文件路径全部存在。
- 合入已有 AwomoPaw 补丁和 49 个新增文件，`source/` 总计 1,521 个文件。
- `source/src/qwenpaw/` 总计 802 个文件，包括基础模块和 AwomoPaw 扩展；保留原 Python 包名。
- 包含控制台、上游网站与文档、依赖描述、锁文件、安装和部署代码、测试与许可证。
- 同步本地近期的 open 任务评测、模型身份校验、跨 seed 记忆、视频复盘和结果查看器；部署脚本对本仓库已有模块使用本地导入名。

框架来源是已发布的 P6000 pre-experiment 框架，加上固定 QwenPaw 上游。保留既有公开打包适配：仿真 embodiment 包、外部 primitive provider 配置以及对应测试 fixture。未用本机私有 primitive 路径覆盖这些配置。上游 README 与本机版本只有结尾换行差异。原有补丁仍可追溯修改，安装时无需再次应用。

## 验证

- 61 项评测、学习、动作 worker、视频/轨迹查看器测试通过。
- 73 项具身执行、编排、RoboDojo adapter 和模型兼容测试通过；2 项需要独立 JetRover 硬件包的测试不在仿真发布的验证范围内。
- 发布准备阶段的全部 618 个 Python 文件通过语法解析；随后新增的清单校验脚本另行验证。
- `scripts/apply_awomopaw.py` 在干净固定版本上组装后，逐文件比对 `source/`。
- `SOURCE_MANIFEST.json` 为除自身以外的每个发布文件记录 SHA-256 和字节数。可运行：

```bash
python scripts/verify_source_manifest.py
```

这些验证确认源码、组装和相关回归行为；没有启动仿真实验，也没有验证新机器上的完整 CUDA/仿真环境。

## 外部依赖

RoboDojo/Isaac、模型权重、凭据、CUDA/感知/IK 依赖、独立真机驱动和实验输出不随源码提交。代码策略的感知/运动 primitive provider 仍由部署环境提供，接口与配置见 `runtime.md`。未重新引入 PhysicalRSI 项目或第三方任务技能。完整性指这里明确列出的框架与接入源码范围。
