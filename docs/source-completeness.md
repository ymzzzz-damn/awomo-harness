# 当前源码与部署验证（2026-10-19）

固定 QwenPaw 上游基础文件均保留；AwomoPaw 修改与 RoboDojo 直接控制接入均在仓库内。当前 source/ 共 1520 个文件，qwenpaw 包共 801 个文件。

按本次要求移除 Python 代码策略 worker、code_policy_runtime 和外部 primitive provider 接口，未引入 SAM3/CuRobo 的自定义封装。GPT 使用公开 RGB、关节状态和 action_spec 生成 1–40 步显式关节目标，通过原生动作接口执行，保留评分、步数限制和执行后监督。

打包修复：主 wheel 包含 embodiment 加载器、RoboDojo Markdown/YAML 描述文件；独立 embodiment wheel 也包含这些资源；删除依赖工作目录的 file:./embodiments 依赖。补入生命周期进程 launcher，提供安装、仿真接线、依赖隔离和 smoke 脚本。旧 patches/ 文件仅用于追溯早期版本。

验证：141 项回归测试通过，2 项独立 JetRover 硬件包测试排除。固定 QwenPaw 上游 1472 个文件逐项存在。主包与独立 embodiment 包均成功构建，空白资源测试环境安装主 wheel 可加载随包的 16 个仿真工具与描述文件。完整依赖安装修复了 ReMe/NumPy 约束冲突和 AgentScope/MCP 主版本不兼容；洁净安装结果与依赖快照见 deployment-validation.json 和 requirements/linux-py310-tested.txt。

在隔离的 RoboDojo checkout 中使用现有 Isaac/官方资产完成原生 200 步关节保持测试、4 次非终止片段三路画面监督，并写出官方结果和三路视频；视频实际解码后抽取 15 张带时间戳的画面。该测试未调用 GPT，不证明模型操作任务成功。

通用学习入口 run_learning_loop.py 支持同任务多场景、跨 seed、失败/成功视频复盘、成功三件套蒸馏、后续技能完整读取及恢复。成功分支与复用通过受控官方结果和模型响应夹具验证；已找到对应接口的本机加密 provider 配置，真实 GPT 图片预检和原生三路视频的 15 帧复盘已通过；两场真实 GPT 直控及后续复用验证仍在运行。真实控制暴露的双臂必填字段与异常后监督卡住问题已修复并增加回归测试。配置可用 provider 后，应先按 runtime.md 执行 --dry-run --probe-model，再运行正式循环。

完整记录见 deployment-validation.json；环境配置和外部二进制依赖见 runtime.md。它们仍需按文档准备，不能把源码 clone 等同于已安装 Isaac、GPU 驱动或已配置 GPT 凭据。

SOURCE_MANIFEST.json 记录除清单自身外每个发布文件的 SHA-256。运行 python scripts/verify_source_manifest.py 核对。
