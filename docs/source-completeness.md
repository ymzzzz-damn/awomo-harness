# 当前源码与部署验证（2026-10-19）

固定 QwenPaw 上游基础文件均保留；AwomoPaw 修改与 RoboDojo 直接控制接入均在仓库内。当前 source/ 共 1520 个文件，qwenpaw 包共 801 个文件。

按本次要求移除 Python 代码策略 worker、code_policy_runtime 和外部 primitive provider 接口，未引入 SAM3/CuRobo 的自定义封装。GPT 使用公开 RGB、关节状态和 action_spec 生成 1–40 步显式关节目标，通过原生动作接口执行，保留评分、步数限制和执行后监督。

打包修复：主 wheel 包含 embodiment 加载器、RoboDojo Markdown/YAML 描述文件；独立 embodiment wheel 也包含这些资源；删除依赖工作目录的 file:./embodiments 依赖。补入生命周期进程 launcher，提供安装、仿真接线、依赖隔离和 smoke 脚本。旧 patches/ 文件仅用于追溯早期版本。

验证：127 项回归测试通过，2 项独立 JetRover 硬件包测试排除。主包与独立 embodiment 包均成功构建；空白 venv 安装主 wheel 后可加载随包的描述文件与直接控制工具（此项只验证资源打包，没有安装全部 agent 依赖）。在隔离的 RoboDojo checkout 中使用现有 Isaac/官方资产完成了原生 200 步关节保持测试并保存官方结果。该测试没有调用 GPT，不证明模型任务成功率。

完整记录见 deployment-validation.json；环境配置和外部二进制依赖见 runtime.md。它们仍需按文档准备，不能把源码 clone 等同于已安装 Isaac、GPU 驱动或已配置 GPT 凭据。

SOURCE_MANIFEST.json 记录除清单自身外每个发布文件的 SHA-256。运行 python scripts/verify_source_manifest.py 核对。
