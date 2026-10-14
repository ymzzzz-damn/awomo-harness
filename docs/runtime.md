# AwomoPaw 运行配置

组装后的基础框架提供 agent/provider API；本仓库提供具身扩展。将 `robodojo-overlay/` 应用到独立 RoboDojo 安装，配置 `robodojo_sim` embodiment 的 bridge 地址和 provider 注册表。仿真器负责 reset、步进、动作验证和评分。

生成 Python policy 的 worker 是可选路径。其底层感知与运动 primitive 由外部环境实现，本仓库不捆绑第三方实现。需要显式设置：

- `ROBODOJO_CODE_POLICY_RUNTIME`：本仓库 `runtime/awomopaw-code-policy` 的绝对路径。
- `ROBODOJO_CODE_POLICY_PYTHON`：安装了 NumPy、Pillow、OpenCV 和 primitive provider 的 Python。
- `ROBODOJO_PRIMITIVES_PATH`：外部 primitive provider 的模块目录。
- `ROBODOJO_PRIMITIVES_MODULE`：模块名，默认 `awomopaw_primitives`。
- `ROBODOJO_CAMERA_CALIBRATION`：包含 `K` 内参与 `T` 外参矩阵的 JSON 文件。
- `ROBODOJO_SUBTASK_MAX_ACTIONS`：动作片段上限；task-artifacts 开发流程默认 40。

外部模块需导出 `PrimitiveClient`、`Robot`、`Sam3TextSegmenter`、`plain`。`PrimitiveClient(episode)` 提供控制契约中声明的动作方法，并通过 `episode.emit(action)` 执行；`Robot()` 提供其所需的运动学后端；`Sam3TextSegmenter().segment(rgb, text, confidence_threshold=...)` 返回 masks 与 scores；`plain(value)` 将数组等转换为 JSON 值。接口调用位置可见 `runtime/awomopaw-code-policy/worker.py`。缺少这些配置时不能启动代码控制。

任务蒸馏读取公开视频、执行代码、日志和官方结束结果；在线控制只读取公开画面和机器人状态。部署后仍需进行实际端到端验证。
