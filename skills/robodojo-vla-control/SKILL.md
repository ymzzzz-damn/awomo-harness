---
name: robodojo-vla-control
description: "在 RoboDojo/Isaac 仿真中用 GPT 规划子任务，优先路由到可插拔 VLA/WAM 生成 action，每个执行窗口后提交 reflection，偏离时立即通过同一 embodiment 由 GPT 接管。用于交互评测、策略切换、失败回退和新模型接入。"
metadata:
  qwenpaw:
    emoji: "🦾"
    requires: {}
---

# RoboDojo 混合策略控制

只用于仿真。`robodojo_sim` embodiment 是 Isaac 的唯一 observation/action
边界；GPT planner、VLA、WAM 都不得绕过它操作仿真。

## 责任分层

- GPT（System 2）观察公开图像与机器人状态，规划或修订一个语义 subtask。
- 已注册 VLA/WAM（System 1）优先把 subtask 和 observation 变成 action chunk。
- 每个 action chunk 执行窗口后由 GPT 提交 reflection，决定继续 VLA/WAM 或立即接管。
- GPT 在明确偏离或 provider 失败时生成单步 direct-action 或新的 code-policy。
- embodiment 校验、缓存并执行所有来源的 action，推进 Isaac，并产生下一 observation。
- RoboDojo 独占 reset、seed、reward、success、评分、视频和正式结果。

架构与进程所有权见 [references/architecture.md](references/architecture.md)。

## 交互控制循环

1. 调用 `robodojo_sim.health`，监督评测须返回 `chunk_supervision.required=true`；
   缺失时报告旧 bridge 尚未启用监督协议，不得声称逐 chunk 监督已生效。
   从 `available_policies` 选择 `ready=true` 的
   `policy_id`。默认使用 `pi05`，除非任务或用户指定另一个已注册 provider。
2. 调用 `robodojo_sim.start_session`。同一仿真 GPU 不并发启动第二个 session。
3. 轮询 `robodojo_sim.session_status`，直到初始 observation 可用或 session 失败。
4. 调用 `robodojo_sim.current_observation`，读取 benchmark 返回的完整
   `instruction`，作为本次任务的权威目标；缺失时停止并报告输入链路错误，
   不得从任务名或图片猜测任务，不得自行补写任务目标。查看 `robot_state`、`action_spec`
   和新的 `observation_id`。再按任务需要调用 `capture_head_frame`、
   `capture_left_wrist_frame` 或 `capture_right_wrist_frame`，让 planner 真正
   看到当前仿真图像；不得只凭文件名推测场景。
5. 维护短而明确的 task plan；每次只选择当前 subtask。子任务和给 provider 的
   prompt 必须保留官方 `instruction` 指定的操作方式、工具、对象、顺序与收尾要求。
   子任务由当前 LLM 从原文和公开观测生成，不在本 skill 中预置具体任务解法。把当前
   `observation_id`、`policy_id`、subtask 和针对该 provider 的 prompt 交给
   `robodojo_sim.request_policy_chunk`。
6. 得到 chunk id 后调用 `robodojo_sim.execute_action_chunk`，设置
   `max_actions=10`（需要更细反馈时用 5）、`discard_remaining=true`。
   embodiment 只执行原始 action 的前缀并丢弃旧尾部，不改写动作值。
   不要在执行前请求第二个 chunk。
7. 每个执行窗口后读取新 observation 并查看三路相机，与执行前图像和
   `supervision.last_execution` 比较。调用 `reflect_action_chunk`，传入当前
   `observation_id`、`supervision.chunk_id`、`verdict` 和具体 `reflection`。
   `continue` 表示正常进展，可继续 VLA/WAM；`correct` 表示明确错误，立即
   GPT 接管；`uncertain` 表示证据不足，重新观察或停止。reflection 要写出
   观察事实、可能原因和不确定性，不能把正常的中间动作当作失败。
   bridge 不允许遗漏 reflection 后请求新 chunk。单步 GPT 纠正后也要复查，
   恢复正常时再返回 VLA/WAM。不要重用旧尾部。完整 chunk 仅用于纯 VLA 基线。
   terminal chunk 后原生 control server 会关闭，其 reflection 转入任务结束后的
   视频反馈，不再请求在线动作。
8. episode 结束后再读取 `evaluation_result`；保存 trace 和 artifact 路径。

详细 action 不变量见 [references/action-contract.md](references/action-contract.md)。

## Provider 失败与 GPT 接管

一次公开反馈已经明确表明偏离 subtask 时，提交 `correct` reflection 后即可
立即切换到 GPT 单步纠正或 Python code-policy，不要求先重试或改 prompt。
先调用 `code_policy_api` 获取通用视觉/IK/移动接口；由当前 LLM 根据官方原文、
公开图像、机器人状态和失败轨迹生成 `def policy(robot, memory)`，再调用
`run_code_policy`，传入当前 observation id、源码、memory 和具体失败依据。
不得加载预置的任务专项代码，不得人工修改模型生成的任务策略。

这个分支持续执行模型生成的程序，期间不再调用 VLA；每个 primitive action
仍通过相同 embodiment 校验、执行并复查。`home()` 使用 episode 初始状态。
用 `code_policy_status` 查看执行状态和源码 hash；worker 返回不代表任务成功。
失败后保存代码与公开执行轨迹，让 LLM 修订下一版；episode 已结束则明确记录
新的尝试。原有的单步 direct-action 回退仍可用于短暂纠正。

以下信号允许进入回退：provider timeout/unavailable、空或非法 chunk、明确
unsupported/OOD、连续 observation 在有限预算内无视觉进展。一次含糊反馈提交 `uncertain`，重新观察；明确错误不能靠继续 VLA 重试绕过接管。

回退时重新调用 `current_observation`，严格按它返回的动态 `action_spec` 生成
1 个 absolute-joint action，连同当前 `observation_id`、subtask 和具体
`fallback_reason` 交给 `submit_direct_action_chunk`，然后仍使用
`execute_action_chunk`。GPT 不得猜测未公布的维度、复用旧 observation、产生
NaN/Inf，或把 direct-action 当作默认高频 policy。

无法从公开 observation 建立安全进展时调用 `abort_session`，保留 trace，停止。
回退判定和停止条件见 [references/failure-fallback.md](references/failure-fallback.md)。

## 任务结束后的自进化

任务执行监督与任务后自进化是两个闭环。结束后由接入的 LLM 看完整任务的
官方结果、三路视频的时间序列采样、每个 chunk 的 reflection 和实际策略，
先诊断成功机制或失败原因，再蒸馏成 task skill 文本、code-policy 和 memory
候选。成功经验也要保留，不只分析失败。每个候选记录 parent revision、
视频/frame hash、反馈 hash 和 validation plan，旧版本保持不变。

本仓库的 `scripts/run_supervised_task.py --evolve` 自动运行两条链路；
`scripts/evolve_task_artifact.py` 可单独对已结束的 development run 做自进化。
新 `task-artifact.json` 可用 `--task-artifact` 显式进入下一次 development run。
加 `--rounds N --evolve` 会自动把每一轮蒸馏的候选交给下一轮，并保存 lineage。
候选的 skill 用于规划，代码可由 LLM 在接管时修订并通过相同 executor 执行。
验证与最终评测不能回流到蒸馏；正式版本选择仍须独立 Self-Harness 比较。
这里更新 skill、代码和 memory，不更新 VLA/WAM/LLM checkpoint 权重。

## 接入其他 VLA/WAM

模型特有的相机映射、state 编码、归一化、checkpoint 和运行环境都留在独立
XPolicyLab adapter 中。注册后必须实现与 Pi0.5 相同的
observation-to-action-chunk 能力；不修改本 skill 的控制循环，也不直接调用
Isaac。接入时读取 [references/new-adapter.md](references/new-adapter.md)。Pi0.5
细节见 [references/pi05-adapter.md](references/pi05-adapter.md)。

## 不可违反的边界

- 不接入或调用 ROS、FCI、USB、真机机械臂或真机相机。
- episode 中不读取 reward、success、ground-truth object pose 或排行榜元数据。
- 不绕过 embodiment 的 stale-id、shape、finite-value、horizon 和 episode 校验。
- 不改变 seed，不覆盖官方结果，不在失败后静默启动新 run。
- 不把 API key、模型凭据或集群凭据写入 skill、trace 或 prompt。
