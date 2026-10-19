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
- GPT 在明确偏离或 provider 失败时生成有界 direct-action 动作块。
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
立即切换到 GPT 关节动作块纠正，不要求先重试或改 prompt。
读取当前三路画面、robot_state 和 action_spec，直接生成 1..40 步关节目标。
使用 submit_direct_action_chunk 提交，再通过 execute_action_chunk 执行并丢弃未执行尾部。
执行后读取最新状态和画面；启用监督时先完成 reflect_action_chunk 再继续动作。
当前没有 Python worker、分割、IK 或笛卡尔运动规划接口。

## 任务结束后的自进化

任务执行监督与任务后自进化是两个闭环。结束后由接入的 LLM 看完整任务的
官方结果、三路视频的时间序列采样、每个 chunk 的 reflection 和实际策略，
先诊断成功机制或失败原因，再蒸馏成 task skill 文本、关节控制策略说明和 memory
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
