---
name: robodojo-vla-control
description: "在 RoboDojo/Isaac 仿真中用 GPT 规划子任务，优先路由到可插拔 VLA/WAM 生成 action，并在模型无法继续时通过同一 embodiment 执行受限的 GPT direct-action。用于交互评测、策略切换、失败回退和新模型接入。"
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
- GPT direct-action 只在 provider 无法继续时生成短 action chunk。
- embodiment 校验、缓存并执行所有来源的 action，推进 Isaac，并产生下一 observation。
- RoboDojo 独占 reset、seed、reward、success、评分、视频和正式结果。

架构与进程所有权见 [references/architecture.md](references/architecture.md)。

## 交互控制循环

1. 调用 `robodojo_sim.health`，从 `available_policies` 选择 `ready=true` 的
   `policy_id`。默认使用 `pi05`，除非任务或用户指定另一个已注册 provider。
2. 调用 `robodojo_sim.start_session`。同一仿真 GPU 不并发启动第二个 session。
3. 轮询 `robodojo_sim.session_status`，直到初始 observation 可用或 session 失败。
4. 调用 `robodojo_sim.current_observation`，查看 `robot_state`、`action_spec`
   和新的 `observation_id`。再按任务需要调用 `capture_head_frame`、
   `capture_left_wrist_frame` 或 `capture_right_wrist_frame`，让 planner 真正
   看到当前仿真图像；不得只凭文件名推测场景。
5. 维护短而明确的 task plan；每次只选择当前 subtask。把当前
   `observation_id`、`policy_id`、subtask 和针对该 provider 的 prompt 交给
   `robodojo_sim.request_policy_chunk`。
6. 得到 chunk id 后调用 `robodojo_sim.execute_action_chunk`。不要在执行前
   请求第二个 chunk，也不要自行修改 provider 产出的 action。
7. 在新的 observation boundary 判断 subtask 是否完成、是否有进展、是否要
   改写 prompt 或重规划。episode 未结束则回到第 4 步。
8. episode 结束后再读取 `evaluation_result`；保存 trace 和 artifact 路径。

详细 action 不变量见 [references/action-contract.md](references/action-contract.md)。

## Provider 失败与 GPT 接管

以下信号允许进入回退：provider timeout/unavailable、空或非法 chunk、明确
unsupported/OOD、连续 observation 在有限预算内无视觉进展。一次含糊失败可先
修订 subtask/prompt；不要盲目重复同一请求。

回退时重新调用 `current_observation`，严格按它返回的动态 `action_spec` 生成
1 个 absolute-joint action，连同当前 `observation_id`、subtask 和具体
`fallback_reason` 交给 `submit_direct_action_chunk`，然后仍使用
`execute_action_chunk`。GPT 不得猜测未公布的维度、复用旧 observation、产生
NaN/Inf，或把 direct-action 当作默认高频 policy。

无法从公开 observation 建立安全进展时调用 `abort_session`，保留 trace，停止。
回退判定和停止条件见 [references/failure-fallback.md](references/failure-fallback.md)。

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
