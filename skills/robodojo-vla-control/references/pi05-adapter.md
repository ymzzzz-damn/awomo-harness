# Pi0.5 adapter

Pi0.5 是 registry 中首个 VLA provider，默认 id 为 `pi05`，运行在 XPolicyLab policy server 中。

## Observation 映射

Pi0.5 adapter 把 RoboDojo observation 转为：

- `state`: 当前机器人 state，`float32`；
- `images.cam_high`: 主视角，CHW `uint8`；
- `images.cam_left_wrist`: 左腕视角，CHW `uint8`；
- `images.cam_right_wrist`: 右腕视角，CHW `uint8`；
- `prompt`: 当前任务语言指令。

原始相机若是 HWC 或浮点图像，由 adapter 做布局和像素类型转换。找不到必需相机时应失败，不能用空图或历史图替代。

## 推理协议

interactive embodiment 通过 msgpack WebSocket 帧把当前 observation 发送给
session 固定的 Pi0.5 server。policy server 调用：

- `update_obs(observation)` 后调用 `get_action()`；或
- `infer(observation)`。

server 返回 Pi0.5 action chunk。embodiment 依据 `env_cfg` 与 `action_type`
解包并校验，创建 opaque chunk id；只有 `execute_action_chunk` 才把它交给
RoboDojo/Isaac 执行。

## 当前固定配置

- provider id: `pi05`
- XPolicyLab policy module: `Pi_05`
- simulator: RoboDojo / Isaac Sim
- action type: `joint`
- policy server 与 simulator client 使用官方 XPolicyLab launcher 启动

checkpoint 名称、provider environment、GPU 分配与 RoboDojo root 属于服务端
registry 配置，不允许从 agent 请求中覆盖。

