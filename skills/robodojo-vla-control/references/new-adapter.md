# 新 VLA/WAM adapter 接入检查表

## 实现

- 创建独立的 XPolicyLab policy 目录，不修改 Pi0.5 或混合控制 skill。
- 实现 `model.py` 中的 `infer(observation)`，或兼容的 `update_obs()` / `get_action()`。
- 把相机、state、prompt 的模型特有映射限制在 adapter 内。
- 返回模型产生的 action chunk，不在 adapter 内调用 simulator。
- 提供固定的 `eval.sh` / policy server 启动入口。

## 注册

- 分配小写、稳定、可审计的 `provider_id`，声明 `kind: vla|wam`。
- 固定 `policy_name`、checkpoint、provider venv；这些路径只在服务端 registry 中出现。
- health 检查必须验证模型文件、启动入口和 RoboDojo 依赖。
- agent 请求只能选择已注册 provider id，不能传任意路径、模块或 shell 参数。

## 验证

- observation contract：必需 camera/state/prompt 完整，类型与布局正确。
- infer contract：返回非空、有限、形状正确的 action chunk。
- protocol contract：hello、prepare/reset、infer、trial end 和 server instance 检查通过。
- execution contract：action 进入 `robodojo.action-chunk/v1`，经 embodiment 校验后控制仿真机械臂。
- isolation：无 ROS/FCI/USB/真实设备接口，无 reward 或 privileged-state 读取。
- smoke test：固定 seed 的单 task、单 episode 能完成并生成官方结果与视频。

## 发布

- 更新 `ROBODOJO_POLICY_REGISTRY_JSON` 和该模型自己的 reference 文档。
- 不修改 planner/router/fallback 控制循环，除非 canonical action protocol 发生版本变化。
- 正式评测前保留 adapter 版本、checkpoint digest、运行配置和测试 receipt。

