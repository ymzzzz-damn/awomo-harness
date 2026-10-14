# 运行时架构

```text
goal + public observation
          |
          v
awomopaw GPT planner (System 2)
          | subtask + provider prompt
          v
provider router -----> registered VLA/WAM (System 1)
          |                         |
          | provider failure        | ActionChunk
          v                         v
GPT direct-action fallback ---> robodojo_sim embodiment
                                      | validate / queue / execute
                                      v
                            RoboDojo client -> Isaac Sim
                                      |
                                      +--> next public observation
                                      +--> post-episode score/artifacts
```

## 所有权

| 层 | 拥有 | 不拥有 |
|---|---|---|
| GPT planner | task plan、subtask、provider 选择、失败归因 | reward、privileged state、Isaac connection |
| Provider registry | stable id、VLA/WAM module、checkpoint、runtime | agent 传入的任意路径或命令 |
| VLA/WAM adapter | observation 映射、模型推理、action chunk | simulator step、score、reset |
| `robodojo_sim` embodiment | public observation、action spec、验证、chunk queue、Isaac execution | 模型特有推理、score override |
| RoboDojo evaluator | reset、seed、场景、reward、success、视频、结果 | 外部模型凭据 |

一个 interactive session 固定一个 provider。切换模型需要启动新的、可审计的
session；不能在 episode 中把同一 stateful policy client 偷换到另一个 server。
GPT fallback 不替换 provider registry，它只是同一 session 内受限的第二 action
source，仍走 embodiment action queue。

## 两条反馈链路

```text
VLA/WAM chunk -> execute short window -> fresh public RGB/state
                                            |
                                   GPT reflection + verdict
                                   /           |          \\
                              continue       correct     uncertain
                                 |              |            |
                           next VLA/WAM   GPT direct/code   observe/abort

completed task -> videos + native result + chunk reflections + current skill/policy
                                            |
                                  LLM diagnosis -> distillation
                                            |
                           immutable task skill/code/memory candidate
                                            |
                            next development run -> repeat feedback
                                            |
                              independent validation -> promotion
```

新监督 bridge 对 verdict 与下一 controller 的对应关系做强制校验。任务后
蒸馏可用 `--evolve --rounds N` 自动回流下一 development run。候选与正式版本
分开；不让 validation/final 反馈进入蒸馏，也不更新模型权重。

## 进程边界

- `:8088`：awomopaw API 和 agent。
- `:8098`：loopback embodiment bridge，暴露为 agent tools。
- `:8099`：episode 内部 control API，仅由 `:8098` 代理。
- XPolicyLab policy server：加载 session 固定的 VLA/WAM。
- RoboDojo client：持有 Isaac 环境并运行 interactive embodiment loop。

这些端口在部署中保持 loopback。外部只通过 awomopaw agent 或集群内受控服务
访问，不公开 episode action API。
