# RoboDojo 推理轨迹 UI

在仓库根目录启动，无需安装前端或 Python 依赖：

```bash
python3 scripts/trajectory_ui.py --port 8790
```

浏览器打开 <http://127.0.0.1:8790>。更换数据目录时传入 `--root /absolute/path/to/eval_result`。
默认只监听本机，界面只读取保存的产物，不调用模型、控制机器人或改写评分。

- 按 task、评测批次、seed、结果筛选；搜索 `006` 或 `1-006` 定位场景。
- 主视角与双腕视频同步播放，支持倍速、动作步拖动、左右方向键步进。
- 查看实际保存的关节目标、阶段、IK 残差、中间结果、可见 Agent 消息和工具调用。
- 点击中间结果跳到动作步；打开原始评分、Harness JSONL 和技能日志核对。
- 场景结果每 15 秒刷新，选择与视频播放保持；“刷新”按钮强制重新扫描。
- URL 保存当前 task、批次、场景和动作步，复制地址即可复现当前位置。

## 产物适配

识别单集 `receipt.json` / `official-result.json`、比较实验的 `results.json`、
原生 PhysicalRSI 的 `official/_result.json`、中断归档的 `run-state.json`、
以及 `official-case-inventory.json` 或实验中的显式 `layouts` 清单。
同一 run 的内部 code worker 回执不会被计为额外场景。

数据目录下的 `ui-catalog.json` 可配置界面保留的评测组：`groups` 数组中的
`id` 对应评测组 ID，`title` 设置名称，`framework` 标注执行框架；按数组顺序显示。
这只控制界面目录，不改写原始实验产物或评分。

逐场自进化实验可在数据目录下用 `experiment.json` 注册：
`schema: "physicalrsi.online-code-evolution/v1"`、`artifact_root`（实验产物绝对路径）、
`task`、`seed`、`title`（界面名称）、`provider` 与 `qualification`（成绩说明）。
读取原生 `official/_result.json`，按 episode 索引关联执行 policy、执行前 memory、
模型修改理由、执行事件和反馈；不会把最后一次未执行的策略更新计为新场景。
最终 SKILL、policy 和 memory 可从原始产物打开，最终 policy 独立标注未单独评测。

优先使用归档的三路视频和 Harness 轨迹；存在本机的、由 run 记录引用的原始日志时，
读取其中的 `ALIGN_*` JSON 事件，展示真实动作目标和 IK 记录。
缺少某项产物时会明确显示缺失；远端路径不存在时不会尝试连接服务器。
中断执行与重建的公共 RGB 回放有独立标记，不计入已评分成功率。

视频时间与动作按帧序**近似**对齐，历史录像通常没有每个动作的媒体时间戳。
关节表是技能输出的目标命令，不代表实时实测状态。
Agent 面板展示已保存的可见消息和工具调用；未保存的推理内容不会被补写。
原生批量评测中没有明确关联到单集的 Agent/worker 记录时，仅显示可确认的评分与视频。

验证：`python3 -m pytest -q tests/test_trajectory_ui.py`。
