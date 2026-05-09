# cue

`cue` 用于运行 CUE，目标是在黑盒 LLM 环境下估计通信效用并只对高价值冲突触发通信。

## 入口

- CLI：`cue_cli`
- 配置：`configs/cue/`
- 运行目录：`runs/cue/<experiment>/<phase>/<run_id>/`
- 报告目录：`reports/cue/`

## 常用命令

```powershell
uv run cue_cli inspect-experiment --experiment configs/cue/experiments/cue_black_box_utility_main.toml
uv run cue_cli run --experiment configs/cue/experiments/cue_black_box_utility_main.toml --phase smoke20 --model xiaomimimo/mimo-v2.5
uv run cue_cli render-report --run-dir runs/cue/cue_black_box_utility_main/smoke20/<run_id>
```

## 维护约定

- `AgentPacket`、`ConflictObject`、`BeliefUpdate` 和 `AuditVerdict` 等结构保持稳定，避免在 runner 里临时拼字段。
- trigger、消息类型和局部审计都围绕统一 utility 框架扩展。
- 共享的 provider、限流和结构化恢复逻辑统一复用 `experiment_core`。
