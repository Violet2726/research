# budget_comm

`budget_comm` 用于运行预算约束通信与包分配策略实验。

## 入口

- CLI：`budget_comm_cli`
- 配置：`configs/budget_comm/`
- 默认运行目录：`local/runs/budget_comm/<experiment>/<phase>/<run_id>/`
- 默认报告目录：`local/reports/budget_comm/`

## 常用命令

```powershell
uv run budget_comm_cli inspect-experiment --experiment configs/budget_comm/experiments/dala_lite_same_context_main.toml
uv run budget_comm_cli run --experiment configs/budget_comm/experiments/dala_lite_same_context_main.toml --phase smoke20 --model xiaomimimo/mimo-v2.5
uv run budget_comm_cli render-report --run-dir local/runs/budget_comm/dala_lite_split_context_main/smoke20/<run_id>
```
