# budget_comm

`budget_comm` 用于运行 DALA-lite 风格的预算约束通信实验。

## 入口

- CLI：`budget_comm_cli`
- 配置：`configs/budget_comm/`
- 运行目录：`runs/budget_comm/<experiment>/<phase>/<run_id>/`
- 报告目录：`reports/budget_comm/`

## 常用命令

```powershell
uv run budget_comm_cli inspect-experiment --experiment configs/budget_comm/experiments/dala_lite_same_context_main.toml
uv run budget_comm_cli run --experiment configs/budget_comm/experiments/dala_lite_split_context_main.toml --phase smoke20 --model xiaomimimo/mimo-v2.5
uv run budget_comm_cli render-report --run-dir runs/budget_comm/dala_lite_split_context_main/smoke20/<run_id>
```

## 维护约定

- same-context / split-context 视图通过配置和数据视图模块切换。
- 预算、tier、knapsack 和 paired design 诊断不要散落到报告层。
- 只依赖 `experiment_core`。
