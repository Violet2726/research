# selective_comm

`selective_comm` 用于运行 trigger / early-exit 选择性通信实验。

## 入口

- CLI：`selective_comm_cli`
- 配置：`configs/selective_comm/`
- 运行目录：`runs/selective_comm/<experiment>/<phase>/<run_id>/`
- 报告目录：`reports/selective_comm/`

## 常用命令

```powershell
uv run selective_comm_cli inspect-experiment --experiment configs/selective_comm/experiments/trigger_early_exit_main.toml
uv run selective_comm_cli run --experiment configs/selective_comm/experiments/voc_trigger_main.toml --phase smoke20 --model xiaomimimo/mimo-v2.5
uv run selective_comm_cli render-report --run-dir runs/selective_comm/voc_trigger_main/smoke20/<run_id>
```

## 维护约定

- `Stage A` 与 `Stage B` 是共享前缀产物，同题不同策略不重复发网络请求。
- 新 trigger 策略优先通过 policy 配置扩展，不在 runner 里散落特判。
- trigger 诊断与 oracle 评估统一写入 `policy_metrics.json` / `policy_diagnostics.json`。
