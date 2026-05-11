# selective_comm

`selective_comm` 用于运行 trigger / early-exit 选择性通信实验。

## 入口

- CLI：`research_cli family selective_comm`
- 配置：`configs/families/selective_comm/`
- 默认运行目录：`local/runs/selective_comm/<experiment>/<phase>/<run_id>/`
- 默认报告目录：`local/reports/selective_comm/`

## 常用命令

```powershell
uv run research_cli family selective_comm inspect-experiment --experiment configs/families/selective_comm/experiments/trigger_early_exit_main.toml
uv run research_cli family selective_comm run --experiment configs/families/selective_comm/experiments/trigger_early_exit_main.toml --phase smoke20 --model xiaomimimo/mimo-v2.5
uv run research_cli family selective_comm render-report --run-dir local/runs/selective_comm/voc_trigger_main/smoke20/<run_id>
```
