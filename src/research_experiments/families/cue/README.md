# cue

`cue` 用于运行 Communication Utility Estimation 实验。

## 入口

- CLI：`research_cli family cue`
- 配置：`configs/families/cue/`
- 默认运行目录：`local/runs/cue/<experiment>/<phase>/<run_id>/`
- 默认报告目录：`local/reports/cue/`

## 常用命令

```powershell
uv run research_cli family cue inspect-experiment --experiment configs/families/cue/experiments/cue_black_box_utility_main.toml
uv run research_cli family cue run --experiment configs/families/cue/experiments/cue_black_box_utility_main.toml --phase smoke20 --model xiaomimimo/mimo-v2.5
uv run research_cli family cue render-report --run-dir local/runs/cue/cue_black_box_utility_main/smoke20/<run_id>
```
