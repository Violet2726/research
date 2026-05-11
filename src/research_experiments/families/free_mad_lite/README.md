# free_mad_lite

`free_mad_lite` 用于运行 Free-MAD-lite 机制验证实验。

## 入口

- CLI：`research_cli family free_mad_lite`
- 配置：`configs/families/free_mad_lite/`
- 默认运行目录：`local/runs/free_mad_lite/<experiment>/<phase>/<run_id>/`
- 默认报告目录：`local/reports/free_mad_lite/`

## 常用命令

```powershell
uv run research_cli family free_mad_lite inspect-experiment --experiment configs/families/free_mad_lite/experiments/free_mad_lite_mechanism_validation.toml
uv run research_cli family free_mad_lite run --experiment configs/families/free_mad_lite/experiments/free_mad_lite_mechanism_validation.toml --phase smoke20 --model xiaomimimo/mimo-v2.5
uv run research_cli family free_mad_lite render-report --run-dir local/runs/free_mad_lite/free_mad_lite_mechanism_validation/smoke20/<run_id>
```
