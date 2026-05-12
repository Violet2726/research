# sid_lite

`sid_lite` 用于运行 SID-lite 早退与压缩机制验证。

## 入口

- CLI：`research_cli family sid_lite`
- 配置：`configs/families/sid_lite/`
- 默认运行目录：`local/runs/sid_lite/<experiment>/<phase>/<run_id>/`
- 默认报告目录：`local/reports/sid_lite/`

## 常用命令

```powershell
uv run research_cli family sid_lite inspect-experiment --experiment configs/families/sid_lite/experiments/sid_lite_mechanism_validation.toml
uv run research_cli family sid_lite run --experiment configs/families/sid_lite/experiments/sid_lite_mechanism_validation.toml --phase count20 --model xiaomimimo/mimo-v2.5
uv run research_cli family sid_lite render-report --run-dir local/runs/sid_lite/sid_lite_mechanism_validation/count20/<run_id>
```
