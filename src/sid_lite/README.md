# sid_lite

`sid_lite` 用于运行 SID-lite 早退与压缩机制验证。

## 入口

- CLI：`sid_lite_cli`
- 配置：`configs/sid_lite/`
- 默认运行目录：`local/runs/sid_lite/<experiment>/<phase>/<run_id>/`
- 默认报告目录：`local/reports/sid_lite/`

## 常用命令

```powershell
uv run sid_lite_cli inspect-experiment --experiment configs/sid_lite/experiments/sid_lite_mechanism_validation.toml
uv run sid_lite_cli run --experiment configs/sid_lite/experiments/sid_lite_mechanism_validation.toml --phase smoke20 --model xiaomimimo/mimo-v2.5
uv run sid_lite_cli render-report --run-dir local/runs/sid_lite/sid_lite_mechanism_validation/smoke20/<run_id>
```
