# sparc

`sparc` 用于运行内容压缩、审计消融与端到端联合机制实验。

## 入口

- CLI：`sparc_cli`
- 配置：`configs/sparc/`
- 默认运行目录：`local/runs/sparc/<experiment>/<phase>/<run_id>/`
- 默认报告目录：`local/reports/sparc/`

## 常用命令

```powershell
uv run sparc_cli inspect-experiment --experiment configs/sparc/experiments/content_ablation.toml
uv run sparc_cli run --experiment configs/sparc/experiments/content_ablation.toml --phase smoke20 --model xiaomimimo/mimo-v2.5
uv run sparc_cli render-report --run-dir local/runs/sparc/content_ablation/smoke20/<run_id>
```
