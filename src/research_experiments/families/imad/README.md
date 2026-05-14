# imad

`imad` 用于复现 Adaptive Stopping / Efficient Debate 路线的 same-context 正式实验。

## 入口

- CLI：`research_cli family imad`
- 配置：`configs/families/imad/`
- 默认运行目录：`local/runs/imad/<experiment>/<phase>/<run_id>/`
- 默认报告目录：`local/reports/imad/`

## 常用命令

```powershell
uv run research_cli family imad inspect-experiment --experiment configs/families/imad/experiments/imad_same_context_main.toml
uv run research_cli family imad run --experiment configs/families/imad/experiments/imad_same_context_main.toml --phase count20 --model xiaomimimo/mimo-v2.5
uv run research_cli family imad render-report --run-dir local/runs/imad/imad_same_context_main/count20/<run_id>
```

