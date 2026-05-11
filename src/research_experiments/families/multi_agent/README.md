# multi_agent

`multi_agent` 用于运行标准多智能体 debate 与 vote 对照实验。

## 入口

- CLI：`research_cli family multi_agent`
- 配置：`configs/families/multi_agent/`
- 默认运行目录：`local/runs/multi_agent/<experiment>/<phase>/<run_id>/`
- 默认报告目录：`local/reports/multi_agent/`

## 常用命令

```powershell
uv run research_cli family multi_agent inspect-experiment --experiment configs/families/multi_agent/experiments/same_context_controlled_debate.toml
uv run research_cli family multi_agent run --experiment configs/families/multi_agent/experiments/same_context_controlled_debate.toml --phase smoke20 --model xiaomimimo/mimo-v2.5
uv run research_cli family multi_agent render-report --run-dir local/runs/multi_agent/same_context_controlled_debate/smoke20/<run_id>
```
