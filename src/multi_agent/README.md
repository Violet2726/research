# multi_agent

`multi_agent` 用于运行标准多智能体 debate 与 vote 对照实验。

## 入口

- CLI：`multi_agent_cli`
- 配置：`configs/multi_agent/`
- 默认运行目录：`local/runs/multi_agent/<experiment>/<phase>/<run_id>/`
- 默认报告目录：`local/reports/multi_agent/`

## 常用命令

```powershell
uv run multi_agent_cli inspect-experiment --experiment configs/multi_agent/experiments/same_context_controlled_debate.toml
uv run multi_agent_cli run --experiment configs/multi_agent/experiments/same_context_controlled_debate.toml --phase smoke20 --model xiaomimimo/mimo-v2.5
uv run multi_agent_cli render-report --run-dir local/runs/multi_agent/same_context_controlled_debate/smoke20/<run_id>
```
