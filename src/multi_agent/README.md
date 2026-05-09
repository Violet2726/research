# multi_agent

`multi_agent` 用于运行标准多智能体 debate 与 vote 对照实验。

## 入口

- CLI：`multi_agent_cli`
- 配置：`configs/multi_agent/`
- 运行目录：`runs/multi_agent/<experiment>/<phase>/<run_id>/`
- 报告目录：`reports/multi_agent/`

## 常用命令

```powershell
uv run multi_agent_cli inspect-experiment --experiment configs/multi_agent/experiments/same_context_controlled_debate.toml
uv run multi_agent_cli run --experiment configs/multi_agent/experiments/same_context_controlled_debate.toml --phase smoke20 --model xiaomimimo/mimo-v2.5
uv run multi_agent_cli report-debate-vs-vote --run-dir runs/multi_agent/same_context_controlled_debate/smoke20/<run_id>
```

## 维护约定

- Debate 协议、matched controls 和 roster 都通过配置组装。
- 默认报告重点是 debate vs vote 的配对比较，不把多实验家族逻辑混进来。
- 只依赖 `experiment_core`。
