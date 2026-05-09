# single_agent

`single_agent` 用于运行单智能体基线实验，当前只保留 `cot` 与 `sc_*` 方法。

## 入口

- CLI：`single_agent_cli`
- 配置：`configs/single_agent/`
- 运行目录：`runs/single_agent/<experiment>/<phase>/<run_id>/`
- 报告目录：`reports/single_agent/`

## 常用命令

```powershell
uv run single_agent_cli inspect-experiment --experiment configs/single_agent/experiments/same_context_core_benchmarks.toml
uv run single_agent_cli run --experiment configs/single_agent/experiments/same_context_core_benchmarks.toml --phase smoke20 --model xiaomimimo/mimo-v2.5
uv run single_agent_cli validate-run --run-dir runs/single_agent/same_context_core_benchmarks/smoke20/<run_id>
uv run single_agent_cli render-report --run-dir runs/single_agent/same_context_core_benchmarks/smoke20/<run_id>
```

## 维护约定

- 只依赖 `experiment_core`，不依赖其他实验包。
- 方法组合通过配置声明，不在 runner 里写死实验矩阵。
- 结构化输出恢复与评分逻辑尽量走共享层。
