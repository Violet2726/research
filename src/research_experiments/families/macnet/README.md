# macnet

`macnet` 用于运行 MacNet DAG 拓扑协作论文复现实验。

## 入口

- CLI：`research_cli family macnet`
- 配置：`configs/families/macnet/`
- 默认运行目录：`local/runs/macnet/<experiment>/<phase>/<run_id>/`
- 默认报告目录：`local/reports/macnet/`

## 常用命令

```powershell
uv run research_cli family macnet inspect-experiment --experiment configs/families/macnet/experiments/macnet_paper_main.toml
uv run research_cli family macnet run --experiment configs/families/macnet/experiments/macnet_paper_main.toml --phase count20 --model xiaomimimo/mimo-v2.5
uv run research_cli family macnet render-report --run-dir local/runs/macnet/macnet_paper_main/count20/<run_id>
```

## 当前口径

- `macnet_paper_main` 是当前项目里的 MacNet 主复现线。
- `macnet_scaling_study` 用来补拓扑规模曲线与调用数匹配控制，不直接改写主表结论。
- 这条线是平行复现支线，当前进入 `reproduction_matrix`，但不进入 `faithful_matrix`。
- canonical benchmark 当前固定为 `MMLU / HumanEval / CommonGen-Hard`。
- `SRDD_Profile` 在 v1 中作为官方角色库资产接入，用于 actor / critic profile 选择，而不是单独的 scored benchmark。

## 设计边界

- v1 只做 `single_agent_cot + 六种官方拓扑 + scaling controls`。
- `AutoGPT / GPTSwarm / AgentVerse` 不进入 canonical 主实验。
- 目标是复现 DAG 拓扑协作、方向差异与规模趋势，不把 MacNet 改写成普通多轮辩论。
- 如果 `count300` 只体现出 token 单调堆叠、没有任务分化或 scale 趋势，这条线应冻结为 secondary reproduction。
