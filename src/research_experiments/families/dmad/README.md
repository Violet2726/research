# dmad

`dmad` 用于运行 Diverse Multi-Agent Debate 的论文对齐复现实验。

## 入口

- CLI: `research_cli family dmad`
- 配置: `configs/families/dmad/`
- 默认运行目录: `local/runs/dmad/<experiment>/<phase>/<run_id>/`
- 默认报告目录: `local/reports/dmad/`

## 常用命令

```powershell
uv run research_cli family dmad inspect-experiment --experiment configs/families/dmad/experiments/dmad_reasoning_main.toml
uv run research_cli family dmad run --experiment configs/families/dmad/experiments/dmad_reasoning_main.toml --phase count20 --model xiaomimimo/mimo-v2.5
uv run research_cli family dmad render-report --run-dir local/runs/dmad/dmad_reasoning_main/count20/<run_id>
```

## 当前口径

- `dmad_reasoning_main` 是当前项目中的 DMAD 正式复现主线。
- 当前 benchmark 覆盖 `MATH500 / MMLU Abstract Algebra / GPQA Diamond / StrategyQA / HotpotQA`。
- 当前 canonical 方法固定为 `single_agent_cot / single_agent_reflection_r1 / mv_6 / vanilla_mad_r1 / persona_diverse_mad_r1 / dmad_strategy_diverse_r1`。
- `single_agent_cot` 与 `mv_6` 复用仓内共享 control catalog 的正式参数口径，而不是在 DMAD family 内部重新定义。

## 论文对齐

- `single_agent_reflection_r1` 按论文图示走三步式 `draft -> feedback -> revise`，不再把 reflection 简化成一次自查。
- `persona_diverse_mad_r1` 保持 `CoT` 求解策略不变，但 persona 角色改成更接近论文示意图的 affirmative / negative / moderator 分工。
- `vanilla_mad_r1 / persona_diverse_mad_r1 / dmad_strategy_diverse_r1` 统一使用 best-solution selector 作为最终聚合器，以对齐作者公开代码里“在候选解之间选择最佳解”的逻辑，而不是只做多数投票。
- `dmad_strategy_diverse_r1` 对齐论文里的 prompting-family 思路:
  `math500 / gsm8k` 使用 `CoT / SBP / PoT`
  其余文本推理任务使用 `CoT / SBP / L2M`
- 当前文本推理 canonical 协议是 `3 agents + 2 debate rounds`。
- 主叙事是“策略异质化是否优于表面 persona 多样化”，而不是简单宣称多智能体一定更强。
