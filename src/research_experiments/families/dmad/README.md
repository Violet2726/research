# dmad

`dmad` 用于复现 ICLR 2025 论文《Breaking Mental Set to Improve Reasoning through Diverse Multi-Agent Debate》的文本 LLM 主线。

## 入口

- CLI：`research_cli family dmad`
- 配置：`configs/families/dmad/`
- 默认运行目录：`local/runs/dmad/<experiment>/<phase>/<run_id>/`

## 当前口径

- `dmad_reasoning_main`：论文正文主结果，只包含 `MATH` 与 `GPQA`
- `dmad_reasoning_appendix`：论文附录 `MMLU abstract_algebra`
- `dmad_reasoning_extended`：扩展验证集 `StrategyQA / HotpotQA`
- 所有主线 MAD/DMAD 方法统一使用 `3 agents + 2 rounds`
- 主线聚合器统一为最终轮自洽投票，不再保留 selector 双轨逻辑

## 论文主表方法

- 单智能体：`cot`、`sbp`、`pot/l2m`
- 自洽：`cot_sc`、`sbp_sc`、`pot_sc/l2m_sc`
- 自反思：`self_refine`
- 对比式自纠错：`self_contrast`
- 动态方法选择：`mrp`
- 固定-MAD：`mad_all_cot`、`mad_all_sbp`、`mad_all_pot/l2m`
- persona-MAD：`mad_persona_d`、`mad_persona_e`
- DMAD：`dmad_cot_sbp_pot` 或 `dmad_cot_sbp_l2m`

## 常用命令

```powershell
uv run research_cli family dmad inspect-experiment --experiment configs/families/dmad/experiments/dmad_reasoning_main.toml
uv run research_cli family dmad run --experiment configs/families/dmad/experiments/dmad_reasoning_main.toml --phase count20 --model xiaomimimo/mimo-v2.5
uv run research_cli family dmad render-report --run-dir local/runs/dmad/dmad_reasoning_main/count20/<run_id>
```
