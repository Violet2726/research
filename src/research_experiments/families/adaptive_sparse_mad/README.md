# adaptive_sparse_mad

`adaptive_sparse_mad` 用于验证 A-SMAD 在 same-context 设定下的主机制：异质 Stage A 求解、稀疏触发式追加求解，以及基于证据门控的聚合与反事实修正。

## 当前保留的实验入口

- `same_context_main_v5.toml`
  - 默认 same-context 主线入口。
  - 基准集为 `hotpotqa / strategyqa / gsm8k`。
  - 主方法是 `hetero_vote_3 + adaptive_counterfactual_v1`。
- `same_context_main_v6.toml`
  - V6 same-context 涓荤嚎鍏ュ彛銆?
  - 鐢ㄤ簬璺戞柊鐨?`adaptive_sparse_rescue_only_v1 / adaptive_sparse_probe_only_v1 / adaptive_sparse_rescue_probe_v1` 鍙樹綋銆?
- `same_context_full_counterfactual_v1_screen.toml`
  - 7 数据集 `count20` 筛选入口。
  - 只保留 `cot_1` 作为 screen 对照，避免在筛选阶段堆无关 baseline。
- `same_context_full_counterfactual_v1.toml`
  - 7 数据集正式验证入口。
  - 用于 `count100` 主结论、显著性检验和主线准入判断。

## 当前结论

- `adaptive_counterfactual_v1` 已通过 `count20` 的 `Promotion Gate`。
- `adaptive_counterfactual_v1` 已通过 `count100` 的 `Mainline Gate`。
- 当前默认保留路线是 `hetero_vote_3 + adaptive_counterfactual_v1`。

## 已清理内容

- 历史单数据集配置、过时主线入口和 `_archive` 配置目录已从正式工程删除。
- 已放弃的 prompt-only、typed-latent、slot-extractor、intersection 等分支不再保留为配置入口。
- CLI smoke 与治理测试现在只覆盖当前 3 个有效实验配置。

## 运行与验证

```powershell
uv run research_cli experiment --family adaptive_sparse_mad inspect-experiment --experiment configs/families/adaptive_sparse_mad/experiments/same_context_main_v5.toml
uv run research_cli experiment --family adaptive_sparse_mad run --experiment configs/families/adaptive_sparse_mad/experiments/same_context_main_v5.toml --phase count100 --model xiaomimimo/mimo-v2.5
uv run research_cli experiment --family adaptive_sparse_mad validate-run --run-dir local/runs/adaptive_sparse_mad/<experiment>/count100/<run_id>
uv run research_cli experiment --family adaptive_sparse_mad render-report --run-dir local/runs/adaptive_sparse_mad/<experiment>/count100/<run_id>
uv run research_cli experiment --family adaptive_sparse_mad refresh-run-artifacts --run-dir local/runs/adaptive_sparse_mad/<experiment>/<phase>/<run_id>
```

```powershell
uv run research_cli experiment --family adaptive_sparse_mad inspect-experiment --experiment configs/families/adaptive_sparse_mad/experiments/same_context_main_v6.toml
uv run research_cli experiment --family adaptive_sparse_mad run --experiment configs/families/adaptive_sparse_mad/experiments/same_context_main_v6.toml --phase count20 --model xiaomimimo/mimo-v2.5
uv run research_cli experiment --family adaptive_sparse_mad run --experiment configs/families/adaptive_sparse_mad/experiments/same_context_main_v6.toml --phase count100 --model xiaomimimo/mimo-v2.5
```

```powershell
uv run research_cli experiment --family adaptive_sparse_mad run --experiment configs/families/adaptive_sparse_mad/experiments/same_context_full_counterfactual_v1_screen.toml --phase count20 --model xiaomimimo/mimo-v2.5
uv run research_cli experiment --family adaptive_sparse_mad run --experiment configs/families/adaptive_sparse_mad/experiments/same_context_full_counterfactual_v1.toml --phase count100 --model xiaomimimo/mimo-v2.5
```

## 研究原则

- `count20` 只负责筛选，不直接承担主结论。
- `count100` 才是 paired significance 和主线准入的正式依据。
- 不针对单一数据集做定向提示词微调。
- 机制若无法跨数据集稳定获益，应及时收缩或删除，而不是长期保留配置入口。
