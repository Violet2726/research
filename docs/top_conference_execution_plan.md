# 顶会投稿导向执行计划

## 1. 当前主线

项目主线冻结为：固定预算下的多智能体高价值分歧处理框架。

后续默认冻结主线 family，不再为了局部分数随意改动方法结构；但允许为高价值论文复现新增少量正式 family，并纳入统一 matrix / validation / reporting 契约。允许继续改动的内容包括：

- 矩阵编排、phase 配置和运行记录管理。
- 解析、评分、统计检验和报告生成。
- 单智能体同预算评估对照。
- 图表、论文包和 appendix 证据组织。

当前唯一明确打开的平行复现支线是：

- `dog_graph_main`：DoG 原论文高保真复现主线，覆盖 `WebQuestions / GrailQA / WebQSP / CWQ / MetaQA 1-hop / 2-hop / 3-hop`，使用动态图检索、enough-answer 判断、三角色顺序问题简化与 direct fallback；独立运行、独立报告，不并入当前 `faithful_matrix`。

方法本体禁止改动：

- 不新增 early exit。
- 不新增 fallback、额外 judge、额外 audit。
- 不修改 trigger rule、通信轮数、agent 数、message set、utility 公式或 allocation 机制。
- 不针对单个数据集单独调阈值或改预算。

## 2. 证据分层

主文只使用 headline 子集，其他实验进入 supporting 或 diagnostic。
`global_total_board` 只用于横向景观扫描，不等于论文主结论表，也不能替代分轨分层解释。

Same-context headline：

- `trigger_early_exit_main`
- `voc_trigger_main`
- `dala_lite_same_context_main`
- `end_to_end_main`

Split-context headline：

- `hotpotqa_split_context_communication_necessity`
- `dala_lite_split_context_main`

Supporting：

- `imad_same_context_main`
- `same_context_controlled_debate`
- `free_mad_lite_mechanism_validation`
- `local_auditing_ablation`

Diagnostic：

- `cue_black_box_utility_main`
- `sid_lite_mechanism_validation`
- `content_ablation`
- `cross_provider_robustness`

Reference：

- `same_context_core_benchmarks`
- `same_context_main_table`

## 3. Confirmatory Phase

新增 phase：`count300`。

矩阵规模固定为 16 个正式 experiment，不包含本地开发项。

数据规模规则：

- GSM8K：`count300_seed42`
- HotpotQA：`count300_seed42`
- StrategyQA：`full229_seed42`
- MATH500：`count300_seed42`
- MMLU-Pro：`count300_seed42`
- GPQA Diamond：`full198_seed42`

预检命令：

```powershell
uv run research_cli matrix inspect-matrix --phase count300
```

正式运行命令：

```powershell
uv run research_cli matrix run --phase count300 --state-root local/runs/faithful_matrix_iterative --reference-state-path local/runs/faithful_matrix_iterative/20260507T121658Z-count100-xiaomimimo-mimo-v2.5
```

若运行中断，续跑命令：

```powershell
uv run research_cli matrix resume --state-path <matrix_run_dir> --reference-state-path local/runs/faithful_matrix_iterative/20260507T121658Z-count100-xiaomimimo-mimo-v2.5
```

## 4. 离线分析链路

每次矩阵完成后，必须生成以下产物：

- `faithful_analysis.json`
- `acceptance_summary.json`
- `paper_statistics.json`
- `bootstrap_ci.json`
- `paired_win_loss.json`
- `mcnemar_tests.json`
- `dataset_breakdown.json`
- `paper_package.json`
- `paper_package.md`
- `family_landscape.json`
- `family_landscape.md`
- `local/reports/faithful_matrix/<run_id>-paper_package.md`
- `local/reports/faithful_matrix/<run_id>-family_landscape.md`
- `<matrix_run_dir>/figures/`

离线重渲染命令：

```powershell
uv run research_cli matrix analyze-faithful --state-path <matrix_run_dir> --reference-state-path local/runs/count20_matrix_iterative/20260505T084626Z-count20-mimo-v2.5
uv run research_cli matrix evaluate-acceptance --analysis-path <matrix_run_dir>
uv run research_cli matrix render-statistics --state-path <matrix_run_dir>
uv run research_cli matrix render-paper-package --state-path <matrix_run_dir>
uv run research_cli matrix render-family-landscape --state-path <matrix_run_dir>
```

## 5. 固定统计比较

统计模块固定比较以下关系：

- `hybrid_trigger` vs `always_communicate`
- `hybrid_trigger` vs `sc_6`
- `voc_trigger_v2` vs `mv_6`
- `dala_lite` same-context vs `all_to_all_full`
- `full_packet_exchange` vs `split_no_comm_mv3`
- `dala_lite` split-context vs `all_to_all_full`

统计规则：

- bootstrap CI 用固定 seed。
- paired win/loss 处理胜、负、平。
- McNemar-style test 只用于同样本二值正确性比较。
- HotpotQA F1 类连续指标只使用 bootstrap / paired delta，不强行套 McNemar。

## 6. 单智能体同预算对照

新增同预算单智能体 reference 只作为评估配套，不进入任何多智能体方法内部。

当前报告包会输出：

- `budget_matched_long_cot`
- `budget_matched_sc`
- `budget_matched_full_context_single`

若没有精确同预算运行，报告必须标注为 `available_proxy_not_exact_budget`，不能把 proxy 写成严格反方结论。

后续若资源允许，应单独补跑精确 matched single-agent baseline，并替换 proxy。

## 7. 论文包图表

每轮最终结果必须输出：

- `budget_frontier_same_context`
- `budget_frontier_split_context`
- `trigger_utility`
- `stage_ceiling_gap`
- `helpful_harmful_comm`

图表用途：

- budget frontier：说明准确率-成本前沿。
- trigger utility：说明触发器是否真的保留有效通信。
- stage ceiling gap：说明方法结构内部还有多少未利用上限。
- helpful/harmful breakdown：说明通信带来修正还是误改。

## 8. 验收口径

工程健康：

- 所有最终保留 run 的 `run_validation.passed = true`。
- `request_failures_total = 0`。
- `output_success_rate >= 0.95`，目标为 `1.0`。

主文结论健康：

- same-context headline 应接近或超过 best no-comm，并显著降低 full communication token。
- split-context headline 应显著优于 split no-comm，并解释相对 full-context single 的剩余差距。
- 如果 confirmatory 结果削弱 headline，就将该结果降级为 supporting 或 diagnostic，不改方法结构追分。

## 9. 最新回放状态

已使用最新 `count100` 矩阵完成离线回放：

- 矩阵：`local/runs/faithful_matrix_iterative/20260507T121658Z-count100-xiaomimimo-mimo-v2.5`
- 论文包：`local/reports/faithful_matrix/20260507T121658Z-count100-xiaomimimo-mimo-v2.5-paper_package.md`
- 景观总览：`local/reports/faithful_matrix/20260507T121658Z-count100-xiaomimimo-mimo-v2.5-family_landscape.md`
- 图表：`local/runs/faithful_matrix_iterative/20260507T121658Z-count100-xiaomimimo-mimo-v2.5/figures/`

该回放用于验证统计和报告链路，不替代后续 `count300` 正式运行。
