# 论文产物包

- generated_at: `2026-05-09T12:58:57.691211+00:00`
- phase_name: `smoke20`
- model_ref: `xiaomimimo/mimo-v2.5`
- counts: `{"completed": 15, "excluded": 0, "semantic_unique_targets": 15}`

## 同上下文主结果表

| experiment_name | primary_method_name | faithful_score | delta_vs_best_no_comm | delta_vs_full_comm | delta_vs_full_context | token_ratio_vs_full_comm | calls_per_question_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dala_lite_same_context_main | dala_lite | 0.750000 | 0.083333 | 0.000000 |  | 0.918221 | 6.000000 |
| trigger_early_exit_main | hybrid_trigger | 0.866667 | 0.000000 | 0.000000 |  | 0.603075 | 3.650000 |
| voc_trigger_main | voc_trigger_v2 | 0.766667 | 0.050000 | 0.000000 |  | 0.818370 | 4.500000 |
| end_to_end_main | sparc_v1 | 0.716667 | 0.050000 | -0.016666 |  | 0.688190 | 4.550000 |

## 分视角主结果表

| experiment_name | primary_method_name | faithful_score | delta_vs_best_no_comm | delta_vs_full_comm | delta_vs_full_context | token_ratio_vs_full_comm | calls_per_question_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dala_lite_split_context_main | dala_lite | 0.650000 | 0.100000 | -0.075000 |  | 0.894359 | 6.000000 |
| hotpotqa_split_context_communication_necessity | full_packet_exchange | 0.500000 | 0.100000 |  | -0.200000 |  | 6.000000 |

## 支撑证据

| experiment_name | primary_method_name | faithful_score | delta_vs_best_no_comm | delta_vs_full_comm | delta_vs_full_context | token_ratio_vs_full_comm | calls_per_question_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| free_mad_lite_mechanism_validation | free_mad_lite_llm_trajectory | 0.816667 | 0.150000 | 0.016667 |  | 1.290219 | 7.000000 |
| same_context_controlled_debate | mad_3a_r1 | 0.816667 | 0.100000 |  |  |  | 6.000000 |
| local_auditing_ablation | local_auditing | 0.733333 | 0.066666 | 0.033333 |  | 1.062695 | 6.316667 |

## 诊断证据

| experiment_name | primary_method_name | faithful_score | delta_vs_best_no_comm | delta_vs_full_comm | delta_vs_full_context | token_ratio_vs_full_comm | calls_per_question_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cue_black_box_utility_main | cue_v1 | 0.620000 | 0.050000 | -0.080000 |  | 0.511998 | 3.640000 |
| sid_lite_mechanism_validation | sid_lite | 0.700000 | 0.000000 | -0.083333 |  | 0.627644 | 4.250000 |
| cross_provider_robustness | cot_1 | 0.850000 | 0.000000 |  |  |  | 1.000000 |
| content_ablation | task_adaptive | 0.683333 | 0.016666 | -0.050000 |  | 0.895563 | 6.000000 |

## 等预算单智能体参照

| experiment_name | baseline_name | matched_budget_source | matched_calls_per_q | matched_total_tokens_mean | score | budget_match_status |
| --- | --- | --- | --- | --- | --- | --- |
| dala_lite_same_context_main | budget_matched_long_cot | same_context_core_benchmarks/cot_1/overall | 1.000000 | 679.850000 | 0.850000 | available_proxy_not_exact_budget |
| dala_lite_same_context_main | budget_matched_sc | same_context_core_benchmarks/sc_5/overall | 5.000000 | 3384.950000 | 0.800000 | available_proxy_not_exact_budget |
| dala_lite_split_context_main | budget_matched_long_cot | same_context_core_benchmarks/cot_1/overall | 1.000000 | 679.850000 | 0.850000 | available_proxy_not_exact_budget |
| dala_lite_split_context_main | budget_matched_sc | same_context_core_benchmarks/sc_5/overall | 5.000000 | 3384.950000 | 0.800000 | available_proxy_not_exact_budget |
| hotpotqa_split_context_communication_necessity | budget_matched_long_cot | same_context_core_benchmarks/cot_1/overall | 1.000000 | 679.850000 | 0.850000 | available_proxy_not_exact_budget |
| hotpotqa_split_context_communication_necessity | budget_matched_sc | same_context_core_benchmarks/sc_5/overall | 5.000000 | 3384.950000 | 0.800000 | available_proxy_not_exact_budget |
| hotpotqa_split_context_communication_necessity | budget_matched_full_context_single | full_context_single |  |  | 0.700000 | same_experiment_full_context_reference |
| trigger_early_exit_main | budget_matched_long_cot | same_context_core_benchmarks/cot_1/overall | 1.000000 | 679.850000 | 0.850000 | available_proxy_not_exact_budget |
| trigger_early_exit_main | budget_matched_sc | same_context_core_benchmarks/sc_5/overall | 5.000000 | 3384.950000 | 0.800000 | available_proxy_not_exact_budget |
| voc_trigger_main | budget_matched_long_cot | same_context_core_benchmarks/cot_1/overall | 1.000000 | 679.850000 | 0.850000 | available_proxy_not_exact_budget |
| voc_trigger_main | budget_matched_sc | same_context_core_benchmarks/sc_5/overall | 5.000000 | 3384.950000 | 0.800000 | available_proxy_not_exact_budget |
| end_to_end_main | budget_matched_long_cot | same_context_core_benchmarks/cot_1/overall | 1.000000 | 679.850000 | 0.850000 | available_proxy_not_exact_budget |
| end_to_end_main | budget_matched_sc | same_context_core_benchmarks/sc_5/overall | 5.000000 | 3384.950000 | 0.800000 | available_proxy_not_exact_budget |

## 固定统计比较

| comparison_id | status | paired_n | mean_delta | delta_ci95 | wins | losses | ties | mcnemar_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_trigger_vs_always_communicate | completed | 60 | 0.000000 | [0.000000, 0.000000] | 0 | 0 | 60 | 1.000000 |
| hybrid_trigger_vs_sc_6 | completed | 60 | 0.000000 | [-0.066667, 0.066667] | 2 | 2 | 56 | 0.617075 |
| voc_trigger_v2_vs_mv_6 | completed | 60 | 0.083333 | [-0.016667, 0.183333] | 8 | 3 | 49 | 0.227800 |
| dala_same_vs_all_to_all_full | completed | 60 | 0.000000 | [-0.083333, 0.083333] | 3 | 3 | 54 | 0.683091 |
| hotpot_split_comm_necessity_vs_split_no_comm_mv3 | completed | 20 | 0.100000 | [0.000000, 0.250000] | 2 | 0 | 18 | 0.479500 |
| dala_split_vs_all_to_all_full | completed | 40 | -0.075000 | [-0.200000, 0.050000] | 2 | 5 | 33 | 0.449692 |

## 有益 / 有害通信

| experiment_name | method_name | sample_method_rows | helpful | harmful | neutral | helpful_rate | harmful_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dala_lite_same_context_main | dala_lite | 60 | 7 | 2 | 51 | 0.116667 | 0.033333 |
| hotpotqa_split_context_communication_necessity | full_packet_exchange | 20 | 2 | 0 | 18 | 0.100000 | 0.000000 |
| trigger_early_exit_main | hybrid_trigger | 60 | 1 | 0 | 59 | 0.016667 | 0.000000 |
| end_to_end_main | sparc_v1 | 60 | 4 | 1 | 55 | 0.066667 | 0.016667 |
| dala_lite_split_context_main | dala_lite | 40 | 5 | 1 | 34 | 0.125000 | 0.025000 |
| voc_trigger_main | voc_trigger_v2 | 60 | 4 | 1 | 55 | 0.066667 | 0.016667 |

## 图表索引

- `budget_frontier_same_context`: `figures/budget_frontier_same_context.svg`
- `budget_frontier_split_context`: `figures/budget_frontier_split_context.svg`
- `helpful_harmful_comm`: `figures/helpful_harmful_comm.svg`
- `stage_ceiling_gap`: `figures/stage_ceiling_gap.svg`
- `trigger_utility`: `figures/trigger_utility.svg`

## 解释边界

- 主结果表中的 headline 行才用于正文主结论。
- supporting 和 diagnostic 行用于解释机制、限制和消融，不直接替代主结论。
- 等预算单智能体行是评测控制组，不代表新增方法步骤。
- 如果 confirmatory 结果削弱了 headline 结论，应下调结论强度，而不是事后改写方法图。

## 图表资产

### 预算前沿：同上下文 headline 方法

![预算前沿：同上下文 headline 方法](../../runs/faithful_matrix_iterative/20260509T125834Z-smoke20-xiaomimimo-mimo-v2.5/figures/budget_frontier_same_context.svg)

*Faithful score 相对于 full communication token 比率的位置关系。*

### 预算前沿：分视角 headline 方法

![预算前沿：分视角 headline 方法](../../runs/faithful_matrix_iterative/20260509T125834Z-smoke20-xiaomimimo-mimo-v2.5/figures/budget_frontier_split_context.svg)

*Faithful score 相对于 full communication 或匹配参照 token 比率的位置关系。*

### Trigger 收益

![Trigger 收益](../../runs/faithful_matrix_iterative/20260509T125834Z-smoke20-xiaomimimo-mimo-v2.5/figures/trigger_utility.svg)

*trigger 类方法相对于最优无通信基线的收益。*

### 距 stage ceiling 的差距

![距 stage ceiling 的差距](../../runs/faithful_matrix_iterative/20260509T125834Z-smoke20-xiaomimimo-mimo-v2.5/figures/stage_ceiling_gap.svg)

*headline 与 supporting 运行中 faithful score 到 stage ceiling 的绝对差距。*

### 有益与有害通信对比

![有益与有害通信对比](../../runs/faithful_matrix_iterative/20260509T125834Z-smoke20-xiaomimimo-mimo-v2.5/figures/helpful_harmful_comm.svg)

*各通信实验中有益通信率与有害通信率的并列比较。*
