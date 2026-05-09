# 论文产物包

- generated_at: `2026-05-09T12:59:44.162714+00:00`
- phase_name: `pilot100`
- model_ref: `xiaomimimo/mimo-v2.5`
- counts: `{"completed": 15, "excluded": 0, "semantic_unique_targets": 15}`

## 同上下文主结果表

| experiment_name | primary_method_name | faithful_score | delta_vs_best_no_comm | delta_vs_full_comm | delta_vs_full_context | token_ratio_vs_full_comm | calls_per_question_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dala_lite_same_context_main | dala_lite | 0.696667 | 0.093334 | -0.016666 |  | 0.920306 | 6.000000 |
| trigger_early_exit_main | hybrid_trigger | 0.846667 | -0.010000 | 0.000000 |  | 0.574502 | 3.610000 |
| voc_trigger_main | voc_trigger_v2 | 0.720000 | 0.073333 | -0.006667 |  | 0.823999 | 4.540000 |
| end_to_end_main | sparc_v1 | 0.676667 | 0.036667 | -0.003333 |  | 0.672796 | 4.503333 |

## 分视角主结果表

| experiment_name | primary_method_name | faithful_score | delta_vs_best_no_comm | delta_vs_full_comm | delta_vs_full_context | token_ratio_vs_full_comm | calls_per_question_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dala_lite_split_context_main | dala_lite | 0.675000 | 0.070000 | -0.065000 |  | 0.896915 | 6.000000 |
| hotpotqa_split_context_communication_necessity | full_packet_exchange | 0.590000 | 0.200000 |  | -0.100000 |  | 6.000000 |

## 支撑证据

| experiment_name | primary_method_name | faithful_score | delta_vs_best_no_comm | delta_vs_full_comm | delta_vs_full_context | token_ratio_vs_full_comm | calls_per_question_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| free_mad_lite_mechanism_validation | free_mad_lite_llm_trajectory | 0.770000 | 0.140000 | 0.006667 |  | 1.288927 | 7.000000 |
| same_context_controlled_debate | mad_3a_r1 | 0.770000 | 0.090000 |  |  |  | 6.000000 |
| local_auditing_ablation | local_auditing | 0.680000 | 0.040000 | 0.016667 |  | 1.046709 | 6.266667 |

## 诊断证据

| experiment_name | primary_method_name | faithful_score | delta_vs_best_no_comm | delta_vs_full_comm | delta_vs_full_context | token_ratio_vs_full_comm | calls_per_question_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cue_black_box_utility_main | cue_v1 | 0.578000 | 0.030000 | -0.058000 |  | 0.522537 | 3.790000 |
| sid_lite_mechanism_validation | sid_lite | 0.623333 | 0.010000 | -0.080000 |  | 0.588660 | 4.190000 |
| cross_provider_robustness | cot_1 | 0.833333 | 0.000000 |  |  |  | 1.000000 |
| content_ablation | task_adaptive | 0.640000 | 0.000000 | -0.090000 |  | 0.897595 | 6.000000 |

## 等预算单智能体参照

| experiment_name | baseline_name | matched_budget_source | matched_calls_per_q | matched_total_tokens_mean | score | budget_match_status |
| --- | --- | --- | --- | --- | --- | --- |
| dala_lite_same_context_main | budget_matched_long_cot | same_context_core_benchmarks/cot_1/overall | 1.000000 | 676.003333 | 0.833333 | available_proxy_not_exact_budget |
| dala_lite_same_context_main | budget_matched_sc | same_context_core_benchmarks/sc_5/overall | 5.000000 | 3378.490000 | 0.806667 | available_proxy_not_exact_budget |
| dala_lite_split_context_main | budget_matched_long_cot | same_context_core_benchmarks/cot_1/overall | 1.000000 | 676.003333 | 0.833333 | available_proxy_not_exact_budget |
| dala_lite_split_context_main | budget_matched_sc | same_context_core_benchmarks/sc_5/overall | 5.000000 | 3378.490000 | 0.806667 | available_proxy_not_exact_budget |
| hotpotqa_split_context_communication_necessity | budget_matched_long_cot | same_context_core_benchmarks/cot_1/overall | 1.000000 | 676.003333 | 0.833333 | available_proxy_not_exact_budget |
| hotpotqa_split_context_communication_necessity | budget_matched_sc | same_context_core_benchmarks/sc_5/overall | 5.000000 | 3378.490000 | 0.806667 | available_proxy_not_exact_budget |
| hotpotqa_split_context_communication_necessity | budget_matched_full_context_single | full_context_single |  |  | 0.690000 | same_experiment_full_context_reference |
| trigger_early_exit_main | budget_matched_long_cot | same_context_core_benchmarks/cot_1/overall | 1.000000 | 676.003333 | 0.833333 | available_proxy_not_exact_budget |
| trigger_early_exit_main | budget_matched_sc | same_context_core_benchmarks/sc_5/overall | 5.000000 | 3378.490000 | 0.806667 | available_proxy_not_exact_budget |
| voc_trigger_main | budget_matched_long_cot | same_context_core_benchmarks/cot_1/overall | 1.000000 | 676.003333 | 0.833333 | available_proxy_not_exact_budget |
| voc_trigger_main | budget_matched_sc | same_context_core_benchmarks/sc_5/overall | 5.000000 | 3378.490000 | 0.806667 | available_proxy_not_exact_budget |
| end_to_end_main | budget_matched_long_cot | same_context_core_benchmarks/cot_1/overall | 1.000000 | 676.003333 | 0.833333 | available_proxy_not_exact_budget |
| end_to_end_main | budget_matched_sc | same_context_core_benchmarks/sc_5/overall | 5.000000 | 3378.490000 | 0.806667 | available_proxy_not_exact_budget |

## 固定统计比较

| comparison_id | status | paired_n | mean_delta | delta_ci95 | wins | losses | ties | mcnemar_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_trigger_vs_always_communicate | completed | 300 | 0.000000 | [0.000000, 0.000000] | 0 | 0 | 300 | 1.000000 |
| hybrid_trigger_vs_sc_6 | completed | 300 | -0.010000 | [-0.036667, 0.016667] | 7 | 10 | 283 | 0.627626 |
| voc_trigger_v2_vs_mv_6 | completed | 300 | 0.076667 | [0.030000, 0.123333] | 37 | 14 | 249 | 0.002066 |
| dala_same_vs_all_to_all_full | completed | 300 | -0.016667 | [-0.046667, 0.013333] | 9 | 14 | 277 | 0.404248 |
| hotpot_split_comm_necessity_vs_split_no_comm_mv3 | completed | 100 | 0.200000 | [0.120000, 0.280000] | 20 | 0 | 80 | 0.000022 |
| dala_split_vs_all_to_all_full | completed | 200 | -0.065000 | [-0.120000, -0.010000] | 9 | 22 | 169 | 0.031141 |

## 有益 / 有害通信

| experiment_name | method_name | sample_method_rows | helpful | harmful | neutral | helpful_rate | harmful_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dala_lite_same_context_main | dala_lite | 300 | 33 | 5 | 262 | 0.110000 | 0.016667 |
| hotpotqa_split_context_communication_necessity | full_packet_exchange | 100 | 20 | 0 | 80 | 0.200000 | 0.000000 |
| trigger_early_exit_main | hybrid_trigger | 300 | 4 | 2 | 294 | 0.013333 | 0.006667 |
| end_to_end_main | sparc_v1 | 300 | 15 | 4 | 281 | 0.050000 | 0.013333 |
| dala_lite_split_context_main | dala_lite | 200 | 23 | 9 | 168 | 0.115000 | 0.045000 |
| voc_trigger_main | voc_trigger_v2 | 300 | 25 | 3 | 272 | 0.083333 | 0.010000 |

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

![预算前沿：同上下文 headline 方法](../../runs/faithful_matrix_iterative/20260509T125858Z-pilot100-xiaomimimo-mimo-v2.5/figures/budget_frontier_same_context.svg)

*Faithful score 相对于 full communication token 比率的位置关系。*

### 预算前沿：分视角 headline 方法

![预算前沿：分视角 headline 方法](../../runs/faithful_matrix_iterative/20260509T125858Z-pilot100-xiaomimimo-mimo-v2.5/figures/budget_frontier_split_context.svg)

*Faithful score 相对于 full communication 或匹配参照 token 比率的位置关系。*

### Trigger 收益

![Trigger 收益](../../runs/faithful_matrix_iterative/20260509T125858Z-pilot100-xiaomimimo-mimo-v2.5/figures/trigger_utility.svg)

*trigger 类方法相对于最优无通信基线的收益。*

### 距 stage ceiling 的差距

![距 stage ceiling 的差距](../../runs/faithful_matrix_iterative/20260509T125858Z-pilot100-xiaomimimo-mimo-v2.5/figures/stage_ceiling_gap.svg)

*headline 与 supporting 运行中 faithful score 到 stage ceiling 的绝对差距。*

### 有益与有害通信对比

![有益与有害通信对比](../../runs/faithful_matrix_iterative/20260509T125858Z-pilot100-xiaomimimo-mimo-v2.5/figures/helpful_harmful_comm.svg)

*各通信实验中有益通信率与有害通信率的并列比较。*
