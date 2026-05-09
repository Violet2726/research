# 论文产物包

- generated_at: `2026-05-09T13:01:25.496458+00:00`
- phase_name: `confirmatory300`
- model_ref: `xiaomimimo/mimo-v2.5`
- counts: `{"completed": 15, "excluded": 0, "semantic_unique_targets": 15}`

## 同上下文主结果表

| experiment_name | primary_method_name | faithful_score | delta_vs_best_no_comm | delta_vs_full_comm | delta_vs_full_context | token_ratio_vs_full_comm | calls_per_question_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dala_lite_same_context_main | dala_lite | 0.708082 | 0.110977 | -0.027744 |  | 0.922340 | 6.000000 |
| trigger_early_exit_main | hybrid_trigger | 0.835947 | -0.010856 | 0.001206 |  | 0.547168 | 3.477684 |
| voc_trigger_main | voc_trigger_v2 | 0.727382 | 0.088058 | -0.007238 |  | 0.820843 | 4.465621 |
| end_to_end_main | sparc_v1 | 0.664656 | 0.051870 | -0.004825 |  | 0.648410 | 4.386007 |

## 分视角主结果表

| experiment_name | primary_method_name | faithful_score | delta_vs_best_no_comm | delta_vs_full_comm | delta_vs_full_context | token_ratio_vs_full_comm | calls_per_question_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dala_lite_split_context_main | dala_lite | 0.648393 | 0.037807 | -0.064272 |  | 0.897040 | 6.000000 |
| hotpotqa_split_context_communication_necessity | full_packet_exchange | 0.576667 | 0.150000 |  | -0.076666 |  | 6.000000 |

## 支撑证据

| experiment_name | primary_method_name | faithful_score | delta_vs_best_no_comm | delta_vs_full_comm | delta_vs_full_context | token_ratio_vs_full_comm | calls_per_question_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| free_mad_lite_mechanism_validation | free_mad_lite_llm_trajectory | 0.772014 | 0.144752 | 0.007237 |  | 1.279580 | 7.000000 |
| same_context_controlled_debate | mad_3a_r1 | 0.786531 | 0.098389 |  |  |  | 6.000000 |
| local_auditing_ablation | local_auditing | 0.669481 | 0.056695 | 0.034982 |  | 1.044240 | 6.268999 |

## 诊断证据

| experiment_name | primary_method_name | faithful_score | delta_vs_best_no_comm | delta_vs_full_comm | delta_vs_full_context | token_ratio_vs_full_comm | calls_per_question_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cue_black_box_utility_main | cue_v1 | 0.572428 | 0.047585 | -0.055983 |  | 0.534714 | 3.850245 |
| sid_lite_mechanism_validation | sid_lite | 0.636912 | 0.020507 | -0.077201 |  | 0.583724 | 4.154403 |
| cross_provider_robustness | cot_1 | 0.831698 | 0.000000 |  |  |  | 1.000000 |
| content_ablation | task_adaptive | 0.626055 | 0.013269 | -0.112184 |  | 0.902542 | 6.000000 |

## 等预算单智能体参照

| experiment_name | baseline_name | matched_budget_source | matched_calls_per_q | matched_total_tokens_mean | score | budget_match_status |
| --- | --- | --- | --- | --- | --- | --- |
| dala_lite_same_context_main | budget_matched_long_cot | same_context_core_benchmarks/cot_1/overall | 1.000000 | 683.431853 | 0.831698 | available_proxy_not_exact_budget |
| dala_lite_same_context_main | budget_matched_sc | same_context_core_benchmarks/sc_5/overall | 5.000000 | 3418.253280 | 0.819442 | available_proxy_not_exact_budget |
| dala_lite_split_context_main | budget_matched_long_cot | same_context_core_benchmarks/cot_1/overall | 1.000000 | 683.431853 | 0.831698 | available_proxy_not_exact_budget |
| dala_lite_split_context_main | budget_matched_sc | same_context_core_benchmarks/sc_5/overall | 5.000000 | 3418.253280 | 0.819442 | available_proxy_not_exact_budget |
| hotpotqa_split_context_communication_necessity | budget_matched_long_cot | same_context_core_benchmarks/cot_1/overall | 1.000000 | 683.431853 | 0.831698 | available_proxy_not_exact_budget |
| hotpotqa_split_context_communication_necessity | budget_matched_sc | same_context_core_benchmarks/sc_5/overall | 5.000000 | 3418.253280 | 0.819442 | available_proxy_not_exact_budget |
| hotpotqa_split_context_communication_necessity | budget_matched_full_context_single | full_context_single |  |  | 0.653333 | same_experiment_full_context_reference |
| trigger_early_exit_main | budget_matched_long_cot | same_context_core_benchmarks/cot_1/overall | 1.000000 | 683.431853 | 0.831698 | available_proxy_not_exact_budget |
| trigger_early_exit_main | budget_matched_sc | same_context_core_benchmarks/sc_5/overall | 5.000000 | 3418.253280 | 0.819442 | available_proxy_not_exact_budget |
| voc_trigger_main | budget_matched_long_cot | same_context_core_benchmarks/cot_1/overall | 1.000000 | 683.431853 | 0.831698 | available_proxy_not_exact_budget |
| voc_trigger_main | budget_matched_sc | same_context_core_benchmarks/sc_5/overall | 5.000000 | 3418.253280 | 0.819442 | available_proxy_not_exact_budget |
| end_to_end_main | budget_matched_long_cot | same_context_core_benchmarks/cot_1/overall | 1.000000 | 683.431853 | 0.831698 | available_proxy_not_exact_budget |
| end_to_end_main | budget_matched_sc | same_context_core_benchmarks/sc_5/overall | 5.000000 | 3418.253280 | 0.819442 | available_proxy_not_exact_budget |

## 固定统计比较

| comparison_id | status | paired_n | mean_delta | delta_ci95 | wins | losses | ties | mcnemar_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_trigger_vs_always_communicate | completed | 829 | 0.001206 | [0.000000, 0.003619] | 1 | 0 | 828 | 1.000000 |
| hybrid_trigger_vs_sc_6 | completed | 829 | -0.010856 | [-0.025332, 0.003619] | 14 | 23 | 792 | 0.188445 |
| voc_trigger_v2_vs_mv_6 | completed | 829 | 0.098914 | [0.072376, 0.126659] | 112 | 30 | 687 | 0.000000 |
| dala_same_vs_all_to_all_full | completed | 829 | -0.027744 | [-0.048251, -0.008444] | 22 | 45 | 762 | 0.007194 |
| hotpot_split_comm_necessity_vs_split_no_comm_mv3 | completed | 300 | 0.150000 | [0.106667, 0.196667] | 48 | 3 | 249 | 0.000000 |
| dala_split_vs_all_to_all_full | completed | 529 | -0.064272 | [-0.096408, -0.035870] | 18 | 52 | 459 | 0.000080 |

## 有益 / 有害通信

| experiment_name | method_name | sample_method_rows | helpful | harmful | neutral | helpful_rate | harmful_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dala_lite_same_context_main | dala_lite | 829 | 104 | 12 | 713 | 0.125452 | 0.014475 |
| hotpotqa_split_context_communication_necessity | full_packet_exchange | 300 | 48 | 3 | 249 | 0.160000 | 0.010000 |
| trigger_early_exit_main | hybrid_trigger | 829 | 7 | 5 | 817 | 0.008444 | 0.006031 |
| end_to_end_main | sparc_v1 | 829 | 52 | 9 | 768 | 0.062726 | 0.010856 |
| dala_lite_split_context_main | dala_lite | 529 | 46 | 26 | 457 | 0.086957 | 0.049149 |
| voc_trigger_main | voc_trigger_v2 | 829 | 84 | 11 | 734 | 0.101327 | 0.013269 |

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

![预算前沿：同上下文 headline 方法](../../runs/faithful_matrix_iterative/20260509T125944Z-confirmatory300-xiaomimimo-mimo-v2.5/figures/budget_frontier_same_context.svg)

*Faithful score 相对于 full communication token 比率的位置关系。*

### 预算前沿：分视角 headline 方法

![预算前沿：分视角 headline 方法](../../runs/faithful_matrix_iterative/20260509T125944Z-confirmatory300-xiaomimimo-mimo-v2.5/figures/budget_frontier_split_context.svg)

*Faithful score 相对于 full communication 或匹配参照 token 比率的位置关系。*

### Trigger 收益

![Trigger 收益](../../runs/faithful_matrix_iterative/20260509T125944Z-confirmatory300-xiaomimimo-mimo-v2.5/figures/trigger_utility.svg)

*trigger 类方法相对于最优无通信基线的收益。*

### 距 stage ceiling 的差距

![距 stage ceiling 的差距](../../runs/faithful_matrix_iterative/20260509T125944Z-confirmatory300-xiaomimimo-mimo-v2.5/figures/stage_ceiling_gap.svg)

*headline 与 supporting 运行中 faithful score 到 stage ceiling 的绝对差距。*

### 有益与有害通信对比

![有益与有害通信对比](../../runs/faithful_matrix_iterative/20260509T125944Z-confirmatory300-xiaomimimo-mimo-v2.5/figures/helpful_harmful_comm.svg)

*各通信实验中有益通信率与有害通信率的并列比较。*
