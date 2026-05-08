# Paper Package

- generated_at: `2026-05-08T02:31:04.721594+00:00`
- phase_name: `pilot100`
- model_ref: `xiaomimimo/mimo-v2.5`
- counts: `{"completed": 16, "excluded": 0, "semantic_unique_targets": 16}`

## Same-Context Main Table

| experiment_name | primary_method_name | faithful_score | delta_vs_best_no_comm | delta_vs_full_comm | delta_vs_full_context | token_ratio_vs_full_comm | calls_per_question_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dala_lite_same_context_v1 | dala_lite | 0.710000 | 0.113333 | -0.013333 |  | 0.918893 | 6.000000 |
| trigger_early_exit_v1 | hybrid_trigger | 0.840000 | -0.016667 | 0.000000 |  | 0.546980 | 3.540000 |
| trigger_voc_v2 | voc_trigger_v2 | 0.730000 | 0.080000 | -0.010000 |  | 0.822325 | 4.530000 |
| sparc_v1_smoke | sparc_v1 | 0.666667 | 0.040000 | 0.000000 |  | 0.657197 | 4.370000 |

## Split-Context Main Table

| experiment_name | primary_method_name | faithful_score | delta_vs_best_no_comm | delta_vs_full_comm | delta_vs_full_context | token_ratio_vs_full_comm | calls_per_question_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dala_lite_split_context_v1 | dala_lite | 0.695000 | 0.085000 | -0.055000 |  | 0.896393 | 6.000000 |
| hotpotqa_split_main | full_packet_exchange | 0.610000 | 0.210000 |  | -0.070000 |  | 6.000000 |

## Supporting Evidence

| experiment_name | primary_method_name | faithful_score | delta_vs_best_no_comm | delta_vs_full_comm | delta_vs_full_context | token_ratio_vs_full_comm | calls_per_question_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| free_mad_lite_v1 | free_mad_lite_llm_trajectory | 0.786667 | 0.160000 | 0.033334 |  | 1.287528 | 7.000000 |
| multi_agent_main | mad_3a_r1 | 0.800000 | 0.130000 |  |  |  | 6.000000 |
| aggregation_auditing_ablation_v1 | local_auditing | 0.666667 | 0.040000 | 0.040000 |  | 1.045324 | 6.260000 |
| auditing_ablation_v1 | local_auditing | 0.666667 | 0.040000 | 0.040000 |  | 1.045324 | 6.260000 |

## Diagnostic Evidence

| experiment_name | primary_method_name | faithful_score | delta_vs_best_no_comm | delta_vs_full_comm | delta_vs_full_context | token_ratio_vs_full_comm | calls_per_question_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cue_v1 | cue_v1 | 0.584000 | 0.030000 | -0.050000 |  | 0.521509 | 3.776000 |
| sid_lite_v1 | sid_lite | 0.620000 | 0.026667 | -0.093333 |  | 0.580106 | 4.110000 |
| robustness | cot_1 | 0.820000 | 0.000000 |  |  |  | 1.000000 |
| content_ablation_v1 | task_adaptive | 0.623333 | -0.003334 | -0.096667 |  | 0.899042 | 6.000000 |

## Equal-Budget Single-Agent References

| experiment_name | baseline_name | matched_budget_source | matched_calls_per_q | matched_total_tokens_mean | score | budget_match_status |
| --- | --- | --- | --- | --- | --- | --- |
| dala_lite_same_context_v1 | budget_matched_long_cot | main_baselines/cot_1/overall | 1.000000 | 675.296667 | 0.820000 | available_proxy_not_exact_budget |
| dala_lite_same_context_v1 | budget_matched_sc | main_baselines/sc_5/overall | 5.000000 | 3380.493333 | 0.830000 | available_proxy_not_exact_budget |
| dala_lite_split_context_v1 | budget_matched_long_cot | main_baselines/cot_1/overall | 1.000000 | 675.296667 | 0.820000 | available_proxy_not_exact_budget |
| dala_lite_split_context_v1 | budget_matched_sc | main_baselines/sc_5/overall | 5.000000 | 3380.493333 | 0.830000 | available_proxy_not_exact_budget |
| hotpotqa_split_main | budget_matched_long_cot | main_baselines/cot_1/overall | 1.000000 | 675.296667 | 0.820000 | available_proxy_not_exact_budget |
| hotpotqa_split_main | budget_matched_sc | main_baselines/sc_5/overall | 5.000000 | 3380.493333 | 0.830000 | available_proxy_not_exact_budget |
| hotpotqa_split_main | budget_matched_full_context_single | full_context_single |  |  | 0.680000 | same_experiment_full_context_reference |
| trigger_early_exit_v1 | budget_matched_long_cot | main_baselines/cot_1/overall | 1.000000 | 675.296667 | 0.820000 | available_proxy_not_exact_budget |
| trigger_early_exit_v1 | budget_matched_sc | main_baselines/sc_5/overall | 5.000000 | 3380.493333 | 0.830000 | available_proxy_not_exact_budget |
| trigger_voc_v2 | budget_matched_long_cot | main_baselines/cot_1/overall | 1.000000 | 675.296667 | 0.820000 | available_proxy_not_exact_budget |
| trigger_voc_v2 | budget_matched_sc | main_baselines/sc_5/overall | 5.000000 | 3380.493333 | 0.830000 | available_proxy_not_exact_budget |
| sparc_v1_smoke | budget_matched_long_cot | main_baselines/cot_1/overall | 1.000000 | 675.296667 | 0.820000 | available_proxy_not_exact_budget |
| sparc_v1_smoke | budget_matched_sc | main_baselines/sc_5/overall | 5.000000 | 3380.493333 | 0.830000 | available_proxy_not_exact_budget |

## Fixed Statistical Comparisons

| comparison_id | status | paired_n | mean_delta | delta_ci95 | wins | losses | ties | mcnemar_p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hybrid_trigger_vs_always_communicate | completed | 300 | 0.000000 | [0.000000, 0.000000] | 0 | 0 | 300 | 1.000000 |
| hybrid_trigger_vs_sc_6 | completed | 300 | -0.016667 | [-0.043333, 0.010000] | 6 | 11 | 283 | 0.331975 |
| voc_trigger_v2_vs_mv_6 | completed | 300 | 0.080000 | [0.036583, 0.126750] | 39 | 15 | 246 | 0.001749 |
| dala_same_vs_all_to_all_full | completed | 300 | -0.013333 | [-0.046667, 0.020000] | 10 | 14 | 276 | 0.540291 |
| hotpotqa_split_main_vs_split_no_comm_mv3 | completed | 100 | 0.210000 | [0.140000, 0.290000] | 21 | 0 | 79 | 0.000013 |
| dala_split_vs_all_to_all_full | completed | 200 | -0.055000 | [-0.100000, -0.010000] | 5 | 16 | 179 | 0.029096 |

## Helpful / Harmful Communication

| experiment_name | method_name | sample_method_rows | helpful | harmful | neutral | helpful_rate | harmful_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dala_lite_same_context_v1 | dala_lite | 300 | 37 | 3 | 260 | 0.123333 | 0.010000 |
| hotpotqa_split_main | full_packet_exchange | 100 | 21 | 0 | 79 | 0.210000 | 0.000000 |
| trigger_early_exit_v1 | hybrid_trigger | 300 | 4 | 5 | 291 | 0.013333 | 0.016667 |
| sparc_v1_smoke | sparc_v1 | 300 | 18 | 6 | 276 | 0.060000 | 0.020000 |
| dala_lite_split_context_v1 | dala_lite | 200 | 23 | 6 | 171 | 0.115000 | 0.030000 |
| trigger_voc_v2 | voc_trigger_v2 | 300 | 38 | 0 | 262 | 0.126667 | 0.000000 |

## Figures

- `budget_frontier_same_context`: `reports/figures/20260507T121658Z-pilot100-xiaomimimo-mimo-v2.5/budget_frontier_same_context.svg`
- `budget_frontier_same_context_data`: `reports/figures/20260507T121658Z-pilot100-xiaomimimo-mimo-v2.5/budget_frontier_same_context.csv`
- `budget_frontier_split_context`: `reports/figures/20260507T121658Z-pilot100-xiaomimimo-mimo-v2.5/budget_frontier_split_context.svg`
- `budget_frontier_split_context_data`: `reports/figures/20260507T121658Z-pilot100-xiaomimimo-mimo-v2.5/budget_frontier_split_context.csv`
- `helpful_harmful_comm`: `reports/figures/20260507T121658Z-pilot100-xiaomimimo-mimo-v2.5/helpful_harmful_comm.svg`
- `helpful_harmful_comm_data`: `reports/figures/20260507T121658Z-pilot100-xiaomimimo-mimo-v2.5/helpful_harmful_comm.csv`
- `stage_ceiling_gap`: `reports/figures/20260507T121658Z-pilot100-xiaomimimo-mimo-v2.5/stage_ceiling_gap.svg`
- `stage_ceiling_gap_data`: `reports/figures/20260507T121658Z-pilot100-xiaomimimo-mimo-v2.5/stage_ceiling_gap.csv`
- `trigger_utility`: `reports/figures/20260507T121658Z-pilot100-xiaomimimo-mimo-v2.5/trigger_utility.svg`
- `trigger_utility_data`: `reports/figures/20260507T121658Z-pilot100-xiaomimimo-mimo-v2.5/trigger_utility.csv`

## Interpretation Guardrails

- Headline tables are the only rows intended for main-text claims.
- Supporting and diagnostic rows explain mechanisms, limits, and ablations.
- Equal-budget single-agent rows are evaluation controls, not new method steps.
- If confirmatory results weaken a headline claim, demote the claim rather than changing the method graph.
