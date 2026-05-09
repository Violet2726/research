# Faithful Analysis

- generated_at: `2026-05-09T12:58:57.082464+00:00`
- experiment_count: `15`

## same_context_overall

| family | experiment_name | evidence_tier | dataset | primary_method_name | faithful_score | best_no_comm_control | delta_vs_best_no_comm | full_comm_reference | delta_vs_full_comm | family_envelope | delta_vs_family_envelope | stage_ceiling | stage_ceiling_gap | engineering_noise_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| budget_comm | dala_lite_same_context_main | headline | overall | dala_lite | 0.750000 | mv_3 | 0.083333 | all_to_all_full | 0.000000 | dala_lite | 0.000000 | 0.783333 | 0.033333 |  |
| cue | cue_black_box_utility_main | diagnostic | overall | cue_v1 | 0.620000 | mv_3 | 0.050000 | always_communicate | -0.080000 | disagreement_triggered | -0.080000 | 0.630000 | 0.010000 |  |
| free_mad_lite | free_mad_lite_mechanism_validation | supporting | overall | free_mad_lite_llm_trajectory | 0.816667 | mv_3_initial | 0.150000 | vanilla_mad_r1_final_vote | 0.016667 | free_mad_lite_llm_trajectory | 0.000000 | 0.850000 | 0.033333 |  |
| multi_agent | same_context_controlled_debate | supporting | overall | mad_3a_r1 | 0.816667 | mv_6 | 0.100000 |  |  | mad_3a_r1 | 0.000000 | 0.816667 | 0.000000 |  |
| selective_comm | trigger_early_exit_main | headline | overall | hybrid_trigger | 0.866667 | mv_6 | 0.000000 | always_communicate | 0.000000 | disagreement_triggered | 0.000000 | 0.866667 | 0.000000 |  |
| selective_comm | voc_trigger_main | headline | overall | voc_trigger_v2 | 0.766667 | mv_3 | 0.050000 | always_communicate | 0.000000 | disagreement_triggered | 0.000000 | 0.783333 | 0.016666 |  |
| sid_lite | sid_lite_mechanism_validation | diagnostic | overall | sid_lite | 0.700000 | mv_3 | 0.000000 | always_full | -0.083333 | always_full | -0.083333 | 0.700000 | 0.000000 |  |
| single_agent | cross_provider_robustness | diagnostic | overall | cot_1 | 0.850000 | cot_1 | 0.000000 |  |  | cot_1 | 0.000000 | 0.850000 | 0.000000 |  |
| single_agent | same_context_core_benchmarks | reference | overall | sc_5 | 0.800000 | cot_1 | -0.050000 |  |  | cot_1 | -0.050000 | 0.800000 | 0.000000 |  |
| single_agent | same_context_main_table | reference | overall | sc_5 | 0.662500 | sc_5 | 0.000000 |  |  | sc_5 | 0.000000 | 0.662500 | 0.000000 |  |
| sparc | content_ablation | diagnostic | overall | task_adaptive | 0.683333 | mv_3 | 0.016666 | full_cot | -0.050000 | full_cot | -0.050000 | 0.716667 | 0.033334 |  |
| sparc | end_to_end_main | headline | overall | sparc_v1 | 0.716667 | mv_3 | 0.050000 | always_communicate | -0.016666 | always_communicate | -0.016666 | 0.733333 | 0.016666 |  |
| sparc | local_auditing_ablation | supporting | overall | local_auditing | 0.733333 | majority_vote | 0.066666 | final_round_vote | 0.033333 | single_judge | -0.050000 | 0.750000 | 0.016667 |  |

## split_context_overall

| family | experiment_name | evidence_tier | dataset | primary_method_name | faithful_score | best_no_comm_control | delta_vs_best_no_comm | full_comm_reference | delta_vs_full_comm | family_envelope | delta_vs_family_envelope | stage_ceiling | stage_ceiling_gap | engineering_noise_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| budget_comm | dala_lite_split_context_main | headline | overall | dala_lite | 0.650000 | mv_3 | 0.100000 | all_to_all_full | -0.075000 | all_to_all_full | -0.075000 | 0.675000 | 0.025000 |  |
| comm_necessary | hotpotqa_split_context_communication_necessity | headline | overall | full_packet_exchange | 0.500000 | split_no_comm_mv3 | 0.100000 |  |  | full_context_single | -0.200000 | 0.500000 | 0.000000 |  |

## same_context_rows

| family | experiment_name | evidence_tier | dataset | primary_method_name | faithful_score | best_no_comm_control | delta_vs_best_no_comm | full_comm_reference | delta_vs_full_comm | family_envelope | delta_vs_family_envelope | stage_ceiling | stage_ceiling_gap | engineering_noise_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| budget_comm | dala_lite_same_context_main | headline | gsm8k | dala_lite | 0.750000 | mv_3 | 0.150000 | all_to_all_full | -0.100000 | dala_lite | 0.000000 | 0.800000 | 0.050000 |  |
| budget_comm | dala_lite_same_context_main | headline | hotpotqa | dala_lite | 0.750000 | mv_3 | 0.000000 | all_to_all_full | 0.000000 | dala_lite | 0.000000 | 0.800000 | 0.050000 |  |
| budget_comm | dala_lite_same_context_main | headline | strategyqa | dala_lite | 0.750000 | mv_3 | 0.100000 | all_to_all_full | 0.100000 | dala_lite | 0.000000 | 0.750000 | 0.000000 |  |
| cue | cue_black_box_utility_main | diagnostic | gsm8k | cue_v1 | 0.750000 | mv_3 | 0.200000 | always_communicate | -0.050000 | disagreement_triggered | -0.050000 | 0.750000 | 0.000000 |  |
| cue | cue_black_box_utility_main | diagnostic | hotpotqa | cue_v1 | 0.700000 | mv_3 | 0.000000 | always_communicate | -0.050000 | disagreement_triggered | -0.050000 | 0.700000 | 0.000000 |  |
| cue | cue_black_box_utility_main | diagnostic | math500 | cue_v1 | 0.400000 | mv_3 | 0.050000 | always_communicate | -0.150000 | disagreement_triggered | -0.150000 | 0.400000 | 0.000000 |  |
| cue | cue_black_box_utility_main | diagnostic | mmlu_pro | cue_v1 | 0.550000 | mv_3 | 0.000000 | always_communicate | -0.150000 | disagreement_triggered | -0.150000 | 0.600000 | 0.050000 |  |
| cue | cue_black_box_utility_main | diagnostic | strategyqa | cue_v1 | 0.700000 | mv_3 | 0.000000 | always_communicate | 0.000000 | disagreement_triggered | 0.000000 | 0.700000 | 0.000000 |  |
| free_mad_lite | free_mad_lite_mechanism_validation | supporting | gsm8k | free_mad_lite_llm_trajectory | 0.950000 | mv_3_initial | 0.350000 | vanilla_mad_r1_final_vote | 0.100000 | free_mad_lite_llm_trajectory | 0.000000 | 0.950000 | 0.000000 |  |
| free_mad_lite | free_mad_lite_mechanism_validation | supporting | hotpotqa | free_mad_lite_llm_trajectory | 0.700000 | mv_3_initial | 0.000000 | vanilla_mad_r1_final_vote | -0.050000 | free_mad_lite_llm_trajectory | 0.000000 | 0.750000 | 0.050000 |  |
| free_mad_lite | free_mad_lite_mechanism_validation | supporting | strategyqa | free_mad_lite_llm_trajectory | 0.800000 | mv_3_initial | 0.100000 | vanilla_mad_r1_final_vote | 0.000000 | free_mad_lite_llm_trajectory | 0.000000 | 0.850000 | 0.050000 |  |
| multi_agent | same_context_controlled_debate | supporting | gsm8k | mad_3a_r1 | 0.950000 | mv_6 | 0.300000 |  |  | mad_3a_r1 | 0.000000 | 0.950000 | 0.000000 |  |
| multi_agent | same_context_controlled_debate | supporting | hotpotqa | mad_3a_r1 | 0.800000 | mv_6 | 0.000000 |  |  | mad_3a_r1 | 0.000000 | 0.800000 | 0.000000 |  |
| multi_agent | same_context_controlled_debate | supporting | strategyqa | mad_3a_r1 | 0.700000 | mv_6 | 0.000000 |  |  | mad_3a_r1 | 0.000000 | 0.700000 | 0.000000 |  |
| selective_comm | trigger_early_exit_main | headline | gsm8k | hybrid_trigger | 1.000000 | mv_6 | 0.000000 | always_communicate | 0.000000 | disagreement_triggered | 0.000000 | 1.000000 | 0.000000 |  |
| selective_comm | trigger_early_exit_main | headline | hotpotqa | hybrid_trigger | 0.750000 | mv_6 | -0.100000 | always_communicate | 0.000000 | disagreement_triggered | 0.000000 | 0.750000 | 0.000000 |  |
| selective_comm | trigger_early_exit_main | headline | strategyqa | hybrid_trigger | 0.850000 | mv_6 | 0.100000 | always_communicate | 0.000000 | disagreement_triggered | 0.000000 | 0.850000 | 0.000000 |  |
| selective_comm | voc_trigger_main | headline | gsm8k | voc_trigger_v2 | 0.700000 | mv_3 | 0.100000 | always_communicate | 0.000000 | disagreement_triggered | 0.000000 | 0.750000 | 0.050000 |  |
| selective_comm | voc_trigger_main | headline | hotpotqa | voc_trigger_v2 | 0.750000 | mv_3 | 0.000000 | always_communicate | 0.000000 | disagreement_triggered | 0.000000 | 0.750000 | 0.000000 |  |
| selective_comm | voc_trigger_main | headline | strategyqa | voc_trigger_v2 | 0.850000 | mv_3 | 0.050000 | always_communicate | 0.000000 | disagreement_triggered | 0.000000 | 0.850000 | 0.000000 |  |
| sid_lite | sid_lite_mechanism_validation | diagnostic | gsm8k | sid_lite | 0.700000 | mv_3 | 0.000000 | always_full | -0.200000 | always_full | -0.200000 | 0.700000 | 0.000000 |  |
| sid_lite | sid_lite_mechanism_validation | diagnostic | hotpotqa | sid_lite | 0.700000 | mv_3 | 0.000000 | always_full | -0.050000 | always_full | -0.050000 | 0.700000 | 0.000000 |  |
| sid_lite | sid_lite_mechanism_validation | diagnostic | strategyqa | sid_lite | 0.700000 | mv_3 | 0.000000 | always_full | 0.000000 | always_full | 0.000000 | 0.700000 | 0.000000 |  |
| single_agent | cross_provider_robustness | diagnostic | gsm8k | cot_1 | 1.000000 | cot_1 | 0.000000 |  |  | cot_1 | 0.000000 | 1.000000 | 0.000000 |  |
| single_agent | cross_provider_robustness | diagnostic | hotpotqa | cot_1 | 0.700000 | cot_1 | 0.000000 |  |  | cot_1 | 0.000000 | 0.700000 | 0.000000 |  |
| single_agent | cross_provider_robustness | diagnostic | strategyqa | cot_1 | 0.850000 | cot_1 | 0.000000 |  |  | cot_1 | 0.000000 | 0.850000 | 0.000000 |  |
| single_agent | same_context_core_benchmarks | reference | gsm8k | sc_5 | 0.950000 | cot_1 | -0.050000 |  |  | cot_1 | -0.050000 | 0.950000 | 0.000000 |  |
| single_agent | same_context_core_benchmarks | reference | hotpotqa | sc_5 | 0.750000 | cot_1 | 0.050000 |  |  | cot_1 | 0.050000 | 0.750000 | 0.000000 |  |
| single_agent | same_context_core_benchmarks | reference | strategyqa | sc_5 | 0.700000 | cot_1 | -0.150000 |  |  | cot_1 | -0.150000 | 0.700000 | 0.000000 |  |
| single_agent | same_context_main_table | reference | gpqa_diamond | sc_5 | 0.450000 | sc_5 | 0.000000 |  |  | sc_5 | 0.000000 | 0.450000 | 0.000000 |  |
| single_agent | same_context_main_table | reference | hotpotqa | sc_5 | 0.750000 | sc_5 | 0.000000 |  |  | sc_5 | 0.000000 | 0.750000 | 0.000000 |  |
| single_agent | same_context_main_table | reference | math500 | sc_5 | 0.700000 | sc_5 | 0.000000 |  |  | sc_5 | 0.000000 | 0.700000 | 0.000000 |  |
| single_agent | same_context_main_table | reference | mmlu_pro | sc_5 | 0.750000 | sc_5 | 0.000000 |  |  | sc_5 | 0.000000 | 0.750000 | 0.000000 |  |
| sparc | content_ablation | diagnostic | gsm8k | task_adaptive | 0.700000 | mv_3 | 0.050000 | full_cot | -0.100000 | full_cot | -0.100000 | 0.800000 | 0.100000 |  |
| sparc | content_ablation | diagnostic | hotpotqa | task_adaptive | 0.650000 | mv_3 | 0.000000 | full_cot | -0.050000 | full_cot | -0.050000 | 0.650000 | 0.000000 |  |
| sparc | content_ablation | diagnostic | strategyqa | task_adaptive | 0.700000 | mv_3 | 0.000000 | full_cot | 0.000000 | full_cot | 0.000000 | 0.700000 | 0.000000 |  |
| sparc | end_to_end_main | headline | gsm8k | sparc_v1 | 0.750000 | mv_3 | 0.100000 | always_communicate | 0.000000 | always_communicate | 0.000000 | 0.750000 | 0.000000 |  |
| sparc | end_to_end_main | headline | hotpotqa | sparc_v1 | 0.700000 | mv_3 | 0.050000 | always_communicate | -0.050000 | always_communicate | -0.050000 | 0.700000 | 0.000000 |  |
| sparc | end_to_end_main | headline | strategyqa | sparc_v1 | 0.700000 | mv_3 | 0.000000 | always_communicate | 0.000000 | always_communicate | 0.000000 | 0.750000 | 0.050000 |  |
| sparc | local_auditing_ablation | supporting | gsm8k | local_auditing | 0.750000 | majority_vote | 0.100000 | final_round_vote | 0.050000 | single_judge | -0.200000 | 0.750000 | 0.000000 |  |
| sparc | local_auditing_ablation | supporting | hotpotqa | local_auditing | 0.750000 | majority_vote | 0.100000 | final_round_vote | 0.050000 | single_judge | 0.050000 | 0.750000 | 0.000000 |  |
| sparc | local_auditing_ablation | supporting | strategyqa | local_auditing | 0.700000 | majority_vote | 0.000000 | final_round_vote | 0.000000 | single_judge | 0.000000 | 0.750000 | 0.050000 |  |

## split_context_rows

| family | experiment_name | evidence_tier | dataset | primary_method_name | faithful_score | best_no_comm_control | delta_vs_best_no_comm | full_comm_reference | delta_vs_full_comm | family_envelope | delta_vs_family_envelope | stage_ceiling | stage_ceiling_gap | engineering_noise_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| budget_comm | dala_lite_split_context_main | headline | hotpotqa | dala_lite | 0.450000 | mv_3 | 0.200000 | all_to_all_full | -0.150000 | all_to_all_full | -0.150000 | 0.500000 | 0.050000 |  |
| budget_comm | dala_lite_split_context_main | headline | strategyqa | dala_lite | 0.850000 | mv_3 | 0.000000 | all_to_all_full | 0.000000 | all_to_all_full | 0.000000 | 0.850000 | 0.000000 |  |
| comm_necessary | hotpotqa_split_context_communication_necessity | headline | hotpotqa | full_packet_exchange | 0.500000 | split_no_comm_mv3 | 0.100000 |  |  | full_context_single | -0.200000 | 0.500000 | 0.000000 |  |
