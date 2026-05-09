# Faithful Analysis

- generated_at: `2026-05-09T12:59:41.726176+00:00`
- experiment_count: `15`

## same_context_overall

| family | experiment_name | evidence_tier | dataset | primary_method_name | faithful_score | best_no_comm_control | delta_vs_best_no_comm | full_comm_reference | delta_vs_full_comm | family_envelope | delta_vs_family_envelope | stage_ceiling | stage_ceiling_gap | engineering_noise_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| budget_comm | dala_lite_same_context_main | headline | overall | dala_lite | 0.696667 | mv_3 | 0.093334 | all_to_all_full | -0.016666 | all_to_all_full | -0.016666 | 0.713333 | 0.016666 | -0.053333 |
| cue | cue_black_box_utility_main | diagnostic | overall | cue_v1 | 0.578000 | mv_3 | 0.030000 | always_communicate | -0.058000 | always_communicate | -0.058000 | 0.588000 | 0.010000 | -0.042000 |
| free_mad_lite | free_mad_lite_mechanism_validation | supporting | overall | free_mad_lite_llm_trajectory | 0.770000 | mv_3_initial | 0.140000 | vanilla_mad_r1_final_vote | 0.006667 | free_mad_lite_llm_trajectory | 0.000000 | 0.796667 | 0.026667 | -0.046667 |
| multi_agent | same_context_controlled_debate | supporting | overall | mad_3a_r1 | 0.770000 | mv_6 | 0.090000 |  |  | mad_3a_r1 | 0.000000 | 0.776667 | 0.006667 |  |
| selective_comm | trigger_early_exit_main | headline | overall | hybrid_trigger | 0.846667 | mv_6 | -0.010000 | always_communicate | 0.000000 | mv_6 | -0.010000 | 0.853333 | 0.006666 | -0.020000 |
| selective_comm | voc_trigger_main | headline | overall | voc_trigger_v2 | 0.720000 | mv_3 | 0.073333 | always_communicate | -0.006667 | always_communicate | -0.006667 | 0.736667 | 0.016667 | -0.046667 |
| sid_lite | sid_lite_mechanism_validation | diagnostic | overall | sid_lite | 0.623333 | mv_3 | 0.010000 | always_full | -0.080000 | always_full | -0.080000 | 0.630000 | 0.006667 | -0.076667 |
| single_agent | cross_provider_robustness | diagnostic | overall | cot_1 | 0.833333 | cot_1 | 0.000000 |  |  | cot_1 | 0.000000 | 0.833333 | 0.000000 |  |
| single_agent | same_context_core_benchmarks | reference | overall | sc_5 | 0.806667 | cot_1 | -0.026666 |  |  | cot_1 | -0.026666 | 0.806667 | 0.000000 |  |
| single_agent | same_context_main_table | reference | overall | sc_5 | 0.675000 | sc_5 | 0.000000 |  |  | sc_5 | 0.000000 | 0.675000 | 0.000000 |  |
| sparc | content_ablation | diagnostic | overall | task_adaptive | 0.640000 | mv_3 | 0.000000 | full_cot | -0.090000 | full_cot | -0.090000 | 0.660000 | 0.020000 | -0.043333 |
| sparc | end_to_end_main | headline | overall | sparc_v1 | 0.676667 | mv_3 | 0.036667 | always_communicate | -0.003333 | always_communicate | -0.003333 | 0.696667 | 0.020000 | -0.040000 |
| sparc | local_auditing_ablation | supporting | overall | local_auditing | 0.680000 | majority_vote | 0.040000 | final_round_vote | 0.016667 | single_judge | -0.050000 | 0.700000 | 0.020000 | -0.053333 |

## split_context_overall

| family | experiment_name | evidence_tier | dataset | primary_method_name | faithful_score | best_no_comm_control | delta_vs_best_no_comm | full_comm_reference | delta_vs_full_comm | family_envelope | delta_vs_family_envelope | stage_ceiling | stage_ceiling_gap | engineering_noise_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| budget_comm | dala_lite_split_context_main | headline | overall | dala_lite | 0.675000 | mv_3 | 0.070000 | all_to_all_full | -0.065000 | all_to_all_full | -0.065000 | 0.720000 | 0.045000 | 0.025000 |
| comm_necessary | hotpotqa_split_context_communication_necessity | headline | overall | full_packet_exchange | 0.590000 | split_no_comm_mv3 | 0.200000 |  |  | full_context_single | -0.100000 | 0.590000 | 0.000000 | 0.090000 |

## same_context_rows

| family | experiment_name | evidence_tier | dataset | primary_method_name | faithful_score | best_no_comm_control | delta_vs_best_no_comm | full_comm_reference | delta_vs_full_comm | family_envelope | delta_vs_family_envelope | stage_ceiling | stage_ceiling_gap | engineering_noise_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| budget_comm | dala_lite_same_context_main | headline | gsm8k | dala_lite | 0.760000 | mv_3 | 0.230000 | all_to_all_full | -0.100000 | all_to_all_full | -0.100000 | 0.770000 | 0.010000 | 0.010000 |
| budget_comm | dala_lite_same_context_main | headline | hotpotqa | dala_lite | 0.660000 | mv_3 | 0.010000 | all_to_all_full | 0.010000 | all_to_all_full | 0.010000 | 0.670000 | 0.010000 | -0.090000 |
| budget_comm | dala_lite_same_context_main | headline | strategyqa | dala_lite | 0.670000 | mv_3 | 0.040000 | all_to_all_full | 0.040000 | all_to_all_full | 0.040000 | 0.700000 | 0.030000 | -0.080000 |
| cue | cue_black_box_utility_main | diagnostic | gsm8k | cue_v1 | 0.600000 | mv_3 | 0.110000 | always_communicate | -0.120000 | always_communicate | -0.120000 | 0.610000 | 0.010000 | -0.150000 |
| cue | cue_black_box_utility_main | diagnostic | hotpotqa | cue_v1 | 0.660000 | mv_3 | 0.000000 | always_communicate | -0.020000 | always_communicate | -0.020000 | 0.670000 | 0.010000 | -0.040000 |
| cue | cue_black_box_utility_main | diagnostic | math500 | cue_v1 | 0.330000 | mv_3 | 0.030000 | always_communicate | -0.100000 | always_communicate | -0.100000 | 0.340000 | 0.010000 | -0.070000 |
| cue | cue_black_box_utility_main | diagnostic | mmlu_pro | cue_v1 | 0.610000 | mv_3 | -0.010000 | always_communicate | -0.050000 | always_communicate | -0.050000 | 0.630000 | 0.020000 | 0.060000 |
| cue | cue_black_box_utility_main | diagnostic | strategyqa | cue_v1 | 0.690000 | mv_3 | 0.020000 | always_communicate | 0.000000 | always_communicate | 0.000000 | 0.690000 | 0.000000 | -0.010000 |
| free_mad_lite | free_mad_lite_mechanism_validation | supporting | gsm8k | free_mad_lite_llm_trajectory | 0.900000 | mv_3_initial | 0.340000 | vanilla_mad_r1_final_vote | 0.040000 | free_mad_lite_llm_trajectory | 0.000000 | 0.900000 | 0.000000 | -0.050000 |
| free_mad_lite | free_mad_lite_mechanism_validation | supporting | hotpotqa | free_mad_lite_llm_trajectory | 0.700000 | mv_3_initial | 0.020000 | vanilla_mad_r1_final_vote | -0.020000 | free_mad_lite_llm_trajectory | 0.000000 | 0.740000 | 0.040000 | 0.000000 |
| free_mad_lite | free_mad_lite_mechanism_validation | supporting | strategyqa | free_mad_lite_llm_trajectory | 0.710000 | mv_3_initial | 0.060000 | vanilla_mad_r1_final_vote | 0.000000 | free_mad_lite_llm_trajectory | 0.000000 | 0.750000 | 0.040000 | -0.090000 |
| multi_agent | same_context_controlled_debate | supporting | gsm8k | mad_3a_r1 | 0.870000 | mv_6 | 0.210000 |  |  | mad_3a_r1 | 0.000000 | 0.870000 | 0.000000 | -0.080000 |
| multi_agent | same_context_controlled_debate | supporting | hotpotqa | mad_3a_r1 | 0.730000 | mv_6 | 0.050000 |  |  | mad_3a_r1 | 0.000000 | 0.740000 | 0.010000 | -0.070000 |
| multi_agent | same_context_controlled_debate | supporting | strategyqa | mad_3a_r1 | 0.710000 | mv_6 | 0.010000 |  |  | mad_3a_r1 | 0.000000 | 0.720000 | 0.010000 | 0.010000 |
| selective_comm | trigger_early_exit_main | headline | gsm8k | hybrid_trigger | 0.970000 | mv_6 | 0.000000 | always_communicate | 0.000000 | mv_6 | 0.000000 | 0.970000 | 0.000000 | -0.030000 |
| selective_comm | trigger_early_exit_main | headline | hotpotqa | hybrid_trigger | 0.740000 | mv_6 | -0.030000 | always_communicate | 0.000000 | mv_6 | -0.030000 | 0.750000 | 0.010000 | -0.010000 |
| selective_comm | trigger_early_exit_main | headline | strategyqa | hybrid_trigger | 0.830000 | mv_6 | 0.000000 | always_communicate | 0.000000 | mv_6 | 0.000000 | 0.840000 | 0.010000 | -0.020000 |
| selective_comm | voc_trigger_main | headline | gsm8k | voc_trigger_v2 | 0.770000 | mv_3 | 0.230000 | always_communicate | -0.010000 | always_communicate | -0.010000 | 0.790000 | 0.020000 | 0.070000 |
| selective_comm | voc_trigger_main | headline | hotpotqa | voc_trigger_v2 | 0.680000 | mv_3 | -0.010000 | always_communicate | 0.000000 | always_communicate | 0.000000 | 0.690000 | 0.010000 | -0.070000 |
| selective_comm | voc_trigger_main | headline | strategyqa | voc_trigger_v2 | 0.710000 | mv_3 | 0.000000 | always_communicate | -0.010000 | always_communicate | -0.010000 | 0.730000 | 0.020000 | -0.140000 |
| sid_lite | sid_lite_mechanism_validation | diagnostic | gsm8k | sid_lite | 0.540000 | mv_3 | 0.020000 | always_full | -0.210000 | always_full | -0.210000 | 0.560000 | 0.020000 | -0.160000 |
| sid_lite | sid_lite_mechanism_validation | diagnostic | hotpotqa | sid_lite | 0.670000 | mv_3 | 0.010000 | always_full | -0.020000 | always_full | -0.020000 | 0.670000 | 0.000000 | -0.030000 |
| sid_lite | sid_lite_mechanism_validation | diagnostic | strategyqa | sid_lite | 0.660000 | mv_3 | 0.000000 | always_full | -0.010000 | always_full | -0.010000 | 0.660000 | 0.000000 | -0.040000 |
| single_agent | cross_provider_robustness | diagnostic | gsm8k | cot_1 | 0.960000 | cot_1 | 0.000000 |  |  | cot_1 | 0.000000 | 0.960000 | 0.000000 | -0.040000 |
| single_agent | cross_provider_robustness | diagnostic | hotpotqa | cot_1 | 0.710000 | cot_1 | 0.000000 |  |  | cot_1 | 0.000000 | 0.710000 | 0.000000 | 0.010000 |
| single_agent | cross_provider_robustness | diagnostic | strategyqa | cot_1 | 0.830000 | cot_1 | 0.000000 |  |  | cot_1 | 0.000000 | 0.830000 | 0.000000 | -0.020000 |
| single_agent | same_context_core_benchmarks | reference | gsm8k | sc_5 | 0.960000 | cot_1 | 0.000000 |  |  | cot_1 | 0.000000 | 0.960000 | 0.000000 | 0.010000 |
| single_agent | same_context_core_benchmarks | reference | hotpotqa | sc_5 | 0.730000 | cot_1 | 0.020000 |  |  | cot_1 | 0.020000 | 0.730000 | 0.000000 | -0.020000 |
| single_agent | same_context_core_benchmarks | reference | strategyqa | sc_5 | 0.730000 | cot_1 | -0.100000 |  |  | cot_1 | -0.100000 | 0.730000 | 0.000000 | 0.030000 |
| single_agent | same_context_main_table | reference | gpqa_diamond | sc_5 | 0.550000 | sc_5 | 0.000000 |  |  | sc_5 | 0.000000 | 0.550000 | 0.000000 | 0.100000 |
| single_agent | same_context_main_table | reference | hotpotqa | sc_5 | 0.730000 | sc_5 | 0.000000 |  |  | sc_5 | 0.000000 | 0.730000 | 0.000000 | -0.020000 |
| single_agent | same_context_main_table | reference | math500 | sc_5 | 0.660000 | sc_5 | 0.000000 |  |  | sc_5 | 0.000000 | 0.660000 | 0.000000 | -0.040000 |
| single_agent | same_context_main_table | reference | mmlu_pro | sc_5 | 0.760000 | sc_5 | 0.000000 |  |  | sc_5 | 0.000000 | 0.760000 | 0.000000 | 0.010000 |
| sparc | content_ablation | diagnostic | gsm8k | task_adaptive | 0.570000 | mv_3 | 0.000000 | full_cot | -0.260000 | full_cot | -0.260000 | 0.630000 | 0.060000 | -0.130000 |
| sparc | content_ablation | diagnostic | hotpotqa | task_adaptive | 0.660000 | mv_3 | 0.000000 | full_cot | -0.020000 | full_cot | -0.020000 | 0.660000 | 0.000000 | 0.010000 |
| sparc | content_ablation | diagnostic | strategyqa | task_adaptive | 0.690000 | mv_3 | 0.000000 | full_cot | 0.010000 | full_cot | 0.010000 | 0.690000 | 0.000000 | -0.010000 |
| sparc | end_to_end_main | headline | gsm8k | sparc_v1 | 0.650000 | mv_3 | 0.080000 | always_communicate | 0.000000 | always_communicate | 0.000000 | 0.670000 | 0.020000 | -0.100000 |
| sparc | end_to_end_main | headline | hotpotqa | sparc_v1 | 0.690000 | mv_3 | 0.030000 | always_communicate | -0.010000 | always_communicate | -0.010000 | 0.690000 | 0.000000 | -0.010000 |
| sparc | end_to_end_main | headline | strategyqa | sparc_v1 | 0.690000 | mv_3 | 0.000000 | always_communicate | 0.000000 | always_communicate | 0.000000 | 0.730000 | 0.040000 | -0.010000 |
| sparc | local_auditing_ablation | supporting | gsm8k | local_auditing | 0.650000 | majority_vote | 0.080000 | final_round_vote | 0.060000 | single_judge | -0.120000 | 0.670000 | 0.020000 | -0.100000 |
| sparc | local_auditing_ablation | supporting | hotpotqa | local_auditing | 0.700000 | majority_vote | 0.040000 | final_round_vote | 0.020000 | single_judge | 0.000000 | 0.700000 | 0.000000 | -0.050000 |
| sparc | local_auditing_ablation | supporting | strategyqa | local_auditing | 0.690000 | majority_vote | 0.000000 | final_round_vote | -0.030000 | single_judge | -0.030000 | 0.730000 | 0.040000 | -0.010000 |

## split_context_rows

| family | experiment_name | evidence_tier | dataset | primary_method_name | faithful_score | best_no_comm_control | delta_vs_best_no_comm | full_comm_reference | delta_vs_full_comm | family_envelope | delta_vs_family_envelope | stage_ceiling | stage_ceiling_gap | engineering_noise_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| budget_comm | dala_lite_split_context_main | headline | hotpotqa | dala_lite | 0.480000 | mv_3 | 0.080000 | all_to_all_full | -0.130000 | all_to_all_full | -0.130000 | 0.520000 | 0.040000 | 0.030000 |
| budget_comm | dala_lite_split_context_main | headline | strategyqa | dala_lite | 0.870000 | mv_3 | 0.060000 | all_to_all_full | 0.000000 | all_to_all_full | 0.000000 | 0.920000 | 0.050000 | 0.020000 |
| comm_necessary | hotpotqa_split_context_communication_necessity | headline | hotpotqa | full_packet_exchange | 0.590000 | split_no_comm_mv3 | 0.200000 |  |  | full_context_single | -0.100000 | 0.590000 | 0.000000 | 0.090000 |
