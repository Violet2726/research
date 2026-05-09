# Faithful Analysis

- generated_at: `2026-05-09T13:01:18.393961+00:00`
- experiment_count: `15`

## same_context_overall

| family | experiment_name | evidence_tier | dataset | primary_method_name | faithful_score | best_no_comm_control | delta_vs_best_no_comm | full_comm_reference | delta_vs_full_comm | family_envelope | delta_vs_family_envelope | stage_ceiling | stage_ceiling_gap | engineering_noise_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| budget_comm | dala_lite_same_context_main | headline | overall | dala_lite | 0.708082 | mv_3 | 0.110977 | all_to_all_full | -0.027744 | all_to_all_full | -0.027744 | 0.722557 | 0.014475 | 0.011415 |
| cue | cue_black_box_utility_main | diagnostic | overall | cue_v1 | 0.572428 | mv_3 | 0.047585 | always_communicate | -0.055983 | always_communicate | -0.055983 | 0.577327 | 0.004899 | -0.005572 |
| free_mad_lite | free_mad_lite_mechanism_validation | supporting | overall | free_mad_lite_llm_trajectory | 0.772014 | mv_3_initial | 0.144752 | vanilla_mad_r1_final_vote | 0.007237 | free_mad_lite_llm_trajectory | 0.000000 | 0.804584 | 0.032570 | 0.002014 |
| multi_agent | same_context_controlled_debate | supporting | overall | mad_3a_r1 | 0.786531 | mv_6 | 0.098389 |  |  | mad_3a_r1 | 0.000000 | 0.798552 | 0.012021 |  |
| selective_comm | trigger_early_exit_main | headline | overall | hybrid_trigger | 0.835947 | mv_6 | -0.010856 | always_communicate | 0.001206 | mv_6 | -0.010856 | 0.841978 | 0.006031 | -0.010720 |
| selective_comm | voc_trigger_main | headline | overall | voc_trigger_v2 | 0.727382 | mv_3 | 0.088058 | always_communicate | -0.007238 | always_communicate | -0.007238 | 0.749095 | 0.021713 | 0.007382 |
| sid_lite | sid_lite_mechanism_validation | diagnostic | overall | sid_lite | 0.636912 | mv_3 | 0.020507 | always_full | -0.077201 | always_full | -0.077201 | 0.641737 | 0.004825 | 0.013579 |
| single_agent | cross_provider_robustness | diagnostic | overall | cot_1 | 0.831698 | cot_1 | 0.000000 |  |  | cot_1 | 0.000000 | 0.831122 | -0.000576 |  |
| single_agent | same_context_core_benchmarks | reference | overall | sc_5 | 0.819442 | cot_1 | -0.012256 |  |  | cot_1 | -0.012256 | 0.822678 | 0.003236 |  |
| single_agent | same_context_main_table | reference | overall | sc_5 | 0.647601 | sc_5 | 0.000000 |  |  | sc_5 | 0.000000 | 0.657559 | 0.009958 |  |
| sparc | content_ablation | diagnostic | overall | task_adaptive | 0.626055 | mv_3 | 0.013269 | full_cot | -0.112184 | full_cot | -0.112184 | 0.644150 | 0.018095 | -0.013945 |
| sparc | end_to_end_main | headline | overall | sparc_v1 | 0.664656 | mv_3 | 0.051870 | always_communicate | -0.004825 | always_communicate | -0.004825 | 0.681544 | 0.016888 | -0.012011 |
| sparc | local_auditing_ablation | supporting | overall | local_auditing | 0.669481 | majority_vote | 0.056695 | final_round_vote | 0.034982 | single_judge | -0.049457 | 0.683957 | 0.014476 | -0.010519 |

## split_context_overall

| family | experiment_name | evidence_tier | dataset | primary_method_name | faithful_score | best_no_comm_control | delta_vs_best_no_comm | full_comm_reference | delta_vs_full_comm | family_envelope | delta_vs_family_envelope | stage_ceiling | stage_ceiling_gap | engineering_noise_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| budget_comm | dala_lite_split_context_main | headline | overall | dala_lite | 0.648393 | mv_3 | 0.037807 | all_to_all_full | -0.064272 | all_to_all_full | -0.064272 | 0.697543 | 0.049150 | -0.026607 |
| comm_necessary | hotpotqa_split_context_communication_necessity | headline | overall | full_packet_exchange | 0.576667 | split_no_comm_mv3 | 0.150000 |  |  | full_context_single | -0.076666 | 0.576667 | 0.000000 | -0.013333 |

## same_context_rows

| family | experiment_name | evidence_tier | dataset | primary_method_name | faithful_score | best_no_comm_control | delta_vs_best_no_comm | full_comm_reference | delta_vs_full_comm | family_envelope | delta_vs_family_envelope | stage_ceiling | stage_ceiling_gap | engineering_noise_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| budget_comm | dala_lite_same_context_main | headline | gsm8k | dala_lite | 0.746667 | mv_3 | 0.256667 | all_to_all_full | -0.103333 | all_to_all_full | -0.103333 | 0.756667 | 0.010000 | -0.013333 |
| budget_comm | dala_lite_same_context_main | headline | hotpotqa | dala_lite | 0.656667 | mv_3 | 0.013334 | all_to_all_full | 0.010000 | all_to_all_full | 0.010000 | 0.670000 | 0.013333 | -0.003333 |
| budget_comm | dala_lite_same_context_main | headline | strategyqa | dala_lite | 0.724891 | mv_3 | 0.048035 | all_to_all_full | 0.021834 | all_to_all_full | 0.021834 | 0.746725 | 0.021834 | 0.054891 |
| cue | cue_black_box_utility_main | diagnostic | gsm8k | cue_v1 | 0.610000 | mv_3 | 0.133333 | always_communicate | -0.133333 | always_communicate | -0.133333 | 0.616667 | 0.006667 | 0.010000 |
| cue | cue_black_box_utility_main | diagnostic | hotpotqa | cue_v1 | 0.643333 | mv_3 | 0.000000 | always_communicate | -0.006667 | always_communicate | -0.006667 | 0.650000 | 0.006667 | -0.016667 |
| cue | cue_black_box_utility_main | diagnostic | math500 | cue_v1 | 0.340000 | mv_3 | 0.036667 | always_communicate | -0.080000 | always_communicate | -0.080000 | 0.346667 | 0.006667 | 0.010000 |
| cue | cue_black_box_utility_main | diagnostic | mmlu_pro | cue_v1 | 0.560000 | mv_3 | 0.040000 | always_communicate | -0.053333 | always_communicate | -0.053333 | 0.563333 | 0.003333 | -0.050000 |
| cue | cue_black_box_utility_main | diagnostic | strategyqa | cue_v1 | 0.751092 | mv_3 | 0.021834 | always_communicate | 0.008734 | always_communicate | 0.008734 | 0.751092 | 0.000000 | 0.061092 |
| free_mad_lite | free_mad_lite_mechanism_validation | supporting | gsm8k | free_mad_lite_llm_trajectory | 0.890000 | mv_3_initial | 0.373333 | vanilla_mad_r1_final_vote | 0.043333 | free_mad_lite_llm_trajectory | 0.000000 | 0.896667 | 0.006667 | -0.010000 |
| free_mad_lite | free_mad_lite_mechanism_validation | supporting | hotpotqa | free_mad_lite_llm_trajectory | 0.663333 | mv_3_initial | 0.003333 | vanilla_mad_r1_final_vote | -0.010000 | free_mad_lite_llm_trajectory | 0.000000 | 0.700000 | 0.036667 | -0.036667 |
| free_mad_lite | free_mad_lite_mechanism_validation | supporting | strategyqa | free_mad_lite_llm_trajectory | 0.759825 | mv_3_initial | 0.030567 | vanilla_mad_r1_final_vote | -0.017468 | free_mad_lite_llm_trajectory | 0.000000 | 0.820961 | 0.061136 | 0.049825 |
| multi_agent | same_context_controlled_debate | supporting | gsm8k | mad_3a_r1 | 0.896667 | mv_6 | 0.253333 |  |  | mad_3a_r1 | 0.000000 | 0.900000 | 0.003333 | 0.026667 |
| multi_agent | same_context_controlled_debate | supporting | hotpotqa | mad_3a_r1 | 0.690000 | mv_6 | 0.020000 |  |  | mad_3a_r1 | 0.000000 | 0.706667 | 0.016667 | -0.040000 |
| multi_agent | same_context_controlled_debate | supporting | strategyqa | mad_3a_r1 | 0.772926 | mv_6 | 0.021834 |  |  | mad_3a_r1 | 0.000000 | 0.786026 | 0.013100 | 0.062926 |
| selective_comm | trigger_early_exit_main | headline | gsm8k | hybrid_trigger | 0.966667 | mv_6 | -0.006666 | always_communicate | 0.000000 | mv_6 | -0.006666 | 0.966667 | 0.000000 | -0.003333 |
| selective_comm | trigger_early_exit_main | headline | hotpotqa | hybrid_trigger | 0.706667 | mv_6 | -0.013333 | always_communicate | 0.003334 | mv_6 | -0.013333 | 0.716667 | 0.010000 | -0.033333 |
| selective_comm | trigger_early_exit_main | headline | strategyqa | hybrid_trigger | 0.834061 | mv_6 | -0.013101 | always_communicate | 0.000000 | mv_6 | -0.013101 | 0.842795 | 0.008734 | 0.004061 |
| selective_comm | voc_trigger_main | headline | gsm8k | voc_trigger_v2 | 0.773333 | mv_3 | 0.250000 | always_communicate | -0.020000 | always_communicate | -0.020000 | 0.800000 | 0.026667 | 0.003333 |
| selective_comm | voc_trigger_main | headline | hotpotqa | voc_trigger_v2 | 0.656667 | mv_3 | 0.000000 | always_communicate | 0.003334 | always_communicate | 0.003334 | 0.673333 | 0.016666 | -0.023333 |
| selective_comm | voc_trigger_main | headline | strategyqa | voc_trigger_v2 | 0.759825 | mv_3 | -0.008734 | always_communicate | -0.004367 | always_communicate | -0.004367 | 0.781659 | 0.021834 | 0.049825 |
| sid_lite | sid_lite_mechanism_validation | diagnostic | gsm8k | sid_lite | 0.560000 | mv_3 | 0.043333 | always_full | -0.210000 | always_full | -0.210000 | 0.570000 | 0.010000 | 0.020000 |
| sid_lite | sid_lite_mechanism_validation | diagnostic | hotpotqa | sid_lite | 0.666667 | mv_3 | 0.013334 | always_full | -0.003333 | always_full | -0.003333 | 0.670000 | 0.003333 | -0.003333 |
| sid_lite | sid_lite_mechanism_validation | diagnostic | strategyqa | sid_lite | 0.698690 | mv_3 | 0.000000 | always_full | 0.000000 | always_full | 0.000000 | 0.698690 | 0.000000 | 0.038690 |
| single_agent | cross_provider_robustness | diagnostic | gsm8k | cot_1 | 0.973333 | cot_1 | 0.000000 |  |  | cot_1 | 0.000000 | 0.973333 | -0.000000 | 0.013333 |
| single_agent | cross_provider_robustness | diagnostic | hotpotqa | cot_1 | 0.683333 | cot_1 | 0.000000 |  |  | cot_1 | 0.000000 | 0.683333 | -0.000000 | -0.026667 |
| single_agent | cross_provider_robustness | diagnostic | strategyqa | cot_1 | 0.838428 | cot_1 | 0.000000 |  |  | cot_1 | 0.000000 | 0.838428 | 0.000000 | 0.008428 |
| single_agent | same_context_core_benchmarks | reference | gsm8k | sc_5 | 0.973333 | cot_1 | 0.000000 |  |  | cot_1 | 0.000000 | 0.973333 | -0.000000 | 0.013333 |
| single_agent | same_context_core_benchmarks | reference | hotpotqa | sc_5 | 0.703333 | cot_1 | 0.020000 |  |  | cot_1 | 0.020000 | 0.703333 | -0.000000 | -0.026667 |
| single_agent | same_context_core_benchmarks | reference | strategyqa | sc_5 | 0.781659 | cot_1 | -0.056769 |  |  | cot_1 | -0.056769 | 0.781659 | -0.000000 | 0.051659 |
| single_agent | same_context_main_table | reference | gpqa_diamond | sc_5 | 0.540404 | sc_5 | 0.000000 |  |  | sc_5 | 0.000000 | 0.540404 | -0.000000 | -0.009596 |
| single_agent | same_context_main_table | reference | hotpotqa | sc_5 | 0.703333 | sc_5 | 0.000000 |  |  | sc_5 | 0.000000 | 0.703333 | -0.000000 | -0.026667 |
| single_agent | same_context_main_table | reference | math500 | sc_5 | 0.640000 | sc_5 | 0.000000 |  |  | sc_5 | 0.000000 | 0.640000 | 0.000000 | -0.020000 |
| single_agent | same_context_main_table | reference | mmlu_pro | sc_5 | 0.706667 | sc_5 | 0.000000 |  |  | sc_5 | 0.000000 | 0.706667 | 0.000000 | -0.053333 |
| sparc | content_ablation | diagnostic | gsm8k | task_adaptive | 0.540000 | mv_3 | 0.043333 | full_cot | -0.286667 | full_cot | -0.286667 | 0.570000 | 0.030000 | -0.030000 |
| sparc | content_ablation | diagnostic | hotpotqa | task_adaptive | 0.643333 | mv_3 | -0.006667 | full_cot | -0.023334 | full_cot | -0.023334 | 0.653333 | 0.010000 | -0.016667 |
| sparc | content_ablation | diagnostic | strategyqa | task_adaptive | 0.716157 | mv_3 | 0.000000 | full_cot | 0.000000 | full_cot | 0.000000 | 0.729258 | 0.013101 | 0.026157 |
| sparc | end_to_end_main | headline | gsm8k | sparc_v1 | 0.626667 | mv_3 | 0.130000 | always_communicate | -0.006666 | always_communicate | -0.006666 | 0.640000 | 0.013333 | -0.023333 |
| sparc | end_to_end_main | headline | hotpotqa | sparc_v1 | 0.666667 | mv_3 | 0.016667 | always_communicate | -0.003333 | always_communicate | -0.003333 | 0.673333 | 0.006666 | -0.023333 |
| sparc | end_to_end_main | headline | strategyqa | sparc_v1 | 0.711790 | mv_3 | -0.004367 | always_communicate | -0.004367 | always_communicate | -0.004367 | 0.746725 | 0.034935 | 0.021790 |
| sparc | local_auditing_ablation | supporting | gsm8k | local_auditing | 0.633333 | majority_vote | 0.136666 | final_round_vote | 0.090000 | single_judge | -0.126667 | 0.643333 | 0.010000 | -0.016667 |
| sparc | local_auditing_ablation | supporting | hotpotqa | local_auditing | 0.670000 | majority_vote | 0.020000 | final_round_vote | 0.013333 | single_judge | 0.010000 | 0.676667 | 0.006667 | -0.030000 |
| sparc | local_auditing_ablation | supporting | strategyqa | local_auditing | 0.716157 | majority_vote | 0.000000 | final_round_vote | -0.008734 | single_judge | -0.026201 | 0.746725 | 0.030568 | 0.026157 |

## split_context_rows

| family | experiment_name | evidence_tier | dataset | primary_method_name | faithful_score | best_no_comm_control | delta_vs_best_no_comm | full_comm_reference | delta_vs_full_comm | family_envelope | delta_vs_family_envelope | stage_ceiling | stage_ceiling_gap | engineering_noise_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| budget_comm | dala_lite_split_context_main | headline | hotpotqa | dala_lite | 0.466667 | mv_3 | 0.023334 | all_to_all_full | -0.126666 | all_to_all_full | -0.126666 | 0.533333 | 0.066666 | -0.013333 |
| budget_comm | dala_lite_split_context_main | headline | strategyqa | dala_lite | 0.886463 | mv_3 | 0.056769 | all_to_all_full | 0.017467 | all_to_all_full | 0.017467 | 0.912664 | 0.026201 | 0.016463 |
| comm_necessary | hotpotqa_split_context_communication_necessity | headline | hotpotqa | full_packet_exchange | 0.576667 | split_no_comm_mv3 | 0.150000 |  |  | full_context_single | -0.076666 | 0.576667 | 0.000000 | -0.013333 |
