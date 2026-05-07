# Faithful Analysis

- generated_at: `2026-05-07T12:17:42.375304+00:00`
- experiment_count: `16`

## same_context_overall

| family | experiment_name | dataset | primary_method_name | faithful_score | best_no_comm_control | delta_vs_best_no_comm | full_comm_reference | delta_vs_full_comm | family_envelope | delta_vs_family_envelope | stage_ceiling | stage_ceiling_gap | engineering_noise_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| budget_comm | dala_lite_same_context_v1 | overall | dala_lite | 0.710000 | mv_3 | 0.113333 | all_to_all_full | -0.013333 | all_to_all_full | -0.013333 | 0.720000 | 0.010000 | 0.026667 |
| cue | cue_v1 | overall | cue_v1 | 0.584000 | mv_3 | 0.030000 | always_communicate | -0.050000 | consensus_freeze | -0.050000 | 0.602000 | 0.018000 | -0.076000 |
| free_mad_lite | free_mad_lite_v1 | overall | free_mad_lite_llm_trajectory | 0.786667 | mv_3_initial | 0.160000 | vanilla_mad_r1_final_vote | 0.033334 | free_mad_lite_llm_trajectory | 0.000000 | 0.803333 | 0.016666 | 0.003334 |
| multi_agent | multi_agent_main | overall | mad_3a_r1 | 0.800000 | mv_6 | 0.130000 |  |  | mad_3a_r1 | 0.000000 | 0.800000 | 0.000000 |  |
| selective_comm | trigger_early_exit_v1 | overall | hybrid_trigger | 0.840000 | sc_6 | -0.016667 | always_communicate | 0.000000 | sc_6 | -0.016667 | 0.856667 | 0.016667 | -0.026667 |
| selective_comm | trigger_voc_v2 | overall | voc_trigger_v2 | 0.730000 | mv_6 | 0.080000 | always_communicate | -0.010000 | always_communicate | -0.010000 | 0.740000 | 0.010000 | 0.013333 |
| sid_lite | sid_lite_v1 | overall | sid_lite | 0.620000 | mv_3 | 0.026667 | always_full | -0.093333 | always_full | -0.093333 | 0.623333 | 0.003333 | -0.063333 |
| single_agent | main_baselines | overall | sc_5 | 0.830000 | sc_5 | 0.000000 |  |  | sc_5 | 0.000000 | 0.830000 | 0.000000 |  |
| single_agent | main_table_same_context | overall | sc_5 | 0.672500 | sc_5 | 0.000000 |  |  | sc_5 | 0.000000 | 0.672500 | 0.000000 |  |
| single_agent | robustness | overall | cot_1 | 0.820000 | cot_1 | 0.000000 |  |  | sc_5 | -0.010000 | 0.820000 | 0.000000 |  |
| sparc | aggregation_auditing_ablation_v1 | overall | local_auditing | 0.666667 | majority_vote | 0.040000 | final_round_vote | 0.040000 | single_judge | -0.036666 | 0.686667 | 0.020000 | -0.050000 |
| sparc | auditing_ablation_v1 | overall | local_auditing | 0.666667 | majority_vote | 0.040000 | final_round_vote | 0.040000 | single_judge | -0.036666 | 0.686667 | 0.020000 | -0.050000 |
| sparc | content_ablation_v1 | overall | task_adaptive | 0.623333 | mv_3 | -0.003334 | full_cot | -0.096667 | full_cot | -0.096667 | 0.640000 | 0.016667 | -0.043334 |
| sparc | sparc_v1_smoke | overall | sparc_v1 | 0.666667 | mv_3 | 0.040000 | always_communicate | 0.000000 | sparc_v1 | 0.000000 | 0.686667 | 0.020000 | -0.033333 |

## split_context_overall

| family | experiment_name | dataset | primary_method_name | faithful_score | best_no_comm_control | delta_vs_best_no_comm | full_comm_reference | delta_vs_full_comm | family_envelope | delta_vs_family_envelope | stage_ceiling | stage_ceiling_gap | engineering_noise_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| budget_comm | dala_lite_split_context_v1 | overall | dala_lite | 0.695000 | mv_3 | 0.085000 | all_to_all_full | -0.055000 | all_to_all_full | -0.055000 | 0.725000 | 0.030000 | -0.055000 |
| comm_necessary | hotpotqa_split_main | overall | full_packet_exchange | 0.610000 | split_no_comm_mv3 | 0.210000 |  |  | full_context_single | -0.070000 | 0.610000 | 0.000000 |  |

## same_context_rows

| family | experiment_name | dataset | primary_method_name | faithful_score | best_no_comm_control | delta_vs_best_no_comm | full_comm_reference | delta_vs_full_comm | family_envelope | delta_vs_family_envelope | stage_ceiling | stage_ceiling_gap | engineering_noise_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| budget_comm | dala_lite_same_context_v1 | gsm8k | dala_lite | 0.770000 | mv_3 | 0.280000 | all_to_all_full | -0.040000 | all_to_all_full | -0.040000 | 0.780000 | 0.010000 | 0.120000 |
| budget_comm | dala_lite_same_context_v1 | hotpotqa | dala_lite | 0.670000 | mv_3 | 0.030000 | all_to_all_full | 0.000000 | all_to_all_full | 0.000000 | 0.670000 | 0.000000 | -0.030000 |
| budget_comm | dala_lite_same_context_v1 | strategyqa | dala_lite | 0.690000 | mv_3 | 0.030000 | all_to_all_full | 0.000000 | all_to_all_full | 0.000000 | 0.710000 | 0.020000 | -0.010000 |
| cue | cue_v1 | gsm8k | cue_v1 | 0.580000 | mv_3 | 0.110000 | always_communicate | -0.140000 | consensus_freeze | -0.140000 | 0.590000 | 0.010000 | -0.020000 |
| cue | cue_v1 | hotpotqa | cue_v1 | 0.670000 | mv_3 | 0.010000 | always_communicate | -0.020000 | consensus_freeze | -0.020000 | 0.670000 | 0.000000 | -0.180000 |
| cue | cue_v1 | math500 | cue_v1 | 0.390000 | mv_3 | 0.030000 | always_communicate | -0.070000 | consensus_freeze | -0.070000 | 0.410000 | 0.020000 | -0.110000 |
| cue | cue_v1 | mmlu_pro | cue_v1 | 0.600000 | mv_3 | -0.010000 | always_communicate | -0.020000 | consensus_freeze | -0.020000 | 0.640000 | 0.040000 | -0.050000 |
| cue | cue_v1 | strategyqa | cue_v1 | 0.680000 | mv_3 | 0.010000 | always_communicate | 0.000000 | consensus_freeze | 0.000000 | 0.700000 | 0.020000 | -0.020000 |
| free_mad_lite | free_mad_lite_v1 | gsm8k | free_mad_lite_llm_trajectory | 0.910000 | mv_3_initial | 0.380000 | vanilla_mad_r1_final_vote | 0.090000 | free_mad_lite_llm_trajectory | 0.000000 | 0.910000 | 0.000000 | -0.090000 |
| free_mad_lite | free_mad_lite_v1 | hotpotqa | free_mad_lite_llm_trajectory | 0.710000 | mv_3_initial | 0.070000 | vanilla_mad_r1_final_vote | 0.020000 | free_mad_lite_llm_trajectory | 0.000000 | 0.710000 | 0.000000 | 0.060000 |
| free_mad_lite | free_mad_lite_v1 | strategyqa | free_mad_lite_llm_trajectory | 0.740000 | mv_3_initial | 0.030000 | vanilla_mad_r1_final_vote | -0.010000 | free_mad_lite_llm_trajectory | 0.000000 | 0.790000 | 0.050000 | 0.040000 |
| multi_agent | multi_agent_main | gsm8k | mad_3a_r1 | 0.910000 | mv_6 | 0.260000 |  |  | mad_3a_r1 | 0.000000 | 0.910000 | 0.000000 |  |
| multi_agent | multi_agent_main | hotpotqa | mad_3a_r1 | 0.760000 | mv_6 | 0.090000 |  |  | mad_3a_r1 | 0.000000 | 0.760000 | 0.000000 |  |
| multi_agent | multi_agent_main | strategyqa | mad_3a_r1 | 0.730000 | mv_6 | 0.040000 |  |  | mad_3a_r1 | 0.000000 | 0.730000 | 0.000000 |  |
| selective_comm | trigger_early_exit_v1 | gsm8k | hybrid_trigger | 0.960000 | sc_6 | 0.000000 | always_communicate | 0.000000 | sc_6 | 0.000000 | 0.970000 | 0.010000 | -0.040000 |
| selective_comm | trigger_early_exit_v1 | hotpotqa | hybrid_trigger | 0.750000 | sc_6 | -0.020000 | always_communicate | 0.000000 | sc_6 | -0.020000 | 0.770000 | 0.020000 | 0.000000 |
| selective_comm | trigger_early_exit_v1 | strategyqa | hybrid_trigger | 0.810000 | sc_6 | -0.030000 | always_communicate | 0.000000 | sc_6 | -0.030000 | 0.830000 | 0.020000 | -0.040000 |
| selective_comm | trigger_voc_v2 | gsm8k | voc_trigger_v2 | 0.780000 | mv_6 | 0.270000 | always_communicate | -0.030000 | always_communicate | -0.030000 | 0.810000 | 0.030000 | 0.030000 |
| selective_comm | trigger_voc_v2 | hotpotqa | voc_trigger_v2 | 0.680000 | mv_6 | -0.040000 | always_communicate | 0.000000 | always_communicate | 0.000000 | 0.680000 | 0.000000 | -0.020000 |
| selective_comm | trigger_voc_v2 | strategyqa | voc_trigger_v2 | 0.730000 | mv_6 | 0.010000 | always_communicate | 0.000000 | always_communicate | 0.000000 | 0.730000 | 0.000000 | 0.030000 |
| sid_lite | sid_lite_v1 | gsm8k | sid_lite | 0.510000 | mv_3 | 0.050000 | always_full | -0.270000 | always_full | -0.270000 | 0.510000 | 0.000000 | 0.010000 |
| sid_lite | sid_lite_v1 | hotpotqa | sid_lite | 0.680000 | mv_3 | 0.010000 | always_full | 0.000000 | always_full | 0.000000 | 0.680000 | 0.000000 | -0.120000 |
| sid_lite | sid_lite_v1 | strategyqa | sid_lite | 0.670000 | mv_3 | 0.020000 | always_full | -0.010000 | always_full | -0.010000 | 0.680000 | 0.010000 | -0.080000 |
| single_agent | main_baselines | gsm8k | sc_5 | 0.970000 | sc_5 | 0.000000 |  |  | sc_5 | 0.000000 | 0.970000 | 0.000000 | -0.030000 |
| single_agent | main_baselines | hotpotqa | sc_5 | 0.700000 | sc_5 | 0.000000 |  |  | sc_5 | 0.000000 | 0.700000 | 0.000000 | -0.150000 |
| single_agent | main_baselines | strategyqa | sc_5 | 0.820000 | sc_5 | 0.000000 |  |  | sc_5 | 0.000000 | 0.820000 | 0.000000 | -0.030000 |
| single_agent | main_table_same_context | gpqa_diamond | sc_5 | 0.540000 | sc_5 | 0.000000 |  |  | sc_5 | 0.000000 | 0.540000 | 0.000000 | 0.090000 |
| single_agent | main_table_same_context | hotpotqa | sc_5 | 0.700000 | sc_5 | 0.000000 |  |  | sc_5 | 0.000000 | 0.700000 | 0.000000 | -0.150000 |
| single_agent | main_table_same_context | math500 | sc_5 | 0.660000 | sc_5 | 0.000000 |  |  | sc_5 | 0.000000 | 0.660000 | 0.000000 | -0.140000 |
| single_agent | main_table_same_context | mmlu_pro | sc_5 | 0.790000 | sc_5 | 0.000000 |  |  | sc_5 | 0.000000 | 0.790000 | 0.000000 | 0.090000 |
| single_agent | robustness | gsm8k | cot_1 | 0.960000 | cot_1 | 0.000000 |  |  | sc_5 | -0.010000 | 0.960000 | 0.000000 | -0.040000 |
| single_agent | robustness | hotpotqa | cot_1 | 0.690000 | cot_1 | 0.000000 |  |  | sc_5 | -0.010000 | 0.690000 | 0.000000 | -0.010000 |
| single_agent | robustness | strategyqa | cot_1 | 0.810000 | cot_1 | 0.000000 |  |  | sc_5 | -0.010000 | 0.810000 | 0.000000 | 0.110000 |
| sparc | aggregation_auditing_ablation_v1 | gsm8k | local_auditing | 0.590000 | majority_vote | 0.030000 | final_round_vote | 0.060000 | single_judge | -0.150000 | 0.630000 | 0.040000 | -0.060000 |
| sparc | aggregation_auditing_ablation_v1 | hotpotqa | local_auditing | 0.720000 | majority_vote | 0.020000 | final_round_vote | 0.010000 | single_judge | 0.000000 | 0.730000 | 0.010000 | -0.030000 |
| sparc | aggregation_auditing_ablation_v1 | strategyqa | local_auditing | 0.690000 | majority_vote | 0.070000 | final_round_vote | 0.050000 | single_judge | 0.040000 | 0.700000 | 0.010000 | -0.060000 |
| sparc | auditing_ablation_v1 | gsm8k | local_auditing | 0.590000 | majority_vote | 0.030000 | final_round_vote | 0.060000 | single_judge | -0.150000 | 0.630000 | 0.040000 | -0.060000 |
| sparc | auditing_ablation_v1 | hotpotqa | local_auditing | 0.720000 | majority_vote | 0.020000 | final_round_vote | 0.010000 | single_judge | 0.000000 | 0.730000 | 0.010000 | -0.030000 |
| sparc | auditing_ablation_v1 | strategyqa | local_auditing | 0.690000 | majority_vote | 0.070000 | final_round_vote | 0.050000 | single_judge | 0.040000 | 0.700000 | 0.010000 | -0.060000 |
| sparc | content_ablation_v1 | gsm8k | task_adaptive | 0.560000 | mv_3 | 0.000000 | full_cot | -0.240000 | full_cot | -0.240000 | 0.590000 | 0.030000 | -0.040000 |
| sparc | content_ablation_v1 | hotpotqa | task_adaptive | 0.700000 | mv_3 | 0.000000 | full_cot | 0.000000 | full_cot | 0.000000 | 0.710000 | 0.010000 | 0.000000 |
| sparc | content_ablation_v1 | strategyqa | task_adaptive | 0.610000 | mv_3 | -0.010000 | full_cot | -0.050000 | full_cot | -0.050000 | 0.620000 | 0.010000 | -0.090000 |
| sparc | sparc_v1_smoke | gsm8k | sparc_v1 | 0.590000 | mv_3 | 0.030000 | always_communicate | 0.000000 | sparc_v1 | 0.000000 | 0.630000 | 0.040000 | -0.060000 |
| sparc | sparc_v1_smoke | hotpotqa | sparc_v1 | 0.720000 | mv_3 | 0.020000 | always_communicate | 0.000000 | sparc_v1 | 0.000000 | 0.730000 | 0.010000 | 0.020000 |
| sparc | sparc_v1_smoke | strategyqa | sparc_v1 | 0.690000 | mv_3 | 0.070000 | always_communicate | 0.000000 | sparc_v1 | 0.000000 | 0.700000 | 0.010000 | -0.060000 |

## split_context_rows

| family | experiment_name | dataset | primary_method_name | faithful_score | best_no_comm_control | delta_vs_best_no_comm | full_comm_reference | delta_vs_full_comm | family_envelope | delta_vs_family_envelope | stage_ceiling | stage_ceiling_gap | engineering_noise_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| budget_comm | dala_lite_split_context_v1 | hotpotqa | dala_lite | 0.520000 | mv_3 | 0.110000 | all_to_all_full | -0.140000 | all_to_all_full | -0.140000 | 0.570000 | 0.050000 | -0.080000 |
| budget_comm | dala_lite_split_context_v1 | strategyqa | dala_lite | 0.870000 | mv_3 | 0.060000 | all_to_all_full | 0.030000 | all_to_all_full | 0.030000 | 0.880000 | 0.010000 | -0.030000 |
| comm_necessary | hotpotqa_split_main | hotpotqa | full_packet_exchange | 0.610000 | split_no_comm_mv3 | 0.210000 |  |  | full_context_single | -0.070000 | 0.610000 | 0.000000 |  |
