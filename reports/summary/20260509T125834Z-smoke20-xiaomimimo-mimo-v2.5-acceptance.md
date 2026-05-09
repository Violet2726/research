# Faithful Acceptance Summary

- generated_at: `2026-05-09T12:58:57.099828+00:00`
- counts: `{"evaluated": 15, "accepted_same_context": 12, "accepted_split_context": 1, "negative_control_family": 2}`

## accepted_same_context

| family | experiment_name | evaluation_track | primary_method_name | faithful_score | delta_vs_best_no_comm | delta_vs_full_comm | delta_vs_full_context | token_ratio_vs_full_comm | stage_ceiling_gap | engineering_noise_gap | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| budget_comm | dala_lite_same_context_main | same_context | dala_lite | 0.750000 | 0.083333 | 0.000000 |  | 0.918221 | 0.033333 |  | accepted_same_context | passed_same_context_gate |
| cue | cue_black_box_utility_main | same_context | cue_v1 | 0.620000 | 0.050000 | -0.080000 |  | 0.511998 | 0.010000 |  | accepted_same_context | passed_same_context_gate |
| free_mad_lite | free_mad_lite_mechanism_validation | same_context | free_mad_lite_llm_trajectory | 0.816667 | 0.150000 | 0.016667 |  | 1.290219 | 0.033333 |  | accepted_same_context | passed_same_context_gate |
| multi_agent | same_context_controlled_debate | same_context | mad_3a_r1 | 0.816667 | 0.100000 |  |  |  | 0.000000 |  | accepted_same_context | passed_same_context_gate |
| selective_comm | trigger_early_exit_main | same_context | hybrid_trigger | 0.866667 | 0.000000 | 0.000000 |  | 0.603075 | 0.000000 |  | accepted_same_context | passed_same_context_gate |
| selective_comm | voc_trigger_main | same_context | voc_trigger_v2 | 0.766667 | 0.050000 | 0.000000 |  | 0.818370 | 0.016666 |  | accepted_same_context | passed_same_context_gate |
| sid_lite | sid_lite_mechanism_validation | same_context | sid_lite | 0.700000 | 0.000000 | -0.083333 |  | 0.627644 | 0.000000 |  | accepted_same_context | passed_same_context_gate |
| single_agent | cross_provider_robustness | same_context | cot_1 | 0.850000 | 0.000000 |  |  |  | 0.000000 |  | accepted_same_context | passed_same_context_gate |
| single_agent | same_context_main_table | same_context | sc_5 | 0.662500 | 0.000000 |  |  |  | 0.000000 |  | accepted_same_context | passed_same_context_gate |
| sparc | content_ablation | same_context | task_adaptive | 0.683333 | 0.016666 | -0.050000 |  | 0.895563 | 0.033334 |  | accepted_same_context | passed_same_context_gate |
| sparc | end_to_end_main | same_context | sparc_v1 | 0.716667 | 0.050000 | -0.016666 |  | 0.688190 | 0.016666 |  | accepted_same_context | passed_same_context_gate |
| sparc | local_auditing_ablation | same_context | local_auditing | 0.733333 | 0.066666 | 0.033333 |  | 1.062695 | 0.016667 |  | accepted_same_context | passed_same_context_gate |

## accepted_split_context

| family | experiment_name | evaluation_track | primary_method_name | faithful_score | delta_vs_best_no_comm | delta_vs_full_comm | delta_vs_full_context | token_ratio_vs_full_comm | stage_ceiling_gap | engineering_noise_gap | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| budget_comm | dala_lite_split_context_main | split_context | dala_lite | 0.650000 | 0.100000 | -0.075000 |  | 0.894359 | 0.025000 |  | accepted_split_context | passed_split_context_gate |

## negative_control_family

| family | experiment_name | evaluation_track | primary_method_name | faithful_score | delta_vs_best_no_comm | delta_vs_full_comm | delta_vs_full_context | token_ratio_vs_full_comm | stage_ceiling_gap | engineering_noise_gap | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single_agent | same_context_core_benchmarks | same_context | sc_5 | 0.800000 | -0.050000 |  |  |  | 0.000000 |  | negative_control_family | best_no_comm_non_inferiority_failed |
| comm_necessary | hotpotqa_split_context_communication_necessity | split_context | full_packet_exchange | 0.500000 | 0.100000 |  | -0.200000 |  | 0.000000 |  | negative_control_family | full_context_gap_recovery_failed |
