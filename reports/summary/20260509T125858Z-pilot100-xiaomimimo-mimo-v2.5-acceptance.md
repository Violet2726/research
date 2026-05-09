# Faithful Acceptance Summary

- generated_at: `2026-05-09T12:59:41.743645+00:00`
- counts: `{"evaluated": 15, "accepted_same_context": 12, "accepted_split_context": 2, "negative_control_family": 1}`

## accepted_same_context

| family | experiment_name | evaluation_track | primary_method_name | faithful_score | delta_vs_best_no_comm | delta_vs_full_comm | delta_vs_full_context | token_ratio_vs_full_comm | stage_ceiling_gap | engineering_noise_gap | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| budget_comm | dala_lite_same_context_main | same_context | dala_lite | 0.696667 | 0.093334 | -0.016666 |  | 0.920306 | 0.016666 | -0.053333 | accepted_same_context | passed_same_context_gate |
| cue | cue_black_box_utility_main | same_context | cue_v1 | 0.578000 | 0.030000 | -0.058000 |  | 0.522537 | 0.010000 | -0.042000 | accepted_same_context | passed_same_context_gate |
| free_mad_lite | free_mad_lite_mechanism_validation | same_context | free_mad_lite_llm_trajectory | 0.770000 | 0.140000 | 0.006667 |  | 1.288927 | 0.026667 | -0.046667 | accepted_same_context | passed_same_context_gate |
| multi_agent | same_context_controlled_debate | same_context | mad_3a_r1 | 0.770000 | 0.090000 |  |  |  | 0.006667 |  | accepted_same_context | passed_same_context_gate |
| selective_comm | trigger_early_exit_main | same_context | hybrid_trigger | 0.846667 | -0.010000 | 0.000000 |  | 0.574502 | 0.006666 | -0.020000 | accepted_same_context | passed_same_context_gate |
| selective_comm | voc_trigger_main | same_context | voc_trigger_v2 | 0.720000 | 0.073333 | -0.006667 |  | 0.823999 | 0.016667 | -0.046667 | accepted_same_context | passed_same_context_gate |
| sid_lite | sid_lite_mechanism_validation | same_context | sid_lite | 0.623333 | 0.010000 | -0.080000 |  | 0.588660 | 0.006667 | -0.076667 | accepted_same_context | passed_same_context_gate |
| single_agent | cross_provider_robustness | same_context | cot_1 | 0.833333 | 0.000000 |  |  |  | 0.000000 |  | accepted_same_context | passed_same_context_gate |
| single_agent | same_context_main_table | same_context | sc_5 | 0.675000 | 0.000000 |  |  |  | 0.000000 |  | accepted_same_context | passed_same_context_gate |
| sparc | content_ablation | same_context | task_adaptive | 0.640000 | 0.000000 | -0.090000 |  | 0.897595 | 0.020000 | -0.043333 | accepted_same_context | passed_same_context_gate |
| sparc | end_to_end_main | same_context | sparc_v1 | 0.676667 | 0.036667 | -0.003333 |  | 0.672796 | 0.020000 | -0.040000 | accepted_same_context | passed_same_context_gate |
| sparc | local_auditing_ablation | same_context | local_auditing | 0.680000 | 0.040000 | 0.016667 |  | 1.046709 | 0.020000 | -0.053333 | accepted_same_context | passed_same_context_gate |

## accepted_split_context

| family | experiment_name | evaluation_track | primary_method_name | faithful_score | delta_vs_best_no_comm | delta_vs_full_comm | delta_vs_full_context | token_ratio_vs_full_comm | stage_ceiling_gap | engineering_noise_gap | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| budget_comm | dala_lite_split_context_main | split_context | dala_lite | 0.675000 | 0.070000 | -0.065000 |  | 0.896915 | 0.045000 | 0.025000 | accepted_split_context | passed_split_context_gate |
| comm_necessary | hotpotqa_split_context_communication_necessity | split_context | full_packet_exchange | 0.590000 | 0.200000 |  | -0.100000 |  | 0.000000 | 0.090000 | accepted_split_context | passed_split_context_gate |

## negative_control_family

| family | experiment_name | evaluation_track | primary_method_name | faithful_score | delta_vs_best_no_comm | delta_vs_full_comm | delta_vs_full_context | token_ratio_vs_full_comm | stage_ceiling_gap | engineering_noise_gap | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| single_agent | same_context_core_benchmarks | same_context | sc_5 | 0.806667 | -0.026666 |  |  |  | 0.000000 |  | negative_control_family | best_no_comm_non_inferiority_failed |
