# Faithful Acceptance Summary

- generated_at: `2026-05-07T12:17:42.396378+00:00`
- counts: `{"evaluated": 16, "accepted_same_context": 14, "accepted_split_context": 2, "negative_control_family": 0}`

## accepted_same_context

| family | experiment_name | evaluation_track | primary_method_name | faithful_score | delta_vs_best_no_comm | delta_vs_full_comm | delta_vs_full_context | token_ratio_vs_full_comm | stage_ceiling_gap | engineering_noise_gap | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| budget_comm | dala_lite_same_context_v1 | same_context | dala_lite | 0.710000 | 0.113333 | -0.013333 |  | 0.918893 | 0.010000 | 0.026667 | accepted_same_context | passed_same_context_gate |
| cue | cue_v1 | same_context | cue_v1 | 0.584000 | 0.030000 | -0.050000 |  | 0.521509 | 0.018000 | -0.076000 | accepted_same_context | passed_same_context_gate |
| free_mad_lite | free_mad_lite_v1 | same_context | free_mad_lite_llm_trajectory | 0.786667 | 0.160000 | 0.033334 |  | 1.287528 | 0.016666 | 0.003334 | accepted_same_context | passed_same_context_gate |
| multi_agent | multi_agent_main | same_context | mad_3a_r1 | 0.800000 | 0.130000 |  |  |  | 0.000000 |  | accepted_same_context | passed_same_context_gate |
| selective_comm | trigger_early_exit_v1 | same_context | hybrid_trigger | 0.840000 | -0.016667 | 0.000000 |  | 0.546980 | 0.016667 | -0.026667 | accepted_same_context | passed_same_context_gate |
| selective_comm | trigger_voc_v2 | same_context | voc_trigger_v2 | 0.730000 | 0.080000 | -0.010000 |  | 0.822325 | 0.010000 | 0.013333 | accepted_same_context | passed_same_context_gate |
| sid_lite | sid_lite_v1 | same_context | sid_lite | 0.620000 | 0.026667 | -0.093333 |  | 0.580106 | 0.003333 | -0.063333 | accepted_same_context | passed_same_context_gate |
| single_agent | main_baselines | same_context | sc_5 | 0.830000 | 0.000000 |  |  |  | 0.000000 |  | accepted_same_context | passed_same_context_gate |
| single_agent | main_table_same_context | same_context | sc_5 | 0.672500 | 0.000000 |  |  |  | 0.000000 |  | accepted_same_context | passed_same_context_gate |
| single_agent | robustness | same_context | cot_1 | 0.820000 | 0.000000 |  |  |  | 0.000000 |  | accepted_same_context | passed_same_context_gate |
| sparc | aggregation_auditing_ablation_v1 | same_context | local_auditing | 0.666667 | 0.040000 | 0.040000 |  | 1.045324 | 0.020000 | -0.050000 | accepted_same_context | passed_same_context_gate |
| sparc | auditing_ablation_v1 | same_context | local_auditing | 0.666667 | 0.040000 | 0.040000 |  | 1.045324 | 0.020000 | -0.050000 | accepted_same_context | passed_same_context_gate |
| sparc | content_ablation_v1 | same_context | task_adaptive | 0.623333 | -0.003334 | -0.096667 |  | 0.899042 | 0.016667 | -0.043334 | accepted_same_context | passed_same_context_gate |
| sparc | sparc_v1_smoke | same_context | sparc_v1 | 0.666667 | 0.040000 | 0.000000 |  | 0.657197 | 0.020000 | -0.033333 | accepted_same_context | passed_same_context_gate |

## accepted_split_context

| family | experiment_name | evaluation_track | primary_method_name | faithful_score | delta_vs_best_no_comm | delta_vs_full_comm | delta_vs_full_context | token_ratio_vs_full_comm | stage_ceiling_gap | engineering_noise_gap | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| budget_comm | dala_lite_split_context_v1 | split_context | dala_lite | 0.695000 | 0.085000 | -0.055000 |  | 0.896393 | 0.030000 | -0.055000 | accepted_split_context | passed_split_context_gate |
| comm_necessary | hotpotqa_split_main | split_context | full_packet_exchange | 0.610000 | 0.210000 |  | -0.070000 |  | 0.000000 |  | accepted_split_context | passed_split_context_gate |

## negative_control_family

No rows.
