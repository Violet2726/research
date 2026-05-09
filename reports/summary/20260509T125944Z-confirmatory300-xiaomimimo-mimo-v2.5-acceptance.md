# Faithful Acceptance Summary

- generated_at: `2026-05-09T13:01:18.417536+00:00`
- counts: `{"evaluated": 15, "accepted_same_context": 13, "accepted_split_context": 2, "negative_control_family": 0}`

## accepted_same_context

| family | experiment_name | evaluation_track | primary_method_name | faithful_score | delta_vs_best_no_comm | delta_vs_full_comm | delta_vs_full_context | token_ratio_vs_full_comm | stage_ceiling_gap | engineering_noise_gap | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| budget_comm | dala_lite_same_context_main | same_context | dala_lite | 0.708082 | 0.110977 | -0.027744 |  | 0.922340 | 0.014475 | 0.011415 | accepted_same_context | passed_same_context_gate |
| cue | cue_black_box_utility_main | same_context | cue_v1 | 0.572428 | 0.047585 | -0.055983 |  | 0.534714 | 0.004899 | -0.005572 | accepted_same_context | passed_same_context_gate |
| free_mad_lite | free_mad_lite_mechanism_validation | same_context | free_mad_lite_llm_trajectory | 0.772014 | 0.144752 | 0.007237 |  | 1.279580 | 0.032570 | 0.002014 | accepted_same_context | passed_same_context_gate |
| multi_agent | same_context_controlled_debate | same_context | mad_3a_r1 | 0.786531 | 0.098389 |  |  |  | 0.012021 |  | accepted_same_context | passed_same_context_gate |
| selective_comm | trigger_early_exit_main | same_context | hybrid_trigger | 0.835947 | -0.010856 | 0.001206 |  | 0.547168 | 0.006031 | -0.010720 | accepted_same_context | passed_same_context_gate |
| selective_comm | voc_trigger_main | same_context | voc_trigger_v2 | 0.727382 | 0.088058 | -0.007238 |  | 0.820843 | 0.021713 | 0.007382 | accepted_same_context | passed_same_context_gate |
| sid_lite | sid_lite_mechanism_validation | same_context | sid_lite | 0.636912 | 0.020507 | -0.077201 |  | 0.583724 | 0.004825 | 0.013579 | accepted_same_context | passed_same_context_gate |
| single_agent | cross_provider_robustness | same_context | cot_1 | 0.831698 | 0.000000 |  |  |  | -0.000576 |  | accepted_same_context | passed_same_context_gate |
| single_agent | same_context_core_benchmarks | same_context | sc_5 | 0.819442 | -0.012256 |  |  |  | 0.003236 |  | accepted_same_context | passed_same_context_gate |
| single_agent | same_context_main_table | same_context | sc_5 | 0.647601 | 0.000000 |  |  |  | 0.009958 |  | accepted_same_context | passed_same_context_gate |
| sparc | content_ablation | same_context | task_adaptive | 0.626055 | 0.013269 | -0.112184 |  | 0.902542 | 0.018095 | -0.013945 | accepted_same_context | passed_same_context_gate |
| sparc | end_to_end_main | same_context | sparc_v1 | 0.664656 | 0.051870 | -0.004825 |  | 0.648410 | 0.016888 | -0.012011 | accepted_same_context | passed_same_context_gate |
| sparc | local_auditing_ablation | same_context | local_auditing | 0.669481 | 0.056695 | 0.034982 |  | 1.044240 | 0.014476 | -0.010519 | accepted_same_context | passed_same_context_gate |

## accepted_split_context

| family | experiment_name | evaluation_track | primary_method_name | faithful_score | delta_vs_best_no_comm | delta_vs_full_comm | delta_vs_full_context | token_ratio_vs_full_comm | stage_ceiling_gap | engineering_noise_gap | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| budget_comm | dala_lite_split_context_main | split_context | dala_lite | 0.648393 | 0.037807 | -0.064272 |  | 0.897040 | 0.049150 | -0.026607 | accepted_split_context | passed_split_context_gate |
| comm_necessary | hotpotqa_split_context_communication_necessity | split_context | full_packet_exchange | 0.576667 | 0.150000 |  | -0.076666 |  | 0.000000 | -0.013333 | accepted_split_context | passed_split_context_gate |

## negative_control_family

No rows.
