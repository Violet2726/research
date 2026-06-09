# Canonical Baseline Recheck

## 摘要

- Canonical simple baseline 固定为 `cot_1@temp=0.7 / mv_3@temp=0.7 / sc_5@temp=0.7`。
- `old official cot_1` 来自 ceiling summary 的旧 official baseline；`run cot_1` 是被复核 run 自带的 control，二者可能不同。
- `holds_vs_canonical_best` 表示方法超过同数据集上 `cot_1/mv_3/sc_5` 三者中最强的 canonical baseline。
- `only_beats_old_official_cot1` 表示结论只相对旧 `cot_1` 成立，不能再包装成优于 strong simple baseline。
- `borderline_*` 表示差值小于 `0.01`，在 count100 口径下应视作边际信号。
- 本表只用于同上下文/full-context 口径；split-context 结论应继续用 split no-comm baseline 单独复核。

## 输入 runs

- `local/runs/adaptive_sparse_mad/same_context_competition_math_stage_a_v4/count100/20260608T024322Z-xiaomimimo-mimo-v2.5`
- `local/runs/adaptive_sparse_mad/same_context_full_counterfactual_v1/count100/20260608T065426Z-xiaomimimo-mimo-v2.5`
- `local/runs/adaptive_sparse_mad/same_context_hard_transfer_stage_a_v4/count100/20260608T024322Z-xiaomimimo-mimo-v2.5`
- `local/runs/adaptive_sparse_mad/same_context_hotpot_stage_a_v4_ablate/count100/20260608T024231Z-xiaomimimo-mimo-v2.5`
- `local/runs/adaptive_sparse_mad/same_context_hotpot_stage_a_v5/count100/20260608T031630Z-xiaomimimo-mimo-v2.5`
- `local/runs/budget_comm/dala_lite_same_context_main/count100/20260531T151736Z-xiaomimimo-mimo-v2.5`
- `local/runs/consensagent/consensagent_main/count100/20260605T064424Z-xiaomimimo-mimo-v2.5`
- `local/runs/dmad/dmad_reasoning_main/count100/20260601T025609Z-xiaomimimo-mimo-v2.5`
- `local/runs/econ/econ_same_context_main/count100/20260531T135412Z-xiaomimimo-mimo-v2.5`
- `local/runs/free_mad_lite/free_mad_lite_mechanism_validation/count100/20260531T135950Z-xiaomimimo-mimo-v2.5`
- `local/runs/imad/imad_same_context_main/count100/20260531T133346Z-xiaomimimo-mimo-v2.5`
- `local/runs/madjudge/madjudge_main/count100/20260602T145728Z-xiaomimimo-mimo-v2.5`
- `local/runs/multi_agent/same_context_controlled_debate/count100/20260531T133340Z-xiaomimimo-mimo-v2.5`
- `local/runs/selective_comm/trigger_early_exit_main/count100/20260604T125623Z-xiaomimimo-mimo-v2.5`
- `local/runs/sid_lite/sid_lite_mechanism_validation/count100/20260605T025856Z-xiaomimimo-mimo-v2.5`

## Canonical Baseline 表

| Dataset | Old official cot_1 | Canonical cot_1 | Canonical mv_3 | Canonical sc_5 | Canonical Best |
| --- | --- | --- | --- | --- | --- |
| competition_math | 0.6700 | 0.6767 | 0.6900 | 0.7400 | `sc_5` 0.7400 |
| gpqa_diamond | 0.4600 | 0.4633 | 0.4800 | 0.5000 | `sc_5` 0.5000 |
| gsm8k | 0.9700 | 0.9600 | 0.9600 | 0.9700 | `sc_5` 0.9700 |
| hotpotqa | 0.6900 | 0.7100 | 0.7300 | 0.7367 | `sc_5` 0.7367 |
| math500 | 0.6200 | 0.6767 | 0.7167 | 0.7567 | `sc_5` 0.7567 |
| mmlu_pro | 0.6900 | 0.7133 | 0.7300 | 0.7400 | `sc_5` 0.7400 |

## Overlap Aggregate 结论

| Experiment | Method | Datasets | Acc | Old cot_1 | New cot_1 | Canonical Best | Delta Best | Judgement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `dala_lite_same_context_main` | `all_to_all_full` | 2 | 0.7800 | 0.8300 | 0.8350 | 0.8533 | -0.0733 | `does_not_beat_old_official_cot1` |
| `dala_lite_same_context_main` | `budget_confidence` | 2 | 0.7050 | 0.8300 | 0.8350 | 0.8533 | -0.1483 | `does_not_beat_old_official_cot1` |
| `dala_lite_same_context_main` | `budget_random` | 2 | 0.7000 | 0.8300 | 0.8350 | 0.8533 | -0.1533 | `does_not_beat_old_official_cot1` |
| `dala_lite_same_context_main` | `dala_lite` | 2 | 0.7200 | 0.8300 | 0.8350 | 0.8533 | -0.1333 | `does_not_beat_old_official_cot1` |
| `dmad_reasoning_main` | `dmad_cot_sbp_pot` | 2 | 0.8075 | 0.6438 | 0.6500 | 0.7100 | +0.0975 | `holds_vs_canonical_best` |
| `dmad_reasoning_main` | `dmad_mrp` | 2 | 0.4238 | 0.6438 | 0.6500 | 0.7100 | -0.2862 | `does_not_beat_old_official_cot1` |
| `dmad_reasoning_main` | `dmad_self_contrast` | 2 | 0.5763 | 0.6438 | 0.6500 | 0.7100 | -0.1337 | `does_not_beat_old_official_cot1` |
| `dmad_reasoning_main` | `dmad_self_refine` | 2 | 0.6025 | 0.6438 | 0.6500 | 0.7100 | -0.1075 | `does_not_beat_old_official_cot1` |
| `dmad_reasoning_main` | `dmad_single_cot` | 2 | 0.4275 | 0.6438 | 0.6500 | 0.7100 | -0.2825 | `does_not_beat_old_official_cot1` |
| `dmad_reasoning_main` | `dmad_single_cot_sc` | 2 | 0.4238 | 0.6438 | 0.6500 | 0.7100 | -0.2862 | `does_not_beat_old_official_cot1` |
| `dmad_reasoning_main` | `dmad_single_pot_l2m` | 2 | 0.5013 | 0.6438 | 0.6500 | 0.7100 | -0.2087 | `does_not_beat_old_official_cot1` |
| `dmad_reasoning_main` | `dmad_single_pot_l2m_sc` | 2 | 0.4975 | 0.6438 | 0.6500 | 0.7100 | -0.2125 | `does_not_beat_old_official_cot1` |
| `dmad_reasoning_main` | `dmad_single_sbp` | 2 | 0.4225 | 0.6438 | 0.6500 | 0.7100 | -0.2875 | `does_not_beat_old_official_cot1` |
| `dmad_reasoning_main` | `dmad_single_sbp_sc` | 2 | 0.4100 | 0.6438 | 0.6500 | 0.7100 | -0.3000 | `does_not_beat_old_official_cot1` |
| `dmad_reasoning_main` | `mad_all_cot` | 2 | 0.7425 | 0.6438 | 0.6500 | 0.7100 | +0.0325 | `holds_vs_canonical_best` |
| `dmad_reasoning_main` | `mad_all_pot` | 2 | 0.7913 | 0.6438 | 0.6500 | 0.7100 | +0.0813 | `holds_vs_canonical_best` |
| `dmad_reasoning_main` | `mad_all_sbp` | 2 | 0.7775 | 0.6438 | 0.6500 | 0.7100 | +0.0675 | `holds_vs_canonical_best` |
| `dmad_reasoning_main` | `mad_persona_d` | 2 | 0.7675 | 0.6438 | 0.6500 | 0.7100 | +0.0575 | `holds_vs_canonical_best` |
| `dmad_reasoning_main` | `mad_persona_e` | 2 | 0.7600 | 0.6438 | 0.6500 | 0.7100 | +0.0500 | `holds_vs_canonical_best` |
| `econ_same_context_main` | `econ_bne_main` | 2 | 0.6800 | 0.8300 | 0.8350 | 0.8533 | -0.1733 | `does_not_beat_old_official_cot1` |
| `econ_same_context_main` | `econ_full_comm_r1` | 2 | 0.7500 | 0.8300 | 0.8350 | 0.8533 | -0.1033 | `does_not_beat_old_official_cot1` |
| `econ_same_context_main` | `single_agent_cot` | 2 | 0.5950 | 0.8300 | 0.8350 | 0.8533 | -0.2583 | `does_not_beat_old_official_cot1` |
| `econ_same_context_main` | `vote_mv3` | 2 | 0.6300 | 0.8300 | 0.8350 | 0.8533 | -0.2233 | `does_not_beat_old_official_cot1` |
| `free_mad_lite_mechanism_validation` | `anti_conformity_final_vote` | 2 | 0.7600 | 0.8300 | 0.8350 | 0.8533 | -0.0933 | `does_not_beat_old_official_cot1` |
| `free_mad_lite_mechanism_validation` | `free_mad_lite_llm_trajectory` | 2 | 0.8150 | 0.8300 | 0.8350 | 0.8533 | -0.0383 | `does_not_beat_old_official_cot1` |
| `imad_same_context_main` | `imad_adaptive` | 2 | 0.8200 | 0.8300 | 0.8350 | 0.8533 | -0.0333 | `does_not_beat_old_official_cot1` |
| `same_context_competition_math_stage_a_v4` | `adaptive_gate_v4` | 1 | 0.7800 | 0.6700 | 0.6767 | 0.7400 | +0.0400 | `holds_vs_canonical_best` |
| `same_context_competition_math_stage_a_v4` | `hetero_vote_3` | 1 | 0.7800 | 0.6700 | 0.6767 | 0.7400 | +0.0400 | `holds_vs_canonical_best` |
| `same_context_full_counterfactual_v1` | `adaptive_counterfactual_v1` | 6 | 0.7583 | 0.6833 | 0.7000 | 0.7406 | +0.0178 | `holds_vs_canonical_best` |
| `same_context_full_counterfactual_v1` | `adaptive_gate_v4` | 6 | 0.7517 | 0.6833 | 0.7000 | 0.7406 | +0.0111 | `holds_vs_canonical_best` |
| `same_context_full_counterfactual_v1` | `hetero_vote_3` | 6 | 0.7433 | 0.6833 | 0.7000 | 0.7406 | +0.0028 | `borderline_above_canonical_best` |
| `same_context_hard_transfer_stage_a_v4` | `adaptive_gate_v4` | 3 | 0.7500 | 0.5900 | 0.6178 | 0.6656 | +0.0844 | `holds_vs_canonical_best` |
| `same_context_hard_transfer_stage_a_v4` | `hetero_vote_3` | 3 | 0.7467 | 0.5900 | 0.6178 | 0.6656 | +0.0811 | `holds_vs_canonical_best` |
| `same_context_hotpot_stage_a_v4_ablate` | `adaptive_gate_v4` | 1 | 0.7600 | 0.6900 | 0.7100 | 0.7367 | +0.0233 | `holds_vs_canonical_best` |
| `same_context_hotpot_stage_a_v4_ablate` | `always_add_v4` | 1 | 0.7600 | 0.6900 | 0.7100 | 0.7367 | +0.0233 | `holds_vs_canonical_best` |
| `same_context_hotpot_stage_a_v4_ablate` | `dge_ega_v4` | 1 | 0.7600 | 0.6900 | 0.7100 | 0.7367 | +0.0233 | `holds_vs_canonical_best` |
| `same_context_hotpot_stage_a_v4_ablate` | `dge_only_v4` | 1 | 0.7200 | 0.6900 | 0.7100 | 0.7367 | -0.0167 | `beats_canonical_cot_not_best` |
| `same_context_hotpot_stage_a_v4_ablate` | `ega_only_v4` | 1 | 0.7600 | 0.6900 | 0.7100 | 0.7367 | +0.0233 | `holds_vs_canonical_best` |
| `same_context_hotpot_stage_a_v4_ablate` | `hetero_vote_3` | 1 | 0.7400 | 0.6900 | 0.7100 | 0.7367 | +0.0033 | `borderline_above_canonical_best` |
| `same_context_hotpot_stage_a_v5` | `adaptive_dual_open_v5` | 1 | 0.7700 | 0.6900 | 0.7100 | 0.7367 | +0.0333 | `holds_vs_canonical_best` |
| `same_context_hotpot_stage_a_v5` | `hetero_vote_3` | 1 | 0.7400 | 0.6900 | 0.7100 | 0.7367 | +0.0033 | `borderline_above_canonical_best` |
| `sid_lite_mechanism_validation` | `always_full` | 2 | 0.7050 | 0.8300 | 0.8350 | 0.8533 | -0.1483 | `does_not_beat_old_official_cot1` |
| `sid_lite_mechanism_validation` | `compression_only` | 2 | 0.6000 | 0.8300 | 0.8350 | 0.8533 | -0.2533 | `does_not_beat_old_official_cot1` |
| `sid_lite_mechanism_validation` | `sid_lite` | 2 | 0.6000 | 0.8300 | 0.8350 | 0.8533 | -0.2533 | `does_not_beat_old_official_cot1` |
| `trigger_early_exit_main` | `always_communicate` | 2 | 0.8350 | 0.8300 | 0.8350 | 0.8533 | -0.0183 | `only_beats_old_official_cot1` |
| `trigger_early_exit_main` | `confidence_triggered` | 2 | 0.8450 | 0.8300 | 0.8350 | 0.8533 | -0.0083 | `beats_canonical_cot_not_best` |
| `trigger_early_exit_main` | `disagreement_triggered` | 2 | 0.8350 | 0.8300 | 0.8350 | 0.8533 | -0.0183 | `only_beats_old_official_cot1` |
| `trigger_early_exit_main` | `hybrid_trigger` | 2 | 0.8350 | 0.8300 | 0.8350 | 0.8533 | -0.0183 | `only_beats_old_official_cot1` |

## Focus Dataset 结论: math500, mmlu_pro, hotpotqa

| Dataset | Experiment | Method | Acc | Run cot_1 | Old cot_1 | New cot_1 | Canonical Best | Delta Best | Judgement |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `hotpotqa` | `consensagent_main` | `consensagent_3a` | 0.7300 | 0.7300 | 0.6900 | 0.7100 | `sc_5` 0.7367 | -0.0067 | `beats_canonical_cot_not_best` |
| `hotpotqa` | `dala_lite_same_context_main` | `all_to_all_full` | 0.6900 |  | 0.6900 | 0.7100 | `sc_5` 0.7367 | -0.0467 | `does_not_beat_old_official_cot1` |
| `hotpotqa` | `dala_lite_same_context_main` | `budget_confidence` | 0.6600 |  | 0.6900 | 0.7100 | `sc_5` 0.7367 | -0.0767 | `does_not_beat_old_official_cot1` |
| `hotpotqa` | `dala_lite_same_context_main` | `budget_random` | 0.6700 |  | 0.6900 | 0.7100 | `sc_5` 0.7367 | -0.0667 | `does_not_beat_old_official_cot1` |
| `hotpotqa` | `dala_lite_same_context_main` | `dala_lite` | 0.6900 |  | 0.6900 | 0.7100 | `sc_5` 0.7367 | -0.0467 | `does_not_beat_old_official_cot1` |
| `hotpotqa` | `econ_same_context_main` | `econ_bne_main` | 0.6800 |  | 0.6900 | 0.7100 | `sc_5` 0.7367 | -0.0567 | `does_not_beat_old_official_cot1` |
| `hotpotqa` | `econ_same_context_main` | `econ_full_comm_r1` | 0.6900 |  | 0.6900 | 0.7100 | `sc_5` 0.7367 | -0.0467 | `does_not_beat_old_official_cot1` |
| `hotpotqa` | `econ_same_context_main` | `single_agent_cot` | 0.6500 |  | 0.6900 | 0.7100 | `sc_5` 0.7367 | -0.0867 | `does_not_beat_old_official_cot1` |
| `hotpotqa` | `econ_same_context_main` | `vote_mv3` | 0.6700 |  | 0.6900 | 0.7100 | `sc_5` 0.7367 | -0.0667 | `does_not_beat_old_official_cot1` |
| `hotpotqa` | `free_mad_lite_mechanism_validation` | `anti_conformity_final_vote` | 0.6800 |  | 0.6900 | 0.7100 | `sc_5` 0.7367 | -0.0567 | `does_not_beat_old_official_cot1` |
| `hotpotqa` | `free_mad_lite_mechanism_validation` | `free_mad_lite_llm_trajectory` | 0.7200 |  | 0.6900 | 0.7100 | `sc_5` 0.7367 | -0.0167 | `beats_canonical_cot_not_best` |
| `hotpotqa` | `imad_same_context_main` | `imad_adaptive` | 0.7000 |  | 0.6900 | 0.7100 | `sc_5` 0.7367 | -0.0367 | `only_beats_old_official_cot1` |
| `hotpotqa` | `madjudge_main` | `madjudge_7a` | 0.7000 | 0.6900 | 0.6900 | 0.7100 | `sc_5` 0.7367 | -0.0367 | `only_beats_old_official_cot1` |
| `hotpotqa` | `same_context_full_counterfactual_v1` | `adaptive_counterfactual_v1` | 0.7400 | 0.6900 | 0.6900 | 0.7100 | `sc_5` 0.7367 | +0.0033 | `borderline_above_canonical_best` |
| `hotpotqa` | `same_context_full_counterfactual_v1` | `adaptive_gate_v4` | 0.7300 | 0.6900 | 0.6900 | 0.7100 | `sc_5` 0.7367 | -0.0067 | `beats_canonical_cot_not_best` |
| `hotpotqa` | `same_context_full_counterfactual_v1` | `hetero_vote_3` | 0.6800 | 0.6900 | 0.6900 | 0.7100 | `sc_5` 0.7367 | -0.0567 | `does_not_beat_old_official_cot1` |
| `hotpotqa` | `same_context_hotpot_stage_a_v4_ablate` | `adaptive_gate_v4` | 0.7600 | 0.6900 | 0.6900 | 0.7100 | `sc_5` 0.7367 | +0.0233 | `holds_vs_canonical_best` |
| `hotpotqa` | `same_context_hotpot_stage_a_v4_ablate` | `always_add_v4` | 0.7600 | 0.6900 | 0.6900 | 0.7100 | `sc_5` 0.7367 | +0.0233 | `holds_vs_canonical_best` |
| `hotpotqa` | `same_context_hotpot_stage_a_v4_ablate` | `dge_ega_v4` | 0.7600 | 0.6900 | 0.6900 | 0.7100 | `sc_5` 0.7367 | +0.0233 | `holds_vs_canonical_best` |
| `hotpotqa` | `same_context_hotpot_stage_a_v4_ablate` | `dge_only_v4` | 0.7200 | 0.6900 | 0.6900 | 0.7100 | `sc_5` 0.7367 | -0.0167 | `beats_canonical_cot_not_best` |
| `hotpotqa` | `same_context_hotpot_stage_a_v4_ablate` | `ega_only_v4` | 0.7600 | 0.6900 | 0.6900 | 0.7100 | `sc_5` 0.7367 | +0.0233 | `holds_vs_canonical_best` |
| `hotpotqa` | `same_context_hotpot_stage_a_v4_ablate` | `hetero_vote_3` | 0.7400 | 0.6900 | 0.6900 | 0.7100 | `sc_5` 0.7367 | +0.0033 | `borderline_above_canonical_best` |
| `hotpotqa` | `same_context_hotpot_stage_a_v5` | `adaptive_dual_open_v5` | 0.7700 | 0.6900 | 0.6900 | 0.7100 | `sc_5` 0.7367 | +0.0333 | `holds_vs_canonical_best` |
| `hotpotqa` | `same_context_hotpot_stage_a_v5` | `hetero_vote_3` | 0.7400 | 0.6900 | 0.6900 | 0.7100 | `sc_5` 0.7367 | +0.0033 | `borderline_above_canonical_best` |
| `hotpotqa` | `sid_lite_mechanism_validation` | `always_full` | 0.6600 |  | 0.6900 | 0.7100 | `sc_5` 0.7367 | -0.0767 | `does_not_beat_old_official_cot1` |
| `hotpotqa` | `sid_lite_mechanism_validation` | `compression_only` | 0.6600 |  | 0.6900 | 0.7100 | `sc_5` 0.7367 | -0.0767 | `does_not_beat_old_official_cot1` |
| `hotpotqa` | `sid_lite_mechanism_validation` | `sid_lite` | 0.6600 |  | 0.6900 | 0.7100 | `sc_5` 0.7367 | -0.0767 | `does_not_beat_old_official_cot1` |
| `hotpotqa` | `trigger_early_exit_main` | `always_communicate` | 0.7000 |  | 0.6900 | 0.7100 | `sc_5` 0.7367 | -0.0367 | `only_beats_old_official_cot1` |
| `hotpotqa` | `trigger_early_exit_main` | `confidence_triggered` | 0.7200 |  | 0.6900 | 0.7100 | `sc_5` 0.7367 | -0.0167 | `beats_canonical_cot_not_best` |
| `hotpotqa` | `trigger_early_exit_main` | `disagreement_triggered` | 0.7000 |  | 0.6900 | 0.7100 | `sc_5` 0.7367 | -0.0367 | `only_beats_old_official_cot1` |
| `hotpotqa` | `trigger_early_exit_main` | `hybrid_trigger` | 0.7000 |  | 0.6900 | 0.7100 | `sc_5` 0.7367 | -0.0367 | `only_beats_old_official_cot1` |
| `math500` | `madjudge_main` | `madjudge_7a` | 0.8200 | 0.5400 | 0.6200 | 0.6767 | `sc_5` 0.7567 | +0.0633 | `holds_vs_canonical_best` |
| `math500` | `same_context_full_counterfactual_v1` | `adaptive_counterfactual_v1` | 0.7400 | 0.7000 | 0.6200 | 0.6767 | `sc_5` 0.7567 | -0.0167 | `beats_canonical_cot_not_best` |
| `math500` | `same_context_full_counterfactual_v1` | `adaptive_gate_v4` | 0.7300 | 0.7000 | 0.6200 | 0.6767 | `sc_5` 0.7567 | -0.0267 | `beats_canonical_cot_not_best` |
| `math500` | `same_context_full_counterfactual_v1` | `hetero_vote_3` | 0.7300 | 0.7000 | 0.6200 | 0.6767 | `sc_5` 0.7567 | -0.0267 | `beats_canonical_cot_not_best` |
| `math500` | `same_context_hard_transfer_stage_a_v4` | `adaptive_gate_v4` | 0.7900 | 0.7000 | 0.6200 | 0.6767 | `sc_5` 0.7567 | +0.0333 | `holds_vs_canonical_best` |
| `math500` | `same_context_hard_transfer_stage_a_v4` | `hetero_vote_3` | 0.7900 | 0.7000 | 0.6200 | 0.6767 | `sc_5` 0.7567 | +0.0333 | `holds_vs_canonical_best` |
| `mmlu_pro` | `madjudge_main` | `madjudge_7a` | 0.8000 | 0.6900 | 0.6900 | 0.7133 | `sc_5` 0.7400 | +0.0600 | `holds_vs_canonical_best` |
| `mmlu_pro` | `same_context_full_counterfactual_v1` | `adaptive_counterfactual_v1` | 0.7500 | 0.7400 | 0.6900 | 0.7133 | `sc_5` 0.7400 | +0.0100 | `holds_vs_canonical_best` |
| `mmlu_pro` | `same_context_full_counterfactual_v1` | `adaptive_gate_v4` | 0.7300 | 0.7400 | 0.6900 | 0.7133 | `sc_5` 0.7400 | -0.0100 | `beats_canonical_cot_not_best` |
| `mmlu_pro` | `same_context_full_counterfactual_v1` | `hetero_vote_3` | 0.7300 | 0.7400 | 0.6900 | 0.7133 | `sc_5` 0.7400 | -0.0100 | `beats_canonical_cot_not_best` |
| `mmlu_pro` | `same_context_hard_transfer_stage_a_v4` | `adaptive_gate_v4` | 0.8100 | 0.7400 | 0.6900 | 0.7133 | `sc_5` 0.7400 | +0.0700 | `holds_vs_canonical_best` |
| `mmlu_pro` | `same_context_hard_transfer_stage_a_v4` | `hetero_vote_3` | 0.8100 | 0.7400 | 0.6900 | 0.7133 | `sc_5` 0.7400 | +0.0700 | `holds_vs_canonical_best` |

## Judgement Counts

- `beats_canonical_cot_not_best`: 2
- `borderline_above_canonical_best`: 3
- `does_not_beat_old_official_cot1`: 23
- `holds_vs_canonical_best`: 17
- `only_beats_old_official_cot1`: 3
