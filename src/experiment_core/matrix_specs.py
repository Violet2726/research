"""为 smoke20 矩阵与 faithful 分析声明实验对照规格。"""

from __future__ import annotations

from dataclasses import dataclass


TRACK_SAME_CONTEXT = "same_context"
TRACK_SPLIT_CONTEXT = "split_context"


@dataclass(frozen=True)
class ExperimentMatrixSpec:
    """单个实验入口在矩阵评审中的比较基准与轨道定义。"""

    evaluation_track: str
    primary_method_name: str
    best_no_comm_candidates: tuple[str, ...]
    full_comm_reference: str | None = None
    full_context_reference: str | None = None
    token_gate_basis: str = "none"


EXPERIMENT_MATRIX_SPECS: dict[str, ExperimentMatrixSpec] = {
    "configs/single_agent/experiments/main_baselines.toml": ExperimentMatrixSpec(
        evaluation_track=TRACK_SAME_CONTEXT,
        primary_method_name="sc_5",
        best_no_comm_candidates=("cot_1", "sc_5"),
    ),
    "configs/single_agent/experiments/main_table_same_context.toml": ExperimentMatrixSpec(
        evaluation_track=TRACK_SAME_CONTEXT,
        primary_method_name="sc_5",
        best_no_comm_candidates=("cot_1", "sc_5"),
    ),
    "configs/single_agent/experiments/robustness.toml": ExperimentMatrixSpec(
        evaluation_track=TRACK_SAME_CONTEXT,
        primary_method_name="cot_1",
        best_no_comm_candidates=("cot_1",),
    ),
    "configs/multi_agent/experiments/debate_vs_vote_controlled.toml": ExperimentMatrixSpec(
        evaluation_track=TRACK_SAME_CONTEXT,
        primary_method_name="mad_3a_r1",
        best_no_comm_candidates=("mv_6", "sc_6"),
    ),
    "configs/multi_agent/experiments/vanilla_mad_clean_smoke.toml": ExperimentMatrixSpec(
        evaluation_track=TRACK_SAME_CONTEXT,
        primary_method_name="mad_2a_r1",
        best_no_comm_candidates=("mv_4", "sc_4"),
    ),
    "configs/free_mad_lite/experiments/free_mad_lite_v1.toml": ExperimentMatrixSpec(
        evaluation_track=TRACK_SAME_CONTEXT,
        primary_method_name="free_mad_lite_llm_trajectory",
        best_no_comm_candidates=("mv_3_initial",),
        full_comm_reference="vanilla_mad_r1_final_vote",
        token_gate_basis="none",
    ),
    "configs/budget_comm/experiments/dala_lite_same_context_v1.toml": ExperimentMatrixSpec(
        evaluation_track=TRACK_SAME_CONTEXT,
        primary_method_name="dala_lite",
        best_no_comm_candidates=("mv_3",),
        full_comm_reference="all_to_all_full",
        token_gate_basis="communication",
    ),
    "configs/budget_comm/experiments/dala_lite_split_context_v1.toml": ExperimentMatrixSpec(
        evaluation_track=TRACK_SPLIT_CONTEXT,
        primary_method_name="dala_lite",
        best_no_comm_candidates=("mv_3",),
        full_comm_reference="all_to_all_full",
        token_gate_basis="communication",
    ),
    "configs/comm_necessary/experiments/hotpotqa_split_evidence_v1.toml": ExperimentMatrixSpec(
        evaluation_track=TRACK_SPLIT_CONTEXT,
        primary_method_name="full_packet_exchange",
        best_no_comm_candidates=("split_no_comm_mv3",),
        full_context_reference="full_context_single",
    ),
    "configs/comm_necessary/experiments/hotpotqa_split500_main.toml": ExperimentMatrixSpec(
        evaluation_track=TRACK_SPLIT_CONTEXT,
        primary_method_name="full_packet_exchange",
        best_no_comm_candidates=("split_no_comm_mv3",),
        full_context_reference="full_context_single",
    ),
    "configs/sid_lite/experiments/sid_lite_v1.toml": ExperimentMatrixSpec(
        evaluation_track=TRACK_SAME_CONTEXT,
        primary_method_name="sid_lite",
        best_no_comm_candidates=("mv_3",),
        full_comm_reference="always_full",
        token_gate_basis="communication",
    ),
    "configs/selective_comm/experiments/trigger_early_exit_v1.toml": ExperimentMatrixSpec(
        evaluation_track=TRACK_SAME_CONTEXT,
        primary_method_name="hybrid_trigger",
        best_no_comm_candidates=("mv_3", "mv_6", "sc_6"),
        full_comm_reference="always_communicate",
        token_gate_basis="communication",
    ),
    "configs/selective_comm/experiments/trigger_voc_v2.toml": ExperimentMatrixSpec(
        evaluation_track=TRACK_SAME_CONTEXT,
        primary_method_name="voc_trigger_v2",
        best_no_comm_candidates=("mv_3", "mv_6", "sc_6"),
        full_comm_reference="always_communicate",
        token_gate_basis="communication",
    ),
    "configs/selective_comm/experiments/trigger_voc_v2_core_only.toml": ExperimentMatrixSpec(
        evaluation_track=TRACK_SAME_CONTEXT,
        primary_method_name="voc_trigger_v2",
        best_no_comm_candidates=("mv_3",),
        full_comm_reference="always_communicate",
        token_gate_basis="communication",
    ),
    "configs/selective_comm/experiments/trigger_voc_v2_equal_budget_gsm_strategy.toml": ExperimentMatrixSpec(
        evaluation_track=TRACK_SAME_CONTEXT,
        primary_method_name="voc_trigger_v2",
        best_no_comm_candidates=("mv_3", "mv_6", "sc_6"),
        full_comm_reference="always_communicate",
        token_gate_basis="communication",
    ),
    "configs/sparc/experiments/sparc_v1_smoke.toml": ExperimentMatrixSpec(
        evaluation_track=TRACK_SAME_CONTEXT,
        primary_method_name="sparc_v1",
        best_no_comm_candidates=("mv_3",),
        full_comm_reference="always_communicate",
        token_gate_basis="communication",
    ),
    "configs/sparc/experiments/content_ablation_v1.toml": ExperimentMatrixSpec(
        evaluation_track=TRACK_SAME_CONTEXT,
        primary_method_name="task_adaptive",
        best_no_comm_candidates=("mv_3",),
        full_comm_reference="full_cot",
        token_gate_basis="communication",
    ),
    "configs/sparc/experiments/auditing_ablation_v1.toml": ExperimentMatrixSpec(
        evaluation_track=TRACK_SAME_CONTEXT,
        primary_method_name="local_auditing",
        best_no_comm_candidates=("majority_vote",),
        full_comm_reference="final_round_vote",
        token_gate_basis="none",
    ),
    "configs/sparc/experiments/aggregation_auditing_ablation_v1.toml": ExperimentMatrixSpec(
        evaluation_track=TRACK_SAME_CONTEXT,
        primary_method_name="local_auditing",
        best_no_comm_candidates=("majority_vote",),
        full_comm_reference="final_round_vote",
        token_gate_basis="none",
    ),
    "configs/cue/experiments/cue_v1.toml": ExperimentMatrixSpec(
        evaluation_track=TRACK_SAME_CONTEXT,
        primary_method_name="cue_v1",
        best_no_comm_candidates=("mv_3",),
        full_comm_reference="always_communicate",
        token_gate_basis="communication",
    ),
    "configs/multi_agent/experiments/vanilla_mad_minimal.toml": ExperimentMatrixSpec(
        evaluation_track=TRACK_SAME_CONTEXT,
        primary_method_name="mad_2a_r1",
        best_no_comm_candidates=("mv_4", "sc_4"),
    ),
    "configs/single_agent/experiments/local_ollama_smoke.toml": ExperimentMatrixSpec(
        evaluation_track=TRACK_SAME_CONTEXT,
        primary_method_name="cot_1",
        best_no_comm_candidates=("cot_1",),
    ),
}


def get_experiment_matrix_spec(config_path: str) -> ExperimentMatrixSpec:
    """按实验配置路径返回既定的矩阵规格。"""
    try:
        return EXPERIMENT_MATRIX_SPECS[config_path]
    except KeyError as exc:
        raise KeyError(f"Missing matrix spec for config {config_path}") from exc
