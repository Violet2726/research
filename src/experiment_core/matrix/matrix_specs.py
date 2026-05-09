"""为 smoke20 矩阵与 faithful 分析声明实验对照规格。"""

from __future__ import annotations

from dataclasses import dataclass


TRACK_SAME_CONTEXT = "same_context"
TRACK_SPLIT_CONTEXT = "split_context"
EVIDENCE_HEADLINE = "headline"
EVIDENCE_SUPPORTING = "supporting"
EVIDENCE_DIAGNOSTIC = "diagnostic"
EVIDENCE_REFERENCE = "reference"


@dataclass(frozen=True)
class ExperimentMatrixSpec:
    """单个实验入口在矩阵评审中的比较基准与轨道定义。"""

    evaluation_track: str
    evidence_tier: str
    primary_method_name: str
    best_no_comm_candidates: tuple[str, ...]
    full_comm_reference: str | None = None
    full_context_reference: str | None = None
    token_gate_basis: str = "none"


EXPERIMENT_MATRIX_SPECS: dict[str, ExperimentMatrixSpec] = {
    "configs/single_agent/experiments/same_context_core_benchmarks.toml": ExperimentMatrixSpec(
        evaluation_track=TRACK_SAME_CONTEXT,
        evidence_tier=EVIDENCE_REFERENCE,
        primary_method_name="sc_5",
        best_no_comm_candidates=("cot_1", "sc_5"),
    ),
    "configs/single_agent/experiments/same_context_main_table.toml": ExperimentMatrixSpec(
        evaluation_track=TRACK_SAME_CONTEXT,
        evidence_tier=EVIDENCE_REFERENCE,
        primary_method_name="sc_5",
        best_no_comm_candidates=("cot_1", "sc_5"),
    ),
    "configs/single_agent/experiments/cross_provider_robustness.toml": ExperimentMatrixSpec(
        evaluation_track=TRACK_SAME_CONTEXT,
        evidence_tier=EVIDENCE_DIAGNOSTIC,
        primary_method_name="cot_1",
        best_no_comm_candidates=("cot_1",),
    ),
    "configs/multi_agent/experiments/same_context_controlled_debate.toml": ExperimentMatrixSpec(
        evaluation_track=TRACK_SAME_CONTEXT,
        evidence_tier=EVIDENCE_SUPPORTING,
        primary_method_name="mad_3a_r1",
        best_no_comm_candidates=("mv_6",),
    ),
    "configs/free_mad_lite/experiments/free_mad_lite_mechanism_validation.toml": ExperimentMatrixSpec(
        evaluation_track=TRACK_SAME_CONTEXT,
        evidence_tier=EVIDENCE_SUPPORTING,
        primary_method_name="free_mad_lite_llm_trajectory",
        best_no_comm_candidates=("mv_3_initial",),
        full_comm_reference="vanilla_mad_r1_final_vote",
        token_gate_basis="none",
    ),
    "configs/budget_comm/experiments/dala_lite_same_context_main.toml": ExperimentMatrixSpec(
        evaluation_track=TRACK_SAME_CONTEXT,
        evidence_tier=EVIDENCE_HEADLINE,
        primary_method_name="dala_lite",
        best_no_comm_candidates=("mv_3",),
        full_comm_reference="all_to_all_full",
        token_gate_basis="communication",
    ),
    "configs/budget_comm/experiments/dala_lite_split_context_main.toml": ExperimentMatrixSpec(
        evaluation_track=TRACK_SPLIT_CONTEXT,
        evidence_tier=EVIDENCE_HEADLINE,
        primary_method_name="dala_lite",
        best_no_comm_candidates=("mv_3",),
        full_comm_reference="all_to_all_full",
        token_gate_basis="communication",
    ),
    "configs/comm_necessary/experiments/hotpotqa_split_context_communication_necessity.toml": ExperimentMatrixSpec(
        evaluation_track=TRACK_SPLIT_CONTEXT,
        evidence_tier=EVIDENCE_HEADLINE,
        primary_method_name="full_packet_exchange",
        best_no_comm_candidates=("split_no_comm_mv3",),
        full_context_reference="full_context_single",
    ),
    "configs/sid_lite/experiments/sid_lite_mechanism_validation.toml": ExperimentMatrixSpec(
        evaluation_track=TRACK_SAME_CONTEXT,
        evidence_tier=EVIDENCE_DIAGNOSTIC,
        primary_method_name="sid_lite",
        best_no_comm_candidates=("mv_3",),
        full_comm_reference="always_full",
        token_gate_basis="communication",
    ),
    "configs/selective_comm/experiments/trigger_early_exit_main.toml": ExperimentMatrixSpec(
        evaluation_track=TRACK_SAME_CONTEXT,
        evidence_tier=EVIDENCE_HEADLINE,
        primary_method_name="hybrid_trigger",
        best_no_comm_candidates=("mv_3", "mv_6", "sc_6"),
        full_comm_reference="always_communicate",
        token_gate_basis="communication",
    ),
    "configs/selective_comm/experiments/voc_trigger_main.toml": ExperimentMatrixSpec(
        evaluation_track=TRACK_SAME_CONTEXT,
        evidence_tier=EVIDENCE_HEADLINE,
        primary_method_name="voc_trigger_v2",
        best_no_comm_candidates=("mv_3", "mv_6", "sc_6"),
        full_comm_reference="always_communicate",
        token_gate_basis="communication",
    ),
    "configs/sparc/experiments/end_to_end_main.toml": ExperimentMatrixSpec(
        evaluation_track=TRACK_SAME_CONTEXT,
        evidence_tier=EVIDENCE_HEADLINE,
        primary_method_name="sparc_v1",
        best_no_comm_candidates=("mv_3",),
        full_comm_reference="always_communicate",
        token_gate_basis="communication",
    ),
    "configs/sparc/experiments/content_ablation.toml": ExperimentMatrixSpec(
        evaluation_track=TRACK_SAME_CONTEXT,
        evidence_tier=EVIDENCE_DIAGNOSTIC,
        primary_method_name="task_adaptive",
        best_no_comm_candidates=("mv_3",),
        full_comm_reference="full_cot",
        token_gate_basis="communication",
    ),
    "configs/sparc/experiments/local_auditing_ablation.toml": ExperimentMatrixSpec(
        evaluation_track=TRACK_SAME_CONTEXT,
        evidence_tier=EVIDENCE_SUPPORTING,
        primary_method_name="local_auditing",
        best_no_comm_candidates=("majority_vote",),
        full_comm_reference="final_round_vote",
        token_gate_basis="none",
    ),
    "configs/cue/experiments/cue_black_box_utility_main.toml": ExperimentMatrixSpec(
        evaluation_track=TRACK_SAME_CONTEXT,
        evidence_tier=EVIDENCE_DIAGNOSTIC,
        primary_method_name="cue_v1",
        best_no_comm_candidates=("mv_3",),
        full_comm_reference="always_communicate",
        token_gate_basis="communication",
    ),

}


def get_experiment_matrix_spec(config_path: str) -> ExperimentMatrixSpec:
    """按实验配置路径返回既定的矩阵规格。"""
    try:
        return EXPERIMENT_MATRIX_SPECS[config_path]
    except KeyError as exc:
        raise KeyError(f"Missing matrix spec for config {config_path}") from exc
