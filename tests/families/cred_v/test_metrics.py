from __future__ import annotations

from research_experiments.families.cred_v.run.sample import build_metrics, build_paired_comparisons


def test_paired_comparison_reports_mcnemar_and_bootstrap_ci() -> None:
    rows = [
        _prediction("d", "1", "sc_5", 1.0),
        _prediction("d", "1", "cred_rfs_adaptive_sc_v1", 1.0),
        _prediction("d", "2", "sc_5", 0.0),
        _prediction("d", "2", "cred_rfs_adaptive_sc_v1", 1.0),
        _prediction("d", "3", "sc_5", 1.0),
        _prediction("d", "3", "cred_rfs_adaptive_sc_v1", 0.0),
        _prediction("d", "4", "sc_5", 0.0),
        _prediction("d", "4", "cred_rfs_adaptive_sc_v1", 1.0),
    ]

    paired = build_paired_comparisons(
        rows,
        dataset_order=["d"],
        method_order=["sc_5", "cred_rfs_adaptive_sc_v1"],
        reference_method="sc_5",
    )
    dataset_row = next(row for row in paired if row["dataset"] == "d")

    assert dataset_row["accuracy_delta"] == 0.25
    assert dataset_row["wins"] == 2
    assert dataset_row["losses"] == 1
    assert dataset_row["ties"] == 1
    assert 0.0 <= dataset_row["mcnemar_p"] <= 1.0
    assert dataset_row["bootstrap_ci_low"] <= dataset_row["accuracy_delta"] <= dataset_row["bootstrap_ci_high"]


def test_vote_anchor_summary_does_not_inherit_safe_select_expansion_counts() -> None:
    rows = [
        _summary_prediction("gpqa_diamond", "1", "sc_5", 1.0),
        _summary_prediction("gpqa_diamond", "1", "cred_rfs_vote_5_anchor", 1.0),
        _summary_prediction(
            "gpqa_diamond",
            "1",
            "cred_rfs_safe_select_v3",
            1.0,
            pairwise_duel_count=3,
            pairwise_duel_win_count=3,
            gpqa_unanimous_duel_count=3,
            method_expansion_call_count=3,
        ),
    ]

    metrics = build_metrics(
        rows,
        dataset_order=["gpqa_diamond"],
        method_order=["sc_5", "cred_rfs_vote_5_anchor", "cred_rfs_safe_select_v3"],
        control_names=["sc_5"],
    )
    summary = metrics["summary"]
    anchor = next(row for row in summary if row["dataset"] == "overall_micro" and row["method_name"] == "cred_rfs_vote_5_anchor")
    safe = next(row for row in summary if row["dataset"] == "overall_micro" and row["method_name"] == "cred_rfs_safe_select_v3")

    assert anchor["pairwise_duel_count"] == 0
    assert anchor["minority_probe_count"] == 0
    assert anchor["method_expansion_call_count"] == 0
    assert safe["pairwise_duel_count"] == 3
    assert safe["gpqa_unanimous_duel_count"] == 3
    assert safe["method_expansion_call_count"] == 3


def test_shadow_counterfactual_metrics_are_reported_separately_from_actual_accuracy() -> None:
    rows = [
        _summary_prediction("gpqa_diamond", "1", "sc_5", 0.0),
        _summary_prediction(
            "gpqa_diamond",
            "1",
            "cred_rfs_shadow_select_v4",
            0.0,
            pairwise_duel_count=5,
            method_expansion_call_count=5,
            shadow_counterfactual_corrected=True,
            shadow_gate_passed=True,
            shadow_net_gain=1,
        ),
        _summary_prediction("gpqa_diamond", "2", "sc_5", 1.0),
        _summary_prediction(
            "gpqa_diamond",
            "2",
            "cred_rfs_shadow_select_v4",
            1.0,
            pairwise_duel_count=5,
            method_expansion_call_count=5,
            shadow_counterfactual_harmed=True,
            shadow_gate_passed=True,
            shadow_net_gain=-1,
            duel_invalid_count=1,
            duel_retry_recoverable_count=1,
        ),
    ]

    metrics = build_metrics(
        rows,
        dataset_order=["gpqa_diamond"],
        method_order=["sc_5", "cred_rfs_shadow_select_v4"],
        control_names=["sc_5"],
    )
    shadow = next(row for row in metrics["summary"] if row["dataset"] == "overall_micro" and row["method_name"] == "cred_rfs_shadow_select_v4")

    assert shadow["accuracy_mean"] == 0.5
    assert shadow["shadow_counterfactual_corrected_count"] == 1
    assert shadow["shadow_counterfactual_harmed_count"] == 1
    assert shadow["shadow_precision"] == 0.5
    assert shadow["shadow_net_gain"] == 0
    assert shadow["shadow_gate_passed_count"] == 2
    assert shadow["duel_invalid_count"] == 1
    assert shadow["duel_retry_recoverable_count"] == 1


def test_v5_incremental_metrics_are_reported_against_v3() -> None:
    rows = [
        _summary_prediction("math500", "1", "sc_5", 0.0),
        _summary_prediction("math500", "1", "cred_rfs_safe_select_v3", 0.0),
        _summary_prediction(
            "math500",
            "1",
            "cred_rfs_evidence_repair_v5",
            1.0,
            math_repair_applied=True,
            resolver="cred_rfs_v5_math_equivalence_repair_v2",
        ),
        _summary_prediction("math500", "2", "sc_5", 1.0),
        _summary_prediction("math500", "2", "cred_rfs_safe_select_v3", 1.0),
        _summary_prediction(
            "math500",
            "2",
            "cred_rfs_evidence_repair_v5",
            0.0,
            resolver="cred_rfs_v5_pairwise_rejected",
        ),
    ]

    metrics = build_metrics(
        rows,
        dataset_order=["math500"],
        method_order=["sc_5", "cred_rfs_safe_select_v3", "cred_rfs_evidence_repair_v5"],
        control_names=["sc_5"],
    )
    v5 = next(row for row in metrics["summary"] if row["dataset"] == "overall_micro" and row["method_name"] == "cred_rfs_evidence_repair_v5")

    assert v5["math_equivalence_repair_v2_count"] == 1
    assert v5["v5_incremental_corrected_vs_v3"] == 1
    assert v5["v5_incremental_harmed_vs_v3"] == 1
    assert v5["v5_actual_gain_vs_v3"] == 0.0


def test_protocol_recovery_metrics_are_aggregated() -> None:
    rows = [
        _summary_prediction(
            "gpqa_diamond",
            "1",
            "cred_rfs_safe_select_v3",
            1.0,
            free_text_recovered_count=2,
            pairwise_json_recovered_count=1,
            json_truncated_count=3,
        )
    ]

    metrics = build_metrics(
        rows,
        dataset_order=["gpqa_diamond"],
        method_order=["cred_rfs_safe_select_v3"],
        control_names=[],
    )
    summary = next(row for row in metrics["summary"] if row["dataset"] == "overall_micro")

    assert summary["free_text_recovered_count"] == 2
    assert summary["pairwise_json_recovered_count"] == 1
    assert summary["json_truncated_count"] == 3


def test_repair_only_and_semantic_selector_metrics_are_separated() -> None:
    rows = [
        _summary_prediction("gpqa_diamond", "1", "sc_5", 0.0),
        _summary_prediction(
            "gpqa_diamond",
            "1",
            "cred_rfs_repair_only_v6",
            1.0,
            corrected_by_debate=True,
            repair_only_corrected=True,
            resolver="cred_rfs_v6_math_repair",
        ),
        _summary_prediction("gpqa_diamond", "2", "cred_rfs_repair_only_v6", 0.0, resolver="cred_rfs_v6_repair_only_rejected"),
        _summary_prediction("gpqa_diamond", "3", "cred_rfs_repair_only_v6", 1.0, resolver="cred_rfs_v6_repair_only_rejected"),
        _summary_prediction("gpqa_diamond", "2", "sc_5", 0.0),
        _summary_prediction(
            "gpqa_diamond",
            "2",
            "cred_rfs_safe_select_v3",
            1.0,
            corrected_by_debate=True,
            semantic_selector_corrected=True,
            resolver="cred_rfs_v3_gpqa_unanimous_pairwise_promoted",
        ),
        _summary_prediction("gpqa_diamond", "3", "sc_5", 1.0),
        _summary_prediction(
            "gpqa_diamond",
            "3",
            "cred_rfs_safe_select_v3",
            0.0,
            harmed_by_debate=True,
            semantic_selector_harmed=True,
            resolver="cred_rfs_v3_gpqa_unanimous_pairwise_promoted",
        ),
    ]

    metrics = build_metrics(
        rows,
        dataset_order=["gpqa_diamond"],
        method_order=["sc_5", "cred_rfs_repair_only_v6", "cred_rfs_safe_select_v3"],
        control_names=["sc_5"],
    )
    v6 = next(row for row in metrics["summary"] if row["dataset"] == "overall_micro" and row["method_name"] == "cred_rfs_repair_only_v6")
    v3 = next(row for row in metrics["summary"] if row["dataset"] == "overall_micro" and row["method_name"] == "cred_rfs_safe_select_v3")

    assert v6["repair_only_corrected_count"] == 1
    assert v6["repair_only_harmed_count"] == 0
    assert v6["repair_only_gain_vs_sc5"] == 0.333334
    assert v3["semantic_selector_corrected_count"] == 1
    assert v3["semantic_selector_harmed_count"] == 1
    assert v3["semantic_selector_precision"] == 0.5
    assert v3["pairwise_duel_precision"] == 0.5


def test_v7_shadow_counterfactual_aliases_and_cross_view_agreement_are_reported() -> None:
    rows = [
        _summary_prediction("gpqa_diamond", "1", "sc_5", 0.0),
        _summary_prediction(
            "gpqa_diamond",
            "1",
            "cred_rfs_shadow_evidence_select_v7",
            0.0,
            method_expansion_call_count=3,
            shadow_counterfactual_corrected=True,
            shadow_gate_passed=True,
            shadow_net_gain=1,
            shadow_cross_view_agreement_count=3,
        ),
    ]

    metrics = build_metrics(
        rows,
        dataset_order=["gpqa_diamond"],
        method_order=["sc_5", "cred_rfs_shadow_evidence_select_v7"],
        control_names=["sc_5"],
    )
    shadow = next(row for row in metrics["summary"] if row["dataset"] == "overall_micro" and row["method_name"] == "cred_rfs_shadow_evidence_select_v7")

    assert shadow["accuracy_mean"] == 0.0
    assert shadow["shadow_counterfactual_precision"] == 1.0
    assert shadow["shadow_counterfactual_net_gain"] == 1
    assert shadow["shadow_cross_view_agreement_count"] == 3


def test_v8_incremental_metrics_are_reported_against_v6() -> None:
    rows = [
        _summary_prediction("math500", "1", "sc_5", 0.0),
        _summary_prediction("math500", "1", "cred_rfs_repair_only_v6", 0.0),
        _summary_prediction(
            "math500",
            "1",
            "cred_rfs_repair_bank_v8",
            1.0,
            corrected_by_debate=True,
            repair_bank_corrected=True,
            resolver="cred_rfs_v8_math_repair",
        ),
        _summary_prediction("math500", "2", "sc_5", 1.0),
        _summary_prediction("math500", "2", "cred_rfs_repair_only_v6", 1.0),
        _summary_prediction(
            "math500",
            "2",
            "cred_rfs_repair_bank_v8",
            1.0,
            resolver="cred_rfs_v8_repair_bank_rejected",
        ),
    ]

    metrics = build_metrics(
        rows,
        dataset_order=["math500"],
        method_order=["sc_5", "cred_rfs_repair_only_v6", "cred_rfs_repair_bank_v8"],
        control_names=["sc_5"],
    )
    v8 = next(row for row in metrics["summary"] if row["dataset"] == "overall_micro" and row["method_name"] == "cred_rfs_repair_bank_v8")

    assert v8["repair_bank_corrected_count"] == 1
    assert v8["repair_bank_harmed_count"] == 0
    assert v8["repair_bank_precision"] == 1.0
    assert v8["v8_incremental_corrected_vs_v6"] == 1
    assert v8["v8_incremental_harmed_vs_v6"] == 0
    assert v8["v8_actual_gain_vs_v6"] == 0.5


def test_v9_certificate_shadow_metrics_are_reported_separately() -> None:
    rows = [
        _summary_prediction("gpqa_diamond", "1", "sc_5", 0.0),
        _summary_prediction(
            "gpqa_diamond",
            "1",
            "cred_rfs_certificate_shadow_v9",
            0.0,
            method_expansion_call_count=3,
            shadow_counterfactual_corrected=True,
            certificate_shadow_corrected=True,
            shadow_gate_passed=True,
            shadow_net_gain=1,
            certificate_shadow_valid_count=3,
        ),
        _summary_prediction("gpqa_diamond", "2", "sc_5", 1.0),
        _summary_prediction(
            "gpqa_diamond",
            "2",
            "cred_rfs_certificate_shadow_v9",
            1.0,
            method_expansion_call_count=3,
            shadow_counterfactual_harmed=True,
            certificate_shadow_harmed=True,
            shadow_gate_passed=True,
            shadow_net_gain=-1,
            certificate_shadow_valid_count=3,
        ),
    ]

    metrics = build_metrics(
        rows,
        dataset_order=["gpqa_diamond"],
        method_order=["sc_5", "cred_rfs_certificate_shadow_v9"],
        control_names=["sc_5"],
    )
    shadow = next(row for row in metrics["summary"] if row["dataset"] == "overall_micro" and row["method_name"] == "cred_rfs_certificate_shadow_v9")

    assert shadow["accuracy_mean"] == 0.5
    assert shadow["certificate_shadow_corrected_count"] == 1
    assert shadow["certificate_shadow_harmed_count"] == 1
    assert shadow["certificate_shadow_precision"] == 0.5
    assert shadow["certificate_shadow_net_gain"] == 0
    assert shadow["certificate_shadow_valid_count"] == 6


def _prediction(dataset: str, sample_id: str, method_name: str, score: float) -> dict:
    return {
        "dataset": dataset,
        "sample_id": sample_id,
        "method_name": method_name,
        "score": score,
    }


def _summary_prediction(
    dataset: str,
    sample_id: str,
    method_name: str,
    score: float,
    *,
    pairwise_duel_count: int = 0,
    pairwise_duel_win_count: int = 0,
    gpqa_unanimous_duel_count: int = 0,
    method_expansion_call_count: int = 0,
    shadow_counterfactual_corrected: bool = False,
    shadow_counterfactual_harmed: bool = False,
    shadow_gate_passed: bool = False,
    shadow_net_gain: int = 0,
    duel_invalid_count: int = 0,
    duel_retry_recoverable_count: int = 0,
    free_text_recovered_count: int = 0,
    pairwise_json_recovered_count: int = 0,
    json_truncated_count: int = 0,
    math_repair_applied: bool = False,
    hotpot_span_repair_applied: bool = False,
    corrected_by_debate: bool = False,
    harmed_by_debate: bool = False,
    repair_only_corrected: bool = False,
    repair_only_harmed: bool = False,
    semantic_selector_corrected: bool = False,
    semantic_selector_harmed: bool = False,
    repair_bank_corrected: bool = False,
    repair_bank_harmed: bool = False,
    certificate_shadow_corrected: bool = False,
    certificate_shadow_harmed: bool = False,
    certificate_shadow_valid_count: int = 0,
    shadow_cross_view_agreement_count: int = 0,
    resolver: str | None = None,
) -> dict:
    return {
        "dataset": dataset,
        "sample_id": sample_id,
        "method_name": method_name,
        "method_type": "control" if method_name == "sc_5" else "mad",
        "model_name": "xiaomimimo/mimo-v2.5",
        "score": score,
        "initial_vote_score": score,
        "total_tokens_per_question": 100.0,
        "prompt_tokens_per_question": 60.0,
        "completion_tokens_per_question": 40.0,
        "latency_ms_per_question": 1.0,
        "debate_total_tokens_per_question": 0.0,
        "debate_prompt_tokens_per_question": 0.0,
        "debate_completion_tokens_per_question": 0.0,
        "debate_latency_ms_per_question": 0.0,
        "calls_per_question": 5 + method_expansion_call_count,
        "debate_rounds": 0,
        "agent_count": 5,
        "initial_consensus": False,
        "final_consensus": True,
        "vote_flipped": False,
        "corrected_by_debate": corrected_by_debate,
        "harmed_by_debate": harmed_by_debate,
        "triggered": method_expansion_call_count > 0,
        "oracle_candidate_correct": score == 1.0,
        "stage_candidate_oracle_correct": score == 1.0,
        "candidate_pool_oracle_correct": score == 1.0,
        "expansion_oracle_correct": False,
        "wrong_majority_some_correct": False,
        "target_correct": None,
        "safe_repair_applied": False,
        "hetero_agreement_applied": False,
        "expansion_call_count": method_expansion_call_count,
        "method_expansion_call_count": method_expansion_call_count,
        "false_consensus_triggered": False,
        "math_repair_applied": math_repair_applied,
        "hotpot_span_repair_applied": hotpot_span_repair_applied,
        "choice_shuffle_agreement_count": 0,
        "single_pro_promotion_blocked": False,
        "strong_majority_locked": False,
        "pairwise_duel_count": pairwise_duel_count,
        "pairwise_duel_win_count": pairwise_duel_win_count,
        "safe_selector_corrected": False,
        "safe_selector_harmed": False,
        "repair_only_corrected": repair_only_corrected,
        "repair_only_harmed": repair_only_harmed,
        "semantic_selector_corrected": semantic_selector_corrected,
        "semantic_selector_harmed": semantic_selector_harmed,
        "repair_bank_corrected": repair_bank_corrected,
        "repair_bank_harmed": repair_bank_harmed,
        "certificate_shadow_corrected": certificate_shadow_corrected,
        "certificate_shadow_harmed": certificate_shadow_harmed,
        "certificate_shadow_valid_count": certificate_shadow_valid_count,
        "gpqa_unanimous_duel_count": gpqa_unanimous_duel_count,
        "blocked_2of3_pairwise_count": 0,
        "blocked_mmlu_pairwise_count": 0,
        "blocked_strategyqa_probe_count": 0,
        "shadow_counterfactual_corrected": shadow_counterfactual_corrected,
        "shadow_counterfactual_harmed": shadow_counterfactual_harmed,
        "shadow_gate_passed": shadow_gate_passed,
        "shadow_net_gain": shadow_net_gain,
        "shadow_cross_view_agreement_count": shadow_cross_view_agreement_count,
        "duel_invalid_count": duel_invalid_count,
        "duel_retry_recoverable_count": duel_retry_recoverable_count,
        "free_text_recovered_count": free_text_recovered_count,
        "pairwise_json_recovered_count": pairwise_json_recovered_count,
        "json_truncated_count": json_truncated_count,
        "minority_probe_count": 0,
        "non_answer_candidate_blocked": False,
        "false_consensus_recovered": False,
        "protocol_failures_per_question": 0,
        "reason_missing_turns_per_question": 0,
        "resolver": resolver or ("no_comm_control" if method_name == "sc_5" else "cred_rfs_v3_rejected"),
        "router_reasons": [],
    }
