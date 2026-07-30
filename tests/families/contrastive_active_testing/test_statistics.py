from __future__ import annotations

import pytest

from research_experiments.families.contrastive_active_testing.statistics import (
    _certificate_v2_diagnostics,
    _clopper_pearson_lower,
    _kernel_diagnostics,
    _summary,
    _v3_observation_diagnostics,
    _v3_panel_dependence,
    materialize_development_catch,
)
from research_experiments.reporting.paired_inference import paired_statistics


def test_development_grid_selection_is_global_and_materializes_one_primary_method() -> None:
    predictions = []
    routers = []
    for index in range(10):
        base = {
            "sample_id": str(index),
            "dataset": "bbeh",
            "task": "t",
            "candidate_oracle_correct": True,
            "triggered": True,
            "total_tokens_per_question": 8,
        }
        predictions.extend(
            [
                {
                    **base,
                    "method_name": "catch_d2_m1",
                    "score": 1,
                    "override_accepted": True,
                    "corrected_by_debate": index < 5,
                    "harmed_by_debate": False,
                },
                {
                    **base,
                    "method_name": "catch_d3_m2",
                    "score": int(index < 7),
                    "override_accepted": index < 7,
                    "corrected_by_debate": index < 4,
                    "harmed_by_debate": False,
                },
            ]
        )
        routers.append(
            {
                "triggered": True,
                "catch_variants": [
                    {"d_min": 2, "margin": 1, "pair_distances": {"B": 2}},
                    {"d_min": 3, "margin": 2, "pair_distances": {"B": 3}},
                ],
            }
        )

    materialized, selection = materialize_development_catch(predictions, routers)
    assert selection["selected"]["method_name"] == "catch_d2_m1"
    assert len([row for row in materialized if row["method_name"] == "catch"]) == 10
    assert not any(row["method_name"] == "catch_d2_m1" for row in materialized)


def test_exact_one_sided_override_precision_bound_is_conservative() -> None:
    assert _clopper_pearson_lower(0, 0, alpha=0.05) == 0.0
    assert _clopper_pearson_lower(8, 10, alpha=0.05) < 0.65
    assert _clopper_pearson_lower(20, 20, alpha=0.05) > 0.85


def test_bbeh_full_paired_inference_uses_adjusted_harmonic_scale() -> None:
    rows = [
        {"dataset": "bbeh", "sample_id": "1", "task": "t1", "method_name": "d3", "score": 1},
        {"dataset": "bbeh", "sample_id": "1", "task": "t1", "method_name": "sc8", "score": 0},
        {"dataset": "bbeh", "sample_id": "2", "task": "t2", "method_name": "d3", "score": 0},
        {"dataset": "bbeh", "sample_id": "2", "task": "t2", "method_name": "sc8", "score": 0},
        {"dataset": "musr", "sample_id": "m1", "task": "team_allocation", "method_name": "d3", "score": 1},
        {"dataset": "musr", "sample_id": "m1", "task": "team_allocation", "method_name": "sc8", "score": 0},
    ]
    result = paired_statistics(
        rows,
        reference="d3",
        competitors=["sc8"],
        seed=42,
        bootstrap_samples=100,
        bbeh_adjusted_harmonic=True,
    )
    test = next(item for item in result["tests"] if item["dataset"] == "bbeh")
    expected = 2 / (1 / 1.01 + 1 / 0.01) - 0.01
    assert test["accuracy_metric"] == "task_stratified_micro_accuracy"
    assert test["mean_accuracy_delta"] == pytest.approx(0.5)
    assert test["bbeh_adjusted_harmonic_delta"] == pytest.approx(expected)
    assert result["bbeh_resampling"] == "within_task_stratified_micro_with_secondary_adjusted_harmonic"
    assert result["holm_scope"] == ["bbeh", "musr", "gpqa_diamond"]
    musr = next(item for item in result["tests"] if item["dataset"] == "musr")
    assert musr["accuracy_metric"] == "task_macro_accuracy"
    assert musr["holm_adjusted_p"] is not None


def test_gpqa_summary_reports_available_domains_without_inventing_reasoning_labels() -> None:
    rows = [
        {"score": 1, "high_level_domain": "Physics", "subdomain": "Optics"},
        {"score": 0, "high_level_domain": "Physics", "subdomain": "Mechanics"},
        {"score": 1, "high_level_domain": "Chemistry", "subdomain": "Organic"},
    ]
    summary = _summary("catch_kernel", rows)
    assert summary["per_domain_accuracy"] == {"Physics": 0.5, "Chemistry": 1.0}
    assert summary["per_subdomain_accuracy"]["Optics"] == 1.0
    assert summary["per_reasoning_type_accuracy"] == {}
    assert summary["reasoning_type_stratification_available"] is False


def test_science_paired_inference_uses_domain_macro_and_rejects_duplicates() -> None:
    rows = []
    for sample_id, domain, left, right in (
        ("p1", "Physics", 1, 0),
        ("p2", "Physics", 0, 0),
        ("p3", "Physics", 0, 0),
        ("c1", "Chemistry", 0, 1),
    ):
        rows.extend(
            [
                {
                    "dataset": "supergpqa_science",
                    "sample_id": sample_id,
                    "high_level_domain": domain,
                    "method_name": "d4",
                    "score": left,
                },
                {
                    "dataset": "supergpqa_science",
                    "sample_id": sample_id,
                    "high_level_domain": domain,
                    "method_name": "sc5",
                    "score": right,
                },
            ]
        )
    result = paired_statistics(rows, reference="d4", competitors=["sc5"], seed=42, bootstrap_samples=20)
    test = result["tests"][0]
    assert test["accuracy_metric"] == "domain_macro_accuracy"
    assert test["mean_accuracy_delta"] == pytest.approx((1 / 3 - 1) / 2)
    assert result["holm_scope"] == ["bbeh", "musr", "supergpqa_science"]
    with pytest.raises(ValueError, match="Duplicate paired-inference row"):
        paired_statistics([*rows, rows[0]], reference="d4", competitors=["sc5"], seed=42)


def test_v3_reports_correlated_false_passes_and_position_diagnostics() -> None:
    routers = []
    for index, passes in enumerate(((True, True), (True, False), (False, False))):
        routers.append(
            {
                "protocol_version": "catch_v3",
                "gold_candidate_key": "A",
                "decision": {
                    "panel_diagnostics": [
                        {"challenger_key": "B", "panel_index": 1, "passed": passes[0]},
                        {"challenger_key": "B", "panel_index": 2, "passed": passes[1]},
                    ]
                },
                "witness_panels": [
                    {"observations": {"C0": "B" if index < 2 else "ERASURE"}},
                    {"observations": {"C0": "B" if index == 0 else "A"}},
                ],
            }
        )
    dependence = _v3_panel_dependence(routers)
    assert dependence["panel_1_false_pass_rate"] == 2 / 3
    assert dependence["panel_2_false_pass_rate"] == 1 / 3
    assert dependence["joint_false_pass_rate"] == 1 / 3
    assert dependence["bernoulli_correlation"] == pytest.approx(0.5)

    turns = [
        {
            "role": "icv_witness",
            "witness_packet": {"public_to_internal": {"X0": "C0", "X1": "C1"}},
            "validated_output": {
                "answers": [
                    {"contrast_id": "X0", "verdict": "LEFT_ONLY"},
                    {"contrast_id": "X1", "verdict": "RIGHT_ONLY"},
                ]
            },
        }
    ]
    observations = _v3_observation_diagnostics(routers, turns)
    assert observations["inverse_mapped_panel_agreement_rate"] == 0.5
    assert observations["left_only_share_among_decisive"] == 0.5


def test_cert_v2_summary_and_mechanism_include_answer_link_obligation_and_adapter_metrics() -> None:
    rows = [
        {
            "method_name": "catch_cert_v2",
            "score": 1,
            "initial_vote_score": 0,
            "corrected_by_debate": True,
            "harmed_by_debate": False,
            "candidate_oracle_correct": True,
            "target_oracle_correct": True,
            "total_tokens_per_question": 1_000,
            "calls_per_question": 6,
            "certificate_coverage": 1,
            "override_accepted": True,
            "certificate_abstained": False,
            "answer_link_coverage": 1,
            "obligation_coverage": 1,
            "adapter_executed_test_count": 2,
            "verifier_format_repair_count": 1,
        }
    ]
    summary = _summary("catch_cert_v2", rows)
    assert summary["correct_per_1000_tokens"] == 1
    assert summary["answer_link_coverage"] == 1
    assert summary["obligation_coverage"] == 1
    assert summary["adapter_executed_test_count"] == 2
    assert summary["verifier_format_repair_count"] == 1

    diagnostics = _certificate_v2_diagnostics(
        [
            {
                "protocol_version": "catch_cert_v2",
                "triggered": True,
                "answer_link_coverage": 1,
                "obligation_coverage": 1,
                "certificate_tests": [{"test_id": "T0"}],
                "certificates": [{"candidate_key_anon": "H1"}],
                "adapter_results": {"T0": {"execution_status": "EXECUTED"}},
                "verifier_panels": [],
                "decision": {"override_accepted": True},
            }
        ]
    )
    assert diagnostics["adapter_executed_test_count"] == 1
    assert diagnostics["override_count"] == 1


def test_kernel_summary_separates_protocol_semantics_jurisdiction_and_proofs() -> None:
    row = {
        "method_name": "catch_kernel",
        "score": 1,
        "initial_vote_score": 0,
        "corrected_by_debate": True,
        "harmed_by_debate": False,
        "candidate_oracle_correct": True,
        "target_oracle_correct": True,
        "total_tokens_per_question": 1_000,
        "calls_per_question": 5,
        "latency_ms_per_question": 125,
        "cache_hits_per_question": 4,
        "network_calls_per_question": 1,
        "certificate_coverage": 1,
        "override_accepted": True,
        "certificate_abstained": False,
        "answer_link_coverage": 1,
        "obligation_coverage": 1,
        "syntax_validity": 1,
        "schema_validity": 1,
        "typed_compilation_validity": 1,
        "semantic_validity": None,
        "contract_accuracy": None,
        "verifier_jurisdiction_coverage": 1,
        "proof_completeness": 1,
        "proof_pass_count": 1,
        "proof_conflict_count": 0,
        "proof_unsupported_count": 0,
        "proof_unknown_count": 0,
    }
    summary = _summary("catch_kernel", [row])
    assert summary["syntax_validity"] == 1
    assert summary["typed_compilation_validity"] == 1
    assert summary["semantic_validity"] is None
    assert summary["contract_accuracy"] is None
    assert summary["verifier_jurisdiction_coverage"] == 1
    assert summary["proof_completeness"] == 1
    assert summary["mean_latency_ms_per_question"] == 125
    assert summary["mean_cache_hits_per_question"] == 4
    assert summary["mean_network_calls_per_question"] == 1
    diagnostics = _kernel_diagnostics(
        [
            {
                "protocol_version": "catch_kernel_v1",
                "triggered": True,
                "verifier_bindings": {
                    "T0": {
                        "binding_status": "BOUND",
                        "verifier_kind": "deterministic.seq_plan",
                    }
                },
                "proof_results": [
                    {
                            "status": "PASS",
                            "provenance_valid": True,
                            "entailment_valid": True,
                            "obligation_valid": True,
                        "sufficiency_valid": True,
                    }
                ],
                "kernel_decision": {"failure_layer": "none"},
            }
        ]
    )
    assert diagnostics["jurisdiction_coverage"] == 1
    assert diagnostics["proof_completeness"] == 1


def test_kernel_route_quality_accepts_multiple_scorer_valid_answer_classes() -> None:
    diagnostics = _kernel_diagnostics(
        [
            {
                "protocol_version": "catch_kernel_v1",
                "triggered": True,
                "anchor_key": "A",
                "gold_candidate_key": "A",
                "gold_candidate_keys": ["A", "B"],
                "candidate_public_to_answer_class_key": {"H1": "B"},
                "verifier_bindings": {
                    "T0": {"binding_status": "BOUND", "verifier_kind": "model.bounded_semantic_panel"}
                },
                "proof_results": [
                    {
                        "test_id": "T0",
                        "verifier_kind": "model.bounded_semantic_panel",
                        "status": "PASS",
                        "provenance_valid": True,
                        "entailment_valid": True,
                        "obligation_valid": True,
                        "sufficiency_valid": True,
                    }
                ],
                "kernel_decision": {
                    "decision": "OVERRIDE",
                    "challenger_id": "H1",
                    "accepted_proofs": ["T0"],
                    "failure_layer": "none",
                },
            }
        ]
    )
    quality = diagnostics["verifier_route_quality"]["model.bounded_semantic_panel"]
    assert quality["correct_to_correct"] == 1
    assert quality["harm_rate"] == 0
    assert diagnostics["cross_jurisdiction_fallback_count"] == 0
