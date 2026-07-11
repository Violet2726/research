from __future__ import annotations

from research_experiments.families.blind_reconstructive_mad.pilot_gate import evaluate_pilot_gate
from research_experiments.families.blind_reconstructive_mad.run.sample import apply_bbeh_harmonic_primary
from research_experiments.families.blind_reconstructive_mad.statistics import paired_method_statistics


def test_bbeh_statistics_are_paired_and_holm_adjusted() -> None:
    rows = []
    for task, sample, brd, comparator in [("t1", "a", 1.0, 0.0), ("t1", "b", 0.0, 0.0), ("t2", "c", 1.0, 1.0), ("t2", "d", 1.0, 0.0)]:
        for method, score in [("brd_quorum_3", brd), ("conditional_resample_3", comparator)]:
            rows.append({"dataset": "bbeh", "model_name": "m", "sample_id": sample, "task": task, "method_name": method, "score": score})

    result = paired_method_statistics(rows, bootstrap_samples=100, seed=3)

    assert result["bbeh_resampling"] == "within_task_stratified"
    test = result["tests"][0]
    assert test["absolute_accuracy_delta"] > 0
    assert "holm_adjusted_p_within_dataset" in test


def test_pilot_gate_requires_every_pre_registered_condition() -> None:
    prediction_rows = []
    for dataset in ("omni_math_2_filtered", "bbeh"):
        for index in range(20):
            sample_id = f"{dataset}-{index}"
            brd = {"dataset": dataset, "sample_id": sample_id, "method_name": "brd_quorum_3", "score": 1.0, "override_accepted": True, "corrected_by_debate": True, "harmed_by_debate": False}
            prediction_rows.extend(
                [
                    brd,
                    {"dataset": dataset, "sample_id": sample_id, "method_name": "sc_5", "score": 0.0},
                    {"dataset": dataset, "sample_id": sample_id, "method_name": "conditional_resample_3", "score": 0.0},
                ]
            )
    diagnostics = {
        "summary_rows": [
            {"dataset": "omni_math_2_filtered", "method_name": "brd_quorum_3", "candidate_oracle_gap_over_anchor": 0.03},
            {"dataset": "bbeh", "method_name": "brd_quorum_3", "candidate_oracle_gap_over_anchor": 0.04},
        ]
    }

    gate = evaluate_pilot_gate(prediction_rows=prediction_rows, turn_rows=[], diagnostics=diagnostics, model_name="m")

    assert gate["passed"]
    assert all(gate["conditions"].values())


def test_bbeh_task_harmonic_replaces_primary_metric_and_keeps_micro() -> None:
    rows = [
        {"dataset": "bbeh", "model_name": "m", "method_name": "sc_5", "task": "t1", "score": 1.0, "initial_vote_score": 1.0},
        {"dataset": "bbeh", "model_name": "m", "method_name": "sc_5", "task": "t2", "score": 0.5, "initial_vote_score": 0.5},
    ]
    metrics = {
        "summary": [
            {"dataset": "bbeh", "aggregate_kind": "dataset", "model_name": "m", "method_name": "sc_5", "accuracy_mean": 0.75, "initial_vote_accuracy_mean": 0.75, "total_tokens_mean": 10.0},
            {"dataset": "overall", "aggregate_kind": "macro", "model_name": "m", "method_name": "sc_5", "accuracy_mean": 0.75, "initial_vote_accuracy_mean": 0.75, "total_tokens_mean": 10.0},
        ]
    }

    result = apply_bbeh_harmonic_primary(metrics, rows, control_names=["sc_5"])
    bbeh = result["summary"][0]

    assert bbeh["primary_accuracy_metric"] == "task_harmonic"
    assert bbeh["micro_accuracy_mean"] == 0.75
    assert bbeh["accuracy_mean"] == 2 / 3


def test_bbeh_count_metric_keeps_micro_as_primary() -> None:
    rows = [
        {"dataset": "bbeh", "model_name": "m", "method_name": "sc_5", "task": "t1", "score": 1.0, "initial_vote_score": 1.0},
        {"dataset": "bbeh", "model_name": "m", "method_name": "sc_5", "task": "t2", "score": 0.0, "initial_vote_score": 0.0},
    ]
    metrics = {
        "summary": [
            {"dataset": "bbeh", "aggregate_kind": "dataset", "model_name": "m", "method_name": "sc_5", "accuracy_mean": 0.5, "initial_vote_accuracy_mean": 0.5, "total_tokens_mean": 10.0},
        ]
    }

    result = apply_bbeh_harmonic_primary(metrics, rows, control_names=["sc_5"], use_harmonic=False)
    bbeh = result["summary"][0]

    assert result["bbeh_metric"] == {"primary": "micro_accuracy", "secondary": None}
    assert bbeh["accuracy_mean"] == 0.5
    assert bbeh["micro_accuracy_mean"] == 0.5
    assert bbeh["bbeh_task_harmonic_accuracy"] is None
    assert bbeh["primary_accuracy_metric"] == "micro_accuracy"
