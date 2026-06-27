from __future__ import annotations

from research_experiments.families.cred_v.run.sample import build_paired_comparisons


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


def _prediction(dataset: str, sample_id: str, method_name: str, score: float) -> dict:
    return {
        "dataset": dataset,
        "sample_id": sample_id,
        "method_name": method_name,
        "score": score,
    }
