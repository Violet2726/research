from __future__ import annotations

import pytest

from research_experiments.families.risk_controlled_trace_mad.run.metrics import build_metrics
from research_experiments.families.risk_controlled_trace_mad.statistics import paired_statistics


def _row(dataset: str, sample_id: str, task: str, method: str, score: float) -> dict:
    return {
        "dataset": dataset,
        "sample_id": sample_id,
        "task": task,
        "method_name": method,
        "score": score,
        "total_tokens_per_question": 1.0,
        "logical_calls_per_question": 1,
    }


def test_bbeh_primary_metric_and_bootstrap_use_task_harmonic() -> None:
    rows = []
    left = {"a": [1, 1, 1, 0], "b": [1, 1, 1, 1]}
    right = {"a": [1, 1, 0, 0], "b": [1, 1, 1, 1]}
    for task in ("a", "b"):
        for index, (left_score, right_score) in enumerate(zip(left[task], right[task], strict=True)):
            sample_id = f"{task}-{index}"
            rows.append(_row("bbeh", sample_id, task, "rcta_1", left_score))
            rows.append(_row("bbeh", sample_id, task, "sc_9", right_score))
    metrics = build_metrics(rows, dataset_order=["bbeh"], method_order=["rcta_1", "sc_9"], bbeh_harmonic=True)
    summary = {(row["dataset"], row["method_name"]): row for row in metrics["summary"]}
    assert summary[("bbeh", "rcta_1")]["accuracy_mean"] == pytest.approx(6 / 7)
    paired = paired_statistics(
        rows,
        reference="rcta_1",
        competitors=["sc_9"],
        seed=42,
        bootstrap_samples=200,
        bbeh_harmonic=True,
    )
    test = paired["tests"][0]
    assert test["accuracy_metric"] == "task_harmonic"
    assert test["mean_accuracy_delta"] == pytest.approx((6 / 7) - (2 / 3))
    assert test["holm_adjusted_p"] >= test["mcnemar_exact_p"]


def test_gpqa_is_not_in_primary_holm_family() -> None:
    rows = [
        _row("gpqa_diamond", "x", "science", "rcta_1", 1),
        _row("gpqa_diamond", "x", "science", "sc_9", 0),
    ]
    result = paired_statistics(rows, reference="rcta_1", competitors=["sc_9"], seed=42, bootstrap_samples=20)
    assert "holm_adjusted_p" not in result["tests"][0]
