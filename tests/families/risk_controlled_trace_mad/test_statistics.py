import pytest

from research_experiments.families.risk_controlled_trace_mad.run.metrics import build_metrics
from research_experiments.families.risk_controlled_trace_mad.statistics import paired_statistics


def _row(dataset, sample, task, method, score):
    return {
        "dataset": dataset,
        "sample_id": sample,
        "task": task,
        "method_name": method,
        "score": score,
        "total_tokens_per_question": 1,
        "logical_calls_per_question": 1,
    }


def test_bbeh_harmonic_and_stratified_bootstrap() -> None:
    rows = []
    for sample, task, left, right in (("1", "a", 1, 1), ("2", "a", 1, 0), ("3", "b", 1, 1), ("4", "b", 1, 1)):
        rows += [_row("bbeh", sample, task, "evf_mad_1", left), _row("bbeh", sample, task, "qwen_sc_9", right)]
    metrics = build_metrics(rows, dataset_order=["bbeh"], method_order=["evf_mad_1", "qwen_sc_9"], bbeh_harmonic=True)
    summary = {(row["dataset"], row["method_name"]): row for row in metrics["summary"]}
    assert summary[("bbeh", "evf_mad_1")]["accuracy_mean"] == pytest.approx(1.0)
    result = paired_statistics(
        rows, reference="evf_mad_1", competitors=["qwen_sc_9"], seed=42, bootstrap_samples=50, bbeh_harmonic=True
    )
    assert result["bbeh_resampling"] == "within_task_stratified_harmonic"
