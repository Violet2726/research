"""SGSA-MAD 的 count100_seed42 到全量运行安全门。"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

PRIMARY_DATASETS = ("omni_math_2_filtered", "bbeh")
PRIMARY_METHOD = "sgsa_unanimous_3"


def evaluate_count100_gate(
    *,
    prediction_rows: list[dict[str, Any]],
    turn_rows: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    del diagnostics
    primary = [
        row
        for row in prediction_rows
        if row.get("method_name") == PRIMARY_METHOD and row.get("dataset") in PRIMARY_DATASETS
    ]
    by_dataset = {
        dataset: [row for row in primary if row.get("dataset") == dataset]
        for dataset in PRIMARY_DATASETS
    }
    oracle = {
        dataset: _mean(rows, "candidate_oracle_correct") - _mean(rows, "initial_vote_score")
        for dataset, rows in by_dataset.items()
    }
    deltas_sc5 = _paired_by_dataset(prediction_rows, PRIMARY_METHOD, "sc_5")
    deltas_resample = _paired_by_dataset(prediction_rows, PRIMARY_METHOD, "conditional_resample_3")
    corrected_by_dataset = {
        dataset: sum(bool(row.get("corrected_by_debate")) for row in rows)
        for dataset, rows in by_dataset.items()
    }
    harmed_by_dataset = {
        dataset: sum(bool(row.get("harmed_by_debate")) for row in rows)
        for dataset, rows in by_dataset.items()
    }
    overrides = [row for row in primary if row.get("override_accepted")]
    corrected = sum(corrected_by_dataset.values())
    harmed = sum(harmed_by_dataset.values())
    decisive = corrected + harmed
    request_failures = sum(
        bool(row.get("request_error")) or row.get("request_status") == "request_fail"
        for row in turn_rows
    )
    protocol_failures = sum(row.get("protocol_parse_status") == "failed" for row in turn_rows)
    conditions = {
        "zero_request_failures": request_failures == 0,
        "zero_protocol_failures": protocol_failures == 0,
        "oracle_gap_at_least_3pp_on_both_primary_sets": all(
            oracle[dataset] >= 0.03 for dataset in PRIMARY_DATASETS
        ),
        "sgsa_positive_vs_sc5_on_each_primary_set": all(
            deltas_sc5[dataset]["net_score_delta"] > 0 for dataset in PRIMARY_DATASETS
        ),
        "sgsa_positive_vs_conditional_resample_on_each_primary_set": all(
            deltas_resample[dataset]["net_score_delta"] > 0 for dataset in PRIMARY_DATASETS
        ),
        "corrected_exceeds_harmed_on_each_primary_set": all(
            corrected_by_dataset[dataset] > harmed_by_dataset[dataset]
            for dataset in PRIMARY_DATASETS
        ),
        "at_least_20_overrides": len(overrides) >= 20,
        "decisive_override_precision_at_least_two_thirds": (
            corrected / decisive >= 2 / 3 if decisive else False
        ),
    }
    return {
        "gate_name": "sgsa_mad_count100_seed42_v1",
        "model_name": model_name,
        "primary_method": PRIMARY_METHOD,
        "primary_datasets": list(PRIMARY_DATASETS),
        "conditions": conditions,
        "passed": all(conditions.values()),
        "evidence": {
            "candidate_oracle_gap_over_anchor": oracle,
            "paired_vs_sc5": deltas_sc5,
            "paired_vs_conditional_resample": deltas_resample,
            "corrected_by_dataset": corrected_by_dataset,
            "harmed_by_dataset": harmed_by_dataset,
            "override_count": len(overrides),
            "corrected": corrected,
            "harmed": harmed,
            "decisive_override_precision": corrected / decisive if decisive else 0.0,
            "request_failures": request_failures,
            "protocol_failures": protocol_failures,
        },
    }


def _paired_by_dataset(
    rows: list[dict[str, Any]], reference: str, competitor: str
) -> dict[str, dict[str, float | int]]:
    keyed: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        dataset = str(row.get("dataset") or "")
        method = str(row.get("method_name") or "")
        if dataset in PRIMARY_DATASETS and method in {reference, competitor}:
            keyed[(dataset, str(row.get("sample_id") or ""))][method] = float(row.get("score") or 0.0)
    result: dict[str, dict[str, float | int]] = {}
    for dataset in PRIMARY_DATASETS:
        deltas = [
            values[reference] - values[competitor]
            for (row_dataset, _), values in keyed.items()
            if row_dataset == dataset and reference in values and competitor in values
        ]
        result[dataset] = {
            "paired_question_count": len(deltas),
            "net_score_delta": sum(deltas),
            "mean_accuracy_delta": sum(deltas) / len(deltas) if deltas else 0.0,
        }
    return result


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    return sum(float(row.get(field) or 0.0) for row in rows) / len(rows) if rows else 0.0
