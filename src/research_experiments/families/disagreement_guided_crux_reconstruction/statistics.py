"""DGCR 指标及预注册门控判定。"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


METHOD_ORDER = ("sc_5", "adaptive_sc_8", "dgcr")


def build_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        values = [row for row in predictions if row.get("method_name") == method]
        rows.append(_summary(method, values))
    return {"summary": rows}


def evaluate_gate(*, phase_name: str, predictions: list[dict[str, Any]], turns: list[dict[str, Any]], routers: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = {row["method_name"]: row for row in build_metrics(predictions)["summary"]}
    sc5 = summaries.get("sc_5", {})
    adaptive = summaries.get("adaptive_sc_8", {})
    dgcr = summaries.get("dgcr", {})
    dgcr_rows = [row for row in predictions if row.get("method_name") == "dgcr"]
    triggered = [row for row in routers if row.get("triggered")]
    valid_crux_rate = _ratio(sum(bool(row.get("crux_valid")) for row in triggered), len(triggered))
    overrides = [row for row in dgcr_rows if row.get("override_accepted")]
    correct_overrides = [row for row in overrides if float(row.get("score") or 0) == 1.0]
    precision = _ratio(len(correct_overrides), len(overrides))
    correction = sum(bool(row.get("corrected_by_debate")) for row in dgcr_rows)
    harm = sum(bool(row.get("harmed_by_debate")) for row in dgcr_rows)
    structured_turns = [row for row in turns if row.get("role") in {"crux_proposer", "reconstruction_panel"}]
    parse_rate = _ratio(sum(row.get("protocol_parse_status") == "ok" for row in structured_turns), len(structured_turns))
    all_request_success = all(not row.get("request_error") for row in turns)
    all_actual_tokens_reported = bool(turns) and all(
        row.get("usage_source") == "reported"
        and row.get("actual_total_tokens") is not None
        and row.get("reasoning_tokens") is not None
        for row in turns
    )
    common = {
        "zero_final_request_failures": all_request_success,
        "all_turns_report_actual_tokens": all_actual_tokens_reported,
        "structured_parse_rate_at_least_99_5_percent": parse_rate >= 0.995,
        "actual_tokens_not_above_adaptive": float(dgcr.get("mean_total_tokens") or 0) <= float(adaptive.get("mean_total_tokens") or 0),
    }
    if phase_name == "development":
        conditions = {
            **common,
            "legal_crux_on_40_percent_of_disagreements": valid_crux_rate >= 0.40,
            "candidate_oracle_at_least_5pp_over_sc5": float(dgcr.get("candidate_oracle_task_harmonic") or 0) - float(sc5.get("task_harmonic_accuracy") or 0) >= 0.05,
            "dgcr_at_least_3pp_over_adaptive": float(dgcr.get("task_harmonic_accuracy") or 0) - float(adaptive.get("task_harmonic_accuracy") or 0) >= 0.03,
            "net_corrections_at_least_three": correction - harm >= 3,
            "override_precision_at_least_65_percent": precision >= 0.65,
        }
    elif phase_name == "heldout":
        conditions = {
            **common,
            "dgcr_at_least_2pp_over_adaptive": float(dgcr.get("task_harmonic_accuracy") or 0) - float(adaptive.get("task_harmonic_accuracy") or 0) >= 0.02,
            "corrected_exceeds_harmed": correction > harm,
            "override_precision_one_sided_95_lower_bound_above_half": _wilson_lower(len(correct_overrides), len(overrides), 1.644854) > 0.5,
        }
    else:
        raise ValueError(f"Unsupported DGCR gate phase {phase_name!r}.")
    return {
        "gate_name": f"dgcr_{phase_name}_v1",
        "passed": all(conditions.values()),
        "conditions": conditions,
        "evidence": {
            "summary": summaries,
            "triggered_count": len(triggered),
            "valid_crux_rate": valid_crux_rate,
            "structured_parse_rate": parse_rate,
            "all_actual_tokens_reported": all_actual_tokens_reported,
            "override_count": len(overrides),
            "correct_override_count": len(correct_overrides),
            "override_precision": precision,
            "override_precision_one_sided_95_lower_bound": _wilson_lower(len(correct_overrides), len(overrides), 1.644854),
            "corrected": correction,
            "harmed": harm,
        },
    }


def _summary(method: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(row.get("score") or 0) for row in rows]
    per_task: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        per_task[str(row.get("task") or "unknown")].append(float(row.get("score") or 0))
    task_accuracies = {task: sum(values) / len(values) for task, values in per_task.items() if values}
    harmonic = _harmonic_mean(task_accuracies.values())
    oracle_per_task: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        oracle_per_task[str(row.get("task") or "unknown")].append(float(bool(row.get("candidate_oracle_correct"))))
    oracle_harmonic = _harmonic_mean(sum(values) / len(values) for values in oracle_per_task.values() if values)
    return {
        "method_name": method,
        "sample_count": len(rows),
        "micro_accuracy": _ratio(sum(scores), len(scores)),
        "task_harmonic_accuracy": harmonic,
        "candidate_oracle_task_harmonic": oracle_harmonic,
        "mean_total_tokens": _ratio(sum(float(row.get("total_tokens_per_question") or 0) for row in rows), len(rows)),
    }


def _harmonic_mean(values) -> float:
    materialized = [float(value) for value in values]
    if not materialized or any(value <= 0 for value in materialized):
        return 0.0
    return len(materialized) / sum(1 / value for value in materialized)


def _ratio(numerator: float, denominator: int) -> float:
    return float(numerator) / denominator if denominator else 0.0


def _wilson_lower(successes: int, total: int, z: float) -> float:
    if total <= 0:
        return 0.0
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = proportion + z**2 / (2 * total)
    margin = z * math.sqrt((proportion * (1 - proportion) + z**2 / (4 * total)) / total)
    return (centre - margin) / denominator
