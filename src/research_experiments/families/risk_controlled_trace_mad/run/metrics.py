"""EVF-MAD 指标、风险分解和分阶段晋级门。"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scipy.stats import beta


def build_metrics(
    rows: list[dict[str, Any]], *, dataset_order: list[str], method_order: list[str], bbeh_harmonic: bool
) -> dict[str, Any]:
    summaries: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), str(row["method_name"]))].append(row)
    for dataset in dataset_order:
        for method in method_order:
            items = grouped.get((dataset, method), [])
            if items:
                summaries.append(_summary(dataset, method, items, harmonic=bbeh_harmonic and dataset == "bbeh"))
    for method in method_order:
        items = [row for row in rows if row.get("method_name") == method]
        if items:
            summaries.append(_summary("overall", method, items, harmonic=False))
    return {
        "summary": summaries,
        "dataset_order": dataset_order,
        "method_order": method_order,
        "bbeh_metric": {
            "primary": "task_harmonic_accuracy" if bbeh_harmonic else "micro_accuracy",
            "secondary": "micro_accuracy",
        },
    }


def build_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    methods: dict[str, dict[str, Any]] = {}
    for method in sorted({str(row["method_name"]) for row in rows}):
        items = [row for row in rows if row.get("method_name") == method]
        evidence = [entry for row in items for entry in list(row.get("evidence_results") or [])]
        methods[method] = {
            "question_count": len(items),
            "trigger_count": sum(bool(row.get("triggered")) for row in items),
            "override_count": sum(bool(row.get("override_accepted")) for row in items),
            "corrected": sum(bool(row.get("corrected_by_debate")) for row in items),
            "harmed": sum(bool(row.get("harmed_by_debate")) for row in items),
            "wrong_to_wrong": sum(
                bool(row.get("vote_flipped")) and not row.get("corrected_by_debate") and not row.get("harmed_by_debate")
                for row in items
            ),
            "provider_abstentions": sum(int(row.get("provider_abstentions_per_question") or 0) for row in items),
            "protocol_failures": sum(int(row.get("protocol_failures_per_question") or 0) for row in items),
            "evidence_status": dict(Counter(str(entry.get("status")) for entry in evidence)),
            "evidence_types": dict(Counter(str(entry.get("test_type")) for entry in evidence)),
            "candidate_oracle_count": sum(bool(row.get("candidate_oracle_correct")) for row in items),
            "novel_promotions": sum(
                bool(row.get("override_accepted")) and bool(row.get("novel_answer")) for row in items
            ),
        }
    return {"methods": methods}


def evaluate_gate(
    phase_name: str, rows: list[dict[str, Any]], paired: dict[str, Any], abstention_limit: float
) -> dict[str, Any]:
    if any(row.get("method_name") == "hsgsa_unanimous_3" for row in rows):
        return _evaluate_hsgsa_gate(phase_name, rows, paired, abstention_limit)
    if phase_name == "count20_seed42":
        return {"phase": phase_name, "evaluated": False, "passed": None, "reason": "engineering_only"}
    evf = [row for row in rows if row.get("method_name") == "evf_mad_1"]
    if not evf:
        return {"phase": phase_name, "evaluated": True, "passed": False, "failures": ["missing_evf_rows"]}
    failures: list[str] = []
    protocol_failures = sum(int(row.get("protocol_failures_per_question") or 0) for row in evf)
    request_failures = sum(int(row.get("request_failures_per_question") or 0) for row in evf)
    abstentions = sum(int(row.get("provider_abstentions_per_question") or 0) for row in evf)
    logical_calls = sum(int(row.get("logical_calls_per_question") or 0) for row in evf)
    abstention_rate = abstentions / logical_calls if logical_calls else 0.0
    if protocol_failures:
        failures.append("nonzero_protocol_failures")
    if request_failures:
        failures.append("nonzero_request_failures")
    if abstention_rate >= abstention_limit:
        failures.append("provider_abstention_limit")
    coverage = [row for row in evf if row.get("override_accepted")]
    corrected = sum(bool(row.get("corrected_by_debate")) for row in coverage)
    harmed = sum(bool(row.get("harmed_by_debate")) for row in coverage)
    precision = corrected / (corrected + harmed) if corrected + harmed else 0.0
    dataset_checks = {}
    for dataset in ("omni_math_2_filtered", "bbeh"):
        d_evf = [row for row in evf if row.get("dataset") == dataset]
        anchor_accuracy = _mean(float(row.get("initial_vote_score") or 0) for row in d_evf)
        oracle = _mean(float(bool(row.get("candidate_oracle_correct"))) for row in d_evf)
        gap = oracle - anchor_accuracy
        d_corrected = sum(bool(row.get("corrected_by_debate")) for row in d_evf)
        d_harmed = sum(bool(row.get("harmed_by_debate")) for row in d_evf)
        scores = {
            method: _mean(
                float(row.get("score") or 0)
                for row in rows
                if row.get("dataset") == dataset and row.get("method_name") == method
            )
            for method in ("evf_mad_1", "heterogeneous_mv_5", "qwen_sc_5", "qwen_sc_9")
        }
        dataset_checks[dataset] = {
            "candidate_oracle_gap": gap,
            "corrected": d_corrected,
            "harmed": d_harmed,
            "accuracy": scores,
        }
        if gap < 0.03:
            failures.append(f"{dataset}:candidate_oracle_gap")
        if d_corrected <= d_harmed:
            failures.append(f"{dataset}:nonpositive_net_correction")
        if any(
            scores["evf_mad_1"] < scores[competitor] for competitor in ("heterogeneous_mv_5", "qwen_sc_5", "qwen_sc_9")
        ):
            failures.append(f"{dataset}:accuracy_gate")
    minimum_coverage = 10 if phase_name == "count100_seed42" else 30
    if len(coverage) < minimum_coverage:
        failures.append("minimum_override_count")
    if corrected <= harmed or precision < 2 / 3:
        failures.append("coverage_precision")
    risk_upper = _clopper_pearson_upper(harmed, corrected + harmed, 0.05)
    if phase_name in {"count300_seed42", "full_seed42"}:
        if risk_upper > 1 / 3:
            failures.append("harm_risk_upper_bound")
        required = {"qwen_sc_9", "heterogeneous_mv_5", "hcp_mad_budget10", "minority_sentinel_reproduction"}
        tests = [
            test
            for test in paired.get("tests", [])
            if test.get("comparison_method") in required and test.get("dataset") in {"omni_math_2_filtered", "bbeh"}
        ]
        if len(tests) != 2 * len(required) or any(float(test["bootstrap_ci_95"][0]) <= 0 for test in tests):
            failures.append("paired_ci_gate")
        summaries = {
            method: _mean(
                float(row.get("total_tokens_per_question") or 0) for row in rows if row.get("method_name") == method
            )
            for method in required | {"evf_mad_1"}
        }
        accuracy = {
            method: _mean(float(row.get("score") or 0) for row in rows if row.get("method_name") == method)
            for method in required
        }
        strongest = max(accuracy, key=accuracy.get) if accuracy else ""
        if strongest and summaries["evf_mad_1"] > summaries[strongest]:
            failures.append("token_gate")
    return {
        "phase": phase_name,
        "evaluated": True,
        "passed": not failures,
        "failures": sorted(set(failures)),
        "coverage": len(coverage),
        "corrected": corrected,
        "harmed": harmed,
        "precision": precision,
        "harm_fraction_upper_95": risk_upper,
        "provider_abstention_rate": abstention_rate,
        "protocol_failures": protocol_failures,
        "request_failures": request_failures,
        "dataset_checks": dataset_checks,
    }


def _evaluate_hsgsa_gate(phase_name, rows, paired, abstention_limit):
    hsgsa = [row for row in rows if row.get("method_name") == "hsgsa_unanimous_3"]
    adaptive = [row for row in rows if row.get("method_name") == "adaptive_sc_8"]
    failures: list[str] = []
    if not hsgsa or not adaptive:
        return {
            "phase": phase_name,
            "evaluated": True,
            "passed": False,
            "failures": ["missing_primary_comparison_rows"],
        }
    reviewer_calls = sum(int(row.get("reviewer_calls_per_question") or 0) for row in hsgsa)
    reviewer_protocol_failures = sum(
        int(row.get("reviewer_protocol_failures_per_question") or 0) for row in hsgsa
    )
    reviewer_parse_rate = 1.0 - reviewer_protocol_failures / reviewer_calls if reviewer_calls else 1.0
    request_failures = sum(
        int(
            row.get("shared_physical_request_failures_per_question")
            if row.get("shared_physical_request_failures_per_question") is not None
            else row.get("request_failures_per_question")
            or 0
        )
        for row in hsgsa
    )
    abstentions = sum(int(row.get("provider_abstentions_per_question") or 0) for row in hsgsa)
    logical_calls = sum(int(row.get("logical_calls_per_question") or 0) for row in hsgsa)
    abstention_rate = abstentions / logical_calls if logical_calls else 0.0
    if reviewer_parse_rate < 0.995:
        failures.append("reviewer_parse_rate_below_99_5_percent")
    if request_failures:
        failures.append("nonzero_request_failures")
    if abstention_rate >= abstention_limit:
        failures.append("provider_abstention_limit")

    bbeh_hsgsa = [row for row in hsgsa if row.get("dataset") == "bbeh"]
    coverage = [row for row in bbeh_hsgsa if row.get("override_accepted")]
    corrected = sum(bool(row.get("corrected_by_debate")) for row in coverage)
    harmed = sum(bool(row.get("harmed_by_debate")) for row in coverage)
    precision = corrected / (corrected + harmed) if corrected + harmed else 0.0
    precision_lower = _clopper_pearson_lower(corrected, corrected + harmed, 0.05)
    if corrected <= harmed:
        failures.append("nonpositive_net_correction")
    if precision_lower <= 0.5:
        failures.append("coverage_precision_lower_bound")

    primary = next(
        (
            item
            for item in paired.get("tests", [])
            if item.get("dataset") == "bbeh" and item.get("comparison_method") == "adaptive_sc_8"
        ),
        None,
    )
    if primary is None:
        failures.append("missing_bbeh_hsgsa_vs_adaptive_sc8_test")
        point, interval, adjusted_p = 0.0, [0.0, 0.0], 1.0
    else:
        point = float(primary.get("mean_accuracy_delta") or 0.0)
        interval = list(primary.get("bootstrap_ci_95") or [0.0, 0.0])
        adjusted_p = float(primary.get("holm_adjusted_p") or 1.0)
        if point < 0.01:
            failures.append("bbeh_harmonic_delta_below_1pp")
        if float(interval[0]) <= 0.0:
            failures.append("bbeh_stratified_ci_not_positive")
        if adjusted_p >= 0.05:
            failures.append("bbeh_mcnemar_holm_not_significant")

    h_calls = _mean(float(row.get("logical_calls_per_question") or 0) for row in hsgsa)
    a_calls = _mean(float(row.get("logical_calls_per_question") or 0) for row in adaptive)
    h_tokens = _mean(float(row.get("total_tokens_per_question") or 0) for row in hsgsa)
    a_tokens = _mean(float(row.get("total_tokens_per_question") or 0) for row in adaptive)
    if abs(h_calls - a_calls) > 1e-12:
        failures.append("logical_call_budget_mismatch")
    if h_tokens > a_tokens + 1e-9:
        failures.append("token_budget_dominated_by_adaptive_sc8")

    return {
        "phase": phase_name,
        "evaluated": True,
        "passed": not failures,
        "failures": sorted(set(failures)),
        "primary_comparison": "hsgsa_unanimous_3_vs_adaptive_sc_8",
        "bbeh_harmonic_delta": point,
        "bbeh_stratified_bootstrap_ci_95": interval,
        "bbeh_mcnemar_holm_adjusted_p": adjusted_p,
        "mean_logical_calls": {"hsgsa_unanimous_3": h_calls, "adaptive_sc_8": a_calls},
        "mean_total_tokens": {"hsgsa_unanimous_3": h_tokens, "adaptive_sc_8": a_tokens},
        "coverage": len(coverage),
        "corrected": corrected,
        "harmed": harmed,
        "coverage_precision": precision,
        "coverage_precision_lower_one_sided_95": precision_lower,
        "reviewer_parse_rate": reviewer_parse_rate,
        "reviewer_protocol_failures": reviewer_protocol_failures,
        "request_failures": request_failures,
        "provider_abstention_rate": abstention_rate,
    }


def write_paper_summary(path: str | Path, metrics: dict[str, Any]) -> None:
    fields = [
        "dataset",
        "method_name",
        "question_count",
        "accuracy_mean",
        "micro_accuracy",
        "total_tokens_mean",
        "calls_per_question_mean",
        "corrected_count",
        "harmed_count",
        "provider_abstention_rate",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in metrics["summary"]:
            writer.writerow({key: row.get(key) for key in fields})


def _summary(dataset: str, method: str, rows: list[dict[str, Any]], *, harmonic: bool) -> dict[str, Any]:
    micro = _mean(float(row.get("score") or 0) for row in rows)
    primary = _task_harmonic(rows) if harmonic else micro
    tokens = _mean(float(row.get("total_tokens_per_question") or 0) for row in rows)
    calls = _mean(float(row.get("logical_calls_per_question") or 0) for row in rows)
    abstentions = sum(int(row.get("provider_abstentions_per_question") or 0) for row in rows)
    return {
        "dataset": dataset,
        "aggregate_kind": "dataset" if dataset != "overall" else "macro",
        "method_name": method,
        "question_count": len(rows),
        "prediction_rows": len(rows),
        "accuracy_mean": primary,
        "micro_accuracy": micro,
        "primary_accuracy_metric": "task_harmonic" if harmonic else "micro_accuracy",
        "total_tokens_mean": tokens,
        "prompt_tokens_mean": _mean(float(row.get("prompt_tokens_per_question") or 0) for row in rows),
        "completion_tokens_mean": _mean(float(row.get("completion_tokens_per_question") or 0) for row in rows),
        "latency_ms_mean": _mean(float(row.get("latency_ms_per_question") or 0) for row in rows),
        "calls_per_question_mean": calls,
        "network_attempts_mean": _mean(float(row.get("network_attempts_per_question") or 0) for row in rows),
        "accuracy_per_1k_tokens": primary * 1000 / tokens if tokens else 0.0,
        "corrected_count": sum(bool(row.get("corrected_by_debate")) for row in rows),
        "harmed_count": sum(bool(row.get("harmed_by_debate")) for row in rows),
        "changed_answer_rate": _mean(float(bool(row.get("vote_flipped"))) for row in rows),
        "provider_abstention_rate": abstentions / (calls * len(rows)) if calls else 0.0,
    }


def _task_harmonic(rows: list[dict[str, Any]]) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("task") or "unknown")].append(float(row.get("score") or 0))
    scores = [_mean(values) for values in grouped.values()]
    if not scores or any(value <= 0 for value in scores):
        return 0.0
    return len(scores) / sum(1 / value for value in scores)


def _clopper_pearson_upper(harmed: int, total: int, alpha: float) -> float:
    if total <= 0:
        return 1.0
    if harmed >= total:
        return 1.0
    return float(beta.ppf(1 - alpha, harmed + 1, total - harmed))


def _clopper_pearson_lower(successes: int, total: int, alpha: float) -> float:
    if total <= 0 or successes <= 0:
        return 0.0
    if successes >= total:
        return float(alpha ** (1 / total))
    return float(beta.ppf(alpha, successes, total - successes + 1))


def _mean(values) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0
