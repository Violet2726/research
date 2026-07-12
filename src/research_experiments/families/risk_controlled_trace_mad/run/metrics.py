"""RCTA 单 run 指标、诊断与论文CSV。"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


def build_metrics(rows: list[dict[str, Any]], *, dataset_order: list[str], method_order: list[str], bbeh_harmonic: bool) -> dict[str, Any]:
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
        "bbeh_metric": {"primary": "task_harmonic_accuracy" if bbeh_harmonic else "micro_accuracy", "secondary": "micro_accuracy"},
    }


def build_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    methods: dict[str, dict[str, Any]] = {}
    for method in sorted({str(row["method_name"]) for row in rows}):
        items = [row for row in rows if row.get("method_name") == method]
        triggered = [row for row in items if row.get("triggered")]
        overrides = [row for row in items if row.get("override_accepted")]
        methods[method] = {
            "question_count": len(items), "trigger_count": len(triggered), "override_count": len(overrides),
            "corrected": sum(bool(row.get("corrected_by_debate")) for row in items),
            "harmed": sum(bool(row.get("harmed_by_debate")) for row in items),
            "certificate_pass": sum((row.get("certificate") or {}).get("status") == "pass" for row in items),
            "certificate_fail": sum((row.get("certificate") or {}).get("status") == "fail" for row in items),
            "certificate_unsupported": sum((row.get("certificate") or {}).get("status") == "unsupported" for row in items),
            "novel_synthesis": sum(bool(row.get("synthesis_answer")) and not row.get("synthesis_existing_candidate") for row in items),
            "final_answer_scalar_normalized": sum("final_answer_json_scalar_to_text" in (row.get("protocol_normalization_flags") or []) for row in items),
            "reasoning_summary_truncated": sum("reasoning_summary_truncated_to_word_limit" in (row.get("protocol_normalization_flags") or []) for row in items),
        }
    return {"methods": methods}


def write_paper_summary(path: str | Path, metrics: dict[str, Any]) -> None:
    fields = ["dataset", "method_name", "question_count", "accuracy_mean", "micro_accuracy", "total_tokens_mean", "calls_per_question_mean", "corrected_count", "harmed_count"]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in metrics["summary"]:
            writer.writerow({key: row.get(key) for key in fields})


def _summary(dataset: str, method: str, rows: list[dict[str, Any]], *, harmonic: bool) -> dict[str, Any]:
    micro = _mean(float(row.get("score") or 0.0) for row in rows)
    primary = _task_harmonic(rows) if harmonic else micro
    tokens = _mean(float(row.get("total_tokens_per_question") or 0.0) for row in rows)
    return {
        "dataset": dataset, "aggregate_kind": "dataset" if dataset != "overall" else "macro",
        "method_name": method, "question_count": len(rows), "prediction_rows": len(rows),
        "accuracy_mean": primary, "micro_accuracy": micro, "primary_accuracy_metric": "task_harmonic" if harmonic else "micro_accuracy",
        "total_tokens_mean": tokens,
        "prompt_tokens_mean": _mean(float(row.get("prompt_tokens_per_question") or 0.0) for row in rows),
        "completion_tokens_mean": _mean(float(row.get("completion_tokens_per_question") or 0.0) for row in rows),
        "latency_ms_mean": _mean(float(row.get("latency_ms_per_question") or 0.0) for row in rows),
        "calls_per_question_mean": _mean(float(row.get("logical_calls_per_question") or row.get("calls_per_question") or 0.0) for row in rows),
        "network_attempts_mean": _mean(float(row.get("network_attempts_per_question") or 0.0) for row in rows),
        "accuracy_per_1k_tokens": (primary * 1000.0 / tokens) if tokens else 0.0,
        "corrected_count": sum(bool(row.get("corrected_by_debate")) for row in rows),
        "harmed_count": sum(bool(row.get("harmed_by_debate")) for row in rows),
        "changed_answer_rate": _mean(float(bool(row.get("vote_flipped"))) for row in rows),
    }


def _task_harmonic(rows: list[dict[str, Any]]) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("task") or "unknown")].append(float(row.get("score") or 0.0))
    accuracies = [_mean(values) for values in grouped.values()]
    if not accuracies or any(value <= 0 for value in accuracies):
        return 0.0
    return len(accuracies) / sum(1.0 / value for value in accuracies)


def _mean(values) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0
