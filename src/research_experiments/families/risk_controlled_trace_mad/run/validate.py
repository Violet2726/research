"""MAD 创新实验的结构、预算、数据切分与安全验证。"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from research_experiments.family_runtime.artifact_index import (
    named_diagnostic_paths,
    named_export_paths,
    named_turn_record_paths,
    resolve_run_artifact_index,
)
from research_experiments.family_runtime.validation import (
    load_json,
    missing_relative_paths,
    summarize_turn_statuses,
    validate_rate_limit_check,
    validate_shared_contracts,
)

FAMILY_NAME = "risk_controlled_trace_mad"
METHOD_LIMITS = {
    "cot_1": 1,
    "sc_3": 3,
    "sc_5": 5,
    "adaptive_sc_8": 8,
    "conditional_resample_3": 8,
    "blind_gsa_1": 6,
    "blind_gsa_quorum_3": 8,
    "hsgsa_unanimous_3": 8,
}


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    index = resolve_run_artifact_index(run_dir, family_name=FAMILY_NAME)
    root = index.run_dir
    turns = named_turn_record_paths(root, family_name=FAMILY_NAME)
    diagnostics = named_diagnostic_paths(root, family_name=FAMILY_NAME)
    exports = named_export_paths(root, family_name=FAMILY_NAME)
    manifest = load_json(index.manifest_path)
    is_hsgsa = str(manifest.get("active_version")) == "v5_hsgsa"
    diagnostic_key = "hsgsa_diagnostics.json" if is_hsgsa else "evf_diagnostics.json"
    comparison_key = "hsgsa_comparison.json" if is_hsgsa else "evf_comparison.json"
    required = [
        index.manifest_path,
        turns["agent_turns.jsonl"],
        turns["debate_messages.jsonl"],
        turns["router_decisions.jsonl"],
        index.prediction_records_path,
        index.metrics_view_path,
        diagnostics[diagnostic_key],
        diagnostics["paired_statistics.json"],
        diagnostics["output_protocol_diagnostics.json"],
        exports[comparison_key],
        exports["paper_summary.csv"],
        index.report_path,
        index.figure_manifest_path,
        index.archive_manifest_path,
    ]
    missing = missing_relative_paths(root, required)
    turn_rows = [_compact_turn(row) for row in _iter_jsonl(turns["agent_turns.jsonl"])]
    decisions = [_compact_decision(row) for row in _iter_jsonl(turns["router_decisions.jsonl"])]
    predictions = [_compact_prediction(row) for row in _iter_jsonl(index.prediction_records_path)]
    statuses = summarize_turn_statuses(turn_rows)
    coverage = _method_coverage(manifest, predictions)
    stage = _hsgsa_stage_check(turn_rows, decisions) if is_hsgsa else {"passed": True, "historical": True}
    safety = _hsgsa_safety_check(predictions) if is_hsgsa else {"passed": True, "historical": True}
    budget = _budget_check(predictions, manifest)
    split = _split_check(manifest)
    network = _network_check(turn_rows, manifest)
    profile = dict((manifest.get("runtime_profiles") or {}).get("primary") or {})
    rate = validate_rate_limit_check(
        index.progress_path,
        turn_rows,
        requests_per_minute_limit=int(profile.get("requests_per_minute_limit") or 0),
    )
    if network["actual"] and not rate["event_replay_available"]:
        rate = {**rate, "passed": False, "reason": "network attempts exist without replay events"}
    protocol = _protocol_health(turn_rows)
    shared = validate_shared_contracts(root)
    passed = (
        not missing
        and statuses["schema_failures"] == 0
        and statuses["request_failures"] == 0
        and coverage["passed"]
        and stage["passed"]
        and safety["passed"]
        and budget["passed"]
        and split["passed"]
        and network["passed"]
        and rate["passed"]
        and protocol["passed"]
        and shared["figure_contract"]["passed"]
        and shared["archive_contract"]["passed"]
    )
    return {
        "run_dir": str(root),
        "passed": passed,
        "missing_files": missing,
        "request_failures": statuses["request_failures"],
        "protocol_failures": statuses["protocol_failures"],
        "schema_failures": statuses["schema_failures"],
        "provider_abstentions": sum(bool(row.get("provider_abstention")) for row in turn_rows),
        "prediction_rows": len(predictions),
        "methods": dict(Counter(str(row.get("method_name")) for row in predictions)),
        "checks": {
            "method_coverage": coverage,
            "shared_physical_stages": stage,
            "safety": safety,
            "logical_budget": budget,
            "split_isolation": split,
            "network_attempt_hard_cap": network,
            "rate_limit": rate,
            "protocol_health": protocol,
            "figure_contract": shared["figure_contract"],
            "archive_contract": shared["archive_contract"],
        },
    }


def _method_coverage(manifest, rows):
    expected = set(manifest.get("method_order") or [])
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        grouped[(str(row.get("dataset")), str(row.get("sample_id")))].add(str(row.get("method_name")))
    invalid = [
        {"dataset": key[0], "sample_id": key[1], "missing": sorted(expected - methods)}
        for key, methods in grouped.items()
        if expected - methods
    ]
    return {"passed": bool(grouped) and not invalid, "invalid_samples": invalid[:20], "expected_methods": sorted(expected)}


def _hsgsa_stage_check(rows, decisions):
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("dataset")), str(row.get("sample_id")), str(row.get("method_name")))].append(row)
    triggered = {
        (str(row.get("dataset")), str(row.get("sample_id")))
        for row in decisions
        if row.get("triggered")
    }
    samples = {(dataset, sample) for dataset, sample, method in grouped if method == "hsgsa_stage_a_shared"}
    invalid = []
    for key in samples:
        stage = grouped.get((*key, "hsgsa_stage_a_shared"), [])
        if sorted(int(row.get("agent_id") or 0) for row in stage) != [1, 2, 3, 4, 5]:
            invalid.append({"dataset": key[0], "sample_id": key[1], "reason": "stage_a_not_exactly_five"})
        if len({str(row.get("model_name")) for row in stage}) != 1 or any(
            row.get("model_lineage") != "mimo" for row in stage
        ):
            invalid.append({"dataset": key[0], "sample_id": key[1], "reason": "non_homogeneous_stage_a"})
        expected_extra = 3 if key in triggered else 0
        for method in ("hsgsa_resample_shared", "hsgsa_blind_reviewer_shared"):
            if len(grouped.get((*key, method), [])) != expected_extra:
                invalid.append({"dataset": key[0], "sample_id": key[1], "reason": f"{method}_count"})
    return {"passed": bool(samples) and not invalid, "sample_count": len(samples), "invalid_samples": invalid[:20]}


_DEBATE_METHODS = {"blind_gsa_1", "blind_gsa_quorum_3", "hsgsa_unanimous_3"}


def _hsgsa_safety_check(rows):
    invalid = []
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(str(row.get("dataset")), str(row.get("sample_id")))][str(row.get("method_name"))] = row
        if row.get("novel_answer") and row.get("method_name") in _DEBATE_METHODS:
            invalid.append({"sample_id": row.get("sample_id"), "reason": "novel_answer_promoted"})
        if (
            row.get("method_name") == "hsgsa_unanimous_3"
            and row.get("override_accepted")
            and row.get("resolver") != "support_blind_3_of_3_override"
        ):
            invalid.append({"sample_id": row.get("sample_id"), "reason": "override_without_three_of_three"})
    for key, methods in grouped.items():
        hsgsa = methods.get("hsgsa_unanimous_3")
        adaptive = methods.get("adaptive_sc_8")
        if hsgsa and adaptive and hsgsa.get("logical_calls_per_question") != adaptive.get("logical_calls_per_question"):
            invalid.append({"dataset": key[0], "sample_id": key[1], "reason": "primary_call_budget_mismatch"})
    return {"passed": not invalid, "invalid_rows": invalid[:20]}


def _budget_check(rows, manifest):
    invalid = []
    protocol_max = int(manifest.get("max_logical_calls_per_question") or 11)
    for row in rows:
        calls = int(row.get("logical_calls_per_question") or 0)
        limit = METHOD_LIMITS.get(str(row.get("method_name")), protocol_max)
        if calls > limit:
            invalid.append({"sample_id": row.get("sample_id"), "method": row.get("method_name"), "calls": calls, "limit": limit})
    return {"passed": not invalid, "invalid_rows": invalid[:20], "limits": METHOD_LIMITS}


def _split_check(manifest):
    datasets = dict((manifest.get("split_audit") or {}).get("datasets") or {})
    invalid = []
    for dataset, payload in datasets.items():
        if int(payload.get("overlap_count") or 0):
            invalid.append(f"{dataset}:development_confirmation_overlap")
        if dataset == "bbeh" and payload.get("excluded_split") and int(payload.get("task_count") or 0) != 23:
            invalid.append("bbeh:missing_confirmation_task")
    return {"passed": bool(datasets) and not invalid, "failures": invalid, "datasets": datasets}


def _network_check(rows, manifest):
    actual = sum(int(row.get("network_attempt_count") or 0) for row in rows)
    limit = int(manifest.get("max_network_attempts") or 50_000)
    return {"passed": actual <= limit, "actual": actual, "limit": limit}


def _protocol_health(rows):
    reviewers = [row for row in rows if row.get("method_name") == "hsgsa_blind_reviewer_shared"]
    failures = sum(row.get("protocol_parse_status") == "failed" for row in reviewers)
    rate = 1.0 - failures / len(reviewers) if reviewers else 1.0
    return {"passed": rate >= 0.995, "reviewer_count": len(reviewers), "failures": failures, "parse_rate": rate}


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _compact_turn(row):
    keys = {
        "dataset", "sample_id", "method_name", "agent_id", "model_name", "model_lineage", "output_status",
        "protocol_parse_status", "provider_abstention", "network_attempt_count", "cache_hit",
        "request_started_at", "request_started_at_events",
    }
    return {key: value for key, value in row.items() if key in keys}


def _compact_decision(row):
    return {key: row.get(key) for key in ("dataset", "sample_id", "triggered")}


def _compact_prediction(row):
    keys = {
        "dataset", "sample_id", "method_name", "logical_calls_per_question", "novel_answer",
        "override_accepted", "resolver",
    }
    return {key: value for key, value in row.items() if key in keys}
