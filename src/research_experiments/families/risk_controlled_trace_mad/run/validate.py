"""EVF-MAD 的完整性、预算、安全和双 provider 限流验证。"""

from __future__ import annotations

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
    load_jsonl_if_present,
    missing_relative_paths,
    summarize_turn_statuses,
    validate_rate_limit_check,
    validate_shared_contracts,
)

FAMILY_NAME = "risk_controlled_trace_mad"
METHOD_LIMITS = {
    "cot_1": 1,
    "qwen_sc_5": 5,
    "qwen_sc_9": 9,
    "mimo_sc_5": 5,
    "mimo_sc_9": 9,
    "heterogeneous_mv_5": 5,
    "heterogeneous_gsa_1": 6,
    "mad_5a_r1": 10,
    "hcp_mad_budget10": 10,
    "minority_sentinel_reproduction": 5,
    "evf_mad_1": 10,
}


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    index = resolve_run_artifact_index(run_dir, family_name=FAMILY_NAME)
    root = index.run_dir
    turns = named_turn_record_paths(root, family_name=FAMILY_NAME)
    diagnostics = named_diagnostic_paths(root, family_name=FAMILY_NAME)
    exports = named_export_paths(root, family_name=FAMILY_NAME)
    required = [
        index.manifest_path,
        turns["agent_turns.jsonl"],
        turns["debate_messages.jsonl"],
        turns["router_decisions.jsonl"],
        index.prediction_records_path,
        index.metrics_view_path,
        diagnostics["evf_diagnostics.json"],
        diagnostics["paired_statistics.json"],
        diagnostics["output_protocol_diagnostics.json"],
        exports["evf_comparison.json"],
        exports["paper_summary.csv"],
        index.report_path,
        index.figure_manifest_path,
        index.archive_manifest_path,
    ]
    missing = missing_relative_paths(root, required)
    manifest = load_json(index.manifest_path)
    turn_rows = load_jsonl_if_present(turns["agent_turns.jsonl"])
    predictions = load_jsonl_if_present(index.prediction_records_path)
    statuses = summarize_turn_statuses(turn_rows)
    abstentions = sum(row.get("output_status") == "abstain" for row in turn_rows)
    coverage = _method_coverage(manifest, predictions)
    stage = _stage_check(turn_rows)
    safety = _safety_check(predictions)
    budget = _budget_check(predictions)
    rates = {}
    for lineage in ("qwen", "mimo"):
        profile = dict((manifest.get("runtime_profiles") or {}).get(lineage) or {})
        lineage_rows = [row for row in turn_rows if row.get("model_lineage") == lineage]
        rates[lineage] = validate_rate_limit_check(
            index.progress_path,
            lineage_rows,
            requests_per_minute_limit=int(profile.get("requests_per_minute_limit") or 0),
        )
        network = sum(int(row.get("network_attempt_count") or 0) for row in lineage_rows)
        if network and not rates[lineage]["event_replay_available"]:
            rates[lineage] = {
                **rates[lineage],
                "passed": False,
                "reason": "network attempts exist without replay events",
            }
    abstention_check = _abstention_check(manifest, turn_rows)
    shared = validate_shared_contracts(root)
    passed = (
        not missing
        and statuses["request_failures"] == 0
        and statuses["protocol_failures"] == 0
        and statuses["schema_failures"] == 0
        and coverage["passed"]
        and stage["passed"]
        and safety["passed"]
        and budget["passed"]
        and all(item["passed"] for item in rates.values())
        and abstention_check["passed"]
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
        "provider_abstentions": abstentions,
        "prediction_rows": len(predictions),
        "methods": dict(Counter(str(row.get("method_name")) for row in predictions)),
        "checks": {
            "method_coverage": coverage,
            "stage_pool": stage,
            "safety": safety,
            "logical_budget": budget,
            "rate_limits": rates,
            "provider_abstention": abstention_check,
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
    return {
        "passed": bool(grouped) and not invalid,
        "invalid_samples": invalid[:20],
        "expected_methods": sorted(expected),
    }


def _stage_check(rows):
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("method_name") == "evf_stage_a_shared":
            grouped[(str(row.get("dataset")), str(row.get("sample_id")), str(row.get("model_lineage")))].append(row)
    invalid = []
    sample_keys = {(dataset, sample) for dataset, sample, _ in grouped}
    for dataset, sample in sample_keys:
        for lineage in ("qwen", "mimo"):
            items = grouped.get((dataset, sample, lineage), [])
            agents = sorted(int(row.get("agent_id") or 0) for row in items)
            if agents != list(range(1, 10)):
                invalid.append({"dataset": dataset, "sample_id": sample, "lineage": lineage, "agent_ids": agents})
    return {
        "passed": bool(sample_keys) and not invalid,
        "expected_per_sample": {"qwen": 9, "mimo": 9},
        "invalid_samples": invalid[:20],
    }


def _safety_check(rows):
    invalid = []
    for row in rows:
        if row.get("method_name") != "evf_mad_1":
            continue
        calls = int(row.get("logical_calls_per_question") or 0)
        if not row.get("triggered") and calls != 5:
            invalid.append({"sample_id": row.get("sample_id"), "reason": "unanimous_must_stop_at_five"})
        if row.get("override_accepted") and row.get("novel_answer"):
            invalid.append({"sample_id": row.get("sample_id"), "reason": "novel_answer_promoted"})
        if row.get("override_accepted") and not row.get("evf_gate_passed"):
            invalid.append({"sample_id": row.get("sample_id"), "reason": "override_without_gate"})
        for evidence in list(row.get("evidence_results") or []):
            if set(evidence) & {"dataset", "task", "model", "model_name", "sample_id", "gold", "confidence"}:
                invalid.append({"sample_id": row.get("sample_id"), "reason": "forbidden_decision_feature"})
    return {"passed": not invalid, "invalid_rows": invalid[:20]}


def _budget_check(rows):
    invalid = []
    for row in rows:
        calls = float(row.get("logical_calls_per_question") or 0)
        limit = METHOD_LIMITS.get(str(row.get("method_name")), 10)
        if calls > limit:
            invalid.append(
                {"sample_id": row.get("sample_id"), "method": row.get("method_name"), "calls": calls, "limit": limit}
            )
    return {"passed": not invalid, "invalid_rows": invalid[:20], "limits": METHOD_LIMITS}


def _abstention_check(manifest, rows):
    total = len(rows)
    abstentions = sum(row.get("output_status") == "abstain" for row in rows)
    rate = abstentions / total if total else 0.0
    phase = str(manifest.get("phase_name") or "")
    limit = float((manifest.get("protocol") or {}).get("provider_abstention_limit") or 0.01)
    passed = phase == "count20_seed42" or rate < limit
    return {
        "passed": passed,
        "abstentions": abstentions,
        "total_turns": total,
        "rate": rate,
        "limit": limit,
        "engineering_phase_exempt": phase == "count20_seed42",
    }
