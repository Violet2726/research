"""RCTA 完整运行、预算、安全与RPM事件回放验证。"""

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


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    index = resolve_run_artifact_index(run_dir, family_name="risk_controlled_trace_mad")
    root = index.run_dir
    turns = named_turn_record_paths(root, family_name="risk_controlled_trace_mad")
    diagnostics = named_diagnostic_paths(root, family_name="risk_controlled_trace_mad")
    exports = named_export_paths(root, family_name="risk_controlled_trace_mad")
    required = [index.manifest_path, turns["agent_turns.jsonl"], turns["debate_messages.jsonl"], turns["router_decisions.jsonl"], index.prediction_records_path, index.metrics_view_path, diagnostics["rcta_diagnostics.json"], diagnostics["paired_statistics.json"], diagnostics["output_protocol_diagnostics.json"], exports["rcta_comparison.json"], exports["paper_summary.csv"], index.report_path, index.figure_manifest_path, index.archive_manifest_path]
    missing = missing_relative_paths(root, required)
    manifest = load_json(index.manifest_path)
    turn_rows = load_jsonl_if_present(turns["agent_turns.jsonl"])
    prediction_rows = load_jsonl_if_present(index.prediction_records_path)
    statuses = summarize_turn_statuses(turn_rows)
    coverage = _method_coverage(manifest, prediction_rows)
    stage = _stage_check(turn_rows)
    safety = _safety_check(prediction_rows)
    budget = _budget_check(prediction_rows)
    rate = validate_rate_limit_check(index.progress_path, turn_rows, manifest=manifest)
    network_attempts = sum(int(row.get("network_attempt_count") or 0) for row in turn_rows)
    if network_attempts > 0 and not rate["event_replay_available"]:
        rate = {**rate, "passed": False, "reason": "network attempts exist but request-start replay is unavailable"}
    full_router = {"passed": manifest.get("phase_name") != "full_seed42" or bool(manifest.get("router_artifact_sha256")), "router_artifact_sha256": manifest.get("router_artifact_sha256")}
    shared = validate_shared_contracts(root)
    passed = not missing and statuses["request_failures"] == 0 and statuses["protocol_failures"] == 0 and statuses["schema_failures"] == 0 and coverage["passed"] and stage["passed"] and safety["passed"] and budget["passed"] and rate["passed"] and full_router["passed"] and shared["figure_contract"]["passed"] and shared["archive_contract"]["passed"]
    return {
        "run_dir": str(root), "passed": passed, "missing_files": missing,
        "request_failures": statuses["request_failures"], "protocol_failures": statuses["protocol_failures"], "schema_failures": statuses["schema_failures"],
        "prediction_rows": len(prediction_rows), "methods": dict(Counter(str(row.get("method_name")) for row in prediction_rows)),
        "checks": {"method_coverage": coverage, "stage_pool": stage, "safety": safety, "logical_budget": budget, "rate_limit": rate, "full_router": full_router, "figure_contract": shared["figure_contract"], "archive_contract": shared["archive_contract"]},
    }


def _method_coverage(manifest, rows):
    expected = set(manifest.get("method_order") or [])
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        grouped[(str(row.get("dataset")), str(row.get("sample_id")))].add(str(row.get("method_name")))
    invalid = [{"dataset": key[0], "sample_id": key[1], "missing": sorted(expected - methods)} for key, methods in grouped.items() if expected - methods]
    return {"passed": bool(grouped) and not invalid, "invalid_samples": invalid[:20], "expected_methods": sorted(expected)}


def _stage_check(rows):
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("method_name") == "rcta_stage_a_shared":
            grouped[(str(row.get("dataset")), str(row.get("sample_id")))].append(row)
    invalid = []
    for key, items in grouped.items():
        agents = sorted(int(row.get("agent_id") or 0) for row in items)
        seeds = sorted(int((row.get("payload") or {}).get("seed")) for row in items if (row.get("payload") or {}).get("seed") is not None)
        if agents != list(range(1, 10)) or len(seeds) != 9:
            invalid.append({"dataset": key[0], "sample_id": key[1], "agent_ids": agents, "seeds": seeds})
    return {"passed": bool(grouped) and not invalid, "expected_per_sample": 9, "invalid_samples": invalid[:20]}


def _safety_check(rows):
    invalid = []
    for row in rows:
        if row.get("method_type") != "rcta":
            continue
        calls = int(row.get("logical_calls_per_question") or row.get("calls_per_question") or 0)
        if not row.get("triggered") and calls != 5:
            invalid.append({"sample_id": row.get("sample_id"), "method": row.get("method_name"), "reason": "unanimous_must_not_synthesize"})
        if row.get("triggered") and row.get("method_name") in {"gsa_trace_1", "rcta_1", "rcta_no_certificate", "rcta_existing_only", "rcta_certificate_shadow_1"} and row.get("candidate_board_all_candidates_visible") is not True:
            invalid.append({"sample_id": row.get("sample_id"), "method": row.get("method_name"), "reason": "trace_board_missing"})
        vector = row.get("feature_vector")
        if isinstance(vector, dict) and set(vector) & {"dataset", "task", "model", "model_name", "sample_id", "gold", "question", "confidence"}:
            invalid.append({"sample_id": row.get("sample_id"), "method": row.get("method_name"), "reason": "forbidden_router_feature"})
        if row.get("method_name") == "rcta_existing_only" and row.get("override_accepted") and not row.get("synthesis_existing_candidate"):
            invalid.append({"sample_id": row.get("sample_id"), "method": row.get("method_name"), "reason": "novel_existing_only_promotion"})
    return {"passed": not invalid, "invalid_rows": invalid[:20]}


def _budget_check(rows):
    invalid = []
    for row in rows:
        calls = float(row.get("logical_calls_per_question") or row.get("calls_per_question") or 0)
        limit = 6 if row.get("method_name") in {"gsa_trace_1", "rcta_1", "rcta_no_certificate", "rcta_existing_only", "rcta_certificate_shadow_1"} else 10
        if calls > limit:
            invalid.append(
                {
                    "sample_id": row.get("sample_id"),
                    "method": row.get("method_name"),
                    "calls": calls,
                    "limit": limit,
                }
            )
    return {"passed": not invalid, "invalid_rows": invalid[:20], "primary_max_calls": 10, "rcta_max_calls": 6}
