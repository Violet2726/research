"""BRD-MAD 运行的产物与协议验证。"""

from __future__ import annotations

from collections import Counter
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


def validate_run(run_dir: str | Path, *, family_name: str = "blind_reconstructive_mad") -> dict[str, Any]:
    index = resolve_run_artifact_index(run_dir, family_name=family_name)
    root = index.run_dir
    turns = named_turn_record_paths(root, family_name=family_name)
    diagnostics = named_diagnostic_paths(root, family_name=family_name)
    exports = named_export_paths(root, family_name=family_name)
    gate_path = diagnostics["count100_gate.json"] if family_name == "selective_gsa_mad" else diagnostics["pilot_gate.json"]
    required = [
        index.manifest_path, turns["agent_turns.jsonl"], turns["debate_messages.jsonl"], turns["router_decisions.jsonl"],
        index.prediction_records_path, index.metrics_view_path, diagnostics["brd_diagnostics.json"],
        diagnostics["paired_statistics.json"], diagnostics["output_protocol_diagnostics.json"], gate_path, exports["brd_comparison.json"],
        exports["paper_summary.csv"], index.report_path, index.figure_manifest_path, index.archive_manifest_path,
    ]
    missing = missing_relative_paths(root, required)
    manifest = load_json(index.manifest_path)
    turn_rows = load_jsonl_if_present(turns["agent_turns.jsonl"])
    prediction_rows = load_jsonl_if_present(index.prediction_records_path)
    router_rows = load_jsonl_if_present(turns["router_decisions.jsonl"])
    statuses = summarize_turn_statuses(turn_rows)
    coverage = _method_coverage(manifest, prediction_rows)
    stage = _stage_a_check(manifest, turn_rows)
    safety = _safety_check(prediction_rows)
    router = _router_check(router_rows, prediction_rows)
    rate = validate_rate_limit_check(index.progress_path, turn_rows, manifest=manifest)
    shared = validate_shared_contracts(root)
    passed = not missing and statuses["request_failures"] == 0 and statuses["protocol_failures"] == 0 and coverage["passed"] and stage["passed"] and safety["passed"] and router["passed"] and rate["passed"] and shared["figure_contract"]["passed"] and shared["archive_contract"]["passed"]
    return {
        "run_dir": str(root), "passed": passed, "missing_files": missing,
        "request_failures": statuses["request_failures"], "protocol_failures": statuses["protocol_failures"],
        "prediction_rows": len(prediction_rows), "methods": dict(Counter(str(row.get("method_name")) for row in prediction_rows)),
        "checks": {"method_coverage": coverage, "stage_a": stage, "safety": safety, "router": router, "rate_limit": rate, "figure_contract": shared["figure_contract"], "archive_contract": shared["archive_contract"]},
    }


def _method_coverage(manifest, rows) -> dict[str, Any]:
    expected = set(manifest.get("method_order") or [])
    observed: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        observed.setdefault((str(row.get("dataset")), str(row.get("sample_id"))), set()).add(str(row.get("method_name")))
    missing = [{"dataset": key[0], "sample_id": key[1], "methods": sorted(expected - values)} for key, values in observed.items() if expected - values]
    return {"passed": bool(observed) and not missing, "expected_methods": sorted(expected), "missing_samples": missing[:20]}


def _stage_a_check(manifest, rows) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("method_name") == "brd_stage_a_shared":
            grouped.setdefault((str(row.get("dataset")), str(row.get("sample_id"))), []).append(row)
    expected = int((manifest.get("protocol") or {}).get("stage_a_candidates") or 5)
    invalid = [key for key, values in grouped.items() if len(values) != expected]
    return {"passed": bool(grouped) and not invalid, "expected_per_sample": expected, "invalid_samples": invalid[:20]}


def _safety_check(rows) -> dict[str, Any]:
    brd = [row for row in rows if row.get("method_type") == "brd"]
    invalid = []
    for row in brd:
        promoted = row.get("promoted_answer")
        candidates = set(row.get("candidate_answers") or [])
        if promoted is not None and promoted not in candidates:
            invalid.append({"sample_id": row.get("sample_id"), "method": row.get("method_name"), "reason": "novel_answer_promoted"})
        if row.get("anchor_support") == 4 and row.get("override_accepted") and int(row.get("quorum_required") or 0) != 3:
            invalid.append({"sample_id": row.get("sample_id"), "method": row.get("method_name"), "reason": "4-1_not_three_vote_gate"})
        if not row.get("triggered") and int(row.get("calls_per_question") or 0) != 5:
            invalid.append({"sample_id": row.get("sample_id"), "method": row.get("method_name"), "reason": "unanimous_stage_should_not_review"})
        if (
            row.get("triggered")
            and row.get("method_name") != "conditional_resample_3"
            and row.get("candidate_board_all_candidates_visible") is not True
        ):
            invalid.append({"sample_id": row.get("sample_id"), "method": row.get("method_name"), "reason": "candidate_board_missing_family"})
    return {"passed": bool(brd) and not invalid, "invalid_rows": invalid[:20]}


def _router_check(router_rows, prediction_rows) -> dict[str, Any]:
    expected = {(str(row.get("dataset")), str(row.get("sample_id"))) for row in prediction_rows if row.get("method_type") == "brd"}
    observed = {(str(row.get("dataset")), str(row.get("sample_id"))) for row in router_rows}
    return {"passed": expected == observed, "router_samples": len(observed), "brd_samples": len(expected)}
