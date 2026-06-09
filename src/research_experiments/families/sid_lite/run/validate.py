"""SID-lite 运行产物校验。

校验共享前端、公平比较与机制约束：
early-exit 零通信、共享 Stage A 哈希一致性，
以及 confidence fail-open 行为。
"""

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

REQUIRED_FILES = [
    "manifest.json",
    "stage_a_turns.jsonl",
    "message_packets.jsonl",
    "belief_updates.jsonl",
    "final_predictions.jsonl",
    "metrics.json",
    "diagnostics.json",
    "progress.json",
    "report.md",
    "figure_manifest.json",
    "archive_manifest.json",
]


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    """Validate SID-lite run meets smoke experiment contract."""
    index = resolve_run_artifact_index(run_dir, family_name="sid_lite")
    root = index.run_dir
    turn_paths = named_turn_record_paths(root, family_name="sid_lite")
    diagnostic_paths = named_diagnostic_paths(root, family_name="sid_lite")
    export_paths = named_export_paths(root, family_name="sid_lite")
    required_paths = [
        index.manifest_path,
        turn_paths["stage_a_turns.jsonl"],
        turn_paths["message_packets.jsonl"],
        turn_paths["belief_updates.jsonl"],
        index.prediction_records_path,
        index.metrics_view_path,
        diagnostic_paths["diagnostics.json"],
        index.report_path,
        export_paths["paper_summary.csv"],
        index.figure_manifest_path,
        index.archive_manifest_path,
    ]
    missing = missing_relative_paths(root, required_paths)
    manifest = load_json(index.manifest_path)
    stage_a_rows = load_jsonl_if_present(turn_paths["stage_a_turns.jsonl"])
    packet_rows = load_jsonl_if_present(turn_paths["message_packets.jsonl"])
    belief_rows = load_jsonl_if_present(turn_paths["belief_updates.jsonl"])
    prediction_rows = load_jsonl_if_present(index.prediction_records_path)

    status_summary = summarize_turn_statuses(stage_a_rows + belief_rows)
    manifest = load_json(index.manifest_path)
    paired_check = _paired_design_check(manifest, prediction_rows)
    stage_hash_check = _shared_stage_hash_check(prediction_rows)
    early_exit_check = _early_exit_zero_comm_check(prediction_rows)
    packet_cap_check = _packet_cap_check(packet_rows)
    fail_open_check = _invalid_confidence_fail_open_check(prediction_rows)
    rate_limit_check = validate_rate_limit_check(index.progress_path, stage_a_rows + belief_rows, manifest=manifest)
    shared_contracts = validate_shared_contracts(root)
    figure_contract = shared_contracts["figure_contract"]
    archive_contract = shared_contracts["archive_contract"]
    passed = (
        not missing
        and status_summary["request_failures"] == 0
        and status_summary["schema_failures"] == 0
        and paired_check["passed"]
        and stage_hash_check["passed"]
        and early_exit_check["passed"]
        and packet_cap_check["passed"]
        and fail_open_check["passed"]
        and rate_limit_check["passed"]
        and figure_contract["passed"]
        and archive_contract["passed"]
    )
    return {
        "run_dir": str(root),
        "passed": passed,
        "missing_files": missing,
        "request_failures": status_summary["request_failures"],
        "schema_failures": status_summary["schema_failures"],
        "checks": {
            "paired_design_check": paired_check,
            "shared_stage_a_hash_check": stage_hash_check,
            "early_exit_zero_comm_check": early_exit_check,
            "packet_cap_check": packet_cap_check,
            "invalid_confidence_fail_open_check": fail_open_check,
            "rate_limit_check": rate_limit_check,
            "figure_contract": figure_contract,
            "archive_contract": archive_contract,
        },
        "rate_limit_check": rate_limit_check,
        "methods": dict(Counter(row.get("method_name") for row in prediction_rows)),
    }


def _paired_design_check(manifest: dict[str, Any], prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    methods = list(manifest.get("methods") or ["mv_3", "always_full", "compression_only", "sid_lite"])
    by_sample: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in prediction_rows:
        by_sample[(str(row.get("dataset")), str(row.get("sample_id")))].add(str(row.get("method_name")))
    missing = [
        {"dataset": dataset, "sample_id": sample_id, "missing_methods": sorted(set(methods) - observed)}
        for (dataset, sample_id), observed in sorted(by_sample.items())
        if set(methods) - observed
    ]
    counts = Counter(row.get("method_name") for row in prediction_rows)
    expected_count = len(by_sample)
    count_mismatches = [
        {"method_name": method, "expected": expected_count, "observed": counts.get(method, 0)}
        for method in methods
        if counts.get(method, 0) != expected_count
    ]
    return {
        "passed": not missing and not count_mismatches and expected_count > 0,
        "sample_count": expected_count,
        "missing_methods": missing[:20],
        "count_mismatches": count_mismatches,
    }


def _shared_stage_hash_check(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    mismatches = []
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in prediction_rows:
        grouped[(str(row.get("dataset")), str(row.get("sample_id")))].add(str(row.get("stage_a_trace_hash")))
    for (dataset, sample_id), hashes in sorted(grouped.items()):
        hashes.discard("")
        hashes.discard("None")
        if len(hashes) != 1:
            mismatches.append({"dataset": dataset, "sample_id": sample_id, "hashes": sorted(hashes)})
    return {"passed": not mismatches, "mismatch_count": len(mismatches), "mismatches": mismatches[:20]}


def _early_exit_zero_comm_check(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    violations = [
        row
        for row in prediction_rows
        if row.get("method_name") == "sid_lite"
        and row.get("early_exit")
        and float(row.get("communication_tokens_per_question") or 0.0) != 0.0
    ]
    return {"passed": not violations, "violation_count": len(violations), "violations": _compact_rows(violations)}


def _packet_cap_check(packet_rows: list[dict[str, Any]]) -> dict[str, Any]:
    violations = [
        row
        for row in packet_rows
        if int(row.get("approx_packet_tokens") or 0) > int(row.get("token_cap") or 0)
    ]
    return {"passed": not violations, "violation_count": len(violations), "violations": _compact_rows(violations)}


def _invalid_confidence_fail_open_check(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    violations = [
        row
        for row in prediction_rows
        if row.get("method_name") == "sid_lite"
        and row.get("trigger_reason") == "invalid_confidence_fail_open"
        and row.get("early_exit")
    ]
    return {"passed": not violations, "violation_count": len(violations), "violations": _compact_rows(violations)}


def _compact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "dataset": row.get("dataset"),
            "sample_id": row.get("sample_id"),
            "method_name": row.get("method_name"),
            "agent_id": row.get("agent_id"),
        }
        for row in rows[:20]
    ]
