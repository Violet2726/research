"""comm_necessary run artifact validation.

Checks split-context design integrity, message packet cap compliance,
rate limit constraints, and HotpotQA prediction file export correctness.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from research_experiments.families.comm_necessary.algorithms import METHOD_ORDER
from research_experiments.family_runtime.artifact_index import (
    named_diagnostic_paths,
    named_export_paths,
    named_turn_record_paths,
    resolve_run_artifact_index,
)
from research_experiments.family_runtime.validation import summarize_turn_statuses, validate_shared_contracts

REQUIRED_FILES = [
    "manifest.json",
    "sample_views.jsonl",
    "stage_a_turns.jsonl",
    "message_packets.jsonl",
    "stage_b_turns.jsonl",
    "final_predictions.jsonl",
    "metrics.json",
    "diagnostics.json",
    "progress.json",
    "report.md",
    "paper_summary.csv",
    "figure_manifest.json",
    "archive_manifest.json",
]


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    """Validate comm_necessary run meets experiment contract."""
    index = resolve_run_artifact_index(run_dir, family_name="comm_necessary")
    root = index.run_dir
    turn_paths = named_turn_record_paths(root, family_name="comm_necessary")
    diagnostic_paths = named_diagnostic_paths(root, family_name="comm_necessary")
    export_paths = named_export_paths(root, family_name="comm_necessary")
    required_paths = [
        index.manifest_path,
        turn_paths["sample_views.jsonl"],
        turn_paths["stage_a_turns.jsonl"],
        turn_paths["message_packets.jsonl"],
        turn_paths["stage_b_turns.jsonl"],
        index.prediction_records_path,
        index.metrics_view_path,
        diagnostic_paths["diagnostics.json"],
        index.progress_path,
        index.report_path,
        export_paths["paper_summary.csv"],
        index.figure_manifest_path,
        index.archive_manifest_path,
    ]
    missing = [path.relative_to(root).as_posix() for path in required_paths if not path.exists()]
    manifest = _load_json(index.manifest_path)
    sample_views = _load_jsonl(turn_paths["sample_views.jsonl"])
    stage_a_rows = _load_jsonl(turn_paths["stage_a_turns.jsonl"])
    packet_rows = _load_jsonl(turn_paths["message_packets.jsonl"])
    stage_b_rows = _load_jsonl(turn_paths["stage_b_turns.jsonl"])
    prediction_rows = _load_jsonl(index.prediction_records_path)
    turn_rows = stage_a_rows + stage_b_rows

    status_summary = summarize_turn_statuses(turn_rows)
    paired_check = _paired_design_check(manifest, prediction_rows)
    context_leak_check = _context_leak_check(sample_views)
    shard_union_check = _shard_union_check(sample_views)
    packet_cap_check = _packet_cap_check(packet_rows)
    hotpot_prediction_check = _hotpot_prediction_files_check(export_paths["hotpot_predictions"], prediction_rows)
    rate_limit_check = _rate_limit_check(turn_rows, manifest)
    shared_contracts = validate_shared_contracts(root)
    figure_contract = shared_contracts["figure_contract"]
    archive_contract = shared_contracts["archive_contract"]

    passed = all(
        [
            not missing,
            status_summary["request_failures"] == 0,
            status_summary["schema_failures"] == 0,
            paired_check["passed"],
            context_leak_check["passed"],
            shard_union_check["passed"],
            packet_cap_check["passed"],
            hotpot_prediction_check["passed"],
            rate_limit_check["passed"],
            figure_contract["passed"],
            archive_contract["passed"],
            bool(prediction_rows),
        ]
    )
    return {
        "run_dir": str(root),
        "passed": passed,
        "missing_files": missing,
        "request_failures": status_summary["request_failures"],
        "schema_failures": status_summary["schema_failures"],
        "checks": {
            "paired_design_check": paired_check,
            "context_leak_check": context_leak_check,
            "shard_union_check": shard_union_check,
            "packet_cap_check": packet_cap_check,
            "hotpot_prediction_files_check": hotpot_prediction_check,
            "rate_limit_check": rate_limit_check,
            "figure_contract": figure_contract,
            "archive_contract": archive_contract,
        },
        "methods": dict(Counter(row.get("method_name") for row in prediction_rows)),
    }


def _paired_design_check(manifest: dict[str, Any], prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    methods = list(manifest.get("methods") or METHOD_ORDER)
    by_sample: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in prediction_rows:
        by_sample[(str(row.get("dataset")), str(row.get("sample_id")))].add(str(row.get("method_name")))
    missing_methods = [
        {"dataset": dataset, "sample_id": sample_id, "missing_methods": sorted(set(methods) - observed)}
        for (dataset, sample_id), observed in sorted(by_sample.items())
        if set(methods) - observed
    ]
    counts = Counter(row.get("method_name") for row in prediction_rows)
    expected = len(by_sample)
    count_mismatches = [
        {"method_name": method, "expected": expected, "observed": counts.get(method, 0)}
        for method in methods
        if counts.get(method, 0) != expected
    ]
    return {
        "passed": expected > 0 and not missing_methods and not count_mismatches,
        "sample_count": expected,
        "missing_methods": missing_methods[:20],
        "count_mismatches": count_mismatches,
    }


def _context_leak_check(sample_views: list[dict[str, Any]]) -> dict[str, Any]:
    violations = [
        {
            "dataset": row.get("dataset"),
            "sample_id": row.get("sample_id"),
            "agent_id": row.get("agent_id"),
            "view_kind": row.get("view_kind"),
        }
        for row in sample_views
        if int(row.get("agent_id") or -1) in {1, 2, 3}
        and (row.get("includes_full_context") or row.get("view_context_hash") == row.get("full_context_hash"))
    ]
    return {"passed": not violations, "violation_count": len(violations), "violations": violations[:20]}


def _shard_union_check(sample_views: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sample_views:
        if int(row.get("agent_id") or -1) in {1, 2, 3}:
            grouped[(str(row.get("dataset")), str(row.get("sample_id")))].append(row)
    violations = []
    for (dataset, sample_id), rows in sorted(grouped.items()):
        required = set()
        covered = set()
        for row in rows:
            required.update(str(item) for item in row.get("required_titles", []) if str(item).strip())
            covered.update(str(item) for item in row.get("coverage_titles", []) if str(item).strip())
        if required and not required.issubset(covered):
            violations.append({"dataset": dataset, "sample_id": sample_id, "required": sorted(required), "covered": sorted(covered)})
    return {"passed": not violations, "violation_count": len(violations), "violations": violations[:20]}


def _packet_cap_check(packet_rows: list[dict[str, Any]]) -> dict[str, Any]:
    violations = [
        {
            "dataset": row.get("dataset"),
            "sample_id": row.get("sample_id"),
            "method_name": row.get("method_name"),
            "agent_id": row.get("agent_id"),
            "approx_packet_tokens": row.get("approx_packet_tokens"),
            "token_cap": row.get("token_cap"),
        }
        for row in packet_rows
        if int(row.get("approx_packet_tokens") or 0) > int(row.get("token_cap") or 0)
    ]
    return {"passed": not violations, "violation_count": len(violations), "violations": violations[:20]}


def _hotpot_prediction_files_check(output_dir: Path, prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    missing = [method for method in METHOD_ORDER if not (output_dir / f"{method}.json").exists()]
    invalid: list[dict[str, Any]] = []
    expected_ids_by_method = {
        method: {str(row["sample_id"]) for row in prediction_rows if row.get("method_name") == method}
        for method in METHOD_ORDER
    }
    for method in METHOD_ORDER:
        path = output_dir / f"{method}.json"
        if not path.exists():
            continue
        payload = _load_json(path)
        answer = payload.get("answer")
        sp = payload.get("sp")
        expected_ids = expected_ids_by_method.get(method, set())
        if not isinstance(answer, dict) or not isinstance(sp, dict) or set(answer) != expected_ids or set(sp) != expected_ids:
            invalid.append({"method_name": method, "expected_count": len(expected_ids)})
    return {"passed": not missing and not invalid, "missing_methods": missing, "invalid_files": invalid}


def _rate_limit_check(turn_rows: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    rpm = manifest.get("requests_per_minute_limit")
    tpm = manifest.get("tokens_per_minute_limit")
    if not rpm and not tpm:
        return {"passed": True, "enabled": False}
    events = []
    for row in turn_rows:
        if row.get("cache_hit"):
            continue
        timestamp = row.get("request_started_at")
        if not timestamp:
            continue
        events.append((_parse_timestamp(str(timestamp)), int(row.get("estimated_request_tokens") or 0), row))
    events.sort(key=lambda item: item[0])
    violations: list[dict[str, Any]] = []
    for index, (timestamp, _, row) in enumerate(events):
        window = [(ts, tokens, item) for ts, tokens, item in events[: index + 1] if (timestamp - ts).total_seconds() < 60.0]
        request_count = len(window)
        token_count = sum(tokens for _, tokens, _ in window)
        if rpm and request_count > int(rpm):
            violations.append(_rate_violation("rpm", request_count, int(rpm), row))
        if tpm and token_count > int(tpm):
            violations.append(_rate_violation("tpm", token_count, int(tpm), row))
    return {
        "passed": not violations,
        "enabled": True,
        "network_event_count": len(events),
        "violation_count": len(violations),
        "violations": violations[:20],
    }


def _rate_violation(kind: str, observed: int, limit: int, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": kind,
        "observed": observed,
        "limit": limit,
        "dataset": row.get("dataset"),
        "sample_id": row.get("sample_id"),
        "method_name": row.get("method_name"),
        "agent_id": row.get("agent_id"),
    }


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _load_json(path: Path) -> dict[str, Any]:
    """Read a UTF-8 JSON file; return empty dict if missing."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSONL file; return empty list if missing."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

