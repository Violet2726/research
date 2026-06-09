"""comm_necessary 运行产物校验。

检查 split-context 设计完整性、消息包上限遵守情况、
限流约束，以及 HotpotQA 预测文件导出的正确性。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from research_experiments.families.comm_necessary.algorithms import METHOD_ORDER
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
        index.report_path,
        export_paths["paper_summary.csv"],
        index.figure_manifest_path,
        index.archive_manifest_path,
    ]
    missing = missing_relative_paths(root, required_paths)
    manifest = load_json(index.manifest_path)
    sample_views = load_jsonl_if_present(turn_paths["sample_views.jsonl"])
    stage_a_rows = load_jsonl_if_present(turn_paths["stage_a_turns.jsonl"])
    packet_rows = load_jsonl_if_present(turn_paths["message_packets.jsonl"])
    stage_b_rows = load_jsonl_if_present(turn_paths["stage_b_turns.jsonl"])
    prediction_rows = load_jsonl_if_present(index.prediction_records_path)
    turn_rows = stage_a_rows + stage_b_rows

    status_summary = summarize_turn_statuses(turn_rows)
    paired_check = _paired_design_check(manifest, prediction_rows)
    context_leak_check = _context_leak_check(sample_views)
    shard_union_check = _shard_union_check(sample_views)
    packet_cap_check = _packet_cap_check(packet_rows)
    hotpot_prediction_check = _hotpot_prediction_files_check(export_paths["hotpot_predictions"], prediction_rows)
    rate_limit_check = validate_rate_limit_check(index.progress_path, turn_rows, manifest=manifest)
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
        "rate_limit_check": rate_limit_check,
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
        payload = load_json(path)
        answer = payload.get("answer")
        sp = payload.get("sp")
        expected_ids = expected_ids_by_method.get(method, set())
        if not isinstance(answer, dict) or not isinstance(sp, dict) or set(answer) != expected_ids or set(sp) != expected_ids:
            invalid.append({"method_name": method, "expected_count": len(expected_ids)})
    return {"passed": not missing and not invalid, "missing_methods": missing, "invalid_files": invalid}
