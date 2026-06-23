"""CRED-MAD 运行产物校验。"""

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


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    index = resolve_run_artifact_index(run_dir, family_name="cred_mad")
    root = index.run_dir
    turn_paths = named_turn_record_paths(root, family_name="cred_mad")
    diagnostic_paths = named_diagnostic_paths(root, family_name="cred_mad")
    export_paths = named_export_paths(root, family_name="cred_mad")
    required_paths = [
        index.manifest_path,
        turn_paths["agent_turns.jsonl"],
        turn_paths["debate_messages.jsonl"],
        turn_paths["router_decisions.jsonl"],
        index.prediction_records_path,
        index.metrics_view_path,
        diagnostic_paths["debate_diagnostics.json"],
        diagnostic_paths["router_eval.json"],
        diagnostic_paths["output_protocol_diagnostics.json"],
        export_paths["cred_comparison.json"],
        export_paths["paper_summary.csv"],
        index.report_path,
        index.figure_manifest_path,
        index.archive_manifest_path,
    ]
    missing = missing_relative_paths(root, required_paths)
    manifest = load_json(index.manifest_path)
    turn_rows = load_jsonl_if_present(turn_paths["agent_turns.jsonl"])
    prediction_rows = load_jsonl_if_present(index.prediction_records_path)
    router_rows = load_jsonl_if_present(turn_paths["router_decisions.jsonl"])
    status_summary = summarize_turn_statuses(turn_rows)
    method_coverage = _method_coverage_check(manifest, prediction_rows)
    router_check = _router_check(manifest, router_rows)
    rate_limit_check = validate_rate_limit_check(index.progress_path, turn_rows, manifest=manifest)
    shared_contracts = validate_shared_contracts(root)
    passed = (
        not missing
        and status_summary["request_failures"] == 0
        and status_summary["protocol_failures"] == 0
        and bool(prediction_rows)
        and method_coverage["passed"]
        and router_check["passed"]
        and rate_limit_check["passed"]
        and shared_contracts["figure_contract"]["passed"]
        and shared_contracts["archive_contract"]["passed"]
    )
    return {
        "run_dir": str(root),
        "passed": passed,
        "missing_files": missing,
        "request_failures": status_summary["request_failures"],
        "protocol_failures": status_summary["protocol_failures"],
        "prediction_rows": len(prediction_rows),
        "methods": dict(Counter(str(row.get("method_name")) for row in prediction_rows)),
        "checks": {
            "method_coverage_check": method_coverage,
            "router_check": router_check,
            "rate_limit_check": rate_limit_check,
            "figure_contract": shared_contracts["figure_contract"],
            "archive_contract": shared_contracts["archive_contract"],
        },
    }


def _method_coverage_check(manifest: dict[str, Any], prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    expected_methods = list(manifest.get("method_order") or [])
    expected = set(expected_methods)
    grouped: dict[tuple[str, str], set[str]] = {}
    for row in prediction_rows:
        grouped.setdefault((str(row.get("dataset")), str(row.get("sample_id"))), set()).add(str(row.get("method_name")))
    missing = [
        {"dataset": dataset, "sample_id": sample_id, "missing_methods": sorted(expected - observed)}
        for (dataset, sample_id), observed in sorted(grouped.items())
        if expected - observed
    ]
    return {
        "passed": bool(grouped) and not missing,
        "sample_count": len(grouped),
        "expected_methods": expected_methods,
        "missing_samples": missing[:20],
    }


def _router_check(manifest: dict[str, Any], router_rows: list[dict[str, Any]]) -> dict[str, Any]:
    cred_methods = list(manifest.get("cred_methods") or [])
    required = "cred_refute_queue_v1_lock" in cred_methods
    return {
        "passed": (not required) or bool(router_rows),
        "required": required,
        "row_count": len(router_rows),
    }
