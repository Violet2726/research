"""A-SMAD run validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from research_experiments.families.adaptive_sparse_mad.config import ADAPTIVE_POLICY_METHODS
from research_experiments.family_runtime.artifact_index import (
    named_diagnostic_paths,
    named_turn_record_paths,
    resolve_run_artifact_index,
)
from research_experiments.family_runtime.validation import (
    load_json,
    load_jsonl,
    missing_relative_paths,
    summarize_turn_statuses,
    validate_rate_limit_check,
    validate_shared_contracts,
)

_LEGACY_ADAPTIVE_POLICY_METHODS = frozenset(
    {
        "dge_only_v4",
        "dge_ega_v4",
        "always_add_v4",
        "adaptive_intersection_v8",
        "adaptive_evidence_sc_v10",
    }
)


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    index = resolve_run_artifact_index(run_dir, family_name="adaptive_sparse_mad")
    root = index.run_dir
    turn_paths = named_turn_record_paths(root, family_name="adaptive_sparse_mad")
    diagnostic_paths = named_diagnostic_paths(root, family_name="adaptive_sparse_mad")
    required_paths = [
        index.manifest_path,
        turn_paths["stage_a_turns.jsonl"],
        turn_paths["stage_b_turns.jsonl"],
        turn_paths["judge_turns.jsonl"],
        turn_paths["control_turns.jsonl"],
        turn_paths["router_decisions.jsonl"],
        index.prediction_records_path,
        index.metrics_view_path,
        _diagnostic_path(diagnostic_paths, root, "router_eval.json"),
        _diagnostic_path(diagnostic_paths, root, "policy_diagnostics.json"),
        _diagnostic_path(diagnostic_paths, root, "stage_a_resolver_breakdown.json"),
        _diagnostic_path(diagnostic_paths, root, "stage_a_error_buckets.json"),
        _diagnostic_path(diagnostic_paths, root, "stage_a_solver_contributions.json"),
        index.report_path,
        index.figure_manifest_path,
        index.archive_manifest_path,
    ]
    missing = missing_relative_paths(root, required_paths)

    stage_a_rows = load_jsonl(turn_paths["stage_a_turns.jsonl"])
    stage_b_rows = load_jsonl(turn_paths["stage_b_turns.jsonl"])
    judge_rows = load_jsonl(turn_paths["judge_turns.jsonl"])
    control_rows = load_jsonl(turn_paths["control_turns.jsonl"])
    router_rows = load_jsonl(turn_paths["router_decisions.jsonl"])
    prediction_rows = load_jsonl(index.prediction_records_path)
    manifest = load_json(index.manifest_path)

    all_turn_rows = stage_a_rows + stage_b_rows + judge_rows + control_rows
    status_summary = summarize_turn_statuses(all_turn_rows)
    router_empty_check = _validate_router_rows(router_rows, manifest=manifest, prediction_rows=prediction_rows)
    stage_b_judge_empty_check = _validate_empty_legacy_rows(
        "stage_b_and_judge",
        stage_b_rows + judge_rows,
    )
    rate_limit_check = validate_rate_limit_check(index.progress_path, all_turn_rows, manifest=manifest)
    shared_contracts = validate_shared_contracts(root)
    figure_contract = shared_contracts["figure_contract"]
    archive_contract = shared_contracts["archive_contract"]

    passed = all(
        [
            not missing,
            status_summary["request_failures"] == 0,
            status_summary["schema_failures"] == 0,
            status_summary["output_success_rate"] >= 0.95,
            router_empty_check["passed"],
            stage_b_judge_empty_check["passed"],
            rate_limit_check["passed"],
            figure_contract["passed"],
            archive_contract["passed"],
        ]
    )

    return {
        "run_dir": str(root),
        "passed": passed,
        "missing_files": missing,
        "request_failures": status_summary["request_failures"],
        "schema_failures": status_summary["schema_failures"],
        "output_success_rate": status_summary["output_success_rate"],
        "checks": {
            "router_empty_check": router_empty_check,
            "stage_b_judge_empty_check": stage_b_judge_empty_check,
            "rate_limit_check": rate_limit_check,
            "figure_contract": figure_contract,
            "archive_contract": archive_contract,
        },
        "rate_limit_check": rate_limit_check,
        "method_counts": _count_by_method(prediction_rows),
    }


def _validate_empty_legacy_rows(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "passed": len(rows) == 0,
        "name": name,
        "row_count": len(rows),
        "examples": rows[:20],
    }


def _validate_router_rows(
    rows: list[dict[str, Any]],
    *,
    manifest: dict[str, Any],
    prediction_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    aggregate_methods = {
        str(method_name)
        for method_name in manifest.get("aggregate_methods", [])
        if str(method_name).strip()
    }
    if not aggregate_methods:
        aggregate_methods = {
            str(row.get("method_name") or "")
            for row in prediction_rows
            if str(row.get("method_kind") or "") == "aggregate"
        }
    adaptive_policies = {
        method_name
        for method_name in aggregate_methods
        if method_name in (ADAPTIVE_POLICY_METHODS | _LEGACY_ADAPTIVE_POLICY_METHODS)
    }
    if not adaptive_policies:
        return _validate_empty_legacy_rows("router_decisions", rows)
    passed = all(
        str(row.get("policy_name") or "") in adaptive_policies
        and isinstance(row.get("triggered"), bool)
        and "selected_addon_solver" in row
        for row in rows
    )
    return {
        "passed": passed,
        "name": "router_decisions",
        "row_count": len(rows),
        "examples": rows[:20],
    }


def _count_by_method(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        method_name = str(row.get("method_name") or "")
        counts[method_name] = counts.get(method_name, 0) + 1
    return counts


def _diagnostic_path(
    diagnostic_paths: dict[str, Path],
    root: Path,
    filename: str,
) -> Path:
    return diagnostic_paths.get(filename, root / "diagnostics" / filename)
