"""iMAD run artifact validation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from research_experiments.family_runtime.artifact_index import (
    named_diagnostic_paths,
    named_turn_record_paths,
    resolve_run_artifact_index,
)
from research_experiments.family_runtime.validation import (
    load_json,
    load_jsonl,
    summarize_turn_statuses,
    validate_rate_limit_check,
    validate_shared_contracts,
)

REQUIRED_PREDICTION_FIELDS = {
    "executed_round_count",
    "stopped_early",
    "stop_reason",
    "round_1_score",
    "round_2_score",
    "round_3_score",
    "ks_statistic_last",
    "posterior_mean_last",
}


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    """Check whether an iMAD run meets the minimum analysis contract."""

    index = resolve_run_artifact_index(run_dir, family_name="imad")
    root = index.run_dir
    turn_paths = named_turn_record_paths(root, family_name="imad")
    diagnostic_paths = named_diagnostic_paths(root, family_name="imad")
    required_paths = [
        index.manifest_path,
        turn_paths["agent_turns.jsonl"],
        turn_paths["debate_messages.jsonl"],
        turn_paths["round_diagnostics.jsonl"],
        index.prediction_records_path,
        index.metrics_view_path,
        diagnostic_paths["stability_diagnostics.json"],
        index.progress_path,
        index.report_path,
        index.figure_manifest_path,
        index.archive_manifest_path,
    ]
    missing = [path.relative_to(root).as_posix() for path in required_paths if not path.exists()]
    manifest = load_json(index.manifest_path)
    agent_rows = load_jsonl(turn_paths["agent_turns.jsonl"]) if turn_paths["agent_turns.jsonl"].exists() else []
    prediction_rows = load_jsonl(index.prediction_records_path) if index.prediction_records_path.exists() else []
    round_rows = load_jsonl(turn_paths["round_diagnostics.jsonl"]) if turn_paths["round_diagnostics.jsonl"].exists() else []

    status_summary = summarize_turn_statuses(agent_rows)
    methods = Counter(row.get("method_name") for row in prediction_rows)
    rate_limit_check = validate_rate_limit_check(index.progress_path, agent_rows, manifest=manifest)
    missing_prediction_fields = sorted(
        field
        for field in REQUIRED_PREDICTION_FIELDS
        if prediction_rows and any(field not in row for row in prediction_rows if row.get("method_type") == "mad")
    )
    invalid_round_counts = [
        row["sample_id"]
        for row in prediction_rows
        if row.get("method_type") == "mad" and int(row.get("executed_round_count") or 0) > 3
    ]
    adaptive_rows = [row for row in prediction_rows if row.get("method_name") == "imad_adaptive"]
    adaptive_stop_flag_mismatches = [
        row["sample_id"]
        for row in adaptive_rows
        if bool(row.get("stopped_early")) != (int(row.get("executed_round_count") or 0) < int(row.get("configured_round_limit") or 0) and row.get("stop_reason") == "stability_gate")
    ]
    shared_contracts = validate_shared_contracts(root)
    figure_contract = shared_contracts["figure_contract"]
    archive_contract = shared_contracts["archive_contract"]
    return {
        "run_dir": str(root),
        "passed": not missing
        and status_summary["request_failures"] == 0
        and status_summary["schema_failures"] == 0
        and bool(prediction_rows)
        and bool(round_rows)
        and not missing_prediction_fields
        and not invalid_round_counts
        and not adaptive_stop_flag_mismatches
        and rate_limit_check["passed"]
        and figure_contract["passed"]
        and archive_contract["passed"],
        "missing_files": missing,
        "request_failures": status_summary["request_failures"],
        "schema_failures": status_summary["schema_failures"],
        "prediction_rows": len(prediction_rows),
        "round_diagnostic_rows": len(round_rows),
        "methods": dict(methods),
        "missing_prediction_fields": missing_prediction_fields,
        "invalid_round_counts": invalid_round_counts[:20],
        "adaptive_stop_flag_mismatches": adaptive_stop_flag_mismatches[:20],
        "rate_limit_check": rate_limit_check,
        "figure_contract": figure_contract,
        "archive_contract": archive_contract,
    }

