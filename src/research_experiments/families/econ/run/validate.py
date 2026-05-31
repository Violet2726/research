"""ECON run artifact validation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from research_experiments.family_runtime.artifact_index import named_turn_record_paths, resolve_run_artifact_index
from research_experiments.family_runtime.validation import (
    load_json,
    load_jsonl,
    summarize_turn_statuses,
    validate_rate_limit_check,
    validate_shared_contracts,
)

REQUIRED_PREDICTION_FIELDS = {
    "initial_answer",
    "final_answer",
    "selected_action",
    "belief_score",
    "expected_gain",
    "communication_cost",
    "changed_after_coordination",
    "coordination_mode",
}

ALLOWED_ACTIONS = {"none", "adopt_vote", "keep_local", "query_best_peer", "query_two_peers", "query_all_peers"}


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    """Check whether an ECON run meets the minimum analysis contract."""

    index = resolve_run_artifact_index(run_dir, family_name="econ")
    root = index.run_dir
    turn_paths = named_turn_record_paths(root, family_name="econ")
    required_paths = [
        index.manifest_path,
        turn_paths["agent_turns.jsonl"],
        turn_paths["belief_trace.jsonl"],
        turn_paths["equilibrium_trace.jsonl"],
        turn_paths["communication_trace.jsonl"],
        index.prediction_records_path,
        index.metrics_view_path,
        index.progress_path,
        index.report_path,
        index.figure_manifest_path,
        index.archive_manifest_path,
    ]
    missing = [path.relative_to(root).as_posix() for path in required_paths if not path.exists()]
    manifest = load_json(index.manifest_path)
    turn_rows = load_jsonl(turn_paths["agent_turns.jsonl"]) if turn_paths["agent_turns.jsonl"].exists() else []
    belief_rows = load_jsonl(turn_paths["belief_trace.jsonl"]) if turn_paths["belief_trace.jsonl"].exists() else []
    equilibrium_rows = load_jsonl(turn_paths["equilibrium_trace.jsonl"]) if turn_paths["equilibrium_trace.jsonl"].exists() else []
    communication_rows = load_jsonl(turn_paths["communication_trace.jsonl"]) if turn_paths["communication_trace.jsonl"].exists() else []
    prediction_rows = load_jsonl(index.prediction_records_path) if index.prediction_records_path.exists() else []

    status_summary = summarize_turn_statuses(turn_rows)
    methods = Counter(row.get("method_name") for row in prediction_rows)
    rate_limit_check = validate_rate_limit_check(index.progress_path, turn_rows, manifest=manifest)
    missing_prediction_fields = sorted(
        field for field in REQUIRED_PREDICTION_FIELDS if prediction_rows and any(field not in row for row in prediction_rows)
    )
    invalid_actions = [
        row.get("selected_action")
        for row in prediction_rows
        if row.get("selected_action") not in ALLOWED_ACTIONS
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
        and bool(belief_rows)
        and bool(equilibrium_rows)
        and bool(communication_rows)
        and not missing_prediction_fields
        and not invalid_actions
        and rate_limit_check["passed"]
        and figure_contract["passed"]
        and archive_contract["passed"],
        "missing_files": missing,
        "request_failures": status_summary["request_failures"],
        "schema_failures": status_summary["schema_failures"],
        "prediction_rows": len(prediction_rows),
        "belief_trace_rows": len(belief_rows),
        "equilibrium_trace_rows": len(equilibrium_rows),
        "communication_trace_rows": len(communication_rows),
        "methods": dict(methods),
        "missing_prediction_fields": missing_prediction_fields,
        "invalid_actions": invalid_actions[:20],
        "rate_limit_check": rate_limit_check,
        "figure_contract": figure_contract,
        "archive_contract": archive_contract,
    }

