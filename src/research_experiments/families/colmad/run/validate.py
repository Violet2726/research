"""ColMAD run artifact validation."""

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
    "task_name",
    "candidate_response_model",
    "gold",
    "final_verdict",
    "single_agent_verdict",
    "copmad_verdict",
    "colmad_verdict",
    "changed_after_debate",
    "shift_direction",
    "judge_confidence",
    "debate_protocol",
}

REQUIRED_JUDGE_FIELDS = {
    "method_name",
    "debate_protocol",
    "verdict",
    "observed_failure_modes",
}


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    """Check whether a ColMAD run meets the minimum analysis contract."""

    index = resolve_run_artifact_index(run_dir, family_name="colmad")
    root = index.run_dir
    turn_paths = named_turn_record_paths(root, family_name="colmad")
    diagnostic_paths = named_diagnostic_paths(root, family_name="colmad")
    required_paths = [
        index.manifest_path,
        turn_paths["debate_trace.jsonl"],
        turn_paths["judge_trace.jsonl"],
        index.prediction_records_path,
        index.metrics_view_path,
        diagnostic_paths["protocol_diagnostics.json"],
        index.progress_path,
        index.report_path,
        index.figure_manifest_path,
        index.archive_manifest_path,
    ]
    missing = [path.relative_to(root).as_posix() for path in required_paths if not path.exists()]
    manifest = load_json(index.manifest_path)
    debate_rows = load_jsonl(turn_paths["debate_trace.jsonl"]) if turn_paths["debate_trace.jsonl"].exists() else []
    judge_rows = load_jsonl(turn_paths["judge_trace.jsonl"]) if turn_paths["judge_trace.jsonl"].exists() else []
    prediction_rows = load_jsonl(index.prediction_records_path) if index.prediction_records_path.exists() else []

    status_summary = summarize_turn_statuses(debate_rows + judge_rows)
    methods = Counter(row.get("method_name") for row in prediction_rows)
    rate_limit_check = validate_rate_limit_check(index.progress_path, debate_rows + judge_rows, manifest=manifest)
    missing_prediction_fields = sorted(
        field for field in REQUIRED_PREDICTION_FIELDS if prediction_rows and any(field not in row for row in prediction_rows)
    )
    missing_judge_fields = sorted(
        field for field in REQUIRED_JUDGE_FIELDS if judge_rows and any(field not in row for row in judge_rows)
    )
    shared_contracts = validate_shared_contracts(root)
    figure_contract = shared_contracts["figure_contract"]
    archive_contract = shared_contracts["archive_contract"]
    return {
        "run_dir": str(root),
        "passed": not missing
        and status_summary["request_failures"] == 0
        and status_summary["schema_failures"] == 0
        and bool(prediction_rows)
        and bool(debate_rows)
        and bool(judge_rows)
        and not missing_prediction_fields
        and not missing_judge_fields
        and rate_limit_check["passed"]
        and figure_contract["passed"]
        and archive_contract["passed"],
        "missing_files": missing,
        "request_failures": status_summary["request_failures"],
        "schema_failures": status_summary["schema_failures"],
        "debate_trace_rows": len(debate_rows),
        "judge_trace_rows": len(judge_rows),
        "prediction_rows": len(prediction_rows),
        "methods": dict(methods),
        "missing_prediction_fields": missing_prediction_fields,
        "missing_judge_fields": missing_judge_fields,
        "rate_limit_check": rate_limit_check,
        "figure_contract": figure_contract,
        "archive_contract": archive_contract,
    }

