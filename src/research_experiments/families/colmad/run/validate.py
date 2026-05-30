"""ColMAD run artifact validation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from research_experiments.families.shared.validate_common import load_jsonl, summarize_turn_statuses, validate_shared_contracts

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

    root = Path(run_dir)
    required = [
        "manifest.json",
        "debate_trace.jsonl",
        "judge_trace.jsonl",
        "final_predictions.jsonl",
        "metrics.json",
        "protocol_diagnostics.json",
        "report.md",
        "figure_manifest.json",
        "archive_manifest.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    debate_rows = load_jsonl(root / "debate_trace.jsonl") if (root / "debate_trace.jsonl").exists() else []
    judge_rows = load_jsonl(root / "judge_trace.jsonl") if (root / "judge_trace.jsonl").exists() else []
    prediction_rows = load_jsonl(root / "final_predictions.jsonl") if (root / "final_predictions.jsonl").exists() else []

    status_summary = summarize_turn_statuses(debate_rows + judge_rows)
    methods = Counter(row.get("method_name") for row in prediction_rows)
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
        "figure_contract": figure_contract,
        "archive_contract": archive_contract,
    }