"""MacNet run artifact validation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from research_experiments.families.shared.validate_common import (
    load_jsonl,
    summarize_turn_statuses,
    validate_shared_contracts,
)

REQUIRED_PREDICTION_FIELDS = {
    "topology_type",
    "node_scale",
    "dataset",
    "sample_id",
    "initial_artifact",
    "final_artifact",
    "final_answer",
    "artifact_revision_count",
    "inbound_instruction_count",
    "max_context_tokens_observed",
    "topology_direction_mode",
}


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    """Check whether a MacNet run meets the minimum analysis contract."""

    root = Path(run_dir)
    required = [
        "manifest.json",
        "artifact_trace.jsonl",
        "instruction_trace.jsonl",
        "final_predictions.jsonl",
        "topology_manifest.json",
        "scaling_summary.json",
        "metrics.json",
        "report.md",
        "figure_manifest.json",
        "archive_manifest.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    artifact_rows = load_jsonl(root / "artifact_trace.jsonl") if (root / "artifact_trace.jsonl").exists() else []
    instruction_rows = load_jsonl(root / "instruction_trace.jsonl") if (root / "instruction_trace.jsonl").exists() else []
    prediction_rows = load_jsonl(root / "final_predictions.jsonl") if (root / "final_predictions.jsonl").exists() else []

    status_summary = summarize_turn_statuses(artifact_rows + instruction_rows)
    methods = Counter(row.get("method_name") for row in prediction_rows)
    missing_prediction_fields = sorted(
        field for field in REQUIRED_PREDICTION_FIELDS if prediction_rows and any(field not in row for row in prediction_rows)
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
        and bool(artifact_rows)
        and not missing_prediction_fields
        and figure_contract["passed"]
        and archive_contract["passed"],
        "missing_files": missing,
        "request_failures": status_summary["request_failures"],
        "schema_failures": status_summary["schema_failures"],
        "artifact_trace_rows": len(artifact_rows),
        "instruction_trace_rows": len(instruction_rows),
        "prediction_rows": len(prediction_rows),
        "methods": dict(methods),
        "missing_prediction_fields": missing_prediction_fields,
        "figure_contract": figure_contract,
        "archive_contract": archive_contract,
    }
