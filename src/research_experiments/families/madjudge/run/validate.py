"""MADJudge 运行产物校验。"""

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
    missing_relative_paths,
    summarize_turn_statuses,
    validate_rate_limit_check,
)


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    index = resolve_run_artifact_index(run_dir, family_name="madjudge")
    root = index.run_dir
    turn_paths = named_turn_record_paths(root, family_name="madjudge")
    diagnostic_paths = named_diagnostic_paths(root, family_name="madjudge")
    required_paths = [
        index.manifest_path,
        turn_paths["turns.jsonl"],
        turn_paths["debate_messages.jsonl"],
        index.prediction_records_path,
        index.metrics_view_path,
        diagnostic_paths["debate_diagnostics.json"],
        diagnostic_paths["cost_breakdown.json"],
    ]
    missing = missing_relative_paths(root, required_paths)

    manifest = load_json(index.manifest_path)
    turn_rows = load_jsonl(turn_paths["turns.jsonl"]) if turn_paths["turns.jsonl"].exists() else []
    prediction_rows = load_jsonl(index.prediction_records_path) if index.prediction_records_path.exists() else []

    status_summary = summarize_turn_statuses(turn_rows)
    methods = Counter(str(row.get("method_name")) for row in prediction_rows)
    rate_limit_check = validate_rate_limit_check(index.progress_path, turn_rows, manifest=manifest)

    passed = (
        not missing
        and status_summary["request_failures"] == 0
        and status_summary["schema_failures"] == 0
        and bool(prediction_rows)
        and rate_limit_check["passed"]
    )
    return {
        "run_dir": str(root),
        "passed": passed,
        "missing_files": missing,
        "request_failures": status_summary["request_failures"],
        "schema_failures": status_summary["schema_failures"],
        "output_success_rate": status_summary["output_success_rate"],
        "turn_rows": status_summary["total_turns"],
        "prediction_rows": len(prediction_rows),
        "methods": dict(methods),
        "rate_limit_check": rate_limit_check,
    }

