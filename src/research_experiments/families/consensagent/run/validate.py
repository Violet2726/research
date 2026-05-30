"""CONSENSAGENT run validation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from research_experiments.families.shared.validate_common import (
    load_jsonl,
    summarize_turn_statuses,
    validate_shared_contracts,
)


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)

    required_files = [
        "manifest.json",
        "turns.jsonl",
        "debate_messages.jsonl",
        "predictions.jsonl",
        "metrics.json",
        "cost_breakdown.json",
        "debate_diagnostics.json",
        "report.md",
        "figure_manifest.json",
        "archive_manifest.json",
    ]
    missing = [name for name in required_files if not (root / name).exists()]

    turn_rows = load_jsonl(root / "turns.jsonl") if (root / "turns.jsonl").exists() else []
    prediction_rows = load_jsonl(root / "predictions.jsonl") if (root / "predictions.jsonl").exists() else []

    status_summary = summarize_turn_statuses(turn_rows)
    methods = Counter(str(row.get("method_name")) for row in prediction_rows)

    shared_contracts = validate_shared_contracts(root)
    figure_contract = shared_contracts["figure_contract"]
    archive_contract = shared_contracts["archive_contract"]

    passed = (
        not missing
        and status_summary["request_failures"] == 0
        and status_summary["schema_failures"] == 0
        and bool(prediction_rows)
        and figure_contract["passed"]
        and archive_contract["passed"]
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
        "figure_contract": figure_contract,
        "archive_contract": archive_contract,
    }
