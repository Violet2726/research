"""DMAD run validation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from research_experiments.families.shared.validate_common import (
    load_json,
    load_jsonl,
    summarize_turn_statuses,
    validate_shared_contracts,
)


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    required = [
        "manifest.json",
        "agent_turns.jsonl",
        "debate_messages.jsonl",
        "final_predictions.jsonl",
        "metrics.json",
        "strategy_diagnostics.json",
        "cost_breakdown.json",
        "paper_tables.json",
        "report.md",
        "figure_manifest.json",
        "archive_manifest.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    agent_rows = load_jsonl(root / "agent_turns.jsonl") if (root / "agent_turns.jsonl").exists() else []
    prediction_rows = load_jsonl(root / "final_predictions.jsonl") if (root / "final_predictions.jsonl").exists() else []
    diagnostics = load_json(root / "strategy_diagnostics.json") if (root / "strategy_diagnostics.json").exists() else {"rows": []}

    status_summary = summarize_turn_statuses(agent_rows)
    methods = Counter(str(row.get("method_name")) for row in prediction_rows)
    shared_contracts = validate_shared_contracts(root)
    figure_contract = shared_contracts["figure_contract"]
    archive_contract = shared_contracts["archive_contract"]
    passed = (
        not missing
        and status_summary["request_failures"] == 0
        and status_summary["schema_failures"] == 0
        and bool(prediction_rows)
        and bool(diagnostics.get("rows"))
        and figure_contract["passed"]
        and archive_contract["passed"]
    )
    return {
        "run_dir": str(root),
        "passed": passed,
        "missing_files": missing,
        "request_failures": status_summary["request_failures"],
        "schema_failures": status_summary["schema_failures"],
        "prediction_rows": len(prediction_rows),
        "methods": dict(methods),
        "strategy_diagnostics_present": bool(diagnostics.get("rows")),
        "figure_contract": figure_contract,
        "archive_contract": archive_contract,
    }
