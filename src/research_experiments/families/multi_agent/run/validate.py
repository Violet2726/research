"""Multi-agent run result validation.

The goal is to quickly confirm whether a multi-agent run has the minimum
conditions for continued analysis: key files present, no request/format
failures, non-empty question-level predictions, and paired analysis report.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from research_experiments.families.shared.validate_common import load_jsonl, summarize_turn_statuses, validate_shared_contracts


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    """Check whether key artifacts in a multi-agent run directory are present and basically usable."""
    root = Path(run_dir)
    required = [
        "manifest.json",
        "agent_turns.jsonl",
        "debate_messages.jsonl",
        "final_predictions.jsonl",
        "metrics.json",
        "cost_breakdown.json",
        "debate_diagnostics.json",
        "report.md",
        "figure_manifest.json",
        "archive_manifest.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    agent_rows = load_jsonl(root / "agent_turns.jsonl") if (root / "agent_turns.jsonl").exists() else []
    prediction_rows = load_jsonl(root / "final_predictions.jsonl") if (root / "final_predictions.jsonl").exists() else []

    status_summary = summarize_turn_statuses(agent_rows)
    methods = Counter(row.get("method_name") for row in prediction_rows)
    shared_contracts = validate_shared_contracts(root)
    figure_contract = shared_contracts["figure_contract"]
    archive_contract = shared_contracts["archive_contract"]
    return {
        "run_dir": str(root),
        "passed": (
            not missing
            and status_summary["request_failures"] == 0
            and status_summary["schema_failures"] == 0
            and bool(prediction_rows)
            and figure_contract["passed"]
            and archive_contract["passed"]
        ),
        "missing_files": missing,
        "request_failures": status_summary["request_failures"],
        "schema_failures": status_summary["schema_failures"],
        "prediction_rows": len(prediction_rows),
        "methods": dict(methods),
        "paired_analysis_present": (root / "paired_debate_vs_vote.json").exists(),
        "paired_report_present": (root / "report.md").exists(),
        "figure_contract": figure_contract,
        "archive_contract": archive_contract,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSONL file."""
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]