"""DMAD 运行产物校验。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from research_experiments.family_runtime.artifact_index import (
    named_diagnostic_paths,
    named_export_paths,
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


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    index = resolve_run_artifact_index(run_dir, family_name="dmad")
    root = index.run_dir
    turn_paths = named_turn_record_paths(root, family_name="dmad")
    diagnostic_paths = named_diagnostic_paths(root, family_name="dmad")
    export_paths = named_export_paths(root, family_name="dmad")
    required_paths = [
        index.manifest_path,
        turn_paths["agent_turns.jsonl"],
        turn_paths["debate_messages.jsonl"],
        index.prediction_records_path,
        index.metrics_view_path,
        diagnostic_paths["strategy_diagnostics.json"],
        diagnostic_paths["cost_breakdown.json"],
        export_paths["paper_tables.json"],
        index.report_path,
        index.figure_manifest_path,
        index.archive_manifest_path,
    ]
    missing = missing_relative_paths(root, required_paths)
    agent_rows = load_jsonl(turn_paths["agent_turns.jsonl"]) if turn_paths["agent_turns.jsonl"].exists() else []
    prediction_rows = load_jsonl(index.prediction_records_path) if index.prediction_records_path.exists() else []
    diagnostics = load_json(diagnostic_paths["strategy_diagnostics.json"]) if diagnostic_paths["strategy_diagnostics.json"].exists() else {"rows": []}

    status_summary = summarize_turn_statuses(agent_rows)
    methods = Counter(str(row.get("method_name")) for row in prediction_rows)
    manifest = load_json(index.manifest_path)
    rate_limit_check = validate_rate_limit_check(index.progress_path, agent_rows, manifest=manifest)
    shared_contracts = validate_shared_contracts(root)
    figure_contract = shared_contracts["figure_contract"]
    archive_contract = shared_contracts["archive_contract"]
    passed = (
        not missing
        and status_summary["request_failures"] == 0
        and status_summary["schema_failures"] == 0
        and bool(prediction_rows)
        and bool(diagnostics.get("rows"))
        and rate_limit_check["passed"]
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
        "rate_limit_check": rate_limit_check,
        "figure_contract": figure_contract,
        "archive_contract": archive_contract,
    }

