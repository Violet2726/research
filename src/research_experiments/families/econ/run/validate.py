"""ECON 运行产物验证。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from research_experiments.families.shared.validate_common import load_jsonl, validate_shared_contracts


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
    """检查 ECON run 是否满足最小分析契约。"""

    root = Path(run_dir)
    required = [
        "manifest.json",
        "agent_turns.jsonl",
        "belief_trace.jsonl",
        "equilibrium_trace.jsonl",
        "communication_trace.jsonl",
        "final_predictions.jsonl",
        "metrics.json",
        "report.md",
        "figure_manifest.json",
        "archive_manifest.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    turn_rows = load_jsonl(root / "agent_turns.jsonl") if (root / "agent_turns.jsonl").exists() else []
    belief_rows = load_jsonl(root / "belief_trace.jsonl") if (root / "belief_trace.jsonl").exists() else []
    equilibrium_rows = load_jsonl(root / "equilibrium_trace.jsonl") if (root / "equilibrium_trace.jsonl").exists() else []
    communication_rows = load_jsonl(root / "communication_trace.jsonl") if (root / "communication_trace.jsonl").exists() else []
    prediction_rows = load_jsonl(root / "final_predictions.jsonl") if (root / "final_predictions.jsonl").exists() else []
    request_failures = sum(1 for row in turn_rows if row.get("output_status") == "request_fail")
    schema_failures = sum(1 for row in turn_rows if row.get("output_status") == "schema_fail")
    methods = Counter(row.get("method_name") for row in prediction_rows)
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
        and request_failures == 0
        and schema_failures == 0
        and bool(prediction_rows)
        and bool(belief_rows)
        and bool(equilibrium_rows)
        and bool(communication_rows)
        and not missing_prediction_fields
        and not invalid_actions
        and figure_contract["passed"]
        and archive_contract["passed"],
        "missing_files": missing,
        "request_failures": request_failures,
        "schema_failures": schema_failures,
        "prediction_rows": len(prediction_rows),
        "belief_trace_rows": len(belief_rows),
        "equilibrium_trace_rows": len(equilibrium_rows),
        "communication_trace_rows": len(communication_rows),
        "methods": dict(methods),
        "missing_prediction_fields": missing_prediction_fields,
        "invalid_actions": invalid_actions[:20],
        "figure_contract": figure_contract,
        "archive_contract": archive_contract,
    }

