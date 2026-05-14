"""iMAD 运行产物验证。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from research_experiments.families.shared.validate_common import load_jsonl, validate_shared_contracts


REQUIRED_PREDICTION_FIELDS = {
    "executed_round_count",
    "stopped_early",
    "stop_reason",
    "round_1_score",
    "round_2_score",
    "round_3_score",
    "ks_statistic_last",
    "posterior_mean_last",
}


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    """检查 iMAD run 是否满足最小分析契约。"""

    root = Path(run_dir)
    required = [
        "manifest.json",
        "agent_turns.jsonl",
        "debate_messages.jsonl",
        "round_diagnostics.jsonl",
        "final_predictions.jsonl",
        "metrics.json",
        "stability_diagnostics.json",
        "report.md",
        "figure_manifest.json",
        "archive_manifest.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    agent_rows = load_jsonl(root / "agent_turns.jsonl") if (root / "agent_turns.jsonl").exists() else []
    prediction_rows = load_jsonl(root / "final_predictions.jsonl") if (root / "final_predictions.jsonl").exists() else []
    round_rows = load_jsonl(root / "round_diagnostics.jsonl") if (root / "round_diagnostics.jsonl").exists() else []
    request_failures = sum(1 for row in agent_rows if row.get("output_status") == "request_fail")
    schema_failures = sum(1 for row in agent_rows if row.get("output_status") == "schema_fail")
    methods = Counter(row.get("method_name") for row in prediction_rows)
    missing_prediction_fields = sorted(
        field
        for field in REQUIRED_PREDICTION_FIELDS
        if prediction_rows and any(field not in row for row in prediction_rows if row.get("method_type") == "mad")
    )
    invalid_round_counts = [
        row["sample_id"]
        for row in prediction_rows
        if row.get("method_type") == "mad" and int(row.get("executed_round_count") or 0) > 3
    ]
    adaptive_rows = [row for row in prediction_rows if row.get("method_name") == "imad_adaptive"]
    adaptive_stop_flag_mismatches = [
        row["sample_id"]
        for row in adaptive_rows
        if bool(row.get("stopped_early")) != (int(row.get("executed_round_count") or 0) < int(row.get("configured_round_limit") or 0) and row.get("stop_reason") == "stability_gate")
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
        and bool(round_rows)
        and not missing_prediction_fields
        and not invalid_round_counts
        and not adaptive_stop_flag_mismatches
        and figure_contract["passed"]
        and archive_contract["passed"],
        "missing_files": missing,
        "request_failures": request_failures,
        "schema_failures": schema_failures,
        "prediction_rows": len(prediction_rows),
        "round_diagnostic_rows": len(round_rows),
        "methods": dict(methods),
        "missing_prediction_fields": missing_prediction_fields,
        "invalid_round_counts": invalid_round_counts[:20],
        "adaptive_stop_flag_mismatches": adaptive_stop_flag_mismatches[:20],
        "figure_contract": figure_contract,
        "archive_contract": archive_contract,
    }
