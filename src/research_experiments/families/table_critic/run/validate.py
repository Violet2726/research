"""Table-Critic 运行产物验证。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from research_experiments.families.shared.validate_common import load_json, load_jsonl, validate_shared_contracts


REQUIRED_PREDICTION_FIELDS = {
    "initial_answer",
    "final_answer",
    "judge_error_detected",
    "judge_error_step",
    "critic_feedback",
    "refinement_round_count",
    "stopped_reason",
    "template_ids_used",
    "table_id",
    "question_type",
}

REQUIRED_CRITIC_TRACE_FIELDS = {
    "judge_error_detected",
    "judge_error_step",
    "critic_feedback",
    "template_ids_used",
    "template_reuse_count",
}

REQUIRED_REFINEMENT_TRACE_FIELDS = {
    "round_index",
    "initial_answer",
    "previous_answer",
    "refined_answer",
    "stopped_reason",
}


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    """检查 Table-Critic run 是否满足最小分析契约。"""

    root = Path(run_dir)
    required = [
        "manifest.json",
        "agent_turns.jsonl",
        "critic_trace.jsonl",
        "refinement_trace.jsonl",
        "final_predictions.jsonl",
        "metrics.json",
        "template_tree.json",
        "error_analysis.json",
        "report.md",
        "figure_manifest.json",
        "archive_manifest.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    turn_rows = load_jsonl(root / "agent_turns.jsonl") if (root / "agent_turns.jsonl").exists() else []
    critic_rows = load_jsonl(root / "critic_trace.jsonl") if (root / "critic_trace.jsonl").exists() else []
    refinement_rows = load_jsonl(root / "refinement_trace.jsonl") if (root / "refinement_trace.jsonl").exists() else []
    prediction_rows = load_jsonl(root / "final_predictions.jsonl") if (root / "final_predictions.jsonl").exists() else []
    request_failures = sum(1 for row in turn_rows if row.get("output_status") == "request_fail")
    schema_failures = sum(1 for row in turn_rows if row.get("output_status") == "schema_fail")
    methods = Counter(row.get("method_name") for row in prediction_rows)
    missing_prediction_fields = sorted(
        field for field in REQUIRED_PREDICTION_FIELDS if prediction_rows and any(field not in row for row in prediction_rows)
    )
    missing_critic_fields = sorted(
        field for field in REQUIRED_CRITIC_TRACE_FIELDS if critic_rows and any(field not in row for row in critic_rows)
    )
    missing_refinement_fields = sorted(
        field for field in REQUIRED_REFINEMENT_TRACE_FIELDS if refinement_rows and any(field not in row for row in refinement_rows)
    )
    shared_contracts = validate_shared_contracts(root)
    figure_contract = shared_contracts["figure_contract"]
    archive_contract = shared_contracts["archive_contract"]
    return {
        "run_dir": str(root),
        "passed": not missing
        and request_failures == 0
        and schema_failures == 0
        and bool(prediction_rows)
        and bool(critic_rows)
        and bool(refinement_rows)
        and not missing_prediction_fields
        and not missing_critic_fields
        and not missing_refinement_fields
        and figure_contract["passed"]
        and archive_contract["passed"],
        "missing_files": missing,
        "request_failures": request_failures,
        "schema_failures": schema_failures,
        "prediction_rows": len(prediction_rows),
        "critic_trace_rows": len(critic_rows),
        "refinement_trace_rows": len(refinement_rows),
        "methods": dict(methods),
        "missing_prediction_fields": missing_prediction_fields,
        "missing_critic_fields": missing_critic_fields,
        "missing_refinement_fields": missing_refinement_fields,
        "figure_contract": figure_contract,
        "archive_contract": archive_contract,
    }

