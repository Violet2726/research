"""CUE 运行产物的完整性与一致性校验。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
from typing import Any

from experiment_core.reporting.run_figures import validate_figure_contract


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    """校验单次 CUE 运行目录是否满足最小可复现要求。"""
    root = Path(run_dir)
    required = [
        "manifest.json",
        "stage_a_turns.jsonl",
        "communication_turns.jsonl",
        "audit_turns.jsonl",
        "policy_predictions.jsonl",
        "policy_metrics.json",
        "policy_diagnostics.json",
        "oracle_trigger_eval.json",
        "progress.json",
        "report.md",
        "figure_manifest.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    stage_a_rows = _load_jsonl(root / "stage_a_turns.jsonl")
    communication_rows = _load_jsonl(root / "communication_turns.jsonl")
    audit_rows = _load_jsonl(root / "audit_turns.jsonl")
    control_rows = _load_jsonl(root / "control_turns.jsonl")
    prediction_rows = _load_jsonl(root / "policy_predictions.jsonl")
    all_turn_rows = stage_a_rows + communication_rows + audit_rows + control_rows
    request_failures = sum(1 for row in all_turn_rows if row.get("output_status") == "request_fail")
    output_success_count = sum(1 for row in all_turn_rows if row.get("output_status") == "ok")
    output_success_rate = output_success_count / len(all_turn_rows) if all_turn_rows else 0.0
    stage_a_hash_check = _validate_stage_a_hashes(prediction_rows)
    figure_contract = validate_figure_contract(root)
    passed = not missing and request_failures == 0 and output_success_rate >= 0.90 and stage_a_hash_check["passed"] and figure_contract["passed"]
    return {
        "run_dir": str(root),
        "passed": passed,
        "missing_files": missing,
        "checks": {
            "request_failures_total": request_failures,
            "output_success_rate": round(output_success_rate, 6),
            "stage_a_hash_check": stage_a_hash_check,
            "figure_contract": figure_contract,
        },
        "policy_methods": dict(Counter(row.get("method_name") for row in prediction_rows)),
    }


def _validate_stage_a_hashes(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in prediction_rows:
        grouped.setdefault((row["dataset"], row["sample_id"]), []).append(row)
    for (dataset, sample_id), rows in grouped.items():
        stage_a_hashes = {row.get("stage_a_trace_hash") for row in rows if row.get("method_name") != "mv_6" and row.get("method_name") != "sc_6"}
        stage_a_hashes.discard(None)
        if len(stage_a_hashes) > 1:
            mismatches.append({"dataset": dataset, "sample_id": sample_id, "issue": "stage_a_hash_mismatch", "values": sorted(stage_a_hashes)})
    return {"passed": len(mismatches) == 0, "mismatch_count": len(mismatches), "mismatches": mismatches[:20]}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
