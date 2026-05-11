"""选择性通信运行结果校验。

该校验器同时验证工程正确性与实验设计正确性：
既检查关键文件、请求成功率，也检查共享哈希、always/disagreement 规则、
early-exit 零通信约束和非法置信度比例。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
from typing import Any

from experiment_core.foundation.run_archives import validate_archive_contract
from experiment_core.reporting.run_figures import validate_figure_contract


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    """检查选择性通信运行目录的关键产物与约束是否满足。"""
    root = Path(run_dir)
    required = [
        "manifest.json",
        "stage_a_turns.jsonl",
        "stage_b_turns.jsonl",
        "trigger_decisions.jsonl",
        "policy_predictions.jsonl",
        "policy_metrics.json",
        "policy_diagnostics.json",
        "oracle_trigger_eval.json",
        "policy_reference_summary.json",
        "progress.json",
        "report.md",
        "figure_manifest.json",
        "archive_manifest.json",
    ]
    missing = [name for name in required if not (root / name).exists()]

    stage_a_rows = _load_jsonl(root / "stage_a_turns.jsonl")
    stage_b_rows = _load_jsonl(root / "stage_b_turns.jsonl")
    control_rows = _load_jsonl(root / "control_turns.jsonl")
    trigger_rows = _load_jsonl(root / "trigger_decisions.jsonl")
    prediction_rows = _load_jsonl(root / "policy_predictions.jsonl")
    diagnostics = _load_json(root / "policy_diagnostics.json") if (root / "policy_diagnostics.json").exists() else {}

    all_turn_rows = stage_a_rows + stage_b_rows + control_rows
    request_failures = sum(1 for row in all_turn_rows if row.get("output_status") == "request_fail")
    output_success_count = sum(1 for row in all_turn_rows if row.get("output_status") == "ok")
    output_success_rate = output_success_count / len(all_turn_rows) if all_turn_rows else 0.0

    shared_hash_check = _validate_shared_hashes(prediction_rows)
    disagreement_check = _validate_disagreement_policy(trigger_rows)
    early_exit_check = _validate_early_exit_tokens(prediction_rows)
    trigger_rate_check = _validate_always_trigger_rate(trigger_rows)
    invalid_confidence_check = _confidence_invalid_ratio(trigger_rows)
    figure_contract = validate_figure_contract(root)
    archive_contract = validate_archive_contract(root)

    passed = all(
        [
            not missing,
            request_failures == 0,
            output_success_rate >= 0.95,
            shared_hash_check["passed"],
            disagreement_check["passed"],
            early_exit_check["passed"],
            trigger_rate_check["passed"],
            figure_contract["passed"],
            archive_contract["passed"],
        ]
    )

    return {
        "run_dir": str(root),
        "passed": passed,
        "missing_files": missing,
        "checks": {
            "request_failures_total": request_failures,
            "output_success_rate": round(output_success_rate, 6),
            "output_success_threshold": 0.95,
            "shared_hash_check": shared_hash_check,
            "always_trigger_rate_check": trigger_rate_check,
            "disagreement_policy_check": disagreement_check,
            "early_exit_zero_comm_check": early_exit_check,
            "invalid_confidence_ratio": invalid_confidence_check,
            "figure_contract": figure_contract,
            "archive_contract": archive_contract,
        },
        "policy_methods": dict(Counter(row.get("method_name") for row in prediction_rows)),
        "diagnostic_recommendation": diagnostics.get("recommended_next_default_policy"),
    }


def _validate_shared_hashes(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """检查各策略是否共享同一份 Stage A / Stage B trace 哈希。"""
    mismatches: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        if row.get("method_kind") != "policy":
            continue
        grouped[(row["dataset"], row["sample_id"])].append(row)

    for (dataset, sample_id), rows in grouped.items():
        stage_a_hashes = {row.get("stage_a_trace_hash") for row in rows}
        if len(stage_a_hashes) != 1:
            mismatches.append(
                {
                    "dataset": dataset,
                    "sample_id": sample_id,
                    "issue": "stage_a_hash_mismatch",
                    "values": sorted(value for value in stage_a_hashes if value),
                }
            )
        triggered_rows = [row for row in rows if row.get("triggered")]
        stage_b_hashes = {row.get("stage_b_trace_hash_used") for row in triggered_rows}
        stage_b_hashes.discard(None)
        if triggered_rows and len(stage_b_hashes) != 1:
            mismatches.append(
                {
                    "dataset": dataset,
                    "sample_id": sample_id,
                    "issue": "stage_b_hash_mismatch",
                    "values": sorted(stage_b_hashes),
                }
            )
    return {"passed": len(mismatches) == 0, "mismatch_count": len(mismatches), "mismatches": mismatches[:20]}


def _validate_always_trigger_rate(trigger_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """检查 ``always_communicate`` 是否总会触发。"""
    rows = [row for row in trigger_rows if row.get("policy_name") == "always_communicate"]
    total = len(rows)
    triggered = sum(1 for row in rows if row.get("triggered"))
    rate = triggered / total if total else 0.0
    return {"passed": total > 0 and rate == 1.0, "total_rows": total, "trigger_rate": round(rate, 6)}


def _validate_disagreement_policy(trigger_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """检查 disagreement 策略是否严格等于 ``initial_disagreement``。"""
    mismatches = []
    rows = [row for row in trigger_rows if row.get("policy_name") == "disagreement_triggered"]
    for row in rows:
        if bool(row.get("triggered")) != bool(row.get("initial_disagreement")):
            mismatches.append(
                {
                    "dataset": row["dataset"],
                    "sample_id": row["sample_id"],
                    "triggered": row["triggered"],
                    "initial_disagreement": row["initial_disagreement"],
                }
            )
    return {"passed": len(mismatches) == 0, "mismatch_count": len(mismatches), "mismatches": mismatches[:20]}


def _validate_early_exit_tokens(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """检查所有 early exit 题目的通信 token 是否为 0。"""
    mismatches = []
    for row in prediction_rows:
        if row.get("method_kind") != "policy":
            continue
        if row.get("early_exit") and float(row.get("communication_tokens_per_question") or 0.0) != 0.0:
            mismatches.append(
                {
                    "dataset": row["dataset"],
                    "sample_id": row["sample_id"],
                    "method_name": row["method_name"],
                    "communication_tokens_per_question": row["communication_tokens_per_question"],
                }
            )
    return {"passed": len(mismatches) == 0, "mismatch_count": len(mismatches), "mismatches": mismatches[:20]}


def _confidence_invalid_ratio(trigger_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """统计 confidence 非法值比例。"""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trigger_rows:
        grouped[row["dataset"]].append(row)
    per_dataset = {
        dataset: round(sum(1 for row in rows if row.get("any_invalid_confidence")) / len(rows), 6)
        for dataset, rows in grouped.items()
        if rows
    }
    overall_denominator = len(trigger_rows)
    overall_numerator = sum(1 for row in trigger_rows if row.get("any_invalid_confidence"))
    return {
        "overall_ratio": round(overall_numerator / overall_denominator, 6) if overall_denominator else 0.0,
        "per_dataset": per_dataset,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 UTF-8 JSONL 文件。"""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_json(path: Path) -> dict[str, Any]:
    """读取 UTF-8 JSON 文件。"""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
