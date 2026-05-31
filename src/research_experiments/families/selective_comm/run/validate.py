"""选择性通信运行校验。

同时覆盖工程正确性与实验设计正确性：
关键产物、请求成功率、共享哈希、一致/分歧规则、
early-exit 零通信约束，以及无效置信度比例。
"""

from __future__ import annotations

from collections import Counter, defaultdict
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
    validate_shared_contracts,
)


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    """Check selective communication run directory meets key artifact and constraint requirements."""
    index = resolve_run_artifact_index(run_dir, family_name="selective_comm")
    root = index.run_dir
    turn_paths = named_turn_record_paths(root, family_name="selective_comm")
    diagnostic_paths = named_diagnostic_paths(root, family_name="selective_comm")
    policy_reference_path = root / "exports" / "policy_reference_summary.json"
    required_paths = [
        index.manifest_path,
        turn_paths["stage_a_turns.jsonl"],
        turn_paths["stage_b_turns.jsonl"],
        turn_paths["trigger_decisions.jsonl"],
        index.prediction_records_path,
        index.metrics_view_path,
        diagnostic_paths["policy_diagnostics.json"],
        diagnostic_paths["oracle_trigger_eval.json"],
        policy_reference_path,
        index.report_path,
        index.figure_manifest_path,
        index.archive_manifest_path,
    ]
    missing = missing_relative_paths(root, required_paths)

    stage_a_rows = load_jsonl(turn_paths["stage_a_turns.jsonl"])
    stage_b_rows = load_jsonl(turn_paths["stage_b_turns.jsonl"])
    control_rows = load_jsonl(turn_paths["control_turns.jsonl"])
    trigger_rows = load_jsonl(turn_paths["trigger_decisions.jsonl"])
    prediction_rows = load_jsonl(index.prediction_records_path)
    diagnostics = load_json(diagnostic_paths["policy_diagnostics.json"]) if diagnostic_paths["policy_diagnostics.json"].exists() else {}

    all_turn_rows = stage_a_rows + stage_b_rows + control_rows
    status_summary = summarize_turn_statuses(all_turn_rows)

    manifest = load_json(index.manifest_path)
    shared_hash_check = _validate_shared_hashes(prediction_rows)
    disagreement_check = _validate_disagreement_policy(trigger_rows)
    early_exit_check = _validate_early_exit_tokens(prediction_rows)
    trigger_rate_check = _validate_always_trigger_rate(trigger_rows)
    invalid_confidence_check = _confidence_invalid_ratio(trigger_rows)
    rate_limit_check = validate_rate_limit_check(index.progress_path, all_turn_rows, manifest=manifest)
    shared_contracts = validate_shared_contracts(root)
    figure_contract = shared_contracts["figure_contract"]
    archive_contract = shared_contracts["archive_contract"]

    passed = all(
        [
            not missing,
            status_summary["request_failures"] == 0,
            status_summary["schema_failures"] == 0,
            status_summary["output_success_rate"] >= 0.95,
            shared_hash_check["passed"],
            disagreement_check["passed"],
            early_exit_check["passed"],
            trigger_rate_check["passed"],
            rate_limit_check["passed"],
            figure_contract["passed"],
            archive_contract["passed"],
        ]
    )

    return {
        "run_dir": str(root),
        "passed": passed,
        "missing_files": missing,
        "request_failures": status_summary["request_failures"],
        "schema_failures": status_summary["schema_failures"],
        "output_success_rate": status_summary["output_success_rate"],
        "checks": {
            "output_success_threshold": 0.95,
            "shared_hash_check": shared_hash_check,
            "always_trigger_rate_check": trigger_rate_check,
            "disagreement_policy_check": disagreement_check,
            "early_exit_zero_comm_check": early_exit_check,
            "invalid_confidence_ratio": invalid_confidence_check,
            "rate_limit_check": rate_limit_check,
            "figure_contract": figure_contract,
            "archive_contract": archive_contract,
        },
        "rate_limit_check": rate_limit_check,
        "policy_methods": dict(Counter(row.get("method_name") for row in prediction_rows)),
        "diagnostic_recommendation": diagnostics.get("recommended_next_default_policy"),
    }


def _validate_shared_hashes(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Check that each strategy shares the same Stage A / Stage B trace hash."""
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
    """Check that `always_communicate` always triggers."""
    rows = [row for row in trigger_rows if row.get("policy_name") == "always_communicate"]
    total = len(rows)
    triggered = sum(1 for row in rows if row.get("triggered"))
    rate = triggered / total if total else 0.0
    return {"passed": total > 0 and rate == 1.0, "total_rows": total, "trigger_rate": round(rate, 6)}


def _validate_disagreement_policy(trigger_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Check that disagreement strategy strictly equals `initial_disagreement`."""
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
    """Check that all early-exit questions have communication tokens equal to 0."""
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
    """Count invalid confidence value ratio."""
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

