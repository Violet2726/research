"""A-SMAD 运行产物校验。

本模块检查 run 目录是否满足 family 产物契约，并确认当前 A-SMAD 主线没有遗留 Stage B / judge 行。
校验结果用于 CLI、报告刷新和自动化回归检查。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from research_experiments.families.adaptive_sparse_mad.config import (
    ADAPTIVE_POLICY_METHODS,
    COT_MAD_GLOBAL_SYNC_METHOD,
    ADAPTIVE_SPARSE_DEBATE_METHOD,
)
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
    """校验单个 A-SMAD run 目录并返回结构化检查结果。"""
    index = resolve_run_artifact_index(run_dir, family_name="adaptive_sparse_mad")
    root = index.run_dir
    turn_paths = named_turn_record_paths(root, family_name="adaptive_sparse_mad")
    diagnostic_paths = named_diagnostic_paths(root, family_name="adaptive_sparse_mad")
    legacy_stage_b_path = root / "turns" / "stage_b_turns.jsonl"
    legacy_judge_path = root / "turns" / "judge_turns.jsonl"
    manifest = load_json(index.manifest_path)
    debate_messages_path = turn_paths.get("debate_messages.jsonl", root / "turns" / "debate_messages.jsonl")
    debate_method_enabled = _manifest_includes_any_method(
        manifest,
        {ADAPTIVE_SPARSE_DEBATE_METHOD, COT_MAD_GLOBAL_SYNC_METHOD},
    )
    required_paths = [
        index.manifest_path,
        turn_paths["stage_a_turns.jsonl"],
        turn_paths["control_turns.jsonl"],
        turn_paths["router_decisions.jsonl"],
        index.prediction_records_path,
        index.metrics_view_path,
        _diagnostic_path(diagnostic_paths, root, "router_eval.json"),
        _diagnostic_path(diagnostic_paths, root, "policy_diagnostics.json"),
        _diagnostic_path(diagnostic_paths, root, "stage_a_resolver_breakdown.json"),
        _diagnostic_path(diagnostic_paths, root, "stage_a_error_buckets.json"),
        _diagnostic_path(diagnostic_paths, root, "stage_a_solver_contributions.json"),
        index.report_path,
        index.figure_manifest_path,
        index.archive_manifest_path,
    ]
    if debate_method_enabled:
        required_paths.append(debate_messages_path)
    missing = missing_relative_paths(root, required_paths)

    stage_a_rows = load_jsonl(turn_paths["stage_a_turns.jsonl"])
    stage_b_rows = load_jsonl(legacy_stage_b_path) if legacy_stage_b_path.exists() else []
    judge_rows = load_jsonl(legacy_judge_path) if legacy_judge_path.exists() else []
    control_rows = load_jsonl(turn_paths["control_turns.jsonl"])
    debate_message_rows = load_jsonl(debate_messages_path) if debate_messages_path.exists() else []
    router_rows = load_jsonl(turn_paths["router_decisions.jsonl"])
    prediction_rows = load_jsonl(index.prediction_records_path)
    router_eval_payload = load_json(_diagnostic_path(diagnostic_paths, root, "router_eval.json"))
    policy_diagnostics_payload = load_json(_diagnostic_path(diagnostic_paths, root, "policy_diagnostics.json"))

    all_turn_rows = stage_a_rows + stage_b_rows + judge_rows + control_rows
    status_summary = summarize_turn_statuses(all_turn_rows)
    router_empty_check = _validate_router_rows(router_rows, manifest=manifest, prediction_rows=prediction_rows)
    router_diagnostics_check = _validate_router_diagnostics(
        router_eval_payload,
        policy_diagnostics_payload,
        manifest=manifest,
        prediction_rows=prediction_rows,
    )
    debate_messages_check = _validate_debate_messages(
        debate_message_rows,
        required=debate_method_enabled,
        path_exists=debate_messages_path.exists(),
    )
    stage_b_judge_empty_check = _validate_empty_legacy_rows(
        "stage_b_and_judge",
        stage_b_rows + judge_rows,
        files_present=[
            legacy_stage_b_path.exists(),
            legacy_judge_path.exists(),
        ],
    )
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
            router_empty_check["passed"],
            router_diagnostics_check["passed"],
            debate_messages_check["passed"],
            stage_b_judge_empty_check["passed"],
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
            "router_empty_check": router_empty_check,
            "router_diagnostics_check": router_diagnostics_check,
            "debate_messages_check": debate_messages_check,
            "stage_b_judge_empty_check": stage_b_judge_empty_check,
            "rate_limit_check": rate_limit_check,
            "figure_contract": figure_contract,
            "archive_contract": archive_contract,
        },
        "rate_limit_check": rate_limit_check,
        "method_counts": _count_by_method(prediction_rows),
    }


def _validate_empty_legacy_rows(
    name: str,
    rows: list[dict[str, Any]],
    *,
    files_present: list[bool] | None = None,
) -> dict[str, Any]:
    """确认已退役产物没有继续产生记录。"""
    return {
        "passed": len(rows) == 0,
        "name": name,
        "row_count": len(rows),
        "files_present_count": sum(1 for flag in (files_present or []) if flag),
        "examples": rows[:20],
    }


def _validate_router_rows(
    rows: list[dict[str, Any]],
    *,
    manifest: dict[str, Any],
    prediction_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """根据启用的自适应策略校验 router_decisions 记录形态。"""
    aggregate_methods = {
        str(method_name)
        for method_name in manifest.get("aggregate_methods", [])
        if str(method_name).strip()
    }
    if not aggregate_methods:
        aggregate_methods = {
            str(row.get("method_name") or "")
            for row in prediction_rows
            if str(row.get("method_kind") or "") == "aggregate"
        }
    adaptive_policies = {
        method_name
        for method_name in aggregate_methods
        if method_name in ADAPTIVE_POLICY_METHODS
    }
    if not adaptive_policies:
        return _validate_empty_legacy_rows("router_decisions", rows)
    passed = all(
        str(row.get("policy_name") or "") in adaptive_policies
        and isinstance(row.get("triggered"), bool)
        and "selected_addon_solver" in row
        for row in rows
    )
    return {
        "passed": passed,
        "name": "router_decisions",
        "row_count": len(rows),
        "examples": rows[:20],
    }


def _validate_debate_messages(
    rows: list[dict[str, Any]],
    *,
    required: bool,
    path_exists: bool,
) -> dict[str, Any]:
    if not required:
        return {
            "passed": True,
            "name": "debate_messages",
            "required": False,
            "path_exists": path_exists,
            "row_count": len(rows),
            "examples": rows[:20],
        }
    required_fields = {"sender_agent_id", "recipient_agent_id", "sender_answer", "gate_reasons"}
    malformed = [
        row
        for row in rows
        if not required_fields.issubset(row)
    ]
    return {
        "passed": path_exists and not malformed,
        "name": "debate_messages",
        "required": True,
        "path_exists": path_exists,
        "row_count": len(rows),
        "malformed_count": len(malformed),
        "examples": rows[:20],
    }


def _validate_router_diagnostics(
    router_eval_payload: dict[str, Any],
    policy_diagnostics_payload: dict[str, Any],
    *,
    manifest: dict[str, Any],
    prediction_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    aggregate_methods = {
        str(method_name)
        for method_name in manifest.get("aggregate_methods", [])
        if str(method_name).strip()
    }
    if not aggregate_methods:
        aggregate_methods = {
            str(row.get("method_name") or "")
            for row in prediction_rows
            if str(row.get("method_kind") or "") == "aggregate"
        }
    adaptive_policies = {
        method_name
        for method_name in aggregate_methods
        if method_name in ADAPTIVE_POLICY_METHODS
    }
    if not adaptive_policies:
        return {"passed": True, "name": "router_diagnostics", "required": False}
    summary_rows = list(router_eval_payload.get("summary_rows", []))
    policy_rows = list(policy_diagnostics_payload.get("policy_rows", []))
    required_router_keys = {
        "stage_a_oracle_accuracy",
        "oracle_gap_vs_hetero",
        "oracle_gap_capture_by_preroute",
        "high_value_trigger_precision",
        "high_value_trigger_recall",
        "all_three_wrong_trigger_rate",
        "correct_to_wrong_rate_on_stage_a_correct",
        "stage_a_oracle_3core",
        "stage_a_oracle_5expert",
        "all_three_wrong_before_expansion_rate",
        "all_three_wrong_after_expansion_rate",
        "specialist_pair_override_precision",
        "arbiter_precision",
    }
    overall_rows = [
        row
        for row in summary_rows
        if row.get("dataset") == "overall" and str(row.get("policy_name") or "") in adaptive_policies
    ]
    malformed_rows = [row for row in overall_rows if not required_router_keys.issubset(set(row))]
    policy_router_fields_present = "router_summary_rows" in policy_diagnostics_payload and "router_bucket_rows" in policy_diagnostics_payload
    return {
        "passed": bool(overall_rows) and not malformed_rows and policy_router_fields_present,
        "name": "router_diagnostics",
        "overall_summary_count": len(overall_rows),
        "policy_row_count": len(policy_rows),
        "malformed_count": len(malformed_rows),
        "examples": malformed_rows[:10],
    }


def _manifest_includes_any_method(manifest: dict[str, Any], method_names: set[str]) -> bool:
    return bool(method_names & {str(item) for item in manifest.get("aggregate_methods", [])})


def _count_by_method(rows: list[dict[str, Any]]) -> dict[str, int]:
    """统计预测视图中每个方法的记录数。"""
    counts: dict[str, int] = {}
    for row in rows:
        method_name = str(row.get("method_name") or "")
        counts[method_name] = counts.get(method_name, 0) + 1
    return counts


def _diagnostic_path(
    diagnostic_paths: dict[str, Path],
    root: Path,
    filename: str,
) -> Path:
    """解析诊断文件路径，兼容旧 run 缺少索引登记的情况。"""
    return diagnostic_paths.get(filename, root / "diagnostics" / filename)
