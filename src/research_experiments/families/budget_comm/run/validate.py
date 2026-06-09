"""budget_comm 运行产物校验。

校验重点不是单纯检查文件是否齐全，而是约束研究口径：
预算超支、分片泄漏、配对设计完整性、背包选择可重放性，
以及 DALA-lite 档位分配一致性。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from research_experiments.families.budget_comm.algorithms import METHOD_ORDER, assign_density_tiers, solve_knapsack
from research_experiments.family_runtime.artifact_index import (
    named_diagnostic_paths,
    named_export_paths,
    named_turn_record_paths,
    resolve_run_artifact_index,
)
from research_experiments.family_runtime.validation import (
    load_json,
    load_jsonl_if_present,
    missing_relative_paths,
    summarize_turn_statuses,
    validate_rate_limit_check,
    validate_shared_contracts,
)


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    """Check budget_comm run directory key artifacts and experiment constraints."""
    index = resolve_run_artifact_index(run_dir, family_name="budget_comm")
    root = index.run_dir
    turn_paths = named_turn_record_paths(root, family_name="budget_comm")
    diagnostic_paths = named_diagnostic_paths(root, family_name="budget_comm")
    export_paths = named_export_paths(root, family_name="budget_comm")
    required_paths = [
        index.manifest_path,
        turn_paths["sample_views.jsonl"],
        turn_paths["stage_a_turns.jsonl"],
        turn_paths["candidate_packets.jsonl"],
        turn_paths["auction_decisions.jsonl"],
        turn_paths["belief_updates.jsonl"],
        index.prediction_records_path,
        index.metrics_view_path,
        diagnostic_paths["budget_diagnostics.json"],
        index.report_path,
        export_paths["paper_summary.csv"],
        index.figure_manifest_path,
        index.archive_manifest_path,
    ]
    missing = missing_relative_paths(root, required_paths)

    manifest = load_json(index.manifest_path)
    sample_views = load_jsonl_if_present(turn_paths["sample_views.jsonl"])
    stage_a_rows = load_jsonl_if_present(turn_paths["stage_a_turns.jsonl"])
    candidate_rows = load_jsonl_if_present(turn_paths["candidate_packets.jsonl"])
    auction_rows = load_jsonl_if_present(turn_paths["auction_decisions.jsonl"])
    belief_rows = load_jsonl_if_present(turn_paths["belief_updates.jsonl"])
    prediction_rows = load_jsonl_if_present(index.prediction_records_path)

    status_summary = summarize_turn_statuses(stage_a_rows + belief_rows)

    budget_check = _validate_budget_overrun(auction_rows)
    silent_check = _validate_silent_zero_tokens(candidate_rows)
    tier_check = _validate_dala_tier_match(candidate_rows)
    knapsack_check = _validate_knapsack_replay(auction_rows)
    paired_check = _validate_paired_design(prediction_rows)
    leak_check = _validate_context_leak(sample_views, manifest)
    shard_union_check = _validate_shard_union(sample_views, manifest)
    rate_limit_check = validate_rate_limit_check(index.progress_path, stage_a_rows + belief_rows, manifest=manifest)
    shared_contracts = validate_shared_contracts(root)
    figure_contract = shared_contracts["figure_contract"]
    archive_contract = shared_contracts["archive_contract"]

    passed = all(
        [
            not missing,
            status_summary["request_failures"] == 0,
            status_summary["schema_failures"] == 0,
            budget_check["passed"],
            silent_check["passed"],
            tier_check["passed"],
            knapsack_check["passed"],
            paired_check["passed"],
            leak_check["passed"],
            shard_union_check["passed"],
            rate_limit_check["passed"],
            figure_contract["passed"],
            archive_contract["passed"],
            bool(prediction_rows),
        ]
    )
    return {
        "run_dir": str(root),
        "passed": passed,
        "missing_files": missing,
        "request_failures": status_summary["request_failures"],
        "schema_failures": status_summary["schema_failures"],
        "checks": {
            "budget_overrun_check": budget_check,
            "silent_zero_token_check": silent_check,
            "dala_tier_match_check": tier_check,
            "knapsack_replay_check": knapsack_check,
            "paired_design_check": paired_check,
            "context_leak_check": leak_check,
            "shard_union_check": shard_union_check,
            "rate_limit_check": rate_limit_check,
            "figure_contract": figure_contract,
            "archive_contract": archive_contract,
        },
        "rate_limit_check": rate_limit_check,
        "methods": dict(Counter(row.get("method_name") for row in prediction_rows)),
    }


def _validate_budget_overrun(auction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Check whether any budget decision shows overrun."""
    violations = [
        {
            "dataset": row["dataset"],
            "sample_id": row["sample_id"],
            "method_name": row["method_name"],
            "total_cost": row["total_cost"],
            "round_budget_tokens": row["round_budget_tokens"],
        }
        for row in auction_rows
        if row.get("round_budget_tokens") is not None and float(row.get("total_cost") or 0.0) > float(row.get("round_budget_tokens") or 0.0)
    ]
    return {"passed": len(violations) == 0, "violation_count": len(violations), "violations": violations[:20]}


def _validate_silent_zero_tokens(candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Check that candidates marked as silence actually consume 0 tokens."""
    mismatches = [
        {
            "dataset": row["dataset"],
            "sample_id": row["sample_id"],
            "method_name": row["method_name"],
            "agent_id": row["agent_id"],
            "selected_packet_tokens": row["selected_packet_tokens"],
        }
        for row in candidate_rows
        if row.get("selected_mode") == "silence" and float(row.get("selected_packet_tokens") or 0.0) != 0.0
    ]
    return {"passed": len(mismatches) == 0, "mismatch_count": len(mismatches), "mismatches": mismatches[:20]}


def _validate_dala_tier_match(candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Replay density tier assignment and check consistency with logged values."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        if row.get("method_name") == "dala_lite":
            grouped[(str(row["dataset"]), str(row["sample_id"]))].append(row)
    mismatches: list[dict[str, Any]] = []
    for (dataset, sample_id), rows in grouped.items():
        tier_map = assign_density_tiers(
            {
                int(row["agent_id"]): float(row["density_score"])
                for row in rows
                if float(row["density_score"]) > 0.0
            }
        )
        for row in rows:
            expected = tier_map.get(int(row["agent_id"]), "silence")
            if float(row["density_score"]) <= 0.0:
                expected = "silence"
            if row.get("dala_assigned_mode") != expected:
                mismatches.append(
                    {
                        "dataset": dataset,
                        "sample_id": sample_id,
                        "agent_id": row["agent_id"],
                        "expected_mode": expected,
                        "observed_mode": row.get("dala_assigned_mode"),
                    }
                )
    return {"passed": len(mismatches) == 0, "mismatch_count": len(mismatches), "mismatches": mismatches[:20]}


def _validate_knapsack_replay(auction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Replay knapsack with candidate scores and costs, check winner set consistency."""
    mismatches: list[dict[str, Any]] = []
    for row in auction_rows:
        selection_rule = str(row.get("selection_rule"))
        round_budget_tokens = row.get("round_budget_tokens")
        if round_budget_tokens is None:
            continue
        if selection_rule not in {"knapsack_random_full", "knapsack_confidence_full", "knapsack_density_tiered"}:
            continue
        candidate_scores = row.get("candidate_scores", {})
        candidate_costs = row.get("candidate_costs", {})
        items = [
            {
                "agent_id": int(agent_id),
                "score": float(candidate_scores[agent_id]),
                "cost": int(candidate_costs.get(agent_id) or 0),
            }
            for agent_id in candidate_scores
            if int(candidate_costs.get(agent_id) or 0) > 0
        ]
        replay = solve_knapsack(items, int(round_budget_tokens))
        observed = tuple(sorted(int(agent_id) for agent_id in row.get("winner_agent_ids", [])))
        if observed != replay.winner_agent_ids:
            mismatches.append(
                {
                    "dataset": row["dataset"],
                    "sample_id": row["sample_id"],
                    "method_name": row["method_name"],
                    "observed": observed,
                    "expected": replay.winner_agent_ids,
                }
            )
    return {"passed": len(mismatches) == 0, "mismatch_count": len(mismatches), "mismatches": mismatches[:20]}


def _validate_paired_design(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Check that all methods are compared on the exact same sample set."""
    sample_sets = {
        method_name: {(row["dataset"], row["sample_id"]) for row in prediction_rows if row.get("method_name") == method_name}
        for method_name in METHOD_ORDER
    }
    reference = sample_sets.get(METHOD_ORDER[0], set())
    mismatches = [
        {
            "method_name": method_name,
            "missing_from_method": sorted(reference - sample_set)[:20],
            "extra_in_method": sorted(sample_set - reference)[:20],
        }
        for method_name, sample_set in sample_sets.items()
        if sample_set != reference
    ]
    return {"passed": len(mismatches) == 0, "mismatch_count": len(mismatches), "mismatches": mismatches}


def _validate_context_leak(sample_views: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    """Check whether split_context track accidentally leaks full context."""
    track_name = manifest.get("context_view", {}).get("track_name")
    if track_name != "split_context":
        return {"passed": True, "enabled": False}
    violations = [
        {
            "dataset": row["dataset"],
            "sample_id": row["sample_id"],
            "agent_id": row["agent_id"],
        }
        for row in sample_views
        if row.get("includes_full_context") or row.get("view_context_hash") == row.get("full_context_hash")
    ]
    return {"passed": len(violations) == 0, "enabled": True, "violation_count": len(violations), "violations": violations[:20]}


def _validate_shard_union(sample_views: list[dict[str, Any]], manifest: dict[str, Any]) -> dict[str, Any]:
    """Check that shard union covers required key information."""
    track_name = manifest.get("context_view", {}).get("track_name")
    if track_name != "split_context":
        return {"passed": True, "enabled": False}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sample_views:
        grouped[(str(row["dataset"]), str(row["sample_id"]))].append(row)
    violations = []
    for (dataset, sample_id), rows in grouped.items():
        required = set()
        covered = set()
        for row in rows:
            required.update(str(item) for item in row.get("required_coverage_items", []) if str(item).strip())
            covered.update(str(item) for item in row.get("coverage_items", []) if str(item).strip())
        if dataset in {"strategyqa", "hotpotqa"} and required and not required.issubset(covered):
            violations.append(
                {
                    "dataset": dataset,
                    "sample_id": sample_id,
                    "required": sorted(required),
                    "covered": sorted(covered),
                }
            )
    return {"passed": len(violations) == 0, "enabled": True, "violation_count": len(violations), "violations": violations[:20]}
