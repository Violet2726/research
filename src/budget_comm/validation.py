"""`budget_comm` 运行结果校验。

本模块从研究约束而非单纯文件完整性的角度验证一次运行：
既检查关键产物是否齐全，也检查预算是否超支、分片是否泄漏、配对设计是否被破坏，
以及 DALA-lite 的 tier 分配与背包选择是否可以被重放。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
from typing import Any

from budget_comm.logic import METHOD_ORDER, assign_density_tiers, solve_knapsack
from experiment_core.reporting.run_figures import validate_figure_contract


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    """检查 `budget_comm` 运行目录的关键产物与实验约束。"""
    root = Path(run_dir)
    required = [
        "manifest.json",
        "sample_views.jsonl",
        "stage_a_turns.jsonl",
        "candidate_packets.jsonl",
        "auction_decisions.jsonl",
        "belief_updates.jsonl",
        "final_predictions.jsonl",
        "metrics.json",
        "budget_diagnostics.json",
        "progress.json",
        "report.md",
        "paper_summary.csv",
        "figure_manifest.json",
    ]
    missing = [name for name in required if not (root / name).exists()]

    manifest = _load_json(root / "manifest.json")
    sample_views = _load_jsonl(root / "sample_views.jsonl")
    stage_a_rows = _load_jsonl(root / "stage_a_turns.jsonl")
    candidate_rows = _load_jsonl(root / "candidate_packets.jsonl")
    auction_rows = _load_jsonl(root / "auction_decisions.jsonl")
    belief_rows = _load_jsonl(root / "belief_updates.jsonl")
    prediction_rows = _load_jsonl(root / "final_predictions.jsonl")

    turn_rows = stage_a_rows + belief_rows
    request_failures = sum(1 for row in turn_rows if row.get("output_status") == "request_fail")
    schema_failures = sum(1 for row in turn_rows if row.get("output_status") == "schema_fail")

    budget_check = _validate_budget_overrun(auction_rows)
    silent_check = _validate_silent_zero_tokens(candidate_rows)
    tier_check = _validate_dala_tier_match(candidate_rows)
    knapsack_check = _validate_knapsack_replay(auction_rows)
    paired_check = _validate_paired_design(prediction_rows)
    leak_check = _validate_context_leak(sample_views, manifest)
    shard_union_check = _validate_shard_union(sample_views, manifest)
    figure_contract = validate_figure_contract(root)

    passed = all(
        [
            not missing,
            request_failures == 0,
            schema_failures == 0,
            budget_check["passed"],
            silent_check["passed"],
            tier_check["passed"],
            knapsack_check["passed"],
            paired_check["passed"],
            leak_check["passed"],
            shard_union_check["passed"],
            figure_contract["passed"],
            bool(prediction_rows),
        ]
    )
    return {
        "run_dir": str(root),
        "passed": passed,
        "missing_files": missing,
        "checks": {
            "request_failures_total": request_failures,
            "schema_failures_total": schema_failures,
            "budget_overrun_check": budget_check,
            "silent_zero_token_check": silent_check,
            "dala_tier_match_check": tier_check,
            "knapsack_replay_check": knapsack_check,
            "paired_design_check": paired_check,
            "context_leak_check": leak_check,
            "shard_union_check": shard_union_check,
            "figure_contract": figure_contract,
        },
        "methods": dict(Counter(row.get("method_name") for row in prediction_rows)),
    }


def _validate_budget_overrun(auction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """检查任何一条预算决策是否出现超支。"""
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
    """检查被标记为 `silence` 的候选是否真的消耗 0 token。"""
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
    """回放 density tier 分配，检查与日志中的档位是否一致。"""
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
    """用候选分数与成本重放 knapsack，检查赢家集合是否一致。"""
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
    """检查所有方法是否在完全相同的样本集合上比较。"""
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
    """检查 `split_context` 轨道是否意外泄漏完整上下文。"""
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
    """检查各分片并集是否覆盖了设计上必须暴露的关键信息。"""
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


def _load_json(path: Path) -> dict[str, Any]:
    """读取 UTF-8 JSON；不存在时返回空字典。"""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 UTF-8 JSONL；不存在时返回空列表。"""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
