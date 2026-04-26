"""`budget_comm` 的机制级纯逻辑函数。

这里承载 DALA-lite 相关的核心算法，但不直接依赖 I/O：
包括消息包压缩、value density 特征构造、tier 分配、背包求解、
VCG 诊断支付、belief update 合并，以及是否值得进入 full DALA 的门槛判断。
这使得核心机制可以被单测、validation 与 runner 共同复用。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
import random
from typing import Any

from experiment_core.selective_signals import normalize_confidence


METHOD_ORDER = [
    "mv_3",
    "all_to_all_full",
    "budget_random",
    "budget_confidence",
    "dala_lite",
]
PACKET_MODE_ORDER = ["full", "summary", "keywords", "silence"]


@dataclass(frozen=True)
class KnapsackDecision:
    """一次预算子集选择的结果。"""

    winner_agent_ids: tuple[int, ...]
    total_score: float
    total_cost: int


def approximate_token_count(text: str) -> int:
    """沿用仓库中的轻量 token 估算。"""
    cleaned = text.strip()
    if not cleaned:
        return 0
    return max(1, len(cleaned) // 4)


def trim_text_to_token_cap(text: str | None, token_cap: int) -> str:
    """按近似 token cap 截断文本。"""
    normalized = (text or "").strip()
    if not normalized:
        return ""
    char_cap = max(1, token_cap * 4)
    if len(normalized) <= char_cap:
        return normalized
    return normalized[: max(1, char_cap - 3)].rstrip() + "..."


def normalize_keyword_clues(value: object) -> list[str]:
    """把 `keyword_clues` 收敛为稳定的字符串列表。"""
    if isinstance(value, str):
        normalized = value.strip()
        return [normalized] if normalized else []
    if not isinstance(value, list):
        return []
    clues: list[str] = []
    for item in value:
        normalized = str(item).strip()
        if normalized:
            clues.append(normalized)
    return clues


def build_shared_candidate_features(stage_a_rows: list[dict[str, Any]], auction_policy) -> list[dict[str, Any]]:
    """从共享 Stage A 输出构造 value density 所需特征。

    这是 DALA-lite 的关键入口：它会把每个 agent 的初始答案、证据、关键词和置信度
    转换成可比较的候选表示，并进一步计算 utility、density 与预分配消息档位。
    """
    answer_counts = Counter(str(row.get("normalized_answer") or "") for row in stage_a_rows if row.get("normalized_answer"))
    keyword_sets = {
        int(row["agent_id"]): _keyword_set(normalize_keyword_clues(row.get("keyword_clues")))
        for row in stage_a_rows
    }

    candidates: list[dict[str, Any]] = []
    utility_scores: list[float] = []
    for row in stage_a_rows:
        # 先计算与答案分歧、证据充分性、新颖性、置信度相关的局部特征，
        # 再按配置权重合成为 utility 分数。
        packet_variants = build_packet_variants(row, auction_policy)
        disagreement_score = _disagreement_component(str(row.get("normalized_answer") or ""), answer_counts)
        evidence_score = _evidence_component(row)
        novelty_score = _novelty_component(int(row["agent_id"]), keyword_sets)
        confidence_score = _confidence_component(row.get("confidence_raw"))
        utility_score = round(
            auction_policy.disagreement_weight * disagreement_score
            + auction_policy.evidence_weight * evidence_score
            + auction_policy.novelty_weight * novelty_score
            + auction_policy.confidence_weight * confidence_score,
            6,
        )
        utility_scores.append(utility_score)
        candidates.append(
            {
                "agent_id": int(row["agent_id"]),
                "normalized_answer": str(row.get("normalized_answer") or ""),
                "keyword_clues": normalize_keyword_clues(row.get("keyword_clues")),
                "keyword_set": sorted(keyword_sets.get(int(row["agent_id"]), set())),
                "confidence_value": row.get("confidence_value"),
                "confidence_valid": bool(row.get("confidence_valid")),
                "confidence_score": confidence_score,
                "disagreement_score": disagreement_score,
                "evidence_score": evidence_score,
                "novelty_score": novelty_score,
                "utility_score": utility_score,
                "packet_variants": packet_variants,
            }
        )

    mean_utility = sum(utility_scores) / len(utility_scores) if utility_scores else 0.0
    variance = sum((score - mean_utility) ** 2 for score in utility_scores) / len(utility_scores) if utility_scores else 0.0
    std_utility = math.sqrt(variance)
    density_values = []
    for candidate in candidates:
        # DALA-lite 使用“标准化效用 / Full 包成本”的近似 value density。
        zscore = (candidate["utility_score"] - mean_utility) / std_utility if std_utility > 0 else 0.0
        full_tokens = candidate["packet_variants"]["full"]["packet_tokens"] or 1
        density_score = round(zscore / full_tokens, 6)
        candidate["density_score"] = density_score
        density_values.append(density_score)

    tier_map = assign_density_tiers(
        {
            candidate["agent_id"]: float(candidate["density_score"])
            for candidate in candidates
            if float(candidate["density_score"]) > auction_policy.positive_density_threshold
        }
    )
    for candidate in candidates:
        # 只有正 density 候选才有资格获得某种消息档位，否则直接静默。
        density_score = float(candidate["density_score"])
        assigned_mode = tier_map.get(candidate["agent_id"], "silence")
        if density_score <= auction_policy.positive_density_threshold:
            assigned_mode = "silence"
        candidate["dala_assigned_mode"] = assigned_mode
        candidate["dala_candidate_cost"] = 0 if assigned_mode == "silence" else int(candidate["packet_variants"][assigned_mode]["packet_tokens"])
    return candidates


def build_packet_variants(stage_a_row: dict[str, Any], auction_policy) -> dict[str, dict[str, Any]]:
    """把 Stage A solver 输出投影为三档消息包。

    三档分别对应：
    - `full`：较完整的推理与证据；
    - `summary`：保留争议 claim 与关键证据；
    - `keywords`：只保留答案与关键词线索。
    """
    final_answer = trim_text_to_token_cap(str(stage_a_row.get("normalized_answer") or ""), 24)
    reasoning_trace = trim_text_to_token_cap(stage_a_row.get("reasoning_trace"), 128)
    claim_span = trim_text_to_token_cap(stage_a_row.get("claim_span"), 48)
    key_evidence = trim_text_to_token_cap(stage_a_row.get("key_evidence"), 64)
    keyword_clues = [trim_text_to_token_cap(item, 8) for item in normalize_keyword_clues(stage_a_row.get("keyword_clues"))]

    variants = {
        "full": {"final_answer": final_answer, "reasoning_trace": reasoning_trace, "key_evidence": key_evidence},
        "summary": {"final_answer": final_answer, "claim_span": claim_span, "key_evidence": key_evidence},
        "keywords": {"final_answer": final_answer, "keyword_clues": keyword_clues},
    }
    primary_keys = {
        "full": ["reasoning_trace", "key_evidence"],
        "summary": ["claim_span", "key_evidence"],
        "keywords": ["keyword_clues"],
    }
    token_caps = {
        "full": int(auction_policy.full_token_cap),
        "summary": int(auction_policy.summary_token_cap),
        "keywords": int(auction_policy.keywords_token_cap),
    }
    rendered: dict[str, dict[str, Any]] = {}
    for mode, packet_fields in variants.items():
        normalized_fields = {key: value for key, value in packet_fields.items() if _has_non_empty_content(value)}
        fitted_fields = _fit_packet_to_cap(normalized_fields, token_caps[mode], primary_keys[mode])
        packet_text = json.dumps(fitted_fields, ensure_ascii=False, sort_keys=True)
        rendered[mode] = {
            "packet_fields": fitted_fields,
            "packet_text": packet_text,
            "packet_tokens": approximate_token_count(packet_text),
        }
    return rendered


def assign_density_tiers(positive_densities: dict[int, float]) -> dict[int, str]:
    """按正密度候选的相对位置映射 `full / summary / keywords`。"""
    if not positive_densities:
        return {}
    ordered = sorted(positive_densities.items(), key=lambda item: (item[1], item[0]))
    if len(ordered) == 1:
        return {ordered[0][0]: "full"}
    tier_map: dict[int, str] = {}
    denominator = max(1, len(ordered) - 1)
    for rank, (agent_id, _) in enumerate(ordered):
        position = rank / denominator
        if position <= (1.0 / 3.0):
            tier_map[agent_id] = "keywords"
        elif position <= (2.0 / 3.0):
            tier_map[agent_id] = "summary"
        else:
            tier_map[agent_id] = "full"
    return tier_map


def build_all_to_all_full_decision(shared_candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """构造 `all_to_all_full` 决策，即固定广播所有 Full 消息。"""
    selected_modes = {candidate["agent_id"]: "full" for candidate in shared_candidates if candidate["packet_variants"]["full"]["packet_tokens"] > 0}
    candidate_rows = _materialize_candidate_rows(
        shared_candidates,
        method_name="all_to_all_full",
        selected_modes=selected_modes,
        selection_rule="broadcast_all_full",
        round_budget_tokens=None,
        score_lookup={candidate["agent_id"]: 1.0 for candidate in shared_candidates},
        vcg_payments={candidate["agent_id"]: 0.0 for candidate in shared_candidates},
    )
    total_cost = sum(int(row["selected_packet_tokens"]) for row in candidate_rows)
    return {
        "method_name": "all_to_all_full",
        "selection_rule": "broadcast_all_full",
        "round_budget_tokens": None,
        "winner_agent_ids": sorted(selected_modes),
        "winner_modes": {str(agent_id): mode for agent_id, mode in selected_modes.items()},
        "total_score": float(len(selected_modes)),
        "total_cost": total_cost,
        "budget_utilization": None,
        "vcg_payments": {str(candidate["agent_id"]): 0.0 for candidate in shared_candidates},
        "candidate_rows": candidate_rows,
    }


def build_budget_random_decision(
    shared_candidates: list[dict[str, Any]],
    *,
    round_budget_tokens: int,
    seed: int,
) -> dict[str, Any]:
    """构造 `budget_random` 决策，即随机打分的 Full-only 预算基线。"""
    rng = random.Random(seed)
    score_lookup = {candidate["agent_id"]: round(rng.random(), 6) for candidate in shared_candidates}
    items = [
        {
            "agent_id": candidate["agent_id"],
            "score": score_lookup[candidate["agent_id"]],
            "cost": int(candidate["packet_variants"]["full"]["packet_tokens"]),
        }
        for candidate in shared_candidates
        if int(candidate["packet_variants"]["full"]["packet_tokens"]) > 0
    ]
    decision = solve_knapsack(items, round_budget_tokens)
    selected_modes = {agent_id: "full" for agent_id in decision.winner_agent_ids}
    candidate_rows = _materialize_candidate_rows(
        shared_candidates,
        method_name="budget_random",
        selected_modes=selected_modes,
        selection_rule="knapsack_random_full",
        round_budget_tokens=round_budget_tokens,
        score_lookup=score_lookup,
        vcg_payments={candidate["agent_id"]: 0.0 for candidate in shared_candidates},
    )
    return {
        "method_name": "budget_random",
        "selection_rule": "knapsack_random_full",
        "round_budget_tokens": round_budget_tokens,
        "winner_agent_ids": list(decision.winner_agent_ids),
        "winner_modes": {str(agent_id): "full" for agent_id in decision.winner_agent_ids},
        "total_score": round(decision.total_score, 6),
        "total_cost": decision.total_cost,
        "budget_utilization": round(decision.total_cost / round_budget_tokens, 6) if round_budget_tokens else 0.0,
        "vcg_payments": {str(candidate["agent_id"]): 0.0 for candidate in shared_candidates},
        "candidate_rows": candidate_rows,
    }


def build_budget_confidence_decision(
    shared_candidates: list[dict[str, Any]],
    *,
    round_budget_tokens: int,
) -> dict[str, Any]:
    """构造 `budget_confidence` 决策，即按置信度排序的 Full-only 预算基线。"""
    score_lookup = {
        candidate["agent_id"]: float(candidate["confidence_score"])
        for candidate in shared_candidates
    }
    items = [
        {
            "agent_id": candidate["agent_id"],
            "score": score_lookup[candidate["agent_id"]],
            "cost": int(candidate["packet_variants"]["full"]["packet_tokens"]),
        }
        for candidate in shared_candidates
        if int(candidate["packet_variants"]["full"]["packet_tokens"]) > 0
    ]
    decision = solve_knapsack(items, round_budget_tokens)
    selected_modes = {agent_id: "full" for agent_id in decision.winner_agent_ids}
    candidate_rows = _materialize_candidate_rows(
        shared_candidates,
        method_name="budget_confidence",
        selected_modes=selected_modes,
        selection_rule="knapsack_confidence_full",
        round_budget_tokens=round_budget_tokens,
        score_lookup=score_lookup,
        vcg_payments={candidate["agent_id"]: 0.0 for candidate in shared_candidates},
    )
    return {
        "method_name": "budget_confidence",
        "selection_rule": "knapsack_confidence_full",
        "round_budget_tokens": round_budget_tokens,
        "winner_agent_ids": list(decision.winner_agent_ids),
        "winner_modes": {str(agent_id): "full" for agent_id in decision.winner_agent_ids},
        "total_score": round(decision.total_score, 6),
        "total_cost": decision.total_cost,
        "budget_utilization": round(decision.total_cost / round_budget_tokens, 6) if round_budget_tokens else 0.0,
        "vcg_payments": {str(candidate["agent_id"]): 0.0 for candidate in shared_candidates},
        "candidate_rows": candidate_rows,
    }


def build_dala_lite_decision(
    shared_candidates: list[dict[str, Any]],
    *,
    round_budget_tokens: int,
    positive_density_threshold: float,
) -> dict[str, Any]:
    """构造 `dala_lite` 决策。

    该方法先根据 density 决定每个 agent 的候选消息档位，
    再在预算约束下用背包求解器选出赢家集合。
    """
    items = []
    for candidate in shared_candidates:
        density_score = float(candidate["density_score"])
        assigned_mode = str(candidate["dala_assigned_mode"])
        if density_score <= positive_density_threshold or assigned_mode == "silence":
            continue
        items.append(
            {
                "agent_id": candidate["agent_id"],
                "score": density_score,
                "cost": int(candidate["packet_variants"][assigned_mode]["packet_tokens"]),
            }
        )
    decision = solve_knapsack(items, round_budget_tokens)
    selected_modes = {
        candidate["agent_id"]: str(candidate["dala_assigned_mode"])
        for candidate in shared_candidates
        if candidate["agent_id"] in decision.winner_agent_ids
    }
    score_lookup = {
        candidate["agent_id"]: float(candidate["density_score"])
        for candidate in shared_candidates
    }
    vcg_payments = compute_vcg_payments(items, decision, budget_tokens=round_budget_tokens)
    candidate_rows = _materialize_candidate_rows(
        shared_candidates,
        method_name="dala_lite",
        selected_modes=selected_modes,
        selection_rule="knapsack_density_tiered",
        round_budget_tokens=round_budget_tokens,
        score_lookup=score_lookup,
        vcg_payments=vcg_payments,
    )
    return {
        "method_name": "dala_lite",
        "selection_rule": "knapsack_density_tiered",
        "round_budget_tokens": round_budget_tokens,
        "winner_agent_ids": list(decision.winner_agent_ids),
        "winner_modes": {str(agent_id): mode for agent_id, mode in selected_modes.items()},
        "total_score": round(decision.total_score, 6),
        "total_cost": decision.total_cost,
        "budget_utilization": round(decision.total_cost / round_budget_tokens, 6) if round_budget_tokens else 0.0,
        "vcg_payments": {str(agent_id): payment for agent_id, payment in sorted(vcg_payments.items())},
        "candidate_rows": candidate_rows,
    }


def solve_knapsack(items: list[dict[str, Any]], budget_tokens: int) -> KnapsackDecision:
    """用穷举法求解小规模 knapsack，并实现固定 tie-break。

    当前实验中的候选数量很小，因此这里优先选择可解释、可重放的穷举法，
    便于 validation 对拍和论文级诊断。
    """
    best = KnapsackDecision(winner_agent_ids=tuple(), total_score=0.0, total_cost=0)
    candidate_count = len(items)
    for mask in range(1 << candidate_count):
        winner_ids: list[int] = []
        total_score = 0.0
        total_cost = 0
        for index in range(candidate_count):
            if not (mask & (1 << index)):
                continue
            total_cost += int(items[index]["cost"])
            if total_cost > budget_tokens:
                break
            total_score += float(items[index]["score"])
            winner_ids.append(int(items[index]["agent_id"]))
        else:
            normalized_ids = tuple(sorted(winner_ids))
            if _is_better_decision(
                score=round(total_score, 12),
                cost=total_cost,
                winner_ids=normalized_ids,
                current=best,
            ):
                best = KnapsackDecision(
                    winner_agent_ids=normalized_ids,
                    total_score=round(total_score, 12),
                    total_cost=total_cost,
                )
    return best


def compute_vcg_payments(
    items: list[dict[str, Any]],
    decision: KnapsackDecision,
    *,
    budget_tokens: int,
) -> dict[int, float]:
    """计算 DALA-lite 的诊断性 VCG payment。"""
    item_by_agent = {int(item["agent_id"]): item for item in items}
    payments = {int(item["agent_id"]): 0.0 for item in items}
    for winner_agent_id in decision.winner_agent_ids:
        filtered_items = [item for item in items if int(item["agent_id"]) != winner_agent_id]
        best_without_winner = solve_knapsack(filtered_items, budget_tokens)
        current_other_value = sum(
            float(item_by_agent[agent_id]["score"])
            for agent_id in decision.winner_agent_ids
            if agent_id != winner_agent_id
        )
        payments[winner_agent_id] = round(best_without_winner.total_score - current_other_value, 6)
    return payments


def apply_belief_update(
    *,
    stage_a_row: dict[str, Any],
    belief_row: dict[str, Any],
) -> dict[str, Any]:
    """把 Stage B belief update 应用到 Stage A 候选。

    如果 belief update 失败或缺字段，这里会保守回退到 Stage A 的原始答案，
    避免由于格式问题把机制收益误判为通信收益。
    """
    validated = belief_row.get("validated_output", {}) if belief_row.get("output_status") == "ok" else {}
    changed_answer = bool(validated.get("changed_answer")) if validated else False
    previous_answer = str(stage_a_row.get("normalized_answer") or "")
    new_answer = str(validated.get("new_answer") or previous_answer).strip() if validated else previous_answer
    confidence_delta = validated.get("confidence_delta") if validated else None
    previous_confidence = stage_a_row.get("confidence_value")
    previous_valid = bool(stage_a_row.get("confidence_valid"))
    if previous_valid and previous_confidence is not None:
        updated_confidence = float(previous_confidence)
        if confidence_delta is not None:
            updated_confidence = min(1.0, max(0.0, updated_confidence + float(confidence_delta)))
        updated_confidence = round(updated_confidence, 6)
    else:
        updated_confidence = None
    return {
        "agent_id": stage_a_row["agent_id"],
        "changed_answer": changed_answer,
        "final_answer": new_answer,
        "normalized_answer": new_answer,
        "previous_answer": previous_answer,
        "confidence_value": updated_confidence,
        "confidence_valid": updated_confidence is not None,
        "reason_for_change": str(validated.get("reason_for_change") or "").strip() if validated else "",
        "remaining_disagreement": str(validated.get("remaining_disagreement") or "").strip() if validated else "",
    }


def evaluate_full_dala_gate(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """根据整体 summary 评估是否值得进入 full DALA。"""
    by_method = {row["method_name"]: row for row in summary_rows if row.get("dataset") == "overall"}
    dala_row = by_method.get("dala_lite")
    full_row = by_method.get("all_to_all_full")
    random_row = by_method.get("budget_random")
    confidence_row = by_method.get("budget_confidence")
    if not all([dala_row, full_row, random_row, confidence_row]):
        return {
            "ready_for_full_dala": False,
            "reason": "missing_overall_rows",
            "conditions": {},
        }
    acc_per_1k_vs_random = float(dala_row["acc_per_1k_tokens"]) > float(random_row["acc_per_1k_tokens"])
    acc_per_1k_vs_confidence = float(dala_row["acc_per_1k_tokens"]) > float(confidence_row["acc_per_1k_tokens"])
    accuracy_gap = float(full_row["accuracy_mean"]) - float(dala_row["accuracy_mean"])
    communication_ratio = (
        float(dala_row["communication_tokens_mean"]) / float(full_row["communication_tokens_mean"])
        if float(full_row["communication_tokens_mean"])
        else math.inf
    )
    conditions = {
        "dala_beats_budget_random_on_acc_per_1k": acc_per_1k_vs_random,
        "dala_beats_budget_confidence_on_acc_per_1k": acc_per_1k_vs_confidence,
        "dala_accuracy_gap_vs_all_to_all_full_le_3pp": accuracy_gap <= 0.03,
        "dala_communication_le_60pct_of_all_to_all_full": communication_ratio <= 0.60,
    }
    return {
        "ready_for_full_dala": all(conditions.values()),
        "reason": "all_conditions_met" if all(conditions.values()) else "gate_not_met",
        "conditions": conditions,
        "accuracy_gap_vs_all_to_all_full": round(accuracy_gap, 6),
        "communication_ratio_vs_all_to_all_full": round(communication_ratio, 6) if math.isfinite(communication_ratio) else None,
    }


def _materialize_candidate_rows(
    shared_candidates: list[dict[str, Any]],
    *,
    method_name: str,
    selected_modes: dict[int, str],
    selection_rule: str,
    round_budget_tokens: int | None,
    score_lookup: dict[int, float],
    vcg_payments: dict[int, float],
) -> list[dict[str, Any]]:
    """把候选、选择分数与最终中标结果展开成可落盘的诊断行。"""
    rows: list[dict[str, Any]] = []
    for candidate in shared_candidates:
        agent_id = int(candidate["agent_id"])
        selected_mode = selected_modes.get(agent_id, "silence")
        is_winner = agent_id in selected_modes
        packet_variant = candidate["packet_variants"].get(selected_mode if selected_mode != "silence" else "full")
        if selection_rule in {"broadcast_all_full", "knapsack_random_full", "knapsack_confidence_full"}:
            candidate_cost = int(candidate["packet_variants"]["full"]["packet_tokens"])
        elif selection_rule == "knapsack_density_tiered":
            assigned_mode = str(candidate["dala_assigned_mode"])
            candidate_cost = 0 if assigned_mode == "silence" else int(candidate["packet_variants"][assigned_mode]["packet_tokens"])
        else:
            candidate_cost = 0
        rows.append(
            {
                "method_name": method_name,
                "agent_id": agent_id,
                "normalized_answer": candidate["normalized_answer"],
                "keyword_clues": candidate["keyword_clues"],
                "keyword_set": candidate["keyword_set"],
                "confidence_value": candidate["confidence_value"],
                "confidence_valid": candidate["confidence_valid"],
                "confidence_score": candidate["confidence_score"],
                "disagreement_score": candidate["disagreement_score"],
                "evidence_score": candidate["evidence_score"],
                "novelty_score": candidate["novelty_score"],
                "utility_score": candidate["utility_score"],
                "density_score": candidate["density_score"],
                "dala_assigned_mode": candidate["dala_assigned_mode"],
                "selection_rule": selection_rule,
                "selection_score": round(float(score_lookup.get(agent_id, 0.0)), 6),
                "round_budget_tokens": round_budget_tokens,
                "candidate_cost": candidate_cost,
                "is_winner": is_winner,
                "selected_mode": selected_mode,
                "selected_packet_tokens": 0 if selected_mode == "silence" else int(packet_variant["packet_tokens"]),
                "selected_packet_text": "" if selected_mode == "silence" else str(packet_variant["packet_text"]),
                "full_packet_tokens": int(candidate["packet_variants"]["full"]["packet_tokens"]),
                "summary_packet_tokens": int(candidate["packet_variants"]["summary"]["packet_tokens"]),
                "keywords_packet_tokens": int(candidate["packet_variants"]["keywords"]["packet_tokens"]),
                "full_packet_text": str(candidate["packet_variants"]["full"]["packet_text"]),
                "summary_packet_text": str(candidate["packet_variants"]["summary"]["packet_text"]),
                "keywords_packet_text": str(candidate["packet_variants"]["keywords"]["packet_text"]),
                "vcg_payment": round(float(vcg_payments.get(agent_id, 0.0)), 6),
            }
        )
    return rows


def _disagreement_component(answer: str, answer_counts: Counter[str]) -> float:
    """计算答案分歧分量：少数观点比多数观点更有潜在通信价值。"""
    if not answer or not answer_counts:
        return 0.0
    if len(answer_counts) == 1:
        return 0.0
    if answer_counts[answer] == 1:
        return 1.0
    return 0.35


def _evidence_component(stage_a_row: dict[str, Any]) -> float:
    """计算证据分量：有关键证据和可争议 claim 的候选更值得发送。"""
    has_evidence = bool(str(stage_a_row.get("key_evidence") or "").strip())
    has_claim = bool(str(stage_a_row.get("claim_span") or "").strip()) or bool(normalize_keyword_clues(stage_a_row.get("keyword_clues")))
    if has_evidence and has_claim:
        return 1.0
    if has_evidence or has_claim:
        return 0.5
    return 0.0


def _novelty_component(agent_id: int, keyword_sets: dict[int, set[str]]) -> float:
    """计算新颖性分量：关键词越不与同伴重合，分数越高。"""
    current = keyword_sets.get(agent_id, set())
    if not current:
        return 0.0
    peer_overlaps = []
    for peer_id, peer_set in keyword_sets.items():
        if peer_id == agent_id:
            continue
        peer_overlaps.append(_jaccard(current, peer_set))
    if not peer_overlaps:
        return 1.0
    return round(1.0 - max(peer_overlaps), 6)


def _confidence_component(raw_confidence: object) -> float:
    """计算置信度分量；无效置信度回退到中性分数。"""
    value, valid, _ = normalize_confidence(raw_confidence)
    if valid and value is not None:
        return float(value)
    return 0.5


def _fit_packet_to_cap(packet_fields: dict[str, Any], token_cap: int, primary_keys: list[str]) -> dict[str, Any]:
    """在保留主要字段语义的前提下，把消息包裁剪到 token 上限内。"""
    fields = json.loads(json.dumps(packet_fields, ensure_ascii=False))
    while approximate_token_count(json.dumps(fields, ensure_ascii=False, sort_keys=True)) > token_cap and primary_keys:
        trimmed = False
        for key in primary_keys:
            value = fields.get(key)
            if isinstance(value, str) and value:
                next_value = value[:-8].rstrip()
                fields[key] = next_value + "..." if next_value else ""
                trimmed = True
            elif isinstance(value, list) and value:
                last_item = str(value[-1]).strip()
                if len(value) > 1:
                    value.pop()
                    trimmed = True
                elif last_item:
                    next_value = last_item[:-4].rstrip()
                    value[-1] = next_value + "..." if next_value else ""
                    trimmed = True
            fields = {field_key: field_value for field_key, field_value in fields.items() if _has_non_empty_content(field_value)}
            if approximate_token_count(json.dumps(fields, ensure_ascii=False, sort_keys=True)) <= token_cap:
                break
        if not trimmed:
            break
    return {key: value for key, value in fields.items() if _has_non_empty_content(value)}


def _has_non_empty_content(value: object) -> bool:
    """判断某个字段是否仍然包含可传递内容。"""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return any(str(item).strip() for item in value)
    return True


def _keyword_set(keyword_clues: list[str]) -> set[str]:
    """把关键词线索列表转换成归一化后的 token 集合。"""
    normalized: set[str] = set()
    for clue in keyword_clues:
        for token in clue.lower().replace(",", " ").split():
            token = token.strip()
            if token:
                normalized.add(token)
    return normalized


def _jaccard(left: set[str], right: set[str]) -> float:
    """计算两个关键词集合的 Jaccard 相似度。"""
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return round(len(left & right) / len(union), 6)


def _is_better_decision(
    *,
    score: float,
    cost: int,
    winner_ids: tuple[int, ...],
    current: KnapsackDecision,
) -> bool:
    """比较两个 knapsack 候选解，按得分、成本、ID 顺序稳定决策。"""
    if score > current.total_score + 1e-12:
        return True
    if math.isclose(score, current.total_score, abs_tol=1e-12) and cost < current.total_cost:
        return True
    if math.isclose(score, current.total_score, abs_tol=1e-12) and cost == current.total_cost and winner_ids < current.winner_agent_ids:
        return True
    return False
