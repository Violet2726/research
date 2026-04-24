"""Free-MAD-lite 纯逻辑辅助函数。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from typing import Any


METHOD_ORDER = [
    "mv_3_initial",
    "vanilla_mad_r1_final_vote",
    "anti_conformity_final_vote",
    "free_mad_lite_llm_trajectory",
]


@dataclass(frozen=True)
class TrajectoryDecision:
    """轨迹裁决结果。"""

    final_answer: str
    selected_agent_id: int | None
    rationale: str
    fallback_used: bool
    fallback_reason: str | None


def majority_vote_with_counts(answers: list[str]) -> tuple[str, dict[str, int], bool]:
    """多数投票并返回是否共识。"""
    ordered = [answer for answer in answers if answer]
    counts = Counter(ordered)
    if not counts:
        return "", {}, False
    winner = max(counts.items(), key=lambda item: (item[1], -ordered.index(item[0])))[0]
    return winner, dict(counts), len(counts) == 1


def deterministic_trajectory_fallback(
    initial_rows: list[dict[str, Any]],
    anti_rows: list[dict[str, Any]],
) -> TrajectoryDecision:
    """当 LLM trajectory judge 失败时，用稳定规则回退。"""
    anti_answers = [str(row.get("normalized_answer") or "") for row in anti_rows]
    anti_vote, anti_counts, _ = majority_vote_with_counts(anti_answers)
    if anti_vote:
        selected = _first_agent_with_answer(anti_rows, anti_vote)
        return TrajectoryDecision(
            final_answer=anti_vote,
            selected_agent_id=selected,
            rationale="fallback_to_anti_conformity_majority",
            fallback_used=True,
            fallback_reason="judge_unavailable_or_invalid",
        )
    initial_answers = [str(row.get("normalized_answer") or "") for row in initial_rows]
    initial_vote, _, _ = majority_vote_with_counts(initial_answers)
    return TrajectoryDecision(
        final_answer=initial_vote,
        selected_agent_id=_first_agent_with_answer(initial_rows, initial_vote),
        rationale="fallback_to_initial_majority",
        fallback_used=True,
        fallback_reason="judge_unavailable_or_invalid",
    )


def build_trajectory_decision(
    judge_row: dict[str, Any],
    initial_rows: list[dict[str, Any]],
    anti_rows: list[dict[str, Any]],
) -> TrajectoryDecision:
    """把 judge 输出转换成裁决，失败时使用确定性 fallback。"""
    if judge_row.get("output_status") != "ok":
        return deterministic_trajectory_fallback(initial_rows, anti_rows)
    validated = judge_row.get("validated_output") or {}
    final_answer = str(validated.get("final_answer") or "").strip()
    if not final_answer:
        return deterministic_trajectory_fallback(initial_rows, anti_rows)
    selected_agent_id = validated.get("selected_agent_id")
    return TrajectoryDecision(
        final_answer=final_answer,
        selected_agent_id=int(selected_agent_id) if selected_agent_id is not None else None,
        rationale=str(validated.get("rationale") or "").strip(),
        fallback_used=False,
        fallback_reason=None,
    )


def approximate_token_count(text: str) -> int:
    """沿用仓库内轻量 token 估算。"""
    cleaned = text.strip()
    if not cleaned:
        return 0
    return max(1, len(cleaned) // 4)


def trajectory_hash(rows: list[dict[str, Any]]) -> str:
    """生成轨迹内容的稳定 JSON 表示，供 runner 再做哈希。"""
    payload = [
        {
            "agent_id": row.get("agent_id"),
            "round_index": row.get("round_index"),
            "role": row.get("role"),
            "normalized_answer": row.get("normalized_answer"),
            "validated_output": row.get("validated_output"),
        }
        for row in rows
    ]
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _first_agent_with_answer(rows: list[dict[str, Any]], answer: str) -> int | None:
    for row in rows:
        if str(row.get("normalized_answer") or "") == answer:
            return int(row["agent_id"])
    return None
