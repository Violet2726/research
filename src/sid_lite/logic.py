"""SID-lite 纯逻辑辅助函数。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from typing import Any


METHOD_ORDER = ["mv_3", "always_full", "compression_only", "sid_lite"]


@dataclass(frozen=True)
class SidEarlyExitDecision:
    """SID-lite 高置信早退判断结果。"""

    early_exit: bool
    triggered: bool
    reason: str
    mean_confidence: float | None
    confidence_spread: float | None
    any_invalid_confidence: bool
    initial_consensus: bool


def approximate_token_count(text: str) -> int:
    """沿用仓库内轻量 token 估算。"""
    cleaned = text.strip()
    if not cleaned:
        return 0
    return max(1, len(cleaned) // 4)


def trim_text_to_token_cap(text: object, token_cap: int) -> str:
    """把任意字段收敛到近似 token 上限。"""
    normalized = str(text or "").strip()
    if not normalized:
        return ""
    char_cap = max(1, token_cap * 4)
    if len(normalized) <= char_cap:
        return normalized
    return normalized[: max(1, char_cap - 3)].rstrip() + "..."


def decide_early_exit(
    stage_a_rows: list[dict[str, Any]],
    *,
    mean_conf_threshold: float,
    conf_spread_threshold: float,
) -> SidEarlyExitDecision:
    """按 SID-lite 的高置信一致规则判断是否跳过通信。"""
    answers = [str(row.get("normalized_answer") or "").strip() for row in stage_a_rows]
    non_empty_answers = [answer for answer in answers if answer]
    initial_consensus = bool(non_empty_answers) and len(set(non_empty_answers)) == 1 and len(non_empty_answers) == len(stage_a_rows)
    valid_confidences = [
        float(row["confidence_value"])
        for row in stage_a_rows
        if row.get("confidence_valid") and row.get("confidence_value") is not None
    ]
    any_invalid_confidence = len(valid_confidences) != len(stage_a_rows)
    mean_confidence = round(sum(valid_confidences) / len(valid_confidences), 6) if valid_confidences else None
    confidence_spread = round(max(valid_confidences) - min(valid_confidences), 6) if len(valid_confidences) >= 2 else (0.0 if valid_confidences else None)

    if not initial_consensus:
        reason = "answer_disagreement_or_empty"
        early_exit = False
    elif any_invalid_confidence:
        # 置信度不可用时 fail-open 到通信，避免把解析失败误判为高置信。
        reason = "invalid_confidence_fail_open"
        early_exit = False
    elif mean_confidence is None or mean_confidence < mean_conf_threshold:
        reason = "low_mean_confidence"
        early_exit = False
    elif confidence_spread is not None and confidence_spread > conf_spread_threshold:
        reason = "high_confidence_spread"
        early_exit = False
    else:
        reason = "early_exit_by_high_confidence_consensus"
        early_exit = True

    return SidEarlyExitDecision(
        early_exit=early_exit,
        triggered=not early_exit,
        reason=reason,
        mean_confidence=mean_confidence,
        confidence_spread=confidence_spread,
        any_invalid_confidence=any_invalid_confidence,
        initial_consensus=initial_consensus,
    )


def project_message_packet(stage_a_row: dict[str, Any], *, mode: str, token_cap: int) -> dict[str, Any]:
    """把 Stage A 输出投影为 full 或 compressed 可通信消息。"""
    if mode not in {"full", "compressed"}:
        raise ValueError(f"Unsupported SID-lite packet mode: {mode}")

    final_answer = trim_text_to_token_cap(stage_a_row.get("normalized_answer"), 24)
    confidence_value = stage_a_row.get("confidence_value") if stage_a_row.get("confidence_valid") else None
    fields: dict[str, Any] = {"final_answer": final_answer}
    if confidence_value is not None:
        fields["confidence_raw"] = confidence_value

    if mode == "full":
        fields.update(
            {
                "reasoning_trace": trim_text_to_token_cap(stage_a_row.get("reasoning_trace"), 128),
                "claim_span": trim_text_to_token_cap(stage_a_row.get("claim_span"), 48),
                "key_evidence": trim_text_to_token_cap(stage_a_row.get("key_evidence"), 64),
                "uncertain_point": trim_text_to_token_cap(stage_a_row.get("uncertain_point"), 32),
            }
        )
        primary_keys = ["reasoning_trace", "key_evidence", "claim_span", "uncertain_point"]
    else:
        fields.update(
            {
                "semantic_focus": trim_text_to_token_cap(
                    stage_a_row.get("claim_span")
                    or stage_a_row.get("key_evidence")
                    or stage_a_row.get("uncertain_point"),
                    56,
                ),
                "uncertain_point": trim_text_to_token_cap(stage_a_row.get("uncertain_point"), 24),
            }
        )
        primary_keys = ["semantic_focus", "uncertain_point"]

    fitted = _fit_packet_to_cap(_drop_empty(fields), token_cap, primary_keys)
    packet_text = json.dumps(fitted, ensure_ascii=False, sort_keys=True)
    return {
        "agent_id": int(stage_a_row["agent_id"]),
        "packet_mode": mode,
        "packet_fields": fitted,
        "packet_text": packet_text,
        "approx_packet_tokens": approximate_token_count(packet_text),
        "token_cap": int(token_cap),
        "final_answer": fitted.get("final_answer", ""),
        "confidence_value": confidence_value,
    }


def apply_belief_update(stage_a_row: dict[str, Any], belief_row: dict[str, Any]) -> dict[str, Any]:
    """把 belief update 应用到候选答案，schema 失败时保留原答案。"""
    validated = belief_row.get("validated_output", {}) if belief_row.get("output_status") == "ok" else {}
    previous_answer = str(stage_a_row.get("normalized_answer") or "")
    changed_answer = bool(validated.get("changed_answer")) if validated else False
    new_answer = str(validated.get("new_answer") or previous_answer).strip() if validated else previous_answer
    confidence_delta = validated.get("confidence_delta") if validated else None
    previous_confidence = stage_a_row.get("confidence_value")
    if stage_a_row.get("confidence_valid") and previous_confidence is not None:
        updated_confidence = float(previous_confidence)
        if confidence_delta is not None:
            updated_confidence = min(1.0, max(0.0, updated_confidence + float(confidence_delta)))
        updated_confidence = round(updated_confidence, 6)
    else:
        updated_confidence = None
    return {
        "agent_id": int(stage_a_row["agent_id"]),
        "previous_answer": previous_answer,
        "final_answer": new_answer,
        "normalized_answer": new_answer,
        "changed_answer": changed_answer,
        "confidence_value": updated_confidence,
        "confidence_valid": updated_confidence is not None,
        "reason_for_change": str(validated.get("reason_for_change") or "").strip() if validated else "",
        "remaining_disagreement": str(validated.get("remaining_disagreement") or "").strip() if validated else "",
    }


def compression_ratio(compressed_packets: list[dict[str, Any]], full_packets: list[dict[str, Any]]) -> float | None:
    """计算 compressed 相对 full 的通信 token 比例。"""
    full_tokens = sum(int(row.get("approx_packet_tokens") or 0) for row in full_packets)
    if full_tokens <= 0:
        return None
    compressed_tokens = sum(int(row.get("approx_packet_tokens") or 0) for row in compressed_packets)
    return round(compressed_tokens / full_tokens, 6)


def majority_vote_with_counts(answers: list[str]) -> tuple[str, dict[str, int], bool]:
    """多数投票并返回是否共识。"""
    ordered = [answer for answer in answers if answer]
    counts = Counter(ordered)
    if not counts:
        return "", {}, False
    winner = max(counts.items(), key=lambda item: (item[1], -ordered.index(item[0])))[0]
    return winner, dict(counts), len(counts) == 1


def _fit_packet_to_cap(fields: dict[str, Any], token_cap: int, primary_keys: list[str]) -> dict[str, Any]:
    fitted = dict(fields)
    while approximate_token_count(json.dumps(fitted, ensure_ascii=False, sort_keys=True)) > token_cap and primary_keys:
        trimmed = False
        for key in primary_keys:
            value = str(fitted.get(key) or "").strip()
            if not value:
                continue
            next_value = value[:-8].rstrip()
            fitted[key] = next_value + "..." if next_value else ""
            trimmed = True
            fitted = _drop_empty(fitted)
            if approximate_token_count(json.dumps(fitted, ensure_ascii=False, sort_keys=True)) <= token_cap:
                break
        if not trimmed:
            break
    return _drop_empty(fitted)


def _drop_empty(fields: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in fields.items()
        if value is not None and value != "" and value != []
    }
