"""SPARC 的机制级纯逻辑函数。

这里集中实现 SPARC 的核心算法：消息包投影与退化、belief update 合并、
带置信度的聚合、审计候选选择，以及局部审计器可见的最小候选包构造。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any
import json


MESSAGE_MODE_ORDER = [
    "full_cot",
    "answer_only",
    "answer_confidence",
    "disagreement_step_only",
    "critical_evidence_only",
]
FIELD_TOKEN_CAPS = {
    "reasoning_trace": 160,
    "claim_span": 48,
    "key_evidence": 64,
    "uncertain_point": 32,
    "answer_only": 24,
    "answer_confidence": 32,
}
MESSAGE_MODE_TOKEN_CAPS = {
    "full_cot": 192,
    "answer_only": 24,
    "answer_confidence": 32,
    "disagreement_step_only": 80,
    "critical_evidence_only": 80,
}
DEFAULT_MESSAGE_MODE_BY_DATASET = {
    "gsm8k": "disagreement_step_only",
    "strategyqa": "disagreement_step_only",
    "hotpotqa": "critical_evidence_only",
}
AGGREGATION_METHOD_ORDER = [
    "majority_vote",
    "single_judge",
    "final_round_vote",
    "local_auditing",
]


def approximate_token_count(text: str) -> int:
    """沿用项目内的轻量 token 估算。"""
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


def project_message_packet(stage_a_row: dict[str, Any], requested_mode: str) -> dict[str, Any]:
    """把 solver 输出投影成一个可发送消息包，并执行显式退化规则。"""
    final_answer = trim_text_to_token_cap(
        str(stage_a_row.get("validated_output", {}).get("final_answer", "")).strip(),
        FIELD_TOKEN_CAPS["answer_only"],
    )
    confidence_raw_display = stage_a_row.get("confidence_raw_display", "")
    confidence_valid = bool(stage_a_row.get("confidence_valid"))
    reasoning_trace = trim_text_to_token_cap(stage_a_row.get("reasoning_trace", ""), FIELD_TOKEN_CAPS["reasoning_trace"])
    claim_span = trim_text_to_token_cap(stage_a_row.get("claim_span", ""), FIELD_TOKEN_CAPS["claim_span"])
    key_evidence = trim_text_to_token_cap(stage_a_row.get("key_evidence", ""), FIELD_TOKEN_CAPS["key_evidence"])

    effective_mode = requested_mode
    degradation_reason = ""
    if requested_mode == "answer_confidence" and (not confidence_valid or confidence_raw_display in {"", None}):
        effective_mode = "answer_only"
        degradation_reason = "invalid_confidence"
    elif requested_mode == "disagreement_step_only" and not claim_span:
        if confidence_valid and confidence_raw_display not in {"", None}:
            effective_mode = "answer_confidence"
            degradation_reason = "missing_claim_span"
        else:
            effective_mode = "answer_only"
            degradation_reason = "missing_claim_span_and_invalid_confidence"
    elif requested_mode == "critical_evidence_only" and not key_evidence:
        if confidence_valid and confidence_raw_display not in {"", None}:
            effective_mode = "answer_confidence"
            degradation_reason = "missing_key_evidence"
        else:
            effective_mode = "answer_only"
            degradation_reason = "missing_key_evidence_and_invalid_confidence"

    packet_fields: dict[str, Any] = {"final_answer": final_answer}
    if effective_mode == "full_cot":
        if reasoning_trace:
            packet_fields["reasoning_trace"] = reasoning_trace
        if confidence_valid and confidence_raw_display not in {"", None}:
            packet_fields["confidence_raw"] = confidence_raw_display
        if key_evidence:
            packet_fields["key_evidence"] = key_evidence
    elif effective_mode == "answer_confidence":
        packet_fields["confidence_raw"] = confidence_raw_display
    elif effective_mode == "disagreement_step_only":
        packet_fields["claim_span"] = claim_span
    elif effective_mode == "critical_evidence_only":
        packet_fields["key_evidence"] = key_evidence

    packet_fields = _fit_packet_to_mode_cap(packet_fields, effective_mode)
    packet_text = json.dumps(packet_fields, ensure_ascii=False, sort_keys=True)
    return {
        "requested_message_mode": requested_mode,
        "effective_message_mode": effective_mode,
        "degradation_reason": degradation_reason or None,
        "packet_fields": packet_fields,
        "packet_text": packet_text,
        "approx_packet_tokens": approximate_token_count(packet_text),
        "final_answer": packet_fields.get("final_answer", ""),
        "confidence_raw_display": packet_fields.get("confidence_raw", ""),
        "reasoning_trace": packet_fields.get("reasoning_trace", ""),
        "claim_span": packet_fields.get("claim_span", ""),
        "key_evidence": packet_fields.get("key_evidence", ""),
        "confidence_value": stage_a_row.get("confidence_value"),
        "confidence_valid": stage_a_row.get("confidence_valid"),
    }


def apply_belief_update(
    *,
    stage_a_row: dict[str, Any],
    message_packet: dict[str, Any],
    belief_row: dict[str, Any],
) -> dict[str, Any]:
    """把 belief update 应用到 Stage A 候选上，生成 post-debate candidate。"""
    validated = belief_row.get("validated_output", {}) if belief_row.get("output_status") == "ok" else {}
    changed_answer = bool(validated.get("changed_answer")) if validated else False
    previous_answer = stage_a_row.get("normalized_answer", "")
    new_answer = str(validated.get("new_answer", previous_answer)).strip() if validated else str(previous_answer)
    confidence_delta = validated.get("confidence_delta") if validated else None
    previous_confidence = stage_a_row.get("confidence_value")
    previous_valid = bool(stage_a_row.get("confidence_valid"))
    if previous_valid and previous_confidence is not None:
        if confidence_delta is None:
            updated_confidence = float(previous_confidence)
        else:
            updated_confidence = min(1.0, max(0.0, float(previous_confidence) + float(confidence_delta)))
        updated_confidence = round(updated_confidence, 6)
        updated_confidence_valid = True
    else:
        updated_confidence = None
        updated_confidence_valid = False

    claim_span = message_packet.get("claim_span") or stage_a_row.get("claim_span", "")
    remaining_disagreement = str(validated.get("remaining_disagreement") or "").strip() if validated else ""
    if remaining_disagreement and message_packet.get("effective_message_mode") in {"full_cot", "disagreement_step_only"}:
        claim_span = trim_text_to_token_cap(remaining_disagreement, FIELD_TOKEN_CAPS["claim_span"])

    return {
        "agent_id": stage_a_row["agent_id"],
        "changed_answer": changed_answer,
        "final_answer": new_answer,
        "normalized_answer": new_answer,
        "previous_answer": previous_answer,
        "confidence_value": updated_confidence,
        "confidence_valid": updated_confidence_valid,
        "confidence_raw_display": updated_confidence if updated_confidence_valid else stage_a_row.get("confidence_raw_display", ""),
        "confidence_delta": confidence_delta,
        "reason_for_change": str(validated.get("reason_for_change") or "").strip() if validated else "",
        "remaining_disagreement": remaining_disagreement,
        "claim_span": claim_span,
        "key_evidence": message_packet.get("key_evidence") or stage_a_row.get("key_evidence", ""),
        "reasoning_trace": message_packet.get("reasoning_trace") or stage_a_row.get("reasoning_trace", ""),
        "requested_message_mode": message_packet.get("requested_message_mode"),
        "effective_message_mode": message_packet.get("effective_message_mode"),
        "degradation_reason": message_packet.get("degradation_reason"),
        "stage_a_agent_id": stage_a_row["agent_id"],
    }


def aggregate_with_confidence_tiebreak(candidates: list[dict[str, Any]]) -> tuple[str, dict[str, int]]:
    """在平票时按更高 confidence、再按更小 agent_id 决定。"""
    ordered_answers = [str(candidate.get("normalized_answer", "")).strip() for candidate in candidates if candidate.get("normalized_answer")]
    counts = Counter(ordered_answers)
    if not counts:
        return "", {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        answer = str(candidate.get("normalized_answer", "")).strip()
        if answer:
            grouped[answer].append(candidate)

    def _rank(answer: str) -> tuple[int, float, int]:
        items = grouped[answer]
        best_confidence = max(
            (
                float(item.get("confidence_value"))
                for item in items
                if item.get("confidence_valid") and item.get("confidence_value") is not None
            ),
            default=-1.0,
        )
        min_agent_id = min(int(item.get("agent_id", 10**6)) for item in items)
        return counts[answer], best_confidence, -min_agent_id

    winner = max(counts, key=_rank)
    return winner, dict(counts)


def select_audit_candidate_pair(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """为 local auditing 选择最多两个候选。"""
    valid_candidates = [candidate for candidate in candidates if candidate.get("normalized_answer")]
    answer_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in valid_candidates:
        answer_groups[str(candidate["normalized_answer"])].append(candidate)
    if len(answer_groups) <= 1:
        only_answer = next(iter(answer_groups), "")
        return {
            "skipped": True,
            "skip_reason": "consensus",
            "pair_type": "consensus",
            "candidate_a": None,
            "candidate_b": None,
            "fallback_answer": only_answer,
        }

    if len(answer_groups) == 2 and sorted(len(items) for items in answer_groups.values()) == [1, 2]:
        majority_answer = next(answer for answer, items in answer_groups.items() if len(items) == 2)
        minority_answer = next(answer for answer, items in answer_groups.items() if len(items) == 1)
        return {
            "skipped": False,
            "skip_reason": None,
            "pair_type": "two_way_majority",
            "candidate_a": _best_candidate(answer_groups[majority_answer]),
            "candidate_b": _best_candidate(answer_groups[minority_answer]),
            "fallback_answer": majority_answer,
        }

    ranked = sorted(
        valid_candidates,
        key=lambda item: (
            1 if item.get("confidence_valid") and item.get("confidence_value") is not None else 0,
            float(item.get("confidence_value") or -1.0),
            -int(item.get("agent_id", 10**6)),
        ),
        reverse=True,
    )
    fallback_answer, _ = aggregate_with_confidence_tiebreak(valid_candidates)
    return {
        "skipped": False,
        "skip_reason": None,
        "pair_type": "three_way",
        "candidate_a": ranked[0],
        "candidate_b": ranked[1],
        "fallback_answer": fallback_answer,
    }


def build_prompt_packet(candidate: dict[str, Any]) -> dict[str, Any]:
    """抽出可安全暴露给 judge / auditor 的最小候选包。"""
    return {
        "agent_id": candidate.get("agent_id"),
        "final_answer": candidate.get("final_answer") or candidate.get("normalized_answer") or "",
        "confidence_value": candidate.get("confidence_value"),
        "claim_span": candidate.get("claim_span") or None,
        "key_evidence": candidate.get("key_evidence") or None,
        "reason_for_change": candidate.get("reason_for_change") or None,
    }


def _fit_packet_to_mode_cap(packet_fields: dict[str, Any], effective_mode: str) -> dict[str, Any]:
    cap = MESSAGE_MODE_TOKEN_CAPS[effective_mode]
    fields = dict(packet_fields)
    primary_keys = {
        "full_cot": ["reasoning_trace", "key_evidence"],
        "disagreement_step_only": ["claim_span"],
        "critical_evidence_only": ["key_evidence"],
    }.get(effective_mode, [])
    while approximate_token_count(json.dumps(fields, ensure_ascii=False, sort_keys=True)) > cap and primary_keys:
        trimmed = False
        for key in primary_keys:
            value = str(fields.get(key, "")).strip()
            if not value:
                continue
            next_value = value[:-8].rstrip()
            fields[key] = next_value + "..." if next_value else ""
            trimmed = True
            if approximate_token_count(json.dumps(fields, ensure_ascii=False, sort_keys=True)) <= cap:
                break
        if not trimmed:
            break
    return fields


def _best_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        candidates,
        key=lambda item: (
            1 if item.get("confidence_valid") and item.get("confidence_value") is not None else 0,
            float(item.get("confidence_value") or -1.0),
            -int(item.get("agent_id", 10**6)),
        ),
        reverse=True,
    )[0]
