"""ECON 的 belief 构造、动作打分与受控聚合逻辑。"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


ACTION_ORDER = ("keep_local", "adopt_vote", "query_best_peer", "query_two_peers")


def approximate_token_count(text: str) -> int:
    """沿用仓内轻量 token 估算规则。"""

    cleaned = text.strip()
    if not cleaned:
        return 0
    return max(1, len(cleaned) // 4)


def build_peer_packet(stage_a_row: dict[str, Any], token_cap: int) -> dict[str, Any]:
    """把 Stage A 行压缩成可传播的 peer packet。"""

    packet_fields = {
        "final_answer": str(stage_a_row.get("final_answer") or stage_a_row.get("normalized_answer") or "").strip(),
        "confidence": stage_a_row.get("confidence_value"),
        "claim_span": str(stage_a_row.get("claim_span") or "").strip(),
        "key_evidence": str(stage_a_row.get("key_evidence") or "").strip(),
        "reasoning_trace": str(stage_a_row.get("reasoning_trace") or "").strip(),
        "keyword_clues": [str(item).strip() for item in (stage_a_row.get("keyword_clues") or []) if str(item).strip()][:2],
    }
    packet_text = _trim_packet_text(packet_fields, token_cap)
    return {
        "agent": f"agent_{stage_a_row['agent_id']}",
        "agent_id": int(stage_a_row["agent_id"]),
        "final_answer": packet_fields["final_answer"],
        "confidence": packet_fields["confidence"],
        "claim_span": packet_fields["claim_span"],
        "key_evidence": packet_fields["key_evidence"],
        "reasoning_trace": packet_fields["reasoning_trace"],
        "keyword_clues": packet_fields["keyword_clues"],
        "packet_text": packet_text,
        "approx_packet_tokens": approximate_token_count(packet_text),
    }


def build_belief_state(
    stage_a_rows: list[dict[str, Any]],
    *,
    protocol: Any,
) -> dict[str, Any]:
    """从 Stage A 输出构造 sample 级 belief state。"""

    normalized_answers = [str(row.get("normalized_answer") or "").strip() for row in stage_a_rows if row.get("normalized_answer")]
    confidences = [
        float(row.get("confidence_value"))
        for row in stage_a_rows
        if row.get("confidence_valid") and row.get("confidence_value") is not None
    ]
    vote_answer, vote_counts = aggregate_majority_with_counts(stage_a_rows)
    answer_counter = Counter(normalized_answers)
    max_count = max(answer_counter.values(), default=0)
    second_count = sorted(answer_counter.values(), reverse=True)[1] if len(answer_counter) >= 2 else 0
    agreement_ratio = round(max_count / max(1, len(stage_a_rows)), 6)
    disagreement_ratio = round(1.0 - agreement_ratio, 6)
    confidence_dispersion = round((max(confidences) - min(confidences)) if len(confidences) >= 2 else 0.0, 6)
    rationale_overlap = pairwise_token_jaccard_mean(
        [
            " | ".join(
                [
                    str(row.get("claim_span") or "").strip(),
                    str(row.get("key_evidence") or "").strip(),
                    str(row.get("reasoning_trace") or "").strip(),
                ]
            ).strip()
            for row in stage_a_rows
        ]
    )
    rationale_conflict = round(1.0 - rationale_overlap, 6)
    majority_margin = round((max_count - second_count) / max(1, len(stage_a_rows)), 6)
    expected_gain = round(
        min(
            1.0,
            float(protocol.disagreement_weight) * disagreement_ratio
            + float(protocol.confidence_dispersion_weight) * confidence_dispersion
            + float(protocol.rationale_conflict_weight) * rationale_conflict
            + float(protocol.expected_gain_weight) * (1.0 - majority_margin),
        ),
        6,
    )

    sample_packet_rows = [build_peer_packet(row, protocol.peer_packet_token_cap) for row in stage_a_rows]
    packet_lookup = {int(row["agent_id"]): row for row in sample_packet_rows}
    best_packet_cost = max((int(row["approx_packet_tokens"]) for row in sample_packet_rows), default=0)
    two_packet_cost = sum(sorted((int(row["approx_packet_tokens"]) for row in sample_packet_rows), reverse=True)[:2])

    action_scores = _score_actions(
        agreement_ratio=agreement_ratio,
        disagreement_ratio=disagreement_ratio,
        confidence_dispersion=confidence_dispersion,
        rationale_conflict=rationale_conflict,
        expected_gain=expected_gain,
        best_packet_cost=best_packet_cost,
        two_packet_cost=two_packet_cost,
        protocol=protocol,
    )
    selected_action = max(action_scores, key=lambda row: (float(row["belief_score"]), -ACTION_ORDER.index(str(row["action"]))))["action"]
    return {
        "initial_vote_answer": vote_answer,
        "initial_vote_counts": vote_counts,
        "initial_vote_support": agreement_ratio,
        "agreement_ratio": agreement_ratio,
        "disagreement_ratio": disagreement_ratio,
        "confidence_dispersion": confidence_dispersion,
        "rationale_overlap": rationale_overlap,
        "rationale_conflict": rationale_conflict,
        "majority_margin": majority_margin,
        "expected_gain": expected_gain,
        "action_scores": action_scores,
        "selected_action": selected_action,
        "sample_packet_rows": sample_packet_rows,
        "packet_lookup": packet_lookup,
    }


def pick_action_packets(
    stage_a_rows: list[dict[str, Any]],
    packet_lookup: dict[int, dict[str, Any]],
    *,
    selected_action: str,
) -> dict[int, list[dict[str, Any]]]:
    """按动作类型为每个 agent 选 peer packets。"""

    packets_by_agent: dict[int, list[dict[str, Any]]] = {}
    ordered_rows = sorted(stage_a_rows, key=lambda row: int(row["agent_id"]))
    for row in ordered_rows:
        agent_id = int(row["agent_id"])
        peers = [peer for peer in ordered_rows if int(peer["agent_id"]) != agent_id]
        ranked_peers = sorted(
            peers,
            key=lambda peer: (
                str(peer.get("normalized_answer") or "") != str(row.get("normalized_answer") or ""),
                float(peer.get("confidence_value") or 0.0),
            ),
            reverse=True,
        )
        if selected_action == "query_best_peer":
            selected = ranked_peers[:1]
        elif selected_action == "query_two_peers":
            selected = ranked_peers[:2]
        elif selected_action == "econ_full_comm_r1":
            selected = peers
        else:
            selected = []
        packets_by_agent[agent_id] = [packet_lookup[int(peer["agent_id"])] for peer in selected]
    return packets_by_agent


def aggregate_majority_with_counts(rows: list[dict[str, Any]]) -> tuple[str, dict[str, int]]:
    """按多数投票聚合 Stage A 候选。"""

    answers = [str(row.get("normalized_answer") or "").strip() for row in rows if str(row.get("normalized_answer") or "").strip()]
    counts = Counter(answers)
    if not counts:
        return "", {}
    winner = max(
        counts,
        key=lambda answer: (
            counts[answer],
            max(
                float(row.get("confidence_value") or 0.0)
                for row in rows
                if str(row.get("normalized_answer") or "").strip() == answer
            ),
            -min(
                int(row.get("agent_id") or 10**6)
                for row in rows
                if str(row.get("normalized_answer") or "").strip() == answer
            ),
        ),
    )
    return winner, dict(counts)


def aggregate_confidence_weighted(rows: list[dict[str, Any]]) -> tuple[str, dict[str, float]]:
    """按置信度加权聚合候选。"""

    grouped_weights: dict[str, float] = defaultdict(float)
    grouped_counts: dict[str, int] = defaultdict(int)
    best_confidence: dict[str, float] = defaultdict(float)
    min_agent_id: dict[str, int] = defaultdict(lambda: 10**6)
    for row in rows:
        answer = str(row.get("normalized_answer") or "").strip()
        if not answer:
            continue
        confidence = float(row.get("confidence_value") or 0.5) if row.get("confidence_valid") else 0.5
        grouped_weights[answer] += confidence
        grouped_counts[answer] += 1
        best_confidence[answer] = max(best_confidence[answer], confidence)
        min_agent_id[answer] = min(min_agent_id[answer], int(row.get("agent_id") or 10**6))
    if not grouped_weights:
        return "", {}
    winner = max(
        grouped_weights,
        key=lambda answer: (
            grouped_weights[answer],
            grouped_counts[answer],
            best_confidence[answer],
            -min_agent_id[answer],
        ),
    )
    return winner, {answer: round(value, 6) for answer, value in grouped_weights.items()}


def apply_belief_answer_safeguard(
    *,
    stage_a_row: dict[str, Any],
    belief_row: dict[str, Any],
    selected_peer_packets: list[dict[str, Any]],
) -> dict[str, Any]:
    """把 belief update 结果安全合并回候选状态。"""

    previous_answer = str(stage_a_row.get("normalized_answer") or "")
    previous_confidence = stage_a_row.get("confidence_value")
    previous_valid = bool(stage_a_row.get("confidence_valid"))
    validated = belief_row.get("validated_output", {}) if belief_row.get("output_status") == "ok" else {}
    changed_answer = bool(validated.get("changed_answer")) if validated else False
    new_answer = str(validated.get("new_answer") or previous_answer).strip()
    if not selected_peer_packets:
        changed_answer = False
        new_answer = previous_answer
    if changed_answer and not new_answer:
        changed_answer = False
        new_answer = previous_answer

    confidence_delta = validated.get("confidence_delta") if validated else None
    if previous_valid and previous_confidence is not None:
        if confidence_delta is None:
            updated_confidence = float(previous_confidence)
        else:
            updated_confidence = min(1.0, max(0.0, float(previous_confidence) + float(confidence_delta)))
        confidence_valid = True
        confidence_value = round(updated_confidence, 6)
    else:
        confidence_valid = False
        confidence_value = None
    return {
        "agent_id": int(stage_a_row["agent_id"]),
        "final_answer": new_answer,
        "normalized_answer": new_answer,
        "changed_answer": changed_answer,
        "previous_answer": previous_answer,
        "confidence_value": confidence_value,
        "confidence_valid": confidence_valid,
        "confidence_delta": confidence_delta,
        "reason_for_change": str(validated.get("reason_for_change") or "").strip() if validated else "",
        "remaining_disagreement": str(validated.get("remaining_disagreement") or "").strip() if validated else "",
        "claim_span": str(stage_a_row.get("claim_span") or "").strip(),
        "key_evidence": str(stage_a_row.get("key_evidence") or "").strip(),
        "reasoning_trace": str(stage_a_row.get("reasoning_trace") or "").strip(),
        "keyword_clues": [str(item).strip() for item in (stage_a_row.get("keyword_clues") or []) if str(item).strip()][:2],
    }


def pairwise_token_jaccard_mean(values: list[str]) -> float:
    """计算多段文本两两之间的 token Jaccard 平均值。"""

    cleaned = [value.strip() for value in values if value.strip()]
    if len(cleaned) <= 1:
        return 1.0
    token_sets = [_tokenize(value) for value in cleaned]
    scores: list[float] = []
    for left_index in range(len(token_sets)):
        for right_index in range(left_index + 1, len(token_sets)):
            scores.append(_token_jaccard(token_sets[left_index], token_sets[right_index]))
    if not scores:
        return 1.0
    return round(sum(scores) / len(scores), 6)


def _score_actions(
    *,
    agreement_ratio: float,
    disagreement_ratio: float,
    confidence_dispersion: float,
    rationale_conflict: float,
    expected_gain: float,
    best_packet_cost: int,
    two_packet_cost: int,
    protocol: Any,
) -> list[dict[str, Any]]:
    """为有限动作集打分。"""

    best_cost = min(1.0, best_packet_cost / max(1.0, float(protocol.communication_cost_divisor)))
    two_cost = min(1.0, two_packet_cost / max(1.0, float(protocol.communication_cost_divisor)))
    keep_score = round(agreement_ratio * 0.35 - disagreement_ratio * 0.30 - rationale_conflict * 0.05, 6)
    vote_score = round(
        float(protocol.vote_bonus)
        + agreement_ratio * 0.25
        + (1.0 - confidence_dispersion) * 0.05
        - disagreement_ratio * 0.20
        - rationale_conflict * 0.15,
        6,
    )
    query_best_score = round(
        float(protocol.query_best_peer_bonus)
        + expected_gain
        + disagreement_ratio * 0.25
        + confidence_dispersion * 0.15
        + rationale_conflict * 0.15
        - float(protocol.communication_cost_weight) * best_cost * 0.5,
        6,
    )
    query_two_score = round(
        float(protocol.query_two_peers_bonus)
        + expected_gain
        + disagreement_ratio * 0.20
        + rationale_conflict * 0.20
        - float(protocol.communication_cost_weight) * two_cost * 0.6,
        6,
    )
    return [
        {
            "action": "keep_local",
            "belief_score": keep_score,
            "expected_gain": 0.0,
            "communication_cost": 0.0,
        },
        {
            "action": "adopt_vote",
            "belief_score": vote_score,
            "expected_gain": round(max(0.0, expected_gain * 0.5), 6),
            "communication_cost": 0.0,
        },
        {
            "action": "query_best_peer",
            "belief_score": query_best_score,
            "expected_gain": expected_gain,
            "communication_cost": round(best_cost, 6),
        },
        {
            "action": "query_two_peers",
            "belief_score": query_two_score,
            "expected_gain": round(min(1.0, expected_gain * 1.05), 6),
            "communication_cost": round(two_cost, 6),
        },
    ]


def _trim_packet_text(packet_fields: dict[str, Any], token_cap: int) -> str:
    """把 peer packet 压缩到近似 token 上限。"""

    lines = [
        f"answer={packet_fields['final_answer']}",
        f"confidence={packet_fields['confidence']}",
        f"claim={packet_fields['claim_span']}",
        f"evidence={packet_fields['key_evidence']}",
    ]
    if packet_fields["keyword_clues"]:
        lines.append("keywords=" + ", ".join(packet_fields["keyword_clues"]))
    if packet_fields["reasoning_trace"]:
        lines.append(f"reasoning={packet_fields['reasoning_trace']}")
    text = "\n".join(line for line in lines if line.strip()).strip()
    words = text.split()
    while words and approximate_token_count(" ".join(words)) > token_cap:
        words.pop()
    return " ".join(words).strip()


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in text.split() if token.strip()}


def _token_jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)
