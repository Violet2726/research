from __future__ import annotations

from collections import Counter, defaultdict
from math import log
from typing import Any
import json

from cue.schemas import ConflictObject, UtilityBreakdown


def approximate_token_count(text: str) -> int:
    cleaned = text.strip()
    if not cleaned:
        return 0
    return max(1, len(cleaned) // 4)


def flatten_short_list(values: list[str] | None, *, fallback: str = "") -> str:
    cleaned = [str(item).strip() for item in (values or []) if str(item).strip()]
    if cleaned:
        return " | ".join(cleaned[:2])
    return fallback.strip()


def pairwise_token_jaccard_mean(values: list[str]) -> float:
    if len(values) <= 1:
        return 1.0
    token_sets = [_tokenize(value) for value in values]
    scores: list[float] = []
    for left_index in range(len(token_sets)):
        for right_index in range(left_index + 1, len(token_sets)):
            scores.append(_token_jaccard(token_sets[left_index], token_sets[right_index]))
    if not scores:
        return 1.0
    return round(sum(scores) / len(scores), 6)


def summarize_cue_signals(stage_a_rows: list[dict[str, Any]], message_token_cap: int) -> dict[str, float | int | bool | None]:
    normalized_answers = [str(row.get("normalized_answer") or "").strip() for row in stage_a_rows if row.get("normalized_answer")]
    raw_answers = [str(row.get("final_answer") or "").strip() for row in stage_a_rows if row.get("final_answer")]
    confidences = [
        float(row["confidence_value"])
        for row in stage_a_rows
        if row.get("confidence_valid") and row.get("confidence_value") is not None
    ]
    mean_confidence = round(sum(confidences) / len(confidences), 6) if confidences else None
    if len(confidences) >= 2:
        confidence_spread = round(max(confidences) - min(confidences), 6)
    elif len(confidences) == 1:
        confidence_spread = 0.0
    else:
        confidence_spread = None

    answer_entropy = _normalized_entropy(normalized_answers)
    claim_texts = [flatten_short_list(row.get("top_claims"), fallback=str(row.get("reasoning_sketch") or "")) for row in stage_a_rows]
    evidence_texts = [flatten_short_list(row.get("evidence_items"), fallback=str(row.get("final_answer") or "")) for row in stage_a_rows]
    claim_conflict_rate = round(1.0 - pairwise_token_jaccard_mean(claim_texts), 6)
    evidence_gap = round(1.0 - pairwise_token_jaccard_mean(evidence_texts), 6)
    initial_disagreement = len(set(answer for answer in normalized_answers if answer)) > 1
    fragile_consensus = _fragile_consensus(
        initial_disagreement=initial_disagreement,
        mean_confidence=mean_confidence,
        claim_conflict_rate=claim_conflict_rate,
        evidence_gap=evidence_gap,
    )
    format_conflict_risk = _format_conflict_risk(raw_answers, normalized_answers)
    majority_pressure_risk = _majority_pressure_risk(stage_a_rows)
    any_invalid_confidence = any(not row.get("confidence_valid") for row in stage_a_rows)
    estimated_comm_cost = float(message_token_cap * max(1, len(stage_a_rows)))
    return {
        "initial_disagreement": initial_disagreement,
        "answer_entropy": answer_entropy,
        "mean_confidence": mean_confidence,
        "confidence_spread": confidence_spread,
        "claim_conflict_rate": claim_conflict_rate,
        "evidence_gap": evidence_gap,
        "fragile_consensus": fragile_consensus,
        "format_conflict_risk": format_conflict_risk,
        "majority_pressure_risk": majority_pressure_risk,
        "any_invalid_confidence": any_invalid_confidence,
        "estimated_comm_cost": estimated_comm_cost,
    }


def build_conflict_object(stage_a_rows: list[dict[str, Any]], signals: dict[str, Any], message_token_cap: int) -> ConflictObject:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stage_a_rows:
        grouped[str(row.get("normalized_answer") or "")].append(row)
    ranked_groups = sorted(
        grouped.items(),
        key=lambda item: (len(item[1]), _best_confidence(item[1])),
        reverse=True,
    )
    candidate_a = _best_candidate(ranked_groups[0][1]) if ranked_groups else stage_a_rows[0]
    if len(ranked_groups) >= 2:
        candidate_b = _best_candidate(ranked_groups[1][1])
    else:
        candidate_b = _most_divergent_peer(stage_a_rows, candidate_a)

    if bool(signals.get("initial_disagreement")):
        if float(signals.get("evidence_gap") or 0.0) >= max(0.45, float(signals.get("claim_conflict_rate") or 0.0)):
            conflict_type = "evidence"
        elif float(signals.get("claim_conflict_rate") or 0.0) >= 0.40:
            conflict_type = "step"
        else:
            conflict_type = "answer"
    else:
        if float(signals.get("fragile_consensus") or 0.0) >= 0.50:
            conflict_type = "fragile_consensus"
        else:
            conflict_type = "assumption"

    claim_a = flatten_short_list(candidate_a.get("top_claims"), fallback=str(candidate_a.get("reasoning_sketch") or candidate_a.get("final_answer") or ""))
    claim_b = flatten_short_list(candidate_b.get("top_claims"), fallback=str(candidate_b.get("reasoning_sketch") or candidate_b.get("final_answer") or ""))
    support_a = [str(item).strip() for item in (candidate_a.get("evidence_items") or []) if str(item).strip()][:2]
    support_b = [str(item).strip() for item in (candidate_b.get("evidence_items") or []) if str(item).strip()][:2]
    specificity = _specificity_score(claim_a, claim_b, support_a, support_b)
    estimated_comm_cost = float(message_token_cap * len(stage_a_rows) + 40)
    message_type = choose_message_type(conflict_type)
    return ConflictObject(
        conflict_type=conflict_type,
        claim_a=claim_a,
        claim_b=claim_b,
        support_a=support_a,
        support_b=support_b,
        specificity=specificity,
        estimated_comm_cost=estimated_comm_cost,
        evidence_gap=float(signals.get("evidence_gap") or 0.0),
        message_type=message_type,
    )


def compute_utility(signals: dict[str, Any], conflict_object: ConflictObject, policy: Any) -> UtilityBreakdown:
    low_confidence = 1.0 - float(signals.get("mean_confidence") if signals.get("mean_confidence") is not None else 0.5)
    correction_potential = (
        float(policy.correction_answer_entropy_weight) * float(signals.get("answer_entropy") or 0.0)
        + float(policy.correction_low_confidence_weight) * low_confidence
        + float(policy.correction_confidence_spread_weight) * float(signals.get("confidence_spread") or 0.0)
        + float(policy.correction_claim_conflict_weight) * float(signals.get("claim_conflict_rate") or 0.0)
        + float(policy.correction_fragile_consensus_weight) * float(signals.get("fragile_consensus") or 0.0)
    )
    evidence_overlap = 1.0 - float(conflict_object.evidence_gap)
    resolvability = (
        float(policy.resolvability_specificity_weight) * float(conflict_object.specificity)
        + float(policy.resolvability_evidence_overlap_weight) * evidence_overlap
    )
    vagueness_risk = _vagueness_risk(conflict_object)
    collapse_risk = (
        float(policy.collapse_format_risk_weight) * float(signals.get("format_conflict_risk") or 0.0)
        + float(policy.collapse_vagueness_risk_weight) * vagueness_risk
        + float(policy.collapse_majority_pressure_weight) * float(signals.get("majority_pressure_risk") or 0.0)
    )
    normalized_cost = min(1.0, float(conflict_object.estimated_comm_cost) / max(1.0, float(policy.normalized_cost_divisor)))
    utility = correction_potential * resolvability - float(policy.cost_weight) * normalized_cost - float(policy.collapse_weight) * collapse_risk
    return UtilityBreakdown(
        correction_potential=round(correction_potential, 6),
        resolvability=round(resolvability, 6),
        collapse_risk=round(collapse_risk, 6),
        normalized_cost=round(normalized_cost, 6),
        utility=round(utility, 6),
    )


def decide_policy_trigger(policy: Any, signals: dict[str, Any], utility: UtilityBreakdown) -> tuple[bool, str]:
    trigger_type = str(policy.trigger_type)
    if trigger_type == "always_communicate":
        return True, "always_on"
    if trigger_type == "disagreement_triggered":
        triggered = bool(signals.get("initial_disagreement"))
        return triggered, "answer_disagreement" if triggered else "early_exit_by_agreement"
    if trigger_type == "consensus_freeze":
        strong_consensus = (
            not bool(signals.get("initial_disagreement"))
            and float(signals.get("mean_confidence") if signals.get("mean_confidence") is not None else 0.0) >= float(policy.freeze_confidence_threshold)
            and float(signals.get("claim_conflict_rate") or 0.0) <= float(policy.freeze_claim_conflict_threshold)
            and float(signals.get("evidence_gap") or 0.0) <= float(policy.freeze_evidence_gap_threshold)
        )
        return (not strong_consensus), ("freeze_consensus" if strong_consensus else "communicate_non_frozen_case")
    if trigger_type == "cue_utility":
        triggered = float(utility.utility) > float(policy.tau)
        return triggered, "utility_above_tau" if triggered else "utility_below_tau"
    raise ValueError(f"Unsupported cue trigger_type: {trigger_type}")


def choose_message_type(conflict_type: str) -> str:
    mapping = {
        "answer": "answer_critique",
        "step": "step_check",
        "evidence": "evidence_packet",
        "assumption": "assumption_challenge",
        "fragile_consensus": "assumption_challenge",
    }
    return mapping.get(conflict_type, "answer_critique")


def build_peer_packet(stage_a_row: dict[str, Any], message_type: str, max_tokens: int) -> dict[str, Any]:
    base = {
        "final_answer": str(stage_a_row.get("final_answer") or stage_a_row.get("normalized_answer") or "").strip(),
        "confidence": stage_a_row.get("confidence_value"),
    }
    if message_type == "step_check":
        base["top_claims"] = [str(item).strip() for item in (stage_a_row.get("top_claims") or []) if str(item).strip()][:2]
    elif message_type == "evidence_packet":
        base["evidence_items"] = [str(item).strip() for item in (stage_a_row.get("evidence_items") or []) if str(item).strip()][:2]
    else:
        base["reasoning_sketch"] = str(stage_a_row.get("reasoning_sketch") or "").strip()
        base["counter_answer"] = stage_a_row.get("counter_answer")
    packet_text = trim_json_to_token_cap(base, max_tokens)
    return {"packet_fields": base, "packet_text": packet_text, "approx_packet_tokens": approximate_token_count(packet_text)}


def apply_belief_update(
    *,
    stage_a_row: dict[str, Any],
    belief_row: dict[str, Any],
    conflict_object: ConflictObject,
) -> dict[str, Any]:
    validated = belief_row.get("validated_output", {}) if belief_row.get("output_status") == "ok" else {}
    changed_answer = bool(validated.get("changed_answer")) if validated else False
    previous_answer = str(stage_a_row.get("normalized_answer") or "")
    new_answer = str(validated.get("new_answer") or previous_answer).strip()
    confidence_delta = validated.get("confidence_delta") if validated else None
    previous_confidence = stage_a_row.get("confidence_value")
    previous_valid = bool(stage_a_row.get("confidence_valid"))
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
        "agent_id": stage_a_row["agent_id"],
        "changed_answer": changed_answer,
        "final_answer": new_answer,
        "normalized_answer": new_answer,
        "previous_answer": previous_answer,
        "confidence_value": confidence_value,
        "confidence_valid": confidence_valid,
        "confidence_delta": confidence_delta,
        "reason_for_change": str(validated.get("reason_for_change") or "").strip() if validated else "",
        "remaining_disagreement": str(validated.get("remaining_disagreement") or "").strip() if validated else "",
        "top_claims": stage_a_row.get("top_claims") or [conflict_object.claim_a],
        "evidence_items": stage_a_row.get("evidence_items") or conflict_object.support_a,
        "reasoning_sketch": stage_a_row.get("reasoning_sketch") or "",
        "counter_answer": stage_a_row.get("counter_answer"),
    }


def aggregate_with_confidence_tiebreak(candidates: list[dict[str, Any]]) -> tuple[str, dict[str, int]]:
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


def aggregate_weighted_vote(candidates: list[dict[str, Any]]) -> tuple[str, dict[str, float]]:
    grouped_weights: dict[str, float] = defaultdict(float)
    grouped_counts: dict[str, int] = defaultdict(int)
    best_confidence: dict[str, float] = defaultdict(lambda: -1.0)
    min_agent_id: dict[str, int] = defaultdict(lambda: 10**6)
    for candidate in candidates:
        answer = str(candidate.get("normalized_answer", "")).strip()
        if not answer:
            continue
        confidence = float(candidate.get("confidence_value")) if candidate.get("confidence_valid") and candidate.get("confidence_value") is not None else 0.5
        grouped_weights[answer] += confidence
        grouped_counts[answer] += 1
        best_confidence[answer] = max(best_confidence[answer], confidence)
        min_agent_id[answer] = min(min_agent_id[answer], int(candidate.get("agent_id", 10**6)))
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
    return winner, {answer: round(weight, 6) for answer, weight in grouped_weights.items()}


def select_audit_candidate_pair(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    valid_candidates = [candidate for candidate in candidates if candidate.get("normalized_answer")]
    answer_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in valid_candidates:
        answer_groups[str(candidate["normalized_answer"])].append(candidate)
    if len(answer_groups) <= 1:
        only_answer = next(iter(answer_groups), "")
        return {"skipped": True, "skip_reason": "consensus", "candidate_a": None, "candidate_b": None, "fallback_answer": only_answer}
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
    return {"skipped": False, "skip_reason": None, "candidate_a": ranked[0], "candidate_b": ranked[1], "fallback_answer": fallback_answer}


def build_prompt_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "agent_id": candidate.get("agent_id"),
        "final_answer": candidate.get("final_answer") or candidate.get("normalized_answer") or "",
        "confidence_value": candidate.get("confidence_value"),
        "top_claims": candidate.get("top_claims") or None,
        "evidence_items": candidate.get("evidence_items") or None,
        "reason_for_change": candidate.get("reason_for_change") or None,
    }


def trim_json_to_token_cap(payload: dict[str, Any], token_cap: int) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    if approximate_token_count(text) <= token_cap:
        return text
    trimmed = dict(payload)
    for key in ["reasoning_sketch", "top_claims", "evidence_items", "counter_answer"]:
        if key not in trimmed:
            continue
        value = trimmed[key]
        if isinstance(value, list):
            trimmed[key] = [str(item)[:32] for item in value[:1]]
        elif isinstance(value, str):
            trimmed[key] = value[: max(0, token_cap * 4 // 2)]
        text = json.dumps(trimmed, ensure_ascii=False, sort_keys=True)
        if approximate_token_count(text) <= token_cap:
            return text
    return text[: max(1, token_cap * 4 - 3)] + "..."


def _best_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        rows,
        key=lambda item: (
            1 if item.get("confidence_valid") and item.get("confidence_value") is not None else 0,
            float(item.get("confidence_value") or -1.0),
            -int(item.get("agent_id", 10**6)),
        ),
        reverse=True,
    )[0]


def _best_confidence(rows: list[dict[str, Any]]) -> float:
    return max(
        (
            float(row.get("confidence_value"))
            for row in rows
            if row.get("confidence_valid") and row.get("confidence_value") is not None
        ),
        default=-1.0,
    )


def _most_divergent_peer(rows: list[dict[str, Any]], anchor: dict[str, Any]) -> dict[str, Any]:
    anchor_claim = flatten_short_list(anchor.get("top_claims"), fallback=str(anchor.get("reasoning_sketch") or ""))
    ranked = sorted(
        [row for row in rows if row["agent_id"] != anchor["agent_id"]],
        key=lambda row: 1.0 - pairwise_token_jaccard_mean(
            [
                anchor_claim,
                flatten_short_list(row.get("top_claims"), fallback=str(row.get("reasoning_sketch") or "")),
            ]
        ),
        reverse=True,
    )
    return ranked[0] if ranked else anchor


def _normalized_entropy(values: list[str]) -> float:
    materialized = [value for value in values if value]
    if len(set(materialized)) <= 1:
        return 0.0
    counts = Counter(materialized)
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        probability = count / total
        entropy -= probability * log(probability)
    max_entropy = log(len(counts))
    return round(entropy / max_entropy, 6) if max_entropy > 0 else 0.0


def _fragile_consensus(
    *,
    initial_disagreement: bool,
    mean_confidence: float | None,
    claim_conflict_rate: float,
    evidence_gap: float,
) -> float:
    if initial_disagreement:
        return 0.0
    low_conf = 1.0 - float(mean_confidence if mean_confidence is not None else 0.5)
    return round(max(low_conf, claim_conflict_rate * 0.8, evidence_gap * 0.8), 6)


def _format_conflict_risk(raw_answers: list[str], normalized_answers: list[str]) -> float:
    if len(set(normalized_answers)) == 1 and len(set(raw_answers)) > 1:
        return 1.0
    return 0.0


def _majority_pressure_risk(stage_a_rows: list[dict[str, Any]]) -> float:
    answer_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in stage_a_rows:
        answer_groups[str(row.get("normalized_answer") or "")].append(row)
    if len(answer_groups) != 2:
        return 0.0
    ranked = sorted(answer_groups.values(), key=len, reverse=True)
    majority = ranked[0]
    minority = ranked[1]
    majority_conf = sum(
        float(row.get("confidence_value"))
        for row in majority
        if row.get("confidence_valid") and row.get("confidence_value") is not None
    ) / max(
        1,
        sum(1 for row in majority if row.get("confidence_valid") and row.get("confidence_value") is not None),
    )
    minority_best = max(
        (
            float(row.get("confidence_value"))
            for row in minority
            if row.get("confidence_valid") and row.get("confidence_value") is not None
        ),
        default=0.0,
    )
    if minority_best >= majority_conf:
        return 0.8
    if minority_best + 0.05 >= majority_conf:
        return 0.5
    return 0.2


def _specificity_score(claim_a: str, claim_b: str, support_a: list[str], support_b: list[str]) -> float:
    text = " ".join([claim_a, claim_b, *support_a, *support_b]).strip()
    tokens = [token for token in _tokenize(text) if token]
    if not tokens:
        return 0.0
    richness = min(1.0, len(tokens) / 12.0)
    claim_presence = 1.0 if claim_a and claim_b else 0.6 if (claim_a or claim_b) else 0.0
    evidence_presence = 1.0 if support_a and support_b else 0.5 if (support_a or support_b) else 0.0
    return round(0.4 * richness + 0.3 * claim_presence + 0.3 * evidence_presence, 6)


def _vagueness_risk(conflict_object: ConflictObject) -> float:
    generic_markers = {"unknown", "not sure", "maybe", "unclear", "something"}
    text = " ".join([conflict_object.claim_a, conflict_object.claim_b]).lower()
    marker_hit = 1.0 if any(marker in text for marker in generic_markers) else 0.0
    return round(min(1.0, (1.0 - conflict_object.specificity) * 0.8 + marker_hit * 0.4), 6)


def _tokenize(value: str) -> set[str]:
    cleaned = "".join(character.lower() if character.isalnum() else " " for character in value)
    return {token for token in cleaned.split() if token}


def _token_jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return round(len(left & right) / len(union), 6)
