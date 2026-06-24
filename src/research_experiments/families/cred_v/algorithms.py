"""CRED-V 验证器路由与聚合逻辑。"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

_NON_ANSWERS = {"", "unknown", "n/a", "none"}


@dataclass(frozen=True)
class CredRouterDecision:
    triggered: bool
    reasons: tuple[str, ...]
    leading_answer: str
    vote_counts: dict[str, int]
    risk_count: int
    evidence_quality_mean: float


@dataclass(frozen=True)
class CredAggregateDecision:
    final_answer: str
    support: dict[str, float]
    resolver: str
    changed: bool
    source: str


def answer_family_key(dataset: str, answer: str) -> str:
    normalized = str(answer or "").strip()
    if not normalized:
        return "unknown"
    if dataset in {"gpqa_diamond", "mmlu", "mmlu_abstract_algebra", "mmlu_pro"}:
        match = re.search(r"\b([A-J])\b", normalized.upper())
        return f"mc:{match.group(1)}" if match else "mc_text:" + _coarse_text(normalized)
    if dataset == "strategyqa":
        lowered = normalized.lower()
        if lowered.startswith("yes"):
            return "bool:yes"
        if lowered.startswith("no"):
            return "bool:no"
    return _coarse_text(normalized)


def build_router_decision(rows: list[dict[str, Any]], *, protocol) -> CredRouterDecision:
    dataset = _dataset_from_rows(rows)
    grouped, _ = _stage_candidate_groups(dataset, rows)
    stage_decision = aggregate_stage_a_vote(rows)
    leading_answer = stage_decision.final_answer
    leading_family = answer_family_key(dataset, leading_answer)
    vote_counts = {key: int(value) for key, value in stage_decision.support.items()}
    leading_count = len(grouped.get(leading_family, [])) if leading_answer else 0
    risk_count = sum(1 for row in rows if _row_risk_level(row) in {"medium", "high"})
    evidence_mean = _mean(evidence_quality(row) for row in rows)
    reasons: list[str] = []
    if not leading_answer:
        reasons.append("no_valid_stage_a_answer")
    elif leading_count < int(protocol.strong_majority_count):
        reasons.append("weak_split_no_strong_majority")
    clean_skip = bool(leading_answer) and leading_count >= int(protocol.strong_majority_count)
    return CredRouterDecision(
        triggered=not clean_skip,
        reasons=tuple(reasons if not clean_skip else ("strong_majority_skip",)),
        leading_answer=leading_answer,
        vote_counts=vote_counts,
        risk_count=risk_count,
        evidence_quality_mean=round(evidence_mean, 6),
    )


def aggregate_stage_a_vote(rows: list[dict[str, Any]]) -> CredAggregateDecision:
    dataset = _dataset_from_rows(rows)
    grouped, ordered_families = _stage_candidate_groups(dataset, rows)
    if not grouped:
        return CredAggregateDecision(
            final_answer="",
            support={},
            resolver="cred_v_vote_5_empty",
            changed=False,
            source="stage_a_vote",
        )
    count_family = max(ordered_families, key=lambda family: (len(grouped[family]), -ordered_families.index(family)))
    winner = _representative_answer(dataset, grouped[count_family])
    counts = _stage_support_counts(dataset, grouped, ordered_families)
    return CredAggregateDecision(
        final_answer=winner,
        support={answer: float(count) for answer, count in counts.items()},
        resolver="cred_v_vote_5_family_majority",
        changed=False,
        source="stage_a_vote",
    )


def aggregate_task_verification(
    *,
    dataset: str,
    stage_rows: list[dict[str, Any]],
    verifier_rows: list[dict[str, Any]],
    stage_winner: str,
    promotion_confidence_min: float,
    promotion_score_margin: float,
    concrete_evidence_min_chars: int,
) -> CredAggregateDecision:
    stage_support = dict(aggregate_stage_a_vote(stage_rows).support)
    if not verifier_rows or not stage_winner:
        return CredAggregateDecision(
            stage_winner,
            stage_support,
            "cred_v_task_verify_empty_fallback",
            False,
            "stage_a",
        )

    leader_family = answer_family_key(dataset, stage_winner)
    eligible: list[tuple[float, float, float, str, dict[str, Any]]] = []
    for row in verifier_rows:
        candidate = _row_answer(row)
        if not _is_candidate(candidate):
            continue
        candidate_family = answer_family_key(dataset, candidate)
        if candidate_family == leader_family:
            continue
        if not _stage_has_family(dataset, stage_rows, candidate_family):
            continue
        if not _row_bool(row, "promote"):
            continue
        if not _has_concrete_evidence(row, concrete_evidence_min_chars):
            continue
        confidence = _row_confidence(row)
        if confidence < float(promotion_confidence_min):
            continue
        leader_score = _row_float(row, "leader_score", default=0.5)
        challenger_score = _row_float(row, "challenger_score", default=0.5)
        margin = challenger_score - leader_score
        if margin < float(promotion_score_margin):
            continue
        eligible.append((margin, confidence, challenger_score, candidate, row))

    if not eligible:
        return CredAggregateDecision(
            stage_winner,
            _verification_support(stage_support, verifier_rows, stage_winner),
            "cred_v_task_verify_rejected",
            False,
            "stage_a",
        )

    margin, confidence, challenger_score, winner, _ = max(eligible, key=lambda item: (item[0], item[1], item[2], item[3]))
    support = _verification_support(stage_support, verifier_rows, stage_winner)
    support[winner] = round(max(float(support.get(winner, 0.0)), 5.0 + margin + confidence + challenger_score), 6)
    return CredAggregateDecision(
        winner,
        support,
        "cred_v_task_verify_promoted",
        True,
        "task_verifier",
    )


def select_verification_targets(
    *,
    dataset: str,
    rows: list[dict[str, Any]],
    leading_answer: str,
    max_verifications: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        answer = _row_answer(row)
        if _is_candidate(answer):
            grouped[answer_family_key(dataset, answer)].append(row)
    leading_family = answer_family_key(dataset, leading_answer)
    candidates = [
        max(items, key=lambda row: (_row_confidence(row), evidence_quality(row)))
        for family, items in grouped.items()
        if family != leading_family
    ]
    candidates.sort(
        key=lambda row: (_row_confidence(row), evidence_quality(row), str(row.get("agent_role") or "")),
        reverse=True,
    )
    return candidates[: int(max_verifications)]


def evidence_quality(row: dict[str, Any]) -> float:
    text = " ".join(
        str(row.get(key) or row.get("validated_output", {}).get(key) or "")
        for key in ("key_evidence", "evidence", "contract", "risk_summary", "reasoning")
    )
    cleaned = " ".join(text.split())
    score = 0.0
    if len(cleaned) >= 12:
        score += 0.35
    if any(char.isdigit() for char in cleaned):
        score += 0.15
    if any(token in cleaned.lower() for token in ("because", "therefore", "context", "option", "constraint", "calculate", "span")):
        score += 0.2
    if _is_candidate(_row_answer(row)) and _row_answer(row).lower() in cleaned.lower():
        score += 0.2
    return min(1.0, round(score, 6))


def _verification_support(stage_support: dict[str, float], verifier_rows: list[dict[str, Any]], stage_winner: str) -> dict[str, float]:
    support = {answer: float(value) for answer, value in stage_support.items()}
    if stage_winner:
        support.setdefault(stage_winner, 0.0)
    for row in verifier_rows:
        answer = _row_answer(row)
        if not _is_candidate(answer):
            continue
        leader_score = _row_float(row, "leader_score", default=0.5)
        challenger_score = _row_float(row, "challenger_score", default=0.5)
        if _row_bool(row, "promote"):
            support[answer] = max(float(support.get(answer, 0.0)), 4.0 + challenger_score + _row_confidence(row))
            if stage_winner:
                support[stage_winner] = max(float(support.get(stage_winner, 0.0)), 4.0 + leader_score)
        elif stage_winner:
            support[stage_winner] = max(float(support.get(stage_winner, 0.0)), 4.0 + leader_score + _row_confidence(row))
    return {answer: round(value, 6) for answer, value in support.items()}


def _row_answer(row: dict[str, Any]) -> str:
    return str(row.get("normalized_answer") or row.get("prediction") or "").strip()


def _dataset_from_rows(rows: list[dict[str, Any]]) -> str:
    return str(next((row.get("dataset") for row in rows if row.get("dataset")), ""))


def _stage_candidate_groups(dataset: str, rows: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ordered_families: list[str] = []
    for row in rows:
        answer = _row_answer(row)
        if not _is_candidate(answer):
            continue
        family = answer_family_key(dataset, answer)
        if family not in grouped:
            ordered_families.append(family)
        grouped[family].append(row)
    return grouped, ordered_families


def _stage_support_counts(
    dataset: str,
    grouped: dict[str, list[dict[str, Any]]],
    ordered_families: list[str],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for family in ordered_families:
        representative = _representative_answer(dataset, grouped[family])
        counts[representative] = counts.get(representative, 0) + len(grouped[family])
    return counts


def _representative_answer(dataset: str, rows: list[dict[str, Any]]) -> str:
    del dataset
    if not rows:
        return ""
    row = max(
        rows,
        key=lambda item: (
            _row_confidence(item),
            evidence_quality(item),
            -len(_row_answer(item)),
        ),
    )
    return _row_answer(row)


def _stage_has_family(dataset: str, rows: list[dict[str, Any]], family: str) -> bool:
    return any(answer_family_key(dataset, _row_answer(row)) == family for row in rows)


def _row_confidence(row: dict[str, Any]) -> float:
    value = row.get("confidence_value")
    if value is None and isinstance(row.get("validated_output"), dict):
        value = row["validated_output"].get("confidence")
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def _row_risk_level(row: dict[str, Any]) -> str:
    payload = row.get("validated_output") if isinstance(row.get("validated_output"), dict) else {}
    normalized = str(row.get("risk_level") or payload.get("risk_level") or "").strip().lower()
    return normalized if normalized in {"none", "low", "medium", "high"} else "none"


def _is_candidate(answer: str) -> bool:
    return str(answer or "").strip().lower() not in _NON_ANSWERS


def _has_concrete_evidence(row: dict[str, Any], min_chars: int) -> bool:
    payload = row.get("validated_output") if isinstance(row.get("validated_output"), dict) else {}
    text = str(row.get("key_evidence") or payload.get("key_evidence") or payload.get("evidence") or "").strip()
    return len(text) >= int(min_chars)


def _row_bool(row: dict[str, Any], key: str) -> bool:
    payload = row.get("validated_output") if isinstance(row.get("validated_output"), dict) else {}
    value = payload.get(key, row.get(key))
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"true", "yes", "1"}


def _row_float(row: dict[str, Any], key: str, *, default: float) -> float:
    payload = row.get("validated_output") if isinstance(row.get("validated_output"), dict) else {}
    value = payload.get(key, row.get(key))
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _mean(values) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return sum(items) / len(items)


def _coarse_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()
