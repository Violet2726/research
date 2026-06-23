"""CRED-MAD 路由与聚合的纯逻辑。"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from research_experiments.core.data.evaluation import aggregate_majority

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
    answers = [_row_answer(row) for row in rows if _is_candidate(_row_answer(row))]
    leading_answer, vote_counts = aggregate_majority(answers)
    leading_count = int(vote_counts.get(leading_answer, 0)) if leading_answer else 0
    risk_count = sum(1 for row in rows if _risk_is_material(_row_risk_level(row)))
    evidence_mean = _mean(evidence_quality(row) for row in rows)
    falsifier_specific_failure = _falsifier_reports_specific_failure(rows, leading_answer=leading_answer)
    reasons: list[str] = []
    if not leading_answer:
        reasons.append("no_valid_stage_a_answer")
    if leading_count <= int(protocol.weak_majority_count):
        reasons.append("weak_or_split_vote")
    if risk_count >= int(protocol.risk_trigger_count):
        reasons.append("material_risk_count")
    if leading_count < int(protocol.strong_majority_count):
        reasons.append("no_strong_majority")
    if evidence_mean < float(protocol.min_evidence_quality):
        reasons.append("weak_evidence_quality")
    if falsifier_specific_failure:
        reasons.append("falsifier_specific_failure")
    clean_skip = (
        leading_count >= int(protocol.strong_majority_count)
        and evidence_mean >= float(protocol.min_evidence_quality)
        and risk_count < int(protocol.risk_trigger_count)
        and not falsifier_specific_failure
    )
    return CredRouterDecision(
        triggered=not clean_skip,
        reasons=tuple(reasons if not clean_skip else ("strong_evidence_majority_skip",)),
        leading_answer=leading_answer,
        vote_counts=vote_counts,
        risk_count=risk_count,
        evidence_quality_mean=round(evidence_mean, 6),
    )


def aggregate_stage_a_vote(rows: list[dict[str, Any]]) -> CredAggregateDecision:
    answers = [_row_answer(row) for row in rows if _is_candidate(_row_answer(row))]
    winner, counts = aggregate_majority(answers)
    return CredAggregateDecision(
        final_answer=winner,
        support={answer: float(count) for answer, count in counts.items()},
        resolver="cred_vote_5_majority",
        changed=False,
        source="stage_a_vote",
    )


def aggregate_survival(
    *,
    dataset: str,
    stage_rows: list[dict[str, Any]],
    refutation_rows: list[dict[str, Any]],
    defense_rows: list[dict[str, Any]],
    judge_row: dict[str, Any] | None,
    stage_winner: str,
    override_margin: float,
    concrete_evidence_min_chars: int,
    locked: bool,
) -> CredAggregateDecision:
    scores: dict[str, float] = defaultdict(float)
    for row in stage_rows:
        answer = _row_answer(row)
        if _is_candidate(answer):
            scores[answer] += 1.0 + 0.35 * _row_confidence(row) + 0.25 * evidence_quality(row)
            scores[answer] -= _risk_penalty(_row_risk_level(row))
    for row in refutation_rows:
        answer = _row_answer(row)
        if _is_candidate(answer) and _has_concrete_evidence(row, concrete_evidence_min_chars):
            scores[answer] += 0.55 + 0.25 * _row_confidence(row)
    for row in defense_rows:
        answer = _row_answer(row)
        if _is_candidate(answer) and _has_concrete_evidence(row, concrete_evidence_min_chars):
            scores[answer] += 0.35 + 0.15 * _row_confidence(row)
    if judge_row is not None:
        answer = _row_answer(judge_row)
        if _is_candidate(answer):
            scores[answer] += 0.65 + 0.25 * _row_confidence(judge_row)
    if not scores:
        return CredAggregateDecision(stage_winner, {}, "cred_survival_empty_fallback", False, "fallback")

    winner = max(scores, key=lambda item: (scores[item], item == stage_winner, item))
    stage_score = float(scores.get(stage_winner, 0.0))
    winner_score = float(scores[winner])
    changed = bool(stage_winner and winner != stage_winner)
    if changed:
        margin = winner_score - stage_score
        concrete = any(_row_answer(row) == winner and _has_concrete_evidence(row, concrete_evidence_min_chars) for row in [*refutation_rows, *defense_rows])
        if margin < float(override_margin) or not concrete:
            return CredAggregateDecision(
                stage_winner,
                dict(scores),
                "cred_survival_override_rejected_locked" if locked else "cred_survival_override_rejected",
                False,
                "stage_a_locked" if locked else "stage_a",
            )
    return CredAggregateDecision(
        winner,
        dict(scores),
        "cred_survival_score_locked" if locked else "cred_survival_score",
        changed,
        "survival",
    )


def select_refutation_targets(
    *,
    dataset: str,
    rows: list[dict[str, Any]],
    leading_answer: str,
    max_refutations: int,
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
    candidates.sort(key=lambda row: (_row_confidence(row), evidence_quality(row), str(row.get("agent_role") or "")), reverse=True)
    if candidates:
        return candidates[:max_refutations]
    falsifier = next(
        (
            row
            for row in rows
            if str(row.get("agent_role") or "") == "counterfactual_falsifier"
            and _falsifier_reports_specific_failure([row], leading_answer=leading_answer)
        ),
        None,
    )
    return [falsifier] if falsifier is not None and max_refutations > 0 else []


def evidence_quality(row: dict[str, Any]) -> float:
    text = " ".join(
        str(row.get(key) or row.get("validated_output", {}).get(key) or "")
        for key in ("key_evidence", "evidence", "contract", "risk_summary")
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


def _row_answer(row: dict[str, Any]) -> str:
    return str(row.get("normalized_answer") or row.get("prediction") or "").strip()


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


def _risk_is_material(risk_level: str) -> bool:
    return str(risk_level or "").strip().lower() in {"medium", "high"}


def _risk_penalty(risk_level: str) -> float:
    normalized = str(risk_level or "").strip().lower()
    if normalized == "high":
        return 0.35
    if normalized == "medium":
        return 0.18
    return 0.0


def _has_concrete_evidence(row: dict[str, Any], min_chars: int) -> bool:
    payload = row.get("validated_output") if isinstance(row.get("validated_output"), dict) else {}
    text = str(row.get("key_evidence") or payload.get("key_evidence") or payload.get("evidence") or "").strip()
    return len(text) >= int(min_chars)


def _falsifier_reports_specific_failure(rows: list[dict[str, Any]], *, leading_answer: str) -> bool:
    for row in rows:
        if str(row.get("agent_role") or "") != "counterfactual_falsifier":
            continue
        answer = _row_answer(row)
        if (
            _row_risk_level(row) == "high"
            and _is_candidate(answer)
            and answer_family_key(str(row.get("dataset") or ""), answer)
            != answer_family_key(str(row.get("dataset") or ""), leading_answer)
            and evidence_quality(row) >= 0.55
        ):
            return True
    return False


def _mean(values) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return sum(items) / len(items)


def _coarse_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()
