"""CRED-V 验证器路由与聚合逻辑。"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

_NON_ANSWERS = {"", "unknown", "n/a", "none"}
_MATH_DATASETS = {"gsm8k", "math500", "competition_math"}
_SPAN_DATASETS = {"hotpotqa"}
_HETERO_DATASETS = {"gpqa_diamond", "mmlu", "mmlu_abstract_algebra", "mmlu_pro", "strategyqa"}
_HOTPOT_BANNED_EXTRA_TOKENS = {
    "and",
    "or",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
}
_HOTPOT_ALLOWED_EXTRA_TOKENS = {
    "captain",
    "center",
    "centre",
    "city",
    "club",
    "college",
    "company",
    "county",
    "country",
    "episode",
    "episodes",
    "excellence",
    "fc",
    "film",
    "football",
    "inc",
    "language",
    "ltd",
    "of",
    "organization",
    "organisation",
    "province",
    "river",
    "school",
    "state",
    "student",
    "students",
    "team",
    "the",
    "title",
    "university",
}


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


def aggregate_safe_verification(
    *,
    dataset: str,
    question: str,
    context: str,
    stage_rows: list[dict[str, Any]],
    verifier_rows: list[dict[str, Any]],
    hetero_verifier_rows: list[dict[str, Any]],
    stage_winner: str,
    verification_modes: tuple[str, ...] | list[str],
    allow_same_model_promotion: bool,
    concrete_evidence_min_chars: int,
    strong_majority_count: int,
) -> CredAggregateDecision:
    del question
    stage_support = dict(aggregate_stage_a_vote(stage_rows).support)
    if not stage_rows or not stage_winner:
        return CredAggregateDecision(
            stage_winner,
            stage_support,
            "cred_verify_safe_empty_fallback",
            False,
            "stage_a",
        )

    modes = {str(mode) for mode in verification_modes}
    challengers = _safe_challenger_rows(dataset, stage_rows, stage_winner)
    if "deterministic_repair" in modes:
        for challenger in challengers:
            candidate = _row_answer(challenger)
            if _deterministic_repair_allows(dataset, stage_winner, candidate, context):
                return _safe_promoted_decision(
                    dataset=dataset,
                    stage_support=stage_support,
                    stage_rows=stage_rows,
                    stage_winner=stage_winner,
                    winner=candidate,
                    resolver="cred_verify_safe_deterministic_repair",
                    source="deterministic_repair",
                )

    if "tool_verified" in modes:
        for challenger in challengers:
            candidate = _row_answer(challenger)
            if _tool_verification_allows(dataset, stage_winner, candidate, context):
                return _safe_promoted_decision(
                    dataset=dataset,
                    stage_support=stage_support,
                    stage_rows=stage_rows,
                    stage_winner=stage_winner,
                    winner=candidate,
                    resolver="cred_verify_safe_tool_verified",
                    source="tool_verified",
                )

    hetero_rows = list(hetero_verifier_rows)
    if allow_same_model_promotion:
        hetero_rows.extend(verifier_rows)
    if "hetero_verified" in modes and dataset in _HETERO_DATASETS:
        leader_family = answer_family_key(dataset, stage_winner)
        leader_count = _stage_family_count(dataset, stage_rows, leader_family)
        if leader_count < int(strong_majority_count):
            eligible: list[tuple[float, str, dict[str, Any]]] = []
            for row in hetero_rows:
                candidate = _row_answer(row)
                candidate_family = answer_family_key(dataset, candidate)
                if not _is_candidate(candidate) or candidate_family == leader_family:
                    continue
                if not _stage_has_family(dataset, stage_rows, candidate_family):
                    continue
                if not _row_bool(row, "promote"):
                    continue
                if _row_bool(row, "leader_pass") or not _row_bool(row, "challenger_pass"):
                    continue
                if not _has_concrete_evidence(row, concrete_evidence_min_chars):
                    continue
                eligible.append((_row_confidence(row), candidate, row))
            if eligible:
                _, winner, _ = max(eligible, key=lambda item: (item[0], item[1]))
                return _safe_promoted_decision(
                    dataset=dataset,
                    stage_support=stage_support,
                    stage_rows=stage_rows,
                    stage_winner=stage_winner,
                    winner=winner,
                    resolver="cred_verify_safe_hetero_promoted",
                    source="hetero_verifier",
                )

    return CredAggregateDecision(
        stage_winner,
        _verification_support(stage_support, [*verifier_rows, *hetero_verifier_rows], stage_winner),
        "cred_verify_safe_rejected",
        False,
        "stage_a",
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


def _safe_promoted_decision(
    *,
    dataset: str,
    stage_support: dict[str, float],
    stage_rows: list[dict[str, Any]],
    stage_winner: str,
    winner: str,
    resolver: str,
    source: str,
) -> CredAggregateDecision:
    support = dict(stage_support)
    support.setdefault(stage_winner, 0.0)
    support[winner] = max(
        float(support.get(winner, 0.0)),
        float(_stage_family_count(dataset, stage_rows, answer_family_key(dataset, winner))) + 0.5,
    )
    return CredAggregateDecision(
        winner,
        {answer: round(float(value), 6) for answer, value in support.items()},
        resolver,
        answer_family_key(dataset, winner) != answer_family_key(dataset, stage_winner),
        source,
    )


def _safe_challenger_rows(dataset: str, stage_rows: list[dict[str, Any]], stage_winner: str) -> list[dict[str, Any]]:
    leading_family = answer_family_key(dataset, stage_winner)
    grouped, ordered_families = _stage_candidate_groups(dataset, stage_rows)
    challengers = [
        max(grouped[family], key=lambda row: (_row_confidence(row), evidence_quality(row), -len(_row_answer(row))))
        for family in ordered_families
        if family != leading_family
    ]
    challengers.sort(key=lambda row: (_row_confidence(row), evidence_quality(row)), reverse=True)
    return challengers


def _deterministic_repair_allows(dataset: str, leader: str, challenger: str, context: str) -> bool:
    if dataset in _MATH_DATASETS:
        return _canonical_math_text(leader) == _canonical_math_text(challenger) and str(leader).strip() != str(challenger).strip()
    if dataset in _SPAN_DATASETS:
        return _hotpot_span_repair_allows(leader, challenger, context)
    return False


def _tool_verification_allows(dataset: str, leader: str, challenger: str, context: str) -> bool:
    if dataset in _MATH_DATASETS:
        return _math_equivalent(leader, challenger) and str(leader).strip() != str(challenger).strip()
    if dataset in _SPAN_DATASETS:
        return _hotpot_span_repair_allows(leader, challenger, context)
    return False


def _canonical_math_text(value: str) -> str:
    text = str(value or "").strip().lower()
    replacements = {
        "\\left": "",
        "\\right": "",
        "\\,": "",
        "$": "",
        "`": "",
        "∞": "inf",
        "\\infty": "inf",
        "infinity": "inf",
        "+inf": "inf",
        "+oo": "inf",
        "oo": "inf",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\s+", "", text)
    text = text.replace("{", "").replace("}", "")
    return text


def _math_equivalent(left: str, right: str) -> bool:
    if _canonical_math_text(left) == _canonical_math_text(right):
        return True
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import (
            convert_xor,
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )
    except Exception:
        return False
    transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
    try:
        left_expr = parse_expr(_sympy_math_text(left), transformations=transformations, evaluate=True)
        right_expr = parse_expr(_sympy_math_text(right), transformations=transformations, evaluate=True)
        return bool(sp.simplify(left_expr - right_expr) == 0)
    except Exception:
        return False


def _sympy_math_text(value: str) -> str:
    text = _canonical_math_text(value)
    text = text.replace("inf", "oo")
    text = text.replace("\\pi", "pi")
    text = text.replace("π", "pi")
    text = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", text)
    text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", text)
    return text


def _hotpot_span_repair_allows(leader: str, challenger: str, context: str) -> bool:
    leader_tokens = _span_tokens(leader)
    challenger_tokens = _span_tokens(challenger)
    if not leader_tokens or not challenger_tokens:
        return False
    if _unsafe_span_answer(challenger):
        return False
    if not _context_supports_answer(context, challenger):
        return False
    span_index = _subsequence_index(challenger_tokens, leader_tokens)
    if span_index < 0:
        return False
    extra = challenger_tokens[:span_index] + challenger_tokens[span_index + len(leader_tokens) :]
    if not extra or len(extra) > 3:
        return False
    if any(token in _HOTPOT_BANNED_EXTRA_TOKENS for token in extra):
        return False
    return all(token in _HOTPOT_ALLOWED_EXTRA_TOKENS for token in extra)


def _context_supports_answer(context: str, answer: str) -> bool:
    context_norm = _normalized_span(context)
    answer_norm = _normalized_span(answer)
    return bool(answer_norm and answer_norm in context_norm)


def _unsafe_span_answer(answer: str) -> bool:
    normalized = _normalized_span(answer)
    return any(phrase in normalized for phrase in ("not mentioned", "unknown", "not in context", "cannot determine"))


def _span_tokens(value: str) -> list[str]:
    return _normalized_span(value).split()


def _normalized_span(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text.lower()).strip()
    return re.sub(r"\s+", " ", text)


def _subsequence_index(items: list[str], needle: list[str]) -> int:
    if not needle or len(needle) > len(items):
        return -1
    for index in range(0, len(items) - len(needle) + 1):
        if items[index : index + len(needle)] == needle:
            return index
    return -1


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


def _stage_family_count(dataset: str, rows: list[dict[str, Any]], family: str) -> int:
    return sum(1 for row in rows if answer_family_key(dataset, _row_answer(row)) == family)


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
