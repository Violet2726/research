"""A-SMAD 路由与聚合逻辑。"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

DEFAULT_CONFIDENCE = 0.5
_NON_ANSWER_VALUES = {"", "unknown"}


def aggregate_confidence_weighted(rows: list[dict[str, Any]]) -> tuple[str, dict[str, float]]:
    grouped_weights: dict[str, float] = defaultdict(float)
    grouped_counts: dict[str, int] = defaultdict(int)
    best_confidence: dict[str, float] = defaultdict(float)
    for row in rows:
        answer = str(row.get("normalized_answer") or "").strip()
        if not _is_answer_candidate(answer):
            continue
        confidence = _row_confidence(row)
        grouped_weights[answer] += confidence
        grouped_counts[answer] += 1
        best_confidence[answer] = max(best_confidence[answer], confidence)
    if not grouped_weights:
        return "", {}
    winner = max(
        grouped_weights,
        key=lambda answer: (
            grouped_weights[answer],
            best_confidence[answer],
            -grouped_counts[answer],
            answer,
        ),
    )
    return winner, {answer: round(value, 6) for answer, value in grouped_weights.items()}


def aggregate_anchor_protected(rows: list[dict[str, Any]]) -> tuple[str, dict[str, float]]:
    answer, support = aggregate_confidence_weighted(rows)
    anchor_row = next((row for row in rows if row.get("solver_mode") == "solver_cot"), None)
    if anchor_row is None:
        return answer, support
    anchor_answer = str(anchor_row.get("normalized_answer") or "").strip()
    if not anchor_answer:
        return answer, support

    grouped = _group_answer_rows(rows)
    if len(grouped) <= 1:
        return answer, support
    if len(grouped.get(anchor_answer, [])) >= 2:
        return anchor_answer, support

    majority_answers = [candidate for candidate, candidate_rows in grouped.items() if len(candidate_rows) >= 2]
    if majority_answers:
        majority_answer = max(
            majority_answers,
            key=lambda candidate: (
                len(grouped[candidate]),
                sum(_row_confidence(row) for row in grouped[candidate]),
                candidate,
            ),
        )
        if _should_prefer_clean_anchor_over_degraded_majority(
            anchor_row=anchor_row,
            anchor_answer=anchor_answer,
            majority_answer=majority_answer,
            grouped=grouped,
        ):
            return anchor_answer, support
        return majority_answer, support

    return anchor_answer, support


def aggregate_constraint_aware_stage_a(rows: list[dict[str, Any]]) -> tuple[str, dict[str, float], str]:
    answer, support = aggregate_anchor_protected(rows)
    if not rows:
        return answer, support, "constraint_aware_anchor_vote"

    grouped = _group_answer_rows(rows)
    selected_answer = str(answer or "").strip() or "unknown"
    selected_rows = grouped.get(selected_answer, [])
    clean_count = sum(1 for row in selected_rows if not _row_is_degraded(row))
    anchor_row = next((row for row in rows if row.get("solver_mode") == "solver_cot"), None)
    anchor_answer = str(anchor_row.get("normalized_answer") or "").strip() if anchor_row is not None else ""

    if selected_answer.lower() in _NON_ANSWER_VALUES:
        clean_candidates = [
            candidate
            for candidate, candidate_rows in grouped.items()
            if candidate.lower() not in _NON_ANSWER_VALUES
            and sum(1 for row in candidate_rows if not _row_is_degraded(row)) >= 2
        ]
        if len(clean_candidates) == 1:
            return clean_candidates[0], support, "constraint_aware_non_unknown_rescue"

    if selected_rows and clean_count == 0:
        clean_majority_candidates = [
            candidate
            for candidate, candidate_rows in grouped.items()
            if candidate != selected_answer
            and sum(1 for row in candidate_rows if not _row_is_degraded(row)) >= 2
        ]
        if len(clean_majority_candidates) == 1:
            return clean_majority_candidates[0], support, "constraint_aware_clean_group_rescue"

    if (
        anchor_row is not None
        and anchor_answer
        and anchor_answer != selected_answer
        and not _row_is_degraded(anchor_row)
        and _majority_pattern(grouped) == "two_to_one"
        and len(grouped.get(anchor_answer, [])) == 1
        and all(str(row.get("solver_mode") or "") != "solver_cot" for row in selected_rows)
    ):
        if _rows_form_clean_expression_consensus(selected_rows):
            return selected_answer, support, "constraint_aware_clean_expression_majority_keep"
        if _rows_form_clean_slot_majority(selected_rows):
            return selected_answer, support, "constraint_aware_clean_slot_majority_keep"
        return anchor_answer, support, "constraint_aware_clean_anchor_minority_override"

    skeptic_row = next((row for row in rows if row.get("solver_mode") == "solver_skeptic"), None)
    skeptic_answer = str(skeptic_row.get("normalized_answer") or "").strip() if skeptic_row is not None else ""
    if (
        skeptic_row is not None
        and skeptic_answer
        and skeptic_answer != selected_answer
        and not _row_is_degraded(skeptic_row)
        and _majority_pattern(grouped) == "two_to_one"
        and len(grouped.get(skeptic_answer, [])) == 1
        and tuple(sorted(str(row.get("solver_mode") or "") for row in selected_rows)) == ("solver_cot", "solver_l2m")
        and _row_has_structured_constraint_fields(skeptic_row)
        and all(not _reasoning_looks_mathy(str(row.get("reasoning") or "")) for row in selected_rows)
    ):
        return skeptic_answer, support, "constraint_aware_clean_skeptic_minority_override"

    if _majority_pattern(grouped) == "two_to_one":
        minority_rows = [
            answer_rows[0]
            for answer, answer_rows in grouped.items()
            if answer != selected_answer and len(answer_rows) == 1
        ]
        if len(selected_rows) == 2 and len(minority_rows) == 1:
            minority_row = minority_rows[0]
            if _should_prefer_typed_minority(minority_row=minority_row, majority_rows=selected_rows):
                minority_answer = str(minority_row.get("normalized_answer") or "").strip()
                if _is_answer_candidate(minority_answer):
                    return minority_answer, support, "constraint_aware_typed_minority_override"

    if (
        anchor_row is None
        and skeptic_row is not None
        and skeptic_answer
        and skeptic_answer != selected_answer
        and not _row_is_degraded(skeptic_row)
        and _majority_pattern(grouped) == "two_to_one"
        and len(grouped.get(skeptic_answer, [])) == 1
        and tuple(sorted(str(row.get("solver_mode") or "") for row in selected_rows)) == ("solver_cot", "solver_l2m")
        and _row_has_structured_constraint_fields(skeptic_row)
        and all(not _reasoning_looks_mathy(str(row.get("reasoning") or "")) for row in selected_rows)
    ):
        return skeptic_answer, support, "constraint_aware_clean_skeptic_minority_override"

    return answer, support, "constraint_aware_anchor_vote"


def aggregate_evidence_grounded_stage_a(
    rows: list[dict[str, Any]],
    *,
    anchor_answer: str = "",
    question: str = "",
) -> tuple[str, dict[str, float], str]:
    grouped = _group_answer_rows(rows)
    if not grouped:
        normalized_anchor = str(anchor_answer or "").strip() or "unknown"
        return normalized_anchor, {normalized_anchor: 0.0}, "evidence_grounded_empty"

    score_by_answer: dict[str, float] = {}
    clean_support_by_answer: dict[str, float] = {}
    total_support_by_answer: dict[str, float] = {}
    diversity_by_answer: dict[str, int] = {}
    evidence_count_by_answer: dict[str, int] = {}

    for answer, answer_rows in grouped.items():
        clean_support = sum(_row_confidence(row) for row in answer_rows if not _row_is_degraded(row))
        total_support = sum(_row_confidence(row) for row in answer_rows)
        evidence_count = sum(1 for row in answer_rows if _row_has_meaningful_evidence(row))
        claim_match_count = sum(1 for row in answer_rows if _claim_or_evidence_matches_answer(row, answer))
        structured_count = sum(1 for row in answer_rows if _row_has_structured_constraint_fields(row))
        solver_diversity = len(
            {
                str(row.get("solver_mode") or row.get("method_name") or "").strip()
                for row in answer_rows
                if str(row.get("solver_mode") or row.get("method_name") or "").strip()
            }
        )
        degradation_penalty = sum(0.35 for row in answer_rows if _row_is_degraded(row))
        type_conflict_penalty = 0.15 if _answer_group_has_type_conflict(answer_rows) else 0.0
        non_answer_penalty = 0.75 if not _is_answer_candidate(answer) else 0.0
        anchor_bonus = 0.05 if anchor_answer and answer == anchor_answer else 0.0
        slot_complete_bonus = _slot_complete_preference_bonus(
            answer=answer,
            grouped=grouped,
            question=question,
        )
        score = (
            clean_support
            + 0.12 * evidence_count
            + 0.12 * claim_match_count
            + 0.08 * max(0, solver_diversity - 1)
            + 0.05 * structured_count
            + anchor_bonus
            + slot_complete_bonus
            - degradation_penalty
            - type_conflict_penalty
            - non_answer_penalty
        )
        score_by_answer[answer] = round(score, 6)
        clean_support_by_answer[answer] = clean_support
        total_support_by_answer[answer] = total_support
        diversity_by_answer[answer] = solver_diversity
        evidence_count_by_answer[answer] = evidence_count

    winner = max(
        score_by_answer,
        key=lambda answer: (
            score_by_answer[answer],
            clean_support_by_answer[answer],
            total_support_by_answer[answer],
            evidence_count_by_answer[answer],
            diversity_by_answer[answer],
            answer == anchor_answer,
            answer,
        ),
    )
    return winner, score_by_answer, "evidence_grounded_score_vote"


def _group_answer_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        answer = str(row.get("normalized_answer") or "").strip()
        if not _is_answer_candidate(answer):
            continue
        grouped[answer].append(row)
    if not grouped:
        for row in rows:
            answer = str(row.get("normalized_answer") or "").strip() or "unknown"
            grouped[answer].append(row)
    return grouped
def _is_answer_candidate(answer: str) -> bool:
    return str(answer or "").strip().lower() not in _NON_ANSWER_VALUES


def _should_prefer_clean_anchor_over_degraded_majority(
    *,
    anchor_row: dict[str, Any],
    anchor_answer: str,
    majority_answer: str,
    grouped: dict[str, list[dict[str, Any]]],
) -> bool:
    if not anchor_answer or anchor_answer == majority_answer:
        return False
    if _row_is_degraded(anchor_row):
        return False
    majority_rows = grouped.get(majority_answer, [])
    if len(majority_rows) < 2:
        return False
    if any(str(row.get("solver_mode") or "") == "solver_cot" for row in majority_rows):
        return False
    return any(_row_is_degraded(row) for row in majority_rows)


def _row_is_degraded(row: dict[str, Any]) -> bool:
    if bool(row.get("stage_a_safe_retry_used")):
        return True
    validated_output = row.get("validated_output")
    if isinstance(validated_output, dict) and validated_output.get("stage_a_recovery_fallback"):
        return True
    answer = str(row.get("normalized_answer") or "").strip().lower()
    return answer in _NON_ANSWER_VALUES


def _row_has_structured_constraint_fields(row: dict[str, Any]) -> bool:
    validated_output = row.get("validated_output")
    if not isinstance(validated_output, dict):
        return False
    return bool(str(validated_output.get("answer_type") or "").strip()) and bool(
        str(validated_output.get("key_constraints") or "").strip()
    )


def _row_has_meaningful_evidence(row: dict[str, Any]) -> bool:
    claim_span = str(row.get("claim_span") or "").strip()
    key_evidence = str(row.get("key_evidence") or "").strip()
    if claim_span and claim_span.lower() not in _NON_ANSWER_VALUES:
        return True
    return bool(key_evidence and key_evidence.lower() not in _NON_ANSWER_VALUES)


def _claim_or_evidence_matches_answer(row: dict[str, Any], answer: str) -> bool:
    normalized_answer = _coarse_normalize_text(answer)
    if not normalized_answer:
        return False
    claim_span = _coarse_normalize_text(str(row.get("claim_span") or ""))
    key_evidence = _coarse_normalize_text(str(row.get("key_evidence") or ""))
    if claim_span and (claim_span == normalized_answer or normalized_answer in claim_span or claim_span in normalized_answer):
        return True
    return bool(key_evidence and normalized_answer in key_evidence)


def _answer_group_has_type_conflict(rows: list[dict[str, Any]]) -> bool:
    normalized_types = sorted({_normalized_answer_type(row) for row in rows if _normalized_answer_type(row)})
    return len(normalized_types) > 1


def _coarse_normalize_text(text: str) -> str:
    lowered = str(text or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", " ", lowered).strip()


def _slot_complete_preference_bonus(
    *,
    answer: str,
    grouped: dict[str, list[dict[str, Any]]],
    question: str,
) -> float:
    normalized_answer = _coarse_normalize_text(answer)
    if not normalized_answer:
        return 0.0
    bonus = 0.0
    for other_answer in grouped:
        if other_answer == answer:
            continue
        normalized_other = _coarse_normalize_text(other_answer)
        if not normalized_other or normalized_other == normalized_answer:
            continue
        if _is_boolean_explanatory_overrun(longer=normalized_answer, shorter=normalized_other):
            shorter_support = sum(_row_confidence(row) for row in grouped.get(other_answer, []))
            bonus -= 1.0 * shorter_support
            continue
        if _is_boolean_explanatory_overrun(longer=normalized_other, shorter=normalized_answer):
            longer_support = sum(_row_confidence(row) for row in grouped.get(other_answer, []))
            bonus += 0.5 * longer_support
            continue
        if normalized_answer in normalized_other and len(normalized_other) > len(normalized_answer):
            if _longer_answer_looks_slot_complete(question=question, longer=normalized_other, shorter=normalized_answer):
                longer_support = sum(_row_confidence(row) for row in grouped.get(other_answer, []))
                bonus -= 0.15 * longer_support
        elif (
            normalized_other in normalized_answer
            and len(normalized_answer) > len(normalized_other)
            and _longer_answer_looks_slot_complete(
                question=question,
                longer=normalized_answer,
                shorter=normalized_other,
            )
        ):
            shorter_support = sum(_row_confidence(row) for row in grouped.get(other_answer, []))
            bonus += 1.0 * shorter_support
    return bonus


def _longer_answer_looks_slot_complete(*, question: str, longer: str, shorter: str) -> bool:
    if shorter in {"yes", "no"}:
        return False
    longer_tokens = [token for token in longer.split() if token]
    shorter_tokens = {token for token in shorter.split() if token}
    extra_tokens = [token for token in longer_tokens if token not in shorter_tokens]
    if not extra_tokens:
        return False
    if re.match(r"^\d{4}$", extra_tokens[0]):
        return True
    slot_words = {
        "students",
        "episodes",
        "language",
        "languages",
        "film",
        "storm",
        "title",
        "court",
        "courts",
        "county",
        "city",
        "state",
        "province",
        "club",
        "team",
        "album",
        "novel",
        "song",
        "season",
    }
    if any(token in slot_words for token in extra_tokens):
        return True
    question_tokens = {
        token
        for token in _coarse_normalize_text(question).split()
        if len(token) >= 4 and token not in {"what", "which", "that", "from", "with", "this", "their", "there"}
    }
    return any(token in question_tokens for token in extra_tokens)


def _is_boolean_explanatory_overrun(*, longer: str, shorter: str) -> bool:
    if shorter not in {"yes", "no"}:
        return False
    longer_tokens = [token for token in longer.split() if token]
    return len(longer_tokens) >= 3 and longer_tokens[0] == shorter


def _should_prefer_typed_minority(
    *,
    minority_row: dict[str, Any],
    majority_rows: list[dict[str, Any]],
) -> bool:
    if _row_is_degraded(minority_row) or not _row_has_structured_constraint_fields(minority_row):
        return False
    if any(_row_is_degraded(row) for row in majority_rows):
        return False

    majority_types = sorted({_normalized_answer_type(row) for row in majority_rows if _normalized_answer_type(row)})
    minority_type = _normalized_answer_type(minority_row)
    if len(majority_types) != 1 or not minority_type:
        return False

    preferred_pairs = {
        ("symmetry_group", "molecular_symmetry_group"),
        ("integer", "option"),
        ("expression", "numeric"),
    }
    return (majority_types[0], minority_type) in preferred_pairs


def _rows_form_clean_expression_consensus(rows: list[dict[str, Any]]) -> bool:
    if len(rows) < 2:
        return False
    return all(
        not _row_is_degraded(row)
        and _normalized_answer_type(row) == "expression"
        and _row_has_structured_constraint_fields(row)
        for row in rows
    )


def _rows_form_clean_slot_majority(rows: list[dict[str, Any]]) -> bool:
    if len(rows) < 2:
        return False
    if not all(not _row_is_degraded(row) and _row_has_structured_constraint_fields(row) for row in rows):
        return False
    families = {_normalized_answer_type(row) for row in rows if _normalized_answer_type(row)}
    if not families:
        return False
    return "boolean" in families or "character" in families or "location" in families or "span" in families


def _normalized_answer_type(row: dict[str, Any]) -> str:
    validated_output = row.get("validated_output")
    raw_value = ""
    if isinstance(validated_output, dict):
        raw_value = str(validated_output.get("answer_type") or "").strip()
    normalized = raw_value.lower().replace("-", "_").replace(" ", "_").replace("/", "_")
    if not normalized:
        return ""
    if normalized in {"multiple_choice", "multiple_choice_letter", "option_letter", "option"}:
        return "option"
    if normalized in {"boolean", "yes_no", "yes_no_judgment"}:
        return "boolean"
    if normalized in {"number", "numeric", "percentage"}:
        return "numeric"
    if "expression" in normalized:
        return "expression"
    if "location" in normalized:
        return "location"
    if "character" in normalized:
        return "character"
    if "phrase" in normalized or "span" in normalized:
        return "span"
    return normalized


def _reasoning_looks_mathy(text: str) -> bool:
    normalized = str(text or "")
    if not normalized:
        return False
    math_chars = set("=+-*/^\\\\[](){}<>")
    return any(char in math_chars for char in normalized) or any(char.isdigit() for char in normalized)


def _majority_pattern(grouped: dict[str, list[dict[str, Any]]]) -> str:
    sizes = sorted((len(rows) for rows in grouped.values()), reverse=True)
    if sizes == [3]:
        return "three_to_zero"
    if sizes == [2, 1]:
        return "two_to_one"
    if sizes == [1, 1, 1]:
        return "three_way_split"
    return "other"


def _row_confidence(row: dict[str, Any]) -> float:
    value = row.get("confidence_value")
    if value is None:
        return DEFAULT_CONFIDENCE
    try:
        return float(value)
    except (TypeError, ValueError):
        return DEFAULT_CONFIDENCE
