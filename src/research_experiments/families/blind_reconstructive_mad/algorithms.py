"""BRD-MAD 的确定性、与 provider 无关的决策规则。

This module deliberately contains no prompting or request code.  Keeping the
anonymisation and promotion policy pure makes the safety claims testable:
reviewers may only promote an answer already present in Stage A; novel answers
remain diagnostic-only shadows in V1.
"""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CandidateFamily:
    """One normalized Stage-A answer and a single representative rationale."""

    answer: str
    representative_reasoning: str
    representative_agent_id: int
    support_count: int


@dataclass(frozen=True)
class StageADecision:
    anchor_answer: str
    families: tuple[CandidateFamily, ...]
    vote_counts: dict[str, int]
    disagreement_pattern: str

    @property
    def triggered(self) -> bool:
        return len(self.families) > 1

    @property
    def candidate_answers(self) -> set[str]:
        return {family.answer for family in self.families if family.answer}

    @property
    def anchor_support(self) -> int:
        return int(self.vote_counts.get(self.anchor_answer, 0))


@dataclass(frozen=True)
class ReviewerBoard:
    """A reviewer-specific anonymous rendering of candidate families."""

    labels: tuple[str, ...]
    families: tuple[CandidateFamily, ...]
    show_support: bool

    def rendered(self, *, max_chars: int | None = None) -> str:
        """Render every candidate, sharing a total rationale budget fairly.

        A prefix slice of the complete board is not label-invariant: long early
        rationales can remove later candidates entirely after permutation.  We
        therefore reserve the headers and final answers for every family first,
        then divide the remaining budget across rationales.  Truncated
        rationales retain both their beginning and conclusion.
        """

        separator = "\n\n---\n\n"
        prefixes: list[str] = []
        suffixes: list[str] = []
        for label, family in zip(self.labels, self.families, strict=True):
            prefixes.append(
                f"Candidate {label}\n"
                f"Proposed final answer: {family.answer}\n"
                "One anonymous representative derivation:\n"
            )
            suffixes.append(
                f"\nObserved independent answers supporting it: {family.support_count}"
                if self.show_support
                else ""
            )

        if max_chars is None:
            return separator.join(
                prefix + family.representative_reasoning + suffix
                for prefix, family, suffix in zip(prefixes, self.families, suffixes, strict=True)
            )

        fixed_chars = sum(len(prefix) + len(suffix) for prefix, suffix in zip(prefixes, suffixes, strict=True))
        fixed_chars += len(separator) * max(0, len(self.families) - 1)
        rationale_budget = max(0, int(max_chars) - fixed_chars)
        family_count = max(1, len(self.families))
        base_budget, remainder = divmod(rationale_budget, family_count)
        blocks: list[str] = []
        for index, (prefix, family, suffix) in enumerate(zip(prefixes, self.families, suffixes, strict=True)):
            budget = base_budget + (1 if index < remainder else 0)
            reasoning = _balanced_excerpt(family.representative_reasoning, budget)
            blocks.append(prefix + reasoning + suffix)
        return separator.join(blocks)

    def label_to_answer(self) -> dict[str, str]:
        return {label: family.answer for label, family in zip(self.labels, self.families, strict=True)}


def _balanced_excerpt(value: str, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    if max_chars <= 0:
        return ""
    marker = "\n[representative truncated]\n"
    if max_chars <= len(marker):
        return text[:max_chars]
    content_budget = max_chars - len(marker)
    head_chars = (content_budget + 1) // 2
    tail_chars = content_budget - head_chars
    tail = text[-tail_chars:] if tail_chars else ""
    return text[:head_chars] + marker + tail


@dataclass(frozen=True)
class QuorumDecision:
    final_answer: str
    promoted_answer: str | None
    quorum_required: int
    quorum_met: bool
    override_accepted: bool
    reviewer_votes: dict[str, int]
    shadow_answers: tuple[str, ...]
    resolver: str


def build_stage_a_decision(stage_rows: list[dict[str, Any]]) -> StageADecision:
    """Group normalized answers without using model confidence or row ordering.

    Ties are resolved by the smallest agent id, which is a fixed, pre-model
    convention rather than a hidden majority cue.  Candidate family order is
    also deterministic and is subsequently shuffled per reviewer.
    """

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in stage_rows:
        answer = str(row.get("normalized_answer") or row.get("prediction") or "").strip()
        if answer:
            grouped.setdefault(answer, []).append(row)
    if not grouped:
        return StageADecision("", (), {}, "0")

    def first_agent(rows: list[dict[str, Any]]) -> int:
        return min(int(row.get("agent_id") or 0) for row in rows)

    ordered_answers = sorted(grouped, key=lambda answer: (-len(grouped[answer]), first_agent(grouped[answer]), answer))
    families = tuple(
        CandidateFamily(
            answer=answer,
            representative_reasoning=str(grouped[answer][0].get("assistant_text") or "").strip(),
            representative_agent_id=int(grouped[answer][0].get("agent_id") or 0),
            support_count=len(grouped[answer]),
        )
        for answer in ordered_answers
    )
    counts = {answer: len(grouped[answer]) for answer in ordered_answers}
    return StageADecision(
        anchor_answer=ordered_answers[0],
        families=families,
        vote_counts=counts,
        disagreement_pattern="-".join(str(counts[answer]) for answer in ordered_answers),
    )


def build_reviewer_board(
    decision: StageADecision,
    *,
    global_seed: int,
    sample_id: str,
    method_name: str,
    reviewer_id: int,
    show_support: bool,
) -> ReviewerBoard:
    """Shuffle labels and family order independently for every reviewer."""

    families = list(decision.families)
    rng = random.Random(f"brd-v1:{global_seed}:{sample_id}:{method_name}:{reviewer_id}")
    rng.shuffle(families)
    labels = [chr(ord("A") + offset) for offset in range(len(families))]
    return ReviewerBoard(labels=tuple(labels), families=tuple(families), show_support=show_support)


def decide_existing_candidate_quorum(
    stage: StageADecision,
    reviewer_answers: list[str],
    *,
    strong_majority_quorum: int = 3,
    default_quorum: int = 2,
) -> QuorumDecision:
    """Apply BRD's safe coverage rule.

    A 4--1 Stage-A split may only be overturned by unanimous (3/3) review
    support for the existing minority.  Every other disagreement uses a 2/3
    quorum.  Review-generated answers not in Stage A are preserved as shadows
    and never promoted.
    """

    valid_answers = stage.candidate_answers
    existing = [str(answer).strip() for answer in reviewer_answers if str(answer).strip() in valid_answers]
    shadows = tuple(sorted({str(answer).strip() for answer in reviewer_answers if str(answer).strip() and str(answer).strip() not in valid_answers}))
    counts = Counter(existing)
    quorum = strong_majority_quorum if stage.anchor_support == 4 else default_quorum

    eligible = [answer for answer, count in counts.items() if count >= quorum]
    if not eligible:
        return QuorumDecision(
            final_answer=stage.anchor_answer,
            promoted_answer=None,
            quorum_required=quorum,
            quorum_met=False,
            override_accepted=False,
            reviewer_votes=dict(counts),
            shadow_answers=shadows,
            resolver="anchor_fallback_no_quorum",
        )

    # Prefer higher independent reviewer support.  Stage-A support and answer
    # text only break an otherwise exact tie, yielding a fully reproducible rule.
    chosen = sorted(
        eligible,
        key=lambda answer: (-counts[answer], -stage.vote_counts.get(answer, 0), answer),
    )[0]
    override = chosen != stage.anchor_answer
    return QuorumDecision(
        final_answer=chosen,
        promoted_answer=chosen if override else None,
        quorum_required=quorum,
        quorum_met=True,
        override_accepted=override,
        reviewer_votes=dict(counts),
        shadow_answers=shadows,
        resolver="existing_candidate_quorum" if override else "anchor_confirmed_by_quorum",
    )


def reviewer_error_correlation(rows: list[dict[str, Any]], reviewer_count: int = 3) -> dict[str, Any]:
    """Estimate pairwise reviewer-error correlation and effective panel size.

    `rows` must include a boolean sequence under ``reviewer_correctness`` for
    triggered samples.  Missing/invalid reviewer outputs are excluded from a
    pair; this is reported explicitly rather than silently imputed as correct.
    """

    errors: list[list[float | None]] = []
    for row in rows:
        values = row.get("reviewer_correctness")
        if not isinstance(values, list):
            continue
        errors.append([None if value is None else 1.0 - float(bool(value)) for value in values[:reviewer_count]])

    pairwise: list[dict[str, Any]] = []
    correlations: list[float] = []
    for left in range(reviewer_count):
        for right in range(left + 1, reviewer_count):
            pairs = [(row[left], row[right]) for row in errors if len(row) > right and row[left] is not None and row[right] is not None]
            correlation = _pearson([pair[0] for pair in pairs], [pair[1] for pair in pairs]) if len(pairs) >= 2 else None
            pairwise.append({"reviewer_a": left + 1, "reviewer_b": right + 1, "n": len(pairs), "error_correlation": correlation})
            if correlation is not None:
                correlations.append(correlation)
    mean_correlation = sum(correlations) / len(correlations) if correlations else None
    effective = None
    if mean_correlation is not None:
        denominator = 1.0 + (reviewer_count - 1) * mean_correlation
        if denominator > 0:
            effective = reviewer_count / denominator
    return {
        "triggered_samples": len(errors),
        "pairwise": pairwise,
        "mean_error_correlation": mean_correlation,
        "effective_reviewer_count": effective,
    }


def quorum_error_probability_independent(error_rate: float) -> float:
    """The 2-of-3 error probability under an explicitly idealized IID model."""

    error_rate = min(1.0, max(0.0, float(error_rate)))
    return 3 * error_rate**2 - 2 * error_rate**3


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    numerator = sum((x - mean_left) * (y - mean_right) for x, y in zip(left, right, strict=True))
    denominator_left = sum((x - mean_left) ** 2 for x in left)
    denominator_right = sum((y - mean_right) ** 2 for y in right)
    if denominator_left <= 0 or denominator_right <= 0:
        return None
    return numerator / math.sqrt(denominator_left * denominator_right)
