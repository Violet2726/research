"""DGCR 纯决策层：不进行提示、provider 或文件系统操作。"""

from __future__ import annotations

import hashlib
import random
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CandidateClass:
    key: str
    answer: str
    support_count: int


@dataclass(frozen=True)
class StageDecision:
    anchor_key: str
    anchor_answer: str
    candidates: tuple[CandidateClass, ...]
    vote_counts: dict[str, int]
    valid_count: int

    @property
    def triggered(self) -> bool:
        return len(self.candidates) > 1


@dataclass(frozen=True)
class CruxSpan:
    start_char: int
    end_char: int
    hidden_text: str
    masked_question: str


def build_stage_decision(rows: list[dict[str, Any]], *, seed: int, sample_id: str) -> StageDecision:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get("answer_class_key") or "").strip()
        answer = str(row.get("normalized_answer") or row.get("prediction") or "").strip()
        if key and answer:
            grouped.setdefault(key, []).append(row)
    if not grouped:
        return StageDecision("", "", (), {}, 0)
    ordered_keys = sorted(
        grouped,
        key=lambda key: (-len(grouped[key]), _stable_rank(seed, sample_id, f"stage:{key}"), key),
    )
    candidates = tuple(
        CandidateClass(
            key=key,
            answer=str(grouped[key][0].get("normalized_answer") or grouped[key][0].get("prediction") or ""),
            support_count=len(grouped[key]),
        )
        for key in ordered_keys
    )
    return StageDecision(
        anchor_key=ordered_keys[0],
        anchor_answer=candidates[0].answer,
        candidates=candidates,
        vote_counts={key: len(grouped[key]) for key in ordered_keys},
        valid_count=sum(len(items) for items in grouped.values()),
    )


def validate_crux_span(question: str, *, start_char: int, end_char: int) -> CruxSpan | None:
    """Validate one non-option, contiguous source span and build the mask."""

    source = str(question or "")
    try:
        start = int(start_char)
        end = int(end_char)
    except (TypeError, ValueError):
        return None
    if not (0 <= start < end <= len(source)):
        return None
    options_index = source.rfind("\nOptions:")
    if options_index >= 0 and end > options_index:
        return None
    hidden = source[start:end]
    if not (8 <= len(hidden) <= 256) or not any(character.isalnum() for character in hidden):
        return None
    return CruxSpan(
        start_char=start,
        end_char=end,
        hidden_text=hidden,
        masked_question=source[:start] + "[DGCR_HIDDEN_CRUX]" + source[end:],
    )


def build_panel_labels(
    candidates: tuple[CandidateClass, ...],
    *,
    seed: int,
    sample_id: str,
    panel_index: int,
) -> dict[str, str]:
    """Independently permute anonymous labels; never emit support counts."""

    keys = [candidate.key for candidate in candidates]
    random.Random(f"dgcr-v1:{seed}:{sample_id}:panel:{panel_index}").shuffle(keys)
    return {chr(ord("A") + index): key for index, key in enumerate(keys)}


def exact_span_match(reconstruction: str, hidden_text: str) -> bool:
    return _match_normalize(reconstruction) == _match_normalize(hidden_text)


def panel_successes(
    reconstructions: dict[str, str],
    *,
    label_to_key: dict[str, str],
    span: CruxSpan,
) -> dict[str, bool] | None:
    """Map one complete panel response back to candidate keys.

    Missing labels, unknown labels, or non-string values invalidate the whole
    panel so malformed structured output can never cause an override.
    """

    if set(reconstructions) != set(label_to_key):
        return None
    results: dict[str, bool] = {}
    for label, key in label_to_key.items():
        value = reconstructions.get(label)
        if not isinstance(value, str):
            return None
        results[key] = exact_span_match(value, span.hidden_text)
    return results


def decide_override(
    stage: StageDecision,
    panel_results: list[dict[str, bool] | None],
) -> tuple[str, bool, str]:
    """Accept a unique challenger only under the frozen double-panel rule."""

    if not stage.anchor_key:
        return "", False, "no_valid_stage_answer"
    if len(panel_results) != 2 or any(result is None for result in panel_results):
        return stage.anchor_answer, False, "panel_protocol_failure"
    first, second = panel_results
    assert first is not None and second is not None
    anchor_matches = bool(first.get(stage.anchor_key)) or bool(second.get(stage.anchor_key))
    passing = [
        candidate
        for candidate in stage.candidates
        if candidate.key != stage.anchor_key and bool(first.get(candidate.key)) and bool(second.get(candidate.key))
    ]
    if anchor_matches:
        return stage.anchor_answer, False, "anchor_reconstructed"
    if len(passing) != 1:
        return stage.anchor_answer, False, "no_unique_challenger" if not passing else "multiple_challengers"
    return passing[0].answer, True, "unique_double_panel_override"


def _match_normalize(value: str) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).replace("\r\n", "\n")


def _stable_rank(seed: int, sample_id: str, purpose: str) -> str:
    return hashlib.sha256(f"dgcr-v1:{seed}:{sample_id}:{purpose}".encode("utf-8")).hexdigest()
