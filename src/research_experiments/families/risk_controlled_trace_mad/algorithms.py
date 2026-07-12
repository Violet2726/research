"""RCTA-MAD 的纯决策、候选板与数据集无关特征。"""

from __future__ import annotations

import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

FEATURE_VERSION = "rcta_router_features_v1"
FEATURE_NAMES = (
    "max_support_rate",
    "answer_entropy",
    "answer_family_rate",
    "support_margin_rate",
    "trace_length_cv",
    "trace_jaccard_distance",
    "synthesis_existing",
    "synthesis_support_rate",
    "source_trace_rate",
    "certificate_pass",
    "certificate_fail",
    "certificate_unsupported",
    "certificate_arithmetic",
    "certificate_symbolic",
    "certificate_ordering",
    "certificate_boolean",
)
FORBIDDEN_FEATURE_KEYS = frozenset({"dataset", "task", "model", "model_name", "sample_id", "gold", "question", "confidence"})


@dataclass(frozen=True)
class StageDecision:
    anchor_answer: str
    vote_counts: dict[str, int]
    disagreement_pattern: str

    @property
    def triggered(self) -> bool:
        return len(self.vote_counts) > 1


def normalized_answer(row: dict[str, Any]) -> str:
    return str(row.get("normalized_answer") or row.get("prediction") or "").strip()


def stage_decision(rows: list[dict[str, Any]]) -> StageDecision:
    grouped: dict[str, list[int]] = {}
    for row in rows:
        answer = normalized_answer(row)
        if answer:
            grouped.setdefault(answer, []).append(int(row.get("agent_id") or 0))
    if not grouped:
        return StageDecision("", {}, "0")
    order = sorted(grouped, key=lambda value: (-len(grouped[value]), min(grouped[value]), value))
    counts = {answer: len(grouped[answer]) for answer in order}
    return StageDecision(order[0], counts, "-".join(str(counts[item]) for item in order))


def majority_with_anchor_fallback(rows: list[dict[str, Any]], anchor: str) -> tuple[str, dict[str, int], str]:
    counts = Counter(answer for answer in (normalized_answer(row) for row in rows) if answer)
    if not counts:
        return anchor, {}, "anchor_fallback_no_valid_votes"
    top = max(counts.values())
    tied = sorted(answer for answer, count in counts.items() if count == top)
    if len(tied) == 1:
        return tied[0], dict(counts), "majority_vote"
    return anchor, dict(counts), "anchor_fallback_multiclass_tie"


def build_trace_board(rows: list[dict[str, Any]], *, seed: int, sample_id: str, trace_max_chars: int = 1200, board_max_chars: int = 7000) -> tuple[str, dict[str, int]]:
    ordered = list(rows)
    random.Random(f"rcta-board-v1:{seed}:{sample_id}").shuffle(ordered)
    blocks: list[str] = []
    char_counts: dict[str, int] = {}
    for index, row in enumerate(ordered, start=1):
        trace_id = f"T{index}"
        answer = normalized_answer(row)
        reasoning = str((row.get("validated_output") or {}).get("reasoning") or row.get("assistant_text") or "").strip()
        excerpt = balanced_excerpt(reasoning, trace_max_chars)
        block = f"Trace {trace_id}\nFinal answer: {answer}\nReasoning:\n{excerpt}"
        blocks.append(block)
        char_counts[trace_id] = len(excerpt)
    rendered = "\n\n---\n\n".join(blocks)
    if len(rendered) > board_max_chars:
        raise ValueError("Balanced trace board exceeds board_max_chars; fixed headers were not budgeted correctly.")
    return rendered, char_counts


def balanced_excerpt(value: str, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    marker = "\n[trace truncated]\n"
    if max_chars <= len(marker):
        return text[:max_chars]
    room = max_chars - len(marker)
    head = (room + 1) // 2
    tail = room - head
    return text[:head] + marker + (text[-tail:] if tail else "")


def build_feature_vector(
    stage_rows: list[dict[str, Any]],
    synthesis: dict[str, Any],
    certificate: dict[str, Any],
) -> dict[str, float]:
    decision = stage_decision(stage_rows)
    counts = sorted(decision.vote_counts.values(), reverse=True)
    total = max(1, len(stage_rows))
    probabilities = [count / total for count in counts]
    entropy = -sum(value * math.log(value) for value in probabilities if value > 0)
    entropy /= math.log(total) if total > 1 else 1.0
    lengths = [len(str((row.get("validated_output") or {}).get("reasoning") or row.get("assistant_text") or "")) for row in stage_rows]
    mean_length = sum(lengths) / len(lengths) if lengths else 0.0
    variance = sum((value - mean_length) ** 2 for value in lengths) / len(lengths) if lengths else 0.0
    synthesis_answer = str(synthesis.get("final_answer") or "").strip()
    certificate_status = str(certificate.get("status") or "unsupported")
    certificate_type = str(certificate.get("certificate_type") or "unsupported")
    vector = {
        "max_support_rate": (counts[0] / total) if counts else 0.0,
        "answer_entropy": entropy,
        "answer_family_rate": len(counts) / total,
        "support_margin_rate": ((counts[0] - counts[1]) / total) if len(counts) > 1 else 1.0,
        "trace_length_cv": (math.sqrt(variance) / mean_length) if mean_length else 0.0,
        "trace_jaccard_distance": _mean_jaccard_distance(stage_rows),
        "synthesis_existing": float(synthesis_answer in decision.vote_counts),
        "synthesis_support_rate": decision.vote_counts.get(synthesis_answer, 0) / total,
        "source_trace_rate": min(1.0, len(set(synthesis.get("source_trace_ids") or [])) / total),
        "certificate_pass": float(certificate_status == "pass"),
        "certificate_fail": float(certificate_status == "fail"),
        "certificate_unsupported": float(certificate_status == "unsupported"),
        "certificate_arithmetic": float(certificate_type == "arithmetic"),
        "certificate_symbolic": float(certificate_type == "symbolic"),
        "certificate_ordering": float(certificate_type == "ordering"),
        "certificate_boolean": float(certificate_type == "boolean"),
    }
    validate_feature_vector(vector)
    return vector


def validate_feature_vector(vector: dict[str, Any]) -> None:
    forbidden = FORBIDDEN_FEATURE_KEYS & set(vector)
    if forbidden:
        raise ValueError("Forbidden router feature(s): " + ", ".join(sorted(forbidden)))
    if tuple(vector) != FEATURE_NAMES:
        raise ValueError(f"Router features must exactly match {FEATURE_NAMES!r}.")
    for key, value in vector.items():
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Router feature {key} is not finite.")


def _mean_jaccard_distance(rows: list[dict[str, Any]]) -> float:
    token_sets = []
    for row in rows:
        text = str((row.get("validated_output") or {}).get("reasoning") or row.get("assistant_text") or "").lower()
        token_sets.append(set(re.findall(r"[a-z0-9_]+", text)))
    distances: list[float] = []
    for left in range(len(token_sets)):
        for right in range(left + 1, len(token_sets)):
            union = token_sets[left] | token_sets[right]
            overlap = token_sets[left] & token_sets[right]
            distances.append(1.0 - len(overlap) / len(union) if union else 0.0)
    return sum(distances) / len(distances) if distances else 0.0
