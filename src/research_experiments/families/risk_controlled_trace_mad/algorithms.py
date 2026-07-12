"""EVF-MAD 的确定性投票、匿名候选板与安全覆盖规则。"""

from __future__ import annotations

import hashlib
import random
from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StageDecision:
    anchor_answer: str
    vote_counts: dict[str, int]
    disagreement_pattern: str
    resolver: str
    valid_trace_count: int

    @property
    def triggered(self) -> bool:
        return len(self.vote_counts) > 1


def normalized_answer(row: dict[str, Any]) -> str:
    return str(row.get("normalized_answer") or row.get("prediction") or "").strip()


def stage_decision(rows: list[dict[str, Any]], *, qwen_rows: list[dict[str, Any]] | None = None) -> StageDecision:
    valid = [row for row in rows if normalized_answer(row)]
    counts = Counter(normalized_answer(row) for row in valid)
    if not counts:
        return StageDecision("", {}, "0", "no_valid_votes", 0)
    top = max(counts.values())
    tied = {answer for answer, count in counts.items() if count == top}
    resolver = "heterogeneous_majority"
    if len(tied) == 1:
        anchor = next(iter(tied))
    else:
        qwen_valid = [row for row in (qwen_rows or []) if normalized_answer(row) in tied]
        qwen_counts = Counter(normalized_answer(row) for row in qwen_valid)
        if qwen_counts:
            qwen_top = max(qwen_counts.values())
            qwen_tied = {answer for answer, count in qwen_counts.items() if count == qwen_top}
        else:
            qwen_tied = set(tied)
        if len(qwen_tied) == 1:
            anchor = next(iter(qwen_tied))
            resolver = "qwen_three_tie_fallback"
        else:
            anchor = next(normalized_answer(row) for row in valid if normalized_answer(row) in qwen_tied)
            resolver = "fixed_stage_order_tie_fallback"
    ordered = sorted(counts, key=lambda answer: (-counts[answer], answer))
    return StageDecision(
        anchor_answer=anchor,
        vote_counts={answer: counts[answer] for answer in ordered},
        disagreement_pattern="-".join(map(str, sorted(counts.values(), reverse=True))),
        resolver=resolver,
        valid_trace_count=len(valid),
    )


def majority_with_anchor_fallback(rows: list[dict[str, Any]], anchor: str) -> tuple[str, dict[str, int], str]:
    counts = Counter(normalized_answer(row) for row in rows if normalized_answer(row))
    if not counts:
        return anchor, {}, "anchor_fallback_no_valid_votes"
    top = max(counts.values())
    tied = [answer for answer, count in counts.items() if count == top]
    if len(tied) == 1:
        return tied[0], dict(counts), "majority_vote"
    return anchor, dict(counts), "anchor_fallback_multiclass_tie"


def balanced_excerpt(value: str, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    marker = "\n[trace truncated]\n"
    room = max(0, max_chars - len(marker))
    head = (room + 1) // 2
    tail = room - head
    return text[:head] + marker + (text[-tail:] if tail else "")


def build_candidate_board(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    sample_id: str,
    purpose: str,
    trace_max_chars: int,
    board_max_chars: int,
    restrict_answers: set[str] | None = None,
) -> tuple[str, dict[str, str], dict[str, int]]:
    """构造不显示票数的候选板，并对每次调用独立置换标签和轨迹。"""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        answer = normalized_answer(row)
        if answer and (restrict_answers is None or answer in restrict_answers):
            grouped.setdefault(answer, []).append(row)
    answers = list(grouped)
    rng = random.Random(f"evf-board-v1:{seed}:{sample_id}:{purpose}")
    rng.shuffle(answers)
    labels = [chr(ord("A") + index) for index in range(len(answers))]
    label_to_answer = dict(zip(labels, answers, strict=True))
    blocks: list[str] = []
    trace_counts: dict[str, int] = {}
    for label, answer in label_to_answer.items():
        representatives = list(grouped[answer])
        rng.shuffle(representatives)
        row = representatives[0]
        reasoning = str((row.get("validated_output") or {}).get("reasoning") or row.get("assistant_text") or "")
        excerpt = balanced_excerpt(reasoning, trace_max_chars)
        blocks.append(f"Candidate {label}\nAnswer: {answer}\nReasoning:\n{excerpt}")
        trace_counts[label] = len(excerpt)
    rendered = "\n\n---\n\n".join(blocks)
    if len(rendered) > board_max_chars:
        raise ValueError("EVF candidate board exceeds board_max_chars.")
    return rendered, label_to_answer, trace_counts


def deterministic_challenger_fallback(stage: StageDecision, *, seed: int, sample_id: str) -> str:
    candidates = [answer for answer in stage.vote_counts if answer != stage.anchor_answer]
    if not candidates:
        return ""
    maximum = max(stage.vote_counts[answer] for answer in candidates)
    tied = sorted(answer for answer in candidates if stage.vote_counts[answer] == maximum)
    index = int(hashlib.sha256(f"{seed}:{sample_id}".encode()).hexdigest(), 16) % len(tied)
    return tied[index]


def decide_override(
    *,
    anchor: str,
    challenger: str,
    audits: list[dict[str, Any]],
    challenger_required_passes: int,
    anchor_required_falsifications: int,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not challenger or challenger == anchor:
        return False, ["missing_existing_challenger"]
    if len(audits) < 2 or any(audit.get("preferred_answer") != challenger for audit in audits):
        reasons.append("auditor_disagreement")
    challenger_passes = sum(
        evidence.get("target_answer") == challenger and evidence.get("status") == "pass"
        for audit in audits
        for evidence in audit.get("evidence_results", [])
    )
    anchor_falsifications = sum(
        evidence.get("target_answer") == anchor
        and evidence.get("claim_kind") == "falsify"
        and evidence.get("status") == "pass"
        for audit in audits
        for evidence in audit.get("evidence_results", [])
    )
    challenger_failures = sum(
        evidence.get("target_answer") == challenger and evidence.get("status") == "fail"
        for audit in audits
        for evidence in audit.get("evidence_results", [])
    )
    if challenger_passes < challenger_required_passes:
        reasons.append("insufficient_challenger_evidence")
    if anchor_falsifications < anchor_required_falsifications:
        reasons.append("insufficient_anchor_falsification")
    if challenger_failures:
        reasons.append("challenger_evidence_failed")
    return not reasons, reasons or ["executable_override_gate_passed"]
