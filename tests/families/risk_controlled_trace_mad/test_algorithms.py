from __future__ import annotations

import json

import pytest

from research_experiments.families.risk_controlled_trace_mad.algorithms import (
    FEATURE_NAMES,
    build_feature_vector,
    build_trace_board,
    majority_with_anchor_fallback,
    stage_decision,
    validate_feature_vector,
)
from research_experiments.families.risk_controlled_trace_mad.run.sample import parse_synthesis_output


def _row(agent_id: int, answer: str, reasoning: str = "reasoning") -> dict:
    return {"agent_id": agent_id, "normalized_answer": answer, "validated_output": {"reasoning": reasoning}, "assistant_text": reasoning}


def test_stage_decision_and_five_zero_trigger() -> None:
    unanimous = stage_decision([_row(i, "A") for i in range(1, 6)])
    assert unanimous.anchor_answer == "A"
    assert unanimous.triggered is False
    split = stage_decision([_row(1, "A"), _row(2, "A"), _row(3, "B"), _row(4, "B"), _row(5, "C")])
    assert split.triggered is True
    assert split.disagreement_pattern == "2-2-1"


def test_nine_vote_multiclass_tie_falls_back_to_sc5_anchor() -> None:
    rows = [_row(1, "A"), _row(2, "A"), _row(3, "B"), _row(4, "B"), _row(5, "C"), _row(6, "C"), _row(7, "D"), _row(8, "D"), _row(9, "E")]
    answer, _, resolver = majority_with_anchor_fallback(rows, "A")
    assert answer == "A"
    assert resolver == "anchor_fallback_multiclass_tie"
    answer, _, resolver = majority_with_anchor_fallback(rows, "Z")
    assert answer == "Z"
    assert resolver == "anchor_fallback_multiclass_tie"


def test_board_is_balanced_complete_and_deterministic() -> None:
    rows = [_row(i, str(i), reasoning=(str(i) * 5000)) for i in range(1, 6)]
    first, counts = build_trace_board(rows, seed=42, sample_id="x")
    second, _ = build_trace_board(rows, seed=42, sample_id="x")
    assert first == second
    assert len(first) <= 7000
    assert set(counts) == {"T1", "T2", "T3", "T4", "T5"}
    assert all(f"Trace T{i}" in first for i in range(1, 6))


def test_feature_contract_rejects_forbidden_or_reordered_features() -> None:
    rows = [_row(1, "A"), _row(2, "A"), _row(3, "B"), _row(4, "C"), _row(5, "D")]
    synthesis = {"final_answer": "B", "source_trace_ids": ["T1", "T2"]}
    vector = build_feature_vector(rows, synthesis, {"status": "pass", "certificate_type": "arithmetic"})
    assert tuple(vector) == FEATURE_NAMES
    bad = dict(vector)
    bad["dataset"] = 1.0
    with pytest.raises(ValueError, match="Forbidden"):
        validate_feature_vector(bad)


def test_synthesis_schema_accepts_novel_answer_and_rejects_confidence_only_card() -> None:
    payload = {"reasoning_summary": "combine two decisive steps", "final_answer": "novel", "source_trace_ids": ["T1", "T3"], "decisive_claim": "the invariant holds", "certificate_type": "unsupported", "certificate_payload": {}}
    assert parse_synthesis_output(json.dumps(payload))["final_answer"] == "novel"
    with pytest.raises(ValueError, match="Missing"):
        parse_synthesis_output(json.dumps({"answer": "A", "confidence": 1.0}))


def test_synthesis_schema_normalizes_json_scalar_answer_and_bounded_summary() -> None:
    payload = {
        "reasoning_summary": " ".join(["word"] * 130),
        "final_answer": 42,
        "source_trace_ids": ["T1"],
        "decisive_claim": "direct recomputation",
        "certificate_type": "unsupported",
        "certificate_payload": {},
    }
    parsed = parse_synthesis_output(json.dumps(payload), reasoning_word_limit=120)
    assert parsed["final_answer"] == "42"
    assert len(parsed["reasoning_summary"].split()) == 120
