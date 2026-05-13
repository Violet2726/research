
"""覆盖选择性通信信号聚合与触发决策行为。"""

from __future__ import annotations

from research_experiments.core.controls.selective_signals import (
    decide_trigger,
    summarize_confidence_rows,
    summarize_divergence_rows,
)


def test_selective_signal_summary_and_decision() -> None:
    summary = summarize_confidence_rows(
        [
            {"agent_id": 1, "confidence_valid": True, "confidence_value": 0.9},
            {"agent_id": 2, "confidence_valid": True, "confidence_value": 0.4},
            {"agent_id": 3, "confidence_valid": False, "confidence_value": None},
        ]
    )
    assert summary.mean_confidence == 0.65
    assert summary.confidence_spread == 0.5
    assert summary.invalid_agent_ids == [3]
    decision = decide_trigger(
        trigger_type="hybrid_trigger",
        initial_disagreement=False,
        mean_confidence=summary.mean_confidence,
        confidence_spread=summary.confidence_spread,
        any_invalid_confidence=summary.any_invalid_confidence,
        fail_open_to_always=False,
    )
    assert decision.triggered is True

def test_missing_confidence_is_not_counted_as_invalid() -> None:
    summary = summarize_confidence_rows(
        [
            {"agent_id": 1, "confidence_valid": False, "confidence_value": None, "confidence_source": "missing"},
            {"agent_id": 2, "confidence_valid": False, "confidence_value": None, "confidence_source": "missing"},
            {"agent_id": 3, "confidence_valid": False, "confidence_value": None, "confidence_source": "missing"},
        ]
    )
    assert summary.invalid_agent_ids == []
    assert summary.any_invalid_confidence is False
    assert summary.mean_confidence is None

def test_selective_divergence_summary_supports_voc_signals() -> None:
    summary = summarize_divergence_rows(
        [
            {
                "normalized_answer": "A",
                "claim_span": "shirt 17.5 shorts 24.5 total 42",
                "uncertainty_type": "calculation",
            },
            {
                "normalized_answer": "A",
                "claim_span": "50 minus 42 leaves 8",
                "uncertainty_type": "evidence_selection",
            },
            {
                "normalized_answer": "A",
                "claim_span": "50 minus 42 leaves 8",
                "uncertainty_type": "evidence_selection",
            },
        ]
    )
    assert summary.answer_unique_count == 1
    assert summary.answer_divergence_score == 0.0
    assert 0.0 <= summary.claim_similarity_mean <= 1.0
    assert summary.claim_divergence_score > 0.55
    assert summary.uncertainty_type_diversity_score == 0.5

def test_voc_trigger_v2_requires_spread_for_confidence_only_case() -> None:
    decision = decide_trigger(
        trigger_type="voc_trigger_v2",
        initial_disagreement=False,
        answer_divergence_score=0.0,
        claim_divergence_score=0.2,
        uncertainty_type_diversity_score=0.0,
        mean_confidence=0.94,
        confidence_spread=0.05,
        any_invalid_confidence=False,
        fail_open_to_always=False,
    )
    assert decision.triggered is False

def test_voc_trigger_v2_ignores_missing_confidence_when_other_signals_are_weak() -> None:
    decision = decide_trigger(
        trigger_type="voc_trigger_v2",
        initial_disagreement=False,
        answer_divergence_score=0.0,
        claim_divergence_score=0.1,
        uncertainty_type_diversity_score=0.0,
        mean_confidence=None,
        confidence_spread=None,
        any_invalid_confidence=True,
        fail_open_to_always=True,
    )
    assert decision.triggered is False
    assert decision.fail_open_applied is False

