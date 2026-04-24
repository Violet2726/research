from __future__ import annotations

from sparc.logic import (
    aggregate_with_confidence_tiebreak,
    project_message_packet,
    select_audit_candidate_pair,
)


def test_project_message_packet_degrades_when_claim_span_missing() -> None:
    packet = project_message_packet(
        {
            "validated_output": {"final_answer": "42"},
            "confidence_raw_display": 0.8,
            "confidence_valid": True,
            "confidence_value": 0.8,
            "reasoning_trace": "x" * 500,
            "claim_span": "",
            "key_evidence": "evidence",
        },
        "disagreement_step_only",
    )
    assert packet["effective_message_mode"] == "answer_confidence"
    assert packet["approx_packet_tokens"] <= 32


def test_project_message_packet_respects_full_cot_cap() -> None:
    packet = project_message_packet(
        {
            "validated_output": {"final_answer": "42"},
            "confidence_raw_display": 0.7,
            "confidence_valid": True,
            "confidence_value": 0.7,
            "reasoning_trace": "reason " * 300,
            "claim_span": "claim",
            "key_evidence": "evidence " * 200,
        },
        "full_cot",
    )
    assert packet["approx_packet_tokens"] <= 192


def test_select_audit_candidate_pair_prefers_majority_vs_minority() -> None:
    pair = select_audit_candidate_pair(
        [
            {"agent_id": 1, "normalized_answer": "yes", "confidence_valid": True, "confidence_value": 0.6},
            {"agent_id": 2, "normalized_answer": "yes", "confidence_valid": True, "confidence_value": 0.8},
            {"agent_id": 3, "normalized_answer": "no", "confidence_valid": True, "confidence_value": 0.9},
        ]
    )
    assert pair["pair_type"] == "two_way_majority"
    assert pair["candidate_a"]["agent_id"] == 2
    assert pair["candidate_b"]["agent_id"] == 3


def test_select_audit_candidate_pair_handles_three_way_conflict() -> None:
    pair = select_audit_candidate_pair(
        [
            {"agent_id": 1, "normalized_answer": "a", "confidence_valid": True, "confidence_value": 0.2},
            {"agent_id": 2, "normalized_answer": "b", "confidence_valid": True, "confidence_value": 0.9},
            {"agent_id": 3, "normalized_answer": "c", "confidence_valid": True, "confidence_value": 0.8},
        ]
    )
    assert pair["pair_type"] == "three_way"
    assert {pair["candidate_a"]["agent_id"], pair["candidate_b"]["agent_id"]} == {2, 3}


def test_select_audit_candidate_pair_skips_consensus() -> None:
    pair = select_audit_candidate_pair(
        [
            {"agent_id": 1, "normalized_answer": "yes", "confidence_valid": True, "confidence_value": 0.6},
            {"agent_id": 2, "normalized_answer": "yes", "confidence_valid": True, "confidence_value": 0.8},
            {"agent_id": 3, "normalized_answer": "yes", "confidence_valid": True, "confidence_value": 0.7},
        ]
    )
    assert pair["skipped"] is True
    assert pair["skip_reason"] == "consensus"
    assert pair["fallback_answer"] == "yes"


def test_aggregate_with_confidence_tiebreak_uses_confidence_then_agent_id() -> None:
    winner, counts = aggregate_with_confidence_tiebreak(
        [
            {"agent_id": 1, "normalized_answer": "a", "confidence_valid": True, "confidence_value": 0.7},
            {"agent_id": 2, "normalized_answer": "b", "confidence_valid": True, "confidence_value": 0.9},
        ]
    )
    assert counts == {"a": 1, "b": 1}
    assert winner == "b"
