from __future__ import annotations

from research_experiments.families.cred_v.algorithms import (
    aggregate_stage_a_vote,
    aggregate_task_verification,
    select_verification_targets,
)


def test_task_verifier_promotes_supported_challenger() -> None:
    stage_rows = [
        _row("gpqa_diamond", "A", confidence=0.62),
        _row("gpqa_diamond", "A", confidence=0.61),
        _row("gpqa_diamond", "A", confidence=0.60),
        _row("gpqa_diamond", "B", confidence=0.74),
        _row("gpqa_diamond", "B", confidence=0.73),
    ]
    verifier_rows = [
        _row(
            "gpqa_diamond",
            "B",
            confidence=0.84,
            payload={
                "promote": True,
                "leader_score": 0.25,
                "challenger_score": 0.88,
                "key_evidence": "option B matches the decisive concept while option A misses the stated constraint",
            },
        )
    ]

    decision = aggregate_task_verification(
        dataset="gpqa_diamond",
        stage_rows=stage_rows,
        verifier_rows=verifier_rows,
        stage_winner="A",
        promotion_confidence_min=0.72,
        promotion_score_margin=0.15,
        concrete_evidence_min_chars=12,
    )

    assert decision.final_answer == "B"
    assert decision.changed is True
    assert decision.resolver == "cred_v_task_verify_promoted"


def test_task_verifier_rejects_weak_certificate() -> None:
    stage_rows = [
        _row("gpqa_diamond", "A"),
        _row("gpqa_diamond", "A"),
        _row("gpqa_diamond", "A"),
        _row("gpqa_diamond", "B"),
        _row("gpqa_diamond", "B"),
    ]
    verifier_rows = [
        _row(
            "gpqa_diamond",
            "B",
            confidence=0.90,
            payload={
                "promote": True,
                "leader_score": 0.70,
                "challenger_score": 0.78,
                "key_evidence": "thin",
            },
        )
    ]

    decision = aggregate_task_verification(
        dataset="gpqa_diamond",
        stage_rows=stage_rows,
        verifier_rows=verifier_rows,
        stage_winner="A",
        promotion_confidence_min=0.72,
        promotion_score_margin=0.15,
        concrete_evidence_min_chars=12,
    )

    assert decision.final_answer == "A"
    assert decision.changed is False
    assert decision.resolver == "cred_v_task_verify_rejected"


def test_select_verification_targets_excludes_current_leader() -> None:
    stage_rows = [
        _row("gpqa_diamond", "A", confidence=0.90),
        _row("gpqa_diamond", "A", confidence=0.70),
        _row("gpqa_diamond", "B", confidence=0.80),
        _row("gpqa_diamond", "C", confidence=0.65),
    ]

    targets = select_verification_targets(
        dataset="gpqa_diamond",
        rows=stage_rows,
        leading_answer=aggregate_stage_a_vote(stage_rows).final_answer,
        max_verifications=1,
    )

    assert [row["normalized_answer"] for row in targets] == ["B"]


def _row(dataset: str, answer: str, *, confidence: float = 0.5, payload: dict | None = None) -> dict:
    validated_output = {
        "answer": answer,
        "final_answer": answer,
        "confidence": confidence,
        "key_evidence": "option clue supports the answer",
        "risk_level": "none",
    }
    if payload:
        validated_output.update(payload)
    return {
        "dataset": dataset,
        "prediction": answer,
        "normalized_answer": answer,
        "confidence_value": confidence,
        "key_evidence": validated_output.get("key_evidence", ""),
        "risk_level": validated_output.get("risk_level", "none"),
        "validated_output": validated_output,
    }
