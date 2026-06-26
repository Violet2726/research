from __future__ import annotations

from research_experiments.families.cred_v.algorithms import (
    aggregate_safe_verification,
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


def test_safe_verification_blocks_same_model_promotion() -> None:
    stage_rows = [
        _row("gpqa_diamond", "A", confidence=0.80),
        _row("gpqa_diamond", "A", confidence=0.75),
        _row("gpqa_diamond", "A", confidence=0.70),
        _row("gpqa_diamond", "B", confidence=0.90),
        _row("gpqa_diamond", "B", confidence=0.85),
    ]
    same_model_verifier = [
        _row(
            "gpqa_diamond",
            "B",
            confidence=0.95,
            payload={
                "promote": True,
                "leader_pass": False,
                "challenger_pass": True,
                "key_evidence": "independent verifier claims option B passes and option A fails",
            },
        )
    ]

    decision = aggregate_safe_verification(
        dataset="gpqa_diamond",
        question="Which option is best?",
        context="",
        stage_rows=stage_rows,
        verifier_rows=same_model_verifier,
        hetero_verifier_rows=[],
        stage_winner="A",
        verification_modes=("hetero_verified",),
        allow_same_model_promotion=False,
        concrete_evidence_min_chars=12,
        strong_majority_count=4,
    )

    assert decision.final_answer == "A"
    assert decision.changed is False
    assert decision.resolver == "cred_verify_safe_rejected"


def test_safe_math_equivalence_repair_promotes_canonical_interval() -> None:
    stage_rows = [
        _row("math500", "(2,infinity)", confidence=0.70),
        _row("math500", "(2,infinity)", confidence=0.69),
        _row("math500", "(2,infinity)", confidence=0.68),
        _row("math500", "(2,\\infty)", confidence=0.90),
        _row("math500", "(2,\\infty)", confidence=0.89),
    ]

    decision = aggregate_safe_verification(
        dataset="math500",
        question="Solve the interval.",
        context="",
        stage_rows=stage_rows,
        verifier_rows=[],
        hetero_verifier_rows=[],
        stage_winner="(2,infinity)",
        verification_modes=("deterministic_repair", "tool_verified"),
        allow_same_model_promotion=False,
        concrete_evidence_min_chars=12,
        strong_majority_count=4,
    )

    assert decision.final_answer == "(2,\\infty)"
    assert decision.changed is True
    assert decision.resolver == "cred_verify_safe_deterministic_repair"


def test_safe_hotpot_span_repair_requires_context_supported_complete_span() -> None:
    stage_rows = [
        _row("hotpotqa", "John Underhill", confidence=0.72),
        _row("hotpotqa", "John Underhill", confidence=0.71),
        _row("hotpotqa", "John Underhill", confidence=0.70),
        _row("hotpotqa", "Captain John Underhill", confidence=0.88),
    ]

    decision = aggregate_safe_verification(
        dataset="hotpotqa",
        question="Who led the expedition?",
        context="The expedition was led by Captain John Underhill before the colony changed command.",
        stage_rows=stage_rows,
        verifier_rows=[],
        hetero_verifier_rows=[],
        stage_winner="John Underhill",
        verification_modes=("deterministic_repair", "tool_verified"),
        allow_same_model_promotion=False,
        concrete_evidence_min_chars=12,
        strong_majority_count=4,
    )

    assert decision.final_answer == "Captain John Underhill"
    assert decision.changed is True

    unsupported = aggregate_safe_verification(
        dataset="hotpotqa",
        question="Who led the expedition?",
        context="The expedition was led by John Underhill before the colony changed command.",
        stage_rows=stage_rows,
        verifier_rows=[],
        hetero_verifier_rows=[],
        stage_winner="John Underhill",
        verification_modes=("deterministic_repair", "tool_verified"),
        allow_same_model_promotion=False,
        concrete_evidence_min_chars=12,
        strong_majority_count=4,
    )

    assert unsupported.final_answer == "John Underhill"
    assert unsupported.changed is False


def test_safe_mmlu_requires_hetero_agreement() -> None:
    stage_rows = [
        _row("mmlu_pro", "A", confidence=0.75),
        _row("mmlu_pro", "A", confidence=0.74),
        _row("mmlu_pro", "A", confidence=0.73),
        _row("mmlu_pro", "B", confidence=0.90),
        _row("mmlu_pro", "B", confidence=0.89),
    ]
    hetero_verifier = [
        _row(
            "mmlu_pro",
            "B",
            confidence=0.92,
            payload={
                "promote": True,
                "leader_pass": False,
                "challenger_pass": True,
                "key_evidence": "heterogeneous verifier confirms option B and identifies the failed clue in option A",
            },
        )
    ]

    without_hetero = aggregate_safe_verification(
        dataset="mmlu_pro",
        question="Which option is best?",
        context="",
        stage_rows=stage_rows,
        verifier_rows=[],
        hetero_verifier_rows=[],
        stage_winner="A",
        verification_modes=("hetero_verified",),
        allow_same_model_promotion=False,
        concrete_evidence_min_chars=12,
        strong_majority_count=4,
    )
    with_hetero = aggregate_safe_verification(
        dataset="mmlu_pro",
        question="Which option is best?",
        context="",
        stage_rows=stage_rows,
        verifier_rows=[],
        hetero_verifier_rows=hetero_verifier,
        stage_winner="A",
        verification_modes=("hetero_verified",),
        allow_same_model_promotion=False,
        concrete_evidence_min_chars=12,
        strong_majority_count=4,
    )

    assert without_hetero.final_answer == "A"
    assert without_hetero.changed is False
    assert with_hetero.final_answer == "B"
    assert with_hetero.changed is True
    assert with_hetero.resolver == "cred_verify_safe_hetero_promoted"


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
