from __future__ import annotations

from research_experiments.families.cred_mad.algorithms import (
    aggregate_stage_a_vote,
    aggregate_survival,
    build_router_decision,
)
from research_experiments.families.cred_mad.config import CredMadProtocolConfig


def _protocol() -> CredMadProtocolConfig:
    return CredMadProtocolConfig(
        stage_a_agent_count=5,
        max_refutations=2,
        stage_a_temperature=0.7,
        debate_temperature=0.4,
        judge_temperature=0.0,
        top_p=1.0,
        strong_majority_count=4,
        min_evidence_quality=0.45,
        risk_trigger_count=2,
        weak_majority_count=3,
        override_margin=0.75,
        locked_override_margin=1.0,
        concrete_evidence_min_chars=12,
    )


def _row(answer: str, *, role: str = "cot_builder", risk: str = "none", evidence: str = "because 2+2=4") -> dict:
    return {
        "normalized_answer": answer,
        "prediction": answer,
        "agent_role": role,
        "confidence_value": 0.8,
        "key_evidence": evidence,
        "failure_risk": risk,
        "validated_output": {
            "confidence": 0.8,
            "key_evidence": evidence,
            "failure_risk": risk,
        },
    }


def test_router_skips_clean_strong_majority() -> None:
    rows = [_row("A"), _row("A"), _row("A"), _row("A"), _row("B", role="counterfactual_falsifier")]

    decision = build_router_decision(rows, protocol=_protocol())

    assert decision.triggered is False
    assert decision.leading_answer == "A"


def test_router_triggers_on_material_risk() -> None:
    rows = [
        _row("A"),
        _row("A", risk="possible option trap"),
        _row("A"),
        _row("A"),
        _row("B", role="counterfactual_falsifier", risk="leading answer misses the requested slot"),
    ]

    decision = build_router_decision(rows, protocol=_protocol())

    assert decision.triggered is True
    assert "material_risk_count" in decision.reasons


def test_survival_requires_margin_and_concrete_evidence_to_override() -> None:
    stage_rows = [_row("A"), _row("B"), _row("A"), _row("B"), _row("C")]
    stage = aggregate_stage_a_vote(stage_rows)
    refute = [
        _row("B", role="refuter", evidence="specific contradiction in option B evidence"),
        _row("B", role="refuter", evidence="another concrete contradiction in option B evidence"),
    ]
    judge = _row("B", role="judge", evidence="judge accepts the concrete contradiction")

    decision = aggregate_survival(
        dataset="gpqa_diamond",
        stage_rows=stage_rows,
        refutation_rows=refute,
        defense_rows=[],
        judge_row=judge,
        stage_winner=stage.final_answer,
        override_margin=10.0,
        concrete_evidence_min_chars=12,
        locked=False,
    )

    assert decision.final_answer == "A"
    assert decision.resolver == "cred_survival_override_rejected"
