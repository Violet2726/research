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
        locked_override_margin=1.0,
        concrete_evidence_min_chars=12,
    )


def _row(
    answer: str,
    *,
    role: str = "cot_builder",
    risk_level: str = "none",
    risk_summary: str = "none",
    evidence: str = "because 2+2=4",
    confidence: float = 0.8,
) -> dict:
    return {
        "dataset": "gpqa_diamond",
        "normalized_answer": answer,
        "prediction": answer,
        "agent_role": role,
        "confidence_value": confidence,
        "key_evidence": evidence,
        "risk_level": risk_level,
        "failure_risk": risk_summary,
        "validated_output": {
            "confidence": confidence,
            "key_evidence": evidence,
            "risk_level": risk_level,
            "risk_summary": risk_summary,
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
        _row("A", risk_level="medium", risk_summary="possible option trap"),
        _row("A"),
        _row("A", risk_level="medium", risk_summary="possible calculation trap"),
        _row(
            "B",
            role="counterfactual_falsifier",
            risk_level="high",
            risk_summary="leading answer misses the requested slot",
            evidence="option B directly matches the requested slot because context says option B",
        ),
    ]

    decision = build_router_decision(rows, protocol=_protocol())

    assert decision.triggered is True
    assert "material_risk_count" in decision.reasons


def test_router_ignores_low_risk_prose_summary() -> None:
    rows = [
        _row("A", risk_level="low", risk_summary="Low risk; all checks agree."),
        _row("A", risk_level="low", risk_summary="Very low; the evidence is direct."),
        _row("A", risk_level="low", risk_summary="None, as the slot is explicit."),
        _row("A", risk_level="none", risk_summary="No major risk."),
        _row("B", role="counterfactual_falsifier", risk_level="low", risk_summary="Low risk; leading answer survives."),
    ]

    decision = build_router_decision(rows, protocol=_protocol())

    assert decision.triggered is False
    assert "material_risk_count" not in decision.reasons


def test_stage_a_uses_evidence_weighting_only_for_weak_splits() -> None:
    rows = [
        _row("A", confidence=0.2, evidence="brief"),
        _row("A", confidence=0.2, evidence="brief"),
        _row("B", confidence=0.95, evidence="option B because direct context span supports B"),
        _row("B", confidence=0.95, evidence="option B because direct calculation supports B"),
    ]

    decision = aggregate_stage_a_vote(rows)

    assert decision.final_answer == "B"
    assert decision.resolver == "cred_vote_5_audit_weighted"


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
        survival_override_margin=10.0,
        concrete_evidence_min_chars=12,
        locked=False,
    )

    assert decision.final_answer == "A"
    assert decision.resolver == "cred_survival_override_rejected"
