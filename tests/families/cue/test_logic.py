"""覆盖 CUE 信号汇总、效用与聚合逻辑的测试。"""

from research_experiments.families.cue.algorithms import build_conflict_object, compute_utility, summarize_cue_signals


def _row(agent_id: int, answer: str, confidence: float, claims: list[str], evidence: list[str]) -> dict:
    return {
        "agent_id": agent_id,
        "final_answer": answer,
        "normalized_answer": answer,
        "confidence_value": confidence,
        "confidence_valid": True,
        "reasoning_sketch": "short reasoning",
        "top_claims": claims,
        "evidence_items": evidence,
        "counter_answer": None,
    }


class _Policy:
    correction_answer_entropy_weight = 0.30
    correction_low_confidence_weight = 0.20
    correction_confidence_spread_weight = 0.15
    correction_claim_conflict_weight = 0.20
    correction_fragile_consensus_weight = 0.15
    resolvability_specificity_weight = 0.60
    resolvability_evidence_overlap_weight = 0.40
    collapse_format_risk_weight = 0.40
    collapse_vagueness_risk_weight = 0.30
    collapse_majority_pressure_weight = 0.30
    normalized_cost_divisor = 420.0
    cost_weight = 0.35
    collapse_weight = 0.25


def test_conflict_object_prefers_answer_or_step_conflict() -> None:
    rows = [
        _row(1, "42", 0.88, ["use subtraction"], ["12 minus 3"]),
        _row(2, "39", 0.71, ["use multiplication"], ["3 times 13"]),
        _row(3, "42", 0.79, ["use subtraction"], ["12 minus 3"]),
    ]
    signals = summarize_cue_signals(rows, 100)
    conflict = build_conflict_object(rows, signals, 100)
    assert conflict.conflict_type in {"answer", "step", "evidence"}
    assert conflict.message_type in {"answer_critique", "step_check", "evidence_packet"}


def test_utility_is_bounded_and_numeric() -> None:
    rows = [
        _row(1, "yes", 0.55, ["claim a"], ["fact a"]),
        _row(2, "no", 0.65, ["claim b"], ["fact b"]),
        _row(3, "yes", 0.60, ["claim a"], ["fact a"]),
    ]
    signals = summarize_cue_signals(rows, 100)
    conflict = build_conflict_object(rows, signals, 100)
    utility = compute_utility(signals, conflict, _Policy())
    assert isinstance(utility.utility, float)
    assert 0.0 <= utility.normalized_cost <= 1.0
