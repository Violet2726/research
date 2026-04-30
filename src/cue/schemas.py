from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentPacket:
    final_answer: str
    reasoning_sketch: str
    confidence: float | None
    uncertain_point: str | None
    top_claims: list[str]
    evidence_items: list[str]
    counter_answer: str | None


@dataclass(frozen=True)
class ConflictObject:
    conflict_type: str
    claim_a: str
    claim_b: str
    support_a: list[str]
    support_b: list[str]
    specificity: float
    estimated_comm_cost: float
    evidence_gap: float
    message_type: str


@dataclass(frozen=True)
class UtilityBreakdown:
    correction_potential: float
    resolvability: float
    collapse_risk: float
    normalized_cost: float
    utility: float

