"""CUE 运行链路中复用的结构化载荷模式。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentPacket:
    """Stage A 或通信轮里单个 agent 对外暴露的压缩状态。"""

    final_answer: str
    reasoning_sketch: str
    confidence: float | None
    uncertain_point: str | None
    top_claims: list[str]
    evidence_items: list[str]
    counter_answer: str | None


@dataclass(frozen=True)
class ConflictObject:
    """CUE 对“局部冲突”进行结构化压缩后的表示。"""

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
    """策略触发时使用的效用分解项。"""

    correction_potential: float
    resolvability: float
    collapse_risk: float
    normalized_cost: float
    utility: float

