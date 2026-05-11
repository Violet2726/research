"""共享结构化输出契约。

本包以“语义 schema”而不是以具体 family 命名来组织结构化输出校验、
轻量恢复与软拒绝回退逻辑，避免共享核心继续吸收实验家族语义。
"""

from __future__ import annotations

from experiment_core.structured_outputs.registry import (
    ARTIFACT_VERSION,
    SCHEMA_ANSWER_CORE,
    SCHEMA_ANSWER_WITH_PROXY_SIGNALS_BUDGET,
    SCHEMA_ANSWER_WITH_PROXY_SIGNALS_DELIBERATION,
    SCHEMA_ANSWER_WITH_PROXY_SIGNALS_SELECTIVE,
    SCHEMA_AUDIT_VERDICT,
    SCHEMA_BELIEF_UPDATE_DELTA,
    SCHEMA_CUE_BLACKBOX_PACKET,
    SCHEMA_DELIBERATION_PACKET,
    SCHEMA_SPLIT_CONTEXT_BELIEF,
    SCHEMA_SPLIT_CONTEXT_SOLVER,
    SchemaId,
    parse_proxy_signal_answer,
    validate_or_recover_structured_output,
    validate_structured_output,
)


__all__ = [
    "ARTIFACT_VERSION",
    "SchemaId",
    "SCHEMA_ANSWER_CORE",
    "SCHEMA_ANSWER_WITH_PROXY_SIGNALS_BUDGET",
    "SCHEMA_ANSWER_WITH_PROXY_SIGNALS_DELIBERATION",
    "SCHEMA_ANSWER_WITH_PROXY_SIGNALS_SELECTIVE",
    "SCHEMA_AUDIT_VERDICT",
    "SCHEMA_BELIEF_UPDATE_DELTA",
    "SCHEMA_CUE_BLACKBOX_PACKET",
    "SCHEMA_DELIBERATION_PACKET",
    "SCHEMA_SPLIT_CONTEXT_BELIEF",
    "SCHEMA_SPLIT_CONTEXT_SOLVER",
    "parse_proxy_signal_answer",
    "validate_or_recover_structured_output",
    "validate_structured_output",
]
