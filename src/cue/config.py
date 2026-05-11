"""CUE 实验的配置加载层。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiment_core.foundation.config import BenchmarkConfig, ResolvedModelConfig
from experiment_core.foundation.config_helpers import (
    load_benchmarks,
    load_toml,
    optional_int,
    phase_metadata,
    resolve_model,
)
from experiment_core.foundation.methods import MethodConfig, load_method_catalog


@dataclass(frozen=True)
class CueProtocolConfig:
    """CUE 共享协议参数。"""

    agent_count: int
    debate_rounds: int
    initial_temperature: float
    debate_temperature: float
    top_p: float
    max_output_tokens: int
    message_token_cap: int
    audit_token_cap: int


@dataclass(frozen=True)
class CuePolicyConfig:
    """CUE 触发与审计效用策略配置。"""

    policy_name: str
    trigger_type: str
    tau: float
    normalized_cost_divisor: float
    collapse_weight: float
    cost_weight: float
    enable_audit: bool
    freeze_confidence_threshold: float
    freeze_claim_conflict_threshold: float
    freeze_evidence_gap_threshold: float
    correction_answer_entropy_weight: float
    correction_low_confidence_weight: float
    correction_confidence_spread_weight: float
    correction_claim_conflict_weight: float
    correction_fragile_consensus_weight: float
    resolvability_specificity_weight: float
    resolvability_evidence_overlap_weight: float
    collapse_format_risk_weight: float
    collapse_vagueness_risk_weight: float
    collapse_majority_pressure_weight: float


@dataclass(frozen=True)
class CueExperimentConfig:
    """CUE 实验入口配置。"""

    name: str
    description: str
    benchmark_configs: list[Path]
    protocol: Path
    policy_configs: list[Path]
    control_catalog: Path | None
    global_seed: int
    prompt_version: str
    max_concurrent_requests: int
    requests_per_minute_limit: int | None
    tokens_per_minute_limit: int | None
    primary_model_ref: str
    raw: dict[str, Any]


def load_protocol_config(path: str | Path) -> CueProtocolConfig:
    """加载 CUE 协议配置。"""
    payload = load_toml(path)
    return CueProtocolConfig(
        agent_count=int(payload["agent_count"]),
        debate_rounds=int(payload["debate_rounds"]),
        initial_temperature=float(payload["initial_temperature"]),
        debate_temperature=float(payload["debate_temperature"]),
        top_p=float(payload["top_p"]),
        max_output_tokens=int(payload["max_output_tokens"]),
        message_token_cap=int(payload.get("message_token_cap", 100)),
        audit_token_cap=int(payload.get("audit_token_cap", 120)),
    )


def load_policy_config(path: str | Path) -> CuePolicyConfig:
    """加载单个 CUE 策略配置。"""
    payload = load_toml(path)
    return CuePolicyConfig(
        policy_name=str(payload["policy_name"]),
        trigger_type=str(payload["trigger_type"]),
        tau=float(payload.get("tau", 0.0)),
        normalized_cost_divisor=float(payload.get("normalized_cost_divisor", 400.0)),
        collapse_weight=float(payload.get("collapse_weight", 0.25)),
        cost_weight=float(payload.get("cost_weight", 0.35)),
        enable_audit=bool(payload.get("enable_audit", True)),
        freeze_confidence_threshold=float(payload.get("freeze_confidence_threshold", 0.8)),
        freeze_claim_conflict_threshold=float(payload.get("freeze_claim_conflict_threshold", 0.25)),
        freeze_evidence_gap_threshold=float(payload.get("freeze_evidence_gap_threshold", 0.30)),
        correction_answer_entropy_weight=float(payload.get("correction_answer_entropy_weight", 0.30)),
        correction_low_confidence_weight=float(payload.get("correction_low_confidence_weight", 0.20)),
        correction_confidence_spread_weight=float(payload.get("correction_confidence_spread_weight", 0.15)),
        correction_claim_conflict_weight=float(payload.get("correction_claim_conflict_weight", 0.20)),
        correction_fragile_consensus_weight=float(payload.get("correction_fragile_consensus_weight", 0.15)),
        resolvability_specificity_weight=float(payload.get("resolvability_specificity_weight", 0.60)),
        resolvability_evidence_overlap_weight=float(payload.get("resolvability_evidence_overlap_weight", 0.40)),
        collapse_format_risk_weight=float(payload.get("collapse_format_risk_weight", 0.40)),
        collapse_vagueness_risk_weight=float(payload.get("collapse_vagueness_risk_weight", 0.30)),
        collapse_majority_pressure_weight=float(payload.get("collapse_majority_pressure_weight", 0.30)),
    )


def load_policies(paths: list[str | Path]) -> list[CuePolicyConfig]:
    """按顺序加载多份 CUE 策略。"""
    return [load_policy_config(path) for path in paths]


def load_control_catalog(path: str | Path | None) -> dict[str, MethodConfig]:
    """加载控制组方法目录；未配置时返回空字典。"""
    if path is None:
        return {}
    resolved = Path(path)
    if not resolved.exists():
        return {}
    return load_method_catalog(resolved)


def load_experiment_config(path: str | Path) -> CueExperimentConfig:
    """加载 CUE 实验入口配置。"""
    payload = load_toml(path)
    control_catalog = payload.get("control_catalog")
    return CueExperimentConfig(
        name=str(payload["name"]),
        description=str(payload["description"]),
        benchmark_configs=[Path(item) for item in payload["benchmark_configs"]],
        protocol=Path(payload["protocol"]),
        policy_configs=[Path(item) for item in payload["policy_configs"]],
        control_catalog=Path(control_catalog) if control_catalog else None,
        global_seed=int(payload["global_seed"]),
        prompt_version=str(payload["prompt_version"]),
        max_concurrent_requests=int(payload["max_concurrent_requests"]),
        requests_per_minute_limit=optional_int(payload, "requests_per_minute_limit"),
        tokens_per_minute_limit=optional_int(payload, "tokens_per_minute_limit"),
        primary_model_ref=str(payload["primary_model_ref"]),
        raw=payload,
    )


