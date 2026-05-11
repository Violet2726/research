"""`budget_comm` 实验配置加载。

本模块负责解析 DALA-lite 风格预算通信实验所需的配置对象，
包括共享协议、拍卖策略、上下文视图以及顶层实验定义。
这些配置共同决定了“谁看见什么上下文”“消息如何压缩”“预算如何分配”。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiment_core.foundation.config import BenchmarkConfig, ResolvedModelConfig
from experiment_core.foundation.config_helpers import (
    load_benchmarks as load_benchmarks_from_experiment,
    load_toml,
    optional_int,
    phase_metadata as phase_metadata_from_raw,
    resolve_model as resolve_shared_model,
)


@dataclass(frozen=True)
class BudgetProtocolConfig:
    """DALA-lite v1 的共享通信协议配置。"""

    agent_count: int
    debate_rounds: int
    initial_temperature: float
    debate_temperature: float
    top_p: float
    max_output_tokens: int


@dataclass(frozen=True)
class AuctionPolicyConfig:
    """预算拍卖与 value density 代理规则。"""

    calibration_fraction: float
    disagreement_weight: float
    evidence_weight: float
    novelty_weight: float
    confidence_weight: float
    positive_density_threshold: float
    full_token_cap: int
    summary_token_cap: int
    keywords_token_cap: int


@dataclass(frozen=True)
class ContextViewConfig:
    """`same-context / split-context` 样本视图配置。"""

    track_name: str
    strategyqa_mode: str
    hotpotqa_mode: str
    allow_full_context: bool


@dataclass(frozen=True)
class BudgetCommExperimentConfig:
    """`budget_comm` 顶层实验配置。"""

    name: str
    description: str
    benchmark_configs: list[Path]
    protocol: Path
    auction_policy: Path
    context_view: Path
    global_seed: int
    prompt_version: str
    calibration_sample_size: int
    max_concurrent_requests: int
    requests_per_minute_limit: int | None
    tokens_per_minute_limit: int | None
    primary_model_ref: str
    raw: dict[str, Any]


def load_protocol_config(path: str | Path) -> BudgetProtocolConfig:
    """加载共享协议配置。"""
    payload = load_toml(path)
    return BudgetProtocolConfig(
        agent_count=int(payload["agent_count"]),
        debate_rounds=int(payload["debate_rounds"]),
        initial_temperature=float(payload["initial_temperature"]),
        debate_temperature=float(payload["debate_temperature"]),
        top_p=float(payload["top_p"]),
        max_output_tokens=int(payload["max_output_tokens"]),
    )


def load_auction_policy_config(path: str | Path) -> AuctionPolicyConfig:
    """加载 value density 与预算规则配置。"""
    payload = load_toml(path)
    return AuctionPolicyConfig(
        calibration_fraction=float(payload["calibration_fraction"]),
        disagreement_weight=float(payload["disagreement_weight"]),
        evidence_weight=float(payload["evidence_weight"]),
        novelty_weight=float(payload["novelty_weight"]),
        confidence_weight=float(payload["confidence_weight"]),
        positive_density_threshold=float(payload.get("positive_density_threshold", 0.0)),
        full_token_cap=int(payload["full_token_cap"]),
        summary_token_cap=int(payload["summary_token_cap"]),
        keywords_token_cap=int(payload["keywords_token_cap"]),
    )


def load_context_view_config(path: str | Path) -> ContextViewConfig:
    """加载上下文视图轨道配置。"""
    payload = load_toml(path)
    return ContextViewConfig(
        track_name=str(payload["track_name"]),
        strategyqa_mode=str(payload["strategyqa_mode"]),
        hotpotqa_mode=str(payload["hotpotqa_mode"]),
        allow_full_context=bool(payload["allow_full_context"]),
    )


def load_experiment_config(path: str | Path) -> BudgetCommExperimentConfig:
    """加载 `budget_comm` 顶层实验配置。"""
    payload = load_toml(path)
    return BudgetCommExperimentConfig(
        name=str(payload["name"]),
        description=str(payload["description"]),
        benchmark_configs=[Path(item) for item in payload["benchmark_configs"]],
        protocol=Path(payload["protocol"]),
        auction_policy=Path(payload["auction_policy"]),
        context_view=Path(payload["context_view"]),
        global_seed=int(payload["global_seed"]),
        prompt_version=str(payload["prompt_version"]),
        calibration_sample_size=int(payload["calibration_sample_size"]),
        max_concurrent_requests=int(payload["max_concurrent_requests"]),
        requests_per_minute_limit=optional_int(payload, "requests_per_minute_limit"),
        tokens_per_minute_limit=optional_int(payload, "tokens_per_minute_limit"),
        primary_model_ref=str(payload["primary_model_ref"]),
        raw=payload,
    )


def phase_metadata(experiment: BudgetCommExperimentConfig, phase_name: str) -> dict[str, Any]:
    """返回指定 phase 的原始配置副本。"""
    return phase_metadata_from_raw(experiment, phase_name)


def load_benchmarks(experiment: BudgetCommExperimentConfig) -> list[BenchmarkConfig]:
    """加载实验声明使用的 benchmark 配置。"""
    return load_benchmarks_from_experiment(experiment)


def resolve_model(model_ref: str) -> ResolvedModelConfig:
    """解析 `budget_comm` 使用的 backbone 模型。"""
    return resolve_shared_model(model_ref)

