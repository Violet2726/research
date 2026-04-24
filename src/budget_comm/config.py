"""budget_comm 实验配置加载。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib

from experiment_core.config import BenchmarkConfig, ResolvedModelConfig, load_benchmark_config, resolve_model_ref


@dataclass(frozen=True)
class BudgetProtocolConfig:
    """DALA-lite v1 的共享协议配置。"""

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
    """same-context / split-context 视图配置。"""

    track_name: str
    strategyqa_mode: str
    hotpotqa_mode: str
    allow_full_context: bool


@dataclass(frozen=True)
class BudgetCommExperimentConfig:
    """budget_comm 顶层实验配置。"""

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
    primary_backbone: str
    raw: dict[str, Any]


def _load_toml(path: str | Path) -> dict[str, Any]:
    """读取 TOML 文件。"""
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def load_protocol_config(path: str | Path) -> BudgetProtocolConfig:
    """加载共享协议配置。"""
    payload = _load_toml(path)
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
    payload = _load_toml(path)
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
    payload = _load_toml(path)
    return ContextViewConfig(
        track_name=str(payload["track_name"]),
        strategyqa_mode=str(payload["strategyqa_mode"]),
        hotpotqa_mode=str(payload["hotpotqa_mode"]),
        allow_full_context=bool(payload["allow_full_context"]),
    )


def load_experiment_config(path: str | Path) -> BudgetCommExperimentConfig:
    """加载 budget_comm 顶层实验配置。"""
    payload = _load_toml(path)
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
        requests_per_minute_limit=_optional_int(payload, "requests_per_minute_limit"),
        tokens_per_minute_limit=_optional_int(payload, "tokens_per_minute_limit"),
        primary_backbone=str(payload["primary_backbone"]),
        raw=payload,
    )


def phase_metadata(experiment: BudgetCommExperimentConfig, phase_name: str) -> dict[str, Any]:
    """返回指定 phase 的原始配置副本。"""
    return dict(experiment.raw["phases"][phase_name])


def load_benchmarks(experiment: BudgetCommExperimentConfig) -> list[BenchmarkConfig]:
    """加载实验声明使用的 benchmark 配置。"""
    return [load_benchmark_config(path) for path in experiment.benchmark_configs]


def resolve_backbone(model_ref: str) -> ResolvedModelConfig:
    """解析 budget_comm 使用的 backbone 模型。"""
    return resolve_model_ref(model_ref)


def _optional_int(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    return int(value)
