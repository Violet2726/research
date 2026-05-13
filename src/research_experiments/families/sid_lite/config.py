"""SID-lite 实验配置加载。

本模块负责解析 SID-lite 所需的共享协议、顶层实验配置与 phase 元数据，
为“高置信一致时早退、否则进入压缩通信”的机制实验提供统一配置入口。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research_experiments.core.config import BenchmarkConfig, ResolvedModelConfig
from research_experiments.families.shared.config_loading import (
    load_benchmarks,
    load_toml,
    optional_int,
    phase_metadata,
    resolve_model,
)


@dataclass(frozen=True)
class SidLiteProtocolConfig:
    """SID-lite 共享协议配置。"""

    agent_count: int
    debate_rounds: int
    initial_temperature: float
    debate_temperature: float
    top_p: float
    max_output_tokens: int
    mean_conf_threshold: float
    conf_spread_threshold: float
    full_token_cap: int
    compressed_token_cap: int


@dataclass(frozen=True)
class SidLiteExperimentConfig:
    """SID-lite 顶层实验配置。"""

    name: str
    description: str
    benchmark_configs: list[Path]
    protocol: Path
    methods: list[str]
    global_seed: int
    prompt_version: str
    max_concurrent_requests: int
    requests_per_minute_limit: int | None
    tokens_per_minute_limit: int | None
    primary_model_ref: str
    raw: dict[str, Any]


def load_protocol_config(path: str | Path) -> SidLiteProtocolConfig:
    """加载 SID-lite 协议配置。"""
    payload = load_toml(path)
    return SidLiteProtocolConfig(
        agent_count=int(payload["agent_count"]),
        debate_rounds=int(payload["debate_rounds"]),
        initial_temperature=float(payload["initial_temperature"]),
        debate_temperature=float(payload["debate_temperature"]),
        top_p=float(payload["top_p"]),
        max_output_tokens=int(payload["max_output_tokens"]),
        mean_conf_threshold=float(payload["mean_conf_threshold"]),
        conf_spread_threshold=float(payload["conf_spread_threshold"]),
        full_token_cap=int(payload["full_token_cap"]),
        compressed_token_cap=int(payload["compressed_token_cap"]),
    )


def load_experiment_config(path: str | Path) -> SidLiteExperimentConfig:
    """加载 SID-lite 顶层配置。"""
    payload = load_toml(path)
    return SidLiteExperimentConfig(
        name=str(payload["name"]),
        description=str(payload["description"]),
        benchmark_configs=[Path(item) for item in payload["benchmark_configs"]],
        protocol=Path(payload["protocol"]),
        methods=[str(item) for item in payload["methods"]],
        global_seed=int(payload["global_seed"]),
        prompt_version=str(payload["prompt_version"]),
        max_concurrent_requests=int(payload["max_concurrent_requests"]),
        requests_per_minute_limit=optional_int(payload, "requests_per_minute_limit"),
        tokens_per_minute_limit=optional_int(payload, "tokens_per_minute_limit"),
        primary_model_ref=str(payload["primary_model_ref"]),
        raw=payload,
    )



