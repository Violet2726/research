"""Free-MAD-lite 实验配置加载。

本模块负责解析 Free-MAD-lite 所需的协议与顶层实验配置，
支撑“单轮反从众辩论 + 轨迹裁决”的轻量机制验证。
"""

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


@dataclass(frozen=True)
class FreeMadLiteProtocolConfig:
    """Free-MAD-lite 共享协议配置。"""

    agent_count: int
    debate_rounds: int
    initial_temperature: float
    debate_temperature: float
    judge_temperature: float
    top_p: float
    max_output_tokens: int
    judge_max_output_tokens: int


@dataclass(frozen=True)
class FreeMadLiteExperimentConfig:
    """Free-MAD-lite 顶层实验配置。"""

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


def load_protocol_config(path: str | Path) -> FreeMadLiteProtocolConfig:
    """加载 Free-MAD-lite 协议配置。"""
    payload = load_toml(path)
    return FreeMadLiteProtocolConfig(
        agent_count=int(payload["agent_count"]),
        debate_rounds=int(payload["debate_rounds"]),
        initial_temperature=float(payload["initial_temperature"]),
        debate_temperature=float(payload["debate_temperature"]),
        judge_temperature=float(payload["judge_temperature"]),
        top_p=float(payload["top_p"]),
        max_output_tokens=int(payload["max_output_tokens"]),
        judge_max_output_tokens=int(payload["judge_max_output_tokens"]),
    )


def load_experiment_config(path: str | Path) -> FreeMadLiteExperimentConfig:
    """加载 Free-MAD-lite 顶层配置。"""
    payload = load_toml(path)
    return FreeMadLiteExperimentConfig(
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


