"""`comm_necessary` 实验配置加载。

本模块解析 HotpotQA 通信必要性实验所需的协议、方法集合与顶层实验配置，
支持在 split-context 设定下比较不同消息强度对 answer / evidence 整合的影响。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research_experiments.core.foundation.config import BenchmarkConfig, ResolvedModelConfig
from research_experiments.core.foundation.config_helpers import (
    load_benchmarks,
    load_toml,
    optional_int,
    phase_metadata,
    resolve_model,
)


@dataclass(frozen=True)
class CommNecessaryProtocolConfig:
    """HotpotQA 通信必要性实验协议。"""

    agent_count: int
    debate_rounds: int
    initial_temperature: float
    update_temperature: float
    top_p: float
    max_output_tokens: int
    answer_only_token_cap: int
    evidence_token_cap: int
    full_packet_token_cap: int


@dataclass(frozen=True)
class CommNecessaryExperimentConfig:
    """comm_necessary 顶层实验配置。"""

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


def load_protocol_config(path: str | Path) -> CommNecessaryProtocolConfig:
    """加载协议配置。"""
    payload = load_toml(path)
    return CommNecessaryProtocolConfig(
        agent_count=int(payload["agent_count"]),
        debate_rounds=int(payload["debate_rounds"]),
        initial_temperature=float(payload["initial_temperature"]),
        update_temperature=float(payload["update_temperature"]),
        top_p=float(payload["top_p"]),
        max_output_tokens=int(payload["max_output_tokens"]),
        answer_only_token_cap=int(payload["answer_only_token_cap"]),
        evidence_token_cap=int(payload["evidence_token_cap"]),
        full_packet_token_cap=int(payload["full_packet_token_cap"]),
    )


def load_experiment_config(path: str | Path) -> CommNecessaryExperimentConfig:
    """加载顶层实验配置。"""
    payload = load_toml(path)
    return CommNecessaryExperimentConfig(
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


