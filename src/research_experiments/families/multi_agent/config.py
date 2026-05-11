"""多智能体实验配置加载。

本模块负责解析 Vanilla MAD 风格多智能体实验所需的配置，
包括 debate 协议、agent roster、实验 setup，以及与之配套的等预算控制方法。
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
from research_experiments.core.foundation.methods import MethodConfig, load_method_catalog


@dataclass(frozen=True)
class ProtocolConfig:
    """Vanilla MAD 协议配置。"""

    debate_rounds: int
    initial_temperature: float
    debate_temperature: float
    top_p: float
    max_output_tokens: int


@dataclass(frozen=True)
class RosterConfig:
    """多智能体 roster 配置。"""

    agent_count: int


@dataclass(frozen=True)
class ExperimentSetup:
    """单个多智能体 setup 的声明。"""

    name: str
    protocol: Path
    roster: Path
    matched_controls: list[str]


@dataclass(frozen=True)
class MultiAgentExperimentConfig:
    """多智能体实验的顶层配置。"""

    name: str
    description: str
    benchmark_configs: list[Path]
    control_catalog: Path
    setups: list[ExperimentSetup]
    global_seed: int
    prompt_version: str
    max_concurrent_requests: int
    requests_per_minute_limit: int | None
    tokens_per_minute_limit: int | None
    primary_model_ref: str
    raw: dict[str, Any]


def load_protocol_config(path: str | Path) -> ProtocolConfig:
    """加载多智能体协议信息。"""
    payload = load_toml(path)
    return ProtocolConfig(
        debate_rounds=int(payload["debate_rounds"]),
        initial_temperature=float(payload["initial_temperature"]),
        debate_temperature=float(payload["debate_temperature"]),
        top_p=float(payload["top_p"]),
        max_output_tokens=int(payload["max_output_tokens"]),
    )


def load_roster_config(path: str | Path) -> RosterConfig:
    """加载 agent roster 配置。"""
    payload = load_toml(path)
    return RosterConfig(agent_count=int(payload["agent_count"]))


def load_control_catalog(path: str | Path) -> dict[str, MethodConfig]:
    """加载与多智能体 setup 配套的控制方法目录。"""
    return load_method_catalog(path)


def load_experiment_config(path: str | Path) -> MultiAgentExperimentConfig:
    """加载多智能体实验配置。"""
    payload = load_toml(path)
    setups = [
        ExperimentSetup(
            name=str(item["name"]),
            protocol=Path(item["protocol"]),
            roster=Path(item["roster"]),
            matched_controls=[str(name) for name in item.get("matched_controls", [])],
        )
        for item in payload.get("setups", [])
    ]
    return MultiAgentExperimentConfig(
        name=str(payload["name"]),
        description=str(payload["description"]),
        benchmark_configs=[Path(item) for item in payload["benchmark_configs"]],
        control_catalog=Path(payload["control_catalog"]),
        setups=setups,
        global_seed=int(payload["global_seed"]),
        prompt_version=str(payload["prompt_version"]),
        max_concurrent_requests=int(payload["max_concurrent_requests"]),
        requests_per_minute_limit=optional_int(payload, "requests_per_minute_limit"),
        tokens_per_minute_limit=optional_int(payload, "tokens_per_minute_limit"),
        primary_model_ref=str(payload["primary_model_ref"]),
        raw=payload,
    )


