"""iMAD family 的配置加载逻辑。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research_experiments.core.config import BenchmarkConfig, ResolvedModelConfig
from research_experiments.families.shared.config_loading import (
    load_benchmarks,
    load_toml,
    optional_float,
    optional_int,
    phase_metadata,
    resolve_model,
)
from research_experiments.families.shared.method_catalog import MethodConfig, load_method_catalog


@dataclass(frozen=True)
class ProtocolConfig:
    """iMAD 多轮 debate 与稳定性检测协议。"""

    agent_count: int
    max_rounds: int
    initial_temperature: float
    debate_temperature: float
    top_p: float
    max_output_tokens: int
    posterior_sample_count: int
    stability_ks_threshold: float
    stable_posterior_mean_threshold: float


@dataclass(frozen=True)
class DebateMethodSpec:
    """iMAD 实验中的单个 debate 方法声明。"""

    name: str
    mode: str
    round_limit: int
    matched_controls: list[str]


@dataclass(frozen=True)
class ImadExperimentConfig:
    """iMAD 实验的顶层配置。"""

    name: str
    description: str
    benchmark_configs: list[Path]
    control_catalog: Path
    protocol: Path
    methods: list[DebateMethodSpec]
    global_seed: int
    prompt_version: str
    max_concurrent_requests: int
    requests_per_minute_limit: int | None
    tokens_per_minute_limit: int | None
    primary_model_ref: str
    raw: dict[str, Any]


def load_protocol_config(path: str | Path) -> ProtocolConfig:
    """加载 iMAD 协议配置。"""

    payload = load_toml(path)
    return ProtocolConfig(
        agent_count=int(payload["agent_count"]),
        max_rounds=int(payload["max_rounds"]),
        initial_temperature=float(payload["initial_temperature"]),
        debate_temperature=float(payload["debate_temperature"]),
        top_p=float(payload["top_p"]),
        max_output_tokens=int(payload["max_output_tokens"]),
        posterior_sample_count=int(payload["posterior_sample_count"]),
        stability_ks_threshold=float(payload["stability_ks_threshold"]),
        stable_posterior_mean_threshold=float(payload["stable_posterior_mean_threshold"]),
    )


def load_control_catalog(path: str | Path) -> dict[str, MethodConfig]:
    """加载同预算无通信控制方法目录。"""

    return load_method_catalog(path)


def load_experiment_config(path: str | Path) -> ImadExperimentConfig:
    """加载 iMAD 实验配置。"""

    payload = load_toml(path)
    methods = [
        DebateMethodSpec(
            name=str(item["name"]),
            mode=str(item["mode"]),
            round_limit=int(item["round_limit"]),
            matched_controls=[str(name) for name in item.get("matched_controls", [])],
        )
        for item in payload.get("methods", [])
    ]
    return ImadExperimentConfig(
        name=str(payload["name"]),
        description=str(payload["description"]),
        benchmark_configs=[Path(item) for item in payload["benchmark_configs"]],
        control_catalog=Path(payload["control_catalog"]),
        protocol=Path(payload["protocol"]),
        methods=methods,
        global_seed=int(payload["global_seed"]),
        prompt_version=str(payload["prompt_version"]),
        max_concurrent_requests=int(payload["max_concurrent_requests"]),
        requests_per_minute_limit=optional_int(payload, "requests_per_minute_limit"),
        tokens_per_minute_limit=optional_int(payload, "tokens_per_minute_limit"),
        primary_model_ref=str(payload["primary_model_ref"]),
        raw=payload,
    )

