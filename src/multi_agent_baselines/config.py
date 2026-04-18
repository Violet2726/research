"""多智能体实验配置解析。

这条子系统只负责解析 protocol / roster / experiment 这三层多智能体专属配置；
backbone 解析则直接复用现有单模型实验里的 `ResolvedModelConfig`。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib

from api_baselines.config import BenchmarkConfig, MethodConfig, ResolvedModelConfig, load_benchmark_config, resolve_model_ref


@dataclass(frozen=True)
class ProtocolConfig:
    """Vanilla MAD 协议定义。"""

    name: str
    debate_type: str
    debate_rounds: int
    share_mode: str
    revision_rule: str
    final_aggregator: str
    initial_temperature: float
    debate_temperature: float
    top_p: float
    max_output_tokens: int


@dataclass(frozen=True)
class RosterConfig:
    """Agent 编组定义。"""

    name: str
    roster_type: str
    agent_count: int
    sampling_seed_rule: str


@dataclass(frozen=True)
class ExperimentSetup:
    """一个 setup 把 protocol、roster 和等预算对照方法绑定在一起。"""

    name: str
    protocol: Path
    roster: Path
    matched_controls: list[str]


@dataclass(frozen=True)
class MultiAgentExperimentConfig:
    """多智能体实验规格。"""

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
    raw: dict[str, Any]


def _load_toml(path: str | Path) -> dict[str, Any]:
    """读取 TOML 配置。"""
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def load_protocol_config(path: str | Path) -> ProtocolConfig:
    """加载协议配置。"""
    payload = _load_toml(path)
    return ProtocolConfig(**payload)


def load_roster_config(path: str | Path) -> RosterConfig:
    """加载 agent 编组配置。"""
    payload = _load_toml(path)
    return RosterConfig(**payload)


def load_control_catalog(path: str | Path) -> dict[str, MethodConfig]:
    """加载等预算单模型对照方法目录。"""
    payload = _load_toml(path)
    methods = payload.get("methods", {})
    return {str(name): MethodConfig(name=str(name), **item) for name, item in methods.items()}


def load_experiment_config(path: str | Path) -> MultiAgentExperimentConfig:
    """加载多智能体实验规格。"""
    payload = _load_toml(path)
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
        name=payload["name"],
        description=payload["description"],
        benchmark_configs=[Path(item) for item in payload["benchmark_configs"]],
        control_catalog=Path(payload["control_catalog"]),
        setups=setups,
        global_seed=payload["global_seed"],
        prompt_version=payload["prompt_version"],
        max_concurrent_requests=payload["max_concurrent_requests"],
        requests_per_minute_limit=payload.get("requests_per_minute_limit"),
        tokens_per_minute_limit=payload.get("tokens_per_minute_limit"),
        raw=payload,
    )


def phase_metadata(experiment: MultiAgentExperimentConfig, phase_name: str) -> dict[str, Any]:
    """返回某个 phase 的原始配置。"""
    return dict(experiment.raw["phases"][phase_name])


def load_benchmarks(experiment: MultiAgentExperimentConfig) -> list[BenchmarkConfig]:
    """加载实验涉及的 benchmark 配置。"""
    return [load_benchmark_config(path) for path in experiment.benchmark_configs]


def resolve_backbone(model_ref: str) -> ResolvedModelConfig:
    """解析多智能体实验使用的 backbone。

    这里直接复用单模型实验已有的模型解析逻辑，不再维护一套重复的数据类。
    """
    return resolve_model_ref(model_ref)


def as_provider_model(backbone: ResolvedModelConfig) -> ResolvedModelConfig:
    """保留兼容入口，便于 runner 继续显式表达“backbone -> provider model”。"""
    return backbone
