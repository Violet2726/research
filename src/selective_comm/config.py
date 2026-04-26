"""选择性通信实验配置加载。

本模块解析 trigger / early-exit 实验所需的共享协议、策略集合、控制方法与顶层实验配置，
为后续 runner 提供统一的“共享前缀 + 多策略复用”配置入口。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib

from experiment_core.config import BenchmarkConfig, ResolvedModelConfig, load_benchmark_config, resolve_model_ref
from experiment_core.methods import MethodConfig, load_method_catalog


@dataclass(frozen=True)
class SharedDebateProtocolConfig:
    """共享前缀协议配置。"""

    agent_count: int
    debate_rounds: int
    initial_temperature: float
    debate_temperature: float
    top_p: float
    max_output_tokens: int


@dataclass(frozen=True)
class TriggerPolicyConfig:
    """trigger 策略配置。"""

    policy_name: str
    trigger_type: str
    mean_conf_threshold: float | None
    conf_spread_threshold: float | None
    fail_open_to_always: bool


@dataclass(frozen=True)
class SelectiveCommExperimentConfig:
    """选择性通信实验顶层配置。"""

    name: str
    description: str
    benchmark_configs: list[Path]
    protocol: Path
    policy_configs: list[Path]
    control_catalog: Path
    global_seed: int
    prompt_version: str
    max_concurrent_requests: int
    requests_per_minute_limit: int | None
    tokens_per_minute_limit: int | None
    primary_backbone: str
    raw: dict[str, Any]


def _load_toml(path: str | Path) -> dict[str, Any]:
    """读取 TOML 文件。"""
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def load_protocol_config(path: str | Path) -> SharedDebateProtocolConfig:
    """加载共享前缀协议配置。"""
    payload = _load_toml(path)
    return SharedDebateProtocolConfig(
        agent_count=int(payload["agent_count"]),
        debate_rounds=int(payload["debate_rounds"]),
        initial_temperature=float(payload["initial_temperature"]),
        debate_temperature=float(payload["debate_temperature"]),
        top_p=float(payload["top_p"]),
        max_output_tokens=int(payload["max_output_tokens"]),
    )


def load_policy_config(path: str | Path) -> TriggerPolicyConfig:
    """加载单个 trigger 策略配置。"""
    payload = _load_toml(path)
    return TriggerPolicyConfig(
        policy_name=str(payload["policy_name"]),
        trigger_type=str(payload["trigger_type"]),
        mean_conf_threshold=_optional_float(payload, "mean_conf_threshold"),
        conf_spread_threshold=_optional_float(payload, "conf_spread_threshold"),
        fail_open_to_always=bool(payload.get("fail_open_to_always", True)),
    )


def load_policies(paths: list[str | Path]) -> list[TriggerPolicyConfig]:
    """按顺序加载多条 trigger 策略。"""
    return [load_policy_config(path) for path in paths]


def load_control_catalog(path: str | Path) -> dict[str, MethodConfig]:
    """加载独立控制方法目录。"""
    return load_method_catalog(path)


def load_experiment_config(path: str | Path) -> SelectiveCommExperimentConfig:
    """加载选择性通信实验配置。"""
    payload = _load_toml(path)
    return SelectiveCommExperimentConfig(
        name=str(payload["name"]),
        description=str(payload["description"]),
        benchmark_configs=[Path(item) for item in payload["benchmark_configs"]],
        protocol=Path(payload["protocol"]),
        policy_configs=[Path(item) for item in payload["policy_configs"]],
        control_catalog=Path(payload["control_catalog"]),
        global_seed=int(payload["global_seed"]),
        prompt_version=str(payload["prompt_version"]),
        max_concurrent_requests=int(payload["max_concurrent_requests"]),
        requests_per_minute_limit=_optional_int(payload, "requests_per_minute_limit"),
        tokens_per_minute_limit=_optional_int(payload, "tokens_per_minute_limit"),
        primary_backbone=str(payload["primary_backbone"]),
        raw=payload,
    )


def phase_metadata(experiment: SelectiveCommExperimentConfig, phase_name: str) -> dict[str, Any]:
    """返回指定 phase 的原始配置副本。"""
    return dict(experiment.raw["phases"][phase_name])


def load_benchmarks(experiment: SelectiveCommExperimentConfig) -> list[BenchmarkConfig]:
    """加载实验声明使用的 benchmark 配置。"""
    return [load_benchmark_config(path) for path in experiment.benchmark_configs]


def resolve_backbone(model_ref: str) -> ResolvedModelConfig:
    """解析选择性通信实验使用的 backbone 模型。"""
    return resolve_model_ref(model_ref)


def _optional_int(payload: dict[str, Any], key: str) -> int | None:
    """读取可选整数值。"""
    value = payload.get(key)
    if value is None:
        return None
    return int(value)


def _optional_float(payload: dict[str, Any], key: str) -> float | None:
    """读取可选浮点值。"""
    value = payload.get(key)
    if value is None:
        return None
    return float(value)
