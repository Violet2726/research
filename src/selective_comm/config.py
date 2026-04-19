"""选择性通信实验配置解析。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib

from api_baselines.config import BenchmarkConfig, ResolvedModelConfig, load_benchmark_config, resolve_model_ref


@dataclass(frozen=True)
class SharedDebateProtocolConfig:
    """共享前缀协议定义。"""

    name: str
    roster_type: str
    agent_count: int
    sampling_seed_rule: str
    debate_rounds: int
    share_mode: str
    final_aggregator: str
    initial_temperature: float
    debate_temperature: float
    top_p: float
    max_output_tokens: int


@dataclass(frozen=True)
class TriggerPolicyConfig:
    """触发策略配置。"""

    policy_name: str
    trigger_type: str
    mean_conf_threshold: float | None
    conf_spread_threshold: float | None
    fail_open_to_always: bool


@dataclass(frozen=True)
class ControlMethodConfig:
    """控制方法配置。"""

    name: str
    family: str
    budget_calls: int
    temperature: float
    top_p: float
    max_output_tokens: int


@dataclass(frozen=True)
class SelectiveCommExperimentConfig:
    """选择性通信实验规格。"""

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
    robustness_backbone: str
    backbone_fallback: str | None
    raw: dict[str, Any]


def _load_toml(path: str | Path) -> dict[str, Any]:
    """读取 TOML 配置。"""
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def load_protocol_config(path: str | Path) -> SharedDebateProtocolConfig:
    """加载共享 debate 协议。"""
    payload = _load_toml(path)
    return SharedDebateProtocolConfig(**payload)


def load_policy_config(path: str | Path) -> TriggerPolicyConfig:
    """加载单个触发策略。"""
    payload = _load_toml(path)
    return TriggerPolicyConfig(
        policy_name=str(payload["policy_name"]),
        trigger_type=str(payload["trigger_type"]),
        mean_conf_threshold=_optional_float(payload, "mean_conf_threshold"),
        conf_spread_threshold=_optional_float(payload, "conf_spread_threshold"),
        fail_open_to_always=bool(payload.get("fail_open_to_always", True)),
    )


def load_policies(paths: list[str | Path]) -> list[TriggerPolicyConfig]:
    """批量加载触发策略。"""
    return [load_policy_config(path) for path in paths]


def load_control_catalog(path: str | Path) -> dict[str, ControlMethodConfig]:
    """加载控制方法目录。"""
    payload = _load_toml(path)
    methods = payload.get("methods", {})
    return {
        str(name): ControlMethodConfig(name=str(name), **config)
        for name, config in methods.items()
    }


def load_experiment_config(path: str | Path) -> SelectiveCommExperimentConfig:
    """加载选择性通信实验规格。"""
    payload = _load_toml(path)
    return SelectiveCommExperimentConfig(
        name=payload["name"],
        description=payload["description"],
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
        robustness_backbone=str(payload["robustness_backbone"]),
        backbone_fallback=payload.get("backbone_fallback"),
        raw=payload,
    )


def phase_metadata(experiment: SelectiveCommExperimentConfig, phase_name: str) -> dict[str, Any]:
    """返回某个 phase 的原始配置。"""
    return dict(experiment.raw["phases"][phase_name])


def load_benchmarks(experiment: SelectiveCommExperimentConfig) -> list[BenchmarkConfig]:
    """加载实验中的 benchmark。"""
    return [load_benchmark_config(path) for path in experiment.benchmark_configs]


def resolve_backbone(model_ref: str) -> ResolvedModelConfig:
    """复用现有模型解析逻辑。"""
    return resolve_model_ref(model_ref)


def _optional_int(payload: dict[str, Any], key: str) -> int | None:
    """安全读取可选整数字段。"""
    value = payload.get(key)
    if value is None:
        return None
    return int(value)


def _optional_float(payload: dict[str, Any], key: str) -> float | None:
    """安全读取可选浮点字段。"""
    value = payload.get(key)
    if value is None:
        return None
    return float(value)
