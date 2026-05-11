"""选择性通信实验的配置加载与模型适配检查。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research_experiments.core.foundation.config import BenchmarkConfig, ResolvedModelConfig
from research_experiments.core.foundation.config_helpers import (
    load_benchmarks,
    load_toml,
    optional_float,
    optional_int,
    phase_metadata,
    resolve_model,
)
from research_experiments.core.foundation.methods import MethodConfig, load_method_catalog

GENERAL_QA_BENCHMARKS = {"strategyqa", "hotpotqa"}


@dataclass(frozen=True)
class SharedDebateProtocolConfig:
    """选择性通信共享的辩论协议参数。"""

    agent_count: int
    debate_rounds: int
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
    claim_divergence_threshold: float | None
    uncertainty_type_diversity_threshold: float | None
    fail_open_to_always: bool


@dataclass(frozen=True)
class SelectiveCommExperimentConfig:
    """选择性通信实验入口配置。"""

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
    primary_model_ref: str
    raw: dict[str, Any]


def load_protocol_config(path: str | Path) -> SharedDebateProtocolConfig:
    """加载共享辩论协议。"""
    payload = load_toml(path)
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
    payload = load_toml(path)
    return TriggerPolicyConfig(
        policy_name=str(payload["policy_name"]),
        trigger_type=str(payload["trigger_type"]),
        mean_conf_threshold=optional_float(payload, "mean_conf_threshold"),
        conf_spread_threshold=optional_float(payload, "conf_spread_threshold"),
        claim_divergence_threshold=optional_float(payload, "claim_divergence_threshold"),
        uncertainty_type_diversity_threshold=optional_float(payload, "uncertainty_type_diversity_threshold"),
        fail_open_to_always=bool(payload.get("fail_open_to_always", True)),
    )


def load_policies(paths: list[str | Path]) -> list[TriggerPolicyConfig]:
    """按顺序加载多份 trigger 策略。"""
    return [load_policy_config(path) for path in paths]


def load_control_catalog(path: str | Path) -> dict[str, MethodConfig]:
    """加载控制组方法目录。"""
    return load_method_catalog(path)


def load_experiment_config(path: str | Path) -> SelectiveCommExperimentConfig:
    """加载选择性通信实验入口配置。"""
    payload = load_toml(path)
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
        requests_per_minute_limit=optional_int(payload, "requests_per_minute_limit"),
        tokens_per_minute_limit=optional_int(payload, "tokens_per_minute_limit"),
        primary_model_ref=str(payload["primary_model_ref"]),
        raw=payload,
    )


def describe_backbone_fit(
    experiment: SelectiveCommExperimentConfig,
    backbone: ResolvedModelConfig,
    benchmarks: list[BenchmarkConfig] | None = None,
) -> list[str]:
    """返回骨干模型与 benchmark 组合的兼容性警告。"""
    loaded_benchmarks = benchmarks if benchmarks is not None else load_benchmarks(experiment)
    benchmark_slugs = {benchmark.slug for benchmark in loaded_benchmarks}
    warnings: list[str] = []

    if benchmark_slugs & GENERAL_QA_BENCHMARKS:
        qa_datasets = ", ".join(sorted(benchmark_slugs & GENERAL_QA_BENCHMARKS))
        if "math" in backbone.tags and "general_qa" not in backbone.tags:
            warnings.append(
                f"Backbone {backbone.name} is tagged as math-specialized but this experiment includes "
                f"general-QA datasets: {qa_datasets}. Use a general_qa backbone such as "
                "deepseek/deepseek-v4-flash, or keep a math-specialized backbone on math-only benchmarks."
            )

    return warnings


def ensure_backbone_fit(
    experiment: SelectiveCommExperimentConfig,
    backbone: ResolvedModelConfig,
    benchmarks: list[BenchmarkConfig] | None = None,
) -> None:
    """在模型与 benchmark 明显不匹配时抛出异常。"""
    warnings = describe_backbone_fit(experiment, backbone, benchmarks)
    if warnings:
        raise RuntimeError("Incompatible backbone/benchmark mix:\n- " + "\n- ".join(warnings))


