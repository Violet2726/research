"""A-SMAD 配置加载。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research_experiments.family_runtime.config_helpers import (
    apply_runtime_defaults,
    load_benchmarks,
    load_toml,
)
from research_experiments.family_runtime.method_catalog import MethodConfig, load_method_catalog

ACTIVE_AGGREGATE_METHODS = frozenset(
    {
        "hetero_vote_3",
        "ega_only_v4",
        "adaptive_gate_v4",
        "adaptive_dual_open_v5",
    }
)
ADAPTIVE_POLICY_METHODS = frozenset(
    {
        "adaptive_gate_v4",
        "adaptive_dual_open_v5",
    }
)


@dataclass(frozen=True)
class AdaptiveSparseMadProtocolConfig:
    agent_count: int
    top_p: float
    stage_a_temperature: float
    stage_a_max_output_tokens: int
    consensus_confidence_threshold: float
    majority_confidence_threshold: float
    majority_margin_threshold: float


@dataclass(frozen=True)
class AdaptiveSparseMadExperimentConfig:
    name: str
    description: str
    benchmark_configs: list[Path]
    protocol: Path
    control_catalog: Path
    aggregate_methods: tuple[str, ...]
    max_adaptive_addon_calls: int
    global_seed: int
    prompt_version: str
    stage_a_prompt_version: str
    adaptive_prompt_version: str
    max_concurrent_requests: int
    requests_per_minute_limit: int | None
    tokens_per_minute_limit: int | None
    primary_model_ref: str
    raw: dict[str, Any]


def load_protocol_config(path: str | Path) -> AdaptiveSparseMadProtocolConfig:
    payload = load_toml(path)
    return AdaptiveSparseMadProtocolConfig(
        agent_count=int(payload["agent_count"]),
        top_p=float(payload["top_p"]),
        stage_a_temperature=float(payload["stage_a_temperature"]),
        stage_a_max_output_tokens=int(payload["stage_a_max_output_tokens"]),
        consensus_confidence_threshold=float(payload["consensus_confidence_threshold"]),
        majority_confidence_threshold=float(payload["majority_confidence_threshold"]),
        majority_margin_threshold=float(payload["majority_margin_threshold"]),
    )


def load_control_catalog(path: str | Path) -> dict[str, MethodConfig]:
    return load_method_catalog(path)


def load_experiment_config(path: str | Path) -> AdaptiveSparseMadExperimentConfig:
    payload = load_toml(path)
    runtime = apply_runtime_defaults(payload)
    aggregate_methods = tuple(str(item) for item in payload.get("aggregate_methods", ["hetero_vote_3"]))
    unsupported_methods = sorted(set(aggregate_methods) - ACTIVE_AGGREGATE_METHODS)
    if unsupported_methods:
        raise ValueError(
            "Unsupported adaptive_sparse_mad aggregate_methods: "
            + ", ".join(unsupported_methods)
        )
    max_adaptive_addon_calls = int(
        payload.get(
            "max_adaptive_addon_calls",
            1 if any(method_name in ADAPTIVE_POLICY_METHODS for method_name in aggregate_methods) else 0,
        )
    )
    prompt_version = str(payload["prompt_version"])
    stage_a_prompt_version = str(payload.get("stage_a_prompt_version", prompt_version))
    adaptive_prompt_version = str(payload.get("adaptive_prompt_version", prompt_version))
    return AdaptiveSparseMadExperimentConfig(
        name=str(payload["name"]),
        description=str(payload["description"]),
        benchmark_configs=[Path(item) for item in payload["benchmark_configs"]],
        protocol=Path(payload["protocol"]),
        control_catalog=Path(payload["control_catalog"]),
        aggregate_methods=aggregate_methods,
        max_adaptive_addon_calls=max_adaptive_addon_calls,
        global_seed=int(payload["global_seed"]),
        prompt_version=prompt_version,
        stage_a_prompt_version=stage_a_prompt_version,
        adaptive_prompt_version=adaptive_prompt_version,
        max_concurrent_requests=runtime["max_concurrent_requests"],
        requests_per_minute_limit=runtime["requests_per_minute_limit"],
        tokens_per_minute_limit=runtime["tokens_per_minute_limit"],
        primary_model_ref=str(payload["primary_model_ref"]),
        raw=payload,
    )


def inspect_methods(experiment: AdaptiveSparseMadExperimentConfig) -> list[str]:
    controls = load_control_catalog(experiment.control_catalog)
    return list(controls) + list(experiment.aggregate_methods)


def inspect_benchmarks(experiment: AdaptiveSparseMadExperimentConfig) -> list[str]:
    return [benchmark.slug for benchmark in load_benchmarks(experiment)]
