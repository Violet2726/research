from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib

from experiment_core.config import BenchmarkConfig, ResolvedModelConfig, load_benchmark_config, resolve_model_ref
from experiment_core.methods import MethodConfig, load_method_catalog

GENERAL_QA_BENCHMARKS = {"strategyqa", "hotpotqa"}


@dataclass(frozen=True)
class SharedDebateProtocolConfig:
    agent_count: int
    debate_rounds: int
    initial_temperature: float
    debate_temperature: float
    top_p: float
    max_output_tokens: int


@dataclass(frozen=True)
class TriggerPolicyConfig:
    policy_name: str
    trigger_type: str
    mean_conf_threshold: float | None
    conf_spread_threshold: float | None
    claim_divergence_threshold: float | None
    uncertainty_type_diversity_threshold: float | None
    fail_open_to_always: bool


@dataclass(frozen=True)
class SelectiveCommExperimentConfig:
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
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def load_protocol_config(path: str | Path) -> SharedDebateProtocolConfig:
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
    payload = _load_toml(path)
    return TriggerPolicyConfig(
        policy_name=str(payload["policy_name"]),
        trigger_type=str(payload["trigger_type"]),
        mean_conf_threshold=_optional_float(payload, "mean_conf_threshold"),
        conf_spread_threshold=_optional_float(payload, "conf_spread_threshold"),
        claim_divergence_threshold=_optional_float(payload, "claim_divergence_threshold"),
        uncertainty_type_diversity_threshold=_optional_float(payload, "uncertainty_type_diversity_threshold"),
        fail_open_to_always=bool(payload.get("fail_open_to_always", True)),
    )


def load_policies(paths: list[str | Path]) -> list[TriggerPolicyConfig]:
    return [load_policy_config(path) for path in paths]


def load_control_catalog(path: str | Path) -> dict[str, MethodConfig]:
    return load_method_catalog(path)


def load_experiment_config(path: str | Path) -> SelectiveCommExperimentConfig:
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
    return dict(experiment.raw["phases"][phase_name])


def load_benchmarks(experiment: SelectiveCommExperimentConfig) -> list[BenchmarkConfig]:
    return [load_benchmark_config(path) for path in experiment.benchmark_configs]


def describe_backbone_fit(
    experiment: SelectiveCommExperimentConfig,
    backbone: ResolvedModelConfig,
    benchmarks: list[BenchmarkConfig] | None = None,
) -> list[str]:
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
    warnings = describe_backbone_fit(experiment, backbone, benchmarks)
    if warnings:
        raise RuntimeError("Incompatible backbone/benchmark mix:\n- " + "\n- ".join(warnings))


def resolve_backbone(model_ref: str) -> ResolvedModelConfig:
    return resolve_model_ref(model_ref)


def _optional_int(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    return int(value)


def _optional_float(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    return float(value)
