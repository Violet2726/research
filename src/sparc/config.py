"""SPARC 实验配置加载。

本模块解析 SPARC 及其消融实验所需的配置对象，
包括内容压缩设置、局部审计设置、触发策略、聚合方式与多种实验形态。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tomllib

from experiment_core.config import BenchmarkConfig, ResolvedModelConfig, load_benchmark_config, resolve_model_ref


EXPERIMENT_KIND_VALUES = {"content_ablation", "auditing_ablation", "sparc_v1"}


@dataclass(frozen=True)
class SparcProtocolConfig:
    """SPARC 共享求解 / 通信协议配置。"""

    agent_count: int
    debate_rounds: int
    initial_temperature: float
    debate_temperature: float
    top_p: float
    max_output_tokens: int


@dataclass(frozen=True)
class SparcExperimentConfig:
    """SPARC 顶层实验配置。"""

    name: str
    description: str
    experiment_kind: str
    benchmark_configs: list[Path]
    protocol: Path
    global_seed: int
    prompt_version: str
    max_concurrent_requests: int
    requests_per_minute_limit: int | None
    tokens_per_minute_limit: int | None
    primary_model_ref: str
    confirmatory_model_ref: str | None
    fixed_trigger_policy: str | None
    message_modes: list[str]
    fixed_message_modes: dict[str, str]
    aggregation_methods: list[str]
    default_trigger_policy: str | None
    fallback_trigger_policy: str | None
    trigger_drop_questions: float
    raw: dict[str, Any]


def _load_toml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def load_protocol_config(path: str | Path) -> SparcProtocolConfig:
    payload = _load_toml(path)
    return SparcProtocolConfig(
        agent_count=int(payload["agent_count"]),
        debate_rounds=int(payload["debate_rounds"]),
        initial_temperature=float(payload["initial_temperature"]),
        debate_temperature=float(payload["debate_temperature"]),
        top_p=float(payload["top_p"]),
        max_output_tokens=int(payload["max_output_tokens"]),
    )


def load_experiment_config(path: str | Path) -> SparcExperimentConfig:
    payload = _load_toml(path)
    experiment_kind = str(payload["experiment_kind"])
    if experiment_kind not in EXPERIMENT_KIND_VALUES:
        raise RuntimeError(
            f"Unsupported sparc experiment_kind {experiment_kind!r}. Expected one of {sorted(EXPERIMENT_KIND_VALUES)}."
        )
    content_config = payload.get("content_ablation", {})
    auditing_config = payload.get("auditing_ablation", {})
    sparc_config = payload.get("sparc_v1", {})
    fixed_message_modes = {
        str(key): str(value)
        for key, value in (
            auditing_config.get("fixed_message_modes")
            or sparc_config.get("fixed_message_modes")
            or {}
        ).items()
    }
    return SparcExperimentConfig(
        name=str(payload["name"]),
        description=str(payload["description"]),
        experiment_kind=experiment_kind,
        benchmark_configs=[Path(item) for item in payload["benchmark_configs"]],
        protocol=Path(payload["protocol"]),
        global_seed=int(payload["global_seed"]),
        prompt_version=str(payload["prompt_version"]),
        max_concurrent_requests=int(payload["max_concurrent_requests"]),
        requests_per_minute_limit=_optional_int(payload, "requests_per_minute_limit"),
        tokens_per_minute_limit=_optional_int(payload, "tokens_per_minute_limit"),
        primary_model_ref=str(payload["primary_model_ref"]),
        confirmatory_model_ref=_optional_str(payload, "confirmatory_model_ref"),
        fixed_trigger_policy=_first_str(content_config, "trigger_policy", auditing_config),
        message_modes=[str(item) for item in content_config.get("message_modes", [])],
        fixed_message_modes=fixed_message_modes,
        aggregation_methods=[str(item) for item in auditing_config.get("aggregation_methods", [])],
        default_trigger_policy=_optional_str(sparc_config, "default_trigger_policy"),
        fallback_trigger_policy=_optional_str(sparc_config, "fallback_trigger_policy"),
        trigger_drop_questions=float(sparc_config.get("trigger_drop_questions", 1.0)),
        raw=payload,
    )


def phase_metadata(experiment: SparcExperimentConfig, phase_name: str) -> dict[str, Any]:
    return dict(experiment.raw["phases"][phase_name])


def load_benchmarks(experiment: SparcExperimentConfig) -> list[BenchmarkConfig]:
    return [load_benchmark_config(path) for path in experiment.benchmark_configs]


def resolve_model(model_ref: str) -> ResolvedModelConfig:
    return resolve_model_ref(model_ref)


def _optional_int(payload: dict[str, Any], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    return int(value)


def _optional_str(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _first_str(primary: dict[str, Any], key: str, secondary: dict[str, Any]) -> str | None:
    value = _optional_str(primary, key)
    if value is not None:
        return value
    return _optional_str(secondary, key)
