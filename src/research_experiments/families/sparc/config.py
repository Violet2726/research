"""SPARC 实验配置加载。

本模块解析 SPARC 及其消融实验所需的配置对象，
包括内容压缩设置、局部审计设置、触发策略、聚合方式与多种实验形态。
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
    optional_str,
    phase_metadata,
    resolve_model,
)
from research_experiments.families.reference_runs import TriggerReferenceConfig


VARIANT_NAME_VALUES = {"content_ablation", "auditing_ablation", "sparc_v1"}


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
    variant_name: str
    benchmark_configs: list[Path]
    protocol: Path
    global_seed: int
    prompt_version: str
    max_concurrent_requests: int
    requests_per_minute_limit: int | None
    tokens_per_minute_limit: int | None
    primary_model_ref: str
    fixed_trigger_policy: str | None
    message_modes: list[str]
    fixed_message_modes: dict[str, str]
    aggregation_methods: list[str]
    trigger_reference: TriggerReferenceConfig | None
    raw: dict[str, Any]


def load_protocol_config(path: str | Path) -> SparcProtocolConfig:
    payload = load_toml(path)
    return SparcProtocolConfig(
        agent_count=int(payload["agent_count"]),
        debate_rounds=int(payload["debate_rounds"]),
        initial_temperature=float(payload["initial_temperature"]),
        debate_temperature=float(payload["debate_temperature"]),
        top_p=float(payload["top_p"]),
        max_output_tokens=int(payload["max_output_tokens"]),
    )


def load_experiment_config(path: str | Path) -> SparcExperimentConfig:
    payload = load_toml(path)
    variant_name = str(payload["variant_name"])
    if variant_name not in VARIANT_NAME_VALUES:
        raise RuntimeError(
            f"Unsupported sparc variant_name {variant_name!r}. Expected one of {sorted(VARIANT_NAME_VALUES)}."
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
    trigger_reference_payload = sparc_config.get("trigger_reference")
    if trigger_reference_payload is None and any(
        key in sparc_config for key in ("default_trigger_policy", "fallback_trigger_policy", "trigger_drop_questions")
    ):
        trigger_reference_payload = {
            "source_family": "selective_comm",
            "source_experiment": "trigger_early_exit_main",
            "source_phase": "count20",
            "default_policy": optional_str(sparc_config, "default_trigger_policy") or "hybrid_trigger",
            "fallback_policy": optional_str(sparc_config, "fallback_trigger_policy") or "disagreement_triggered",
            "drop_questions_threshold": float(sparc_config.get("trigger_drop_questions", 1.0)),
        }
    trigger_reference = (
        TriggerReferenceConfig(
            source_family=str(trigger_reference_payload["source_family"]),
            source_experiment=str(trigger_reference_payload["source_experiment"]),
            source_phase=str(trigger_reference_payload["source_phase"]),
            default_policy=str(trigger_reference_payload["default_policy"]),
            fallback_policy=str(trigger_reference_payload["fallback_policy"]),
            drop_questions_threshold=float(trigger_reference_payload["drop_questions_threshold"]),
        )
        if trigger_reference_payload is not None
        else None
    )
    return SparcExperimentConfig(
        name=str(payload["name"]),
        description=str(payload["description"]),
        variant_name=variant_name,
        benchmark_configs=[Path(item) for item in payload["benchmark_configs"]],
        protocol=Path(payload["protocol"]),
        global_seed=int(payload["global_seed"]),
        prompt_version=str(payload["prompt_version"]),
        max_concurrent_requests=int(payload["max_concurrent_requests"]),
        requests_per_minute_limit=optional_int(payload, "requests_per_minute_limit"),
        tokens_per_minute_limit=optional_int(payload, "tokens_per_minute_limit"),
        primary_model_ref=str(payload["primary_model_ref"]),
        fixed_trigger_policy=_first_str(content_config, auditing_config, "trigger_policy"),
        message_modes=[str(item) for item in content_config.get("message_modes", [])],
        fixed_message_modes=fixed_message_modes,
        aggregation_methods=[str(item) for item in auditing_config.get("aggregation_methods", [])],
        trigger_reference=trigger_reference,
        raw=payload,
    )


def _first_str(primary: dict[str, Any], secondary: dict[str, Any], key: str) -> str | None:
    value = optional_str(primary, key)
    if value is not None:
        return value
    return optional_str(secondary, key)

