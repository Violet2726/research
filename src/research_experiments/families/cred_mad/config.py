"""CRED-MAD 配置加载。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research_experiments.core.controls.control_prompts import FREE_TEXT_V1_PROMPT_VERSION
from research_experiments.family_runtime.config_helpers import apply_runtime_defaults, load_benchmarks, load_toml
from research_experiments.family_runtime.json_object_protocol import JSON_OBJECT_ANSWER_PROTOCOL_V3
from research_experiments.family_runtime.method_catalog import MethodConfig, load_method_catalog
from research_experiments.family_runtime.output_protocols import validate_output_protocol

CRED_VOTE_5 = "cred_vote_5"
CRED_REFUTE_QUEUE_V1_LOCK = "cred_refute_queue_v1_lock"
CRED_METHODS = frozenset({CRED_VOTE_5, CRED_REFUTE_QUEUE_V1_LOCK})


@dataclass(frozen=True)
class CredMadProtocolConfig:
    stage_a_agent_count: int
    max_refutations: int
    stage_a_temperature: float
    debate_temperature: float
    judge_temperature: float
    top_p: float
    strong_majority_count: int
    min_evidence_quality: float
    risk_trigger_count: int
    weak_majority_count: int
    locked_override_margin: float
    concrete_evidence_min_chars: int


@dataclass(frozen=True)
class CredMadExperimentConfig:
    name: str
    description: str
    benchmark_configs: list[Path]
    protocol: Path
    control_catalog: Path
    control_methods: list[str]
    cred_methods: list[str]
    method_order: list[str]
    global_seed: int
    control_prompt_version: str
    cred_output_protocol: str
    max_concurrent_requests: int
    requests_per_minute_limit: int | None
    primary_model_ref: str
    raw: dict[str, Any]


def load_protocol_config(path: str | Path) -> CredMadProtocolConfig:
    payload = load_toml(path)
    return CredMadProtocolConfig(
        stage_a_agent_count=int(payload.get("stage_a_agent_count", 5)),
        max_refutations=int(payload.get("max_refutations", 2)),
        stage_a_temperature=float(payload.get("stage_a_temperature", 0.7)),
        debate_temperature=float(payload.get("debate_temperature", 0.4)),
        judge_temperature=float(payload.get("judge_temperature", 0.0)),
        top_p=float(payload.get("top_p", 1.0)),
        strong_majority_count=int(payload.get("strong_majority_count", 4)),
        min_evidence_quality=float(payload.get("min_evidence_quality", 0.45)),
        risk_trigger_count=int(payload.get("risk_trigger_count", 2)),
        weak_majority_count=int(payload.get("weak_majority_count", 3)),
        locked_override_margin=float(payload.get("locked_override_margin", 1.0)),
        concrete_evidence_min_chars=int(payload.get("concrete_evidence_min_chars", 12)),
    )


def load_control_catalog(path: str | Path) -> dict[str, MethodConfig]:
    return load_method_catalog(path)


def load_experiment_config(path: str | Path) -> CredMadExperimentConfig:
    payload = load_toml(path)
    runtime = apply_runtime_defaults(payload)
    control_methods = [str(item) for item in payload.get("control_methods", [])]
    cred_methods = [str(item) for item in payload.get("cred_methods", [])]
    unsupported = sorted(set(cred_methods) - CRED_METHODS)
    if unsupported:
        raise ValueError("Unsupported cred_mad methods: " + ", ".join(unsupported))
    method_order = [str(item) for item in payload.get("method_order", [])]
    if set(method_order) != set(control_methods) | set(cred_methods):
        raise ValueError("cred_mad method_order must exactly cover control_methods and cred_methods.")
    if len(method_order) != len(set(method_order)):
        raise ValueError("cred_mad method_order must not contain duplicates.")
    control_prompt_version = str(payload.get("control_prompt_version", FREE_TEXT_V1_PROMPT_VERSION))
    if control_prompt_version != FREE_TEXT_V1_PROMPT_VERSION:
        raise ValueError(f"cred_mad control_prompt_version must be {FREE_TEXT_V1_PROMPT_VERSION}.")
    cred_output_protocol = validate_output_protocol(str(payload.get("cred_output_protocol", JSON_OBJECT_ANSWER_PROTOCOL_V3)))
    if cred_output_protocol != JSON_OBJECT_ANSWER_PROTOCOL_V3:
        raise ValueError("cred_mad cred_output_protocol must be json_object_answer_v3.")
    return CredMadExperimentConfig(
        name=str(payload["name"]),
        description=str(payload["description"]),
        benchmark_configs=[Path(item) for item in payload["benchmark_configs"]],
        protocol=Path(payload["protocol"]),
        control_catalog=Path(payload["control_catalog"]),
        control_methods=control_methods,
        cred_methods=cred_methods,
        method_order=method_order,
        global_seed=int(payload["global_seed"]),
        control_prompt_version=control_prompt_version,
        cred_output_protocol=cred_output_protocol,
        max_concurrent_requests=runtime["max_concurrent_requests"],
        requests_per_minute_limit=runtime["requests_per_minute_limit"],
        primary_model_ref=str(payload["primary_model_ref"]),
        raw=payload,
    )


def inspect_methods(experiment: CredMadExperimentConfig) -> list[str]:
    return [*experiment.control_methods, *experiment.cred_methods]


def inspect_benchmarks(experiment: CredMadExperimentConfig) -> list[str]:
    return [benchmark.slug for benchmark in load_benchmarks(experiment)]
