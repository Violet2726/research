"""CRED-V 验证器中心实验配置加载。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research_experiments.core.controls.control_prompts import FREE_TEXT_V1_PROMPT_VERSION
from research_experiments.family_runtime.config_helpers import apply_runtime_defaults, load_benchmarks, load_toml
from research_experiments.family_runtime.free_text_protocol import FREE_TEXT_ANSWER_PROTOCOL_V1
from research_experiments.family_runtime.json_object_protocol import JSON_OBJECT_ANSWER_PROTOCOL_V3
from research_experiments.family_runtime.method_catalog import MethodConfig, load_method_catalog
from research_experiments.family_runtime.output_protocols import validate_output_protocol

CRED_V_VOTE_5 = "cred_v_vote_5"
CRED_V_TASK_VERIFY_V3 = "cred_v_task_verify_v3"
CRED_VERIFY_SAFE_V1 = "cred_verify_safe_v1"
CRED_ACS_V1 = "cred_acs_v1"
CRED_METHODS = frozenset({CRED_V_VOTE_5, CRED_V_TASK_VERIFY_V3, CRED_VERIFY_SAFE_V1, CRED_ACS_V1})
CRED_VERIFY_METHODS = frozenset({CRED_V_TASK_VERIFY_V3})
CRED_SAFE_VERIFY_METHODS = frozenset({CRED_VERIFY_SAFE_V1})
CRED_ACS_METHODS = frozenset({CRED_ACS_V1})
CRED_COMM_METHODS = CRED_VERIFY_METHODS | CRED_SAFE_VERIFY_METHODS | CRED_ACS_METHODS
CRED_VOTE_METHODS = frozenset({CRED_V_VOTE_5})


@dataclass(frozen=True)
class CredVProtocolConfig:
    stage_a_agent_count: int
    max_verifications: int
    max_verification_calls: int
    verification_modes: tuple[str, ...]
    expansion_modes: tuple[str, ...]
    expansion_model_refs: tuple[str, ...]
    max_expansion_calls: int
    promotion_min_independent_support: int
    promotion_margin_min: float
    allow_single_verifier_promotion: bool
    false_consensus_probe: bool
    max_trigger_rate: float
    allow_same_model_promotion: bool
    stage_a_temperature: float
    verifier_temperature: float
    top_p: float
    strong_majority_count: int
    weak_majority_count: int
    concrete_evidence_min_chars: int
    stage_a_max_tokens: int
    verifier_max_tokens: int
    promotion_confidence_min: float
    promotion_score_margin: float


@dataclass(frozen=True)
class CredVExperimentConfig:
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
    cred_stage_a_output_protocol: str
    cred_verification_output_protocol: str
    max_concurrent_requests: int
    requests_per_minute_limit: int | None
    primary_model_ref: str
    verifier_model_refs: list[str]
    raw: dict[str, Any]


def load_protocol_config(path: str | Path) -> CredVProtocolConfig:
    payload = load_toml(path)
    max_verifications = int(payload.get("max_verifications", payload.get("max_verification_calls", 1)))
    return CredVProtocolConfig(
        stage_a_agent_count=int(payload.get("stage_a_agent_count", 5)),
        max_verifications=max_verifications,
        max_verification_calls=int(payload.get("max_verification_calls", max_verifications)),
        verification_modes=tuple(
            str(item)
            for item in payload.get(
                "verification_modes",
                ["deterministic_repair", "tool_verified", "hetero_verified"],
            )
        ),
        expansion_modes=tuple(
            str(item)
            for item in payload.get(
                "expansion_modes",
                ["math_symbolic_repair", "hotpot_span_extract", "mc_choice_shuffle", "strategyqa_dual_polarity"],
            )
        ),
        expansion_model_refs=tuple(str(item) for item in payload.get("expansion_model_refs", [])),
        max_expansion_calls=int(payload.get("max_expansion_calls", 0)),
        promotion_min_independent_support=int(payload.get("promotion_min_independent_support", 2)),
        promotion_margin_min=float(payload.get("promotion_margin_min", payload.get("promotion_score_margin", 1.0))),
        allow_single_verifier_promotion=_parse_bool(payload.get("allow_single_verifier_promotion", False)),
        false_consensus_probe=_parse_bool(payload.get("false_consensus_probe", False)),
        max_trigger_rate=float(payload.get("max_trigger_rate", 1.0)),
        allow_same_model_promotion=_parse_bool(payload.get("allow_same_model_promotion", False)),
        stage_a_temperature=float(payload.get("stage_a_temperature", 0.7)),
        verifier_temperature=float(payload.get("verifier_temperature", 0.0)),
        top_p=float(payload.get("top_p", 1.0)),
        strong_majority_count=int(payload.get("strong_majority_count", 4)),
        weak_majority_count=int(payload.get("weak_majority_count", 3)),
        concrete_evidence_min_chars=int(payload.get("concrete_evidence_min_chars", 12)),
        stage_a_max_tokens=int(payload.get("stage_a_max_tokens", 640)),
        verifier_max_tokens=int(payload.get("verifier_max_tokens", 1024)),
        promotion_confidence_min=float(payload.get("promotion_confidence_min", 0.72)),
        promotion_score_margin=float(payload.get("promotion_score_margin", 0.15)),
    )


def load_control_catalog(path: str | Path) -> dict[str, MethodConfig]:
    return load_method_catalog(path)


def load_experiment_config(path: str | Path) -> CredVExperimentConfig:
    payload = load_toml(path)
    runtime = apply_runtime_defaults(payload)
    control_methods = [str(item) for item in payload.get("control_methods", [])]
    cred_methods = [str(item) for item in payload.get("cred_methods", [])]
    unsupported = sorted(set(cred_methods) - CRED_METHODS)
    if unsupported:
        raise ValueError("Unsupported cred_v methods: " + ", ".join(unsupported))
    method_order = [str(item) for item in payload.get("method_order", [])]
    if set(method_order) != set(control_methods) | set(cred_methods):
        raise ValueError("cred_v method_order must exactly cover control_methods and cred_methods.")
    if len(method_order) != len(set(method_order)):
        raise ValueError("cred_v method_order must not contain duplicates.")
    control_prompt_version = str(payload.get("control_prompt_version", FREE_TEXT_V1_PROMPT_VERSION))
    if control_prompt_version != FREE_TEXT_V1_PROMPT_VERSION:
        raise ValueError(f"cred_v control_prompt_version must be {FREE_TEXT_V1_PROMPT_VERSION}.")
    cred_output_protocol = validate_output_protocol(str(payload.get("cred_output_protocol", JSON_OBJECT_ANSWER_PROTOCOL_V3)))
    cred_stage_a_output_protocol = validate_output_protocol(
        str(payload.get("cred_stage_a_output_protocol", payload.get("stage_a_output_protocol", FREE_TEXT_ANSWER_PROTOCOL_V1)))
    )
    cred_verification_output_protocol = validate_output_protocol(
        str(payload.get("cred_verification_output_protocol", JSON_OBJECT_ANSWER_PROTOCOL_V3))
    )
    if cred_stage_a_output_protocol not in {FREE_TEXT_ANSWER_PROTOCOL_V1, JSON_OBJECT_ANSWER_PROTOCOL_V3}:
        raise ValueError("cred_v cred_stage_a_output_protocol must be free_text_answer_v1 or json_object_answer_v3.")
    if cred_verification_output_protocol != JSON_OBJECT_ANSWER_PROTOCOL_V3:
        raise ValueError("cred_v cred_verification_output_protocol must be json_object_answer_v3.")
    return CredVExperimentConfig(
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
        cred_stage_a_output_protocol=cred_stage_a_output_protocol,
        cred_verification_output_protocol=cred_verification_output_protocol,
        max_concurrent_requests=runtime["max_concurrent_requests"],
        requests_per_minute_limit=runtime["requests_per_minute_limit"],
        primary_model_ref=str(payload["primary_model_ref"]),
        verifier_model_refs=[str(item) for item in payload.get("verifier_model_refs", [])],
        raw=payload,
    )


def inspect_methods(experiment: CredVExperimentConfig) -> list[str]:
    return [*experiment.control_methods, *experiment.cred_methods]


def inspect_benchmarks(experiment: CredVExperimentConfig) -> list[str]:
    return [benchmark.slug for benchmark in load_benchmarks(experiment)]


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}
