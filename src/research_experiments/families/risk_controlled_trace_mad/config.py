"""RCTA-MAD 配置、公开方法集合与冻结不变量。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research_experiments.core.controls.control_prompts import FREE_TEXT_V1_PROMPT_VERSION
from research_experiments.family_runtime.config_helpers import apply_runtime_defaults, load_benchmarks, load_toml
from research_experiments.family_runtime.method_catalog import MethodConfig, load_method_catalog

CONTROL_METHODS = ("cot_1", "sc_3", "sc_5", "sc_7", "sc_9")
RCTA_METHODS = (
    "adaptive_sc_9",
    "gsa_trace_1",
    "mad_5a_r1",
    "confidence_mad_5a_r1",
    "rcta_certificate_shadow_1",
    "rcta_1",
    "rcta_no_certificate",
    "rcta_existing_only",
)
FULL_METHODS = frozenset({
    *CONTROL_METHODS,
    "adaptive_sc_9",
    "gsa_trace_1",
    "mad_5a_r1",
    "confidence_mad_5a_r1",
    "rcta_1",
})


@dataclass(frozen=True)
class RuntimeProfile:
    max_concurrent_requests: int
    requests_per_minute_limit: int


@dataclass(frozen=True)
class RctaProtocolConfig:
    stage_a_candidates: int
    sc_ceiling_candidates: int
    trace_synthesizer_count: int
    trigger_mode: str
    trace_max_chars: int
    board_max_chars: int
    reasoning_word_limit: int
    stage_a_temperature: float
    synthesis_temperature: float
    debate_temperature: float
    top_p: float
    stage_a_max_tokens: int
    synthesis_max_tokens: int
    debate_max_tokens: int
    router_artifact: Path
    router_mode: str
    risk_limit: float
    risk_delta: float


@dataclass(frozen=True)
class RctaExperimentConfig:
    name: str
    description: str
    benchmark_configs: list[Path]
    protocol: Path
    control_catalog: Path
    control_methods: list[str]
    rcta_methods: list[str]
    method_order: list[str]
    global_seed: int
    control_prompt_version: str
    primary_model_ref: str
    max_concurrent_requests: int
    requests_per_minute_limit: int
    runtime_profiles: dict[str, RuntimeProfile]
    raw: dict[str, Any]


def load_protocol_config(path: str | Path) -> RctaProtocolConfig:
    payload = load_toml(path)
    protocol = RctaProtocolConfig(
        stage_a_candidates=int(payload.get("stage_a_candidates", 5)),
        sc_ceiling_candidates=int(payload.get("sc_ceiling_candidates", 9)),
        trace_synthesizer_count=int(payload.get("trace_synthesizer_count", 1)),
        trigger_mode=str(payload.get("trigger_mode", "answer_disagreement")),
        trace_max_chars=int(payload.get("trace_max_chars", 1200)),
        board_max_chars=int(payload.get("board_max_chars", 7000)),
        reasoning_word_limit=int(payload.get("reasoning_word_limit", 120)),
        stage_a_temperature=float(payload.get("stage_a_temperature", 0.7)),
        synthesis_temperature=float(payload.get("synthesis_temperature", 0.7)),
        debate_temperature=float(payload.get("debate_temperature", 0.7)),
        top_p=float(payload.get("top_p", 1.0)),
        stage_a_max_tokens=int(payload.get("stage_a_max_tokens", 768)),
        synthesis_max_tokens=int(payload.get("synthesis_max_tokens", 2048)),
        debate_max_tokens=int(payload.get("debate_max_tokens", 2048)),
        router_artifact=Path(str(payload.get("router_artifact", "configs/families/risk_controlled_trace_mad/router/rcta_v1.json"))),
        router_mode=str(payload.get("router_mode", "shadow")),
        risk_limit=float(payload.get("risk_limit", 1.0 / 3.0)),
        risk_delta=float(payload.get("risk_delta", 0.05)),
    )
    fixed = {
        "stage_a_candidates": (protocol.stage_a_candidates, 5),
        "sc_ceiling_candidates": (protocol.sc_ceiling_candidates, 9),
        "trace_synthesizer_count": (protocol.trace_synthesizer_count, 1),
        "trigger_mode": (protocol.trigger_mode, "answer_disagreement"),
        "trace_max_chars": (protocol.trace_max_chars, 1200),
        "board_max_chars": (protocol.board_max_chars, 7000),
    }
    invalid = [f"{key}={actual!r} (required {expected!r})" for key, (actual, expected) in fixed.items() if actual != expected]
    if invalid:
        raise ValueError("RCTA-MAD V1 protocol is frozen: " + "; ".join(invalid))
    if protocol.router_mode not in {"shadow", "frozen"}:
        raise ValueError("router_mode must be 'shadow' or 'frozen'.")
    if not 0.0 < protocol.risk_limit < 1.0 or not 0.0 < protocol.risk_delta < 1.0:
        raise ValueError("risk_limit and risk_delta must lie in (0, 1).")
    return protocol


def load_experiment_config(path: str | Path) -> RctaExperimentConfig:
    payload = load_toml(path)
    runtime = apply_runtime_defaults(payload)
    controls = [str(item) for item in payload.get("control_methods", CONTROL_METHODS)]
    methods = [str(item) for item in payload.get("rcta_methods", RCTA_METHODS)]
    if set(controls) - set(CONTROL_METHODS):
        raise ValueError("Unsupported RCTA control methods: " + ", ".join(sorted(set(controls) - set(CONTROL_METHODS))))
    if set(methods) - set(RCTA_METHODS):
        raise ValueError("Unsupported RCTA methods: " + ", ".join(sorted(set(methods) - set(RCTA_METHODS))))
    order = [str(item) for item in payload.get("method_order", [*controls, *methods])]
    if set(order) != set(controls) | set(methods) or len(order) != len(set(order)):
        raise ValueError("method_order must exactly cover unique control_methods and rcta_methods.")
    if str(payload.get("control_prompt_version", FREE_TEXT_V1_PROMPT_VERSION)) != FREE_TEXT_V1_PROMPT_VERSION:
        raise ValueError("RCTA Stage A must match single_agent_free_text_v1.")
    profiles: dict[str, RuntimeProfile] = {}
    for name, values in dict(payload.get("runtime_profiles") or {}).items():
        profiles[str(name)] = RuntimeProfile(
            max_concurrent_requests=int(values["max_concurrent_requests"]),
            requests_per_minute_limit=int(values["requests_per_minute_limit"]),
        )
    return RctaExperimentConfig(
        name=str(payload["name"]),
        description=str(payload["description"]),
        benchmark_configs=[Path(item) for item in payload["benchmark_configs"]],
        protocol=Path(payload["protocol"]),
        control_catalog=Path(payload["control_catalog"]),
        control_methods=controls,
        rcta_methods=methods,
        method_order=order,
        global_seed=int(payload.get("global_seed", 42)),
        control_prompt_version=FREE_TEXT_V1_PROMPT_VERSION,
        primary_model_ref=str(payload["primary_model_ref"]),
        max_concurrent_requests=int(runtime["max_concurrent_requests"]),
        requests_per_minute_limit=int(runtime["requests_per_minute_limit"]),
        runtime_profiles=profiles,
        raw=payload,
    )


def load_control_catalog(path: str | Path) -> dict[str, MethodConfig]:
    return load_method_catalog(path)


def runtime_for_provider(experiment: RctaExperimentConfig, provider: str) -> RuntimeProfile:
    normalized = str(provider).lower()
    for key, profile in experiment.runtime_profiles.items():
        if key.lower() == normalized:
            return profile
    return RuntimeProfile(experiment.max_concurrent_requests, experiment.requests_per_minute_limit)


def phase_methods(experiment: RctaExperimentConfig, phase_name: str) -> list[str]:
    phase = dict(experiment.raw.get("phases", {}).get(phase_name, {}))
    methods = [str(item) for item in phase.get("rcta_methods", experiment.rcta_methods)]
    if not methods or set(methods) - set(experiment.rcta_methods):
        raise ValueError(f"Phase {phase_name!r} has unsupported or empty rcta_methods.")
    if phase_name == "full_seed42" and set(methods) - FULL_METHODS:
        raise ValueError("full_seed42 may not include development-only ablations.")
    return methods


def inspect_methods(experiment: RctaExperimentConfig) -> list[str]:
    return [*experiment.control_methods, *experiment.rcta_methods]


def inspect_benchmarks(experiment: RctaExperimentConfig) -> list[str]:
    return [item.slug for item in load_benchmarks(experiment)]

