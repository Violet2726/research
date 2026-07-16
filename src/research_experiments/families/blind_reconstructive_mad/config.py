"""BRD-MAD 的配置加载与冻结不变量。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research_experiments.core.controls.control_prompts import FREE_TEXT_V1_PROMPT_VERSION
from research_experiments.family_runtime.config_helpers import apply_runtime_defaults, load_benchmarks, load_toml
from research_experiments.family_runtime.free_text_protocol import FREE_TEXT_ANSWER_PROTOCOL_V1
from research_experiments.family_runtime.method_catalog import MethodConfig, load_method_catalog

CONDITIONAL_RESAMPLE_3 = "conditional_resample_3"
GSA_QUORUM_3 = "gsa_quorum_3"
BRD_QUORUM_3 = "brd_quorum_3"
BRD_VISIBLE_SUPPORT_3 = "brd_visible_support_3"
SGSA_UNANIMOUS_3 = "sgsa_unanimous_3"
SGSA_VISIBLE_SUPPORT_3 = "sgsa_visible_support_3"
CONCISE_BRD_QUORUM_3 = "concise_brd_quorum_3"
CONCISE_BRD_UNANIMOUS_3 = "concise_brd_unanimous_3"
BRD_METHODS = frozenset(
    {
        CONDITIONAL_RESAMPLE_3,
        GSA_QUORUM_3,
        BRD_QUORUM_3,
        BRD_VISIBLE_SUPPORT_3,
        SGSA_UNANIMOUS_3,
        SGSA_VISIBLE_SUPPORT_3,
        CONCISE_BRD_QUORUM_3,
        CONCISE_BRD_UNANIMOUS_3,
    }
)


@dataclass(frozen=True)
class BrdProtocolConfig:
    stage_a_candidates: int
    reviewer_count: int
    trigger_mode: str
    hide_vote_counts: bool
    strong_majority_quorum: int
    default_quorum: int
    novel_answer_mode: str
    stage_a_temperature: float
    reviewer_temperature: float
    top_p: float
    stage_a_max_tokens: int
    reviewer_max_tokens: int
    representative_max_chars: int


@dataclass(frozen=True)
class RuntimeProfile:
    max_concurrent_requests: int
    requests_per_minute_limit: int


@dataclass(frozen=True)
class BrdMadExperimentConfig:
    name: str
    description: str
    benchmark_configs: list[Path]
    protocol: Path
    control_catalog: Path
    control_methods: list[str]
    brd_methods: list[str]
    method_order: list[str]
    global_seed: int
    control_prompt_version: str
    output_protocol: str
    primary_model_ref: str
    max_concurrent_requests: int
    requests_per_minute_limit: int
    runtime_profiles: dict[str, RuntimeProfile]
    raw: dict[str, Any]


def load_protocol_config(path: str | Path) -> BrdProtocolConfig:
    payload = load_toml(path)
    protocol = BrdProtocolConfig(
        stage_a_candidates=int(payload.get("stage_a_candidates", 5)),
        reviewer_count=int(payload.get("reviewer_count", 3)),
        trigger_mode=str(payload.get("trigger_mode", "answer_disagreement")),
        hide_vote_counts=bool(payload.get("hide_vote_counts", True)),
        strong_majority_quorum=int(payload.get("strong_majority_quorum", 3)),
        default_quorum=int(payload.get("default_quorum", 2)),
        novel_answer_mode=str(payload.get("novel_answer_mode", "shadow")),
        stage_a_temperature=float(payload.get("stage_a_temperature", 0.7)),
        reviewer_temperature=float(payload.get("reviewer_temperature", 0.7)),
        top_p=float(payload.get("top_p", 1.0)),
        stage_a_max_tokens=int(payload.get("stage_a_max_tokens", 768)),
        reviewer_max_tokens=int(payload.get("reviewer_max_tokens", 768)),
        representative_max_chars=int(payload.get("representative_max_chars", 6000)),
    )
    _validate_protocol(protocol)
    return protocol


def load_control_catalog(path: str | Path) -> dict[str, MethodConfig]:
    return load_method_catalog(path)


def load_experiment_config(path: str | Path) -> BrdMadExperimentConfig:
    payload = load_toml(path)
    runtime = apply_runtime_defaults(payload)
    control_methods = [str(item) for item in payload.get("control_methods", [])]
    brd_methods = [str(item) for item in payload.get("brd_methods", [])]
    unsupported = sorted(set(brd_methods) - BRD_METHODS)
    if unsupported:
        raise ValueError("Unsupported BRD-MAD methods: " + ", ".join(unsupported))
    method_order = [str(item) for item in payload.get("method_order", [])]
    if set(method_order) != set(control_methods) | set(brd_methods) or len(method_order) != len(set(method_order)):
        raise ValueError("method_order must exactly cover unique control_methods and brd_methods.")
    if str(payload.get("control_prompt_version", FREE_TEXT_V1_PROMPT_VERSION)) != FREE_TEXT_V1_PROMPT_VERSION:
        raise ValueError("BRD-MAD Stage A must use single_agent_free_text_v1, identical to sc_5.")
    if str(payload.get("output_protocol", FREE_TEXT_ANSWER_PROTOCOL_V1)) != FREE_TEXT_ANSWER_PROTOCOL_V1:
        raise ValueError("BRD-MAD must use free_text_answer_v1; JSON forcing is intentionally excluded.")
    profiles = _load_runtime_profiles(payload)
    return BrdMadExperimentConfig(
        name=str(payload["name"]),
        description=str(payload["description"]),
        benchmark_configs=[Path(item) for item in payload["benchmark_configs"]],
        protocol=Path(payload["protocol"]),
        control_catalog=Path(payload["control_catalog"]),
        control_methods=control_methods,
        brd_methods=brd_methods,
        method_order=method_order,
        global_seed=int(payload["global_seed"]),
        control_prompt_version=FREE_TEXT_V1_PROMPT_VERSION,
        output_protocol=FREE_TEXT_ANSWER_PROTOCOL_V1,
        primary_model_ref=str(payload["primary_model_ref"]),
        max_concurrent_requests=runtime["max_concurrent_requests"],
        requests_per_minute_limit=runtime["requests_per_minute_limit"],
        runtime_profiles=profiles,
        raw=payload,
    )


def runtime_for_provider(experiment: BrdMadExperimentConfig, provider_name: str) -> RuntimeProfile:
    key = str(provider_name).strip().lower()
    for alias, profile in experiment.runtime_profiles.items():
        if key == alias.lower():
            return profile
    return RuntimeProfile(experiment.max_concurrent_requests, experiment.requests_per_minute_limit)


def inspect_methods(experiment: BrdMadExperimentConfig) -> list[str]:
    return [*experiment.control_methods, *experiment.brd_methods]


def inspect_benchmarks(experiment: BrdMadExperimentConfig) -> list[str]:
    return [benchmark.slug for benchmark in load_benchmarks(experiment)]


def phase_brd_methods(experiment: BrdMadExperimentConfig, phase_name: str) -> list[str]:
    phase = dict(experiment.raw.get("phases", {}).get(phase_name, {}))
    methods = [str(item) for item in phase.get("brd_methods", experiment.brd_methods)]
    if not methods or set(methods) - set(experiment.brd_methods):
        raise ValueError(f"Phase {phase_name!r} has unsupported or empty brd_methods.")
    return methods


def _load_runtime_profiles(payload: dict[str, Any]) -> dict[str, RuntimeProfile]:
    profiles: dict[str, RuntimeProfile] = {}
    raw = payload.get("runtime_profiles") or {}
    if not isinstance(raw, dict):
        raise ValueError("runtime_profiles must be a TOML table.")
    for name, values in raw.items():
        if not isinstance(values, dict):
            raise ValueError(f"runtime_profiles.{name} must be a table.")
        profiles[str(name)] = RuntimeProfile(
            max_concurrent_requests=int(values["max_concurrent_requests"]),
            requests_per_minute_limit=int(values["requests_per_minute_limit"]),
        )
    return profiles


def _validate_protocol(protocol: BrdProtocolConfig) -> None:
    fixed = {
        "stage_a_candidates": (protocol.stage_a_candidates, 5),
        "reviewer_count": (protocol.reviewer_count, 3),
        "trigger_mode": (protocol.trigger_mode, "answer_disagreement"),
        "hide_vote_counts": (protocol.hide_vote_counts, True),
        "strong_majority_quorum": (protocol.strong_majority_quorum, 3),
        "default_quorum": (protocol.default_quorum, 2),
        "novel_answer_mode": (protocol.novel_answer_mode, "shadow"),
    }
    invalid = [f"{name}={actual!r} (required {expected!r})" for name, (actual, expected) in fixed.items() if actual != expected]
    if invalid:
        raise ValueError("BRD-MAD V1 protocol is frozen: " + "; ".join(invalid))
