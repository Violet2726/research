"""DGCR 的配置加载与冻结不变量。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research_experiments.family_runtime.config_helpers import apply_runtime_defaults, load_benchmarks, load_toml


@dataclass(frozen=True)
class DgcrProtocolConfig:
    stage_candidates: int
    resample_candidates: int
    panel_count: int
    temperature: float
    top_p: float
    solver_max_tokens: int
    role_max_tokens: int
    span_min_chars: int
    span_max_chars: int
    forbid_options_span: bool


@dataclass(frozen=True)
class DgcrExperimentConfig:
    name: str
    description: str
    benchmark_configs: list[Path]
    protocol: Path
    primary_model_ref: str
    global_seed: int
    max_concurrent_requests: int
    requests_per_minute_limit: int
    cache_policy: str
    provider_audit_path: Path
    raw: dict[str, Any]


def load_protocol_config(path: str | Path) -> DgcrProtocolConfig:
    raw = load_toml(path)
    config = DgcrProtocolConfig(
        stage_candidates=int(raw.get("stage_candidates", 5)),
        resample_candidates=int(raw.get("resample_candidates", 3)),
        panel_count=int(raw.get("panel_count", 2)),
        temperature=float(raw.get("temperature", 0.7)),
        top_p=float(raw.get("top_p", 1.0)),
        solver_max_tokens=int(raw.get("solver_max_tokens", 16_384)),
        role_max_tokens=int(raw.get("role_max_tokens", 2_048)),
        span_min_chars=int(raw.get("span_min_chars", 8)),
        span_max_chars=int(raw.get("span_max_chars", 256)),
        forbid_options_span=bool(raw.get("forbid_options_span", True)),
    )
    required = {
        "stage_candidates": (config.stage_candidates, 5),
        "resample_candidates": (config.resample_candidates, 3),
        "panel_count": (config.panel_count, 2),
        "temperature": (config.temperature, 0.7),
        "top_p": (config.top_p, 1.0),
        "solver_max_tokens": (config.solver_max_tokens, 16_384),
        "role_max_tokens": (config.role_max_tokens, 2_048),
        "span_min_chars": (config.span_min_chars, 8),
        "span_max_chars": (config.span_max_chars, 256),
        "forbid_options_span": (config.forbid_options_span, True),
    }
    invalid = [name for name, (actual, expected) in required.items() if actual != expected]
    if invalid:
        raise ValueError("DGCR protocol is frozen; invalid fields: " + ", ".join(invalid))
    return config


def load_experiment_config(path: str | Path) -> DgcrExperimentConfig:
    raw = load_toml(path)
    runtime = apply_runtime_defaults(raw)
    if "cache_namespaces" in raw:
        raise ValueError("DGCR retired cache_namespaces field is forbidden.")
    cache_policy = str(raw.get("cache_policy") or "")
    if cache_policy != "global_validated_response_v3":
        raise ValueError("DGCR requires cache_policy='global_validated_response_v3'.")
    provider_audit_raw = str(raw.get("provider_audit_path") or "").strip()
    if not provider_audit_raw:
        raise ValueError("DGCR requires a provider_audit_path before either gate phase can run.")
    provider_audit_path = Path(provider_audit_raw)
    return DgcrExperimentConfig(
        name=str(raw["name"]),
        description=str(raw["description"]),
        benchmark_configs=[Path(item) for item in raw["benchmark_configs"]],
        protocol=Path(raw["protocol"]),
        primary_model_ref=str(raw["primary_model_ref"]),
        global_seed=int(raw.get("global_seed", 42)),
        max_concurrent_requests=runtime["max_concurrent_requests"],
        requests_per_minute_limit=runtime["requests_per_minute_limit"],
        cache_policy=cache_policy,
        provider_audit_path=provider_audit_path,
        raw=raw,
    )


def phase_metadata(experiment: DgcrExperimentConfig, phase_name: str) -> dict[str, Any]:
    phase = dict(experiment.raw.get("phases", {}).get(phase_name, {}))
    if not phase:
        raise ValueError(f"Unknown DGCR phase {phase_name!r}.")
    if phase_name not in {"development", "heldout"}:
        raise ValueError("DGCR supports only development and heldout phases.")
    return phase


def load_phase_benchmarks(experiment: DgcrExperimentConfig, phase_name: str):
    phase = phase_metadata(experiment, phase_name)
    requested = {str(item) for item in phase.get("benchmark_slugs", ["bbeh"])}
    benchmarks = [item for item in load_benchmarks(experiment) if item.slug in requested]
    if {item.slug for item in benchmarks} != requested:
        raise ValueError("DGCR phase refers to an unknown benchmark.")
    return benchmarks
