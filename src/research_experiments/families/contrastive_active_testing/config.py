"""CATCH 配置加载与冻结协议不变量。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research_experiments.family_runtime.config_helpers import apply_runtime_defaults, load_benchmarks, load_toml


@dataclass(frozen=True)
class CatchProtocolConfig:
    stage_candidates: int
    resample_candidates: int
    witness_count: int
    direct_judge_count: int
    max_proposed_tests: int
    max_selected_tests: int
    temperature: float
    top_p: float
    solver_max_tokens: int
    role_max_tokens: int
    d_min_grid: tuple[int, ...]
    margin_grid: tuple[int, ...]
    max_network_attempts: int


@dataclass(frozen=True)
class CatchExperimentConfig:
    name: str
    description: str
    benchmark_configs: list[Path]
    protocol: Path
    primary_model_ref: str
    global_seed: int
    max_concurrent_requests: int
    requests_per_minute_limit: int
    cache_namespaces: dict[str, str]
    provider_audit_path: Path
    frozen_decoding_path: Path
    human_audit_path: Path
    raw: dict[str, Any]


def load_protocol_config(path: str | Path) -> CatchProtocolConfig:
    raw = load_toml(path)
    config = CatchProtocolConfig(
        stage_candidates=int(raw.get("stage_candidates", 5)),
        resample_candidates=int(raw.get("resample_candidates", 3)),
        witness_count=int(raw.get("witness_count", 2)),
        direct_judge_count=int(raw.get("direct_judge_count", 3)),
        max_proposed_tests=int(raw.get("max_proposed_tests", 6)),
        max_selected_tests=int(raw.get("max_selected_tests", 4)),
        temperature=float(raw.get("temperature", 0.7)),
        top_p=float(raw.get("top_p", 1.0)),
        solver_max_tokens=int(raw.get("solver_max_tokens", 16_384)),
        role_max_tokens=int(raw.get("role_max_tokens", 4_096)),
        d_min_grid=tuple(int(item) for item in raw.get("d_min_grid", [2, 3, 4])),
        margin_grid=tuple(int(item) for item in raw.get("margin_grid", [1, 2])),
        max_network_attempts=int(raw.get("max_network_attempts", 62_000)),
    )
    required = {
        "stage_candidates": (config.stage_candidates, 5),
        "resample_candidates": (config.resample_candidates, 3),
        "witness_count": (config.witness_count, 2),
        "direct_judge_count": (config.direct_judge_count, 3),
        "max_proposed_tests": (config.max_proposed_tests, 6),
        "max_selected_tests": (config.max_selected_tests, 4),
        "temperature": (config.temperature, 0.7),
        "top_p": (config.top_p, 1.0),
        "solver_max_tokens": (config.solver_max_tokens, 16_384),
        "role_max_tokens": (config.role_max_tokens, 4_096),
        "d_min_grid": (config.d_min_grid, (2, 3, 4)),
        "margin_grid": (config.margin_grid, (1, 2)),
        "max_network_attempts": (config.max_network_attempts, 62_000),
    }
    invalid = [name for name, (actual, expected) in required.items() if actual != expected]
    if invalid:
        raise ValueError("CATCH v1 protocol is frozen; invalid fields: " + ", ".join(invalid))
    return config


def load_experiment_config(path: str | Path) -> CatchExperimentConfig:
    raw = load_toml(path)
    runtime = apply_runtime_defaults(raw)
    namespaces = {str(key): str(value) for key, value in dict(raw.get("cache_namespaces") or {}).items()}
    required_namespaces = {"provider_audit", "development", "heldout", "confirmation"}
    if set(namespaces) != required_namespaces:
        raise ValueError(f"CATCH cache_namespaces must exactly contain {sorted(required_namespaces)}.")
    expected_namespaces = {
        "provider_audit": "catch-provider-audit-v1",
        "development": "catch-dev-v1",
        "heldout": "catch-heldout-v1",
        "confirmation": "catch-confirm-v1",
    }
    if namespaces != expected_namespaces:
        raise ValueError("CATCH v1 cache namespaces are frozen and must remain isolated.")
    return CatchExperimentConfig(
        name=str(raw["name"]),
        description=str(raw["description"]),
        benchmark_configs=[Path(item) for item in raw["benchmark_configs"]],
        protocol=Path(raw["protocol"]),
        primary_model_ref=str(raw["primary_model_ref"]),
        global_seed=int(raw.get("global_seed", 42)),
        max_concurrent_requests=runtime["max_concurrent_requests"],
        requests_per_minute_limit=runtime["requests_per_minute_limit"],
        cache_namespaces=namespaces,
        provider_audit_path=Path(str(raw["provider_audit_path"])),
        frozen_decoding_path=Path(str(raw["frozen_decoding_path"])),
        human_audit_path=Path(str(raw["human_audit_path"])),
        raw=raw,
    )


def phase_metadata(experiment: CatchExperimentConfig, phase_name: str) -> dict[str, Any]:
    if phase_name not in {"development", "heldout", "confirmation"}:
        raise ValueError(f"Unsupported CATCH phase {phase_name!r}.")
    phase = dict(experiment.raw.get("phases", {}).get(phase_name, {}))
    if not phase:
        raise ValueError(f"Missing CATCH phase configuration {phase_name!r}.")
    return phase


def load_phase_benchmarks(experiment: CatchExperimentConfig, phase_name: str):
    phase = phase_metadata(experiment, phase_name)
    requested = {str(item) for item in phase.get("benchmark_slugs", ["bbeh"])}
    benchmarks = [item for item in load_benchmarks(experiment) if item.slug in requested]
    if {item.slug for item in benchmarks} != requested:
        raise ValueError("CATCH phase refers to an unknown benchmark.")
    return benchmarks
