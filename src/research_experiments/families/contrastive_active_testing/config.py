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
    protocol_version: str = "catch_v1"
    preflight_sample_count: int = 0
    preflight_quote_alignment_threshold: float = 0.0
    preflight_code_coverage_threshold: float = 0.0
    preflight_coordinate_validity_threshold: float = 0.0
    preflight_usable_pair_threshold: float = 0.0
    coordinates_per_pair: int = 3
    max_selected_contrasts: int = 6
    pair_judge_count: int = 3
    preflight_decisive_threshold: float = 0.0
    preflight_panel_agreement_threshold: float = 0.0
    budget_scope: str = "confirmatory_gate"


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
    baseline_cache_namespaces: dict[str, str]
    provider_audit_path: Path
    frozen_decoding_path: Path
    human_audit_path: Path
    preflight_human_audit_path: Path
    study_type: str
    confirmatory: bool
    config_warnings: tuple[str, ...]
    raw: dict[str, Any]


def load_protocol_config(path: str | Path) -> CatchProtocolConfig:
    raw = load_toml(path)
    protocol_version = str(raw.get("protocol_version") or "catch_v1")
    config = CatchProtocolConfig(
        stage_candidates=int(raw.get("stage_candidates", 5)),
        resample_candidates=int(raw.get("resample_candidates", 3)),
        witness_count=int(raw.get("witness_count", 2)),
        direct_judge_count=int(raw.get("direct_judge_count", 3)),
        max_proposed_tests=int(raw.get("max_proposed_tests", 0 if protocol_version == "catch_v3" else 6)),
        max_selected_tests=int(raw.get("max_selected_tests", 0 if protocol_version == "catch_v3" else 4)),
        temperature=float(raw.get("temperature", 0.7)),
        top_p=float(raw.get("top_p", 1.0)),
        solver_max_tokens=int(raw.get("solver_max_tokens", 16_384)),
        role_max_tokens=int(raw.get("role_max_tokens", 4_096)),
        d_min_grid=tuple(
            int(item) for item in raw.get("d_min_grid", [] if protocol_version == "catch_v3" else [2, 3, 4])
        ),
        margin_grid=tuple(
            int(item) for item in raw.get("margin_grid", [] if protocol_version == "catch_v3" else [1, 2])
        ),
        max_network_attempts=int(raw.get("max_network_attempts", 62_000)),
        protocol_version=protocol_version,
        preflight_sample_count=int(raw.get("preflight_sample_count", 0)),
        preflight_quote_alignment_threshold=float(raw.get("preflight_quote_alignment_threshold", 0.0)),
        preflight_code_coverage_threshold=float(raw.get("preflight_code_coverage_threshold", 0.0)),
        preflight_coordinate_validity_threshold=float(
            raw.get("preflight_coordinate_validity_threshold", 0.0)
        ),
        preflight_usable_pair_threshold=float(raw.get("preflight_usable_pair_threshold", 0.0)),
        coordinates_per_pair=int(raw.get("coordinates_per_pair", 3)),
        max_selected_contrasts=int(raw.get("max_selected_contrasts", 6)),
        pair_judge_count=int(raw.get("pair_judge_count", 3)),
        preflight_decisive_threshold=float(raw.get("preflight_decisive_threshold", 0.0)),
        preflight_panel_agreement_threshold=float(raw.get("preflight_panel_agreement_threshold", 0.0)),
        budget_scope=str(raw.get("budget_scope") or "confirmatory_gate"),
    )
    if protocol_version not in {"catch_v1", "catch_v2", "catch_v3", "catch_cert_v1"}:
        raise ValueError(f"Unsupported CATCH protocol version {protocol_version!r}.")
    minimum_fields = {
        "stage_candidates": config.stage_candidates,
        "resample_candidates": config.resample_candidates,
        "witness_count": config.witness_count,
        "solver_max_tokens": config.solver_max_tokens,
        "role_max_tokens": config.role_max_tokens,
        "max_network_attempts": config.max_network_attempts,
    }
    invalid = [name for name, value in minimum_fields.items() if int(value) <= 0]
    if invalid:
        raise ValueError("CATCH protocol fields must be positive: " + ", ".join(invalid))
    if not 0 <= config.temperature <= 2 or not 0 < config.top_p <= 1:
        raise ValueError("CATCH temperature/top_p are outside executable ranges.")
    return config


def load_experiment_config(path: str | Path) -> CatchExperimentConfig:
    raw = load_toml(path)
    runtime = apply_runtime_defaults(raw)
    namespaces = {str(key): str(value) for key, value in dict(raw.get("cache_namespaces") or {}).items()}
    baseline_namespaces = {
        str(key): str(value)
        for key, value in dict(raw.get("baseline_cache_namespaces") or {}).items()
    }
    study_type = str(raw.get("study_type") or "confirmatory_gate")
    is_boundary = study_type == "post_failure_cross_domain_boundary_audit"
    config_warnings: list[str] = []
    required_namespaces = (
        {"bbeh", "musr", "seqbench", "gpqa_diamond"}
        if is_boundary
        else {"development", "heldout", "confirmation"}
    )
    missing_namespaces = required_namespaces - set(namespaces)
    if missing_namespaces:
        config_warnings.append(f"derived_missing_cache_namespaces:{sorted(missing_namespaces)}")
    protocol_path = Path(str(raw["protocol"]))
    protocol = load_protocol_config(protocol_path)
    namespace_version = protocol.protocol_version.removeprefix("catch_")
    expected_namespaces = (
        {
            "bbeh": "catch-boundary-v3-bbeh",
            "musr": "catch-boundary-v3-musr",
            "seqbench": "catch-boundary-v3-seqbench",
            "gpqa_diamond": "catch-boundary-v3-gpqa",
        }
        if is_boundary
        else {
            "development": "catch-dev-cert-v3-baseline",
            "heldout": "catch-heldout-cert-v3-baseline",
            "confirmation": "catch-confirm-cert-v3-baseline",
        }
        if study_type == "catch_cert_cross_domain_baseline"
        else {
            "development": f"catch-dev-{namespace_version}",
            "heldout": f"catch-heldout-{namespace_version}",
            "confirmation": f"catch-confirm-{namespace_version}",
        }
    )
    for key in missing_namespaces:
        namespaces[key] = expected_namespaces[key]
    if namespaces != expected_namespaces:
        config_warnings.append("cache_namespaces_differ_from_original_study")
    expected_baseline = (
        {"bbeh": "catch-dev-v3,catch-dev-v1"}
        if is_boundary
        else {
            "development": "catch-dev-cert_v1",
            "heldout": "catch-heldout-cert_v1",
            "confirmation": "catch-confirm-cert_v1",
        }
        if study_type == "catch_cert_cross_domain_baseline"
        else {"development": "catch-dev-v1"}
        if protocol.protocol_version in {"catch_v2", "catch_v3"}
        else {}
    )
    if baseline_namespaces != expected_baseline:
        config_warnings.append("baseline_cache_namespaces_differ_from_original_study")
    return CatchExperimentConfig(
        name=str(raw["name"]),
        description=str(raw["description"]),
        benchmark_configs=[Path(item) for item in raw["benchmark_configs"]],
        protocol=protocol_path,
        primary_model_ref=str(raw["primary_model_ref"]),
        global_seed=int(raw.get("global_seed", 42)),
        max_concurrent_requests=runtime["max_concurrent_requests"],
        requests_per_minute_limit=runtime["requests_per_minute_limit"],
        cache_namespaces=namespaces,
        baseline_cache_namespaces=baseline_namespaces,
        provider_audit_path=Path(str(raw.get("provider_audit_path") or "unused/provider_audit.json")),
        frozen_decoding_path=Path(str(raw.get("frozen_decoding_path") or "unused/frozen_decoding.json")),
        human_audit_path=Path(str(raw.get("human_audit_path") or "unused/human_audit.json")),
        preflight_human_audit_path=Path(
            str(
                raw.get("preflight_human_audit_path")
                or raw.get("human_audit_path")
                or "unused/preflight_human_audit.json"
            )
        ),
        study_type=study_type,
        confirmatory=bool(raw.get("confirmatory", not is_boundary)),
        config_warnings=tuple(config_warnings),
        raw=raw,
    )


def phase_metadata(experiment: CatchExperimentConfig, phase_name: str) -> dict[str, Any]:
    allowed = {"boundary_audit"} if experiment.study_type == "post_failure_cross_domain_boundary_audit" else {"development", "heldout", "confirmation"}
    if phase_name not in allowed:
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
