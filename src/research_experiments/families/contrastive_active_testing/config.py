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
    expected_attempt_cap = 3_000 if config.budget_scope == "boundary_audit" else 62_000
    required = {
        "stage_candidates": (config.stage_candidates, 5),
        "resample_candidates": (config.resample_candidates, 3),
        "witness_count": (config.witness_count, 2),
        "direct_judge_count": (config.direct_judge_count, 3),
        "temperature": (config.temperature, 0.7),
        "top_p": (config.top_p, 1.0),
        "solver_max_tokens": (config.solver_max_tokens, 16_384),
        "role_max_tokens": (config.role_max_tokens, 4_096),
        "max_network_attempts": (config.max_network_attempts, expected_attempt_cap),
    }
    invalid = [name for name, (actual, expected) in required.items() if actual != expected]
    if invalid:
        raise ValueError(f"{protocol_version} protocol has invalid frozen fields: " + ", ".join(invalid))
    if protocol_version == "catch_v2":
        v2_required = {
            "max_proposed_tests": (config.max_proposed_tests, 6),
            "max_selected_tests": (config.max_selected_tests, 4),
            "d_min_grid": (config.d_min_grid, (2, 3, 4)),
            "margin_grid": (config.margin_grid, (1, 2)),
            "preflight_sample_count": (config.preflight_sample_count, 20),
            "preflight_quote_alignment_threshold": (
                config.preflight_quote_alignment_threshold,
                0.95,
            ),
            "preflight_code_coverage_threshold": (config.preflight_code_coverage_threshold, 0.60),
            "preflight_coordinate_validity_threshold": (
                config.preflight_coordinate_validity_threshold,
                0.95,
            ),
            "preflight_usable_pair_threshold": (config.preflight_usable_pair_threshold, 0.90),
        }
        invalid_v2 = [name for name, (actual, expected) in v2_required.items() if actual != expected]
        if invalid_v2:
            raise ValueError("CATCH v2 preflight fields are frozen: " + ", ".join(invalid_v2))
    elif protocol_version == "catch_v3":
        v3_required = {
            "retired_max_proposed_tests": (config.max_proposed_tests, 0),
            "retired_max_selected_tests": (config.max_selected_tests, 0),
            "retired_d_min_grid": (config.d_min_grid, ()),
            "retired_margin_grid": (config.margin_grid, ()),
            "preflight_sample_count": (config.preflight_sample_count, 20),
            "coordinates_per_pair": (config.coordinates_per_pair, 3),
            "max_selected_contrasts": (config.max_selected_contrasts, 6),
            "pair_judge_count": (config.pair_judge_count, 3),
            "preflight_code_coverage_threshold": (config.preflight_code_coverage_threshold, 0.60),
            "preflight_coordinate_validity_threshold": (
                config.preflight_coordinate_validity_threshold,
                0.95,
            ),
            "preflight_usable_pair_threshold": (config.preflight_usable_pair_threshold, 0.90),
            "preflight_decisive_threshold": (config.preflight_decisive_threshold, 0.80),
            "preflight_panel_agreement_threshold": (
                config.preflight_panel_agreement_threshold,
                0.70,
            ),
        }
        invalid_v3 = [name for name, (actual, expected) in v3_required.items() if actual != expected]
        if invalid_v3:
            raise ValueError("CATCH v3 indexed-contrast fields are frozen: " + ", ".join(invalid_v3))
    elif protocol_version == "catch_v1":
        legacy_required = {
            "max_proposed_tests": (config.max_proposed_tests, 6),
            "max_selected_tests": (config.max_selected_tests, 4),
            "d_min_grid": (config.d_min_grid, (2, 3, 4)),
            "margin_grid": (config.margin_grid, (1, 2)),
        }
        invalid_legacy = [name for name, (actual, expected) in legacy_required.items() if actual != expected]
        if invalid_legacy:
            raise ValueError("CATCH v1 grid fields are frozen: " + ", ".join(invalid_legacy))
    else:
        raise ValueError(f"Unsupported CATCH protocol version {protocol_version!r}.")
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
    required_namespaces = (
        {"provider_audit", "bbeh", "musr", "seqbench", "gpqa_diamond"}
        if is_boundary
        else {"provider_audit", "development", "heldout", "confirmation"}
    )
    if set(namespaces) != required_namespaces:
        raise ValueError(f"CATCH cache_namespaces must exactly contain {sorted(required_namespaces)}.")
    protocol_path = Path(str(raw["protocol"]))
    protocol = load_protocol_config(protocol_path)
    namespace_version = protocol.protocol_version.removeprefix("catch_")
    expected_namespaces = (
        {
            "provider_audit": "catch-provider-audit-v1",
            "bbeh": "catch-boundary-v3-bbeh",
            "musr": "catch-boundary-v3-musr",
            "seqbench": "catch-boundary-v3-seqbench",
            "gpqa_diamond": "catch-boundary-v3-gpqa",
        }
        if is_boundary
        else {
            "provider_audit": "catch-provider-audit-v1",
            "development": f"catch-dev-{namespace_version}",
            "heldout": f"catch-heldout-{namespace_version}",
            "confirmation": f"catch-confirm-{namespace_version}",
        }
    )
    if namespaces != expected_namespaces:
        raise ValueError(f"{protocol.protocol_version} cache namespaces are frozen and must remain isolated.")
    expected_baseline = (
        {"bbeh": "catch-dev-v3,catch-dev-v1"}
        if is_boundary
        else {"development": "catch-dev-v1"}
        if protocol.protocol_version in {"catch_v2", "catch_v3"}
        else {}
    )
    if baseline_namespaces != expected_baseline:
        raise ValueError(
            f"{protocol.protocol_version} baseline cache namespaces must equal {expected_baseline}."
        )
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
        provider_audit_path=Path(str(raw["provider_audit_path"])),
        frozen_decoding_path=Path(str(raw["frozen_decoding_path"])),
        human_audit_path=Path(str(raw["human_audit_path"])),
        preflight_human_audit_path=Path(
            str(raw.get("preflight_human_audit_path") or raw["human_audit_path"])
        ),
        study_type=study_type,
        confirmatory=bool(raw.get("confirmatory", not is_boundary)),
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
