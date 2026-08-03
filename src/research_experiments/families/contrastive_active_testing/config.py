"""加载 CATCH 配置并执行冻结协议不变量检查。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research_experiments.families.contrastive_active_testing.d4_contract import (
    D4_MAINLINE_PROTOCOL_VERSION,
    D4_SOURCE_COMPILER_SMOKE_FAILED,
    D4_SOURCE_COMPILER_SMOKE_PASSED,
)
from research_experiments.family_runtime.config_helpers import apply_runtime_defaults, load_benchmarks, load_toml

GLOBAL_CACHE_POLICY = "global_validated_response_v3"


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
    judge_max_tokens: int = 4_096
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
    cache_policy: str
    provider_audit_path: Path
    frozen_decoding_path: Path
    human_audit_path: Path
    preflight_human_audit_path: Path
    readiness_assessment_path: Path
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
        max_selected_tests=int(
            raw.get(
                "max_selected_tests",
                0 if protocol_version == "catch_v3" else 6 if protocol_version in {"catch_cert_v2", "catch_kernel_v1"} else 4,
            )
        ),
        temperature=float(raw.get("temperature", 0.7)),
        top_p=float(raw.get("top_p", 1.0)),
        solver_max_tokens=int(raw.get("solver_max_tokens", 16_384)),
        role_max_tokens=int(raw.get("role_max_tokens", 4_096)),
        judge_max_tokens=int(raw.get("judge_max_tokens", raw.get("role_max_tokens", 4_096))),
        d_min_grid=tuple(int(item) for item in raw.get("d_min_grid", [] if protocol_version == "catch_v3" else [2, 3, 4])),
        margin_grid=tuple(int(item) for item in raw.get("margin_grid", [] if protocol_version == "catch_v3" else [1, 2])),
        max_network_attempts=int(raw.get("max_network_attempts", 62_000)),
        protocol_version=protocol_version,
        preflight_sample_count=int(raw.get("preflight_sample_count", 0)),
        preflight_quote_alignment_threshold=float(raw.get("preflight_quote_alignment_threshold", 0.0)),
        preflight_code_coverage_threshold=float(raw.get("preflight_code_coverage_threshold", 0.0)),
        preflight_coordinate_validity_threshold=float(raw.get("preflight_coordinate_validity_threshold", 0.0)),
        preflight_usable_pair_threshold=float(raw.get("preflight_usable_pair_threshold", 0.0)),
        coordinates_per_pair=int(raw.get("coordinates_per_pair", 3)),
        max_selected_contrasts=int(raw.get("max_selected_contrasts", 6)),
        pair_judge_count=int(raw.get("pair_judge_count", 3)),
        preflight_decisive_threshold=float(raw.get("preflight_decisive_threshold", 0.0)),
        preflight_panel_agreement_threshold=float(raw.get("preflight_panel_agreement_threshold", 0.0)),
        budget_scope=str(raw.get("budget_scope") or "confirmatory_gate"),
    )
    if protocol_version not in {
        "catch_v1",
        "catch_v2",
        "catch_v3",
        "catch_cert_v1",
        "catch_cert_v2",
        "catch_kernel_v1",
    }:
        raise ValueError(f"Unsupported CATCH protocol version {protocol_version!r}.")
    positive = {
        "stage_candidates": config.stage_candidates,
        "resample_candidates": config.resample_candidates,
        "witness_count": config.witness_count,
        "solver_max_tokens": config.solver_max_tokens,
        "role_max_tokens": config.role_max_tokens,
        "judge_max_tokens": config.judge_max_tokens,
        "max_network_attempts": config.max_network_attempts,
    }
    invalid = [name for name, value in positive.items() if int(value) <= 0]
    if invalid:
        raise ValueError("CATCH protocol fields must be positive: " + ", ".join(invalid))
    if not 0 <= config.temperature <= 2 or not 0 < config.top_p <= 1:
        raise ValueError("CATCH temperature/top_p are outside executable ranges.")
    if Path(path).name == "catch_kernel_d4_v3.toml" and (
        raw.get("d4_mainline_protocol_version") != D4_MAINLINE_PROTOCOL_VERSION
        or protocol_version != "catch_kernel_v1"
        or (config.solver_max_tokens, config.role_max_tokens, config.judge_max_tokens)
        != (65_536, 65_536, 32_768)
    ):
        raise ValueError("D4 requires its frozen 65536/65536/32768 completion-token protocol.")
    return config


def _validate_d4_mainline_config(
    raw: dict[str, Any],
    *,
    protocol_path: Path,
    protocol: CatchProtocolConfig,
) -> None:
    if protocol_path.name != "catch_kernel_d4_v3.toml":
        raise ValueError("D4 accepts only the frozen catch_kernel_d4_v3 protocol file.")
    if raw.get("d4_mainline_protocol_version") != D4_MAINLINE_PROTOCOL_VERSION:
        raise ValueError(f"D4 requires mainline protocol {D4_MAINLINE_PROTOCOL_VERSION}.")
    smoke_status = str(raw.get("source_compiler_smoke_status") or "")
    smoke_path = str(raw.get("source_compiler_smoke_result_path") or "")
    smoke_sha = str(raw.get("source_compiler_smoke_result_sha256") or "")
    if (
        smoke_status not in {D4_SOURCE_COMPILER_SMOKE_PASSED, D4_SOURCE_COMPILER_SMOKE_FAILED}
        or not smoke_path
        or re.fullmatch(r"[0-9a-f]{64}", smoke_sha) is None
    ):
        raise ValueError("D4 requires an explicit hash-linked source-compiler smoke status.")
    if (
        protocol.protocol_version != "catch_kernel_v1"
        or protocol.stage_candidates != 5
        or protocol.resample_candidates != 3
        or (protocol.solver_max_tokens, protocol.role_max_tokens, protocol.judge_max_tokens)
        != (65_536, 65_536, 32_768)
    ):
        raise ValueError("D4 requires the frozen 5-stage/3-compiler and 65536/65536/32768 protocol.")
    output = raw.get("d4_output")
    if not isinstance(output, dict):
        raise ValueError("D4 requires an explicit tagged-text d4_output configuration.")
    allowed_output = {
        "stage_a_protocol",
        "parse_failure_target",
        "stage_a_quorum_minimum_valid_turns",
        "stage_a_quorum_failure_target",
        "conflicts_fail_closed",
    }
    retired = set(output) - allowed_output
    if retired:
        raise ValueError("D4 retired configuration fields are forbidden: " + ", ".join(sorted(retired)))
    if output.get("stage_a_protocol") != "tagged_text":
        raise ValueError("D4 supports only the tagged_text Stage-A protocol.")
    for key in ("protocol_ab_assessment_path", "prompt_variant", "cache_namespaces", "baseline_cache_namespaces"):
        if key in raw:
            raise ValueError(f"D4 retired configuration field is forbidden: {key}")


def load_experiment_config(path: str | Path) -> CatchExperimentConfig:
    raw = load_toml(path)
    runtime = apply_runtime_defaults(raw)
    retired_cache_fields = {"cache_namespaces", "baseline_cache_namespaces"} & set(raw)
    if retired_cache_fields:
        raise ValueError("CATCH retired cache fields are forbidden: " + ", ".join(sorted(retired_cache_fields)))
    cache_policy = str(raw.get("cache_policy") or "")
    if cache_policy != GLOBAL_CACHE_POLICY:
        raise ValueError(f"CATCH requires cache_policy={GLOBAL_CACHE_POLICY!r}.")
    study_type = str(raw.get("study_type") or "confirmatory_gate")
    is_boundary = study_type == "post_failure_cross_domain_boundary_audit"
    protocol_path = Path(str(raw["protocol"]))
    protocol = load_protocol_config(protocol_path)
    kernel_revision = str(raw.get("kernel_revision") or "d1_pairwise_v1")
    if protocol_path.name == "catch_kernel_d4_v3.toml" and kernel_revision != "d4_proof_carrying_v1":
        raise ValueError("The D4 v3 protocol cannot be paired with a non-D4 kernel revision.")
    if kernel_revision == "d4_proof_carrying_v1":
        _validate_d4_mainline_config(raw, protocol_path=protocol_path, protocol=protocol)
    return CatchExperimentConfig(
        name=str(raw["name"]),
        description=str(raw["description"]),
        benchmark_configs=[Path(item) for item in raw["benchmark_configs"]],
        protocol=protocol_path,
        primary_model_ref=str(raw["primary_model_ref"]),
        global_seed=int(raw.get("global_seed", 42)),
        max_concurrent_requests=runtime["max_concurrent_requests"],
        requests_per_minute_limit=runtime["requests_per_minute_limit"],
        cache_policy=cache_policy,
        provider_audit_path=Path(str(raw.get("provider_audit_path") or "unused/provider_audit.json")),
        frozen_decoding_path=Path(str(raw.get("frozen_decoding_path") or "unused/frozen_decoding.json")),
        human_audit_path=Path(str(raw.get("human_audit_path") or "unused/human_audit.json")),
        preflight_human_audit_path=Path(
            str(raw.get("preflight_human_audit_path") or raw.get("human_audit_path") or "unused/preflight_human_audit.json")
        ),
        readiness_assessment_path=Path(
            str(
                raw.get("readiness_assessment_path")
                or raw.get("readiness_gate_path")
                or "unused/catch_cert_v2_readiness_assessment.json"
            )
        ),
        study_type=study_type,
        confirmatory=bool(raw.get("confirmatory", not is_boundary)),
        config_warnings=(),
        raw=raw,
    )


def phase_metadata(experiment: CatchExperimentConfig, phase_name: str) -> dict[str, Any]:
    allowed = (
        {"boundary_audit"}
        if experiment.study_type == "post_failure_cross_domain_boundary_audit"
        else {"development", "heldout", "confirmation"}
    )
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
