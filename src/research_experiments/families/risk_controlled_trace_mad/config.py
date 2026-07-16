"""统一 MAD 创新实验的版本、模型编组与协议配置。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research_experiments.family_runtime.config_helpers import load_benchmarks, load_toml

VERSION_STATUSES = frozenset({"active", "retired_negative", "retired_prerequisite_failure", "retired_superseded"})
METHODS = (
    "cot_1",
    "qwen_sc_5",
    "qwen_sc_9",
    "mimo_sc_5",
    "mimo_sc_9",
    "heterogeneous_mv_5",
    "heterogeneous_gsa_1",
    "mad_5a_r1",
    "hcp_mad_budget10",
    "minority_sentinel_reproduction",
    "evf_mad_1",
    "sc_3",
    "sc_5",
    "adaptive_sc_8",
    "conditional_resample_3",
    "blind_gsa_1",
    "blind_gsa_quorum_3",
    "hsgsa_unanimous_3",
)


@dataclass(frozen=True)
class RuntimeProfile:
    max_concurrent_requests: int
    requests_per_minute_limit: int


@dataclass(frozen=True)
class VersionRecord:
    version_id: str
    paper_name: str
    status: str
    source_commit: str
    config_sha256: str
    methods: tuple[str, ...]
    representative_run: str
    retirement_reason: str
    claim_boundary: str


@dataclass(frozen=True)
class VersionRegistry:
    active_version: str | None
    versions: dict[str, VersionRecord]


@dataclass(frozen=True)
class EvfProtocolConfig:
    stage_qwen_candidates: int
    stage_mimo_candidates: int
    evf_qwen_candidates: int
    evf_mimo_candidates: int
    trigger_mode: str
    stage_temperature: float
    selector_temperature: float
    audit_temperature: float
    cross_exam_temperature: float
    top_p: float
    selector_max_tokens: int
    audit_max_tokens: int
    cross_exam_max_tokens: int
    trace_max_chars: int
    board_max_chars: int
    max_logical_calls: int
    minimum_valid_stage_traces: int
    provider_abstention_limit: float
    challenger_required_passes: int
    anchor_required_falsifications: int


@dataclass(frozen=True)
class HsgsaProtocolConfig:
    protocol_kind: str
    stage_candidates: int
    resample_candidates: int
    reviewer_count: int
    trigger_mode: str
    stage_temperature: float
    reviewer_temperature: float
    top_p: float
    stage_max_tokens: int
    reviewer_max_tokens: int
    trace_max_chars: int
    board_max_chars: int
    max_logical_calls: int
    max_network_attempts: int
    minimum_valid_stage_traces: int
    provider_abstention_limit: float
    quorum_size: int
    unanimity_size: int
    hide_support_counts: bool
    allow_novel_answer: bool


@dataclass(frozen=True)
class MadInnovationExperimentConfig:
    name: str
    description: str
    active_version: str | None
    version_registry: Path
    protocol: Path
    benchmark_configs: list[Path]
    qwen_model_ref: str
    mimo_model_ref: str
    primary_model_ref: str
    global_seed: int
    max_concurrent_requests: int
    requests_per_minute_limit: int
    methods: list[str]
    method_order: list[str]
    runtime_profiles: dict[str, RuntimeProfile]
    raw: dict[str, Any]


def load_version_registry(path: str | Path) -> VersionRegistry:
    payload = load_toml(path)
    records: dict[str, VersionRecord] = {}
    for version_id, raw in dict(payload.get("versions") or {}).items():
        status = str(raw.get("status") or "")
        if status not in VERSION_STATUSES:
            raise ValueError(f"Unsupported version status for {version_id}: {status!r}")
        records[str(version_id)] = VersionRecord(
            version_id=str(version_id),
            paper_name=str(raw.get("paper_name") or version_id),
            status=status,
            source_commit=str(raw.get("source_commit") or ""),
            config_sha256=str(raw.get("config_sha256") or ""),
            methods=tuple(map(str, raw.get("methods") or [])),
            representative_run=str(raw.get("representative_run") or ""),
            retirement_reason=str(raw.get("retirement_reason") or ""),
            claim_boundary=str(raw.get("claim_boundary") or ""),
        )
    active = str(payload.get("active_version") or "").strip()
    active_records = [record.version_id for record in records.values() if record.status == "active"]
    if not active:
        if active_records:
            raise ValueError("A registry without active_version cannot contain active records.")
        return VersionRegistry(active_version=None, versions=records)
    if active not in records or active_records != [active]:
        raise ValueError("Version registry must contain exactly one active version matching active_version.")
    return VersionRegistry(active_version=active, versions=records)


def require_active_version(registry: VersionRegistry, requested: str | None) -> VersionRecord:
    version_id = requested or registry.active_version
    if version_id is None:
        raise ValueError("This MAD family is historical-only and has no active version to run.")
    if version_id not in registry.versions:
        raise ValueError(f"Unknown MAD innovation version: {version_id}")
    record = registry.versions[version_id]
    if record.status != "active":
        evidence = f"; representative_run={record.representative_run}" if record.representative_run else ""
        raise ValueError(
            f"Version {version_id} is {record.status} and cannot be run; "
            f"checkout source_commit={record.source_commit} for exact reproduction{evidence}."
        )
    return record


def load_protocol_config(path: str | Path) -> EvfProtocolConfig | HsgsaProtocolConfig:
    raw = load_toml(path)
    if str(raw.get("protocol_kind") or "evf") == "hsgsa":
        protocol = HsgsaProtocolConfig(
            **{
                field: caster(raw[field])
                for field, caster in {
                    "protocol_kind": str,
                    "stage_candidates": int,
                    "resample_candidates": int,
                    "reviewer_count": int,
                    "trigger_mode": str,
                    "stage_temperature": float,
                    "reviewer_temperature": float,
                    "top_p": float,
                    "stage_max_tokens": int,
                    "reviewer_max_tokens": int,
                    "trace_max_chars": int,
                    "board_max_chars": int,
                    "max_logical_calls": int,
                    "max_network_attempts": int,
                    "minimum_valid_stage_traces": int,
                    "provider_abstention_limit": float,
                    "quorum_size": int,
                    "unanimity_size": int,
                    "hide_support_counts": bool,
                    "allow_novel_answer": bool,
                }.items()
            }
        )
        required = {
            "stage_candidates": (protocol.stage_candidates, 5),
            "resample_candidates": (protocol.resample_candidates, 3),
            "reviewer_count": (protocol.reviewer_count, 3),
            "trigger_mode": (protocol.trigger_mode, "answer_class_disagreement"),
            "max_logical_calls": (protocol.max_logical_calls, 11),
            "quorum_size": (protocol.quorum_size, 2),
            "unanimity_size": (protocol.unanimity_size, 3),
            "hide_support_counts": (protocol.hide_support_counts, True),
            "allow_novel_answer": (protocol.allow_novel_answer, False),
        }
        errors = [
            f"{key}={actual!r}, required {expected!r}"
            for key, (actual, expected) in required.items()
            if actual != expected
        ]
        if errors:
            raise ValueError("H-SGSA v5 frozen invariant violation: " + "; ".join(errors))
        if protocol.max_network_attempts != 50_000:
            raise ValueError("H-SGSA v5 max_network_attempts must be exactly 50000.")
        if protocol.minimum_valid_stage_traces < 3:
            raise ValueError("minimum_valid_stage_traces must be at least three.")
        if not 0 <= protocol.provider_abstention_limit < 1:
            raise ValueError("provider_abstention_limit must lie in [0, 1).")
        return protocol
    protocol = EvfProtocolConfig(
        **{
            field: caster(raw[field])
            for field, caster in {
                "stage_qwen_candidates": int,
                "stage_mimo_candidates": int,
                "evf_qwen_candidates": int,
                "evf_mimo_candidates": int,
                "trigger_mode": str,
                "stage_temperature": float,
                "selector_temperature": float,
                "audit_temperature": float,
                "cross_exam_temperature": float,
                "top_p": float,
                "selector_max_tokens": int,
                "audit_max_tokens": int,
                "cross_exam_max_tokens": int,
                "trace_max_chars": int,
                "board_max_chars": int,
                "max_logical_calls": int,
                "minimum_valid_stage_traces": int,
                "provider_abstention_limit": float,
                "challenger_required_passes": int,
                "anchor_required_falsifications": int,
            }.items()
        }
    )
    required = {
        "stage_qwen_candidates": (protocol.stage_qwen_candidates, 9),
        "stage_mimo_candidates": (protocol.stage_mimo_candidates, 9),
        "evf_qwen_candidates": (protocol.evf_qwen_candidates, 3),
        "evf_mimo_candidates": (protocol.evf_mimo_candidates, 2),
        "trigger_mode": (protocol.trigger_mode, "answer_disagreement"),
        "max_logical_calls": (protocol.max_logical_calls, 10),
    }
    errors = [
        f"{key}={actual!r}, required {expected!r}" for key, (actual, expected) in required.items() if actual != expected
    ]
    if errors:
        raise ValueError("EVF-MAD v4 frozen invariant violation: " + "; ".join(errors))
    if not 0 <= protocol.provider_abstention_limit < 1:
        raise ValueError("provider_abstention_limit must lie in [0, 1).")
    if protocol.minimum_valid_stage_traces < 3:
        raise ValueError("minimum_valid_stage_traces must be at least three.")
    return protocol


def load_experiment_config(path: str | Path) -> MadInnovationExperimentConfig:
    raw = load_toml(path)
    methods = list(map(str, raw.get("methods") or []))
    unsupported = sorted(set(methods) - set(METHODS))
    if unsupported or not methods or len(methods) != len(set(methods)):
        raise ValueError(f"Invalid MAD innovation methods: unsupported={unsupported}")
    registry_path = Path(str(raw["version_registry"]))
    registry = load_version_registry(registry_path)
    active_version = str(raw.get("active_version") or "").strip() or None
    if active_version != registry.active_version:
        raise ValueError("Experiment active_version must match the version registry.")
    profiles = {
        str(name): RuntimeProfile(
            max_concurrent_requests=int(values["max_concurrent_requests"]),
            requests_per_minute_limit=int(values["requests_per_minute_limit"]),
        )
        for name, values in dict(raw.get("runtime_profiles") or {}).items()
    }
    return MadInnovationExperimentConfig(
        name=str(raw["name"]),
        description=str(raw["description"]),
        active_version=active_version,
        version_registry=registry_path,
        protocol=Path(str(raw["protocol"])),
        benchmark_configs=[Path(item) for item in raw["benchmark_configs"]],
        qwen_model_ref=str(raw.get("qwen_model_ref") or raw["primary_model_ref"]),
        mimo_model_ref=str(raw.get("mimo_model_ref") or raw["primary_model_ref"]),
        primary_model_ref=str(raw.get("primary_model_ref") or raw["qwen_model_ref"]),
        global_seed=int(raw.get("global_seed", 42)),
        max_concurrent_requests=int(raw.get("max_concurrent_requests", 1000)),
        requests_per_minute_limit=int(raw.get("requests_per_minute_limit", 1000)),
        methods=methods,
        method_order=list(methods),
        runtime_profiles=profiles,
        raw=raw,
    )


def phase_methods(experiment: MadInnovationExperimentConfig, phase_name: str) -> list[str]:
    try:
        phase = dict(experiment.raw["phases"][phase_name])
    except KeyError as exc:
        raise ValueError(f"Unknown canonical phase: {phase_name}") from exc
    methods = list(map(str, phase.get("methods") or experiment.methods))
    if not methods or set(methods) - set(experiment.methods):
        raise ValueError(f"Phase {phase_name} contains invalid methods.")
    return methods


def runtime_for_provider(experiment: MadInnovationExperimentConfig, provider: str) -> RuntimeProfile:
    try:
        return experiment.runtime_profiles[str(provider)]
    except KeyError as exc:
        raise ValueError(f"No frozen runtime profile for provider {provider!r}.") from exc


def inspect_methods(experiment: MadInnovationExperimentConfig) -> list[str]:
    return list(experiment.methods)


def inspect_benchmarks(experiment: MadInnovationExperimentConfig) -> list[str]:
    return [item.slug for item in load_benchmarks(experiment)]


# Narrow compatibility alias for shared runtime type hints; old family entry points are removed.
RctaExperimentConfig = MadInnovationExperimentConfig
RctaProtocolConfig = EvfProtocolConfig
