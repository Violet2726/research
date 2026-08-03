"""带隔离缓存、硬门控和可审计生命周期的 CATCH 执行器。"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from typing import Any

from dotenv import load_dotenv

from research_experiments.core.data.datasets import (
    load_samples,
    load_split_ids,
    resolve_dataset_source_path,
    resolve_split_manifest_path,
    select_samples,
)
from research_experiments.core.data.evaluation import score_prediction
from research_experiments.core.execution.cache import CACHE_KEY_POLICY_VERSION, RequestCacheRouter
from research_experiments.core.execution.provider_audit import evaluate_mimo_provider_audit
from research_experiments.core.execution.providers import OpenAICompatibleProvider
from research_experiments.core.execution.rate_limits import RequestThrottle
from research_experiments.core.execution.runtime import RunProgressTracker, build_run_id
from research_experiments.families.contrastive_active_testing.algorithms import build_stage_decision
from research_experiments.families.contrastive_active_testing.cert_prompts import (
    CERT_PROMPT_VERSION,
    CERT_SCHEMA_VERSION,
)
from research_experiments.families.contrastive_active_testing.cert_prompts_v2 import (
    CERT_V2_PROMPT_VERSION,
    CERT_V2_SCHEMA_VERSION,
)
from research_experiments.families.contrastive_active_testing.config import (
    load_experiment_config,
    load_phase_benchmarks,
    load_protocol_config,
    phase_metadata,
)
from research_experiments.families.contrastive_active_testing.d4_audit import (
    evaluate_d4_human_audit,
    validate_d4_gate_evidence,
)
from research_experiments.families.contrastive_active_testing.d4_contract import (
    D4_MAINLINE_PROTOCOL_VERSION,
    D4_SOURCE_COMPILER_SMOKE_PASSED,
)
from research_experiments.families.contrastive_active_testing.d4_data import (
    text_sha256,
    validate_sealed_manifest,
    validate_text_sealed_manifest,
)
from research_experiments.families.contrastive_active_testing.d4_protocol_validation import (
    validate_tagged_protocol_validation_assessment,
)
from research_experiments.families.contrastive_active_testing.icv import build_target_pairs
from research_experiments.families.contrastive_active_testing.kernel import (
    KERNEL_CAPABILITY_VERSION,
    KERNEL_D2_DECODER_VERSION,
    KERNEL_D3_DECODER_VERSION,
    KERNEL_DECODER_VERSION,
    KERNEL_SCHEMA_VERSION,
    KERNEL_SEMANTICS_VERSION,
)
from research_experiments.families.contrastive_active_testing.kernel_d3 import (
    D3_CAPABILITY_REGISTRY_VERSION,
    capability_registry,
)
from research_experiments.families.contrastive_active_testing.kernel_d4 import (
    D4_CAPABILITY_REGISTRY_VERSION,
    D4_DECODER_VERSION,
    D4_IR_SCHEMA,
    D4_PROMPT_VERSION,
    load_risk_evidence,
    risk_gate_snapshot,
)
from research_experiments.families.contrastive_active_testing.kernel_d4 import (
    capability_registry as d4_capability_registry,
)
from research_experiments.families.contrastive_active_testing.kernel_d4 import (
    route_for_sample as d4_route_for_sample,
)
from research_experiments.families.contrastive_active_testing.kernel_prompts import (
    D3_PROMPT_VERSION,
    KERNEL_PROMPT_VERSION,
)
from research_experiments.families.contrastive_active_testing.prompts import (
    CATCH_PROMPT_VERSION,
    CATCH_SCHEMA_VERSION,
)
from research_experiments.families.contrastive_active_testing.replay import replay_canonicalization
from research_experiments.families.contrastive_active_testing.run.d4_ledger import (
    D4CompletionLedger,
)
from research_experiments.families.contrastive_active_testing.run.lifecycle import (
    render_report_with_fallback,
    write_nonblocking_validation,
)
from research_experiments.families.contrastive_active_testing.run.preflight import (
    PreflightJob,
    evaluate_icv_human_audit,
    run_icv_structural_preflight,
)
from research_experiments.families.contrastive_active_testing.run.report import render_report
from research_experiments.families.contrastive_active_testing.run.sample import (
    NetworkAttemptBudget,
    run_catch_sample,
    run_stage_a_only_sample,
)
from research_experiments.families.contrastive_active_testing.statistics import (
    build_best_effort_diagnostics,
    build_metrics,
    materialize_development_catch,
)
from research_experiments.family_runtime.layout import prepare_registered_run_layout
from research_experiments.family_runtime.manifest import finalize_family_manifest
from research_experiments.workspace.layout import default_cache_root, default_runs_root

D4_OUTPUT_PROTOCOL_ONLY_ROLE_PREFIXES = (
    "d4_output_protocol_independent_validation_",
)
D4_INDEPENDENT_PROTOCOL_VALIDATION_ROLE = (
    "d4_output_protocol_independent_validation_tagged_v3"
)
D4_INDEPENDENT_PROTOCOL_VALIDATION_DATA_ROLE = (
    "fresh_independent_protocol_validation_after_prompt_freeze"
)
D4_INDEPENDENT_PROTOCOL_VALIDATION_BENCHMARKS = {
    "bbeh_extension",
    "musr_x",
    "supergpqa_science",
}


def d4_source_compiler_smoke_snapshot(experiment) -> dict[str, Any]:
    """Read the configured smoke result without changing its frozen status."""

    status = str(experiment.raw.get("source_compiler_smoke_status") or "")
    path = Path(str(experiment.raw.get("source_compiler_smoke_result_path") or ""))
    expected_sha = str(experiment.raw.get("source_compiler_smoke_result_sha256") or "")
    snapshot: dict[str, Any] = {
        "configured_status": status,
        "path": path.as_posix(),
        "expected_sha256": expected_sha,
        "artifact_available": path.is_file(),
        "artifact_valid": False,
        "mainline_compatible": False,
        "artifact_passed": False,
    }
    if not path.is_file():
        return snapshot
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return snapshot
    actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path = path.with_name("manifest.json")
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            loaded_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded_manifest, dict):
                manifest = loaded_manifest
        except (OSError, json.JSONDecodeError):
            pass
    manifest_protocol = dict(manifest.get("protocol") or {})
    mainline_compatible = (
        payload.get("d4_mainline_protocol_version") == D4_MAINLINE_PROTOCOL_VERSION
        and manifest.get("d4_mainline_protocol_version") == D4_MAINLINE_PROTOCOL_VERSION
        and manifest.get("protocol_sha256") == payload.get("protocol_sha256")
        and manifest_protocol.get("solver_max_tokens") == 65_536
        and manifest_protocol.get("role_max_tokens") == 65_536
        and manifest_protocol.get("judge_max_tokens") == 32_768
    )
    snapshot.update(
        {
            "actual_sha256": actual_sha,
            "artifact_valid": actual_sha == expected_sha
            and payload.get("schema") == "catch_d4_source_compiler_smoke_v1",
            "mainline_compatible": mainline_compatible,
            "artifact_passed": payload.get("passed") is True,
            "run_id": payload.get("run_id"),
        }
    )
    return snapshot


def require_passing_d4_source_compiler_smoke(experiment) -> dict[str, Any]:
    """Fail closed unless the configured, hash-linked D4 smoke passed."""

    snapshot = d4_source_compiler_smoke_snapshot(experiment)
    if not (
        snapshot["configured_status"] == D4_SOURCE_COMPILER_SMOKE_PASSED
        and snapshot["artifact_valid"]
        and snapshot["mainline_compatible"]
        and snapshot["artifact_passed"]
    ):
        raise RuntimeError(
            "D4 downstream execution is blocked by the frozen SourceIR v3 compiler smoke: "
            f"{snapshot}."
        )
    return snapshot


@dataclass(frozen=True)
class CatchEndpoint:
    backbone: object
    provider: OpenAICompatibleProvider
    cache: object
    throttle: RequestThrottle
    stop_event: Event | None = None
    completion_ledger: D4CompletionLedger | None = None

    def cache_for_role(self, role: str):
        del role
        return self.cache

    def cache_lookup_namespaces_for_role(self, role: str) -> tuple[str, ...]:
        del role
        return ("global_cache",)


@dataclass(frozen=True)
class CatchSampleJob:
    sequence_index: int
    sample: Any
    split_name: str
    endpoint: CatchEndpoint


def run_experiment(
    experiment,
    phase_name: str,
    backbone,
    run_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    *,
    run_mode: str = "full",
    resume_run_dir: str | Path | None = None,
) -> Path:
    if experiment.study_type == "post_failure_cross_domain_boundary_audit":
        if phase_name != "boundary_audit":
            raise ValueError("The cross-domain boundary study exposes only the boundary_audit phase.")
        from research_experiments.families.contrastive_active_testing.run.boundary_execute import (
            run_boundary_audit,
        )

        return run_boundary_audit(experiment, backbone, run_root=run_root, cache_root=cache_root)
    load_dotenv(".env.local", override=False)
    execution_warnings: list[str] = list(getattr(experiment, "config_warnings", ()))
    phase = phase_metadata(experiment, phase_name)
    protocol = load_protocol_config(experiment.protocol)
    is_d4_kernel_run = (
        protocol.protocol_version == "catch_kernel_v1"
        and str(experiment.raw.get("kernel_revision") or "") == "d4_proof_carrying_v1"
    )
    evaluation_role = str(phase.get("evaluation_role") or "")
    is_d4_output_protocol_only = is_d4_kernel_run and evaluation_role.startswith(
        D4_OUTPUT_PROTOCOL_ONLY_ROLE_PREFIXES
    )
    is_d4_independent_protocol_validation = (
        is_d4_kernel_run
        and evaluation_role == D4_INDEPENDENT_PROTOCOL_VALIDATION_ROLE
    )
    if is_d4_independent_protocol_validation or (
        phase_name == "confirmation" and is_d4_kernel_run
    ):
        require_passing_d4_source_compiler_smoke(experiment)
    if (
        phase_name == "confirmation"
        and protocol.protocol_version == "catch_kernel_v1"
        and str(experiment.raw.get("kernel_revision") or "") == "d4_proof_carrying_v1"
        and not bool(phase.get("sealed_data_ready", False))
    ):
        raise RuntimeError(
            "D4 confirmation is blocked: sealed_data_ready=true is required after the BBEH-extension, "
            "MuSR-X, and SuperGPQA Science manifests and selections are frozen."
        )
    if backbone.provider != "xiaomimimo":
        execution_warnings.append(f"provider_differs_from_original_study:{backbone.provider}")
    if run_mode not in {"full", "structural_preflight"}:
        raise ValueError(f"Unsupported CATCH run mode {run_mode!r}.")
    if run_mode == "structural_preflight" and (phase_name != "development" or protocol.protocol_version != "catch_v3"):
        raise ValueError("The one-shot structural preflight is defined only for CATCH-v3 development.")
    if run_mode == "structural_preflight":
        execution_warnings.append("legacy_structural_preflight_request_ignored_running_full_phase")
        run_mode = "full"
    try:
        frozen_components = _frozen_component_hashes(experiment)
    except (OSError, KeyError, ValueError) as exc:
        execution_warnings.append(f"component_hash_unavailable:{type(exc).__name__}:{exc}")
        frozen_components = {"best_effort_hash_status": hashlib.sha256(str(exc).encode("utf-8")).hexdigest()}
    config_sha = _frozen_config_sha(experiment, component_hashes=frozen_components)
    provider_audit = {
        "required": False,
        "status": "not_run",
        "path": experiment.provider_audit_path.as_posix(),
    }
    run_root = Path(run_root or default_runs_root("contrastive_active_testing"))
    cache_root = Path(cache_root or default_cache_root())
    preflight_dependency = None
    frozen_decoding = None
    readiness_assessment = None
    if phase_name in {"heldout", "confirmation"} and protocol.protocol_version == "catch_cert_v2":
        readiness_assessment = _load_cert_v2_readiness_assessment(
            experiment.readiness_assessment_path,
            config_sha=config_sha,
        )
        if readiness_assessment["status"] != "available":
            execution_warnings.append(
                f"cert_v2_readiness_assessment_{readiness_assessment['status']}:results_are_exploratory"
            )
        elif readiness_assessment["unmet_conditions"]:
            execution_warnings.append(
                "cert_v2_readiness_recommendations_unmet:"
                f"{len(readiness_assessment['unmet_conditions'])}:results_are_exploratory"
            )
    if phase_name in {"heldout", "confirmation"}:
        if protocol.protocol_version in {"catch_v3", "catch_cert_v1", "catch_cert_v2", "catch_kernel_v1"}:
            frozen_decoding = _build_frozen_protocol_candidate(
                run_id="built_in_fixed_protocol",
                config_sha=config_sha,
                protocol_version=protocol.protocol_version,
                kernel_revision=str(experiment.raw.get("kernel_revision") or "d1_pairwise_v1"),
            )
            frozen_decoding["source"] = (
                "built_in_fixed_v3_decoder"
                if protocol.protocol_version == "catch_v3"
                else "built_in_fixed_kernel_decoder"
                if protocol.protocol_version == "catch_kernel_v1"
                else "built_in_fixed_cert_v2_decoder"
                if protocol.protocol_version == "catch_cert_v2"
                else "built_in_fixed_cert_decoder"
            )
        else:
            try:
                frozen_decoding = _load_frozen_decoding(
                    experiment.frozen_decoding_path,
                    config_sha=config_sha,
                )
                frozen_decoding["source"] = "optional_frozen_file"
            except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
                execution_warnings.append(f"frozen_decoding_unavailable_using_default_d2_m1:{type(exc).__name__}:{exc}")
                frozen_decoding = {
                    "freeze_kind": "catch_decoding_best_effort_default",
                    "d_min": 2,
                    "margin": 1,
                    "selection_constraints_passed": False,
                    "source": "protocol_default",
                }
    is_d4_confirmation = (
        phase_name == "confirmation"
        and is_d4_kernel_run
    )
    if phase_name in {"heldout", "confirmation"} and not is_d4_confirmation:
        execution_warnings.append("confirmatory_evidence_missing_or_not_enforced")

    cache_policy = "global_validated_response_v3"
    d4_human_audit = None
    d4_sealed_manifests = None
    d4_protocol_validation = None
    d4_risk_evidence = None
    if is_d4_independent_protocol_validation:
        _validate_d4_independent_protocol_validation_config(
            experiment,
            phase_name=phase_name,
            phase=phase,
        )
        d4_sealed_manifests = _require_d4_sealed_manifests(
            dict(phase.get("sealed_manifest_paths") or {}),
            expected_labels=D4_INDEPENDENT_PROTOCOL_VALIDATION_BENCHMARKS,
            expectations=dict(phase.get("sealed_manifest_expectations") or {}),
        )
    if is_d4_output_protocol_only:
        provider_audit = {
            "required": True,
            "status": "passed",
            **_require_passing_provider_audit(
                experiment.provider_audit_path,
                expected_cache_policy="live_only",
                expected_provider=str(backbone.provider),
                expected_model_id=str(backbone.model_id),
            ),
        }
    if is_d4_confirmation:
        if not experiment.confirmatory:
            raise RuntimeError("D4 confirmation is blocked: confirmatory=true is required.")
        expected_hashes = dict(phase.get("expected_selection_sha256") or {})
        if not expected_hashes:
            raise RuntimeError("D4 confirmation is blocked: frozen selection hashes are required.")
        freeze_path = Path(str(experiment.raw.get("kernel_freeze_path") or ""))
        if not freeze_path.is_file():
            raise RuntimeError("D4 confirmation is blocked: kernel freeze is missing.")
        d4_protocol_validation = _require_passing_d4_protocol_validation(
            Path(str(experiment.raw.get("protocol_validation_assessment_path") or ""))
        )
        d4_risk_evidence = _require_passing_d4_risk_evidence(experiment)
        d4_risk = dict(experiment.raw.get("d4_risk") or {})
        if bool(d4_risk.get("semantic_override_enabled", False)):
            d4_human_audit = _require_passing_d4_human_audit(
                experiment.human_audit_path,
                required_kernels=set(d4_risk_evidence["required_semantic_kernels"]),
            )
            if d4_human_audit["sha256"] != d4_risk_evidence["human_audit_sha256"]:
                raise RuntimeError("D4 confirmation is blocked: risk evidence and active human audit differ.")
        else:
            d4_human_audit = {"required": False, "status": "semantic_overrides_disabled"}
        d4_sealed_manifests = _require_d4_sealed_manifests(
            dict(phase.get("sealed_manifest_paths") or {}),
            expected_labels=set(phase.get("benchmark_slugs") or []),
            expectations=dict(phase.get("sealed_manifest_expectations") or {}),
        )
        provider_audit = {
            "required": True,
            "status": "passed",
            **_require_passing_provider_audit(
                experiment.provider_audit_path,
                expected_cache_policy="live_only",
                expected_provider=str(backbone.provider),
                expected_model_id=str(backbone.model_id),
            ),
        }
    resume_root = Path(resume_run_dir).resolve() if resume_run_dir is not None else None
    if resume_root is not None:
        if not is_d4_kernel_run:
            raise ValueError("Only frozen D4 runs support --resume-run-dir.")
        previous_manifest_path = resume_root / "manifest.json"
        if not previous_manifest_path.is_file():
            raise FileNotFoundError("D4 resume requires an existing manifest.json.")
        previous_manifest = json.loads(previous_manifest_path.read_text(encoding="utf-8"))
        if (
            previous_manifest.get("experiment_name") != experiment.name
            or previous_manifest.get("phase_name") != phase_name
            or previous_manifest.get("frozen_config_sha256") != config_sha
            or previous_manifest.get("cache_policy") != cache_policy
            or previous_manifest.get("schema_version") != D4_IR_SCHEMA
        ):
            raise ValueError("D4 resume manifest does not match the frozen experiment, phase, protocol, and cache policy.")
        if str(previous_manifest.get("run_status") or "") in {"completed", "completed_with_errors"}:
            raise ValueError("A terminal D4 run cannot be resumed.")
    provider = OpenAICompatibleProvider(backbone)
    active_router = RequestCacheRouter(cache_root)
    throttle = RequestThrottle.for_model(
        backbone,
        max_concurrent_requests=experiment.max_concurrent_requests,
        requests_per_minute=experiment.requests_per_minute_limit,
    )
    network_budget = NetworkAttemptBudget(protocol.max_network_attempts)
    run_id = resume_root.name if resume_root is not None else build_run_id(backbone.name)
    layout = prepare_registered_run_layout(
        "contrastive_active_testing",
        resume_root.parents[2] if resume_root is not None else run_root,
        experiment.name,
        phase_name,
        run_id,
    )
    if preflight_dependency is not None:
        archived_audit = layout.root / "diagnostics" / "preflight_human_audit.json"
        archived_audit.write_bytes(experiment.preflight_human_audit_path.read_bytes())
        preflight_dependency = {
            **preflight_dependency,
            "archived_human_audit_path": archived_audit.relative_to(layout.root).as_posix(),
        }
    benchmarks = load_phase_benchmarks(experiment, phase_name)
    selected_by_benchmark: dict[str, list[Any]] = {}
    dataset_errors: list[dict[str, Any]] = []
    for benchmark in benchmarks:
        try:
            selected_by_benchmark[benchmark.slug] = _select_phase_samples(benchmark, phase, phase_name)
        except Exception as exc:
            selected_by_benchmark[benchmark.slug] = []
            dataset_errors.append(
                {
                    "dataset": benchmark.slug,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            execution_warnings.append(f"{benchmark.slug}:dataset_skipped:{type(exc).__name__}:{exc}")
    if (is_d4_confirmation or is_d4_independent_protocol_validation) and (
        dataset_errors
        or set(selected_by_benchmark) != {str(item) for item in phase.get("benchmark_slugs") or []}
        or any(not rows for rows in selected_by_benchmark.values())
    ):
        raise RuntimeError(
            "D4 sealed-data run is blocked: every preregistered dataset must resolve to a non-empty sample set "
            f"without loader errors; errors={dataset_errors}."
        )
    sample_count = sum(len(samples) for samples in selected_by_benchmark.values())
    selected_sample_manifest = _selected_sample_manifest(selected_by_benchmark, phase_name=phase_name)
    d3_data_audit = _d3_data_audit(
        experiment,
        benchmarks=benchmarks,
        selected_by_benchmark=selected_by_benchmark,
        phase=phase,
        phase_name=phase_name,
    )
    d3_confirmation_role = str(d3_data_audit.get("primary_confirmation_role") or "")
    if d3_confirmation_role.startswith("independent_") and (
        int(d3_data_audit.get("selected_bbeh_inspected_overlap_count") or 0) > 0
        or int(d3_data_audit.get("selected_bbeh_text_hash_overlap_with_inspected_count") or 0) > 0
    ):
        execution_warnings.append("d3_primary_confirmation_overlaps_previously_inspected_pool")
    expected_selection_hashes = dict(phase.get("expected_selection_sha256") or {})
    for dataset, expected_hash in expected_selection_hashes.items():
        actual_hash = (selected_sample_manifest.get(str(dataset)) or {}).get("sha256")
        if actual_hash != str(expected_hash):
            execution_warnings.append(f"{dataset}:frozen_selection_hash_mismatch")
    if phase_name == "confirmation" and not expected_selection_hashes:
        execution_warnings.append("confirmation_selection_hash_not_preregistered_d1_evidence_only")
    kernel_freeze = None
    if phase_name == "confirmation" and protocol.protocol_version == "catch_kernel_v1":
        freeze_path = experiment.raw.get("kernel_freeze_path")
        if freeze_path:
            kernel_freeze = _validate_kernel_freeze(
                Path(str(freeze_path)),
                component_hashes=frozen_components,
                selection_manifest=selected_sample_manifest,
                expected_metadata=_kernel_freeze_metadata(experiment, protocol=protocol, phase=phase),
            )
            if not kernel_freeze["valid"]:
                execution_warnings.append(f"kernel_freeze_invalid:{kernel_freeze['reason']}")
        else:
            execution_warnings.append("kernel_freeze_missing_development_evidence_only")
    if is_d4_confirmation:
        mismatched = [
            dataset
            for dataset, expected_hash in expected_selection_hashes.items()
            if (selected_sample_manifest.get(str(dataset)) or {}).get("sha256") != str(expected_hash)
        ]
        if mismatched:
            raise RuntimeError(f"D4 confirmation is blocked: selection hash mismatch for {sorted(mismatched)}.")
        if not kernel_freeze or not kernel_freeze.get("valid"):
            reason = (kernel_freeze or {}).get("reason", "freeze_missing")
            raise RuntimeError(f"D4 confirmation is blocked: kernel freeze is invalid: {reason}.")
    if is_d4_independent_protocol_validation:
        mismatched = [
            dataset
            for dataset, expected_hash in expected_selection_hashes.items()
            if (selected_sample_manifest.get(str(dataset)) or {}).get("sha256") != str(expected_hash)
        ]
        if mismatched:
            raise RuntimeError(
                "D4 independent protocol validation is blocked: selection hash mismatch for "
                f"{sorted(mismatched)}."
            )
        if sample_count != 300 or any(
            len(selected_by_benchmark.get(dataset, [])) != 100
            for dataset in D4_INDEPENDENT_PROTOCOL_VALIDATION_BENCHMARKS
        ):
            raise RuntimeError(
                "D4 independent protocol validation requires exactly 100 sealed records from each "
                "of BBEH-extension, MuSR-X, and SuperGPQA Science."
            )
    d4_completion_ledger = (
        D4CompletionLedger(layout.root / "diagnostics" / "d4_completion_ledger.jsonl")
        if is_d4_kernel_run
        else None
    )
    endpoints: dict[str, CatchEndpoint] = {}
    stop_event = Event()
    for benchmark in benchmarks:
        active_cache = active_router.for_request_target(
            provider=backbone.provider,
            request_model=backbone.model_id,
            dataset=benchmark.slug,
        )
        endpoints[benchmark.slug] = CatchEndpoint(
            backbone=backbone,
            provider=provider,
            cache=active_cache,
            throttle=throttle,
            stop_event=stop_event,
            completion_ledger=d4_completion_ledger,
        )
    jobs: list[CatchSampleJob] = []
    sequence_index = 0
    for benchmark in benchmarks:
        split_name = str(phase["split_overrides"][benchmark.slug])
        for sample in selected_by_benchmark[benchmark.slug]:
            jobs.append(CatchSampleJob(sequence_index, sample, split_name, endpoints[benchmark.slug]))
            sequence_index += 1
    resume_turns = _read_jsonl(layout.agent_turns) if resume_root is not None else []
    resume_routers = _read_jsonl(layout.router_decisions) if resume_root is not None else []
    resume_predictions = _read_jsonl(layout.predictions) if resume_root is not None else []
    resume_prediction_counts: dict[tuple[str, str], int] = defaultdict(int)
    for row in resume_predictions:
        resume_prediction_counts[(str(row.get("dataset") or ""), str(row.get("sample_id") or ""))] += 1
    required_resume_predictions = 1 if is_d4_output_protocol_only else 6
    resumed_sample_keys = {
        (str(row.get("dataset") or ""), str(row.get("sample_id") or ""))
        for row in resume_routers
        if row.get("dataset")
        and row.get("sample_id")
        and (
            isinstance(row.get("sample_error"), dict)
            or resume_prediction_counts[(str(row.get("dataset") or ""), str(row.get("sample_id") or ""))]
            >= required_resume_predictions
        )
    }
    if resume_root is not None:
        jobs = [
            job
            for job in jobs
            if (str(job.sample.dataset), str(job.sample.sample_id)) not in resumed_sample_keys
        ]
    cert_screening_mode = bool(
        protocol.protocol_version in {"catch_cert_v1", "catch_cert_v2", "catch_kernel_v1"}
        and phase_name == "development"
        and int(phase.get("screening_sample_count") or 0) > 0
        and int(phase.get("disagreement_sample_count") or 0) > 0
    )
    all_screening_jobs = list(jobs)
    if cert_screening_mode:
        jobs = []
    run_direct_judge = bool(phase.get("run_direct_judge", phase_name != "confirmation"))
    if protocol.protocol_version in {"catch_v3", "catch_cert_v1", "catch_cert_v2", "catch_kernel_v1"}:
        is_d3_kernel = (
            protocol.protocol_version == "catch_kernel_v1"
            and str(experiment.raw.get("kernel_revision") or "") == "d3_source_blind_v1"
        )
        is_d4_kernel = (
            protocol.protocol_version == "catch_kernel_v1"
            and str(experiment.raw.get("kernel_revision") or "") == "d4_proof_carrying_v1"
        )
        is_d4_output_protocol_only = is_d4_kernel and str(
            phase.get("evaluation_role") or ""
        ).startswith(D4_OUTPUT_PROTOCOL_ONLY_ROLE_PREFIXES)
        calls_per_triggered = (
            5 if is_d4_output_protocol_only
            else 17 if is_d3_kernel and run_direct_judge
            else 11 if is_d3_kernel or is_d4_kernel
            else 17 if run_direct_judge
            else 11
        )
        predictions_per_sample = (
            1 if is_d4_output_protocol_only
            else 9 if is_d3_kernel and run_direct_judge
            else 7 if is_d3_kernel
            else 6 if is_d4_kernel
            else 5 if run_direct_judge
            else 3
        )
    else:
        calls_per_triggered = 18 if phase_name == "development" else 14 if run_direct_judge else 11
        predictions_per_sample = 9 if phase_name == "development" else 4 if run_direct_judge else 3
    selected_disagreement_upper_bound = (
        sum(
            min(
                int(phase.get("disagreement_sample_count") or 0),
                len(selected_by_benchmark.get(benchmark.slug, [])),
            )
            for benchmark in benchmarks
        )
        if cert_screening_mode
        else sample_count
    )
    preflight_call_upper_bound = (
        sample_count * protocol.stage_candidates + protocol.preflight_sample_count * (1 + protocol.witness_count)
        if phase_name == "development" and protocol.protocol_version in {"catch_v2", "catch_v3"}
        else 0
    )
    if run_mode == "structural_preflight":
        # Stage-A is loaded for the manifest split, then intervention calls are
        # limited to the frozen 20 disagreement samples.
        total_planned_calls = preflight_call_upper_bound
        total_planned_predictions = 0
        total_planned_samples = protocol.preflight_sample_count
    else:
        total_planned_calls = (
            sample_count * protocol.stage_candidates + selected_disagreement_upper_bound * calls_per_triggered
            if cert_screening_mode
            else sample_count * calls_per_triggered
        ) + (preflight_call_upper_bound if protocol.protocol_version == "catch_v2" else 0)
        total_planned_predictions = selected_disagreement_upper_bound * predictions_per_sample
        total_planned_samples = (
            sample_count + selected_disagreement_upper_bound if cert_screening_mode else sample_count
        )
    progress = RunProgressTracker(
        layout.progress,
        total_planned_calls=total_planned_calls,
        total_planned_predictions=total_planned_predictions,
        total_planned_samples=total_planned_samples,
        planned_calls_are_upper_bound=True,
        target_network_rpm=experiment.requests_per_minute_limit,
        rate_limit_snapshot_provider=lambda: {
            **throttle.snapshot(),
            "network_attempt_budget": network_budget.snapshot(),
        },
    )
    manifest = finalize_family_manifest(
        {
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "family_name": "contrastive_active_testing",
            "paper_method_name": (
                "Risk-Calibrated Proof-Carrying Candidate Completion"
                if protocol.protocol_version == "catch_kernel_v1"
                and str(experiment.raw.get("kernel_revision") or "") == "d4_proof_carrying_v1"
                else "CATCH-Kernel"
                if protocol.protocol_version == "catch_kernel_v1"
                else "CATCH-Cert v2"
                if protocol.protocol_version == "catch_cert_v2"
                else "CATCH-Cert"
                if protocol.protocol_version == "catch_cert_v1"
                else "CATCH-ICV"
                if protocol.protocol_version == "catch_v3"
                else "CATCH"
            ),
            "method_version": protocol.protocol_version,
            "protocol_version": protocol.protocol_version,
            "d4_mainline_protocol_version": (
                D4_MAINLINE_PROTOCOL_VERSION if is_d4_kernel_run else None
            ),
            "kernel_revision": (
                str(experiment.raw.get("kernel_revision") or "d1_pairwise_v1")
                if protocol.protocol_version == "catch_kernel_v1"
                else None
            ),
            "d3_capability_registry_version": (
                D3_CAPABILITY_REGISTRY_VERSION
                if protocol.protocol_version == "catch_kernel_v1"
                and str(experiment.raw.get("kernel_revision") or "") == "d3_source_blind_v1"
                else None
            ),
            "d3_capability_registry": (
                capability_registry()
                if protocol.protocol_version == "catch_kernel_v1"
                and str(experiment.raw.get("kernel_revision") or "") == "d3_source_blind_v1"
                else None
            ),
            "d4_capability_registry_version": (
                D4_CAPABILITY_REGISTRY_VERSION
                if protocol.protocol_version == "catch_kernel_v1"
                and str(experiment.raw.get("kernel_revision") or "") == "d4_proof_carrying_v1"
                else None
            ),
            "d4_capability_registry": (
                d4_capability_registry()
                if protocol.protocol_version == "catch_kernel_v1"
                and str(experiment.raw.get("kernel_revision") or "") == "d4_proof_carrying_v1"
                else None
            ),
            "experiment_name": experiment.name,
            "phase_name": phase_name,
            "run_mode": run_mode,
            "description": experiment.description,
            "resolved_model": asdict(backbone),
            "protocol": asdict(protocol),
            "prompt_version": (
                D4_PROMPT_VERSION
                if protocol.protocol_version == "catch_kernel_v1"
                and str(experiment.raw.get("kernel_revision") or "") == "d4_proof_carrying_v1"
                else D3_PROMPT_VERSION
                if protocol.protocol_version == "catch_kernel_v1"
                and str(experiment.raw.get("kernel_revision") or "") == "d3_source_blind_v1"
                else KERNEL_PROMPT_VERSION
                if protocol.protocol_version == "catch_kernel_v1"
                else CERT_V2_PROMPT_VERSION
                if protocol.protocol_version == "catch_cert_v2"
                else CERT_PROMPT_VERSION
                if protocol.protocol_version == "catch_cert_v1"
                else CATCH_PROMPT_VERSION
            ),
            "schema_version": (
                D4_IR_SCHEMA
                if protocol.protocol_version == "catch_kernel_v1"
                and str(experiment.raw.get("kernel_revision") or "") == "d4_proof_carrying_v1"
                else KERNEL_SCHEMA_VERSION
                if protocol.protocol_version == "catch_kernel_v1"
                else CERT_V2_SCHEMA_VERSION
                if protocol.protocol_version == "catch_cert_v2"
                else CERT_SCHEMA_VERSION
                if protocol.protocol_version == "catch_cert_v1"
                else CATCH_SCHEMA_VERSION
            ),
            "global_seed": experiment.global_seed,
            "cache_policy": cache_policy,
            "cache_key_policy": CACHE_KEY_POLICY_VERSION,
            "request_source": "global_validated_response_cache",
            "provider_audit": provider_audit,
            "preflight_dependency": preflight_dependency,
            "readiness_assessment": readiness_assessment,
            "evidence_interpretation": (
                readiness_assessment.get("recommended_interpretation") if readiness_assessment else "not_applicable"
            ),
            "execution_policy": "best_effort_non_blocking",
            "execution_warnings": execution_warnings,
            "frozen_config_sha256": config_sha,
            "frozen_component_sha256": frozen_components,
            "frozen_decoding": frozen_decoding,
            "kernel_d2_freeze": kernel_freeze,
            "phase_metadata": phase,
            "benchmarks": [asdict(item) for item in benchmarks],
            "dataset_errors": dataset_errors,
            "sample_count": sample_count,
            "source_split_sample_count": sample_count,
            "selected_sample_manifest": selected_sample_manifest,
            "d3_data_audit": d3_data_audit,
            "d4_risk": (
                dict(experiment.raw.get("d4_risk") or {})
                if str(experiment.raw.get("kernel_revision") or "") == "d4_proof_carrying_v1"
                else None
            ),
            "d4_output": (
                dict(experiment.raw.get("d4_output") or {})
                if str(experiment.raw.get("kernel_revision") or "") == "d4_proof_carrying_v1"
                else None
            ),
            "d4_source_compiler_smoke": (
                d4_source_compiler_smoke_snapshot(experiment) if is_d4_kernel_run else None
            ),
            "d4_human_audit": d4_human_audit if is_d4_confirmation else None,
            "d4_sealed_manifests": d4_sealed_manifests,
            "d4_output_protocol_validation": d4_protocol_validation if is_d4_confirmation else None,
            "d4_risk_evidence_validation": d4_risk_evidence if is_d4_confirmation else None,
            "planned_sample_count": total_planned_samples,
            "screening_sample_count_per_dataset": int(phase.get("screening_sample_count") or 0)
            if cert_screening_mode
            else None,
            "selected_disagreement_cap_per_dataset": int(phase.get("disagreement_sample_count") or 0)
            if cert_screening_mode
            else None,
            "selection_rule": "gold_blind_stage_a_disagreement_sha256" if cert_screening_mode else None,
            "method_order": [
                "sc_5",
                *(
                    ["fixed_sc_8", "catch_d3_exact_only", "ssv_raw", "catch_kernel_d4", "adaptive_sc_8"]
                    if protocol.protocol_version == "catch_kernel_v1"
                    and str(experiment.raw.get("kernel_revision") or "") == "d4_proof_carrying_v1"
                    else ["adaptive_sc_8"]
                ),
                *(["fixed_sc_8"] if protocol.protocol_version == "catch_kernel_v1" and str(experiment.raw.get("kernel_revision") or "") == "d3_source_blind_v1" else []),
                *(
                    ["solver_direct"]
                    if protocol.protocol_version == "catch_kernel_v1"
                    and str(experiment.raw.get("kernel_revision") or "") == "d3_source_blind_v1"
                    else []
                ),
                *(
                    [
                        "catch_d3_exact_no_completion",
                        "catch_d3_exact_completion",
                        "catch_d3_semantic_compiler",
                    ]
                    if protocol.protocol_version == "catch_kernel_v1"
                    and str(experiment.raw.get("kernel_revision") or "") == "d3_source_blind_v1"
                    else []
                ),
                *(
                    []
                    if protocol.protocol_version == "catch_kernel_v1"
                    and str(experiment.raw.get("kernel_revision") or "") == "d4_proof_carrying_v1"
                    else [
                        "catch_kernel"
                        if protocol.protocol_version == "catch_kernel_v1"
                        else "catch_cert_v2"
                        if protocol.protocol_version == "catch_cert_v2"
                        else "catch_cert"
                        if protocol.protocol_version == "catch_cert_v1"
                        else "catch"
                    ]
                ),
                *(["direct_judge_3", "pair_judge_3"] if run_direct_judge else []),
            ],
            "max_network_attempts": protocol.max_network_attempts,
            "network_attempt_limit_mode": "soft_warning",
            "calls_per_triggered_question_upper_bound": calls_per_triggered,
            "preflight_call_upper_bound": preflight_call_upper_bound,
            "run_status": "running",
            "resume_source_run_dir": resume_root.as_posix() if resume_root is not None else None,
            "resume_completed_sample_count": len(resumed_sample_keys),
            "d4_completion_ledger": (
                "diagnostics/d4_completion_ledger.jsonl" if d4_completion_ledger is not None else None
            ),
            "dgcr_predecessor_status": "retired_exact_span_reconstruction_channel_failed",
        },
        family_name="contrastive_active_testing",
    )
    layout.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    turns: list[dict[str, Any]] = list(resume_turns)
    routers: list[dict[str, Any]] = list(resume_routers)
    predictions: list[dict[str, Any]] = list(resume_predictions)
    sample_errors: list[dict[str, Any]] = [
        dict(row["sample_error"])
        for row in routers
        if isinstance(row.get("sample_error"), dict)
    ]
    screening_stage_rows_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    screening_selection: dict[str, Any] = {
        "enabled": cert_screening_mode,
        "screening_sample_count": sample_count if cert_screening_mode else 0,
        "selected_disagreement_count": 0,
        "selected_sample_ids": {},
        "selection_rule": "gold_blind_stage_a_disagreement_sha256",
    }
    try:
        preflight = None
        if run_mode == "structural_preflight":
            replay = _run_required_canonicalization_replay(
                run_root,
                experiment_name=experiment.name,
                model_name=backbone.name,
                samples=[job.sample for job in jobs],
                output_path=layout.root / "diagnostics" / "canonicalization_replay.json",
            )
            _update_manifest_fields(
                layout.manifest,
                canonicalization_replay={
                    "passed": replay["passed"],
                    "metrics": replay["metrics"],
                    "hashes": replay["hashes"],
                },
            )
            preflight = run_icv_structural_preflight(
                [PreflightJob(job.sequence_index, job.sample, job.split_name, job.endpoint) for job in jobs],
                run_id=run_id,
                experiment=experiment,
                protocol=protocol,
                network_budget=network_budget,
                progress=progress,
                turns_path=layout.preflight_turns,
                output_path=layout.preflight,
                config_sha=config_sha,
            )
            progress.record_completed_samples(len(preflight.get("selected_sample_ids") or []))
            _finalize_preflight_run(
                layout,
                manifest_path=layout.manifest,
                preflight=preflight,
                network_budget=network_budget,
                planned_sample_count=protocol.preflight_sample_count,
            )
            progress.mark_completed("structural_preflight_completed")
            return layout.root
        with (
            layout.agent_turns.open("w", encoding="utf-8") as turns_handle,
            layout.router_decisions.open("w", encoding="utf-8") as routers_handle,
            (layout.root / "diagnostics" / "certificate_screening.jsonl").open(
                "w", encoding="utf-8"
            ) as screening_handle,
        ):
            for row in turns:
                turns_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            for row in routers:
                routers_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if cert_screening_mode:
                for job, stage_turns, stage_or_router in _execute_jobs_bounded(
                    all_screening_jobs,
                    max_workers=experiment.max_concurrent_requests,
                    worker=lambda screening_job: run_stage_a_only_sample(
                        screening_job.sample,
                        run_id=run_id,
                        split_name=screening_job.split_name,
                        experiment=experiment,
                        protocol=protocol,
                        endpoint=screening_job.endpoint,
                        network_budget=network_budget,
                    ),
                    progress=progress,
                ):
                    for raw in stage_turns:
                        row = {**raw, "sample_sequence_index": job.sequence_index, "run_stage": "screening"}
                        turns.append(row)
                        turns_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        progress.record_call(row)
                    if stage_turns and hasattr(stage_or_router, "triggered"):
                        screening_stage_rows_by_key[(job.sample.dataset, job.sample.sample_id)] = list(stage_turns)
                        stage = stage_or_router
                        screening_handle.write(
                            json.dumps(
                                {
                                    "dataset": job.sample.dataset,
                                    "sample_id": job.sample.sample_id,
                                    "sample_sequence_index": job.sequence_index,
                                    "triggered": stage.triggered,
                                    "anchor_key": stage.anchor_key,
                                    "vote_counts": stage.vote_counts,
                                    "valid_count": stage.valid_count,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        progress.record_phase_sample("stage_a_ready")
                    if isinstance(stage_or_router, dict) and stage_or_router.get("sample_error"):
                        sample_errors.append(dict(stage_or_router["sample_error"]))
                    progress.record_completed_samples(1, method_name="certificate_screening")
                    turns_handle.flush()
                    screening_handle.flush()
                selected_jobs = _select_cert_disagreement_jobs(
                    all_screening_jobs,
                    {
                        key: (
                            rows,
                            build_stage_decision(rows, seed=experiment.global_seed, sample_id=key[1]),
                        )
                        for key, rows in screening_stage_rows_by_key.items()
                    },
                    cap_per_dataset=int(phase.get("disagreement_sample_count") or 0),
                    seed=experiment.global_seed,
                )
                jobs = selected_jobs
                screening_selection["selected_disagreement_count"] = len(selected_jobs)
                screening_selection["selected_sample_ids"] = {
                    dataset: [job.sample.sample_id for job in selected_jobs if job.sample.dataset == dataset]
                    for dataset in sorted({job.sample.dataset for job in selected_jobs})
                }
                progress.reconcile_dynamic_plan(
                    total_planned_samples=sample_count + len(selected_jobs),
                    total_planned_predictions=len(selected_jobs) * predictions_per_sample,
                )
            for job, sample_turns, router_row, sample_predictions in _execute_jobs_bounded(
                jobs,
                max_workers=experiment.max_concurrent_requests,
                worker=lambda job: run_catch_sample(
                    job.sample,
                    run_id=run_id,
                    split_name=job.split_name,
                    experiment=experiment,
                    protocol=protocol,
                    endpoint=job.endpoint,
                    network_budget=network_budget,
                    phase_name=phase_name,
                    frozen_decoding=frozen_decoding,
                    run_direct_judge=run_direct_judge,
                    precomputed_stage_rows=(
                        tuple(screening_stage_rows_by_key.get((job.sample.dataset, job.sample.sample_id), ()))
                        if cert_screening_mode
                        else None
                    ),
                ),
                progress=progress,
            ):
                already_screened = cert_screening_mode and bool(
                    screening_stage_rows_by_key.get((job.sample.dataset, job.sample.sample_id))
                )
                emitted_turns = (
                    [row for row in sample_turns if row.get("role") != "stage_a_solver"]
                    if already_screened
                    else sample_turns
                )
                for raw in emitted_turns:
                    row = {**raw, "sample_sequence_index": job.sequence_index, "run_stage": "main"}
                    turns.append(row)
                    turns_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    progress.record_call(row)
                router_row = {**router_row, "sample_sequence_index": job.sequence_index}
                routers.append(router_row)
                routers_handle.write(json.dumps(router_row, ensure_ascii=False) + "\n")
                indexed_predictions = [
                    {**row, "sample_sequence_index": job.sequence_index} for row in sample_predictions
                ]
                predictions.extend(indexed_predictions)
                if router_row.get("sample_error"):
                    sample_errors.append(dict(router_row["sample_error"]))
                turns_handle.flush()
                routers_handle.flush()
                progress.record_predictions(
                    len(indexed_predictions),
                    job.sample.dataset,
                    "catch_sample",
                    sample_completed=True,
                )

        turns.sort(key=_turn_sort_key)
        routers.sort(key=lambda row: int(row.get("sample_sequence_index") or 0))
        predictions.sort(key=_prediction_sort_key)
        development_selection = None
        if phase_name == "development" and protocol.protocol_version not in {
            "catch_v3",
            "catch_cert_v1",
            "catch_cert_v2",
            "catch_kernel_v1",
        }:
            try:
                predictions, development_selection = materialize_development_catch(predictions, routers)
            except (KeyError, TypeError, ValueError) as exc:
                execution_warnings.append(f"development_selection_unavailable:{type(exc).__name__}:{exc}")
                default_method = "catch_d2_m1"
                predictions = [
                    {**row, "method_name": "catch", "selected_grid_method_name": default_method}
                    if row.get("method_name") == default_method
                    else row
                    for row in predictions
                ]
                development_selection = {
                    "selection_rule": "best_effort_default_d2_m1",
                    "selected": {"method_name": default_method, "d_min": 2, "margin": 1},
                    "positive_constraints_satisfied": False,
                    "error": {"error_type": type(exc).__name__, "message": str(exc)},
                }
        with layout.predictions.open("w", encoding="utf-8") as predictions_handle:
            for row in predictions:
                predictions_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        metrics = build_metrics(predictions)
        if cert_screening_mode:
            metrics["screening"] = _build_cert_screening_metrics(
                selected_by_benchmark,
                screening_stage_rows_by_key,
                seed=experiment.global_seed,
            )
            metrics["screening_selection"] = screening_selection
        planned_for_diagnostics = (
            {
                dataset: len(
                    [sample_id for sample_id in screening_selection.get("selected_sample_ids", {}).get(dataset, [])]
                )
                for dataset in selected_by_benchmark
            }
            if cert_screening_mode
            else {dataset: len(samples) for dataset, samples in selected_by_benchmark.items()}
        )
        metrics.update(
            build_best_effort_diagnostics(
                predictions=predictions,
                turns=turns,
                routers=routers,
                planned_by_dataset=planned_for_diagnostics,
            )
        )
        required_methods = [str(item) for item in phase.get("required_comparison_methods") or []]
        comparison_method_audit = _build_comparison_method_audit(metrics, required_methods)
        for dataset, audit in comparison_method_audit.items():
            if audit["missing"]:
                execution_warnings.append(
                    f"{dataset}:required_comparison_methods_missing:{','.join(audit['missing'])}"
                )
        metrics["comparison_method_audit"] = comparison_method_audit
        request_failure_count = sum(bool(row.get("request_error")) for row in turns)
        parse_failure_count = sum(row.get("protocol_parse_status") == "failed" for row in turns)
        execution = {
            "policy": "best_effort_non_blocking",
            "planned_sample_count": total_planned_samples,
            "screening_sample_count": sample_count if cert_screening_mode else 0,
            "selected_disagreement_count": screening_selection.get("selected_disagreement_count", 0),
            "attempted_sample_count": len(routers) + (sample_count if cert_screening_mode else 0),
            "evaluable_sample_count": len({row.get("sample_id") for row in predictions}),
            "missing_sample_count": max(
                0,
                total_planned_samples
                - len({row.get("sample_id") for row in predictions})
                - (sample_count if cert_screening_mode else 0),
            ),
            "sample_error_count": len(sample_errors),
            "dataset_error_count": len(dataset_errors),
            "request_failure_count": request_failure_count,
            "parse_failure_count": parse_failure_count,
            "network_attempt_budget": network_budget.snapshot(),
            "warnings": execution_warnings,
        }
        metrics["execution"] = execution
        layout.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        layout.run_summary.write_text(
            json.dumps(
                {
                    "metrics": metrics,
                    "execution": execution,
                    "development_selection": development_selection,
                    "preflight": preflight,
                    "planned_sample_count": total_planned_samples,
                    "screening_selection": screening_selection,
                    "sample_errors": sample_errors,
                    "dataset_errors": dataset_errors,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        has_errors = bool(request_failure_count or parse_failure_count or sample_errors or dataset_errors)
        terminal_status = "completed_with_errors" if has_errors else "completed"
        termination_reason = f"{phase_name}_{terminal_status}"
        _update_manifest_status(layout.manifest, terminal_status, termination_reason=termination_reason)
        report_result = render_report_with_fallback(layout.root, render_report)
        report_failed = bool(report_result.get("error_type"))
        if report_failed and terminal_status == "completed":
            terminal_status = "completed_with_errors"
            termination_reason = f"{phase_name}_completed_with_errors"
            _update_manifest_status(layout.manifest, terminal_status, termination_reason=termination_reason)
        validation = write_nonblocking_validation(layout.root)
        if not validation.get("artifact_valid") and terminal_status == "completed":
            terminal_status = "completed_with_errors"
            termination_reason = f"{phase_name}_completed_with_errors"
            _update_manifest_status(layout.manifest, terminal_status, termination_reason=termination_reason)
            write_nonblocking_validation(layout.root)
        if terminal_status == "completed":
            progress.mark_completed(termination_reason)
        else:
            progress.mark_completed_with_errors(
                termination_reason,
                error_count=request_failure_count + len(sample_errors) + len(dataset_errors) + int(report_failed),
                warning_count=len(execution_warnings) + int(parse_failure_count),
            )
        return layout.root
    except BaseException as exc:
        termination_reason = "interrupted_by_user" if isinstance(exc, KeyboardInterrupt) else "fatal_startup_error"
        _write_partial_outputs(
            layout,
            turns=turns,
            routers=routers,
            predictions=predictions,
            termination_reason=termination_reason,
            error=exc,
        )
        _update_manifest_status(
            layout.manifest,
            "interrupted" if isinstance(exc, KeyboardInterrupt) else "fatal_startup_error",
            termination_reason=termination_reason,
        )
        with suppress(BaseException):
            render_report_with_fallback(layout.root, render_report)
            write_nonblocking_validation(layout.root)
        if isinstance(exc, KeyboardInterrupt):
            progress.mark_interrupted(str(exc) or "interrupted by user")
        else:
            progress.mark_fatal_startup_error(type(exc).__name__, str(exc))
        raise
    finally:
        progress.close()
        provider.close()
        active_router.close()


def _execute_jobs_bounded(
    jobs: list[CatchSampleJob],
    *,
    max_workers: int,
    worker,
    progress: RunProgressTracker,
):
    """Keep at most max_workers sample futures alive and yield completed sample blocks."""

    executor = ThreadPoolExecutor(max_workers=max(1, max_workers), thread_name_prefix="catch-sample")
    iterator = iter(jobs)
    pending: dict[Future, CatchSampleJob] = {}

    def submit_one() -> bool:
        try:
            job = next(iterator)
        except StopIteration:
            return False
        pending[executor.submit(worker, job)] = job
        return True

    try:
        for _ in range(min(max(1, max_workers), len(jobs))):
            submit_one()
        progress.update_scheduler_state(
            in_flight_samples=len(pending),
            queued_samples=max(0, len(jobs) - len(pending)),
        )
        completed = 0
        while pending:
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            completed_batch: list[tuple[CatchSampleJob, Any]] = []
            for future in sorted(done, key=lambda item: pending[item].sequence_index):
                job = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = _failed_sample_result(job, exc)
                completed_batch.append((job, result))

            for job, result in completed_batch:
                completed += 1
                submit_one()
                progress.update_scheduler_state(
                    in_flight_samples=len(pending),
                    queued_samples=max(0, len(jobs) - completed - len(pending)),
                )
                yield (job, *result)
    except BaseException:
        for future in pending:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        progress.update_scheduler_state(in_flight_samples=0, queued_samples=0)


def _failed_sample_result(
    job: CatchSampleJob,
    error: Exception,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Represent an unexpected sample exception without cancelling sibling work."""

    sample = job.sample
    dataset = str(getattr(sample, "dataset", "unknown"))
    sample_id = str(getattr(sample, "sample_id", job.sequence_index))
    metadata = getattr(sample, "metadata", {})
    failure = {
        "dataset": dataset,
        "sample_id": sample_id,
        "sample_sequence_index": job.sequence_index,
        "error_type": type(error).__name__,
        "message": str(error),
    }
    router = {
        "dataset": dataset,
        "sample_id": sample_id,
        "task": metadata.get("task") if isinstance(metadata, dict) else None,
        "split": job.split_name,
        "triggered": None,
        "protocol_version": None,
        "resolver": "sample_execution_error",
        "sample_error": failure,
    }
    return [], router, []


def _turn_sort_key(row: dict[str, Any]) -> tuple[int, str, int]:
    return (
        int(row.get("sample_sequence_index") or 0),
        str(row.get("method_name") or ""),
        int(row.get("agent_id") or 0),
    )


def _prediction_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    method_order = {
        "sc_5": "00",
        "adaptive_sc_8": "01",
        "catch": "02",
        "catch_cert": "02",
        "catch_cert_v2": "02",
        "catch_kernel": "02",
        "direct_judge_3": "03",
        "pair_judge_3": "04",
    }
    method = str(row.get("method_name") or "")
    return int(row.get("sample_sequence_index") or 0), method_order.get(method, f"10:{method}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_partial_outputs(
    layout,
    *,
    turns: list[dict[str, Any]],
    routers: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    termination_reason: str,
    error: BaseException,
) -> None:
    """Preserve the core result set for an interrupted or fatal run."""

    layout.agent_turns.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in turns),
        encoding="utf-8",
    )
    layout.router_decisions.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in routers),
        encoding="utf-8",
    )
    layout.predictions.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
        encoding="utf-8",
    )
    manifest_payload = json.loads(layout.manifest.read_text(encoding="utf-8"))
    planned = int(manifest_payload.get("planned_sample_count") or manifest_payload.get("sample_count") or 0)
    completed = len({str(row.get("sample_id")) for row in routers if row.get("sample_id")})
    with suppress(BaseException):
        metrics = build_metrics(predictions)
        metrics["execution"] = {
            "policy": "best_effort_non_blocking",
            "planned_sample_count": planned,
            "attempted_sample_count": completed,
            "evaluable_sample_count": len({str(row.get("sample_id")) for row in predictions if row.get("sample_id")}),
            "missing_sample_count": max(0, planned - completed),
            "termination_reason": termination_reason,
        }
        layout.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if not layout.metrics.exists():
        layout.metrics.write_text(json.dumps({"summary": [], "execution": {}}, indent=2), encoding="utf-8")
    summary = {
        "metrics": json.loads(layout.metrics.read_text(encoding="utf-8")),
        "execution": {
            "policy": "best_effort_non_blocking",
            "termination_reason": termination_reason,
            "planned_sample_count": planned,
            "attempted_sample_count": completed,
            "incomplete_sample_count": max(0, planned - completed),
            "error": {"error_type": type(error).__name__, "message": str(error)},
        },
        "planned_sample_count": planned,
        "sample_errors": [row["sample_error"] for row in routers if isinstance(row.get("sample_error"), dict)],
        "dataset_errors": [],
    }
    layout.run_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _update_manifest_status(path: Path, status: str, *, termination_reason: str | None = None) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["run_status"] = status
    if termination_reason is not None:
        payload["termination_reason"] = termination_reason
    payload["status_updated_at"] = datetime.now(UTC).isoformat()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _update_manifest_fields(path: Path, **fields: Any) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(fields)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_required_canonicalization_replay(
    run_root: Path,
    *,
    experiment_name: str,
    model_name: str,
    samples: list[Any],
    output_path: Path,
) -> dict[str, Any]:
    """Find the immutable v1 dev trace and prove v3 target headroom offline."""

    phase_root = run_root / experiment_name / "development"
    manifests = (
        sorted(
            phase_root.glob("*/manifest.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if phase_root.exists()
        else []
    )
    sources: dict[str, Path] = {}
    for path in manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("cache_namespace") in {"catch-dev-v1", "catch-dev-v2"}
            and payload.get("resolved_model", {}).get("name") == model_name
            and (
                (path.parent / "turns" / "agent_turns.jsonl").exists()
                or (path.parent / "turns" / "preflight_turns.jsonl").exists()
            )
        ):
            sources.setdefault(str(payload.get("cache_namespace")), path.parent)
    if "catch-dev-v1" not in sources:
        raise RuntimeError("CATCH-v3 is blocked: no immutable CATCH-v1 development trace was found for replay.")
    ordered_sources = [sources[key] for key in ("catch-dev-v1", "catch-dev-v2") if key in sources]
    replay = replay_canonicalization(ordered_sources, samples=samples, output_path=output_path)
    if not replay.get("passed"):
        raise RuntimeError(
            f"CATCH-v3 is blocked by canonicalization replay headroom: {replay.get('feasibility_conditions')}"
        )
    return replay


def finalize_partial_run_directory(
    run_dir: str | Path,
    *,
    termination_reason: str = "futility_gate_impossible",
) -> dict[str, Any]:
    """Recover a hard-stopped CATCH run into an auditable failed terminal state."""

    root = Path(run_dir)
    manifest_path = root / "manifest.json"
    progress_path = root / "progress.json"
    if not manifest_path.exists() or not progress_path.exists():
        raise FileNotFoundError("partial finalization requires manifest.json and progress.json")
    _update_manifest_status(manifest_path, "failed", termination_reason=termination_reason)
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    planned_sample_count = int(
        manifest_payload.get("planned_sample_count") or manifest_payload.get("sample_count") or 0
    )
    router_path = root / "turns" / "router_decisions.jsonl"
    completed_sample_ids: set[str] = set()
    if router_path.exists():
        for line in router_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                sample_id = json.loads(line).get("sample_id")
            except (AttributeError, json.JSONDecodeError):
                continue
            if sample_id:
                completed_sample_ids.add(str(sample_id))
    completed_sample_count = len(completed_sample_ids)
    preflight_path = root / "diagnostics" / "preflight.json"
    is_structural_preflight = manifest_payload.get("run_mode") == "structural_preflight"
    preflight_payload: dict[str, Any] = {}
    if is_structural_preflight and preflight_path.exists():
        try:
            preflight_payload = json.loads(preflight_path.read_text(encoding="utf-8"))
            completed_sample_count = len(preflight_payload.get("selected_sample_ids") or [])
        except json.JSONDecodeError:
            preflight_payload = {}
    incomplete_sample_count = max(0, planned_sample_count - completed_sample_count)
    progress_payload = json.loads(progress_path.read_text(encoding="utf-8"))
    progress_payload.update(
        {
            "status": "failed",
            "last_write_reason": "failed",
            "last_updated_at": datetime.now(UTC).isoformat(),
            "failure": {
                "error_type": "FutilityStop",
                "message": "run stopped after its frozen gate became unreachable",
                "termination_reason": termination_reason,
            },
            "termination_reason": termination_reason,
            "total_planned_samples": planned_sample_count,
            "completed_samples": completed_sample_count,
            "incomplete_samples": incomplete_sample_count,
        }
    )
    progress_path.write_text(json.dumps(progress_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    required_empty = (
        root / "turns" / "agent_turns.jsonl",
        root / "turns" / "router_decisions.jsonl",
        root / "turns" / "preflight_turns.jsonl",
        root / "views" / "predictions.jsonl",
    )
    for path in required_empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("", encoding="utf-8")
    metrics_path = root / "views" / "metrics.json"
    gate_path = root / "diagnostics" / "gate.json"
    summary_path = root / "views" / "run_summary.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    if not metrics_path.exists():
        metrics_path.write_text(json.dumps({"summary": []}, indent=2), encoding="utf-8")
    failure_gate = {
        "gate_name": ("catch_v3_structural_preflight" if is_structural_preflight else "catch_failed_partial"),
        "passed": False,
        "run_mode": manifest_payload.get("run_mode"),
        "termination_reason": termination_reason,
        "planned_sample_count": planned_sample_count,
        "completed_sample_count": completed_sample_count,
        "incomplete_sample_count": incomplete_sample_count,
        "performance_gate_applicable": not is_structural_preflight,
        "preflight_status": preflight_payload.get("status") if is_structural_preflight else None,
    }
    gate_path.write_text(json.dumps(failure_gate, ensure_ascii=False, indent=2), encoding="utf-8")
    if not preflight_path.exists():
        preflight_path.write_text(
            json.dumps({"status": "not_applicable_historical_v1", "passed": False}, indent=2),
            encoding="utf-8",
        )
    if not summary_path.exists():
        summary_path.write_text(
            json.dumps({"metrics": {"summary": []}, "gate": failure_gate}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    render_report(root)
    validation = write_nonblocking_validation(root)
    return {"run_dir": root.as_posix(), "termination_reason": termination_reason, "validation": validation}


def _finalize_preflight_run(
    layout,
    *,
    manifest_path: Path,
    preflight: dict[str, Any],
    network_budget: NetworkAttemptBudget,
    planned_sample_count: int,
) -> None:
    """Land a terminal structural-preflight run without a performance fiction."""

    layout.agent_turns.write_text("", encoding="utf-8")
    layout.router_decisions.write_text("", encoding="utf-8")
    layout.predictions.write_text("", encoding="utf-8")
    metrics = {"summary": [], "paired_statistics": {"tests": []}}
    completed = len(preflight.get("selected_sample_ids") or [])
    gate = {
        "gate_name": "catch_v3_structural_preflight",
        "passed": bool(preflight.get("passed")),
        "run_mode": "structural_preflight",
        "termination_reason": "structural_preflight_completed",
        "planned_sample_count": planned_sample_count,
        "completed_sample_count": completed,
        "incomplete_sample_count": max(0, planned_sample_count - completed),
        "actual_network_attempts": network_budget.actual,
        "performance_gate_applicable": False,
        "preflight_status": preflight.get("status"),
    }
    layout.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    layout.gate.write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    layout.run_summary.write_text(
        json.dumps({"metrics": metrics, "gate": gate, "preflight": preflight}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _update_manifest_status(manifest_path, "completed", termination_reason="structural_preflight_completed")
    render_report(layout.root)
    write_nonblocking_validation(layout.root)


def _require_passing_icv_preflight(
    run_root: Path,
    *,
    experiment_name: str,
    model_name: str,
    config_sha: str,
    human_audit_path: Path,
) -> dict[str, Any]:
    """Require the one-shot machine preflight and its blind 40-coordinate audit."""

    phase_root = run_root / experiment_name / "development"
    candidates = (
        sorted(phase_root.glob("*/diagnostics/preflight.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if phase_root.exists()
        else []
    )
    found_run_id: str | None = None
    found_run_dir: Path | None = None
    found_preflight_path: Path | None = None
    for preflight_path in candidates:
        run_dir = preflight_path.parents[1]
        manifest_path = run_dir / "manifest.json"
        validation_path = run_dir / "run_validation.json"
        if not manifest_path.exists() or not validation_path.exists():
            continue
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if (
            preflight.get("passed")
            and validation.get("passed")
            and manifest.get("run_mode") == "structural_preflight"
            and manifest.get("method_version") == "catch_v3"
            and manifest.get("resolved_model", {}).get("name") == model_name
            and manifest.get("frozen_config_sha256") == config_sha
        ):
            found_run_id = str(manifest.get("run_id") or run_dir.name)
            found_run_dir = run_dir
            found_preflight_path = preflight_path
            break
    if found_run_id is None or found_run_dir is None or found_preflight_path is None:
        raise RuntimeError(
            "CATCH-v3 development is blocked until the exact-config structural-preflight command passes."
        )
    sample_path = found_run_dir / "diagnostics" / "preflight_human_audit_sample.json"
    if not sample_path.exists():
        raise RuntimeError("CATCH-v3 development is blocked: the preflight coordinate audit sample is missing.")
    sample_payload = json.loads(sample_path.read_text(encoding="utf-8"))
    expected_hashes = {
        str(item.get("coordinate_sha256") or "")
        for item in sample_payload.get("items") or []
        if isinstance(item, dict) and item.get("coordinate_sha256")
    }
    if len(expected_hashes) != 40:
        raise RuntimeError(
            "CATCH-v3 development is blocked: the preflight audit sample does not contain 40 unique coordinates."
        )
    human_audit = _require_passing_preflight_human_audit(
        human_audit_path,
        expected_run_id=found_run_id,
        expected_config_sha=config_sha,
        expected_coordinate_hashes=expected_hashes,
    )
    return {
        "source_preflight_run_id": found_run_id,
        "preflight_path": found_preflight_path.as_posix(),
        "preflight_sha256": hashlib.sha256(found_preflight_path.read_bytes()).hexdigest(),
        "audit_sample_path": sample_path.as_posix(),
        "audit_sample_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
        "human_audit": human_audit,
    }


def _require_unused_v3_preflight_attempt(run_root: Path, *, experiment_name: str) -> None:
    """Make the registered v3 structural preflight genuinely one shot."""

    phase_root = run_root / experiment_name / "development"
    if not phase_root.exists():
        return
    prior: list[tuple[Path, str]] = []
    for manifest_path in phase_root.glob("*/manifest.json"):
        with suppress(OSError, json.JSONDecodeError):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("method_version") == "catch_v3" and manifest.get("run_mode") == "structural_preflight":
                prior.append((manifest_path.parent, str(manifest.get("run_status") or "unknown")))
    if prior:
        rendered = ", ".join(f"{path.name}:{status}" for path, status in sorted(prior))
        raise RuntimeError("CATCH-v3 structural preflight is one shot and a prior attempt already exists: " + rendered)


def _require_unused_v3_full_phase_attempt(
    run_root: Path,
    *,
    experiment_name: str,
    phase_name: str,
) -> None:
    """Prevent dev/heldout retries from becoming unregistered selection."""

    if phase_name not in {"development", "heldout"}:
        return
    phase_root = run_root / experiment_name / phase_name
    if not phase_root.exists():
        return
    prior: list[tuple[Path, str]] = []
    for manifest_path in phase_root.glob("*/manifest.json"):
        with suppress(OSError, json.JSONDecodeError):
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("method_version") == "catch_v3" and manifest.get("run_mode") == "full":
                prior.append((manifest_path.parent, str(manifest.get("run_status") or "unknown")))
    if prior:
        rendered = ", ".join(f"{path.name}:{status}" for path, status in sorted(prior))
        raise RuntimeError(f"CATCH-v3 {phase_name} is one shot and a prior full attempt already exists: {rendered}")


def _require_passing_preflight_human_audit(
    path: Path,
    *,
    expected_run_id: str,
    expected_config_sha: str,
    expected_coordinate_hashes: set[str],
) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"CATCH-v3 development is blocked: preflight human audit is missing at {path}.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    evaluation = evaluate_icv_human_audit(
        payload,
        expected_coordinate_hashes=expected_coordinate_hashes,
    )
    conditions = {
        "record_level_recomputation_passed": evaluation["passed"],
        "adjudication_complete": bool(payload.get("adjudication_complete")),
        "source_run": payload.get("source_preflight_run_id") == expected_run_id,
        "source_config": payload.get("source_config_sha256") == expected_config_sha,
        "seed_42": int(payload.get("seed") or 0) == 42,
    }
    if not all(conditions.values()):
        raise RuntimeError(f"CATCH-v3 preflight human validity audit failed {conditions}; recomputed={evaluation}.")
    return {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_preflight_run_id": expected_run_id,
        "source_config_sha256": expected_config_sha,
        "recomputed": evaluation,
    }


def _select_phase_samples(benchmark, phase: dict[str, Any], phase_name: str):
    selection_strategy = str(phase.get("selection_strategy") or "")
    if selection_strategy == "d4_sealed_manifest_only":
        if dict(phase.get("exclude_splits") or {}).get(benchmark.slug):
            raise ValueError("D4 sealed selection cannot apply a post-sealing exclusion split.")
        # The sealed manifest is the selection authority.  Requiring an
        # independently generated local split manifest here would introduce a
        # second, mutable source of truth before the sealed IDs are checked.
        selected = list(load_samples(benchmark))
    else:
        split_name = str(phase["split_overrides"][benchmark.slug])
        samples = select_samples(benchmark, split_name)
        excluded_names = dict(phase.get("exclude_splits") or {}).get(benchmark.slug, [])
        excluded: set[str] = set()
        for excluded_name in excluded_names:
            excluded.update(load_split_ids(benchmark.cache_namespace or benchmark.slug, str(excluded_name)))
        selected = [sample for sample in samples if sample.sample_id not in excluded]
    limit = dict(phase.get("sample_limits") or {}).get(benchmark.slug)
    if selection_strategy == "d4_sealed_manifest_only":
        manifest_paths = dict(phase.get("sealed_manifest_paths") or {})
        manifest_path = manifest_paths.get(benchmark.slug)
        if not manifest_path:
            raise ValueError(f"D4 sealed manifest path is missing for {benchmark.slug}.")
        payload = json.loads(Path(str(manifest_path)).read_text(encoding="utf-8"))
        expected_asset_hash = str(
            payload.get("source_asset_sha256") or payload.get("rendered_asset_sha256") or ""
        )
        source_path = resolve_dataset_source_path(benchmark.source_path)
        if (
            not expected_asset_hash
            or not source_path.is_file()
            or hashlib.sha256(source_path.read_bytes()).hexdigest() != expected_asset_hash
        ):
            raise ValueError(f"D4 sealed source asset hash mismatch for {benchmark.slug}.")
        split_name = str(phase.get("sealed_manifest_split") or "confirmation")
        rows = list((payload.get("splits") or {}).get(split_name) or [])
        rows_by_id = {
            str(row.get("record_id") or ""): row
            for row in rows
            if isinstance(row, dict) and str(row.get("record_id") or "")
        }
        if len(rows_by_id) != len(rows):
            raise ValueError(f"D4 sealed manifest has duplicate or empty IDs for {benchmark.slug}.")
        ids = set(rows_by_id)
        selected = [sample for sample in selected if str(sample.sample_id) in ids]
        if len(selected) != len(ids):
            raise ValueError(
                f"D4 sealed manifest IDs do not resolve exactly for {benchmark.slug}: "
                f"manifest={len(ids)} resolved={len(selected)}"
            )
        for sample in selected:
            row = rows_by_id[str(sample.sample_id)]
            if row.get("question_sha256") and text_sha256(sample.question) != str(row["question_sha256"]):
                raise ValueError(f"D4 sealed question hash mismatch for {benchmark.slug}:{sample.sample_id}.")
            if row.get("source_record_sha256") and str(
                sample.metadata.get("source_record_sha256") or ""
            ) != str(row["source_record_sha256"]):
                raise ValueError(f"D4 sealed record hash mismatch for {benchmark.slug}:{sample.sample_id}.")
            if row.get("latent_graph_sha256") and str(
                sample.metadata.get("latent_graph_sha256") or ""
            ) != str(row["latent_graph_sha256"]):
                raise ValueError(f"D4 sealed latent hash mismatch for {benchmark.slug}:{sample.sample_id}.")
            expected_stratum = str(row.get("stratum") or row.get("task") or "")
            actual_stratum = str(
                sample.metadata.get("task")
                or sample.metadata.get("field")
                or sample.metadata.get("domain")
                or ""
            )
            if expected_stratum and actual_stratum != expected_stratum:
                raise ValueError(f"D4 sealed stratum mismatch for {benchmark.slug}:{sample.sample_id}.")
        if limit is not None and len(selected) != max(0, int(limit)):
            raise ValueError(
                f"D4 sealed manifest count differs from the preregistered sample limit for {benchmark.slug}: "
                f"manifest={len(selected)} limit={int(limit)}"
            )
        return selected
    if selection_strategy == "d4_capability_stratified_hash":
        per_dataset = dict(phase.get("capability_sample_limits") or {}).get(benchmark.slug)
        if not isinstance(per_dataset, dict) or not per_dataset:
            raise ValueError(f"D4 capability calibration quotas are missing for {benchmark.slug}.")
        return _select_d4_capability_samples(
            selected,
            benchmark_slug=benchmark.slug,
            phase_name=phase_name,
            seed=int(phase.get("selection_seed", 42)),
            quotas={str(key): int(value) for key, value in per_dataset.items()},
            limit=max(0, int(limit)) if limit is not None else len(selected),
        )
    if selection_strategy == "kernel_confirmation_stratified_sha256":
        return _select_kernel_confirmation_strata(
            selected,
            benchmark_slug=benchmark.slug,
            phase_name=phase_name,
            seed=int(phase.get("selection_seed", 42)),
            limit=max(0, int(limit)) if limit is not None else len(selected),
        )
    if selection_strategy in {
        "d3_task_stratified_hash",
        "d3_confirmation_stratified_hash",
    }:
        return _select_d3_stratified_samples(
            selected,
            benchmark_slug=benchmark.slug,
            phase_name=phase_name,
            seed=int(phase.get("selection_seed", 42)),
            limit=max(0, int(limit)) if limit is not None else len(selected),
        )
    if bool(phase.get("hash_sample_selection", False)):
        selection_seed = int(phase.get("selection_seed", 42))
        selected = sorted(
            selected,
            key=lambda sample: hashlib.sha256(
                f"{selection_seed}\0{phase_name}\0{benchmark.slug}\0{sample.sample_id}\0catch-kernel-confirmation".encode()
            ).hexdigest(),
        )
    if limit is not None:
        selected = selected[: max(0, int(limit))]
    return selected


def _d3_data_audit(
    experiment,
    *,
    benchmarks: list[Any],
    selected_by_benchmark: dict[str, list[Any]],
    phase: dict[str, Any],
    phase_name: str,
) -> dict[str, Any]:
    """Record official-Mini overlap and the role of this phase before scoring."""

    if str(experiment.raw.get("kernel_revision") or "") != "d3_source_blind_v1":
        return {}
    audit_config = dict(experiment.raw.get("d3_data_audit") or {})
    official_split = str(audit_config.get("official_bbeh_mini_split") or "bbeh_mini460_seed42")
    bbeh = next((item for item in benchmarks if item.slug == "bbeh"), None)
    if bbeh is None:
        return {"status": "bbeh_not_in_phase", "phase_name": phase_name}
    try:
        official_ids = set(load_split_ids(bbeh.cache_namespace or bbeh.slug, official_split))
    except (FileNotFoundError, ValueError, OSError) as exc:
        return {
            "status": "official_mini_manifest_unavailable",
            "phase_name": phase_name,
            "official_split": official_split,
            "error": type(exc).__name__,
        }
    inspected_ids: set[str] = set()
    for split in audit_config.get("previously_inspected_splits") or []:
        with suppress(FileNotFoundError, ValueError, OSError):
            inspected_ids.update(load_split_ids(bbeh.cache_namespace or bbeh.slug, str(split)))
    selected_samples = list(selected_by_benchmark.get("bbeh", []))
    selected_ids = {str(item.sample_id) for item in selected_samples}
    sample_by_id = {str(item.sample_id): item for item in load_samples(bbeh)}

    def text_hashes(sample_ids: set[str]) -> set[str]:
        return {
            hashlib.sha256(str(sample_by_id[sample_id].question).encode("utf-8")).hexdigest()
            for sample_id in sample_ids
            if sample_id in sample_by_id
        }

    official_text_hashes = text_hashes(official_ids)
    inspected_text_hashes = text_hashes(inspected_ids)
    selected_text_hashes = {
        hashlib.sha256(str(item.question).encode("utf-8")).hexdigest() for item in selected_samples
    }
    return {
        "status": "audited",
        "phase_name": phase_name,
        "primary_confirmation_role": str(
            phase.get("evaluation_role") or audit_config.get("primary_confirmation_role") or "unknown"
        ),
        "official_mini_role": str(audit_config.get("official_bbeh_mini_role") or "unclassified"),
        "official_split": official_split,
        "official_mini_count": len(official_ids),
        "previously_inspected_count": len(inspected_ids),
        "official_mini_overlap_with_inspected_count": len(official_ids & inspected_ids),
        "official_mini_text_hash_overlap_with_inspected_count": len(
            official_text_hashes & inspected_text_hashes
        ),
        "official_mini_independent_eligible_count": len(official_ids - inspected_ids),
        "selected_bbeh_count": len(selected_ids),
        "selected_bbeh_inspected_overlap_count": len(selected_ids & inspected_ids),
        "selected_bbeh_text_hash_overlap_with_inspected_count": len(
            selected_text_hashes & inspected_text_hashes
        ),
        "selected_bbeh_official_mini_overlap_count": len(selected_ids & official_ids),
        "selected_bbeh_inspected_disjoint": not bool(selected_ids & inspected_ids),
        "selected_bbeh_text_hash_inspected_disjoint": not bool(
            selected_text_hashes & inspected_text_hashes
        ),
        "text_hash_algorithm": "sha256_utf8_question",
        "selection_strategy": str(phase.get("selection_strategy") or ""),
    }


def _select_kernel_confirmation_strata(
    samples: list[Any],
    *,
    benchmark_slug: str,
    phase_name: str,
    seed: int,
    limit: int,
) -> list[Any]:
    """Frozen gold-blind task/native-structure stratification for confirmation."""

    if benchmark_slug == "bbeh":
        return sorted(samples, key=lambda item: _selection_hash(seed, phase_name, benchmark_slug, item.sample_id))[
            :limit
        ]
    if benchmark_slug == "seqbench" and samples:
        ordered_depth = sorted(
            samples,
            key=lambda item: (int(item.metadata.get("logical_depth_L") or 0), str(item.sample_id)),
        )
        denominator = max(1, len(ordered_depth))
        depth_deciles = {item.sample_id: min(9, index * 10 // denominator) for index, item in enumerate(ordered_depth)}
        primary: dict[tuple[int, float], list[Any]] = {}
        for sample in samples:
            key = (
                int(sample.metadata.get("backtracking_count_B") or 0),
                float(sample.metadata.get("noise_ratio_N") or 0),
            )
            primary.setdefault(key, []).append(sample)
        primary_orders: dict[tuple[int, float], list[Any]] = {}
        for key, items in primary.items():
            deciles: dict[int, list[Any]] = {}
            for item in items:
                deciles.setdefault(depth_deciles[item.sample_id], []).append(item)
            for values in deciles.values():
                values.sort(key=lambda item: _selection_hash(seed, phase_name, benchmark_slug, item.sample_id))
            primary_orders[key] = _round_robin_groups(deciles, limit=len(items))
        return _round_robin_groups(primary_orders, limit=limit)
    strata: dict[tuple[Any, ...], list[Any]] = {}
    for sample in samples:
        stratum = (
            (str(sample.metadata.get("task") or "unknown"),)
            if benchmark_slug in {"musr", "musr_x"}
            else ("all",)
        )
        strata.setdefault(stratum, []).append(sample)
    for items in strata.values():
        items.sort(key=lambda item: _selection_hash(seed, phase_name, benchmark_slug, item.sample_id))
    return _round_robin_groups(strata, limit=limit)


def _select_d4_capability_samples(
    samples: list[Any],
    *,
    benchmark_slug: str,
    phase_name: str,
    seed: int,
    quotas: dict[str, int],
    limit: int,
) -> list[Any]:
    """Gold-blind development selection with frozen per-capability quotas."""

    if any(value < 0 for value in quotas.values()) or sum(quotas.values()) > limit:
        raise ValueError(f"Invalid D4 capability quotas for {benchmark_slug}.")
    groups: dict[str, list[Any]] = defaultdict(list)
    for sample in samples:
        groups[d4_route_for_sample(sample).capability_id].append(sample)
    selected: list[Any] = []
    for capability_id, requested in sorted(quotas.items()):
        available = sorted(
            groups.get(capability_id, []),
            key=lambda item: _selection_hash(seed, phase_name, benchmark_slug, item.sample_id),
        )
        if len(available) < requested:
            raise ValueError(
                f"Insufficient D4 development samples for {benchmark_slug}:{capability_id}; "
                f"requested={requested} available={len(available)}."
            )
        selected.extend(available[:requested])
    return sorted(
        selected,
        key=lambda item: _selection_hash(seed, phase_name, benchmark_slug, item.sample_id),
    )


def _select_d3_stratified_samples(
    samples: list[Any],
    *,
    benchmark_slug: str,
    phase_name: str,
    seed: int,
    limit: int,
) -> list[Any]:
    """Gold-blind nested task/domain stratification for D3 development/confirmation."""

    strata: dict[str, list[Any]] = defaultdict(list)
    for sample in samples:
        if benchmark_slug in {"bbeh", "bbeh_extension", "musr", "musr_x"}:
            key = str(sample.metadata.get("task") or "unknown")
        elif benchmark_slug == "gpqa_diamond":
            key = str(sample.metadata.get("high_level_domain") or "unknown").casefold()
        else:
            key = "all"
        strata[key].append(sample)
    for values in strata.values():
        values.sort(key=lambda item: _selection_hash(seed, phase_name, benchmark_slug, item.sample_id))
    return _round_robin_groups(strata, limit=limit)


def _round_robin_groups(groups: dict[Any, list[Any]], *, limit: int) -> list[Any]:
    selected: list[Any] = []
    positions = {key: 0 for key in groups}
    ordered_strata = sorted(groups, key=str)
    while len(selected) < limit:
        added = False
        for stratum in ordered_strata:
            position = positions[stratum]
            if position >= len(groups[stratum]):
                continue
            selected.append(groups[stratum][position])
            positions[stratum] += 1
            added = True
            if len(selected) >= limit:
                break
        if not added:
            break
    return selected


def _selection_hash(seed: int, phase_name: str, dataset: str, sample_id: str) -> str:
    return hashlib.sha256(
        f"{seed}\0{phase_name}\0{dataset}\0{sample_id}\0catch-kernel-confirmation".encode()
    ).hexdigest()


def _selected_sample_manifest(samples_by_dataset: dict[str, list[Any]], *, phase_name: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for dataset, samples in sorted(samples_by_dataset.items()):
        ids = [str(sample.sample_id) for sample in samples]
        raw = json.dumps(ids, ensure_ascii=False, separators=(",", ":"))
        strata: dict[str, int] = defaultdict(int)
        for sample in samples:
            if dataset in {"musr", "musr_x"}:
                key = str(sample.metadata.get("task") or "unknown")
            elif dataset == "seqbench":
                key = (
                    f"B{int(sample.metadata.get('backtracking_count_B') or 0)}_"
                    f"N{float(sample.metadata.get('noise_ratio_N') or 0):g}"
                )
            elif dataset == "gpqa_diamond":
                key = str(sample.metadata.get("high_level_domain") or "unknown")
            else:
                key = str(sample.metadata.get("task") or "all")
            strata[key] += 1
        payload[dataset] = {
            "phase_name": phase_name,
            "count": len(ids),
            "sample_ids": ids,
            "sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
            "stratum_counts": dict(sorted(strata.items())),
        }
    return payload


def write_kernel_d2_freeze(experiment_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Materialize the frozen Kernel components and confirmation IDs before confirmation.

    The historical CLI name is retained for compatibility; D3 writes a D3
    schema and carries its source-blind risk gates in the signed metadata.
    """

    experiment = load_experiment_config(experiment_path)
    protocol = load_protocol_config(experiment.protocol)
    if protocol.protocol_version != "catch_kernel_v1":
        raise ValueError("Kernel D2 freeze requires a catch_kernel_v1 experiment.")
    phase = phase_metadata(experiment, "confirmation")
    selected: dict[str, list[Any]] = {}
    for benchmark in load_phase_benchmarks(experiment, "confirmation"):
        selected[benchmark.slug] = _select_phase_samples(benchmark, phase, "confirmation")
    components = _frozen_component_hashes(experiment)
    unsigned = {
        "schema_version": (
            "catch_kernel_d4_freeze_v1"
            if str(experiment.raw.get("kernel_revision") or "") == "d4_proof_carrying_v1"
            else "catch_kernel_d3_freeze_v1"
            if str(experiment.raw.get("kernel_revision") or "") == "d3_source_blind_v1"
            else "catch_kernel_d2_freeze_v1"
        ),
        "component_sha256": components,
        "selected_sample_manifest": _selected_sample_manifest(selected, phase_name="confirmation"),
        **_kernel_freeze_metadata(experiment, protocol=protocol, phase=phase),
    }
    payload = {
        **unsigned,
        "sha256": hashlib.sha256(
            json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def write_kernel_d4_freeze(experiment_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Materialize a D4 freeze only after sealed confirmation data is ready."""

    experiment = load_experiment_config(experiment_path)
    if str(experiment.raw.get("kernel_revision") or "") != "d4_proof_carrying_v1":
        raise ValueError("D4 freeze requires kernel_revision=d4_proof_carrying_v1.")
    phase = phase_metadata(experiment, "confirmation")
    if not bool(phase.get("sealed_data_ready", False)):
        raise RuntimeError("D4 freeze is blocked until sealed_data_ready=true.")
    if not experiment.confirmatory:
        raise RuntimeError("D4 freeze requires confirmatory=true.")
    _require_passing_d4_protocol_validation(
        Path(str(experiment.raw.get("protocol_validation_assessment_path") or ""))
    )
    risk_evidence = _require_passing_d4_risk_evidence(experiment)
    d4_risk = dict(experiment.raw.get("d4_risk") or {})
    if bool(d4_risk.get("semantic_override_enabled", False)):
        human_audit = _require_passing_d4_human_audit(
            experiment.human_audit_path,
            required_kernels=set(risk_evidence["required_semantic_kernels"]),
        )
        if human_audit["sha256"] != risk_evidence["human_audit_sha256"]:
            raise RuntimeError("D4 freeze is blocked: risk evidence and active human audit differ.")
    _require_d4_sealed_manifests(
        dict(phase.get("sealed_manifest_paths") or {}),
        expected_labels=set(phase.get("benchmark_slugs") or []),
        expectations=dict(phase.get("sealed_manifest_expectations") or {}),
    )
    return write_kernel_d2_freeze(experiment_path, output_path)


def write_d4_independent_protocol_validation_selection(
    experiment_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Inspect sealed validation assets and emit the hashes needed to freeze the run.

    This command is deliberately offline.  It never constructs a provider and
    emits no question text, answer, latent graph, or selected record ID.
    """

    experiment = load_experiment_config(experiment_path)
    require_passing_d4_source_compiler_smoke(experiment)
    phase = phase_metadata(experiment, "development")
    protocol = load_protocol_config(experiment.protocol)
    role = str(phase.get("evaluation_role") or "")
    if (
        protocol.protocol_version != "catch_kernel_v1"
        or str(experiment.raw.get("kernel_revision") or "") != "d4_proof_carrying_v1"
        or role != D4_INDEPENDENT_PROTOCOL_VALIDATION_ROLE
    ):
        raise ValueError("The selection inspector requires the D4 tagged-v3 validation template.")
    conditions = _d4_independent_protocol_validation_config_conditions(
        experiment,
        phase_name="development",
        phase=phase,
    )
    structural_conditions = {
        key: value
        for key, value in conditions.items()
        if key not in {"sealed_data_ready", "selection_hashes_frozen"}
    }
    if not all(structural_conditions.values()):
        raise RuntimeError(
            "D4 protocol-validation selection inspection is blocked by its structural contract: "
            f"{structural_conditions}."
        )
    sealed = _require_d4_sealed_manifests(
        dict(phase.get("sealed_manifest_paths") or {}),
        expected_labels=D4_INDEPENDENT_PROTOCOL_VALIDATION_BENCHMARKS,
        expectations=dict(phase.get("sealed_manifest_expectations") or {}),
    )
    benchmarks = load_phase_benchmarks(experiment, "development")
    selected = {
        benchmark.slug: _select_phase_samples(benchmark, phase, "development")
        for benchmark in benchmarks
    }
    if set(selected) != D4_INDEPENDENT_PROTOCOL_VALIDATION_BENCHMARKS or any(
        len(selected.get(dataset, [])) != 100
        for dataset in D4_INDEPENDENT_PROTOCOL_VALIDATION_BENCHMARKS
    ):
        raise RuntimeError("D4 protocol-validation selection must resolve to exactly 100/100/100 records.")
    selection = _selected_sample_manifest(selected, phase_name="development")
    selection_hashes = {
        dataset: str(row["sha256"])
        for dataset, row in sorted(selection.items())
    }
    frozen_hashes = {
        str(key): str(value)
        for key, value in dict(phase.get("expected_selection_sha256") or {}).items()
    }
    if frozen_hashes and frozen_hashes != selection_hashes:
        raise RuntimeError(
            "D4 protocol-validation selection differs from already frozen expected hashes."
        )
    unsigned = {
        "schema": "catch_d4_independent_protocol_validation_selection_v1",
        "experiment_path": Path(experiment_path).resolve().as_posix(),
        "experiment_sha256": hashlib.sha256(Path(experiment_path).read_bytes()).hexdigest(),
        "evaluation_role": role,
        "validation_data_role": phase.get("validation_data_role"),
        "sealed_manifest_split": phase.get("sealed_manifest_split"),
        "selection_hashes": selection_hashes,
        "selected_counts": {
            dataset: len(rows) for dataset, rows in sorted(selected.items())
        },
        "selected_stratum_counts": {
            dataset: dict(selection[dataset].get("stratum_counts") or {})
            for dataset in sorted(selection)
        },
        "sealed_manifest_sha256": {
            dataset: str(row["sha256"])
            for dataset, row in sorted(sealed.items())
        },
        "contains_question_text": False,
        "contains_gold": False,
        "contains_latent_graph": False,
    }
    payload = {
        **unsigned,
        "sha256": hashlib.sha256(
            json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _validate_kernel_freeze(
    path: Path,
    *,
    component_hashes: dict[str, str],
    selection_manifest: dict[str, Any],
    expected_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not path.exists():
        return {"valid": False, "reason": "freeze_file_missing", "path": path.as_posix()}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"valid": False, "reason": f"freeze_parse_failed:{type(exc).__name__}", "path": path.as_posix()}
    unsigned = {key: value for key, value in payload.items() if key != "sha256"}
    actual_sha = hashlib.sha256(
        json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if payload.get("sha256") != actual_sha:
        return {"valid": False, "reason": "freeze_hash_invalid", "path": path.as_posix()}
    if payload.get("component_sha256") != component_hashes:
        return {"valid": False, "reason": "component_hash_mismatch", "path": path.as_posix()}
    for key, expected in dict(expected_metadata or {}).items():
        if payload.get(key) != expected:
            return {"valid": False, "reason": f"freeze_metadata_mismatch:{key}", "path": path.as_posix()}
    expected_selection = payload.get("selected_sample_manifest") or {}
    expected_hashes = {key: value.get("sha256") for key, value in expected_selection.items()}
    actual_hashes = {key: value.get("sha256") for key, value in selection_manifest.items()}
    if expected_hashes != actual_hashes:
        return {"valid": False, "reason": "selection_hash_mismatch", "path": path.as_posix()}
    return {
        "valid": True,
        "reason": "frozen_components_and_selection_match",
        "path": path.resolve().as_posix(),
        "sha256": actual_sha,
    }


def _kernel_freeze_metadata(experiment, *, protocol, phase: dict[str, Any]) -> dict[str, Any]:
    kernel_revision = str(experiment.raw.get("kernel_revision") or "d1_pairwise_v1")
    return {
        "protocol_version": protocol.protocol_version,
        "d4_mainline_protocol_version": (
            D4_MAINLINE_PROTOCOL_VERSION if kernel_revision == "d4_proof_carrying_v1" else None
        ),
        "prompt_version": (
            D4_PROMPT_VERSION
            if kernel_revision == "d4_proof_carrying_v1"
            else D3_PROMPT_VERSION
            if kernel_revision == "d3_source_blind_v1"
            else KERNEL_PROMPT_VERSION
        ),
        "schema_version_runtime": D4_IR_SCHEMA if kernel_revision == "d4_proof_carrying_v1" else KERNEL_SCHEMA_VERSION,
        "semantics_version": KERNEL_SEMANTICS_VERSION,
        "capability_version": KERNEL_CAPABILITY_VERSION,
        "decoder_version": (
            D4_DECODER_VERSION
            if kernel_revision == "d4_proof_carrying_v1"
            else KERNEL_D3_DECODER_VERSION
            if kernel_revision == "d3_source_blind_v1"
            else KERNEL_D2_DECODER_VERSION
            if kernel_revision == "d2_unary_exact_v1"
            else KERNEL_DECODER_VERSION
        ),
        "kernel_revision": kernel_revision,
        "primary_model_ref": experiment.primary_model_ref,
        "global_seed": experiment.global_seed,
        "cache_policy": "global_validated_response_v3",
        "required_comparison_methods": list(phase.get("required_comparison_methods") or []),
        "d3_capability_registry_version": (
            D3_CAPABILITY_REGISTRY_VERSION if kernel_revision == "d3_source_blind_v1" else None
        ),
        "d3_risk": dict(experiment.raw.get("d3_risk") or {}) if kernel_revision == "d3_source_blind_v1" else None,
        "d4_capability_registry_version": (
            D4_CAPABILITY_REGISTRY_VERSION if kernel_revision == "d4_proof_carrying_v1" else None
        ),
        "d4_risk": dict(experiment.raw.get("d4_risk") or {}) if kernel_revision == "d4_proof_carrying_v1" else None,
        "evaluation_role": str(phase.get("evaluation_role") or "unknown"),
    }


def _select_cert_disagreement_jobs(
    jobs: list[CatchSampleJob],
    stages: dict[tuple[str, str], tuple[list[dict[str, Any]], Any]],
    *,
    cap_per_dataset: int,
    seed: int,
) -> list[CatchSampleJob]:
    """Select a deterministic, gold-blind disagreement subset after screening."""

    selected: list[CatchSampleJob] = []
    by_dataset: dict[str, list[CatchSampleJob]] = {}
    for job in jobs:
        state = stages.get((str(job.sample.dataset), str(job.sample.sample_id)))
        if state is None or not bool(getattr(state[1], "triggered", False)):
            continue
        by_dataset.setdefault(str(job.sample.dataset), []).append(job)
    for dataset, candidates in sorted(by_dataset.items()):
        ordered = sorted(
            candidates,
            key=lambda job: hashlib.sha256(
                f"{seed}\0{dataset}\0{job.sample.sample_id}\0catch-cert-disagreement".encode()
            ).hexdigest(),
        )
        selected.extend(ordered[: max(0, int(cap_per_dataset))])
    return selected


def _build_cert_screening_metrics(
    samples_by_dataset: dict[str, list[Any]],
    stage_rows_by_key: dict[tuple[str, str], list[dict[str, Any]]],
    *,
    seed: int,
) -> dict[str, dict[str, Any]]:
    """Score the gold-free screening decisions only after collection."""

    metrics: dict[str, dict[str, Any]] = {}
    for dataset, samples in sorted(samples_by_dataset.items()):
        rows: list[dict[str, Any]] = []
        for sample in samples:
            stage_rows = stage_rows_by_key.get((sample.dataset, sample.sample_id))
            if not stage_rows:
                continue
            stage = build_stage_decision(stage_rows, seed=seed, sample_id=sample.sample_id)
            target_keys = {stage.anchor_key}
            target_keys.update(
                pair.challenger_key for pair in build_target_pairs(stage, seed=seed, sample_id=sample.sample_id)
            )
            rows.append(
                {
                    "sc5": score_prediction(
                        sample.dataset,
                        stage.anchor_answer,
                        sample.reference_answer,
                        sample=sample,
                    ),
                    "candidate_oracle": any(
                        score_prediction(sample.dataset, candidate.answer, sample.reference_answer, sample=sample)
                        == 1.0
                        for candidate in stage.candidates
                    ),
                    "target_oracle": any(
                        candidate.key in target_keys
                        and score_prediction(sample.dataset, candidate.answer, sample.reference_answer, sample=sample)
                        == 1.0
                        for candidate in stage.candidates
                    ),
                    "triggered": stage.triggered,
                    "valid": stage.valid_count > 0,
                }
            )
        metrics[dataset] = {
            "sample_count": len(rows),
            "sc5_micro_accuracy": sum(row["sc5"] for row in rows) / len(rows) if rows else 0.0,
            "candidate_oracle_micro": sum(row["candidate_oracle"] for row in rows) / len(rows) if rows else 0.0,
            "target_oracle_micro": sum(row["target_oracle"] for row in rows) / len(rows) if rows else 0.0,
            "disagreement_count": sum(bool(row["triggered"]) for row in rows),
            "invalid_stage_answer_count": sum(not bool(row["valid"]) for row in rows),
        }
    return metrics


def _build_comparison_method_audit(
    metrics: dict[str, Any],
    required_methods: list[str],
) -> dict[str, dict[str, Any]]:
    audit: dict[str, dict[str, Any]] = {}
    if not required_methods:
        return audit
    for dataset, payload in dict(metrics.get("datasets") or {}).items():
        available = set(dict(payload.get("methods") or {}))
        missing = [method for method in required_methods if method not in available]
        audit[str(dataset)] = {
            "required": list(required_methods),
            "available": sorted(available),
            "missing": missing,
            "complete": not missing,
        }
    return audit


def _frozen_config_sha(
    experiment,
    *,
    component_hashes: dict[str, str] | None = None,
) -> str:
    protocol_version = load_protocol_config(experiment.protocol).protocol_version
    is_cert_v1 = protocol_version == "catch_cert_v1"
    is_cert_v2 = protocol_version == "catch_cert_v2"
    is_kernel = protocol_version == "catch_kernel_v1"
    kernel_revision = str(experiment.raw.get("kernel_revision") or "d1_pairwise_v1")
    payload = {
        "experiment": experiment.raw,
        "protocol": Path(experiment.protocol).read_text(encoding="utf-8"),
        "prompt_version": (
            D4_PROMPT_VERSION
            if is_kernel and kernel_revision == "d4_proof_carrying_v1"
            else D3_PROMPT_VERSION
            if is_kernel and str(experiment.raw.get("kernel_revision") or "") == "d3_source_blind_v1"
            else KERNEL_PROMPT_VERSION
            if is_kernel
            else CERT_V2_PROMPT_VERSION
            if is_cert_v2
            else CERT_PROMPT_VERSION
            if is_cert_v1
            else CATCH_PROMPT_VERSION
        ),
        "schema_version": (
            D4_IR_SCHEMA
            if is_kernel and kernel_revision == "d4_proof_carrying_v1"
            else KERNEL_SCHEMA_VERSION
            if is_kernel
            else CERT_V2_SCHEMA_VERSION
            if is_cert_v2
            else CERT_SCHEMA_VERSION
            if is_cert_v1
            else CATCH_SCHEMA_VERSION
        ),
        "decoder_version": (
            D4_DECODER_VERSION
            if is_kernel and kernel_revision == "d4_proof_carrying_v1"
            else KERNEL_D3_DECODER_VERSION
            if is_kernel and kernel_revision == "d3_source_blind_v1"
            else KERNEL_D2_DECODER_VERSION
            if is_kernel and kernel_revision == "d2_unary_exact_v1"
            else KERNEL_DECODER_VERSION
            if is_kernel
            else "catch_answer_linked_obligation_decoder_v2"
            if is_cert_v2
            else "catch_certificate_decoder_v1"
            if is_cert_v1
            else "catch_icv_repetition_decoder_v3"
            if protocol_version == "catch_v3"
            else "catch_ecoc_decoder_v2"
        ),
        "component_sha256": component_hashes or _frozen_component_hashes(experiment),
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _frozen_component_hashes(experiment) -> dict[str, str]:
    """Commit prompts, validators, parsers, benchmark config, and split files."""

    repo_root = Path(__file__).resolve().parents[5]
    family_root = Path(__file__).resolve().parents[1]
    paths = {
        Path(experiment.protocol).resolve(),
        *(Path(path).resolve() for path in experiment.benchmark_configs),
        family_root / "algorithms.py",
        family_root / "certificates.py",
        family_root / "certificates_v2.py",
        family_root / "kernel.py",
        family_root / "kernel_d3.py",
        family_root / "kernel_d4.py",
        family_root / "d4_contract.py",
        family_root / "d4_compiler_smoke.py",
        family_root / "d4_audit.py",
        family_root / "d4_data.py",
        family_root / "d4_protocol_validation.py",
        family_root / "kernel_adapters.py",
        family_root / "kernel_mechanism.py",
        family_root / "kernel_prompts.py",
        family_root / "comparison_replay.py",
        family_root / "causal_ledger.py",
        family_root / "cert_prompts.py",
        family_root / "cert_prompts_v2.py",
        family_root / "icv.py",
        family_root / "prompts.py",
        family_root / "replay.py",
        family_root / "statistics.py",
        family_root / "artifact_replay.py",
        family_root / "config.py",
        Path(__file__).resolve(),
        Path(__file__).with_name("sample.py").resolve(),
        Path(__file__).with_name("d4_ledger.py").resolve(),
        Path(__file__).with_name("report.py").resolve(),
        Path(__file__).with_name("preflight.py").resolve(),
        Path(__file__).with_name("validate.py").resolve(),
        repo_root / "src" / "research_experiments" / "core" / "data" / "datasets.py",
        repo_root / "src" / "research_experiments" / "core" / "data" / "evaluation.py",
        repo_root / "src" / "research_experiments" / "core" / "controls" / "control_prompts.py",
        repo_root / "src" / "research_experiments" / "core" / "execution" / "cache.py",
        repo_root / "src" / "research_experiments" / "core" / "execution" / "provider_audit.py",
        repo_root / "src" / "research_experiments" / "core" / "execution" / "rate_limits.py",
        repo_root / "src" / "research_experiments" / "core" / "execution" / "runner_common.py",
        repo_root / "src" / "research_experiments" / "core" / "execution" / "runtime.py",
        repo_root / "src" / "research_experiments" / "core" / "execution" / "providers" / "client.py",
        repo_root / "src" / "research_experiments" / "family_runtime" / "free_text_protocol.py",
        repo_root / "src" / "research_experiments" / "family_runtime" / "output_protocols.py",
        repo_root / "src" / "research_experiments" / "reporting" / "paired_inference.py",
    }
    if experiment.study_type == "post_failure_cross_domain_boundary_audit":
        boundary_phase = dict((experiment.raw.get("phases") or {}).get("boundary_audit") or {})
        selected_manifest = boundary_phase.get("bbeh_selected_manifest")
        if selected_manifest:
            paths.add(Path(str(selected_manifest)).resolve())
        paths.add(family_root / "boundary.py")
        paths.add(Path(__file__).with_name("boundary_execute.py").resolve())
        paths.add(Path(__file__).with_name("boundary_report.py").resolve())
    if str(experiment.raw.get("kernel_revision") or "") == "d4_proof_carrying_v1":
        risk_path = dict(experiment.raw.get("d4_risk") or {}).get("evidence_path")
        if risk_path:
            paths.add(Path(str(risk_path)).resolve())
        protocol_validation_path = Path(
            str(experiment.raw.get("protocol_validation_assessment_path") or "")
        )
        if protocol_validation_path.is_file():
            paths.add(protocol_validation_path.resolve())
        if bool(dict(experiment.raw.get("d4_risk") or {}).get("semantic_override_enabled", False)):
            paths.add(experiment.human_audit_path.resolve())
        confirmation_phase = dict((experiment.raw.get("phases") or {}).get("confirmation") or {})
        for manifest_path in dict(confirmation_phase.get("sealed_manifest_paths") or {}).values():
            paths.add(Path(str(manifest_path)).resolve())
    for phase_name, phase in (experiment.raw.get("phases") or {}).items():
        if str(phase.get("selection_strategy") or "") == "d4_sealed_manifest_only":
            continue
        benchmarks = {item.slug: item for item in load_phase_benchmarks(experiment, str(phase_name))}
        for slug, split_name in dict(phase.get("split_overrides") or {}).items():
            benchmark = benchmarks[str(slug)]
            paths.add(
                resolve_split_manifest_path(
                    benchmark.cache_namespace or benchmark.slug,
                    str(split_name),
                ).resolve()
            )
            for excluded_name in dict(phase.get("exclude_splits") or {}).get(str(slug), []):
                paths.add(
                    resolve_split_manifest_path(
                        benchmark.cache_namespace or benchmark.slug,
                        str(excluded_name),
                    ).resolve()
                )
    missing = sorted(path.as_posix() for path in paths if not path.exists())
    if missing:
        raise FileNotFoundError(f"CATCH frozen component files are missing: {missing}")
    return {
        path.relative_to(repo_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths, key=lambda value: value.as_posix())
    }


def _build_frozen_decoding_candidate(*, run_id: str, config_sha: str, selection: dict[str, Any]) -> dict[str, Any]:
    winner = dict(selection["selected"])
    payload = {
        "freeze_kind": "catch_decoding_v2",
        "source_development_run_id": run_id,
        "source_config_sha256": config_sha,
        "d_min": int(winner["d_min"]),
        "margin": int(winner["margin"]),
        "selection_constraints_passed": bool(selection.get("positive_constraints_satisfied")),
        "prompt_version": CATCH_PROMPT_VERSION,
        "schema_version": CATCH_SCHEMA_VERSION,
        "decoder_version": "catch_ecoc_decoder_v2",
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return payload


def _build_frozen_protocol_candidate(
    *,
    run_id: str,
    config_sha: str,
    protocol_version: str = "catch_v3",
    kernel_revision: str = "d1_pairwise_v1",
) -> dict[str, Any]:
    """Create an immutable fixed-protocol candidate; no dev grid is selected."""

    is_cert_v1 = protocol_version == "catch_cert_v1"
    is_cert_v2 = protocol_version == "catch_cert_v2"
    is_kernel = protocol_version == "catch_kernel_v1"
    is_cert = is_cert_v1 or is_cert_v2 or is_kernel

    payload = {
        "freeze_kind": (
            "catch_kernel_d4_protocol_v1"
            if is_kernel and kernel_revision == "d4_proof_carrying_v1"
            else "catch_kernel_protocol_v1"
            if is_kernel
            else "catch_cert_protocol_v2"
            if is_cert_v2
            else "catch_cert_protocol_v1"
            if is_cert_v1
            else "catch_icv_protocol_v3"
        ),
        "source_development_run_id": run_id,
        "source_config_sha256": config_sha,
        "coordinates_per_pair": None if is_cert else 3,
        "panel_rule": (
            {
                "all_required_conditions": True,
                "derived_anchor_refutation_required": True,
                "dual_panel_agreement_required": True,
                "mandatory_obligation_coverage_required": is_cert_v2 or is_kernel,
                "answer_hash_link_required": is_cert_v2 or is_kernel,
                "verifier_jurisdiction_required": is_kernel,
                "cross_jurisdiction_fallback_allowed": False if is_kernel else None,
            }
            if is_cert
            else {"challenger_votes_at_least": 2, "strictly_more_than_anchor": True}
        ),
        "dual_panel_unique_challenger_required": True,
        "selection_constraints_passed": True,
        "prompt_version": (
            D4_PROMPT_VERSION
            if is_kernel and kernel_revision == "d4_proof_carrying_v1"
            else D3_PROMPT_VERSION
            if is_kernel and kernel_revision == "d3_source_blind_v1"
            else KERNEL_PROMPT_VERSION
            if is_kernel
            else CERT_V2_PROMPT_VERSION
            if is_cert_v2
            else CERT_PROMPT_VERSION
            if is_cert_v1
            else CATCH_PROMPT_VERSION
        ),
        "schema_version": (
            D4_IR_SCHEMA
            if is_kernel and kernel_revision == "d4_proof_carrying_v1"
            else KERNEL_SCHEMA_VERSION
            if is_kernel
            else CERT_V2_SCHEMA_VERSION
            if is_cert_v2
            else CERT_SCHEMA_VERSION
            if is_cert_v1
            else CATCH_SCHEMA_VERSION
        ),
        "decoder_version": (
            D4_DECODER_VERSION
            if is_kernel and kernel_revision == "d4_proof_carrying_v1"
            else KERNEL_D3_DECODER_VERSION
            if is_kernel and kernel_revision == "d3_source_blind_v1"
            else KERNEL_D2_DECODER_VERSION
            if is_kernel and kernel_revision == "d2_unary_exact_v1"
            else KERNEL_DECODER_VERSION
            if is_kernel
            else "catch_answer_linked_obligation_decoder_v2"
            if is_cert_v2
            else "catch_certificate_decoder_v1"
            if is_cert_v1
            else "catch_icv_repetition_decoder_v3"
        ),
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return payload


def _load_frozen_decoding(path: Path, *, config_sha: str) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"CATCH held-out/confirmation is blocked: frozen decoding file is missing at {path}.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_sha = str(payload.get("sha256") or "")
    unsigned = dict(payload)
    unsigned.pop("sha256", None)
    actual_sha = hashlib.sha256(json.dumps(unsigned, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    if expected_sha != actual_sha:
        raise RuntimeError("CATCH frozen decoding hash is invalid.")
    v2 = payload.get("freeze_kind") == "catch_decoding_v2" and payload.get("decoder_version") == "catch_ecoc_decoder_v2"
    v3 = (
        payload.get("freeze_kind") == "catch_icv_protocol_v3"
        and payload.get("decoder_version") == "catch_icv_repetition_decoder_v3"
    )
    if not (v2 or v3):
        raise RuntimeError("CATCH frozen decoding uses an unknown or retired protocol version.")
    if payload.get("source_config_sha256") != config_sha or not payload.get("selection_constraints_passed"):
        raise RuntimeError("CATCH frozen decoding does not match the active config or failed development constraints.")
    if v2 and (int(payload.get("d_min") or 0) not in {2, 3, 4} or int(payload.get("margin") or 0) not in {1, 2}):
        raise RuntimeError("CATCH frozen decoding contains an out-of-grid threshold.")
    if v3 and (
        int(payload.get("coordinates_per_pair") or 0) != 3
        or payload.get("panel_rule")
        != {
            "challenger_votes_at_least": 2,
            "strictly_more_than_anchor": True,
        }
        or not payload.get("dual_panel_unique_challenger_required")
    ):
        raise RuntimeError("CATCH-v3 frozen decoder does not match the preregistered repetition code.")
    return payload


def _load_cert_v2_readiness_assessment(path: Path, *, config_sha: str) -> dict[str, Any]:
    """读取非阻断式就绪度诊断；缺失或失败只改变证据解释。"""

    base = {
        "path": path.as_posix(),
        "enforcement": "advisory_only",
        "blocks_execution": False,
        "status": "missing",
        "all_recommended_conditions_met": False,
        "conditions": {},
        "unmet_conditions": [],
        "recommended_interpretation": "exploratory_diagnostic_evidence",
    }
    if not path.exists():
        return base
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {**base, "status": "invalid", "error": f"{type(exc).__name__}:{exc}"}
    expected_sha = str(payload.get("sha256") or "")
    unsigned = dict(payload)
    unsigned.pop("sha256", None)
    actual_sha = hashlib.sha256(json.dumps(unsigned, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    conditions = payload.get("conditions")
    schema_version = payload.get("schema_version")
    if (
        expected_sha != actual_sha
        or schema_version not in {"catch_cert_v2_readiness_v1", "catch_cert_v2_readiness_assessment_v2"}
        or payload.get("protocol_version") != "catch_cert_v2"
        or not isinstance(conditions, dict)
        or not conditions
    ):
        return {**base, "status": "invalid"}
    if payload.get("source_config_sha256") != config_sha:
        return {
            **base,
            "status": "config_mismatch",
            "source_config_sha256": payload.get("source_config_sha256"),
        }
    unmet = [name for name, met in conditions.items() if met is not True]
    all_met = not unmet
    return {
        **base,
        "status": "available",
        "schema_version": schema_version,
        "source_run_id": payload.get("source_run_id"),
        "source_config_sha256": payload.get("source_config_sha256"),
        "sha256": expected_sha,
        "conditions": conditions,
        "unmet_conditions": unmet,
        "all_recommended_conditions_met": all_met,
        "recommended_interpretation": ("confirmation_candidate" if all_met else "exploratory_diagnostic_evidence"),
        "evidence": payload.get("evidence") or {},
    }


def _load_cert_v2_readiness_gate(path: Path, *, config_sha: str) -> dict[str, Any]:
    """兼容旧调用名称；该函数已不再形成强制门槛。"""

    return _load_cert_v2_readiness_assessment(path, config_sha=config_sha)


def _require_passing_provider_audit(
    path: Path,
    *,
    expected_cache_policy: str,
    expected_provider: str,
    expected_model_id: str,
) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"CATCH gate is blocked: required provider audit is missing at {path}.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("audit_kind") != "mimo_live_provider_contract_v1" or payload.get("cache_mode") != "bypassed":
        raise RuntimeError("CATCH gate is blocked: provider audit has the wrong contract or cache mode.")
    if payload.get("provider") != expected_provider or payload.get("model_id") != expected_model_id:
        raise RuntimeError("CATCH gate is blocked: provider audit used a different provider or model.")
    evaluated = evaluate_mimo_provider_audit(
        payload.get("records") or [],
        expected_cache_policy=expected_cache_policy,
    )
    if not payload.get("passed") or not evaluated.get("passed"):
        raise RuntimeError(f"CATCH gate is blocked: provider audit failed: {evaluated.get('conditions', {})}")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "passed": True,
        "conditions": evaluated["conditions"],
    }


def _require_passing_d4_human_audit(
    path: Path,
    *,
    required_kernels: set[str] | None = None,
) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"D4 confirmation is blocked: blind IR audit is missing at {path}.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if required_kernels:
            payload = {**payload, "required_kernels": sorted(required_kernels)}
        assessment = evaluate_d4_human_audit(payload)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"D4 confirmation is blocked: blind IR audit is invalid: {exc}") from exc
    if not assessment.get("passed"):
        raise RuntimeError("D4 confirmation is blocked: blind IR audit did not pass all kernel conditions.")
    return {
        "path": path.resolve().as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "assessment": assessment,
    }


def _require_passing_d4_protocol_validation(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(
            f"D4 confirmation is blocked: independent tagged protocol validation is missing at {path}."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("D4 confirmation is blocked: tagged protocol validation is invalid.") from exc
    assessment_validation = validate_tagged_protocol_validation_assessment(
        payload,
        verify_source_files=True,
    )
    if not assessment_validation.get("passed"):
        raise RuntimeError(
            "D4 confirmation is blocked: independent tagged protocol validation failed: "
            f"{assessment_validation.get('conditions', {})}."
        )
    return {
        "path": path.resolve().as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "conditions": assessment_validation["conditions"],
        "assessment_validation": assessment_validation,
    }


def _require_passing_d4_risk_evidence(experiment) -> dict[str, Any]:
    d4_risk = dict(experiment.raw.get("d4_risk") or {})
    expected_design = {
        "development_minimum_overrides_per_capability": 50,
        "development_precision_lower_bound": 0.90,
        "development_gate_family_size": 9,
        "development_familywise_alpha": 0.05,
        "development_multiplicity_correction": "bonferroni_fixed_preregistered_capability_family",
        "confirmation_minimum_pooled_overrides": 59,
        "confirmation_precision_lower_bound": 0.95,
        "confirmation_harm_upper_bound": 0.05,
    }
    mismatched_design = {
        key: {"expected": expected, "actual": d4_risk.get(key)}
        for key, expected in expected_design.items()
        if d4_risk.get(key) != expected
    }
    if mismatched_design:
        raise RuntimeError(
            "D4 confirmation is blocked: configured risk design differs from the frozen implementation: "
            f"{mismatched_design}."
        )
    exact_enabled = bool(d4_risk.get("new_exact_override_enabled", False))
    semantic_enabled = bool(d4_risk.get("semantic_override_enabled", False))
    if not (exact_enabled or semantic_enabled):
        raise RuntimeError(
            "D4 confirmation is blocked: at least one independently gated new exact or semantic capability "
            "must be enabled; otherwise the run is only the frozen D3 method under a D4 label."
        )
    path = Path(str(d4_risk.get("evidence_path") or ""))
    if not path.is_file():
        raise RuntimeError(f"D4 confirmation is blocked: risk evidence is missing at {path}.")
    try:
        payload = load_risk_evidence(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"D4 confirmation is blocked: risk evidence is invalid: {exc}") from exc
    validation = validate_d4_gate_evidence(payload, verify_source_files=True)
    if not validation.get("passed"):
        raise RuntimeError(f"D4 confirmation is blocked: risk evidence failed validation: {validation}.")
    active = []
    active_exact: list[str] = []
    active_semantic: list[str] = []
    semantic_kernels: set[str] = set()
    for capability_id, row in sorted(dict(payload.get("capabilities") or {}).items()):
        route = str(row.get("route") or "SOFT_UNSUPPORTED")
        enabled = exact_enabled if route == "EXACT_EXECUTABLE" else semantic_enabled if route == "SEMANTIC_EXECUTABLE" else False
        if not enabled:
            continue
        snapshot = risk_gate_snapshot(
            str(capability_id),
            route=route,
            evidence=payload,
            phase="development",
        )
        if snapshot.route_activation_state == "ACTIVE":
            active.append(str(capability_id))
            if route == "EXACT_EXECUTABLE":
                active_exact.append(str(capability_id))
            elif route == "SEMANTIC_EXECUTABLE":
                active_semantic.append(str(capability_id))
                semantic_kernels.add(str(row.get("kernel_id") or ""))
    if not active:
        raise RuntimeError("D4 confirmation is blocked: no enabled new capability passed the development risk gate.")
    if exact_enabled and not active_exact:
        raise RuntimeError("D4 confirmation is blocked: new exact overrides are enabled but none passed the gate.")
    if semantic_enabled and not active_semantic:
        raise RuntimeError("D4 confirmation is blocked: semantic overrides are enabled but none passed the gate.")
    return {
        "path": path.resolve().as_posix(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "validation": validation,
        "active_capabilities": active,
        "required_semantic_kernels": sorted(item for item in semantic_kernels if item),
        "human_audit_sha256": str((payload.get("source") or {}).get("human_audit_sha256") or ""),
    }


def _require_d4_sealed_manifests(
    paths: dict[str, Any],
    *,
    expected_labels: set[str] | None = None,
    expectations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not paths:
        raise RuntimeError("D4 sealed-data run is blocked: sealed manifest paths are missing.")
    if expected_labels is not None and set(paths) != set(expected_labels):
        raise RuntimeError(
            "D4 sealed-data run is blocked: sealed manifest labels must exactly match the preregistered benchmarks; "
            f"expected={sorted(expected_labels)} actual={sorted(paths)}."
        )
    validated: dict[str, Any] = {}
    expected_specs = dict(expectations or {})
    if set(expected_specs) != set(paths):
        raise RuntimeError(
            "D4 sealed-data run is blocked: every sealed manifest needs a preregistered expectation; "
            f"paths={sorted(paths)} expectations={sorted(expected_specs)}."
        )
    for label, raw_path in sorted(paths.items()):
        path = Path(str(raw_path))
        if not path.exists():
            raise RuntimeError(f"D4 sealed-data run is blocked: sealed manifest is missing: {label}={path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"D4 sealed-data run is blocked: sealed manifest is invalid: {label}") from exc
        schema = str(payload.get("schema") or "")
        spec = dict(expected_specs.get(str(label)) or {})
        raw_counts = spec.get("counts")
        expected = (
            {str(key): int(value) for key, value in dict(raw_counts).items()}
            if isinstance(raw_counts, dict)
            else {}
        )
        expected_strata = tuple(str(item) for item in spec.get("strata") or [])
        if not expected or any(value <= 0 for value in expected.values()) or not expected_strata:
            raise RuntimeError(
                f"D4 sealed-data run is blocked: sealed expectation must preregister positive counts and strata: {label}."
            )
        if schema == "catch_d4_latent_first_sealed_manifest_v3":
            if str(label) not in {"musr", "musr_x"}:
                raise RuntimeError(
                    f"D4 sealed-data run is blocked: latent-first MuSR manifest cannot authorize {label}."
                )
            result = validate_sealed_manifest(
                payload,
                expected_counts=expected,
                expected_strata=expected_strata,
                manifest_path=path,
            )
            if not result.get("passed"):
                raise RuntimeError(f"D4 sealed-data run is blocked: latent manifest failed: {label}:{result}")
        elif schema == "catch_d4_text_sealed_manifest_v1":
            if str(label) not in {"bbeh_extension", "supergpqa_science"}:
                raise RuntimeError(
                    f"D4 sealed-data run is blocked: text-sealed manifest cannot authorize {label}."
                )
            result = validate_text_sealed_manifest(
                payload,
                expected_dataset_id=str(label),
                expected_counts=expected,
                expected_strata=expected_strata,
                manifest_path=path,
            )
            if not result.get("passed"):
                raise RuntimeError(f"D4 sealed-data run is blocked: text manifest failed: {label}:{result}")
        else:
            raise RuntimeError(
                f"D4 sealed-data run is blocked: sealed manifest schema is not implemented or allow-listed: "
                f"{label}:{schema or 'missing'}"
            )
        validated[str(label)] = {
            "path": path.resolve().as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "validation": result,
        }
    return validated


def _validate_d4_independent_protocol_validation_config(
    experiment,
    *,
    phase_name: str,
    phase: dict[str, Any],
) -> None:
    """Fail before data loading or API calls when the validation design is not frozen."""

    conditions = _d4_independent_protocol_validation_config_conditions(
        experiment,
        phase_name=phase_name,
        phase=phase,
    )
    if not all(conditions.values()):
        raise RuntimeError(
            "D4 independent protocol validation is blocked until its fresh sealed design is frozen: "
            f"{conditions}."
        )


def _d4_independent_protocol_validation_config_conditions(
    experiment,
    *,
    phase_name: str,
    phase: dict[str, Any],
) -> dict[str, bool]:
    d4_output = dict(experiment.raw.get("d4_output") or {})
    protocol = load_protocol_config(experiment.protocol)
    expected_hashes = dict(phase.get("expected_selection_sha256") or {})
    benchmark_slugs = {str(item) for item in phase.get("benchmark_slugs") or []}
    sample_limits = {
        str(key): int(value)
        for key, value in dict(phase.get("sample_limits") or {}).items()
    }
    smoke_snapshot = d4_source_compiler_smoke_snapshot(experiment)
    return {
        "development_phase": phase_name == "development",
        "nonconfirmatory_run": not experiment.confirmatory,
        "mainline_protocol_version": experiment.raw.get("d4_mainline_protocol_version")
        == D4_MAINLINE_PROTOCOL_VERSION,
        "source_compiler_smoke_passed": (
            smoke_snapshot["configured_status"] == D4_SOURCE_COMPILER_SMOKE_PASSED
            and smoke_snapshot["artifact_valid"]
            and smoke_snapshot["mainline_compatible"]
            and smoke_snapshot["artifact_passed"]
        ),
        "tagged_mainline_protocol": d4_output.get("stage_a_protocol") == "tagged_text",
        "global_validated_cache": experiment.raw.get("cache_policy") == "global_validated_response_v3",
        "high_completion_caps": protocol.solver_max_tokens == 65_536
        and protocol.role_max_tokens == 65_536
        and protocol.judge_max_tokens == 32_768,
        "frozen_targets": d4_output.get("parse_failure_target") == 0.01
        and d4_output.get("stage_a_quorum_minimum_valid_turns") == 3
        and d4_output.get("stage_a_quorum_failure_target") == 0.01,
        "sealed_data_ready": phase.get("sealed_data_ready") is True,
        "independent_data_role": phase.get("validation_data_role")
        == D4_INDEPENDENT_PROTOCOL_VALIDATION_DATA_ROLE,
        "sealed_selection": phase.get("selection_strategy") == "d4_sealed_manifest_only"
        and phase.get("sealed_manifest_split") == "protocol_validation",
        "three_frozen_benchmarks": benchmark_slugs
        == D4_INDEPENDENT_PROTOCOL_VALIDATION_BENCHMARKS,
        "exact_100_per_benchmark": sample_limits
        == {dataset: 100 for dataset in D4_INDEPENDENT_PROTOCOL_VALIDATION_BENCHMARKS},
        "selection_hashes_frozen": set(expected_hashes)
        == D4_INDEPENDENT_PROTOCOL_VALIDATION_BENCHMARKS
        and all(re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in expected_hashes.values()),
    }


def _require_passing_gate(
    run_root: Path,
    *,
    experiment_name: str,
    phase_name: str,
    model_name: str,
    config_sha: str,
    frozen_sha: str,
) -> None:
    phase_root = run_root / experiment_name / phase_name
    candidates = (
        sorted(phase_root.glob("*/diagnostics/gate.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        if phase_root.exists()
        else []
    )
    for gate_path in candidates:
        run_dir = gate_path.parents[1]
        manifest_path = run_dir / "manifest.json"
        validation_path = run_dir / "run_validation.json"
        if not manifest_path.exists() or not validation_path.exists():
            continue
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        candidate_path = run_dir / "diagnostics" / "frozen_decoding_candidate.json"
        candidate_sha = ""
        if candidate_path.exists():
            candidate_sha = str(json.loads(candidate_path.read_text(encoding="utf-8")).get("sha256") or "")
        manifest_frozen_sha = str((manifest.get("frozen_decoding") or {}).get("sha256") or "")
        if (
            gate.get("passed")
            and validation.get("passed")
            and manifest.get("resolved_model", {}).get("name") == model_name
            and manifest.get("frozen_config_sha256") == config_sha
            and (candidate_sha == frozen_sha if phase_name == "development" else manifest_frozen_sha == frozen_sha)
        ):
            return
    raise RuntimeError(f"CATCH {phase_name} gate is required for this exact model, config, and frozen decoder.")


def _require_passing_human_audit(path: Path) -> None:
    if not path.exists():
        raise RuntimeError(f"CATCH confirmation is blocked: human validity audit is missing at {path}.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("audit_version") == "catch_v3_icv_blind_coordinate_audit_v1":
        conditions = {
            "passed": bool(payload.get("passed")),
            "adjudication_complete": bool(payload.get("adjudication_complete")),
            "decidable_rate": float(payload.get("decidable_rate") or 0) >= 0.90,
            "exclusive_rate": float(payload.get("exclusive_rate") or 0) >= 0.90,
            "atomic_rate": float(payload.get("atomic_rate") or 0) >= 0.90,
            "leakage_zero": float(payload.get("answer_leakage_rate") or 0) == 0.0,
            "item_count": int(payload.get("item_count") or 0) >= 40,
            "annotator_count": int(payload.get("annotator_count") or 0) >= 2,
            "cohen_kappa": float(payload.get("cohen_kappa") or 0) >= 0.60,
        }
    else:
        conditions = {
            "passed": bool(payload.get("passed")),
            "adjudication_complete": bool(payload.get("adjudication_complete")),
            "decidable_rate": float(payload.get("decidable_rate") or 0) >= 0.90,
            "entailment_rate": float(payload.get("entailment_rate") or 0) >= 0.90,
            "leakage_rate": float(payload.get("answer_leakage_rate") or 1) <= 0.05,
            "sample_count": int(payload.get("sample_count") or 0) >= 100,
            "annotator_count": int(payload.get("annotator_count") or 0) >= 2,
        }
    if not all(conditions.values()):
        raise RuntimeError(f"CATCH confirmation is blocked: human validity audit failed {conditions}.")
