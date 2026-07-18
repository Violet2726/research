"""One-shot four-dataset post-failure CATCH-v3 boundary audit.

一次性执行四数据集机制边界审计，并在任何终止路径上保留可重算产物。
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from typing import Any

from dotenv import load_dotenv

from research_experiments.core.data.datasets import load_samples, select_samples
from research_experiments.core.data.evaluation import score_prediction
from research_experiments.core.execution.cache import RequestCacheRouter
from research_experiments.core.execution.providers import OpenAICompatibleProvider
from research_experiments.core.execution.rate_limits import RequestThrottle
from research_experiments.core.execution.runtime import RunProgressTracker, build_run_id
from research_experiments.families.contrastive_active_testing.algorithms import build_stage_decision
from research_experiments.families.contrastive_active_testing.boundary import (
    BOUNDARY_DATASETS,
    boundary_sample_view,
    boundary_stratum,
    select_disagreement_states,
    select_screening_samples,
    stable_payload_hash,
    verify_frozen_v3_mechanism,
    verify_source_asset,
)
from research_experiments.families.contrastive_active_testing.cache_layers import ReadThroughRequestCache
from research_experiments.families.contrastive_active_testing.config import (
    load_phase_benchmarks,
    load_protocol_config,
    phase_metadata,
)
from research_experiments.families.contrastive_active_testing.icv import build_target_pairs
from research_experiments.families.contrastive_active_testing.prompts import (
    CATCH_PROMPT_VERSION,
    CATCH_SCHEMA_VERSION,
)
from research_experiments.families.contrastive_active_testing.run.boundary_report import (
    materialize_boundary_artifacts,
)
from research_experiments.families.contrastive_active_testing.run.lifecycle import (
    render_report_with_fallback,
    write_nonblocking_validation,
)
from research_experiments.families.contrastive_active_testing.run.report import render_report
from research_experiments.families.contrastive_active_testing.run.sample import (
    NetworkAttemptBudget,
    _answer_turn,
    run_catch_sample,
)
from research_experiments.family_runtime.layout import prepare_registered_run_layout
from research_experiments.family_runtime.manifest import finalize_family_manifest
from research_experiments.workspace.layout import default_cache_root, default_runs_root


@dataclass(frozen=True)
class ScreeningState:
    sequence_index: int
    sample: Any
    split_name: str
    endpoint: Any
    rows: tuple[dict[str, Any], ...]
    stage: Any


def run_boundary_audit(
    experiment,
    backbone,
    *,
    run_root: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> Path:
    """Run all four datasets once; never dispatch heldout or confirmation."""

    load_dotenv(".env.local", override=False)
    execution_warnings: list[str] = list(getattr(experiment, "config_warnings", ()))
    if backbone.provider != "xiaomimimo":
        execution_warnings.append(
            f"provider_differs_from_original_study:{backbone.provider}"
        )
    protocol = load_protocol_config(experiment.protocol)
    if protocol.protocol_version != "catch_v3" or protocol.budget_scope != "boundary_audit":
        raise ValueError("The boundary audit requires the frozen catch_v3 boundary protocol.")
    if experiment.confirmatory:
        raise ValueError("The boundary audit must remain explicitly non-confirmatory.")

    # Import here to avoid a module cycle with the standard runner's dispatch.
    from research_experiments.families.contrastive_active_testing.run.execute import (
        CatchEndpoint,
        _frozen_component_hashes,
        _frozen_config_sha,
        _update_manifest_status,
    )

    phase = phase_metadata(experiment, "boundary_audit")
    benchmarks = load_phase_benchmarks(experiment, "boundary_audit")
    observed_order = tuple(benchmark.slug for benchmark in benchmarks)
    if observed_order != BOUNDARY_DATASETS:
        execution_warnings.append(
            f"dataset_order_differs_from_original:{observed_order}"
        )
    source_assets: list[dict[str, Any]] = []
    try:
        mechanism_compatibility = verify_frozen_v3_mechanism()
    except (OSError, RuntimeError, ValueError) as exc:
        execution_warnings.append(
            f"mechanism_hash_mismatch_or_unavailable:{type(exc).__name__}:{exc}"
        )
        mechanism_compatibility = {
            "exact_component_hash_match": False,
            "warning": str(exc),
        }
    provider_audit = {"required": False, "status": "not_run"}
    try:
        frozen_components = _frozen_component_hashes(experiment)
    except (OSError, KeyError, ValueError) as exc:
        execution_warnings.append(f"component_hash_unavailable:{type(exc).__name__}:{exc}")
        frozen_components = {"best_effort_hash_status": stable_payload_hash(str(exc))}
    config_sha = _frozen_config_sha(experiment, component_hashes=frozen_components)

    run_root = Path(run_root or default_runs_root("contrastive_active_testing"))
    cache_root = Path(cache_root or default_cache_root())
    run_id = build_run_id(backbone.name)
    layout = prepare_registered_run_layout(
        "contrastive_active_testing", run_root, experiment.name, "boundary_audit", run_id
    )
    throttle = RequestThrottle.for_model(
        backbone,
        max_concurrent_requests=experiment.max_concurrent_requests,
        requests_per_minute=experiment.requests_per_minute_limit,
    )
    provider = OpenAICompatibleProvider(backbone)
    budget = NetworkAttemptBudget(protocol.max_network_attempts)
    active_routers: dict[str, RequestCacheRouter] = {}
    fallback_routers: dict[str, RequestCacheRouter] = {}
    endpoints: dict[str, CatchEndpoint] = {}
    for benchmark in benchmarks:
        namespace = experiment.cache_namespaces[benchmark.slug]
        active_router = RequestCacheRouter(cache_root, namespace=namespace)
        active_routers[benchmark.slug] = active_router
        active = active_router.for_request_target(
            provider=backbone.provider, request_model=backbone.model_id, dataset=benchmark.slug
        )
        predecessor_namespaces = tuple(
            item.strip()
            for item in str(experiment.baseline_cache_namespaces.get(benchmark.slug) or "").split(",")
            if item.strip()
        )
        fallbacks = []
        for predecessor in predecessor_namespaces:
            router = fallback_routers.setdefault(
                predecessor, RequestCacheRouter(cache_root, namespace=predecessor)
            )
            fallbacks.append(
                (
                    router.for_request_target(
                        provider=backbone.provider,
                        request_model=backbone.model_id,
                        dataset=benchmark.slug,
                    ),
                    predecessor,
                )
            )
        baseline_cache = ReadThroughRequestCache(
            active,
            primary_namespace=namespace,
            fallbacks=fallbacks,
        )
        intervention_fallbacks = [item for item in fallbacks if item[1] == "catch-dev-v3"]
        intervention_cache = ReadThroughRequestCache(
            active,
            primary_namespace=namespace,
            fallbacks=intervention_fallbacks,
        )
        endpoints[benchmark.slug] = CatchEndpoint(
            backbone=backbone,
            provider=provider,
            baseline_cache=baseline_cache,
            intervention_cache=intervention_cache,
            throttle=throttle,
            cache_namespace=namespace,
            baseline_cache_namespace=predecessor_namespaces,
            intervention_cache_namespaces=tuple(item[1] for item in intervention_fallbacks),
            stop_event=Event(),
        )

    screening_count = int(phase.get("screening_sample_count") or 100)
    disagreement_count = int(phase.get("disagreement_sample_count") or 20)
    planned_screening = screening_count * len(benchmarks)
    planned_selected = disagreement_count * len(benchmarks)
    planned_calls = planned_screening * protocol.stage_candidates + planned_selected * 12
    progress = RunProgressTracker(
        layout.progress,
        total_planned_calls=planned_calls,
        total_planned_predictions=planned_selected * 5,
        total_planned_samples=planned_screening + planned_selected,
        planned_calls_are_upper_bound=True,
        target_network_rpm=experiment.requests_per_minute_limit,
        rate_limit_snapshot_provider=lambda: {
            **throttle.snapshot(),
            "network_attempt_budget": budget.snapshot(),
        },
    )
    manifest = finalize_family_manifest(
        {
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "family_name": "contrastive_active_testing",
            "paper_method_name": "CATCH-ICV boundary audit",
            "method_version": "catch_v3",
            "protocol_version": "catch_v3",
            "experiment_name": experiment.name,
            "phase_name": "boundary_audit",
            "run_mode": "cross_domain_boundary_audit",
            "study_type": experiment.study_type,
            "confirmatory": False,
            "description": experiment.description,
            "resolved_model": asdict(backbone),
            "protocol": asdict(protocol),
            "prompt_version": CATCH_PROMPT_VERSION,
            "schema_version": CATCH_SCHEMA_VERSION,
            "global_seed": experiment.global_seed,
            "cache_namespace": "per_dataset_boundary_namespaces",
            "cache_namespaces": {
                key: value for key, value in experiment.cache_namespaces.items() if key != "provider_audit"
            },
            "baseline_read_cache_namespaces": experiment.baseline_cache_namespaces,
            "request_source": "role_aware_cross_domain_boundary_cache",
            "provider_audit": provider_audit,
            "execution_policy": "best_effort_non_blocking",
            "execution_warnings": execution_warnings,
            "frozen_config_sha256": config_sha,
            "frozen_component_sha256": frozen_components,
            "phase_metadata": phase,
            "benchmarks": [asdict(item) for item in benchmarks],
            "sample_count": planned_screening,
            "planned_sample_count": planned_screening + planned_selected,
            "screening_sample_count_per_dataset": screening_count,
            "selected_disagreement_cap_per_dataset": disagreement_count,
            "method_order": ["sc_5", "adaptive_sc_8", "catch", "direct_judge_3", "pair_judge_3"],
            "max_network_attempts": protocol.max_network_attempts,
            "network_attempt_limit_mode": "soft_warning",
            "confirmation_budget_consumed": False,
            "heldout_authorized": False,
            "run_status": "running",
            "predecessor_status": "catch_v3_failed_structural_preflight",
        },
        family_name="contrastive_active_testing",
    )
    layout.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    screening_rows: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []
    routers: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    sample_errors: list[dict[str, Any]] = []
    checkpoints: dict[str, Any] = {}
    actual_selected_sample_count = 0
    reproducibility = {
        "manifest_version": "catch_cross_domain_boundary_audit_v1",
        "study_type": experiment.study_type,
        "confirmatory": False,
        "seed": experiment.global_seed,
        "frozen_config_sha256": config_sha,
        "frozen_component_sha256": frozen_components,
        "prompt_version": CATCH_PROMPT_VERSION,
        "schema_version": CATCH_SCHEMA_VERSION,
        "frozen_bbeh_v3_mechanism_compatibility": mechanism_compatibility,
        "provider_audit": provider_audit,
        "cache_namespaces": {
            key: value for key, value in experiment.cache_namespaces.items() if key != "provider_audit"
        },
        "read_only_predecessor_cache_namespaces": experiment.baseline_cache_namespaces,
        "dataset_sources": source_assets,
        "execution_warnings": execution_warnings,
        "screening_manifests": {},
        "disagreement_manifests": {},
    }
    sequence_offset = 0
    current_dataset: str | None = None
    try:
        with (
            layout.agent_turns.open("w", encoding="utf-8") as turn_handle,
            layout.router_decisions.open("w", encoding="utf-8") as router_handle,
        ):
            for benchmark in benchmarks:
                dataset = benchmark.slug
                current_dataset = dataset
                checkpoints[dataset] = {
                    "status": "running",
                    "started_at": datetime.now(UTC).isoformat(),
                }
                try:
                    source_asset = verify_source_asset(benchmark)
                except (OSError, RuntimeError, ValueError) as exc:
                    warning = f"{dataset}:source_integrity_warning:{type(exc).__name__}:{exc}"
                    execution_warnings.append(warning)
                    source_asset = {"dataset": dataset, "verified": False, "warning": str(exc)}
                source_assets.append(source_asset)
                reproducibility["dataset_sources"] = source_assets
                try:
                    source_samples = load_samples(benchmark)
                    if dataset == "bbeh":
                        selected_screening = select_samples(
                            benchmark,
                            str(phase.get("bbeh_screening_split") or "dgcr_dev100_seed42"),
                        )
                    else:
                        selected_screening = select_screening_samples(
                            dataset,
                            source_samples,
                            count=screening_count,
                            seed=experiment.global_seed,
                        )
                except Exception as exc:
                    error = {
                        "dataset": dataset,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                    checkpoints[dataset].update(
                        {
                            "status": "skipped_with_error",
                            "completed_at": datetime.now(UTC).isoformat(),
                            "dataset_error": error,
                            "screening_sample_count": 0,
                            "selected_disagreement_count": 0,
                        }
                    )
                    execution_warnings.append(
                        f"{dataset}:dataset_skipped:{type(exc).__name__}:{exc}"
                    )
                    current_dataset = None
                    continue
                selected_screening = [boundary_sample_view(sample) for sample in selected_screening]
                screen_manifest = {
                    "dataset": dataset,
                    "seed": experiment.global_seed,
                    "uses_gold": False,
                    "sampling_rule": _screening_rule(dataset),
                    "sample_ids": [sample.sample_id for sample in selected_screening],
                }
                screen_manifest["sha256"] = stable_payload_hash(screen_manifest)
                reproducibility["screening_manifests"][dataset] = screen_manifest
                states = _run_screening_dataset(
                    selected_screening,
                    endpoint=endpoints[dataset],
                    split_name="boundary_screen100_seed42",
                    sequence_offset=sequence_offset,
                    run_id=run_id,
                    protocol=protocol,
                    experiment=experiment,
                    budget=budget,
                    progress=progress,
                    turn_handle=turn_handle,
                    turns=turns,
                    screening_rows=screening_rows,
                    sample_errors=sample_errors,
                )
                # Reserve indices for failed screening jobs too so later
                # datasets never collide in the single-writer artifact stream.
                sequence_offset += len(selected_screening)
                if dataset == "bbeh":
                    try:
                        selected_states = _fixed_bbeh_states(
                            states, Path(str(phase["bbeh_selected_manifest"]))
                        )
                        selection_rule = "fixed_failed_v3_preflight20_manifest"
                    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
                        execution_warnings.append(
                            f"bbeh:fixed_manifest_fallback:{type(exc).__name__}:{exc}"
                        )
                        selected_states = select_disagreement_states(
                            states, count=disagreement_count, seed=experiment.global_seed
                        )
                        selection_rule = "fallback_gold_free_stratum_round_robin_sha256"
                else:
                    selected_states = select_disagreement_states(
                        states, count=disagreement_count, seed=experiment.global_seed
                    )
                    selection_rule = "gold_free_stratum_round_robin_sha256"
                disagreement_manifest = {
                    "dataset": dataset,
                    "seed": experiment.global_seed,
                    "uses_gold": False,
                    "selection_rule": selection_rule,
                    "available_disagreement_count": sum(state.stage.triggered for state in states),
                    "sample_ids": [state.sample.sample_id for state in selected_states],
                    "strata": [boundary_stratum(state.sample) for state in selected_states],
                }
                disagreement_manifest["sha256"] = stable_payload_hash(disagreement_manifest)
                reproducibility["disagreement_manifests"][dataset] = disagreement_manifest
                actual_selected_sample_count += len(selected_states)
                _run_selected_dataset(
                    selected_states,
                    run_id=run_id,
                    experiment=experiment,
                    protocol=protocol,
                    budget=budget,
                    progress=progress,
                    turn_handle=turn_handle,
                    router_handle=router_handle,
                    turns=turns,
                    routers=routers,
                    predictions=predictions,
                    sample_errors=sample_errors,
                )
                checkpoints[dataset].update(
                    {
                        "status": "completed",
                        "completed_at": datetime.now(UTC).isoformat(),
                        "screening_sample_count": len(states),
                        "disagreement_count": sum(state.stage.triggered for state in states),
                        "selected_disagreement_count": len(selected_states),
                        "screening_manifest_sha256": screen_manifest["sha256"],
                        "disagreement_manifest_sha256": disagreement_manifest["sha256"],
                        "actual_network_attempts_cumulative": budget.actual,
                    }
                )
                current_dataset = None

        turns.sort(key=lambda row: (int(row.get("sample_sequence_index") or 0), str(row.get("role") or ""), int(row.get("agent_id") or 0)))
        routers.sort(key=lambda row: int(row.get("sample_sequence_index") or 0))
        predictions.sort(key=lambda row: (int(row.get("sample_sequence_index") or 0), str(row.get("method_name") or "")))
        layout.predictions.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
            encoding="utf-8",
        )
        reproducibility["network_attempt_budget"] = budget.snapshot()
        reproducibility["execution_warnings"] = execution_warnings
        bundle = materialize_boundary_artifacts(
            layout.root,
            screening_rows=screening_rows,
            turns=turns,
            routers=routers,
            predictions=predictions,
            source_manifest=reproducibility,
            checkpoints=checkpoints,
        )
        request_failures = sum(bool(row.get("request_error")) for row in turns)
        parse_failures = sum(row.get("protocol_parse_status") == "failed" for row in turns)
        dataset_errors = [
            dict(row.get("dataset_error") or {})
            for row in checkpoints.values()
            if row.get("dataset_error")
        ]
        execution = {
            "policy": "best_effort_non_blocking",
            "planned_sample_count": planned_screening + actual_selected_sample_count,
            "attempted_sample_count": len(screening_rows) + len(routers),
            "evaluable_selected_sample_count": len(
                {row.get("sample_id") for row in predictions if row.get("sample_id")}
            ),
            "incomplete_sample_count": max(
                0,
                planned_screening
                + actual_selected_sample_count
                - len(screening_rows)
                - len(routers),
            ),
            "selected_disagreement_cap": planned_selected,
            "selected_capacity_unfilled_due_to_insufficient_disagreement": max(
                0, planned_selected - actual_selected_sample_count
            ),
            "request_failure_count": request_failures,
            "parse_failure_count": parse_failures,
            "sample_error_count": len(sample_errors),
            "dataset_error_count": len(dataset_errors),
            "dataset_statuses": checkpoints,
            "network_attempt_budget": budget.snapshot(),
            "warnings": execution_warnings,
        }
        bundle["metrics"]["execution"] = execution
        layout.metrics.write_text(
            json.dumps(bundle["metrics"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
        layout.run_summary.write_text(
            json.dumps(
                {
                    "metrics": bundle["metrics"],
                    "execution": execution,
                    "planned_sample_count": planned_screening + actual_selected_sample_count,
                    "sample_errors": sample_errors,
                    "dataset_errors": dataset_errors,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        progress.reconcile_dynamic_plan(
            total_planned_samples=planned_screening + actual_selected_sample_count,
            total_planned_predictions=len(predictions),
        )
        has_errors = bool(request_failures or parse_failures or sample_errors or dataset_errors)
        terminal_status = "completed_with_errors" if has_errors else "completed"
        termination_reason = f"boundary_audit_{terminal_status}"
        _update_manifest_status(layout.manifest, terminal_status, termination_reason=termination_reason)
        report_result = render_report_with_fallback(layout.root, render_report)
        report_failed = bool(report_result.get("error_type"))
        if report_failed and terminal_status == "completed":
            terminal_status = "completed_with_errors"
            termination_reason = "boundary_audit_completed_with_errors"
            _update_manifest_status(layout.manifest, terminal_status, termination_reason=termination_reason)
        validation = write_nonblocking_validation(layout.root)
        if not validation.get("artifact_valid") and terminal_status == "completed":
            terminal_status = "completed_with_errors"
            termination_reason = "boundary_audit_completed_with_errors"
            _update_manifest_status(layout.manifest, terminal_status, termination_reason=termination_reason)
            write_nonblocking_validation(layout.root)
        if terminal_status == "completed":
            progress.mark_completed(termination_reason)
        else:
            progress.mark_completed_with_errors(
                termination_reason,
                error_count=request_failures + len(sample_errors) + len(dataset_errors) + int(report_failed),
                warning_count=len(execution_warnings) + parse_failures,
            )
        return layout.root
    except BaseException as exc:
        termination = "interrupted_by_user" if isinstance(exc, KeyboardInterrupt) else "fatal_startup_error"
        if current_dataset is not None:
            checkpoints.setdefault(current_dataset, {})
            checkpoints[current_dataset].update(
                {
                    "status": "failed",
                    "failed_at": datetime.now(UTC).isoformat(),
                    "termination_reason": termination,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "actual_network_attempts_cumulative": budget.actual,
                }
            )
        for dataset_name in BOUNDARY_DATASETS:
            checkpoints.setdefault(
                dataset_name,
                {
                    "status": "not_started_due_to_fatal_error",
                    "termination_reason": termination,
                },
            )
        with suppress(BaseException):
            partial = materialize_boundary_artifacts(
                layout.root,
                screening_rows=screening_rows,
                turns=turns,
                routers=routers,
                predictions=predictions,
                source_manifest=reproducibility,
                checkpoints=checkpoints,
            )
            del partial
            layout.predictions.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions), encoding="utf-8"
            )
            layout.run_summary.write_text(
                json.dumps(
                    {
                        "metrics": partial["metrics"],
                        "execution": {
                            "termination_reason": termination,
                            "error": {"error_type": type(exc).__name__, "message": str(exc)},
                        },
                        "planned_sample_count": planned_screening + planned_selected,
                        "sample_errors": sample_errors,
                        "dataset_errors": [
                            row.get("dataset_error") for row in checkpoints.values() if row.get("dataset_error")
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        _update_manifest_status(
            layout.manifest,
            "interrupted" if isinstance(exc, KeyboardInterrupt) else "fatal_startup_error",
            termination_reason=termination,
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
        for router in active_routers.values():
            router.close()
        for router in fallback_routers.values():
            router.close()


def _run_screening_dataset(
    samples,
    *,
    endpoint,
    split_name: str,
    sequence_offset: int,
    run_id: str,
    protocol,
    experiment,
    budget,
    progress,
    turn_handle,
    turns,
    screening_rows,
    sample_errors,
) -> list[ScreeningState]:
    jobs = list(enumerate(samples, start=sequence_offset))

    def worker(job):
        sequence_index, sample = job
        rows = tuple(
            _answer_turn(
                sample,
                run_id=run_id,
                split_name=split_name,
                endpoint=endpoint,
                network_budget=budget,
                method_name="catch_stage_a_shared",
                role="stage_a_solver",
                agent_id=index,
                seed=42_000 + index,
                max_tokens=protocol.solver_max_tokens,
            )
            for index in range(1, protocol.stage_candidates + 1)
        )
        stage = build_stage_decision(list(rows), seed=experiment.global_seed, sample_id=sample.sample_id)
        return ScreeningState(sequence_index, sample, split_name, endpoint, rows, stage)

    states = []
    for job, state in _bounded(
        jobs,
        max_workers=experiment.max_concurrent_requests,
        worker=worker,
        progress=progress,
    ):
        if isinstance(state, Exception):
            sequence_index, sample = job
            sample_errors.append(
                {
                    "dataset": sample.dataset,
                    "sample_id": sample.sample_id,
                    "sample_sequence_index": sequence_index,
                    "run_stage": "boundary_screening",
                    "error_type": type(state).__name__,
                    "message": str(state),
                }
            )
            progress.record_completed_samples(1, method_name="boundary_screening_error")
            continue
        states.append(state)
        for raw in state.rows:
            row = {**raw, "sample_sequence_index": state.sequence_index, "run_stage": "boundary_screening"}
            turns.append(row)
            turn_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            progress.record_call(row)
        turn_handle.flush()
        screening_rows.append(_screening_row(state, seed=experiment.global_seed))
        progress.record_phase_sample("stage_a_ready")
        progress.record_completed_samples(1, method_name="boundary_stage_a_screening")
    return sorted(states, key=lambda state: state.sequence_index)


def _run_selected_dataset(
    states: list[ScreeningState],
    *,
    run_id,
    experiment,
    protocol,
    budget,
    progress,
    turn_handle,
    router_handle,
    turns,
    routers,
    predictions,
    sample_errors,
) -> None:
    def worker(state):
        return run_catch_sample(
            state.sample,
            run_id=run_id,
            split_name="boundary_selected_disagreements_seed42",
            experiment=experiment,
            protocol=protocol,
            endpoint=state.endpoint,
            network_budget=budget,
            phase_name="boundary_audit",
            frozen_decoding=None,
            run_direct_judge=True,
            precomputed_stage_rows=state.rows,
        )

    for state, result in _bounded(states, max_workers=experiment.max_concurrent_requests, worker=worker, progress=progress):
        if isinstance(result, Exception):
            error = {
                "dataset": state.sample.dataset,
                "sample_id": state.sample.sample_id,
                "sample_sequence_index": state.sequence_index,
                "run_stage": "boundary_selected",
                "error_type": type(result).__name__,
                "message": str(result),
            }
            sample_errors.append(error)
            router = {
                "dataset": state.sample.dataset,
                "sample_id": state.sample.sample_id,
                "task": state.sample.metadata.get("task"),
                "sample_sequence_index": state.sequence_index,
                "run_stage": "boundary_selected",
                "sample_error": error,
            }
            routers.append(router)
            router_handle.write(json.dumps(router, ensure_ascii=False) + "\n")
            router_handle.flush()
            progress.record_predictions(
                0,
                state.sample.dataset,
                "boundary_selected_error",
                sample_completed=True,
            )
            continue
        sample_turns, router, sample_predictions = result
        # The exact shared five Stage-A turns are supplied to the selected
        # protocol in-memory and already exist in the screening block.  No
        # duplicate cache lookup or hidden logical call is performed here.
        sample_turns = [row for row in sample_turns if row.get("role") != "stage_a_solver"]
        for raw in sample_turns:
            row = {**raw, "sample_sequence_index": state.sequence_index, "run_stage": "boundary_selected"}
            turns.append(row)
            turn_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            progress.record_call(row)
        indexed_router = {**router, "sample_sequence_index": state.sequence_index, "run_stage": "boundary_selected"}
        routers.append(indexed_router)
        router_handle.write(json.dumps(indexed_router, ensure_ascii=False) + "\n")
        indexed_predictions = [
            {**row, "sample_sequence_index": state.sequence_index, "run_stage": "boundary_selected"}
            for row in sample_predictions
        ]
        predictions.extend(indexed_predictions)
        turn_handle.flush()
        router_handle.flush()
        progress.record_phase_sample("selector_completed")
        if sum(row.get("role") == "icv_witness" for row in sample_turns) == protocol.witness_count:
            progress.record_phase_sample("witness_completed")
        progress.record_predictions(
            len(indexed_predictions), state.sample.dataset, "boundary_matched_methods", sample_completed=True
        )


def _screening_row(state: ScreeningState, *, seed: int) -> dict[str, Any]:
    sample = state.sample
    stage = state.stage
    pairs = build_target_pairs(stage, seed=seed, sample_id=sample.sample_id)
    target_keys = {stage.anchor_key, *(pair.challenger_key for pair in pairs)}
    candidate_oracle = any(
        score_prediction(sample.dataset, candidate.answer, sample.reference_answer, sample=sample) == 1.0
        for candidate in stage.candidates
    )
    target_oracle = any(
        candidate.key in target_keys
        and score_prediction(sample.dataset, candidate.answer, sample.reference_answer, sample=sample) == 1.0
        for candidate in stage.candidates
    )
    return {
        "dataset": sample.dataset,
        "sample_id": sample.sample_id,
        "task": sample.metadata.get("task"),
        "stratum": boundary_stratum(sample),
        "sample_sequence_index": state.sequence_index,
        "sc5_prediction": stage.anchor_answer,
        "sc5_score": score_prediction(sample.dataset, stage.anchor_answer, sample.reference_answer, sample=sample) if stage.anchor_answer else 0.0,
        "triggered": stage.triggered,
        "valid_stage_answer_count": stage.valid_count,
        "invalid_stage_answer_count": len(state.rows) - stage.valid_count,
        "candidate_count": len(stage.candidates),
        "candidate_oracle_correct": candidate_oracle,
        "target_oracle_correct": target_oracle,
        "target_candidate_count": len(target_keys),
        "answer_class_vote_counts": stage.vote_counts,
    }


def _fixed_bbeh_states(states: list[ScreeningState], manifest_path: Path) -> list[ScreeningState]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    ids = [str(value) for value in payload.get("sample_ids") or []]
    if len(ids) != 20 or len(set(ids)) != 20 or payload.get("uses_gold") is not False:
        raise ValueError("The fixed BBEH preflight20 manifest is invalid.")
    by_id = {state.sample.sample_id: state for state in states}
    missing = [sample_id for sample_id in ids if sample_id not in by_id]
    if missing:
        raise ValueError(f"Fixed BBEH preflight samples are absent from screening100: {missing}")
    selected = [by_id[sample_id] for sample_id in ids]
    if not all(state.stage.triggered for state in selected):
        raise RuntimeError("The frozen BBEH preflight20 no longer consists entirely of Stage-A disagreements.")
    return selected


def _bounded(
    jobs: Iterable[Any], *, max_workers: int, worker: Callable[[Any], Any], progress
):
    jobs = list(jobs)
    executor = ThreadPoolExecutor(max_workers=max(1, max_workers), thread_name_prefix="catch-boundary")
    iterator = iter(jobs)
    pending: dict[Future, Any] = {}

    def submit_one() -> None:
        with suppress(StopIteration):
            job = next(iterator)
            pending[executor.submit(worker, job)] = job

    try:
        for _ in range(min(max(1, max_workers), len(jobs))):
            submit_one()
        completed = 0
        progress.update_scheduler_state(in_flight_samples=len(pending), queued_samples=max(0, len(jobs) - len(pending)))
        while pending:
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in sorted(done, key=lambda value: _job_index(pending[value])):
                job = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = exc
                completed += 1
                submit_one()
                progress.update_scheduler_state(
                    in_flight_samples=len(pending),
                    queued_samples=max(0, len(jobs) - completed - len(pending)),
                )
                yield job, result
    except BaseException:
        for future in pending:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        progress.update_scheduler_state(in_flight_samples=0, queued_samples=0)


def _job_index(job: Any) -> int:
    if isinstance(job, tuple):
        return int(job[0])
    return int(getattr(job, "sequence_index", 0))


def _screening_rule(dataset: str) -> str:
    return {
        "bbeh": "existing_seed42_task_stratified_dev100",
        "musr": "fixed_domain_quotas_34_33_33_then_sha256",
        "gpqa_diamond": "high_level_domain_largest_remainder_minimum_one_then_sha256",
        "seqbench": "B_by_N_largest_remainder_minimum_one_with_L_decile_round_robin",
    }[dataset]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
