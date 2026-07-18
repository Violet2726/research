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
from research_experiments.core.execution.provider_audit import run_mimo_provider_audit
from research_experiments.core.execution.providers import OpenAICompatibleProvider
from research_experiments.core.execution.rate_limits import RequestThrottle
from research_experiments.core.execution.runtime import RunProgressTracker, build_run_id, finalize_run_outputs
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
from research_experiments.families.contrastive_active_testing.run.sample import (
    NetworkAttemptBudget,
    NetworkAttemptLimitExceeded,
    _answer_turn,
    run_catch_sample,
)
from research_experiments.families.contrastive_active_testing.run.validate import validate_run
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
    if backbone.provider != "xiaomimimo":
        raise ValueError("The boundary audit is frozen to the audited xiaomimimo provider.")
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
        _require_passing_provider_audit,
        _update_manifest_status,
    )

    phase = phase_metadata(experiment, "boundary_audit")
    benchmarks = load_phase_benchmarks(experiment, "boundary_audit")
    if tuple(benchmark.slug for benchmark in benchmarks) != BOUNDARY_DATASETS:
        raise ValueError(f"Boundary benchmark order must be exactly {BOUNDARY_DATASETS}.")
    source_assets = [verify_source_asset(benchmark) for benchmark in benchmarks]
    mechanism_compatibility = verify_frozen_v3_mechanism()
    if not experiment.provider_audit_path.exists():
        # The boundary audit is intentionally a one-shot command.  When the
        # server has no archived audit file, perform the required ten live,
        # cache-bypassed contract checks before creating a scientific run.
        audit_provider = OpenAICompatibleProvider(backbone)
        try:
            audit_payload = run_mimo_provider_audit(
                backbone=backbone,
                provider=audit_provider,
                cache_namespace=experiment.cache_namespaces["provider_audit"],
            )
        finally:
            audit_provider.close()
        experiment.provider_audit_path.parent.mkdir(parents=True, exist_ok=True)
        experiment.provider_audit_path.write_text(
            json.dumps(audit_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    provider_audit = _require_passing_provider_audit(
        experiment.provider_audit_path,
        expected_cache_namespace=experiment.cache_namespaces["provider_audit"],
        expected_provider=backbone.provider,
        expected_model_id=backbone.model_id,
    )
    frozen_components = _frozen_component_hashes(experiment)
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
        rate_limit_snapshot_provider=throttle.snapshot,
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
            "confirmation_budget_consumed": False,
            "heldout_authorized": False,
            "run_status": "running",
            "predecessor_status": "catch_v3_failed_structural_preflight",
        },
        family_name="contrastive_active_testing",
    )
    layout.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    layout.preflight_turns.write_text("", encoding="utf-8")
    layout.preflight.write_text(
        json.dumps({"status": "not_applicable_post_failure_audit", "passed": False}, indent=2),
        encoding="utf-8",
    )

    screening_rows: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []
    routers: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
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
                source_samples = load_samples(benchmark)
                if dataset == "bbeh":
                    selected_screening = select_samples(
                        benchmark, str(phase.get("bbeh_screening_split") or "dgcr_dev100_seed42")
                    )
                else:
                    selected_screening = select_screening_samples(
                        dataset, source_samples, count=screening_count, seed=experiment.global_seed
                    )
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
                )
                sequence_offset += len(states)
                if dataset == "bbeh":
                    selected_states = _fixed_bbeh_states(
                        states, Path(str(phase["bbeh_selected_manifest"]))
                    )
                    selection_rule = "fixed_failed_v3_preflight20_manifest"
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
                _write_json(layout.root / "diagnostics" / "dataset_checkpoints.json", checkpoints)
                _write_json(layout.root / "reproducibility_manifest.json", reproducibility)
                current_dataset = None

        turns.sort(key=lambda row: (int(row.get("sample_sequence_index") or 0), str(row.get("role") or ""), int(row.get("agent_id") or 0)))
        routers.sort(key=lambda row: int(row.get("sample_sequence_index") or 0))
        predictions.sort(key=lambda row: (int(row.get("sample_sequence_index") or 0), str(row.get("method_name") or "")))
        layout.predictions.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
            encoding="utf-8",
        )
        _write_jsonl(layout.root / "diagnostics" / "screening_samples.jsonl", screening_rows)
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
        gate = {
            "gate_name": "catch_v3_cross_domain_boundary_audit",
            "passed": request_failures == 0 and all(row.get("status") == "completed" for row in checkpoints.values()),
            "artifact_execution_valid": request_failures == 0,
            "scientific_gate_applicable": False,
            "scientific_gate_passed": False,
            "confirmatory": False,
            "heldout_authorized": False,
            "termination_reason": "boundary_audit_completed",
            "planned_sample_count": planned_screening + actual_selected_sample_count,
            "completed_sample_count": planned_screening + len(routers),
            "incomplete_sample_count": max(0, actual_selected_sample_count - len(routers)),
            "selected_disagreement_cap": planned_selected,
            "selected_capacity_unfilled_due_to_insufficient_disagreement": max(
                0, planned_selected - actual_selected_sample_count
            ),
            "actual_network_attempts": budget.actual,
            "network_attempt_cap": protocol.max_network_attempts,
            "request_failure_count": request_failures,
            "mechanism_assessment": bundle["mechanism_assessment"],
        }
        layout.gate.write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
        layout.run_summary.write_text(
            json.dumps(
                {
                    "metrics": bundle["metrics"],
                    "gate": gate,
                    "selector_funnel": bundle["selector_funnel"],
                    "witness_analysis": bundle["witness_analysis"],
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
        _update_manifest_status(layout.manifest, "completed", termination_reason="boundary_audit_completed")
        finalize_run_outputs(layout.root, validator=validate_run, validation_path=layout.validation)
        progress.mark_completed("boundary_audit_completed")
        return layout.root
    except BaseException as exc:
        termination = (
            "network_attempt_cap_reached"
            if isinstance(exc, NetworkAttemptLimitExceeded)
            else "interrupted_by_user"
            if isinstance(exc, KeyboardInterrupt)
            else "boundary_audit_execution_failure"
        )
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
                    "status": "not_started_due_to_prior_failure",
                    "termination_reason": termination,
                },
            )
        with suppress(BaseException):
            _write_jsonl(layout.root / "diagnostics" / "screening_samples.jsonl", screening_rows)
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
            layout.gate.write_text(
                json.dumps(
                    {
                        "gate_name": "catch_v3_cross_domain_boundary_audit",
                        "passed": False,
                        "scientific_gate_applicable": False,
                        "scientific_gate_passed": False,
                        "termination_reason": termination,
                        "planned_sample_count": planned_screening + planned_selected,
                        "completed_sample_count": progress.completed_samples,
                        "incomplete_sample_count": max(0, planned_screening + planned_selected - progress.completed_samples),
                        "actual_network_attempts": budget.actual,
                        "error": {"error_type": type(exc).__name__, "message": str(exc)},
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            layout.run_summary.write_text(
                json.dumps({"metrics": {}, "gate": json.loads(layout.gate.read_text(encoding="utf-8"))}, indent=2),
                encoding="utf-8",
            )
        _update_manifest_status(layout.manifest, "failed", termination_reason=termination)
        with suppress(BaseException):
            finalize_run_outputs(layout.root, validator=validate_run, validation_path=layout.validation)
        progress.mark_failed(type(exc).__name__, str(exc), termination_reason=termination)
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
    for _, state in _bounded(jobs, max_workers=experiment.max_concurrent_requests, worker=worker, progress=progress):
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
                result = future.result()
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
