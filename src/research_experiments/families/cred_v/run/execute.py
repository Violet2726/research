"""CRED-V 运行编排。"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from research_experiments.core.controls.no_comm_controls import run_unified_control_batch
from research_experiments.core.execution.artifacts import BufferedJsonlWriter
from research_experiments.core.execution.cache import RequestCacheRouter
from research_experiments.core.execution.providers import OpenAICompatibleProvider
from research_experiments.core.execution.rate_limits import RequestThrottle
from research_experiments.core.execution.runtime import RunProgressTracker, build_run_id, finalize_run_outputs
from research_experiments.core.structured_outputs import ARTIFACT_VERSION
from research_experiments.families.cred_v.config import (
    CredVExperimentConfig,
    load_control_catalog,
    load_protocol_config,
    validate_experiment_protocol_contract,
)
from research_experiments.families.cred_v.prompts import CRED_PROMPT_VERSION
from research_experiments.families.cred_v.run.report import render_report, summarize_run
from research_experiments.families.cred_v.run.sample import (
    CredVerifierRuntime,
    _execute_turn,
    append_outputs,
    build_control_prediction_row,
    build_debate_diagnostics,
    build_error_bucket_audit,
    build_metrics,
    build_router_eval,
    estimate_work,
    load_selected_samples,
    resolve_split_name,
    run_cred_batch,
)
from research_experiments.families.cred_v.run.validate import validate_run
from research_experiments.family_runtime.config_helpers import load_benchmarks, phase_metadata, resolve_model
from research_experiments.family_runtime.layout import prepare_registered_run_layout
from research_experiments.family_runtime.manifest import finalize_family_manifest
from research_experiments.family_runtime.output_protocols import build_shared_output_protocol_diagnostics
from research_experiments.workspace.layout import default_cache_root, default_runs_root


@dataclass(frozen=True)
class _DatasetRunResult:
    benchmark_index: int
    dataset_slug: str
    turn_rows: list[dict[str, Any]]
    debate_rows: list[dict[str, Any]]
    router_rows: list[dict[str, Any]]
    prediction_rows: list[dict[str, Any]]


def run_experiment(
    experiment: CredVExperimentConfig,
    phase_name: str,
    backbone,
    run_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    *,
    family_name: str = "cred_v",
    display_name: str = "CRED-V",
) -> Path:
    load_dotenv(".env.local", override=False)
    run_root = run_root or default_runs_root(family_name)
    cache_root = cache_root or default_cache_root()
    benchmarks = load_benchmarks(experiment)
    protocol = load_protocol_config(experiment.protocol)
    validate_experiment_protocol_contract(experiment, protocol)
    controls = load_control_catalog(experiment.control_catalog)
    _validate_control_methods(experiment, controls)
    provider = OpenAICompatibleProvider(backbone)
    verifier_backbones = [(model_ref, resolve_model(model_ref)) for model_ref in experiment.verifier_model_refs]
    verifier_providers = [
        (
            model_ref,
            verifier_backbone,
            OpenAICompatibleProvider(verifier_backbone),
            RequestThrottle.for_model(
                verifier_backbone,
                max_concurrent_requests=experiment.max_concurrent_requests,
                requests_per_minute=experiment.requests_per_minute_limit,
            ),
        )
        for model_ref, verifier_backbone in verifier_backbones
    ]
    cache_router = RequestCacheRouter(cache_root)
    throttle = RequestThrottle.for_model(
        backbone,
        max_concurrent_requests=experiment.max_concurrent_requests,
        requests_per_minute=experiment.requests_per_minute_limit,
    )
    run_id = build_run_id(backbone.name)
    run_paths = prepare_registered_run_layout(family_name, run_root, experiment.name, phase_name, run_id)
    total_calls, total_predictions = estimate_work(experiment, phase_name, benchmarks, controls, protocol)
    progress = RunProgressTracker(
        run_paths.progress,
        total_calls,
        total_predictions,
        planned_calls_are_upper_bound=True,
        target_network_rpm=experiment.requests_per_minute_limit,
        rate_limit_snapshot_provider=throttle.snapshot,
    )

    manifest = finalize_family_manifest(
        {
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "family_name": family_name,
            "family_display_name": display_name,
            "experiment_name": experiment.name,
            "phase_name": phase_name,
            "primary_model_ref": experiment.primary_model_ref,
            "resolved_model": asdict(backbone),
            "verifier_model_refs": experiment.verifier_model_refs,
            "legacy_experiment": experiment.legacy_experiment,
            "resolved_verifier_models": [asdict(item[1]) for item in verifier_backbones],
            "experiment": experiment.name,
            "description": experiment.description,
            "phase": phase_name,
            "phase_metadata": phase_metadata(experiment, phase_name),
            "protocol": asdict(protocol),
            "control_prompt_version": experiment.control_prompt_version,
            "cred_prompt_version": CRED_PROMPT_VERSION,
            "cred_output_protocol": experiment.cred_output_protocol,
            "cred_stage_a_output_protocol": experiment.cred_stage_a_output_protocol,
            "cred_verification_output_protocol": experiment.cred_verification_output_protocol,
            "control_output_protocol": "free_text_answer_v1",
            "artifact_version": ARTIFACT_VERSION,
            "global_seed": experiment.global_seed,
            "max_concurrent_requests": experiment.max_concurrent_requests,
            "requests_per_minute_limit": experiment.requests_per_minute_limit,
            "benchmarks": [asdict(item) for item in benchmarks],
            "dataset_order": [item.slug for item in benchmarks],
            "control_method_names": experiment.control_methods,
            "control_methods": {name: asdict(controls[name]) for name in experiment.control_methods},
            "cred_methods": experiment.cred_methods,
            "method_order": experiment.method_order,
            "total_planned_calls": total_calls,
            "total_planned_predictions": total_predictions,
        },
        family_name=family_name,
    )
    run_paths.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    all_turns: list[dict[str, object]] = []
    all_debate_rows: list[dict[str, object]] = []
    all_router_rows: list[dict[str, object]] = []
    all_predictions: list[dict[str, object]] = []
    try:
        with (
            run_paths.agent_turns.open("w", encoding="utf-8") as turn_handle,
            run_paths.debate_messages.open("w", encoding="utf-8") as debate_handle,
            run_paths.router_decisions.open("w", encoding="utf-8") as router_handle,
            run_paths.predictions.open("w", encoding="utf-8") as prediction_handle,
        ):
            turn_writer = BufferedJsonlWriter(turn_handle)
            debate_writer = BufferedJsonlWriter(debate_handle)
            router_writer = BufferedJsonlWriter(router_handle)
            prediction_writer = BufferedJsonlWriter(prediction_handle)
            for dataset_result in _run_dataset_batches_parallel(
                run_id=run_id,
                benchmarks=benchmarks,
                experiment=experiment,
                phase_name=phase_name,
                protocol=protocol,
                controls=controls,
                backbone=backbone,
                provider=provider,
                cache_router=cache_router,
                throttle=throttle,
                verifier_providers=verifier_providers,
            ):
                append_outputs(
                    sample_results=[
                        (
                            dataset_result.benchmark_index,
                            dataset_result.turn_rows,
                            dataset_result.debate_rows,
                            dataset_result.router_rows,
                            dataset_result.prediction_rows,
                        )
                    ],
                    dataset_slug=dataset_result.dataset_slug,
                    progress=progress,
                    turn_writer=turn_writer,
                    debate_writer=debate_writer,
                    router_writer=router_writer,
                    prediction_writer=prediction_writer,
                    all_turns=all_turns,
                    all_debate_rows=all_debate_rows,
                    all_router_rows=all_router_rows,
                    all_predictions=all_predictions,
                )

        metrics = build_metrics(
            all_predictions,
            dataset_order=[benchmark.slug for benchmark in benchmarks],
            method_order=experiment.method_order,
            control_names=experiment.control_methods,
        )
        debate_diagnostics = build_debate_diagnostics(
            all_predictions,
            dataset_order=[benchmark.slug for benchmark in benchmarks],
            method_order=experiment.method_order,
        )
        router_eval = build_router_eval(all_router_rows)
        error_bucket_audit = build_error_bucket_audit(all_predictions, all_turns)
        output_protocol_diagnostics = build_shared_output_protocol_diagnostics(
            all_turns,
            dataset_order=[benchmark.slug for benchmark in benchmarks],
            method_order=experiment.method_order,
        )
        run_paths.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        run_paths.diagnostic_path("debate_diagnostics.json").write_text(json.dumps(debate_diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
        run_paths.diagnostic_path("router_eval.json").write_text(json.dumps(router_eval, ensure_ascii=False, indent=2), encoding="utf-8")
        run_paths.diagnostic_path("error_bucket_audit.json").write_text(
            json.dumps(error_bucket_audit, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        run_paths.diagnostic_path("output_protocol_diagnostics.json").write_text(
            json.dumps(output_protocol_diagnostics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        run_paths.run_summary.write_text(
            json.dumps(summarize_run(run_paths.root, family_name=family_name), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        render_report(run_paths.root, family_name=family_name, display_name=display_name)
        finalize_run_outputs(
            run_paths.root,
            validator=lambda path: validate_run(path, family_name=family_name),
            validation_path=run_paths.validation,
        )
        progress.mark_completed()
        return run_paths.root
    finally:
        progress.close()
        provider.close()
        for _, _, verifier_provider, _ in verifier_providers:
            verifier_provider.close()
        cache_router.close()


def _control_results_as_cred_shape(control_results):
    for sample_index, turn_rows, debate_rows, prediction_row in control_results:
        yield sample_index, turn_rows, debate_rows, [], [prediction_row]


def _run_dataset_batches_parallel(
    *,
    run_id: str,
    benchmarks,
    experiment: CredVExperimentConfig,
    phase_name: str,
    protocol,
    controls,
    backbone,
    provider,
    cache_router,
    throttle,
    verifier_providers,
):
    indexed_benchmarks = list(enumerate(benchmarks))
    if not indexed_benchmarks:
        return
    max_workers = max(1, min(int(experiment.max_concurrent_requests or 1), len(indexed_benchmarks)))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(
                _run_dataset_batch,
                benchmark_index=benchmark_index,
                benchmark=benchmark,
                run_id=run_id,
                experiment=experiment,
                phase_name=phase_name,
                protocol=protocol,
                controls=controls,
                backbone=backbone,
                provider=provider,
                cache_router=cache_router,
                throttle=throttle,
                verifier_providers=verifier_providers,
            ): benchmark_index
            for benchmark_index, benchmark in indexed_benchmarks
        }
        pending: dict[int, _DatasetRunResult] = {}
        next_index = 0
        for future in as_completed(future_to_index):
            result = future.result()
            pending[result.benchmark_index] = result
            while next_index in pending:
                yield pending.pop(next_index)
                next_index += 1


def _run_dataset_batch(
    *,
    benchmark_index: int,
    benchmark,
    run_id: str,
    experiment: CredVExperimentConfig,
    phase_name: str,
    protocol,
    controls,
    backbone,
    provider,
    cache_router,
    throttle,
    verifier_providers,
) -> _DatasetRunResult:
    cache = cache_router.for_request_target(
        provider=backbone.provider,
        request_model=backbone.model_id,
        dataset=benchmark.slug,
    )
    verifier_runtimes = [
        CredVerifierRuntime(
            model_ref=model_ref,
            backbone=verifier_backbone,
            provider=verifier_provider,
            throttle=verifier_throttle,
            cache=cache_router.for_request_target(
                provider=verifier_backbone.provider,
                request_model=verifier_backbone.model_id,
                dataset=benchmark.slug,
            ),
        )
        for model_ref, verifier_backbone, verifier_provider, verifier_throttle in verifier_providers
    ]
    split_name = resolve_split_name(experiment, phase_name, benchmark.slug)
    samples = load_selected_samples(benchmark, split_name)
    turn_rows: list[dict[str, Any]] = []
    debate_rows: list[dict[str, Any]] = []
    router_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    _collect_sample_results(
        run_cred_batch(
            run_id=run_id,
            benchmark_slug=benchmark.slug,
            split_name=split_name,
            samples=samples,
            experiment=experiment,
            protocol=protocol,
            backbone=backbone,
            provider=provider,
            cache=cache,
            throttle=throttle,
            verifier_runtimes=verifier_runtimes,
        ),
        turn_rows=turn_rows,
        debate_rows=debate_rows,
        router_rows=router_rows,
        prediction_rows=prediction_rows,
    )
    for control_name in experiment.control_methods:
        _collect_sample_results(
            _control_results_as_cred_shape(
                run_unified_control_batch(
                    run_id=run_id,
                    samples=samples,
                    control_name=control_name,
                    method=controls[control_name],
                    benchmark_slug=benchmark.slug,
                    split_name=split_name,
                    backbone=backbone,
                    provider=provider,
                    cache=cache,
                    throttle=throttle,
                    global_seed=experiment.global_seed,
                    max_concurrent_requests=experiment.max_concurrent_requests,
                    execute_turn=_execute_turn,
                    build_prediction_row=build_control_prediction_row,
                    prompt_version=experiment.control_prompt_version,
                )
            ),
            turn_rows=turn_rows,
            debate_rows=debate_rows,
            router_rows=router_rows,
            prediction_rows=prediction_rows,
        )
    return _DatasetRunResult(
        benchmark_index=benchmark_index,
        dataset_slug=benchmark.slug,
        turn_rows=turn_rows,
        debate_rows=debate_rows,
        router_rows=router_rows,
        prediction_rows=prediction_rows,
    )


def _collect_sample_results(
    sample_results,
    *,
    turn_rows: list[dict[str, Any]],
    debate_rows: list[dict[str, Any]],
    router_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> None:
    for _, sample_turn_rows, sample_debate_rows, sample_router_rows, sample_prediction_rows in sample_results:
        turn_rows.extend(sample_turn_rows)
        debate_rows.extend(sample_debate_rows)
        router_rows.extend(sample_router_rows)
        prediction_rows.extend(sample_prediction_rows)


def _validate_control_methods(experiment: CredVExperimentConfig, controls) -> None:
    missing = [name for name in experiment.control_methods if name not in controls]
    if missing:
        raise RuntimeError("Unknown cred_v control methods: " + ", ".join(missing))
