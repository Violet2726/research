"""baseline_compare 的执行主链路。"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from research_experiments.core.controls.no_comm_controls import run_unified_control_batch
from research_experiments.core.execution.artifacts import BufferedJsonlWriter
from research_experiments.core.execution.cache import RequestCacheRouter
from research_experiments.core.execution.providers import OpenAICompatibleProvider
from research_experiments.core.execution.rate_limits import RequestThrottle
from research_experiments.core.execution.runtime import RunProgressTracker, build_run_id, finalize_run_outputs
from research_experiments.core.structured_outputs import ARTIFACT_VERSION
from research_experiments.families.baseline_compare.config import (
    BaselineCompareExperimentConfig,
    load_control_catalog,
    load_protocol_config,
    load_roster_config,
)
from research_experiments.families.baseline_compare.run.report import render_report, summarize_run
from research_experiments.families.baseline_compare.run.sample import (
    _active_setups,
    _build_control_prediction_row,
    _build_cost_breakdown,
    _build_debate_diagnostics,
    _build_metrics,
    _estimate_work,
    _execute_turn,
    _load_selected_samples,
    _resolve_split_name,
    _run_mad_setup_batch,
    _write_sample_outputs,
)
from research_experiments.families.baseline_compare.run.validate import validate_run
from research_experiments.family_runtime.config_helpers import load_benchmarks, phase_metadata
from research_experiments.family_runtime.layout import prepare_registered_run_layout
from research_experiments.family_runtime.manifest import finalize_family_manifest
from research_experiments.workspace.layout import default_cache_root, default_runs_root


def run_experiment(
    experiment: BaselineCompareExperimentConfig,
    phase_name: str,
    backbone,
    run_root: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> Path:
    load_dotenv(".env.local", override=False)
    run_root = run_root or default_runs_root("baseline_compare")
    cache_root = cache_root or default_cache_root()
    benchmarks = load_benchmarks(experiment)
    phase = phase_metadata(experiment, phase_name)
    setups = _active_setups(experiment, phase_name)
    controls = load_control_catalog(experiment.control_catalog)
    control_names = _resolve_control_names(experiment, controls)
    provider = OpenAICompatibleProvider(backbone)
    cache_router = RequestCacheRouter(cache_root)
    throttle = RequestThrottle.for_model(
        backbone,
        max_concurrent_requests=experiment.max_concurrent_requests,
        requests_per_minute=experiment.requests_per_minute_limit,
        tokens_per_minute=experiment.tokens_per_minute_limit,
    )
    run_id = build_run_id(backbone.name)
    run_paths = prepare_registered_run_layout(
        "baseline_compare",
        run_root,
        experiment.name,
        phase_name,
        run_id,
    )
    total_calls, total_predictions = _estimate_work(
        experiment,
        phase_name,
        benchmarks,
        setups,
        controls,
    )
    progress = RunProgressTracker(
        run_paths.progress,
        total_calls,
        total_predictions,
        target_network_rpm=experiment.requests_per_minute_limit,
        rate_limit_snapshot_provider=throttle.snapshot,
    )

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "family_name": "baseline_compare",
        "experiment_name": experiment.name,
        "phase_name": phase_name,
        "primary_model_ref": experiment.primary_model_ref,
        "resolved_model": asdict(backbone),
        "experiment": experiment.name,
        "description": experiment.description,
        "phase": phase_name,
        "phase_metadata": phase,
        "prompt_version": experiment.prompt_version,
        "max_concurrent_requests": experiment.max_concurrent_requests,
        "requests_per_minute_limit": experiment.requests_per_minute_limit,
        "tokens_per_minute_limit": experiment.tokens_per_minute_limit,
        "artifact_version": ARTIFACT_VERSION,
        "backbone": asdict(backbone),
        "benchmarks": [asdict(item) for item in benchmarks],
        "dataset_order": [item.slug for item in benchmarks],
        "control_method_names": control_names,
        "control_methods": {name: asdict(controls[name]) for name in control_names},
        "method_order": experiment.method_order,
        "setups": [
            {
                "name": setup.name,
                "protocol": asdict(load_protocol_config(setup.protocol)),
                "roster": asdict(load_roster_config(setup.roster)),
            }
            for setup in setups
        ],
        "total_planned_calls": total_calls,
        "total_planned_predictions": total_predictions,
    }
    manifest = finalize_family_manifest(manifest, family_name="baseline_compare")
    run_paths.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    all_turns: list[dict[str, object]] = []
    debate_messages: list[dict[str, object]] = []
    final_predictions: list[dict[str, object]] = []

    try:
        with (
            run_paths.agent_turns.open("w", encoding="utf-8") as turn_handle,
            run_paths.debate_messages.open("w", encoding="utf-8") as debate_handle,
            run_paths.final_predictions.open("w", encoding="utf-8") as prediction_handle,
        ):
            turn_writer = BufferedJsonlWriter(turn_handle)
            debate_writer = BufferedJsonlWriter(debate_handle)
            prediction_writer = BufferedJsonlWriter(prediction_handle)
            for benchmark in benchmarks:
                cache = cache_router.for_request_target(
                    provider=backbone.provider,
                    request_model=backbone.model_id,
                    dataset=benchmark.slug,
                )
                split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
                samples = _load_selected_samples(benchmark, split_name)

                for setup in setups:
                    protocol = load_protocol_config(setup.protocol)
                    roster = load_roster_config(setup.roster)
                    mad_results = _run_mad_setup_batch(
                        run_id=run_id,
                        benchmark_slug=benchmark.slug,
                        split_name=split_name,
                        samples=samples,
                        setup=setup,
                        protocol=protocol,
                        roster=roster,
                        backbone=backbone,
                        provider=provider,
                        cache=cache,
                        throttle=throttle,
                        global_seed=experiment.global_seed,
                        prompt_version=experiment.prompt_version,
                        max_concurrent_requests=experiment.max_concurrent_requests,
                    )
                    _write_sample_outputs(
                        sample_results=mad_results,
                        dataset_slug=benchmark.slug,
                        progress=progress,
                        turn_handle=turn_writer,
                        debate_handle=debate_writer,
                        prediction_handle=prediction_writer,
                        all_turns=all_turns,
                        debate_messages=debate_messages,
                        final_predictions=final_predictions,
                    )

                for control_name in control_names:
                    control_results = run_unified_control_batch(
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
                        build_prediction_row=_build_control_prediction_row,
                    )
                    _write_sample_outputs(
                        sample_results=control_results,
                        dataset_slug=benchmark.slug,
                        progress=progress,
                        turn_handle=turn_writer,
                        debate_handle=debate_writer,
                        prediction_handle=prediction_writer,
                        all_turns=all_turns,
                        debate_messages=debate_messages,
                        final_predictions=final_predictions,
                    )

        metrics = _build_metrics(
            final_predictions,
            dataset_order=[benchmark.slug for benchmark in benchmarks],
            method_order=experiment.method_order,
            control_names=control_names,
        )
        cost_breakdown = _build_cost_breakdown(
            all_turns,
            dataset_order=[benchmark.slug for benchmark in benchmarks],
            method_order=experiment.method_order,
        )
        debate_diagnostics = _build_debate_diagnostics(
            final_predictions,
            dataset_order=[benchmark.slug for benchmark in benchmarks],
            method_order=experiment.method_order,
        )

        run_paths.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        run_paths.cost_breakdown.write_text(json.dumps(cost_breakdown, ensure_ascii=False, indent=2), encoding="utf-8")
        run_paths.debate_diagnostics.write_text(
            json.dumps(debate_diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        run_paths.run_summary.write_text(
            json.dumps(summarize_run(run_paths.root), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        render_report(run_paths.root)
        finalize_run_outputs(
            run_paths.root,
            validator=validate_run,
            validation_path=run_paths.run_validation,
        )
        progress.mark_completed()
        return run_paths.root
    finally:
        progress.close()
        provider.close()
        cache_router.close()


def _resolve_control_names(
    experiment: BaselineCompareExperimentConfig,
    controls,
) -> list[str]:
    missing = [name for name in experiment.control_methods if name not in controls]
    if missing:
        raise RuntimeError("Unknown baseline_compare control methods: " + ", ".join(missing))
    return list(experiment.control_methods)
