"""iMAD family 的顶层实验执行入口。"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Any

from dotenv import load_dotenv

from research_experiments.core.execution.artifacts import BufferedJsonlWriter
from research_experiments.core.execution.cache import RequestCacheRouter
from research_experiments.core.execution.providers import OpenAICompatibleProvider
from research_experiments.core.execution.rate_limits import SlidingWindowRateLimiter
from research_experiments.core.execution.runtime import RunProgressTracker, build_run_id, finalize_run_outputs
from research_experiments.core.controls.no_comm_controls import run_no_comm_control_batch
from research_experiments.workspace.layout import default_cache_root, default_runs_root
from research_experiments.families.imad.config import load_benchmarks, load_control_catalog, load_protocol_config, ImadExperimentConfig
from research_experiments.families.imad.prompts import build_initial_messages
from research_experiments.families.imad.run.io import _prepare_run_paths
from research_experiments.families.imad.run.report import render_report, summarize_run
from research_experiments.families.imad.run.sample import (
    _active_methods,
    _build_control_prediction_row,
    _build_cost_breakdown,
    _build_metrics,
    _build_stability_diagnostics,
    _estimate_work,
    _load_selected_samples,
    _resolve_split_name,
    _run_method_batch,
    _write_sample_outputs,
    _execute_turn,
)
from research_experiments.families.imad.run.validate import validate_run


def run_experiment(
    experiment: ImadExperimentConfig,
    phase_name: str,
    backbone,
    run_root: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> Path:
    """执行一个 iMAD phase，并写出完整运行目录。"""

    load_dotenv(".env.local", override=False)
    run_root = run_root or default_runs_root("imad")
    cache_root = cache_root or default_cache_root()
    benchmarks = load_benchmarks(experiment)
    methods = _active_methods(experiment)
    controls = load_control_catalog(experiment.control_catalog)
    control_names = sorted({name for method in methods for name in method.matched_controls})
    protocol = load_protocol_config(experiment.protocol)
    provider = OpenAICompatibleProvider(backbone)
    cache_router = RequestCacheRouter(cache_root)
    limiter = SlidingWindowRateLimiter(
        requests_per_minute=experiment.requests_per_minute_limit,
        tokens_per_minute=experiment.tokens_per_minute_limit,
    )
    run_id = build_run_id(backbone.name)
    run_paths = _prepare_run_paths(run_root, experiment.name, phase_name, run_id)
    total_calls, total_predictions = _estimate_work(experiment, phase_name, benchmarks, protocol, methods, controls)
    progress = RunProgressTracker(run_paths.progress, total_calls, total_predictions)

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "family_name": "imad",
        "experiment_name": experiment.name,
        "phase_name": phase_name,
        "primary_model_ref": experiment.primary_model_ref,
        "resolved_model": asdict(backbone),
        "experiment": experiment.name,
        "description": experiment.description,
        "phase": phase_name,
        "phase_metadata": dict(experiment.raw["phases"][phase_name]),
        "prompt_version": experiment.prompt_version,
        "protocol": asdict(protocol),
        "methods": [asdict(method) for method in methods],
        "control_methods": {name: asdict(controls[name]) for name in control_names},
        "benchmarks": [asdict(item) for item in benchmarks],
        "total_planned_calls": total_calls,
        "total_planned_predictions": total_predictions,
    }
    run_paths.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    all_turns: list[dict[str, Any]] = []
    all_round_diagnostics: list[dict[str, Any]] = []
    final_predictions: list[dict[str, Any]] = []

    with (
        run_paths.agent_turns.open("w", encoding="utf-8") as turn_handle_raw,
        run_paths.debate_messages.open("w", encoding="utf-8") as debate_handle_raw,
        run_paths.round_diagnostics.open("w", encoding="utf-8") as round_handle_raw,
        run_paths.final_predictions.open("w", encoding="utf-8") as prediction_handle_raw,
    ):
        turn_handle = BufferedJsonlWriter(turn_handle_raw)
        debate_handle = BufferedJsonlWriter(debate_handle_raw)
        round_handle = BufferedJsonlWriter(round_handle_raw)
        prediction_handle = BufferedJsonlWriter(prediction_handle_raw)
        for benchmark in benchmarks:
            cache = cache_router.for_request_target(
                provider=backbone.provider,
                request_model=backbone.model_id,
                dataset=benchmark.slug,
            )
            split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
            samples = _load_selected_samples(benchmark, split_name)

            for method in methods:
                method_results = _run_method_batch(
                    run_id=run_id,
                    benchmark_slug=benchmark.slug,
                    split_name=split_name,
                    samples=samples,
                    method=method,
                    protocol=protocol,
                    backbone=backbone,
                    provider=provider,
                    cache=cache,
                    limiter=limiter,
                    global_seed=experiment.global_seed,
                    prompt_version=experiment.prompt_version,
                    max_concurrent_requests=experiment.max_concurrent_requests,
                )
                _write_sample_outputs(
                    sample_results=method_results,
                    dataset_slug=benchmark.slug,
                    progress=progress,
                    turn_handle=turn_handle,
                    debate_handle=debate_handle,
                    round_handle=round_handle,
                    prediction_handle=prediction_handle,
                    all_turns=all_turns,
                    all_round_diagnostics=all_round_diagnostics,
                    final_predictions=final_predictions,
                )

            for control_name in control_names:
                control_results = run_no_comm_control_batch(
                    samples=samples,
                    control_name=control_name,
                    method=controls[control_name],
                    run_id=run_id,
                    benchmark_slug=benchmark.slug,
                    split_name=split_name,
                    backbone=backbone,
                    provider=provider,
                    cache=cache,
                    limiter=limiter,
                    global_seed=experiment.global_seed,
                    prompt_version=experiment.prompt_version,
                    max_concurrent_requests=experiment.max_concurrent_requests,
                    build_messages=build_initial_messages,
                    execute_turn=_execute_turn,
                    build_prediction_row=_build_control_prediction_row,
                )
                control_results_with_rounds = [
                    (sample_index, turn_rows, debate_rows, [], prediction_row)
                    for sample_index, turn_rows, debate_rows, prediction_row in control_results
                ]
                _write_sample_outputs(
                    sample_results=control_results_with_rounds,
                    dataset_slug=benchmark.slug,
                    progress=progress,
                    turn_handle=turn_handle,
                    debate_handle=debate_handle,
                    round_handle=round_handle,
                    prediction_handle=prediction_handle,
                    all_turns=all_turns,
                    all_round_diagnostics=all_round_diagnostics,
                    final_predictions=final_predictions,
                )

    metrics = _build_metrics(final_predictions, methods)
    stability_diagnostics = _build_stability_diagnostics(final_predictions, all_round_diagnostics)
    cost_breakdown = _build_cost_breakdown(all_turns)

    run_paths.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    run_paths.stability_diagnostics.write_text(json.dumps(stability_diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    run_paths.cost_breakdown.write_text(json.dumps(cost_breakdown, ensure_ascii=False, indent=2), encoding="utf-8")
    run_paths.run_summary.write_text(json.dumps(summarize_run(run_paths.root), ensure_ascii=False, indent=2), encoding="utf-8")
    render_report(run_paths.root)
    finalize_run_outputs(
        run_paths.root,
        validator=validate_run,
        validation_path=run_paths.run_validation,
    )
    progress.mark_completed()
    provider.close()
    cache_router.close()
    return run_paths.root
