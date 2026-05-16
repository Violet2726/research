"""ECON family 的顶层实验执行入口。"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import json

from dotenv import load_dotenv

from research_experiments.core.execution.artifacts import BufferedJsonlWriter
from research_experiments.core.execution.cache import RequestCacheRouter
from research_experiments.core.execution.providers import OpenAICompatibleProvider
from research_experiments.core.execution.rate_limits import SlidingWindowRateLimiter
from research_experiments.core.execution.runtime import RunProgressTracker, build_run_id, finalize_run_outputs
from research_experiments.families.econ.config import EconExperimentConfig, load_benchmarks, load_protocol_config
from research_experiments.families.econ.run.io import _prepare_run_paths
from research_experiments.families.econ.run.report import render_report, summarize_run
from research_experiments.families.econ.run.sample import (
    _active_methods,
    _build_metrics,
    _estimate_work,
    _load_selected_samples,
    _resolve_split_name,
    _run_sample_batch,
    _write_sample_outputs,
)
from research_experiments.families.econ.run.validate import validate_run
from research_experiments.workspace.layout import default_cache_root, default_runs_root


def run_experiment(
    experiment: EconExperimentConfig,
    phase_name: str,
    backbone,
    run_root: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> Path:
    """执行一个 ECON phase，并写出完整运行目录。"""

    load_dotenv(".env.local", override=False)
    run_root = run_root or default_runs_root("econ")
    cache_root = cache_root or default_cache_root()
    benchmarks = load_benchmarks(experiment)
    methods = _active_methods(experiment)
    protocol = load_protocol_config(experiment.protocol)
    provider = OpenAICompatibleProvider(backbone)
    cache_router = RequestCacheRouter(cache_root)
    limiter = SlidingWindowRateLimiter(
        requests_per_minute=experiment.requests_per_minute_limit,
        tokens_per_minute=experiment.tokens_per_minute_limit,
    )
    run_id = build_run_id(backbone.name)
    run_paths = _prepare_run_paths(run_root, experiment.name, phase_name, run_id)
    total_calls, total_predictions = _estimate_work(experiment, phase_name, benchmarks, protocol, methods)
    progress = RunProgressTracker(
        run_paths.progress,
        total_calls,
        total_predictions,
        planned_calls_are_upper_bound=True,
    )

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "family_name": "econ",
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
        "benchmarks": [asdict(item) for item in benchmarks],
        "total_planned_calls": total_calls,
        "total_planned_predictions": total_predictions,
    }
    run_paths.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    all_turns: list[dict[str, object]] = []
    all_belief_rows: list[dict[str, object]] = []
    all_equilibrium_rows: list[dict[str, object]] = []
    all_communication_rows: list[dict[str, object]] = []
    final_predictions: list[dict[str, object]] = []

    with (
        run_paths.agent_turns.open("w", encoding="utf-8") as turn_handle_raw,
        run_paths.belief_trace.open("w", encoding="utf-8") as belief_handle_raw,
        run_paths.equilibrium_trace.open("w", encoding="utf-8") as equilibrium_handle_raw,
        run_paths.communication_trace.open("w", encoding="utf-8") as communication_handle_raw,
        run_paths.final_predictions.open("w", encoding="utf-8") as prediction_handle_raw,
    ):
        turn_handle = BufferedJsonlWriter(turn_handle_raw)
        belief_handle = BufferedJsonlWriter(belief_handle_raw)
        equilibrium_handle = BufferedJsonlWriter(equilibrium_handle_raw)
        communication_handle = BufferedJsonlWriter(communication_handle_raw)
        prediction_handle = BufferedJsonlWriter(prediction_handle_raw)
        for benchmark in benchmarks:
            cache = cache_router.for_request_target(
                provider=backbone.provider,
                request_model=backbone.model_id,
                dataset=benchmark.cache_namespace or benchmark.slug,
            )
            split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
            samples = _load_selected_samples(benchmark, split_name)
            sample_results = _run_sample_batch(
                run_id=run_id,
                benchmark_slug=benchmark.slug,
                split_name=split_name,
                samples=samples,
                protocol=protocol,
                experiment=experiment,
                backbone=backbone,
                provider=provider,
                cache=cache,
                limiter=limiter,
            )
            _write_sample_outputs(
                sample_results=sample_results,
                dataset_slug=benchmark.slug,
                progress=progress,
                turn_handle=turn_handle,
                belief_handle=belief_handle,
                equilibrium_handle=equilibrium_handle,
                communication_handle=communication_handle,
                prediction_handle=prediction_handle,
                all_turns=all_turns,
                all_belief_rows=all_belief_rows,
                all_equilibrium_rows=all_equilibrium_rows,
                all_communication_rows=all_communication_rows,
                final_predictions=final_predictions,
            )

    metrics = _build_metrics(final_predictions, methods, model_name=backbone.name)
    run_paths.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
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

