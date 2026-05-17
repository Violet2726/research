"""ColMAD family 的顶层实验执行入口。"""

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
from research_experiments.workspace.layout import default_cache_root, default_runs_root
from research_experiments.families.colmad.config import ColmadExperimentConfig, load_protocol_config
from research_experiments.families.colmad.run.io import _prepare_run_paths
from research_experiments.families.colmad.run.report import render_report, summarize_run
from research_experiments.families.colmad.run.sample import (
    _active_methods,
    _build_metrics,
    _build_protocol_diagnostics,
    _estimate_work,
    _load_selected_samples,
    _resolve_split_name,
    _run_sample_batch,
)
from research_experiments.families.colmad.run.validate import validate_run
from research_experiments.families.shared.config_loading import load_benchmarks


def run_experiment(
    experiment: ColmadExperimentConfig,
    phase_name: str,
    backbone,
    run_root: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> Path:
    """执行一个 ColMAD phase，并写出完整运行目录。"""

    load_dotenv(".env.local", override=False)
    run_root = run_root or default_runs_root("colmad")
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
    total_calls, total_predictions = _estimate_work(experiment, phase_name, benchmarks, methods)
    progress = RunProgressTracker(
        run_paths.progress,
        total_calls,
        total_predictions,
        planned_calls_are_upper_bound=True,
    )

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "family_name": "colmad",
        "experiment_name": experiment.name,
        "experiment_kind": experiment.experiment_kind,
        "phase_name": phase_name,
        "primary_model_ref": experiment.primary_model_ref,
        "resolved_model": asdict(backbone),
        "description": experiment.description,
        "prompt_version": experiment.prompt_version,
        "protocol": asdict(protocol),
        "methods": [asdict(method) for method in methods],
        "benchmarks": [asdict(item) for item in benchmarks],
        "total_planned_calls": total_calls,
        "total_planned_predictions": total_predictions,
    }
    run_paths.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    debate_rows: list[dict[str, object]] = []
    judge_rows: list[dict[str, object]] = []
    final_predictions: list[dict[str, object]] = []

    with (
        run_paths.debate_trace.open("w", encoding="utf-8") as debate_handle_raw,
        run_paths.judge_trace.open("w", encoding="utf-8") as judge_handle_raw,
        run_paths.final_predictions.open("w", encoding="utf-8") as prediction_handle_raw,
    ):
        debate_handle = BufferedJsonlWriter(debate_handle_raw)
        judge_handle = BufferedJsonlWriter(judge_handle_raw)
        prediction_handle = BufferedJsonlWriter(prediction_handle_raw)

        for benchmark in benchmarks:
            cache = cache_router.for_request_target(
                provider=backbone.provider,
                request_model=backbone.model_id,
                dataset=benchmark.cache_namespace or benchmark.slug,
            )
            split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
            samples = _load_selected_samples(benchmark, split_name)
            batch_results = _run_sample_batch(
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
            for payload in batch_results:
                for row in payload.debate_rows:
                    row["model_name"] = backbone.name
                    debate_handle.write_row(row)
                    progress.record_call(row, method_key="method_name")
                    debate_rows.append(row)
                for row in payload.judge_rows:
                    row["model_name"] = backbone.name
                    judge_handle.write_row(row)
                    progress.record_call(row, method_key="method_name")
                    judge_rows.append(row)
                for row in payload.prediction_rows:
                    row["model_name"] = backbone.name
                    prediction_handle.write_row(row)
                    progress.record_predictions(1, benchmark.slug, row["method_name"])
                    final_predictions.append(row)

    metrics = _build_metrics(final_predictions, methods, model_name=backbone.name)
    protocol_diagnostics = _build_protocol_diagnostics(final_predictions, judge_rows)

    run_paths.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    run_paths.protocol_diagnostics.write_text(json.dumps(protocol_diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
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
