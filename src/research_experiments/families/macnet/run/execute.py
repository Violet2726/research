"""MacNet 顶层实验执行入口。"""

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
from research_experiments.families.macnet.config import MacnetExperimentConfig, load_benchmarks, load_protocol_config
from research_experiments.families.macnet.profiles import load_profile_bank, summarize_profile_bank
from research_experiments.families.macnet.run.io import _prepare_run_paths
from research_experiments.families.macnet.run.report import render_report
from research_experiments.families.macnet.run.sample import (
    _active_methods,
    _build_metrics,
    _build_scaling_summary,
    _estimate_work,
    _load_selected_samples,
    _resolve_split_name,
    _run_sample_batch,
    _write_sample_outputs,
)
from research_experiments.families.macnet.run.validate import validate_run
from research_experiments.workspace.layout import default_cache_root, default_runs_root


def run_experiment(
    experiment: MacnetExperimentConfig,
    phase_name: str,
    backbone,
    run_root: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> Path:
    """执行一个 MacNet phase，并写出完整运行目录。"""

    load_dotenv(".env.local", override=False)
    run_root = run_root or default_runs_root("macnet")
    cache_root = cache_root or default_cache_root()
    benchmarks = load_benchmarks(experiment)
    methods = _active_methods(experiment)
    protocol = load_protocol_config(experiment.protocol)
    profile_bank = load_profile_bank(protocol.profile_asset_path)
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
        "family_name": "macnet",
        "experiment_name": experiment.name,
        "phase_name": phase_name,
        "primary_model_ref": experiment.primary_model_ref,
        "resolved_model": asdict(backbone),
        "experiment_kind": experiment.experiment_kind,
        "prompt_version": experiment.prompt_version,
        "protocol": _serialize_protocol(protocol),
        "methods": [asdict(method) for method in methods],
        "benchmarks": [asdict(item) for item in benchmarks],
        "profile_bank_summary": summarize_profile_bank(profile_bank),
        "total_planned_calls": total_calls,
        "total_planned_predictions": total_predictions,
        "phase_metadata": dict(experiment.raw["phases"][phase_name]),
    }
    run_paths.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    all_artifact_rows: list[dict[str, object]] = []
    all_instruction_rows: list[dict[str, object]] = []
    final_predictions: list[dict[str, object]] = []
    topology_specs: list[dict[str, object]] = []

    with (
        run_paths.artifact_trace.open("w", encoding="utf-8") as artifact_handle_raw,
        run_paths.instruction_trace.open("w", encoding="utf-8") as instruction_handle_raw,
        run_paths.final_predictions.open("w", encoding="utf-8") as prediction_handle_raw,
    ):
        artifact_handle = BufferedJsonlWriter(artifact_handle_raw)
        instruction_handle = BufferedJsonlWriter(instruction_handle_raw)
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
                phase_name=phase_name,
                samples=samples,
                protocol=protocol,
                experiment=experiment,
                backbone=backbone,
                provider=provider,
                cache=cache,
                limiter=limiter,
                profile_bank=profile_bank,
            )
            _write_sample_outputs(
                sample_results=sample_results,
                dataset_slug=benchmark.slug,
                progress=progress,
                artifact_handle=artifact_handle,
                instruction_handle=instruction_handle,
                prediction_handle=prediction_handle,
                all_artifact_rows=all_artifact_rows,
                all_instruction_rows=all_instruction_rows,
                final_predictions=final_predictions,
                topology_specs=topology_specs,
            )

    metrics = _build_metrics(final_predictions, model_name=backbone.name)
    run_paths.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    run_paths.topology_manifest.write_text(
        json.dumps({"topologies": topology_specs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    run_paths.scaling_summary.write_text(
        json.dumps(_build_scaling_summary(final_predictions), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
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


def _serialize_protocol(protocol) -> dict[str, object]:
    payload = asdict(protocol)
    payload["profile_asset_path"] = str(protocol.profile_asset_path)
    return payload
