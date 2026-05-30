"""`comm_necessary` 实验主运行链路。

本模块把 HotpotQA 通信必要性实验落成完整流程：
构造 split-context 视图、运行不同消息强度的方法、聚合答案与 supporting facts，
并导出联合指标、官方预测文件与报告产物。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from research_experiments.core.config import ResolvedModelConfig
from research_experiments.core.data.datasets import select_samples
from research_experiments.core.execution.artifacts import BufferedJsonlWriter
from research_experiments.core.execution.cache import RequestCacheRouter
from research_experiments.core.execution.providers import OpenAICompatibleProvider
from research_experiments.core.execution.rate_limits import SlidingWindowRateLimiter
from research_experiments.core.execution.runtime import RunProgressTracker, build_run_id, finalize_run_outputs
from research_experiments.families.comm_necessary.config import (
    CommNecessaryExperimentConfig,
    load_protocol_config,
)
from research_experiments.family_runtime.layout import prepare_registered_run_layout
from research_experiments.families.comm_necessary.run.sample import (
    _build_diagnostics,
    _build_metrics,
    _estimate_work,
    _resolve_split_name,
    _run_sample_batch,
    _write_hotpot_predictions,
    _write_paper_summary,
)
from research_experiments.family_runtime.manifest import finalize_family_manifest
from research_experiments.family_runtime.config_helpers import load_benchmarks, phase_metadata
from research_experiments.workspace.layout import default_cache_root, default_runs_root


def run_experiment(
    experiment: CommNecessaryExperimentConfig,
    phase_name: str,
    backbone: ResolvedModelConfig,
    run_root: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> Path:
    """执行一个 comm_necessary phase，并写出完整运行目录。"""
    from research_experiments.families.comm_necessary.run.report import render_report
    from research_experiments.families.comm_necessary.run.validate import validate_run

    load_dotenv(".env.local", override=False)
    run_root = run_root or default_runs_root("comm_necessary")
    cache_root = cache_root or default_cache_root()
    protocol = load_protocol_config(experiment.protocol)
    benchmarks = load_benchmarks(experiment)
    provider = OpenAICompatibleProvider(backbone)
    cache_router = RequestCacheRouter(cache_root)
    limiter = SlidingWindowRateLimiter(
        requests_per_minute=experiment.requests_per_minute_limit,
        tokens_per_minute=experiment.tokens_per_minute_limit,
    )

    run_id = build_run_id(backbone.name)
    paths = prepare_registered_run_layout('comm_necessary', run_root, experiment.name, phase_name, run_id)
    total_calls, total_predictions = _estimate_work(experiment, phase_name, benchmarks, protocol)
    progress = RunProgressTracker(paths.progress, total_calls, total_predictions)

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "experiment": experiment.name,
        "description": experiment.description,
        "phase": phase_name,
        "phase_metadata": phase_metadata(experiment, phase_name),
        "prompt_version": experiment.prompt_version,
        "backbone": asdict(backbone),
        "benchmarks": [asdict(item) for item in benchmarks],
        "protocol": asdict(protocol),
        "methods": experiment.methods,
        "global_seed": experiment.global_seed,
        "max_concurrent_requests": experiment.max_concurrent_requests,
        "requests_per_minute_limit": experiment.requests_per_minute_limit,
        "tokens_per_minute_limit": experiment.tokens_per_minute_limit,
        "total_planned_calls": total_calls,
        "total_planned_predictions": total_predictions,
        "source_note": "HotpotQA distractor provides answer and sentence-level supporting facts; AgentsNet is reserved for later topology experiments.",
    }
    manifest = finalize_family_manifest(manifest, family_name="comm_necessary")
    paths.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    all_views: list[dict[str, Any]] = []
    all_stage_a: list[dict[str, Any]] = []
    all_packets: list[dict[str, Any]] = []
    all_stage_b: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []

    try:
        with (
            paths.sample_views.open("w", encoding="utf-8") as views_handle,
            paths.stage_a_turns.open("w", encoding="utf-8") as stage_a_handle,
            paths.message_packets.open("w", encoding="utf-8") as packets_handle,
            paths.stage_b_turns.open("w", encoding="utf-8") as stage_b_handle,
            paths.final_predictions.open("w", encoding="utf-8") as predictions_handle,
        ):
            views_writer = BufferedJsonlWriter(views_handle)
            stage_a_writer = BufferedJsonlWriter(stage_a_handle)
            packets_writer = BufferedJsonlWriter(packets_handle)
            stage_b_writer = BufferedJsonlWriter(stage_b_handle)
            predictions_writer = BufferedJsonlWriter(predictions_handle)
            for benchmark in benchmarks:
                cache = cache_router.for_request_target(
                    provider=backbone.provider,
                    request_model=backbone.model_id,
                    dataset=benchmark.cache_namespace or benchmark.slug,
                )
                split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
                samples = select_samples(benchmark, split_name)
                results = _run_sample_batch(
                    run_id=run_id,
                    dataset=benchmark.cache_namespace or benchmark.slug,
                    split_name=split_name,
                    samples=samples,
                    protocol=protocol,
                    backbone=backbone,
                    provider=provider,
                    cache=cache,
                    limiter=limiter,
                    global_seed=experiment.global_seed,
                    prompt_version=experiment.prompt_version,
                    max_concurrent_requests=experiment.max_concurrent_requests,
                )
                for result in results:
                    for row in result.sample_views:
                        views_writer.write_row(row)
                    for row in result.stage_a_turns:
                        stage_a_writer.write_row(row)
                        progress.record_call(row, method_key="stage_name")
                    for row in result.message_packets:
                        packets_writer.write_row(row)
                    for row in result.stage_b_turns:
                        stage_b_writer.write_row(row)
                        progress.record_call(row, method_key="method_name")
                    for row in result.final_predictions:
                        predictions_writer.write_row(row)
                        progress.record_predictions(1, str(row["dataset"]), str(row["method_name"]))
                    all_views.extend(result.sample_views)
                    all_stage_a.extend(result.stage_a_turns)
                    all_packets.extend(result.message_packets)
                    all_stage_b.extend(result.stage_b_turns)
                    all_predictions.extend(result.final_predictions)

        metrics = _build_metrics(all_predictions)
        diagnostics = _build_diagnostics(all_predictions, all_views)
        paths.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        paths.diagnostics.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_hotpot_predictions(paths.hotpot_predictions, all_predictions)
        _write_paper_summary(paths.paper_summary, metrics)
        render_report(paths.root)
        finalize_run_outputs(
            paths.root,
            validator=validate_run,
            validation_path=paths.run_validation,
        )
        progress.mark_completed()
        return paths.root
    finally:
        provider.close()
        cache_router.close()
