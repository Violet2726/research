"""Free-MAD-lite 主运行链路。

本模块把 Free-MAD-lite 的完整流程落地为可执行实验：
共享 Stage A、单轮 anti-conformity debate、轨迹裁决器调用、
以及题级指标、诊断与报告产物生成。
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
from research_experiments.core.structured_outputs import (
    ARTIFACT_VERSION,
)
from research_experiments.families.free_mad_lite.config import (
    FreeMadLiteExperimentConfig,
    load_protocol_config,
)
from research_experiments.families.free_mad_lite.prompts import (
    anti_conformity_prompt_hash,
)
from research_experiments.families.free_mad_lite.run.io import _prepare_run_paths
from research_experiments.families.free_mad_lite.run.metrics import (
    build_diagnostics_payload as build_free_mad_diagnostics_payload,
)
from research_experiments.families.free_mad_lite.run.metrics import (
    build_metrics_payload as build_free_mad_metrics_payload,
)
from research_experiments.families.free_mad_lite.run.metrics import (
    write_paper_summary as write_free_mad_paper_summary,
)
from research_experiments.families.free_mad_lite.run.report import render_report, summarize_run
from research_experiments.families.free_mad_lite.run.sample import (
    _estimate_work,
    _resolve_split_name,
    _run_sample_batch,
)
from research_experiments.families.free_mad_lite.run.validate import validate_run
from research_experiments.families.shared.config_loading import load_benchmarks, phase_metadata
from research_experiments.workspace.layout import default_cache_root, default_runs_root


def run_experiment(
    experiment: FreeMadLiteExperimentConfig,
    phase_name: str,
    backbone: ResolvedModelConfig,
    run_root: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> Path:
    """执行一个 Free-MAD-lite phase，并写出完整运行目录。"""
    load_dotenv(".env.local", override=False)
    run_root = run_root or default_runs_root("free_mad_lite")
    cache_root = cache_root or default_cache_root()
    protocol = load_protocol_config(experiment.protocol)
    benchmarks = load_benchmarks(experiment)
    phase = phase_metadata(experiment, phase_name)
    provider = OpenAICompatibleProvider(backbone)
    cache_router = RequestCacheRouter(cache_root)
    limiter = SlidingWindowRateLimiter(
        requests_per_minute=experiment.requests_per_minute_limit,
        tokens_per_minute=experiment.tokens_per_minute_limit,
    )
    run_id = build_run_id(backbone.name)
    run_paths = _prepare_run_paths(run_root, experiment.name, phase_name, run_id)
    total_calls, total_predictions = _estimate_work(experiment, phase_name, benchmarks, protocol)
    progress = RunProgressTracker(run_paths.progress, total_calls, total_predictions)

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "experiment": experiment.name,
        "description": experiment.description,
        "phase": phase_name,
        "phase_metadata": phase,
        "prompt_version": experiment.prompt_version,
        "artifact_version": ARTIFACT_VERSION,
        "backbone": asdict(backbone),
        "benchmarks": [asdict(item) for item in benchmarks],
        "protocol": asdict(protocol),
        "methods": experiment.methods,
        "anti_conformity_prompt_hash": anti_conformity_prompt_hash(),
        "max_concurrent_requests": experiment.max_concurrent_requests,
        "requests_per_minute_limit": experiment.requests_per_minute_limit,
        "tokens_per_minute_limit": experiment.tokens_per_minute_limit,
        "limitation_note": "Free-MAD-lite validates anti-conformity and LLM trajectory judging only; it is not full Free-MAD score-model reproduction.",
        "total_planned_calls": total_calls,
        "total_planned_predictions": total_predictions,
    }
    run_paths.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    all_turns: list[dict[str, Any]] = []
    all_debate_messages: list[dict[str, Any]] = []
    all_scores: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []

    try:
        with (
            run_paths.agent_turns.open("w", encoding="utf-8") as turn_handle,
            run_paths.debate_messages.open("w", encoding="utf-8") as debate_handle,
            run_paths.trajectory_scores.open("w", encoding="utf-8") as score_handle,
            run_paths.final_predictions.open("w", encoding="utf-8") as prediction_handle,
        ):
            turn_writer = BufferedJsonlWriter(turn_handle)
            debate_writer = BufferedJsonlWriter(debate_handle)
            score_writer = BufferedJsonlWriter(score_handle)
            prediction_writer = BufferedJsonlWriter(prediction_handle)
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
                for _, turn_rows, debate_rows, score_rows, prediction_rows in results:
                    for row in turn_rows:
                        turn_writer.write_row(row)
                        progress.record_call(row, method_key="method_name")
                    for row in debate_rows:
                        debate_writer.write_row(row)
                    for row in score_rows:
                        score_writer.write_row(row)
                    for row in prediction_rows:
                        prediction_writer.write_row(row)
                        progress.record_predictions(1, str(row["dataset"]), str(row["method_name"]))
                    all_turns.extend(turn_rows)
                    all_debate_messages.extend(debate_rows)
                    all_scores.extend(score_rows)
                    all_predictions.extend(prediction_rows)

        metrics = build_free_mad_metrics_payload(all_predictions)
        diagnostics = build_free_mad_diagnostics_payload(all_predictions, all_scores)
        run_paths.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        run_paths.diagnostics.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
        write_free_mad_paper_summary(run_paths.paper_summary, metrics)
        render_report(run_paths.root)
        run_paths.run_summary.write_text(json.dumps(summarize_run(run_paths.root), ensure_ascii=False, indent=2), encoding="utf-8")
        finalize_run_outputs(
            run_paths.root,
            validator=validate_run,
            validation_path=run_paths.run_validation,
        )
        progress.mark_completed()
        return run_paths.root
    finally:
        provider.close()
        cache_router.close()
