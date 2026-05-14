"""SID-lite 主运行链路。

本模块把 SID-lite 的完整流程落地为可执行实验：
共享 Stage A、早退判定、压缩消息广播、belief update、题级聚合、
以及指标/诊断/报告产物生成。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
from hashlib import sha256
from pathlib import Path
import json
from typing import Any

from dotenv import load_dotenv

from research_experiments.core.execution.artifacts import BufferedJsonlWriter
from research_experiments.core.execution.cache import RequestCache, RequestCacheRouter, json_dump
from research_experiments.core.config import ResolvedModelConfig
from research_experiments.core.data.datasets import DatasetSample, load_split_ids, select_samples
from research_experiments.core.data.evaluation import normalize_prediction, score_prediction
from research_experiments.families.shared.common import resolve_phase_split_name, safe_mean, stable_trace_hash, summarize_row_cost
from research_experiments.core.execution.providers import OpenAICompatibleProvider
from research_experiments.core.execution.rate_limits import SlidingWindowRateLimiter
from research_experiments.core.execution.runner_common import (
    execute_cached_turn,
    prepare_run_root,
    run_indexed_batch,
)
from research_experiments.core.execution.runtime import RunProgressTracker, build_run_id, finalize_run_outputs
from research_experiments.core.controls.selective_signals import normalize_confidence
from research_experiments.core.structured_outputs import (
    ARTIFACT_VERSION,
    SCHEMA_ANSWER_WITH_PROXY_SIGNALS_DELIBERATION,
    SCHEMA_BELIEF_UPDATE_DELTA,
    validate_or_recover_structured_output,
)
from research_experiments.workspace.layout import default_cache_root, default_runs_root
from research_experiments.families.sid_lite.run.metrics import (
    build_diagnostics_payload as build_sid_diagnostics_payload,
    build_metrics_payload as build_sid_metrics_payload,
    write_paper_summary as write_sid_paper_summary,
)
from research_experiments.families.sid_lite.config import SidLiteExperimentConfig, SidLiteProtocolConfig, load_benchmarks, load_protocol_config, phase_metadata
from research_experiments.families.sid_lite.algorithms import (
    apply_belief_update,
    compression_ratio,
    decide_early_exit,
    majority_vote_with_counts,
    project_message_packet,
)
from research_experiments.families.sid_lite.prompts import build_belief_update_messages, build_solver_messages
from research_experiments.families.sid_lite.run.report import render_report, summarize_run
from research_experiments.families.sid_lite.run.validate import validate_run

from research_experiments.families.sid_lite.run.io import _prepare_run_paths
from research_experiments.families.sid_lite.run.sample import (
    _estimate_work,
    _resolve_split_name,
    _run_sample_batch,
)

def run_experiment(
    experiment: SidLiteExperimentConfig,
    phase_name: str,
    backbone: ResolvedModelConfig,
    run_root: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> Path:
    """执行一个 SID-lite phase，并写出完整运行目录。"""
    load_dotenv(".env.local", override=False)
    run_root = run_root or default_runs_root("sid_lite")
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
        "created_at": datetime.now(timezone.utc).isoformat(),
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
        "max_concurrent_requests": experiment.max_concurrent_requests,
        "requests_per_minute_limit": experiment.requests_per_minute_limit,
        "tokens_per_minute_limit": experiment.tokens_per_minute_limit,
        "limitation_note": "SID-lite uses self-reported confidence and structured fields as black-box proxies; logits and attention are unavailable.",
        "total_planned_calls": total_calls,
        "total_planned_predictions": total_predictions,
    }
    run_paths.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    all_stage_a: list[dict[str, Any]] = []
    all_packets: list[dict[str, Any]] = []
    all_beliefs: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []

    try:
        with (
            run_paths.stage_a_turns.open("w", encoding="utf-8") as stage_a_handle,
            run_paths.message_packets.open("w", encoding="utf-8") as packet_handle,
            run_paths.belief_updates.open("w", encoding="utf-8") as belief_handle,
            run_paths.final_predictions.open("w", encoding="utf-8") as prediction_handle,
        ):
            stage_a_writer = BufferedJsonlWriter(stage_a_handle)
            packet_writer = BufferedJsonlWriter(packet_handle)
            belief_writer = BufferedJsonlWriter(belief_handle)
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
                for _, stage_a_rows, packet_rows, belief_rows, prediction_rows in results:
                    for row in stage_a_rows:
                        stage_a_writer.write_row(row)
                        progress.record_call(row, method_key="stage_name")
                    for row in packet_rows:
                        packet_writer.write_row(row)
                    for row in belief_rows:
                        belief_writer.write_row(row)
                        progress.record_call(row, method_key="method_name")
                    for row in prediction_rows:
                        prediction_writer.write_row(row)
                        progress.record_predictions(1, str(row["dataset"]), str(row["method_name"]))
                    all_stage_a.extend(stage_a_rows)
                    all_packets.extend(packet_rows)
                    all_beliefs.extend(belief_rows)
                    all_predictions.extend(prediction_rows)

        metrics = build_sid_metrics_payload(all_predictions)
        diagnostics = build_sid_diagnostics_payload(all_predictions)
        run_paths.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        run_paths.diagnostics.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
        write_sid_paper_summary(run_paths.paper_summary, metrics)
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
