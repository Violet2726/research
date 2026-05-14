"""SPARC 实验主运行链路。

本模块把 SPARC 的多种实验形态落成完整流程：
共享 Stage A、内容压缩/触发选择、Stage B belief update、single judge 或 local auditing，
以及指标、诊断和报告产物生成。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
from hashlib import sha256
from pathlib import Path
import csv
import json
from typing import Any
from typing import Callable

from dotenv import load_dotenv

from research_experiments.core.execution.artifacts import BufferedJsonlWriter
from research_experiments.core.execution.cache import RequestCache, RequestCacheRouter, json_dump
from research_experiments.core.data.datasets import DatasetSample, load_split_ids, select_samples
from research_experiments.core.data.evaluation import aggregate_majority, normalize_prediction, score_prediction
from research_experiments.families.shared.common import build_question_preview, resolve_phase_split_name, safe_mean, stable_trace_hash, sum_metric
from research_experiments.core.execution.providers import OpenAICompatibleProvider, build_payload, execute_completion_request
from research_experiments.core.execution.rate_limits import SlidingWindowRateLimiter
from research_experiments.core.execution.runner_common import (
    execute_cached_turn,
    prepare_run_root,
    run_indexed_batch,
)
from research_experiments.core.execution.runtime import RunProgressTracker, build_run_id, finalize_run_outputs
from research_experiments.core.controls.selective_signals import decide_trigger, normalize_confidence, summarize_confidence_rows
from research_experiments.families.shared.reference_runs import resolve_trigger_reference_selection
from research_experiments.core.structured_outputs import (
    ARTIFACT_VERSION,
    SCHEMA_ANSWER_CORE,
    SCHEMA_ANSWER_WITH_PROXY_SIGNALS_DELIBERATION,
    SCHEMA_AUDIT_VERDICT,
    SCHEMA_BELIEF_UPDATE_DELTA,
)
from research_experiments.workspace.layout import default_cache_root, default_runs_root
from research_experiments.families.sparc.config import (
    SparcExperimentConfig,
    SparcProtocolConfig,
    load_benchmarks,
    load_protocol_config,
    phase_metadata,
)
from research_experiments.families.sparc.algorithms import (
    AGGREGATION_METHOD_ORDER,
    DEFAULT_MESSAGE_MODE_BY_DATASET,
    MESSAGE_MODE_ORDER,
    aggregate_with_confidence_tiebreak,
    aggregate_weighted_vote,
    build_prompt_packet,
    project_message_packet,
    select_audit_candidate_pair,
    apply_belief_update,
)
from research_experiments.families.sparc.prompts import (
    build_audit_messages,
    build_debate_messages,
    build_single_judge_messages,
    build_solver_messages,
)
from research_experiments.families.sparc.run.report import render_report
from research_experiments.families.sparc.run.validate import validate_run

from research_experiments.families.sparc.run.io import _prepare_run_paths
from research_experiments.families.sparc.run.sample import (
    _build_diagnostics,
    _build_metrics,
    _estimate_work,
    _export_paper_summary,
    _resolve_split_name,
    _resolve_trigger_selection,
    _run_sample_batch,
    _write_sample_result,
)

def run_experiment(
    experiment: SparcExperimentConfig,
    phase_name: str,
    backbone,
    run_root: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> Path:
    """执行一个 SPARC phase，并写出完整运行目录。"""
    load_dotenv(".env.local", override=False)
    run_root = run_root or default_runs_root("sparc")
    cache_root = cache_root or default_cache_root()
    benchmarks = load_benchmarks(experiment)
    protocol = load_protocol_config(experiment.protocol)
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
    trigger_selection = _resolve_trigger_selection(experiment, backbone)

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "family_name": "sparc",
        "experiment_name": experiment.name,
        "phase_name": phase_name,
        "primary_model_ref": experiment.primary_model_ref,
        "resolved_model": asdict(backbone),
        "experiment": experiment.name,
        "description": experiment.description,
        "variant_name": experiment.variant_name,
        "phase": phase_name,
        "phase_metadata": phase_metadata(experiment, phase_name),
        "protocol": asdict(protocol),
        "prompt_version": experiment.prompt_version,
        "artifact_version": ARTIFACT_VERSION,
        "global_seed": experiment.global_seed,
        "max_concurrent_requests": experiment.max_concurrent_requests,
        "requests_per_minute_limit": experiment.requests_per_minute_limit,
        "tokens_per_minute_limit": experiment.tokens_per_minute_limit,
        "benchmarks": [asdict(benchmark) for benchmark in benchmarks],
        "message_modes": experiment.message_modes,
        "fixed_message_modes": experiment.fixed_message_modes,
        "aggregation_methods": experiment.aggregation_methods,
        "fixed_trigger_policy": experiment.fixed_trigger_policy,
        "trigger_reference": asdict(experiment.trigger_reference) if experiment.trigger_reference is not None else None,
        "selected_trigger_policy": trigger_selection["selected_policy"],
        "trigger_selection": trigger_selection,
        "total_planned_calls": total_calls,
        "total_planned_predictions": total_predictions,
    }
    run_paths.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    all_stage_a_turns: list[dict[str, Any]] = []
    all_message_packets: list[dict[str, Any]] = []
    all_belief_updates: list[dict[str, Any]] = []
    all_audit_turns: list[dict[str, Any]] = []
    all_prediction_rows: list[dict[str, Any]] = []

    with (
        run_paths.stage_a_turns.open("w", encoding="utf-8") as stage_a_handle,
        run_paths.message_packets.open("w", encoding="utf-8") as message_handle,
        run_paths.belief_updates.open("w", encoding="utf-8") as belief_handle,
        run_paths.audit_turns.open("w", encoding="utf-8") as audit_handle,
        run_paths.final_predictions.open("w", encoding="utf-8") as prediction_handle,
    ):
        stage_a_writer = BufferedJsonlWriter(stage_a_handle)
        message_writer = BufferedJsonlWriter(message_handle)
        belief_writer = BufferedJsonlWriter(belief_handle)
        audit_writer = BufferedJsonlWriter(audit_handle)
        prediction_writer = BufferedJsonlWriter(prediction_handle)
        for benchmark in benchmarks:
            cache = cache_router.for_request_target(
                provider=backbone.provider,
                request_model=backbone.model_id,
                dataset=benchmark.cache_namespace or benchmark.slug,
            )
            split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
            samples = select_samples(benchmark, split_name)
            _run_sample_batch(
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
                trigger_selection=trigger_selection,
                on_complete=partial(
                    _write_sample_result,
                    stage_a_handle=stage_a_writer,
                    message_handle=message_writer,
                    belief_handle=belief_writer,
                    audit_handle=audit_writer,
                    prediction_handle=prediction_writer,
                    progress=progress,
                    all_stage_a_turns=all_stage_a_turns,
                    all_message_packets=all_message_packets,
                    all_belief_updates=all_belief_updates,
                    all_audit_turns=all_audit_turns,
                    all_prediction_rows=all_prediction_rows,
                ),
            )

    metrics_payload = _build_metrics(all_prediction_rows, experiment.variant_name)
    diagnostics_payload = _build_diagnostics(all_prediction_rows, trigger_selection, experiment.variant_name)
    run_paths.metrics.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    run_paths.diagnostics.write_text(json.dumps(diagnostics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _export_paper_summary(run_paths.paper_summary, metrics_payload["summary"])
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
