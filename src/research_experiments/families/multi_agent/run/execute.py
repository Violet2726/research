"""多智能体实验主运行链路。

本模块把 Vanilla MAD 及其等预算控制方法组织成完整实验流程，
包括共享样本选择、setup 解析、agent turn 执行、debate 消息落盘、
题级投票聚合、成本拆分与最终报告/校验产物生成。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
import json
from typing import Any

from dotenv import load_dotenv

from research_experiments.core.execution.artifacts import BufferedJsonlWriter
from research_experiments.core.execution.cache import RequestCache, RequestCacheRouter, json_dump
from research_experiments.core.data.datasets import DatasetSample, load_split_ids, select_samples
from research_experiments.core.data.evaluation import aggregate_majority, normalize_prediction, score_prediction
from research_experiments.families.shared.common import resolve_phase_split_name
from research_experiments.core.controls.no_comm_controls import run_no_comm_control_batch
from research_experiments.core.execution.providers import OpenAICompatibleProvider
from research_experiments.core.execution.rate_limits import SlidingWindowRateLimiter
from research_experiments.core.execution.runner_common import (
    execute_cached_turn,
    prepare_run_root,
    run_indexed_batch,
)
from research_experiments.core.execution.runtime import RunProgressTracker, build_run_id, finalize_run_outputs
from research_experiments.core.structured_outputs import ARTIFACT_VERSION, SCHEMA_ANSWER_CORE
from research_experiments.workspace.layout import default_cache_root, default_runs_root
from research_experiments.families.multi_agent.config import (
    ExperimentSetup,
    MultiAgentExperimentConfig,
    ProtocolConfig,
    RosterConfig,
    load_benchmarks,
    load_control_catalog,
    load_protocol_config,
    load_roster_config,
    phase_metadata,
)
from research_experiments.families.multi_agent.prompts import build_debate_messages, build_initial_messages
from research_experiments.families.multi_agent.run.report import render_report, summarize_run
from research_experiments.families.multi_agent.run.validate import validate_run

from research_experiments.families.multi_agent.run.io import _prepare_run_paths
from research_experiments.families.multi_agent.run.sample import (
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

def run_experiment(
    experiment: MultiAgentExperimentConfig,
    phase_name: str,
    backbone,
    run_root: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> Path:
    """执行一个多智能体 phase，并写出完整运行目录。"""
    load_dotenv(".env.local", override=False)
    run_root = run_root or default_runs_root("multi_agent")
    cache_root = cache_root or default_cache_root()
    benchmarks = load_benchmarks(experiment)
    phase = phase_metadata(experiment, phase_name)
    setups = _active_setups(experiment, phase_name)
    controls = load_control_catalog(experiment.control_catalog)
    matched_control_names = sorted({name for setup in setups for name in setup.matched_controls})
    provider = OpenAICompatibleProvider(backbone)
    cache_router = RequestCacheRouter(cache_root)
    limiter = SlidingWindowRateLimiter(
        requests_per_minute=experiment.requests_per_minute_limit,
        tokens_per_minute=experiment.tokens_per_minute_limit,
    )
    run_id = build_run_id(backbone.name)
    run_paths = _prepare_run_paths(run_root, experiment.name, phase_name, run_id)
    total_calls, total_predictions = _estimate_work(experiment, phase_name, benchmarks, setups, matched_control_names, controls)
    progress = RunProgressTracker(run_paths.progress, total_calls, total_predictions)

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "family_name": "multi_agent",
        "experiment_name": experiment.name,
        "phase_name": phase_name,
        "primary_model_ref": experiment.primary_model_ref,
        "resolved_model": asdict(backbone),
        "experiment": experiment.name,
        "description": experiment.description,
        "phase": phase_name,
        "phase_metadata": phase,
        "prompt_version": experiment.prompt_version,
        "artifact_version": ARTIFACT_VERSION,
        "backbone": asdict(backbone),
        "benchmarks": [asdict(item) for item in benchmarks],
        "setups": [
            {
                "name": setup.name,
                "protocol": asdict(load_protocol_config(setup.protocol)),
                "roster": asdict(load_roster_config(setup.roster)),
                "matched_controls": setup.matched_controls,
            }
            for setup in setups
        ],
        "control_methods": {name: asdict(controls[name]) for name in matched_control_names},
        "total_planned_calls": total_calls,
        "total_planned_predictions": total_predictions,
    }
    run_paths.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    all_turns: list[dict[str, Any]] = []
    debate_messages: list[dict[str, Any]] = []
    final_predictions: list[dict[str, Any]] = []

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
                dataset=benchmark.cache_namespace or benchmark.slug,
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
                    limiter=limiter,
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

            for control_name in matched_control_names:
                method = controls[control_name]
                control_results = run_no_comm_control_batch(
                    run_id=run_id,
                    samples=samples,
                    control_name=control_name,
                    method=method,
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

    metrics = _build_metrics(final_predictions, experiment, setups)
    diagnostics = _build_debate_diagnostics(final_predictions)
    cost_breakdown = _build_cost_breakdown(all_turns)

    run_paths.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    run_paths.debate_diagnostics.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
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
