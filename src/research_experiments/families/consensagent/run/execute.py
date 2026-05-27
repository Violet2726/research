"""CONSENSAGENT 实验主运行链路。

本模块把 CONSENSAGENT 方法组织成完整实验流程，包括：
- 共享样本选择、setup 解析、agent turn 执行
- debate 消息落盘、题级投票聚合
- 成本拆分与最终报告/校验产物生成
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
from typing import Any

from dotenv import load_dotenv

from research_experiments.core.execution.artifacts import BufferedJsonlWriter
from research_experiments.core.execution.cache import RequestCacheRouter
from research_experiments.core.execution.providers import OpenAICompatibleProvider
from research_experiments.core.execution.rate_limits import SlidingWindowRateLimiter
from research_experiments.core.execution.runtime import RunProgressTracker, build_run_id, finalize_run_outputs
from research_experiments.core.structured_outputs import ARTIFACT_VERSION
from research_experiments.workspace.layout import default_cache_root, default_runs_root
from research_experiments.core.controls.no_comm_controls import run_no_comm_control_batch
from research_experiments.families.consensagent.config import (
    ConsensagentExperimentConfig,
    ExperimentSetup,
    load_benchmarks,
    load_control_catalog,
    load_protocol_config,
    load_roster_config,
    phase_metadata,
)
from research_experiments.families.consensagent.prompts import build_initial_messages
from research_experiments.families.consensagent.run.io import RunPaths
from research_experiments.families.consensagent.run.report import render_report, summarize_run
from research_experiments.families.consensagent.run.validate import validate_run
from research_experiments.families.consensagent.run.sample import (
    _active_setups,
    _build_control_prediction_row,
    _build_cost_breakdown,
    _build_debate_diagnostics,
    _build_metrics,
    _estimate_work,
    _execute_turn,
    _load_selected_samples,
    _resolve_split_name,
    _run_consensagent_sample,
    _run_consensagent_batch,
    _write_sample_outputs,
)


def run_experiment(
    experiment: ConsensagentExperimentConfig,
    phase_name: str,
    backbone,
    run_root: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> Path:
    """执行一个完整的 CONSENSAGENT 实验。"""
    load_dotenv(".env.local", override=False)
    resolved_run_root = Path(run_root) if run_root else default_runs_root("consensagent")
    resolved_cache_root = Path(cache_root) if cache_root else default_cache_root()

    benchmarks = load_benchmarks(experiment)
    phase = phase_metadata(experiment, phase_name)
    setups = _active_setups(experiment, phase_name)

    controls = load_control_catalog(experiment.control_catalog) if experiment.control_catalog else {}
    matched_control_names = sorted({name for setup in setups for name in setup.matched_controls})

    provider = OpenAICompatibleProvider(backbone)
    cache_router = RequestCacheRouter(resolved_cache_root)
    limiter = SlidingWindowRateLimiter(
        requests_per_minute=experiment.requests_per_minute_limit,
        tokens_per_minute=experiment.tokens_per_minute_limit,
    )
    run_id = build_run_id(backbone.name)
    run_root_path = resolved_run_root / experiment.name / phase_name / run_id
    run_root_path.mkdir(parents=True, exist_ok=True)
    paths = RunPaths(run_root=run_root_path)

    total_calls, total_predictions = _estimate_work(
        experiment, phase_name, benchmarks, setups, matched_control_names, controls,
    )
    progress = RunProgressTracker(paths.run_root / "progress.json", total_calls, total_predictions)

    print(f"[CONSENSAGENT] Phase: {phase_name}", flush=True)
    print(f"[CONSENSAGENT] Benchmarks: {[b.slug for b in benchmarks]}", flush=True)
    print(f"[CONSENSAGENT] Setups: {[s.name for s in setups]}", flush=True)
    print(f"[CONSENSAGENT] Estimated calls: {total_calls}, predictions: {total_predictions}", flush=True)

    # 写入 manifest
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "family_name": "consensagent",
        "experiment_name": experiment.name,
        "phase_name": phase_name,
        "primary_model_ref": experiment.primary_model_ref,
        "resolved_model": asdict(backbone),
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
    (paths.run_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    all_turns: list[dict[str, Any]] = []
    debate_messages: list[dict[str, Any]] = []
    final_predictions: list[dict[str, Any]] = []

    with (
        paths.turns_path.open("w", encoding="utf-8") as turn_handle,
        paths.debate_messages_path.open("w", encoding="utf-8") as debate_handle,
        paths.predictions_path.open("w", encoding="utf-8") as prediction_handle,
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

            print(f"[CONSENSAGENT] Running {benchmark.slug} ({len(samples)} samples)...", flush=True)

            for setup in setups:
                protocol = load_protocol_config(setup.protocol)
                roster = load_roster_config(setup.roster)

                sample_results = _run_consensagent_batch(
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
                    sample_results=sample_results,
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
                print(f"[CONSENSAGENT] Running control: {control_name}", flush=True)
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

    paths.metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    paths.debate_diagnostics_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    paths.cost_breakdown_path.write_text(json.dumps(cost_breakdown, ensure_ascii=False, indent=2), encoding="utf-8")
    (paths.run_root / "run_summary.json").write_text(
        json.dumps(summarize_run(paths.run_root), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    render_report(paths.run_root)
    finalize_run_outputs(
        paths.run_root,
        validator=validate_run,
    )
    progress.mark_completed()
    provider.close()
    cache_router.close()
    return paths.run_root
