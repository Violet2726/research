"""MADJudge 实验主运行链路。

本模块把 MADJudge 方法组织成完整实验流程，包括：
- 共享样本选择、setup 解析、agent turn 执行
- debate 消息落盘、题级投票聚合
- 成本拆分与最终报告/校验产物生成
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from research_experiments.core.controls.no_comm_controls import run_unified_control_batch
from research_experiments.core.execution.artifacts import BufferedJsonlWriter
from research_experiments.core.execution.cache import RequestCacheRouter
from research_experiments.core.execution.providers import OpenAICompatibleProvider
from research_experiments.core.execution.rate_limits import RequestThrottle
from research_experiments.core.execution.runtime import RunProgressTracker, build_run_id, finalize_run_outputs
from research_experiments.core.structured_outputs import ARTIFACT_VERSION
from research_experiments.families.madjudge.config import (
    MadJudgeExperimentConfig,
    load_control_catalog,
    load_protocol_config,
    load_roster_config,
    phase_metadata,
)
from research_experiments.families.madjudge.run.report import render_report, summarize_run
from research_experiments.families.madjudge.run.sample import (
    _active_setups,
    _build_control_prediction_row,
    _build_cost_breakdown,
    _build_debate_diagnostics,
    _build_metrics,
    _estimate_work,
    _execute_turn,
    _load_selected_samples,
    _resolve_split_name,
    _run_madjudge_batch_round_by_round,
    _write_sample_outputs,
)
from research_experiments.families.madjudge.run.validate import validate_run
from research_experiments.family_runtime.config_helpers import load_benchmarks
from research_experiments.family_runtime.layout import prepare_registered_run_layout
from research_experiments.family_runtime.manifest import finalize_family_manifest
from research_experiments.workspace.layout import default_cache_root, default_runs_root


def run_experiment(
    experiment: MadJudgeExperimentConfig,
    phase_name: str,
    backbone,
    run_root: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> Path:
    """执行一个完整的 MADJudge 实验。"""
    load_dotenv(".env.local", override=False)
    resolved_run_root = Path(run_root) if run_root else default_runs_root("madjudge")
    resolved_cache_root = Path(cache_root) if cache_root else default_cache_root()

    benchmarks = load_benchmarks(experiment)
    phase_metadata(experiment, phase_name)
    setups = _active_setups(experiment, phase_name)

    controls = load_control_catalog(experiment.control_catalog) if experiment.control_catalog else {}
    matched_control_names = sorted({name for setup in setups for name in setup.matched_controls})

    provider = OpenAICompatibleProvider(backbone)
    cache_router = RequestCacheRouter(resolved_cache_root)
    throttle = RequestThrottle.for_model(
        backbone,
        max_concurrent_requests=experiment.max_concurrent_requests,
        requests_per_minute=experiment.requests_per_minute_limit,
    )
    run_id = build_run_id(backbone.name)
    paths = prepare_registered_run_layout("madjudge", resolved_run_root, experiment.name, phase_name, run_id)

    total_calls, total_predictions = _estimate_work(
        experiment, phase_name, benchmarks, setups, matched_control_names, controls,
    )
    progress = RunProgressTracker(
        paths.progress,
        total_calls,
        total_predictions,
        target_network_rpm=experiment.requests_per_minute_limit,
        rate_limit_snapshot_provider=throttle.snapshot,
    )


    # 写入 manifest
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "family_name": "madjudge",
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
    manifest = finalize_family_manifest(manifest, family_name="madjudge")
    (paths.run_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    all_turns: list[dict[str, Any]] = []
    debate_messages: list[dict[str, Any]] = []
    final_predictions: list[dict[str, Any]] = []

    try:
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
                    dataset=benchmark.slug,
                )
                split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
                samples = _load_selected_samples(benchmark, split_name)


                for setup in setups:
                    protocol = load_protocol_config(setup.protocol)
                    roster = load_roster_config(setup.roster)

                    sample_results = _run_madjudge_batch_round_by_round(
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
                        throttle=throttle,
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
                    control_results = run_unified_control_batch(
                        run_id=run_id,
                        samples=samples,
                        control_name=control_name,
                        method=method,
                        benchmark_slug=benchmark.slug,
                        split_name=split_name,
                        backbone=backbone,
                        provider=provider,
                        cache=cache,
                        throttle=throttle,
                        global_seed=experiment.global_seed,
                        max_concurrent_requests=experiment.max_concurrent_requests,
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
        paths.run_summary.write_text(
            json.dumps(summarize_run(paths.root), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        render_report(paths.root)
        finalize_run_outputs(
            paths.root,
            validator=validate_run,
        )
        progress.mark_completed()
        return paths.root
    finally:
        progress.close()
        provider.close()
        cache_router.close()
