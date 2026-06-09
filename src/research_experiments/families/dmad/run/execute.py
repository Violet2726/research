"""DMAD 顶层实验执行入口。"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from research_experiments.core.execution.artifacts import BufferedJsonlWriter
from research_experiments.core.execution.cache import RequestCacheRouter
from research_experiments.core.execution.providers import OpenAICompatibleProvider
from research_experiments.core.execution.rate_limits import RequestThrottle
from research_experiments.core.execution.runtime import RunProgressTracker, build_run_id, finalize_run_outputs
from research_experiments.families.dmad.config import (
    DmadExperimentConfig,
    load_control_catalog,
    load_protocol_config,
    load_roster_config,
)
from research_experiments.families.dmad.run.report import render_report, summarize_run
from research_experiments.families.dmad.run.sample import (
    _active_methods,
    _build_cost_breakdown,
    _build_metrics,
    _build_paper_tables,
    _build_strategy_diagnostics,
    _estimate_work,
    _load_selected_samples,
    _resolve_split_name,
    _run_sample_batch,
    _write_sample_outputs,
)
from research_experiments.families.dmad.run.validate import validate_run
from research_experiments.family_runtime.config_helpers import load_benchmarks
from research_experiments.family_runtime.layout import prepare_registered_run_layout
from research_experiments.family_runtime.manifest import finalize_family_manifest
from research_experiments.workspace.layout import default_cache_root, default_runs_root


def run_experiment(
    experiment: DmadExperimentConfig,
    phase_name: str,
    backbone,
    run_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    splits_root: str | Path = "configs/core/shared/benchmarks/splits",
) -> Path:
    """Run one DMAD phase and write a complete run directory."""

    load_dotenv(".env.local", override=False)
    run_root = run_root or default_runs_root("dmad")
    cache_root = cache_root or default_cache_root()
    benchmarks = load_benchmarks(experiment)
    methods = _active_methods(experiment)
    protocol = load_protocol_config(experiment.protocol)
    controls = load_control_catalog(experiment.control_catalog) if experiment.control_catalog is not None else {}
    rosters = {
        method.name: load_roster_config(method.roster)
        for method in methods
        if method.roster is not None
    }
    provider = OpenAICompatibleProvider(backbone)
    cache_router = RequestCacheRouter(cache_root)
    throttle = RequestThrottle.for_model(
        backbone,
        max_concurrent_requests=experiment.max_concurrent_requests,
        requests_per_minute=experiment.requests_per_minute_limit,
        tokens_per_minute=experiment.tokens_per_minute_limit,
    )
    run_id = build_run_id(backbone.name)
    run_paths = prepare_registered_run_layout('dmad', run_root, experiment.name, phase_name, run_id)
    total_calls, total_predictions = _estimate_work(
        experiment,
        phase_name,
        benchmarks,
        protocol,
        methods,
        rosters,
        controls,
        splits_root,
    )
    progress = RunProgressTracker(
        run_paths.progress,
        total_calls,
        total_predictions,
        target_network_rpm=experiment.requests_per_minute_limit,
        rate_limit_snapshot_provider=throttle.snapshot,
    )

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "family_name": "dmad",
        "experiment_name": experiment.name,
        "phase_name": phase_name,
        "primary_model_ref": experiment.primary_model_ref,
        "resolved_model": asdict(backbone),
        "experiment": experiment.name,
        "description": experiment.description,
        "evaluation_scope": experiment.evaluation_scope,
        "paper_alignment_version": experiment.paper_alignment_version,
        "phase": phase_name,
        "phase_metadata": dict(experiment.raw["phases"][phase_name]),
        "prompt_version": experiment.prompt_version,
        "protocol": asdict(protocol),
        "control_catalog": None if experiment.control_catalog is None else experiment.control_catalog.as_posix(),
        "methods": [
            {
                "name": method.name,
                "mode": method.mode,
                "roster": None if method.roster is None else method.roster.as_posix(),
                "note": method.note,
                "matched_controls": list(method.matched_controls),
                "roster_config": None if method.roster is None else asdict(rosters[method.name]),
            }
            for method in methods
        ],
        "benchmarks": [asdict(item) for item in benchmarks],
        "total_planned_calls": total_calls,
        "total_planned_predictions": total_predictions,
    }
    manifest = finalize_family_manifest(manifest, family_name="dmad")
    run_paths.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    all_turns: list[dict[str, object]] = []
    all_debate_messages: list[dict[str, object]] = []
    final_predictions: list[dict[str, object]] = []

    try:
        with (
            run_paths.agent_turns.open("w", encoding="utf-8") as turn_handle_raw,
            run_paths.debate_messages.open("w", encoding="utf-8") as debate_handle_raw,
            run_paths.final_predictions.open("w", encoding="utf-8") as prediction_handle_raw,
        ):
            turn_handle = BufferedJsonlWriter(turn_handle_raw)
            debate_handle = BufferedJsonlWriter(debate_handle_raw)
            prediction_handle = BufferedJsonlWriter(prediction_handle_raw)
            for benchmark in benchmarks:
                cache = cache_router.for_request_target(
                    provider=backbone.provider,
                    request_model=backbone.model_id,
                    dataset=benchmark.slug,
                )
                split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
                samples = _load_selected_samples(benchmark, split_name, splits_root)
                sample_results = _run_sample_batch(
                    run_id=run_id,
                    benchmark_slug=benchmark.slug,
                    split_name=split_name,
                    samples=samples,
                    protocol=protocol,
                    methods=methods,
                    rosters=rosters,
                    controls=controls,
                    experiment=experiment,
                    backbone=backbone,
                    provider=provider,
                    cache=cache,
                    throttle=throttle,
                )
                _write_sample_outputs(
                    sample_results=sample_results,
                    dataset_slug=benchmark.slug,
                    progress=progress,
                    turn_handle=turn_handle,
                    debate_handle=debate_handle,
                    prediction_handle=prediction_handle,
                    all_turns=all_turns,
                    debate_messages=all_debate_messages,
                    final_predictions=final_predictions,
                )

        metrics = _build_metrics(final_predictions, model_name=backbone.name)
        diagnostics = _build_strategy_diagnostics(final_predictions, evaluation_scope=experiment.evaluation_scope)
        cost_breakdown = _build_cost_breakdown(all_turns)
        paper_tables = _build_paper_tables(final_predictions, evaluation_scope=experiment.evaluation_scope)

        run_paths.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        run_paths.strategy_diagnostics.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
        run_paths.cost_breakdown.write_text(json.dumps(cost_breakdown, ensure_ascii=False, indent=2), encoding="utf-8")
        run_paths.paper_tables.write_text(json.dumps(paper_tables, ensure_ascii=False, indent=2), encoding="utf-8")
        run_paths.run_summary.write_text(json.dumps(summarize_run(run_paths.root), ensure_ascii=False, indent=2), encoding="utf-8")
        render_report(run_paths.root)
        finalize_run_outputs(
            run_paths.root,
            validator=validate_run,
            validation_path=run_paths.run_validation,
        )
        progress.mark_completed()
        return run_paths.root
    finally:
        progress.close()
        provider.close()
        cache_router.close()
