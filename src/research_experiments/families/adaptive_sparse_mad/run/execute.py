"""A-SMAD experiment execution."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from dotenv import load_dotenv

from research_experiments.core.data.datasets import select_samples
from research_experiments.core.execution.artifacts import BufferedJsonlWriter
from research_experiments.core.execution.cache import RequestCacheRouter
from research_experiments.core.execution.providers import OpenAICompatibleProvider
from research_experiments.core.execution.rate_limits import RequestThrottle
from research_experiments.core.execution.runtime import RunProgressTracker, build_run_id, finalize_run_outputs
from research_experiments.core.io import read_json, read_jsonl, write_json, write_jsonl
from research_experiments.core.structured_outputs import ARTIFACT_VERSION
from research_experiments.families.adaptive_sparse_mad.config import (
    ADAPTIVE_POLICY_METHODS,
    AdaptiveSparseMadExperimentConfig,
    load_control_catalog,
    load_protocol_config,
)
from research_experiments.families.adaptive_sparse_mad.run.report import render_report
from research_experiments.families.adaptive_sparse_mad.run.sample import (
    append_sample_result,
    build_metrics_payload,
    build_policy_diagnostics,
    build_router_eval_payload,
    build_stage_a_error_bucket_payload,
    build_stage_a_resolver_breakdown_payload,
    build_stage_a_solver_contribution_payload,
    estimate_work,
    refresh_stage_a_prediction_rows,
    run_sample_batch,
    summarize_run,
)
from research_experiments.families.adaptive_sparse_mad.run.validate import validate_run
from research_experiments.family_runtime.config_helpers import load_benchmarks, phase_metadata
from research_experiments.family_runtime.layout import prepare_registered_run_layout
from research_experiments.family_runtime.manifest import finalize_family_manifest
from research_experiments.workspace.layout import default_cache_root, default_runs_root
from research_experiments.workspace.run_archives import pack_run_artifacts


def run_experiment(
    experiment: AdaptiveSparseMadExperimentConfig,
    phase_name: str,
    backbone,
    run_root: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> Path:
    load_dotenv(".env.local", override=False)
    run_root = run_root or default_runs_root("adaptive_sparse_mad")
    cache_root = cache_root or default_cache_root()
    benchmarks = load_benchmarks(experiment)
    protocol = load_protocol_config(experiment.protocol)
    controls = load_control_catalog(experiment.control_catalog)
    provider = OpenAICompatibleProvider(backbone)
    cache_router = RequestCacheRouter(cache_root)
    throttle = RequestThrottle.for_model(
        backbone,
        max_concurrent_requests=experiment.max_concurrent_requests,
        requests_per_minute=experiment.requests_per_minute_limit,
        tokens_per_minute=experiment.tokens_per_minute_limit,
    )
    run_id = build_run_id(backbone.name)
    run_paths = prepare_registered_run_layout(
        "adaptive_sparse_mad",
        run_root,
        experiment.name,
        phase_name,
        run_id,
    )
    total_calls, total_predictions = estimate_work(experiment, phase_name, benchmarks, controls)
    progress = RunProgressTracker(
        run_paths.progress,
        total_calls,
        total_predictions,
        planned_calls_are_upper_bound=any(
            method_name in ADAPTIVE_POLICY_METHODS for method_name in experiment.aggregate_methods
        ),
        target_network_rpm=experiment.requests_per_minute_limit,
        rate_limit_snapshot_provider=throttle.snapshot,
    )

    manifest = finalize_family_manifest(
        {
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "experiment": experiment.name,
            "description": experiment.description,
            "phase": phase_name,
            "phase_metadata": phase_metadata(experiment, phase_name),
            "protocol": {
                "agent_count": protocol.agent_count,
                "top_p": protocol.top_p,
                "stage_a_temperature": protocol.stage_a_temperature,
                "stage_a_max_output_tokens": protocol.stage_a_max_output_tokens,
                "consensus_confidence_threshold": protocol.consensus_confidence_threshold,
                "majority_confidence_threshold": protocol.majority_confidence_threshold,
                "majority_margin_threshold": protocol.majority_margin_threshold,
            },
            "controls": {name: method.__dict__ for name, method in controls.items()},
            "aggregate_methods": list(experiment.aggregate_methods),
            "max_adaptive_addon_calls": experiment.max_adaptive_addon_calls,
            "prompt_version": experiment.prompt_version,
            "stage_a_prompt_version": experiment.stage_a_prompt_version,
            "adaptive_prompt_version": experiment.adaptive_prompt_version,
            "artifact_version": ARTIFACT_VERSION,
            "global_seed": experiment.global_seed,
            "max_concurrent_requests": experiment.max_concurrent_requests,
            "requests_per_minute_limit": experiment.requests_per_minute_limit,
            "tokens_per_minute_limit": experiment.tokens_per_minute_limit,
            "family_name": "adaptive_sparse_mad",
            "experiment_name": experiment.name,
            "phase_name": phase_name,
            "primary_model_ref": experiment.primary_model_ref,
            "resolved_model": backbone.__dict__,
            "benchmarks": [benchmark.__dict__ for benchmark in benchmarks],
            "total_planned_calls": total_calls,
            "total_planned_predictions": total_predictions,
        },
        family_name="adaptive_sparse_mad",
    )
    run_paths.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    all_stage_a_turns: list[dict[str, object]] = []
    all_control_turns: list[dict[str, object]] = []
    all_router_rows: list[dict[str, object]] = []
    all_prediction_rows: list[dict[str, object]] = []

    try:
        with (
            run_paths.stage_a_turns.open("w", encoding="utf-8") as stage_a_handle,
            run_paths.stage_b_turns.open("w", encoding="utf-8") as _stage_b_handle,
            run_paths.judge_turns.open("w", encoding="utf-8") as _judge_handle,
            run_paths.control_turns.open("w", encoding="utf-8") as control_handle,
            run_paths.router_decisions.open("w", encoding="utf-8") as router_handle,
            run_paths.predictions.open("w", encoding="utf-8") as prediction_handle,
        ):
            stage_a_writer = BufferedJsonlWriter(stage_a_handle)
            control_writer = BufferedJsonlWriter(control_handle)
            router_writer = BufferedJsonlWriter(router_handle)
            prediction_writer = BufferedJsonlWriter(prediction_handle)

            for benchmark in benchmarks:
                cache = cache_router.for_request_target(
                    provider=backbone.provider,
                    request_model=backbone.model_id,
                    dataset=benchmark.slug,
                )
                split_name = phase_metadata(experiment, phase_name)["split_overrides"][benchmark.slug]
                samples = select_samples(benchmark, split_name)
                run_sample_batch(
                    run_id=run_id,
                    benchmark_slug=benchmark.slug,
                    split_name=split_name,
                    samples=samples,
                    protocol=protocol,
                    controls=controls,
                    experiment=experiment,
                    backbone=backbone,
                    provider=provider,
                    cache=cache,
                    throttle=throttle,
                    on_complete=partial(
                        append_sample_result,
                        stage_a_handle=stage_a_writer,
                        control_handle=control_writer,
                        router_handle=router_writer,
                        prediction_handle=prediction_writer,
                        progress=progress,
                        all_stage_a_turns=all_stage_a_turns,
                        all_control_turns=all_control_turns,
                        all_router_rows=all_router_rows,
                        all_prediction_rows=all_prediction_rows,
                    ),
                )

        metrics_payload = build_metrics_payload(list(all_prediction_rows))
        router_eval_payload = build_router_eval_payload(list(all_router_rows))
        diagnostics_payload = build_policy_diagnostics(list(all_prediction_rows), router_eval_payload)
        stage_a_resolver_breakdown = build_stage_a_resolver_breakdown_payload(
            list(all_stage_a_turns),
            list(all_prediction_rows),
        )
        stage_a_error_buckets = build_stage_a_error_bucket_payload(
            list(all_stage_a_turns),
            list(all_prediction_rows),
        )
        stage_a_solver_contributions = build_stage_a_solver_contribution_payload(list(all_stage_a_turns))
        run_paths.metrics.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        run_paths.router_eval.write_text(json.dumps(router_eval_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        run_paths.policy_diagnostics.write_text(json.dumps(diagnostics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        run_paths.diagnostic_path("stage_a_resolver_breakdown.json").write_text(
            json.dumps(stage_a_resolver_breakdown, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        run_paths.diagnostic_path("stage_a_error_buckets.json").write_text(
            json.dumps(stage_a_error_buckets, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        run_paths.diagnostic_path("stage_a_solver_contributions.json").write_text(
            json.dumps(stage_a_solver_contributions, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        run_paths.run_summary.write_text(json.dumps(summarize_run(run_paths.root), ensure_ascii=False, indent=2), encoding="utf-8")
        render_report(run_paths.root)
        finalize_run_outputs(
            run_paths.root,
            validator=validate_run,
            validation_path=run_paths.validation,
        )
        progress.mark_completed()
        return run_paths.root
    finally:
        progress.close()
        provider.close()
        cache_router.close()


def refresh_stage_a_only_run_artifacts(run_dir: str | Path) -> Path:
    root = Path(run_dir)
    manifest = read_json(root / "manifest.json")
    prompt_version = str(manifest.get("stage_a_prompt_version") or manifest.get("prompt_version") or "")
    stage_a_rows = read_jsonl(root / "turns" / "stage_a_turns.jsonl")
    prediction_rows = read_jsonl(root / "views" / "predictions.jsonl")

    refreshed_predictions = refresh_stage_a_prediction_rows(
        stage_a_rows,
        prediction_rows,
        prompt_version=prompt_version,
    )
    metrics_payload = build_metrics_payload(refreshed_predictions)
    router_rows = read_jsonl(root / "turns" / "router_decisions.jsonl")
    router_eval_payload = build_router_eval_payload(router_rows)
    diagnostics_payload = build_policy_diagnostics(refreshed_predictions, router_eval_payload)
    stage_a_resolver_breakdown = build_stage_a_resolver_breakdown_payload(stage_a_rows, refreshed_predictions)
    stage_a_error_buckets = build_stage_a_error_bucket_payload(stage_a_rows, refreshed_predictions)
    stage_a_solver_contributions = build_stage_a_solver_contribution_payload(stage_a_rows)

    write_jsonl(root / "views" / "predictions.jsonl", refreshed_predictions)
    write_json(root / "views" / "metrics.json", metrics_payload)
    write_json(root / "diagnostics" / "router_eval.json", router_eval_payload)
    write_json(root / "diagnostics" / "policy_diagnostics.json", diagnostics_payload)
    write_json(root / "diagnostics" / "stage_a_resolver_breakdown.json", stage_a_resolver_breakdown)
    write_json(root / "diagnostics" / "stage_a_error_buckets.json", stage_a_error_buckets)
    write_json(root / "diagnostics" / "stage_a_solver_contributions.json", stage_a_solver_contributions)
    write_json(root / "views" / "run_summary.json", summarize_run(root))
    render_report(root)
    pack_run_artifacts(root)
    validation = validate_run(root)
    write_json(root / "run_validation.json", validation)
    return root
