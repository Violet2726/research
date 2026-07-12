"""RCTA-MAD 独立实验编排。"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from research_experiments.core.data.datasets import load_split_ids
from research_experiments.core.execution.artifacts import BufferedJsonlWriter
from research_experiments.core.execution.cache import RequestCacheRouter
from research_experiments.core.execution.providers import OpenAICompatibleProvider
from research_experiments.core.execution.rate_limits import RequestThrottle
from research_experiments.core.execution.runtime import (
    RunProgressTracker,
    build_run_id,
    finalize_run_outputs,
)
from research_experiments.core.structured_outputs import ARTIFACT_VERSION
from research_experiments.families.risk_controlled_trace_mad.config import (
    RctaExperimentConfig,
    load_control_catalog,
    load_protocol_config,
    phase_methods,
    runtime_for_provider,
)
from research_experiments.families.risk_controlled_trace_mad.prompts import (
    RCTA_PROMPT_VERSION,
    RCTA_SCHEMA_VERSION,
)
from research_experiments.families.risk_controlled_trace_mad.router import load_router
from research_experiments.families.risk_controlled_trace_mad.run.metrics import (
    build_diagnostics,
    build_metrics,
    write_paper_summary,
)
from research_experiments.families.risk_controlled_trace_mad.run.report import (
    render_report,
    summarize_run,
)
from research_experiments.families.risk_controlled_trace_mad.run.sample import (
    estimate_work,
    load_selected_samples,
    resolve_split_name,
    run_batch,
)
from research_experiments.families.risk_controlled_trace_mad.run.validate import validate_run
from research_experiments.families.risk_controlled_trace_mad.statistics import paired_statistics
from research_experiments.family_runtime.config_helpers import load_benchmarks, phase_metadata
from research_experiments.family_runtime.layout import prepare_registered_run_layout
from research_experiments.family_runtime.manifest import finalize_family_manifest
from research_experiments.family_runtime.output_protocols import build_shared_output_protocol_diagnostics
from research_experiments.workspace.layout import default_cache_root, default_runs_root


def run_experiment(
    experiment: RctaExperimentConfig,
    phase_name: str,
    backbone,
    run_root: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> Path:
    load_dotenv(".env.local", override=False)
    run_root = run_root or default_runs_root("risk_controlled_trace_mad")
    cache_root = cache_root or default_cache_root()
    phase = phase_metadata(experiment, phase_name)
    benchmarks = load_benchmarks(experiment)
    requested = set(map(str, phase.get("benchmark_slugs") or []))
    if requested:
        benchmarks = [item for item in benchmarks if item.slug in requested]
        missing = requested - {item.slug for item in benchmarks}
        if missing:
            raise ValueError(f"Unknown benchmark(s): {sorted(missing)}")
    protocol = load_protocol_config(experiment.protocol)
    active_methods = phase_methods(experiment, phase_name)
    controls = load_control_catalog(experiment.control_catalog)
    missing_controls = set(experiment.control_methods) - set(controls)
    if missing_controls:
        raise ValueError(f"Unknown controls: {sorted(missing_controls)}")
    router = None
    if protocol.router_mode == "frozen" or "rcta_1" in active_methods and phase_name == "full_seed42":
        router = load_router(protocol.router_artifact, require_passing_gate=phase_name == "full_seed42")
    runtime = runtime_for_provider(experiment, backbone.provider)
    provider = OpenAICompatibleProvider(backbone)
    cache_router = RequestCacheRouter(cache_root)
    throttle = RequestThrottle.for_model(backbone, max_concurrent_requests=runtime.max_concurrent_requests, requests_per_minute=runtime.requests_per_minute_limit)
    run_id = build_run_id(backbone.name)
    paths = prepare_registered_run_layout("risk_controlled_trace_mad", run_root, experiment.name, phase_name, run_id)
    total_calls, total_predictions = estimate_work(experiment, phase_name, benchmarks, active_methods)
    progress = RunProgressTracker(paths.progress, total_calls, total_predictions, planned_calls_are_upper_bound=True, target_network_rpm=runtime.requests_per_minute_limit, rate_limit_snapshot_provider=throttle.snapshot)
    method_order = [*experiment.control_methods, *active_methods]
    manifest = finalize_family_manifest({
        "run_id": run_id, "created_at": datetime.now(UTC).isoformat(), "family_name": "risk_controlled_trace_mad",
        "family_display_name": "RCTA-MAD", "experiment_name": experiment.name, "phase_name": phase_name,
        "experiment": experiment.name, "phase": phase_name, "description": experiment.description,
        "primary_model_ref": experiment.primary_model_ref, "resolved_model": asdict(backbone), "phase_metadata": phase,
        "protocol": asdict(protocol), "control_prompt_version": experiment.control_prompt_version,
        "synthesis_prompt_version": RCTA_PROMPT_VERSION, "synthesis_schema_version": RCTA_SCHEMA_VERSION,
        "artifact_version": ARTIFACT_VERSION, "global_seed": experiment.global_seed,
        "max_concurrent_requests": runtime.max_concurrent_requests, "requests_per_minute_limit": runtime.requests_per_minute_limit,
        "benchmarks": [asdict(item) for item in benchmarks], "dataset_order": [item.slug for item in benchmarks],
        "control_method_names": experiment.control_methods, "rcta_methods": active_methods, "method_order": method_order,
        "max_logical_calls_per_question": 10, "router_artifact_sha256": router.artifact_sha256 if router else None,
        "total_planned_calls": total_calls, "total_planned_predictions": total_predictions,
    }, family_name="risk_controlled_trace_mad")
    paths.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    all_turns, all_messages, all_routers, all_predictions = [], [], [], []
    try:
        with paths.agent_turns.open("w", encoding="utf-8") as turn_handle, paths.debate_messages.open("w", encoding="utf-8") as message_handle, paths.router_decisions.open("w", encoding="utf-8") as router_handle, paths.predictions.open("w", encoding="utf-8") as prediction_handle:
            turn_writer, message_writer, router_writer, prediction_writer = map(BufferedJsonlWriter, (turn_handle, message_handle, router_handle, prediction_handle))
            for benchmark in benchmarks:
                cache = cache_router.for_request_target(provider=backbone.provider, request_model=backbone.model_id, dataset=benchmark.slug)
                split_name = resolve_split_name(experiment, phase_name, benchmark.slug)
                samples = load_selected_samples(benchmark, split_name)
                batch = run_batch(
                    run_id=run_id,
                    dataset=benchmark.slug,
                    split_name=split_name,
                    samples=samples,
                    experiment=experiment,
                    protocol=protocol,
                    active_methods=active_methods,
                    backbone=backbone,
                    provider=provider,
                    cache=cache,
                    throttle=throttle,
                    router=router,
                )
                for _, turns, messages, routers, predictions in batch:
                    for row in turns:
                        turn_writer.write_row(row)
                        progress.record_call(row, method_key="method_name")
                    for row in messages:
                        message_writer.write_row(row)
                    for row in routers:
                        router_writer.write_row(row)
                    for row in predictions:
                        prediction_writer.write_row(row)
                        progress.record_predictions(
                            1,
                            str(row["dataset"]),
                            str(row["method_name"]),
                        )
                    all_turns.extend(turns)
                    all_messages.extend(messages)
                    all_routers.extend(routers)
                    all_predictions.extend(predictions)
        harmonic = phase_name == "full_seed42"
        analysis_predictions = all_predictions
        analysis_mask = {"kind": "all_rows", "excluded_id_count_by_dataset": {}}
        if phase_name == "full_seed42":
            excluded_by_dataset = {
                benchmark.slug: set(load_split_ids(benchmark.cache_namespace or benchmark.slug, "count300_seed42", random_seed=benchmark.random_seed))
                for benchmark in benchmarks
                if benchmark.slug in {"omni_math_2_filtered", "bbeh"}
            }
            analysis_predictions = [
                row for row in all_predictions
                if row.get("dataset") not in excluded_by_dataset or str(row.get("sample_id")) not in excluded_by_dataset[str(row.get("dataset"))]
            ]
            analysis_mask = {"kind": "full_minus_count300_seed42", "excluded_id_count_by_dataset": {key: len(value) for key, value in excluded_by_dataset.items()}}
        metrics = build_metrics(analysis_predictions, dataset_order=[item.slug for item in benchmarks], method_order=method_order, bbeh_harmonic=harmonic)
        metrics["analysis_mask"] = analysis_mask
        if phase_name == "full_seed42":
            metrics["descriptive_full_summary"] = build_metrics(all_predictions, dataset_order=[item.slug for item in benchmarks], method_order=method_order, bbeh_harmonic=harmonic)["summary"]
        diagnostics = build_diagnostics(all_predictions)
        reference = "rcta_1" if "rcta_1" in active_methods else "gsa_trace_1"
        paired = paired_statistics(analysis_predictions, reference=reference, competitors=[name for name in method_order if name != reference], seed=experiment.global_seed, bbeh_harmonic=harmonic)
        protocol_diagnostics = build_shared_output_protocol_diagnostics(all_turns, dataset_order=[item.slug for item in benchmarks], method_order=["rcta_stage_a_shared", "rcta_trace_synthesizer", *method_order])
        paths.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        paths.rcta_diagnostics.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
        paths.paired_statistics.write_text(json.dumps(paired, ensure_ascii=False, indent=2), encoding="utf-8")
        paths.output_protocol_diagnostics.write_text(json.dumps(protocol_diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
        comparison = {"reference_method": reference, "paired_statistics": paired, "claim_scope": "fixed backbone, at most ten logical calls"}
        paths.rcta_comparison.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
        write_paper_summary(paths.paper_summary, metrics)
        render_report(paths.root)
        paths.run_summary.write_text(json.dumps(summarize_run(paths.root), ensure_ascii=False, indent=2), encoding="utf-8")
        finalize_run_outputs(paths.root, validator=validate_run, validation_path=paths.validation)
        progress.mark_completed()
        return paths.root
    finally:
        progress.close()
        provider.close()
        cache_router.close()
