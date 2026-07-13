"""统一 MAD 创新实验的异构编排、恢复和正式产物生成。"""

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
from research_experiments.core.execution.runtime import RunProgressTracker, build_run_id, finalize_run_outputs
from research_experiments.core.structured_outputs import ARTIFACT_VERSION
from research_experiments.families.risk_controlled_trace_mad.config import (
    MadInnovationExperimentConfig,
    load_protocol_config,
    load_version_registry,
    phase_methods,
    require_active_version,
    runtime_for_provider,
)
from research_experiments.families.risk_controlled_trace_mad.prompts import EVF_AUDIT_SCHEMA_VERSION, EVF_PROMPT_VERSION
from research_experiments.families.risk_controlled_trace_mad.run.metrics import (
    build_diagnostics,
    build_metrics,
    evaluate_gate,
    write_paper_summary,
)
from research_experiments.families.risk_controlled_trace_mad.run.report import render_report, summarize_run
from research_experiments.families.risk_controlled_trace_mad.run.sample import (
    ModelEndpoint,
    estimate_work,
    load_selected_samples,
    resolve_split_name,
    run_batch,
)
from research_experiments.families.risk_controlled_trace_mad.run.validate import validate_run
from research_experiments.families.risk_controlled_trace_mad.statistics import paired_statistics
from research_experiments.family_runtime.config_helpers import load_benchmarks, phase_metadata, resolve_model
from research_experiments.family_runtime.layout import prepare_registered_run_layout
from research_experiments.family_runtime.manifest import finalize_family_manifest
from research_experiments.family_runtime.output_protocols import build_shared_output_protocol_diagnostics
from research_experiments.workspace.layout import default_cache_root, default_runs_root

FAMILY_NAME = "risk_controlled_trace_mad"


def run_experiment(
    experiment: MadInnovationExperimentConfig,
    phase_name: str,
    backbone=None,
    run_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    resume_run_dir: str | Path | None = None,
    version: str | None = None,
) -> Path:
    load_dotenv(".env.local", override=False)
    registry = load_version_registry(experiment.version_registry)
    version_record = require_active_version(registry, version)
    protocol = load_protocol_config(experiment.protocol)
    phase = phase_metadata(experiment, phase_name)
    active_methods = phase_methods(experiment, phase_name)
    benchmarks = load_benchmarks(experiment)
    requested = set(map(str, phase.get("benchmark_slugs") or []))
    if requested:
        benchmarks = [benchmark for benchmark in benchmarks if benchmark.slug in requested]
        missing = requested - {benchmark.slug for benchmark in benchmarks}
        if missing:
            raise ValueError(f"Unknown benchmark(s): {sorted(missing)}")

    qwen_model = resolve_model(experiment.qwen_model_ref)
    mimo_model = resolve_model(experiment.mimo_model_ref)
    qwen_runtime = runtime_for_provider(experiment, qwen_model.provider)
    mimo_runtime = runtime_for_provider(experiment, mimo_model.provider)
    qwen_provider = OpenAICompatibleProvider(qwen_model)
    mimo_provider = OpenAICompatibleProvider(mimo_model)
    qwen_throttle = RequestThrottle.for_model(
        qwen_model,
        max_concurrent_requests=qwen_runtime.max_concurrent_requests,
        requests_per_minute=qwen_runtime.requests_per_minute_limit,
    )
    mimo_throttle = RequestThrottle.for_model(
        mimo_model,
        max_concurrent_requests=mimo_runtime.max_concurrent_requests,
        requests_per_minute=mimo_runtime.requests_per_minute_limit,
    )
    cache_router = RequestCacheRouter(cache_root or default_cache_root())
    run_root = Path(run_root or default_runs_root(FAMILY_NAME))
    if resume_run_dir is not None:
        resume = Path(resume_run_dir)
        previous = json.loads((resume / "manifest.json").read_text(encoding="utf-8"))
        if (
            previous.get("experiment_name") != experiment.name
            or previous.get("phase_name") != phase_name
            or previous.get("active_version") != version_record.version_id
        ):
            raise ValueError("Resume run manifest does not match experiment, phase and active version.")
        run_id = resume.name
        paths = prepare_registered_run_layout(FAMILY_NAME, resume.parents[2], experiment.name, phase_name, run_id)
    else:
        run_id = build_run_id("qwen-flash+mimo-v2.5")
        paths = prepare_registered_run_layout(FAMILY_NAME, run_root, experiment.name, phase_name, run_id)

    total_calls, total_predictions = estimate_work(experiment, phase_name, benchmarks, active_methods)
    progress = RunProgressTracker(
        paths.progress,
        total_calls,
        total_predictions,
        planned_calls_are_upper_bound=True,
        target_network_rpm=qwen_runtime.requests_per_minute_limit + mimo_runtime.requests_per_minute_limit,
    )
    manifest = finalize_family_manifest(
        {
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "family_name": FAMILY_NAME,
            "family_display_name": "MAD Innovation / EVF-MAD",
            "experiment_name": experiment.name,
            "phase_name": phase_name,
            "experiment": experiment.name,
            "phase": phase_name,
            "description": experiment.description,
            "active_version": version_record.version_id,
            "version_record": asdict(version_record),
            "phase_metadata": phase,
            "protocol": asdict(protocol),
            "prompt_version": EVF_PROMPT_VERSION,
            "schema_version": EVF_AUDIT_SCHEMA_VERSION,
            "artifact_version": ARTIFACT_VERSION,
            "global_seed": experiment.global_seed,
            "resolved_models": {"qwen": asdict(qwen_model), "mimo": asdict(mimo_model)},
            "runtime_profiles": {"qwen": asdict(qwen_runtime), "mimo": asdict(mimo_runtime)},
            "max_concurrent_requests": qwen_runtime.max_concurrent_requests,
            "requests_per_minute_limit": qwen_runtime.requests_per_minute_limit,
            "benchmarks": [asdict(item) for item in benchmarks],
            "dataset_order": [item.slug for item in benchmarks],
            "method_order": active_methods,
            "methods": active_methods,
            "max_logical_calls_per_question": 10,
            "claim_scope": "fixed Qwen-Flash + MiMo-v2.5 compound system, at most ten logical model calls",
            "total_planned_calls": total_calls,
            "total_planned_predictions": total_predictions,
            "resume_source": str(resume_run_dir) if resume_run_dir is not None else None,
        },
        family_name=FAMILY_NAME,
    )
    paths.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    all_turns: list[dict] = []
    all_predictions: list[dict] = []
    try:
        with (
            paths.agent_turns.open("w", encoding="utf-8") as turn_handle,
            paths.debate_messages.open("w", encoding="utf-8") as message_handle,
            paths.router_decisions.open("w", encoding="utf-8") as decision_handle,
            paths.predictions.open("w", encoding="utf-8") as prediction_handle,
        ):
            turn_writer = BufferedJsonlWriter(turn_handle)
            message_writer = BufferedJsonlWriter(message_handle)
            decision_writer = BufferedJsonlWriter(decision_handle)
            prediction_writer = BufferedJsonlWriter(prediction_handle)
            for benchmark in benchmarks:
                qwen_endpoint = ModelEndpoint(
                    "qwen",
                    qwen_model,
                    qwen_provider,
                    cache_router.for_request_target(
                        provider=qwen_model.provider, request_model=qwen_model.model_id, dataset=benchmark.slug
                    ),
                    qwen_throttle,
                )
                mimo_endpoint = ModelEndpoint(
                    "mimo",
                    mimo_model,
                    mimo_provider,
                    cache_router.for_request_target(
                        provider=mimo_model.provider, request_model=mimo_model.model_id, dataset=benchmark.slug
                    ),
                    mimo_throttle,
                )
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
                    qwen=qwen_endpoint,
                    mimo=mimo_endpoint,
                    max_concurrent_samples=max(
                        1, mimo_runtime.max_concurrent_requests // protocol.stage_mimo_candidates
                    ),
                )
                for _, turns, messages, decisions, predictions in batch:
                    for row in turns:
                        turn_writer.write_row(row)
                        progress.record_call(row, method_key="method_name")
                    for row in messages:
                        message_writer.write_row(row)
                    for row in decisions:
                        decision_writer.write_row(row)
                    for row in predictions:
                        prediction_writer.write_row(row)
                        progress.record_predictions(1, str(row["dataset"]), str(row["method_name"]))
                    all_turns.extend(turns)
                    all_predictions.extend(predictions)

        harmonic = phase_name == "full_seed42"
        analysis_predictions = all_predictions
        analysis_mask = {"kind": "all_rows", "excluded_id_count_by_dataset": {}}
        if harmonic:
            excluded = {
                benchmark.slug: set(
                    load_split_ids(
                        benchmark.cache_namespace or benchmark.slug,
                        "count300_seed42",
                        random_seed=benchmark.random_seed,
                    )
                )
                for benchmark in benchmarks
                if benchmark.slug in {"omni_math_2_filtered", "bbeh"}
            }
            analysis_predictions = [
                row
                for row in all_predictions
                if row.get("dataset") not in excluded
                or str(row.get("sample_id")) not in excluded[str(row.get("dataset"))]
            ]
            analysis_mask = {
                "kind": "full_minus_count300_seed42",
                "excluded_id_count_by_dataset": {key: len(value) for key, value in excluded.items()},
            }
        metrics = build_metrics(
            analysis_predictions,
            dataset_order=[item.slug for item in benchmarks],
            method_order=active_methods,
            bbeh_harmonic=harmonic,
        )
        metrics["analysis_mask"] = analysis_mask
        if harmonic:
            metrics["descriptive_full_summary"] = build_metrics(
                all_predictions,
                dataset_order=[item.slug for item in benchmarks],
                method_order=active_methods,
                bbeh_harmonic=True,
            )["summary"]
        diagnostics = build_diagnostics(all_predictions)
        paired = paired_statistics(
            analysis_predictions,
            reference="evf_mad_1",
            competitors=[name for name in active_methods if name != "evf_mad_1"],
            seed=experiment.global_seed,
            bbeh_harmonic=harmonic,
        )
        protocol_diagnostics = build_shared_output_protocol_diagnostics(
            all_turns,
            dataset_order=[item.slug for item in benchmarks],
            method_order=[
                "evf_stage_a_shared",
                "evf_challenger_selector",
                "evf_symmetric_audit",
                "evf_cross_exam",
                "heterogeneous_gsa_synthesis",
                "mad_5a_r1_update",
            ],
        )
        gate = evaluate_gate(phase_name, all_predictions, paired, protocol.provider_abstention_limit)
        metrics["progression_gate"] = gate
        paths.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        paths.evf_diagnostics.write_text(
            json.dumps({**diagnostics, "progression_gate": gate}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        paths.paired_statistics.write_text(json.dumps(paired, ensure_ascii=False, indent=2), encoding="utf-8")
        paths.output_protocol_diagnostics.write_text(
            json.dumps(protocol_diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        paths.evf_comparison.write_text(
            json.dumps(
                {
                    "reference_method": "evf_mad_1",
                    "paired_statistics": paired,
                    "progression_gate": gate,
                    "claim_scope": manifest["claim_scope"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        write_paper_summary(paths.paper_summary, metrics)
        render_report(paths.root)
        paths.run_summary.write_text(
            json.dumps(summarize_run(paths.root), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        finalize_run_outputs(paths.root, validator=validate_run, validation_path=paths.validation)
        progress.mark_completed()
        return paths.root
    finally:
        progress.close()
        qwen_provider.close()
        mimo_provider.close()
        cache_router.close()


