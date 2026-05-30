"""CUE 实验的主运行链路。"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from research_experiments.core.data.datasets import select_samples
from research_experiments.core.execution.artifacts import BufferedJsonlWriter
from research_experiments.core.execution.cache import RequestCacheRouter
from research_experiments.core.execution.providers import OpenAICompatibleProvider
from research_experiments.core.execution.rate_limits import SlidingWindowRateLimiter
from research_experiments.core.execution.runtime import RunProgressTracker, build_run_id, finalize_run_outputs
from research_experiments.core.structured_outputs import (
    ARTIFACT_VERSION,
)
from research_experiments.families.cue.config import (
    CueExperimentConfig,
    load_control_catalog,
    load_policies,
    load_protocol_config,
)
from research_experiments.families.cue.run.report import render_report
from research_experiments.families.cue.run.sample import (
    _build_metrics_payload,
    _build_oracle_payload,
    _build_policy_diagnostics,
    _estimate_work,
    _resolve_split_name,
    _run_sample_batch,
    _write_sample_result,
)
from research_experiments.families.cue.run.validate import validate_run
from research_experiments.family_runtime.config_helpers import load_benchmarks, phase_metadata
from research_experiments.family_runtime.layout import prepare_registered_run_layout
from research_experiments.family_runtime.manifest import finalize_family_manifest
from research_experiments.workspace.layout import default_cache_root, default_runs_root


def run_experiment(
    experiment: CueExperimentConfig,
    phase_name: str,
    backbone,
    run_root: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> Path:
    """执行一个 CUE 实验 phase，并返回运行目录。"""
    load_dotenv(".env.local", override=False)
    run_root = run_root or default_runs_root("cue")
    cache_root = cache_root or default_cache_root()
    benchmarks = load_benchmarks(experiment)
    protocol = load_protocol_config(experiment.protocol)
    policies = load_policies(experiment.policy_configs)
    controls = load_control_catalog(experiment.control_catalog)
    provider = OpenAICompatibleProvider(backbone)
    cache_router = RequestCacheRouter(cache_root)
    limiter = SlidingWindowRateLimiter(
        requests_per_minute=experiment.requests_per_minute_limit,
        tokens_per_minute=experiment.tokens_per_minute_limit,
    )
    run_id = build_run_id(backbone.name)
    run_paths = prepare_registered_run_layout('cue', run_root, experiment.name, phase_name, run_id)
    total_calls, total_predictions = _estimate_work(experiment, phase_name, benchmarks, protocol, controls, policies)
    progress = RunProgressTracker(run_paths.progress, total_calls, total_predictions)

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "experiment": experiment.name,
        "description": experiment.description,
        "phase": phase_name,
        "phase_metadata": phase_metadata(experiment, phase_name),
        "protocol": asdict(protocol),
        "policies": [asdict(policy) for policy in policies],
        "controls": {name: asdict(control) for name, control in sorted(controls.items())},
        "prompt_version": experiment.prompt_version,
        "artifact_version": ARTIFACT_VERSION,
        "global_seed": experiment.global_seed,
        "max_concurrent_requests": experiment.max_concurrent_requests,
        "requests_per_minute_limit": experiment.requests_per_minute_limit,
        "tokens_per_minute_limit": experiment.tokens_per_minute_limit,
        "family_name": "cue",
        "experiment_name": experiment.name,
        "phase_name": phase_name,
        "primary_model_ref": experiment.primary_model_ref,
        "resolved_model": asdict(backbone),
        "benchmarks": [asdict(benchmark) for benchmark in benchmarks],
        "total_planned_calls": total_calls,
        "total_planned_predictions": total_predictions,
    }
    manifest = finalize_family_manifest(manifest, family_name="cue")
    run_paths.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    all_stage_a_turns: list[dict[str, Any]] = []
    all_communication_turns: list[dict[str, Any]] = []
    all_audit_turns: list[dict[str, Any]] = []
    all_control_turns: list[dict[str, Any]] = []
    all_prediction_rows: list[dict[str, Any]] = []

    with (
        run_paths.stage_a_turns.open("w", encoding="utf-8") as stage_a_handle,
        run_paths.communication_turns.open("w", encoding="utf-8") as communication_handle,
        run_paths.audit_turns.open("w", encoding="utf-8") as audit_handle,
        run_paths.control_turns.open("w", encoding="utf-8") as control_handle,
        run_paths.policy_predictions.open("w", encoding="utf-8") as prediction_handle,
    ):
        stage_a_writer = BufferedJsonlWriter(stage_a_handle)
        communication_writer = BufferedJsonlWriter(communication_handle)
        audit_writer = BufferedJsonlWriter(audit_handle)
        control_writer = BufferedJsonlWriter(control_handle)
        prediction_writer = BufferedJsonlWriter(prediction_handle)
        for benchmark in benchmarks:
            cache = cache_router.for_request_target(
                provider=backbone.provider,
                request_model=backbone.model_id,
                dataset=benchmark.cache_namespace or benchmark.slug,
            )
            split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
            samples = select_samples(benchmark, split_name)
            print(
                f"[cue] start dataset={benchmark.slug} split={split_name} sample_count={len(samples)}",
                flush=True,
            )
            _run_sample_batch(
                run_id=run_id,
                benchmark_slug=benchmark.slug,
                split_name=split_name,
                samples=samples,
                protocol=protocol,
                policies=policies,
                controls=controls,
                experiment=experiment,
                backbone=backbone,
                provider=provider,
                cache=cache,
                limiter=limiter,
                on_complete=partial(
                    _write_sample_result,
                    stage_a_handle=stage_a_writer,
                    communication_handle=communication_writer,
                    audit_handle=audit_writer,
                    control_handle=control_writer,
                    prediction_handle=prediction_writer,
                    progress=progress,
                    all_stage_a_turns=all_stage_a_turns,
                    all_communication_turns=all_communication_turns,
                    all_audit_turns=all_audit_turns,
                    all_control_turns=all_control_turns,
                    all_prediction_rows=all_prediction_rows,
                ),
            )

    metrics_payload = _build_metrics_payload(all_prediction_rows)
    oracle_payload = _build_oracle_payload(all_prediction_rows)
    diagnostics_payload = _build_policy_diagnostics(all_prediction_rows, oracle_payload)
    run_paths.policy_metrics.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    run_paths.oracle_trigger_eval.write_text(json.dumps(oracle_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    run_paths.policy_diagnostics.write_text(json.dumps(diagnostics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
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
