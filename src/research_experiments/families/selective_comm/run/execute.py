"""选择性通信实验主运行链路。

本模块围绕“共享前缀”组织整个实验：
先统一运行 Stage A，再为多种 trigger 策略复用同一份中间结果，
只在需要通信时才共享 Stage B，从而在保证对照公平的同时降低总成本。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
from hashlib import sha256
from pathlib import Path
import json
from typing import Any
from typing import Callable

from dotenv import load_dotenv

from research_experiments.core.execution.artifacts import BufferedJsonlWriter
from research_experiments.core.execution.cache import RequestCache, RequestCacheRouter, json_dump
from research_experiments.core.data.datasets import DatasetSample, load_split_ids, select_samples
from research_experiments.core.data.evaluation import aggregate_majority, normalize_prediction, score_prediction
from research_experiments.families.shared.common import build_question_preview, resolve_phase_split_name, safe_mean, safe_ratio, stable_trace_hash
from research_experiments.core.execution.providers import OpenAICompatibleProvider
from research_experiments.core.execution.rate_limits import SlidingWindowRateLimiter
from research_experiments.core.execution.runner_common import (
    execute_cached_turn,
    prepare_run_root,
    prompt_hash as build_prompt_hash,
    run_indexed_batch,
)
from research_experiments.core.execution.runtime import RunProgressTracker, build_run_id, finalize_run_outputs
from research_experiments.core.controls.selective_signals import (
    confidence_display,
    decide_trigger_from_policy,
    normalize_confidence,
    summarize_divergence_rows,
    summarize_confidence_rows,
)
from research_experiments.families.shared.method_catalog import MethodConfig
from research_experiments.families.shared.reference_runs import write_policy_reference_summary
from research_experiments.core.structured_outputs import (
    ARTIFACT_VERSION,
    SCHEMA_ANSWER_WITH_PROXY_SIGNALS_SELECTIVE,
)
from research_experiments.workspace.layout import default_cache_root, default_runs_root
from research_experiments.families.selective_comm.config import (
    SelectiveCommExperimentConfig,
    SharedDebateProtocolConfig,
    TriggerPolicyConfig,
    load_benchmarks,
    load_control_catalog,
    load_policies,
    load_protocol_config,
    phase_metadata,
)
from research_experiments.families.selective_comm.prompts import build_debate_messages, build_initial_messages
from research_experiments.families.selective_comm.run.report import render_report
from research_experiments.families.selective_comm.run.validate import validate_run

from research_experiments.families.selective_comm.run.io import _prepare_run_paths
from research_experiments.families.selective_comm.run.sample import (
    _build_metrics_payload,
    _build_oracle_payload,
    _build_policy_diagnostics,
    _estimate_work,
    _load_resume_seed_state,
    _resolve_split_name,
    _run_sample_batch,
    _write_sample_result,
    _write_seed_rows,
)

def run_experiment(
    experiment: SelectiveCommExperimentConfig,
    phase_name: str,
    backbone,
    run_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    resume_run_dir: str | Path | None = None,
) -> Path:
    """执行一个选择性通信 phase，并写出完整运行目录。"""
    load_dotenv(".env.local", override=False)
    run_root = run_root or default_runs_root("selective_comm")
    cache_root = cache_root or default_cache_root()
    benchmarks = load_benchmarks(experiment)
    protocol = load_protocol_config(experiment.protocol)
    policies = load_policies(experiment.policy_configs)
    controls = load_control_catalog(experiment.control_catalog)
    resume_state = (
        _load_resume_seed_state(Path(resume_run_dir), protocol, policies, controls)
        if resume_run_dir is not None
        else None
    )
    provider = OpenAICompatibleProvider(backbone)
    cache_router = RequestCacheRouter(cache_root)
    limiter = SlidingWindowRateLimiter(
        requests_per_minute=experiment.requests_per_minute_limit,
        tokens_per_minute=experiment.tokens_per_minute_limit,
    )
    run_id = build_run_id(backbone.name)
    run_paths = _prepare_run_paths(run_root, experiment.name, phase_name, run_id)
    total_calls, total_predictions = _estimate_work(experiment, phase_name, benchmarks, protocol, controls)
    progress = RunProgressTracker(
        run_paths.progress,
        total_calls,
        total_predictions,
        initial_completed_calls=resume_state.initial_completed_calls if resume_state is not None else 0,
        initial_completed_predictions=resume_state.initial_completed_predictions if resume_state is not None else 0,
    )

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
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
        "family_name": "selective_comm",
        "experiment_name": experiment.name,
        "phase_name": phase_name,
        "primary_model_ref": experiment.primary_model_ref,
        "resolved_model": asdict(backbone),
        "benchmarks": [asdict(benchmark) for benchmark in benchmarks],
        "total_planned_calls": total_calls,
        "total_planned_predictions": total_predictions,
    }
    if resume_state is not None:
        manifest["resume_source_run_dir"] = resume_state.source_root.as_posix()
        manifest["resume_source_run_id"] = resume_state.source_run_id
        manifest["resume_completed_sample_count"] = len(resume_state.completed_sample_keys)
    run_paths.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    all_stage_a_turns: list[dict[str, Any]] = list(resume_state.stage_a_turns) if resume_state is not None else []
    all_stage_b_turns: list[dict[str, Any]] = list(resume_state.stage_b_turns) if resume_state is not None else []
    all_control_turns: list[dict[str, Any]] = list(resume_state.control_turns) if resume_state is not None else []
    all_trigger_rows: list[dict[str, Any]] = list(resume_state.trigger_rows) if resume_state is not None else []
    all_prediction_rows: list[dict[str, Any]] = list(resume_state.prediction_rows) if resume_state is not None else []

    with (
        run_paths.stage_a_turns.open("w", encoding="utf-8") as stage_a_handle,
        run_paths.stage_b_turns.open("w", encoding="utf-8") as stage_b_handle,
        run_paths.control_turns.open("w", encoding="utf-8") as control_handle,
        run_paths.trigger_decisions.open("w", encoding="utf-8") as trigger_handle,
        run_paths.policy_predictions.open("w", encoding="utf-8") as prediction_handle,
    ):
        stage_a_writer = BufferedJsonlWriter(stage_a_handle)
        stage_b_writer = BufferedJsonlWriter(stage_b_handle)
        control_writer = BufferedJsonlWriter(control_handle)
        trigger_writer = BufferedJsonlWriter(trigger_handle)
        prediction_writer = BufferedJsonlWriter(prediction_handle)
        if resume_state is not None:
            _write_seed_rows(
                resume_state,
                stage_a_handle=stage_a_writer,
                stage_b_handle=stage_b_writer,
                control_handle=control_writer,
                trigger_handle=trigger_writer,
                prediction_handle=prediction_writer,
            )
        for benchmark in benchmarks:
            cache = cache_router.for_request_target(
                provider=backbone.provider,
                request_model=backbone.model_id,
                dataset=benchmark.slug,
            )
            split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
            samples = select_samples(benchmark, split_name)
            if resume_state is not None:
                samples = [
                    sample
                    for sample in samples
                    if (benchmark.slug, sample.sample_id) not in resume_state.completed_sample_keys
                ]
            if not samples:
                continue
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
                    stage_b_handle=stage_b_writer,
                    control_handle=control_writer,
                    trigger_handle=trigger_writer,
                    prediction_handle=prediction_writer,
                    progress=progress,
                    all_stage_a_turns=all_stage_a_turns,
                    all_stage_b_turns=all_stage_b_turns,
                    all_control_turns=all_control_turns,
                    all_trigger_rows=all_trigger_rows,
                    all_prediction_rows=all_prediction_rows,
                ),
            )

    metrics_payload = _build_metrics_payload(all_prediction_rows)
    oracle_payload = _build_oracle_payload(all_prediction_rows)
    diagnostics_payload = _build_policy_diagnostics(all_prediction_rows, oracle_payload, all_stage_a_turns, all_stage_b_turns)
    run_paths.policy_metrics.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    run_paths.oracle_trigger_eval.write_text(json.dumps(oracle_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    run_paths.policy_diagnostics.write_text(json.dumps(diagnostics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_policy_reference_summary(run_paths.root, manifest=manifest, metrics_payload=metrics_payload)
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
