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

from research_experiments.core.foundation.artifacts import BufferedJsonlWriter
from research_experiments.core.foundation.cache import RequestCache, RequestCacheRouter, json_dump
from research_experiments.core.foundation.datasets import DatasetSample, load_split_ids, select_samples
from research_experiments.core.foundation.evaluation import aggregate_majority, normalize_prediction, score_prediction
from research_experiments.core.foundation.family_helpers import build_question_preview, resolve_phase_split_name, safe_mean, stable_trace_hash, sum_metric
from research_experiments.core.foundation.providers import OpenAICompatibleProvider, build_payload, execute_completion_request
from research_experiments.core.foundation.rate_limits import SlidingWindowRateLimiter
from research_experiments.core.foundation.runner_common import (
    execute_cached_turn,
    prepare_run_root,
    run_indexed_batch,
)
from research_experiments.core.foundation.runtime import RunProgressTracker, build_run_id, finalize_run_outputs
from research_experiments.core.controls.selective_signals import decide_trigger, normalize_confidence, summarize_confidence_rows
from research_experiments.families.reference_runs import resolve_trigger_reference_selection
from research_experiments.core.structured_outputs import (
    ARTIFACT_VERSION,
    SCHEMA_ANSWER_CORE,
    SCHEMA_ANSWER_WITH_PROXY_SIGNALS_DELIBERATION,
    SCHEMA_AUDIT_VERDICT,
    SCHEMA_BELIEF_UPDATE_DELTA,
)
from research_experiments.core.foundation.workspace import default_cache_root, default_runs_root
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


REPORT_NAME_BY_KIND = {
    "content_ablation": "content_ablation_report.md",
    "auditing_ablation": "auditing_ablation_report.md",
    "sparc_v1": "sparc_v1_report.md",
}
E2E_METHOD_ORDER = [
    "mv_3",
    "always_communicate",
    "hybrid_trigger_baseline",
    "final_round_vote_baseline",
    "sparc_v1",
]


@dataclass(frozen=True)
class RunPaths:
    """SPARC 运行目录下的固定产物。"""

    root: Path
    manifest: Path
    stage_a_turns: Path
    message_packets: Path
    belief_updates: Path
    audit_turns: Path
    final_predictions: Path
    metrics: Path
    diagnostics: Path
    progress: Path
    run_validation: Path
    report_markdown: Path
    paper_summary: Path


@dataclass(frozen=True)
class SampleResult:
    """单题运行产物。"""

    stage_a_turns: list[dict[str, Any]]
    message_packets: list[dict[str, Any]]
    belief_updates: list[dict[str, Any]]
    audit_turns: list[dict[str, Any]]
    prediction_rows: list[dict[str, Any]]


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
                dataset=benchmark.slug,
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


def _run_sample_batch(
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    samples: list[DatasetSample],
    protocol: SparcProtocolConfig,
    experiment: SparcExperimentConfig,
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    trigger_selection: dict[str, Any],
    on_complete: Callable[[SampleResult], None] | None = None,
) -> None:
    worker = partial(
        _run_sample,
        run_id=run_id,
        benchmark_slug=benchmark_slug,
        split_name=split_name,
        protocol=protocol,
        experiment=experiment,
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        trigger_selection=trigger_selection,
    )
    for _, result in run_indexed_batch(
        samples,
        worker=worker,
        max_concurrent_requests=experiment.max_concurrent_requests,
    ):
        if on_complete is not None:
            on_complete(result)


def _write_sample_result(
    result: SampleResult,
    stage_a_handle,
    message_handle,
    belief_handle,
    audit_handle,
    prediction_handle,
    progress: RunProgressTracker,
    all_stage_a_turns: list[dict[str, Any]],
    all_message_packets: list[dict[str, Any]],
    all_belief_updates: list[dict[str, Any]],
    all_audit_turns: list[dict[str, Any]],
    all_prediction_rows: list[dict[str, Any]],
) -> None:
    for row in result.stage_a_turns:
        stage_a_handle.write_row(row)
        progress.record_call(row)
    for row in result.message_packets:
        message_handle.write_row(row)
    for row in result.belief_updates:
        belief_handle.write_row(row)
        progress.record_call(row)
    for row in result.audit_turns:
        audit_handle.write_row(row)
        progress.record_call(row)
    for row in result.prediction_rows:
        prediction_handle.write_row(row)
        progress.record_predictions(1, str(row["dataset"]), str(row["method_name"]))
    all_stage_a_turns.extend(result.stage_a_turns)
    all_message_packets.extend(result.message_packets)
    all_belief_updates.extend(result.belief_updates)
    all_audit_turns.extend(result.audit_turns)
    all_prediction_rows.extend(result.prediction_rows)


def _run_sample(
    sample: DatasetSample,
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    protocol: SparcProtocolConfig,
    experiment: SparcExperimentConfig,
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    trigger_selection: dict[str, Any],
) -> SampleResult:
    question_preview = _question_preview(sample.question)
    stage_a_turns = _run_shared_stage_a(
        run_id=run_id,
        benchmark_slug=benchmark_slug,
        split_name=split_name,
        sample=sample,
        protocol=protocol,
        experiment=experiment,
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        question_preview=question_preview,
    )
    stage_a_trace_hash = _trace_hash(
        stage_a_turns,
        keys=["stage_name", "method_name", "round_index", "agent_id", "prompt_hash", "normalized_answer", "confidence_value", "output_status"],
    )
    for row in stage_a_turns:
        row["stage_trace_hash"] = stage_a_trace_hash

    initial_answers = [row["normalized_answer"] for row in stage_a_turns]
    stage_a_vote, stage_a_vote_counts = aggregate_majority(initial_answers)
    stage_a_score = score_prediction(benchmark_slug, stage_a_vote, sample.reference_answer)
    initial_disagreement = len(set(initial_answers)) > 1
    confidence_summary = summarize_confidence_rows(stage_a_turns)

    shared_context = {
        "run_id": run_id,
        "dataset": benchmark_slug,
        "split": split_name,
        "sample": sample,
        "question_preview": question_preview,
        "protocol": protocol,
        "experiment": experiment,
        "backbone": backbone,
        "provider": provider,
        "cache": cache,
        "limiter": limiter,
        "stage_a_turns": stage_a_turns,
        "stage_a_trace_hash": stage_a_trace_hash,
        "stage_a_vote": stage_a_vote,
        "stage_a_vote_counts": stage_a_vote_counts,
        "stage_a_score": stage_a_score,
        "initial_disagreement": initial_disagreement,
        "confidence_summary": confidence_summary,
    }

    if experiment.variant_name == "content_ablation":
        return _run_content_sample(shared_context)
    if experiment.variant_name == "auditing_ablation":
        return _run_auditing_sample(shared_context)
    if experiment.variant_name == "sparc_v1":
        return _run_sparc_sample(shared_context, trigger_selection=trigger_selection)
    raise ValueError(f"Unsupported variant_name: {experiment.variant_name}")


def _run_content_sample(shared_context: dict[str, Any]) -> SampleResult:
    stage_a_turns = shared_context["stage_a_turns"]
    sample = shared_context["sample"]
    benchmark_slug = shared_context["dataset"]
    all_message_rows: list[dict[str, Any]] = []
    all_belief_rows: list[dict[str, Any]] = []
    all_audit_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = [
        _build_prediction_row(
            shared_context,
            method_name="mv_3",
            display_name="mv_3",
            method_kind="baseline",
            prediction=shared_context["stage_a_vote"],
            score=shared_context["stage_a_score"],
            stage_b_prediction=shared_context["stage_a_vote"],
            stage_b_score=shared_context["stage_a_score"],
            message_mode=None,
            trigger_policy="never",
            aggregation_method="majority_vote",
            triggered=False,
            early_exit=True,
            communication_tokens=0.0,
            audit_tokens=0.0,
            prompt_tokens=_sum_metric(stage_a_turns, "prompt_tokens"),
            completion_tokens=_sum_metric(stage_a_turns, "completion_tokens"),
            total_tokens=_sum_metric(stage_a_turns, "total_tokens"),
            latency_ms=_sum_metric(stage_a_turns, "latency_ms"),
            calls_per_question=len(stage_a_turns),
            stage_b_trace_hash_used=None,
            audit_status="not_applicable",
            audit_resolved=False,
            audit_abstained=False,
            wrong_overrule=False,
            minority_rescue=False,
            note="shared_stage_a_vote",
        )
    ]

    mode_rows: dict[str, dict[str, Any]] = {}
    for message_mode in shared_context["experiment"].message_modes:
        message_rows, belief_rows, candidates, stage_b_summary = _run_shared_stage_b(
            shared_context,
            requested_message_mode=message_mode,
        )
        all_message_rows.extend(message_rows)
        all_belief_rows.extend(belief_rows)
        row = _build_prediction_row(
            shared_context,
            method_name=message_mode,
            display_name=message_mode,
            method_kind="message_mode",
            prediction=stage_b_summary["stage_b_vote"],
            score=stage_b_summary["stage_b_score"],
            stage_b_prediction=stage_b_summary["stage_b_vote"],
            stage_b_score=stage_b_summary["stage_b_score"],
            message_mode=message_mode,
            trigger_policy="always_communicate",
            aggregation_method="final_round_vote",
            triggered=True,
            early_exit=False,
            communication_tokens=stage_b_summary["stage_b_total_tokens"],
            audit_tokens=0.0,
            prompt_tokens=_sum_metric(stage_a_turns + belief_rows, "prompt_tokens"),
            completion_tokens=_sum_metric(stage_a_turns + belief_rows, "completion_tokens"),
            total_tokens=_sum_metric(stage_a_turns + belief_rows, "total_tokens"),
            latency_ms=_sum_metric(stage_a_turns + belief_rows, "latency_ms"),
            calls_per_question=len(stage_a_turns) + len(belief_rows),
            stage_b_trace_hash_used=stage_b_summary["stage_b_trace_hash"],
            audit_status="not_applicable",
            audit_resolved=False,
            audit_abstained=False,
            wrong_overrule=False,
            minority_rescue=False,
            note=None,
        )
        mode_rows[message_mode] = row
        prediction_rows.append(row)

    oracle_positive = bool(
        mode_rows.get("full_cot", {}).get("score", shared_context["stage_a_score"]) > shared_context["stage_a_score"]
    )
    for row in prediction_rows:
        row["oracle_positive"] = oracle_positive
    return SampleResult(
        stage_a_turns=stage_a_turns,
        message_packets=all_message_rows,
        belief_updates=all_belief_rows,
        audit_turns=all_audit_rows,
        prediction_rows=prediction_rows,
    )


def _run_auditing_sample(shared_context: dict[str, Any]) -> SampleResult:
    requested_mode = shared_context["experiment"].fixed_message_modes[shared_context["dataset"]]
    message_rows, belief_rows, candidates, stage_b_summary = _run_shared_stage_b(
        shared_context,
        requested_message_mode=requested_mode,
    )
    sample = shared_context["sample"]
    single_judge_rows, single_judge_summary = _run_single_judge(shared_context, candidates)
    local_audit_rows, local_audit_summary = _run_local_auditing(
        shared_context,
        candidates,
        requested_message_mode=requested_mode,
    )
    weighted_vote_prediction, _ = aggregate_weighted_vote(candidates)
    weighted_vote_score = score_prediction(shared_context["dataset"], weighted_vote_prediction, sample.reference_answer)
    oracle_positive = stage_b_summary["stage_b_score"] > shared_context["stage_a_score"]
    prediction_rows = [
        _build_prediction_row(
            shared_context,
            method_name="majority_vote",
            display_name="majority_vote",
            method_kind="aggregation",
            prediction=shared_context["stage_a_vote"],
            score=shared_context["stage_a_score"],
            stage_b_prediction=stage_b_summary["stage_b_vote"],
            stage_b_score=stage_b_summary["stage_b_score"],
            message_mode=requested_mode,
            trigger_policy="always_communicate",
            aggregation_method="majority_vote",
            triggered=False,
            early_exit=True,
            communication_tokens=0.0,
            audit_tokens=0.0,
            prompt_tokens=_sum_metric(shared_context["stage_a_turns"], "prompt_tokens"),
            completion_tokens=_sum_metric(shared_context["stage_a_turns"], "completion_tokens"),
            total_tokens=_sum_metric(shared_context["stage_a_turns"], "total_tokens"),
            latency_ms=_sum_metric(shared_context["stage_a_turns"], "latency_ms"),
            calls_per_question=len(shared_context["stage_a_turns"]),
            stage_b_trace_hash_used=None,
            audit_status="not_applicable",
            audit_resolved=False,
            audit_abstained=False,
            wrong_overrule=False,
            minority_rescue=False,
            note="stage_a_majority_vote",
            oracle_positive=oracle_positive,
        ),
        _build_prediction_row(
            shared_context,
            method_name="weighted_vote_fallback",
            display_name="weighted_vote_fallback",
            method_kind="aggregation",
            prediction=weighted_vote_prediction,
            score=weighted_vote_score,
            stage_b_prediction=stage_b_summary["stage_b_vote"],
            stage_b_score=stage_b_summary["stage_b_score"],
            message_mode=requested_mode,
            trigger_policy="always_communicate",
            aggregation_method="weighted_vote_fallback",
            triggered=True,
            early_exit=False,
            communication_tokens=stage_b_summary["stage_b_total_tokens"],
            audit_tokens=0.0,
            prompt_tokens=_sum_metric(shared_context["stage_a_turns"] + belief_rows, "prompt_tokens"),
            completion_tokens=_sum_metric(shared_context["stage_a_turns"] + belief_rows, "completion_tokens"),
            total_tokens=_sum_metric(shared_context["stage_a_turns"] + belief_rows, "total_tokens"),
            latency_ms=_sum_metric(shared_context["stage_a_turns"] + belief_rows, "latency_ms"),
            calls_per_question=len(shared_context["stage_a_turns"]) + len(belief_rows),
            stage_b_trace_hash_used=stage_b_summary["stage_b_trace_hash"],
            audit_status="not_applicable",
            audit_resolved=False,
            audit_abstained=False,
            wrong_overrule=False,
            minority_rescue=weighted_vote_prediction != shared_context["stage_a_vote"] and weighted_vote_score > shared_context["stage_a_score"],
            note="confidence_weighted_stage_b_vote",
            oracle_positive=oracle_positive,
        ),
        _build_prediction_row(
            shared_context,
            method_name="final_round_vote",
            display_name="final_round_vote",
            method_kind="aggregation",
            prediction=stage_b_summary["stage_b_vote"],
            score=stage_b_summary["stage_b_score"],
            stage_b_prediction=stage_b_summary["stage_b_vote"],
            stage_b_score=stage_b_summary["stage_b_score"],
            message_mode=requested_mode,
            trigger_policy="always_communicate",
            aggregation_method="final_round_vote",
            triggered=True,
            early_exit=False,
            communication_tokens=stage_b_summary["stage_b_total_tokens"],
            audit_tokens=0.0,
            prompt_tokens=_sum_metric(shared_context["stage_a_turns"] + belief_rows, "prompt_tokens"),
            completion_tokens=_sum_metric(shared_context["stage_a_turns"] + belief_rows, "completion_tokens"),
            total_tokens=_sum_metric(shared_context["stage_a_turns"] + belief_rows, "total_tokens"),
            latency_ms=_sum_metric(shared_context["stage_a_turns"] + belief_rows, "latency_ms"),
            calls_per_question=len(shared_context["stage_a_turns"]) + len(belief_rows),
            stage_b_trace_hash_used=stage_b_summary["stage_b_trace_hash"],
            audit_status="not_applicable",
            audit_resolved=False,
            audit_abstained=False,
            wrong_overrule=False,
            minority_rescue=False,
            note=None,
            oracle_positive=oracle_positive,
        ),
        _build_prediction_row(
            shared_context,
            method_name="single_judge",
            display_name="single_judge",
            method_kind="aggregation",
            prediction=single_judge_summary["prediction"],
            score=single_judge_summary["score"],
            stage_b_prediction=stage_b_summary["stage_b_vote"],
            stage_b_score=stage_b_summary["stage_b_score"],
            message_mode=requested_mode,
            trigger_policy="always_communicate",
            aggregation_method="single_judge",
            triggered=True,
            early_exit=False,
            communication_tokens=stage_b_summary["stage_b_total_tokens"],
            audit_tokens=single_judge_summary["audit_tokens"],
            prompt_tokens=_sum_metric(shared_context["stage_a_turns"] + belief_rows + single_judge_rows, "prompt_tokens"),
            completion_tokens=_sum_metric(shared_context["stage_a_turns"] + belief_rows + single_judge_rows, "completion_tokens"),
            total_tokens=_sum_metric(shared_context["stage_a_turns"] + belief_rows + single_judge_rows, "total_tokens"),
            latency_ms=_sum_metric(shared_context["stage_a_turns"] + belief_rows + single_judge_rows, "latency_ms"),
            calls_per_question=len(shared_context["stage_a_turns"]) + len(belief_rows) + len(single_judge_rows),
            stage_b_trace_hash_used=stage_b_summary["stage_b_trace_hash"],
            audit_status="judge",
            audit_resolved=True,
            audit_abstained=False,
            wrong_overrule=single_judge_summary["wrong_overrule"],
            minority_rescue=single_judge_summary["minority_rescue"],
            note=None,
            oracle_positive=oracle_positive,
        ),
        _build_prediction_row(
            shared_context,
            method_name="local_auditing",
            display_name="local_auditing",
            method_kind="aggregation",
            prediction=local_audit_summary["prediction"],
            score=local_audit_summary["score"],
            stage_b_prediction=stage_b_summary["stage_b_vote"],
            stage_b_score=stage_b_summary["stage_b_score"],
            message_mode=requested_mode,
            trigger_policy="always_communicate",
            aggregation_method="local_auditing",
            triggered=True,
            early_exit=False,
            communication_tokens=stage_b_summary["stage_b_total_tokens"],
            audit_tokens=local_audit_summary["audit_tokens"],
            prompt_tokens=_sum_metric(shared_context["stage_a_turns"] + belief_rows + local_audit_rows, "prompt_tokens"),
            completion_tokens=_sum_metric(shared_context["stage_a_turns"] + belief_rows + local_audit_rows, "completion_tokens"),
            total_tokens=_sum_metric(shared_context["stage_a_turns"] + belief_rows + local_audit_rows, "total_tokens"),
            latency_ms=_sum_metric(shared_context["stage_a_turns"] + belief_rows + local_audit_rows, "latency_ms"),
            calls_per_question=len(shared_context["stage_a_turns"]) + len(belief_rows) + len(local_audit_rows),
            stage_b_trace_hash_used=stage_b_summary["stage_b_trace_hash"],
            audit_status=local_audit_summary["audit_status"],
            audit_resolved=local_audit_summary["audit_resolved"],
            audit_abstained=local_audit_summary["audit_abstained"],
            wrong_overrule=local_audit_summary["wrong_overrule"],
            minority_rescue=local_audit_summary["minority_rescue"],
            note=local_audit_summary["note"],
            oracle_positive=oracle_positive,
        ),
    ]
    return SampleResult(
        stage_a_turns=shared_context["stage_a_turns"],
        message_packets=message_rows,
        belief_updates=belief_rows,
        audit_turns=single_judge_rows + local_audit_rows,
        prediction_rows=prediction_rows,
    )


def _run_sparc_sample(shared_context: dict[str, Any], trigger_selection: dict[str, Any]) -> SampleResult:
    requested_mode = shared_context["experiment"].fixed_message_modes[shared_context["dataset"]]
    message_rows, belief_rows, candidates, stage_b_summary = _run_shared_stage_b(
        shared_context,
        requested_message_mode=requested_mode,
    )
    always_audit_rows, always_summary = _run_local_auditing(
        shared_context,
        candidates,
        requested_message_mode=requested_mode,
    )

    confidence_summary = shared_context["confidence_summary"]
    hybrid_decision = decide_trigger(
        trigger_type="hybrid_trigger",
        initial_disagreement=shared_context["initial_disagreement"],
        mean_confidence=confidence_summary.mean_confidence,
        confidence_spread=confidence_summary.confidence_spread,
        any_invalid_confidence=confidence_summary.any_invalid_confidence,
    )
    selected_policy = trigger_selection["selected_policy"]
    selected_decision = decide_trigger(
        trigger_type=selected_policy,
        initial_disagreement=shared_context["initial_disagreement"],
        mean_confidence=confidence_summary.mean_confidence,
        confidence_spread=confidence_summary.confidence_spread,
        any_invalid_confidence=confidence_summary.any_invalid_confidence,
    )
    sparc_audit_rows, sparc_summary = (always_audit_rows, always_summary)
    if not selected_decision.triggered:
        sparc_audit_rows = []
        sparc_summary = {
            "prediction": shared_context["stage_a_vote"],
            "score": shared_context["stage_a_score"],
            "audit_tokens": 0.0,
            "audit_status": "early_exit",
            "audit_resolved": False,
            "audit_abstained": False,
            "wrong_overrule": False,
            "minority_rescue": False,
            "note": selected_decision.decision_reason,
        }
    oracle_positive = stage_b_summary["stage_b_score"] > shared_context["stage_a_score"]
    prediction_rows = [
        _build_prediction_row(
            shared_context,
            method_name="mv_3",
            display_name="mv_3",
            method_kind="baseline",
            prediction=shared_context["stage_a_vote"],
            score=shared_context["stage_a_score"],
            stage_b_prediction=stage_b_summary["stage_b_vote"],
            stage_b_score=stage_b_summary["stage_b_score"],
            message_mode=None,
            trigger_policy="never",
            aggregation_method="majority_vote",
            triggered=False,
            early_exit=True,
            communication_tokens=0.0,
            audit_tokens=0.0,
            prompt_tokens=_sum_metric(shared_context["stage_a_turns"], "prompt_tokens"),
            completion_tokens=_sum_metric(shared_context["stage_a_turns"], "completion_tokens"),
            total_tokens=_sum_metric(shared_context["stage_a_turns"], "total_tokens"),
            latency_ms=_sum_metric(shared_context["stage_a_turns"], "latency_ms"),
            calls_per_question=len(shared_context["stage_a_turns"]),
            stage_b_trace_hash_used=None,
            audit_status="not_applicable",
            audit_resolved=False,
            audit_abstained=False,
            wrong_overrule=False,
            minority_rescue=False,
            note="shared_stage_a_vote",
            oracle_positive=oracle_positive,
            selected_trigger_policy=selected_policy,
        ),
        _build_prediction_row(
            shared_context,
            method_name="always_communicate",
            display_name="always_communicate",
            method_kind="e2e",
            prediction=always_summary["prediction"],
            score=always_summary["score"],
            stage_b_prediction=stage_b_summary["stage_b_vote"],
            stage_b_score=stage_b_summary["stage_b_score"],
            message_mode=requested_mode,
            trigger_policy="always_communicate",
            aggregation_method="local_auditing",
            triggered=True,
            early_exit=False,
            communication_tokens=stage_b_summary["stage_b_total_tokens"],
            audit_tokens=always_summary["audit_tokens"],
            prompt_tokens=_sum_metric(shared_context["stage_a_turns"] + belief_rows + always_audit_rows, "prompt_tokens"),
            completion_tokens=_sum_metric(shared_context["stage_a_turns"] + belief_rows + always_audit_rows, "completion_tokens"),
            total_tokens=_sum_metric(shared_context["stage_a_turns"] + belief_rows + always_audit_rows, "total_tokens"),
            latency_ms=_sum_metric(shared_context["stage_a_turns"] + belief_rows + always_audit_rows, "latency_ms"),
            calls_per_question=len(shared_context["stage_a_turns"]) + len(belief_rows) + len(always_audit_rows),
            stage_b_trace_hash_used=stage_b_summary["stage_b_trace_hash"],
            audit_status=always_summary["audit_status"],
            audit_resolved=always_summary["audit_resolved"],
            audit_abstained=always_summary["audit_abstained"],
            wrong_overrule=always_summary["wrong_overrule"],
            minority_rescue=always_summary["minority_rescue"],
            note=always_summary["note"],
            oracle_positive=oracle_positive,
            selected_trigger_policy=selected_policy,
        ),
        _build_prediction_row(
            shared_context,
            method_name="hybrid_trigger_baseline",
            display_name="hybrid_trigger_baseline",
            method_kind="e2e",
            prediction=stage_b_summary["stage_b_vote"] if hybrid_decision.triggered else shared_context["stage_a_vote"],
            score=stage_b_summary["stage_b_score"] if hybrid_decision.triggered else shared_context["stage_a_score"],
            stage_b_prediction=stage_b_summary["stage_b_vote"],
            stage_b_score=stage_b_summary["stage_b_score"],
            message_mode=requested_mode,
            trigger_policy="hybrid_trigger",
            aggregation_method="final_round_vote",
            triggered=hybrid_decision.triggered,
            early_exit=not hybrid_decision.triggered,
            communication_tokens=stage_b_summary["stage_b_total_tokens"] if hybrid_decision.triggered else 0.0,
            audit_tokens=0.0,
            prompt_tokens=_sum_metric(shared_context["stage_a_turns"], "prompt_tokens") + (_sum_metric(belief_rows, "prompt_tokens") if hybrid_decision.triggered else 0.0),
            completion_tokens=_sum_metric(shared_context["stage_a_turns"], "completion_tokens") + (_sum_metric(belief_rows, "completion_tokens") if hybrid_decision.triggered else 0.0),
            total_tokens=_sum_metric(shared_context["stage_a_turns"], "total_tokens") + (_sum_metric(belief_rows, "total_tokens") if hybrid_decision.triggered else 0.0),
            latency_ms=_sum_metric(shared_context["stage_a_turns"], "latency_ms") + (_sum_metric(belief_rows, "latency_ms") if hybrid_decision.triggered else 0.0),
            calls_per_question=len(shared_context["stage_a_turns"]) + (len(belief_rows) if hybrid_decision.triggered else 0),
            stage_b_trace_hash_used=stage_b_summary["stage_b_trace_hash"] if hybrid_decision.triggered else None,
            audit_status="early_exit" if not hybrid_decision.triggered else "not_applicable",
            audit_resolved=False,
            audit_abstained=False,
            wrong_overrule=False,
            minority_rescue=False,
            note=hybrid_decision.decision_reason,
            oracle_positive=oracle_positive,
            selected_trigger_policy=selected_policy,
        ),
        _build_prediction_row(
            shared_context,
            method_name="final_round_vote_baseline",
            display_name="final_round_vote_baseline",
            method_kind="e2e",
            prediction=stage_b_summary["stage_b_vote"],
            score=stage_b_summary["stage_b_score"],
            stage_b_prediction=stage_b_summary["stage_b_vote"],
            stage_b_score=stage_b_summary["stage_b_score"],
            message_mode=requested_mode,
            trigger_policy="always_communicate",
            aggregation_method="final_round_vote",
            triggered=True,
            early_exit=False,
            communication_tokens=stage_b_summary["stage_b_total_tokens"],
            audit_tokens=0.0,
            prompt_tokens=_sum_metric(shared_context["stage_a_turns"] + belief_rows, "prompt_tokens"),
            completion_tokens=_sum_metric(shared_context["stage_a_turns"] + belief_rows, "completion_tokens"),
            total_tokens=_sum_metric(shared_context["stage_a_turns"] + belief_rows, "total_tokens"),
            latency_ms=_sum_metric(shared_context["stage_a_turns"] + belief_rows, "latency_ms"),
            calls_per_question=len(shared_context["stage_a_turns"]) + len(belief_rows),
            stage_b_trace_hash_used=stage_b_summary["stage_b_trace_hash"],
            audit_status="not_applicable",
            audit_resolved=False,
            audit_abstained=False,
            wrong_overrule=False,
            minority_rescue=False,
            note=None,
            oracle_positive=oracle_positive,
            selected_trigger_policy=selected_policy,
        ),
        _build_prediction_row(
            shared_context,
            method_name="sparc_v1",
            display_name="sparc_v1",
            method_kind="e2e",
            prediction=sparc_summary["prediction"],
            score=sparc_summary["score"],
            stage_b_prediction=stage_b_summary["stage_b_vote"],
            stage_b_score=stage_b_summary["stage_b_score"],
            message_mode=requested_mode,
            trigger_policy=selected_policy,
            aggregation_method="local_auditing",
            triggered=selected_decision.triggered,
            early_exit=not selected_decision.triggered,
            communication_tokens=stage_b_summary["stage_b_total_tokens"] if selected_decision.triggered else 0.0,
            audit_tokens=sparc_summary["audit_tokens"],
            prompt_tokens=_sum_metric(shared_context["stage_a_turns"], "prompt_tokens") + (_sum_metric(belief_rows, "prompt_tokens") if selected_decision.triggered else 0.0) + _sum_metric(sparc_audit_rows, "prompt_tokens"),
            completion_tokens=_sum_metric(shared_context["stage_a_turns"], "completion_tokens") + (_sum_metric(belief_rows, "completion_tokens") if selected_decision.triggered else 0.0) + _sum_metric(sparc_audit_rows, "completion_tokens"),
            total_tokens=_sum_metric(shared_context["stage_a_turns"], "total_tokens") + (_sum_metric(belief_rows, "total_tokens") if selected_decision.triggered else 0.0) + _sum_metric(sparc_audit_rows, "total_tokens"),
            latency_ms=_sum_metric(shared_context["stage_a_turns"], "latency_ms") + (_sum_metric(belief_rows, "latency_ms") if selected_decision.triggered else 0.0) + _sum_metric(sparc_audit_rows, "latency_ms"),
            calls_per_question=len(shared_context["stage_a_turns"]) + (len(belief_rows) if selected_decision.triggered else 0) + len(sparc_audit_rows),
            stage_b_trace_hash_used=stage_b_summary["stage_b_trace_hash"] if selected_decision.triggered else None,
            audit_status=sparc_summary["audit_status"],
            audit_resolved=sparc_summary["audit_resolved"],
            audit_abstained=sparc_summary["audit_abstained"],
            wrong_overrule=sparc_summary["wrong_overrule"],
            minority_rescue=sparc_summary["minority_rescue"],
            note=sparc_summary["note"] or selected_decision.decision_reason,
            oracle_positive=oracle_positive,
            selected_trigger_policy=selected_policy,
        ),
    ]
    return SampleResult(
        stage_a_turns=shared_context["stage_a_turns"],
        message_packets=message_rows,
        belief_updates=belief_rows,
        audit_turns=always_audit_rows + sparc_audit_rows if sparc_audit_rows is not always_audit_rows else always_audit_rows,
        prediction_rows=prediction_rows,
    )


def _run_shared_stage_a(
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    sample: DatasetSample,
    protocol: SparcProtocolConfig,
    experiment: SparcExperimentConfig,
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    question_preview: str,
) -> list[dict[str, Any]]:
    stage_a_turns: list[dict[str, Any]] = []
    for agent_id in range(1, protocol.agent_count + 1):
        messages = build_solver_messages(sample, agent_id, prompt_version=experiment.prompt_version)
        stage_a_turns.append(
            _execute_turn(
                run_id=run_id,
                dataset=benchmark_slug,
                split_name=split_name,
                sample=sample,
                stage_name="stage_a",
                method_name="shared_stage_a",
                role="solver",
                round_index=0,
                agent_id=agent_id,
                visible_peer_count=0,
                messages=messages,
                backbone=backbone,
                provider=provider,
                cache=cache,
                limiter=limiter,
                temperature=protocol.initial_temperature,
                top_p=protocol.top_p,
                max_output_tokens=protocol.max_output_tokens,
                seed=experiment.global_seed + agent_id,
                question_preview=question_preview,
                output_mode=SCHEMA_ANSWER_WITH_PROXY_SIGNALS_DELIBERATION,
            )
        )
    return stage_a_turns


def _run_shared_stage_b(
    shared_context: dict[str, Any],
    *,
    requested_message_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    stage_a_turns = shared_context["stage_a_turns"]
    sample = shared_context["sample"]
    protocol = shared_context["protocol"]
    experiment = shared_context["experiment"]
    message_rows: list[dict[str, Any]] = []
    belief_rows: list[dict[str, Any]] = []
    projected_packets = {
        int(row["agent_id"]): project_message_packet(row, requested_message_mode)
        for row in stage_a_turns
    }
    for recipient_id in range(1, protocol.agent_count + 1):
        previous_packet = projected_packets[recipient_id]
        peer_packets_for_prompt: list[dict[str, object]] = []
        for sender in stage_a_turns:
            if sender["agent_id"] == recipient_id:
                continue
            sender_packet = projected_packets[int(sender["agent_id"])]
            peer_packets_for_prompt.append(
                {
                    "agent": f"agent_{sender['agent_id']}",
                    "effective_message_mode": sender_packet["effective_message_mode"],
                    "packet_text": sender_packet["packet_text"],
                }
            )
            message_rows.append(
                {
                    "run_id": shared_context["run_id"],
                    "dataset": shared_context["dataset"],
                    "split": shared_context["split"],
                    "sample_id": sample.sample_id,
                    "message_mode": requested_message_mode,
                    "requested_message_mode": sender_packet["requested_message_mode"],
                    "effective_message_mode": sender_packet["effective_message_mode"],
                    "degradation_reason": sender_packet["degradation_reason"],
                    "sender_agent_id": sender["agent_id"],
                    "recipient_agent_id": recipient_id,
                    "final_answer": sender_packet["final_answer"],
                    "confidence_raw_display": sender_packet["confidence_raw_display"],
                    "reasoning_trace": sender_packet["reasoning_trace"],
                    "claim_span": sender_packet["claim_span"],
                    "key_evidence": sender_packet["key_evidence"],
                    "approx_packet_tokens": sender_packet["approx_packet_tokens"],
                    "packet_text": sender_packet["packet_text"],
                }
            )
        messages = build_debate_messages(
            sample=sample,
            agent_id=recipient_id,
            round_index=1,
            previous_packet=previous_packet,
            peer_packets=peer_packets_for_prompt,
            requested_message_mode=requested_message_mode,
            prompt_version=experiment.prompt_version,
        )
        belief_row = _execute_turn(
            run_id=shared_context["run_id"],
            dataset=shared_context["dataset"],
            split_name=shared_context["split"],
            sample=sample,
            stage_name="stage_b",
            method_name=f"shared_stage_b::{requested_message_mode}",
            role="belief_update",
            round_index=1,
            agent_id=recipient_id,
            visible_peer_count=len(peer_packets_for_prompt),
            messages=messages,
            backbone=shared_context["backbone"],
            provider=shared_context["provider"],
            cache=shared_context["cache"],
            limiter=shared_context["limiter"],
            temperature=protocol.debate_temperature,
            top_p=protocol.top_p,
            max_output_tokens=protocol.max_output_tokens,
            seed=experiment.global_seed + recipient_id + 100,
            question_preview=shared_context["question_preview"],
            output_mode=SCHEMA_BELIEF_UPDATE_DELTA,
        )
        belief_row["message_mode"] = requested_message_mode
        belief_rows.append(belief_row)

    stage_b_trace_hash = _trace_hash(
        belief_rows,
        keys=["method_name", "round_index", "agent_id", "prompt_hash", "normalized_answer", "output_status"],
    )
    for row in belief_rows:
        row["stage_trace_hash"] = stage_b_trace_hash

    candidates: list[dict[str, Any]] = []
    for stage_a_row, belief_row in zip(stage_a_turns, belief_rows, strict=True):
        candidate = apply_belief_update(
            stage_a_row=stage_a_row,
            message_packet=projected_packets[int(stage_a_row["agent_id"])],
            belief_row=belief_row,
        )
        candidate["normalized_answer"] = normalize_prediction(shared_context["dataset"], str(candidate["final_answer"]))
        candidate["score"] = score_prediction(shared_context["dataset"], candidate["normalized_answer"], sample.reference_answer)
        candidates.append(candidate)

    final_answers = [candidate["normalized_answer"] for candidate in candidates]
    stage_b_vote, stage_b_vote_counts = aggregate_majority(final_answers)
    stage_b_score = score_prediction(shared_context["dataset"], stage_b_vote, sample.reference_answer)
    return (
        message_rows,
        belief_rows,
        candidates,
        {
            "stage_b_vote": stage_b_vote,
            "stage_b_vote_counts": stage_b_vote_counts,
            "stage_b_score": stage_b_score,
            "stage_b_total_tokens": _sum_metric(belief_rows, "total_tokens"),
            "stage_b_trace_hash": stage_b_trace_hash,
        },
    )


def _run_single_judge(
    shared_context: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    messages = build_single_judge_messages(
        shared_context["sample"],
        [build_prompt_packet(candidate) for candidate in candidates],
    )
    judge_row = _execute_turn(
        run_id=shared_context["run_id"],
        dataset=shared_context["dataset"],
        split_name=shared_context["split"],
        sample=shared_context["sample"],
        stage_name="audit",
        method_name="single_judge",
        role="judge",
        round_index=0,
        agent_id=0,
        visible_peer_count=len(candidates),
        messages=messages,
        backbone=shared_context["backbone"],
        provider=shared_context["provider"],
        cache=shared_context["cache"],
        limiter=shared_context["limiter"],
        temperature=0.0,
        top_p=1.0,
        max_output_tokens=shared_context["protocol"].max_output_tokens,
        seed=shared_context["experiment"].global_seed + 999,
        question_preview=shared_context["question_preview"],
        output_mode=SCHEMA_ANSWER_CORE,
    )
    judge_row["input_includes_full_debate"] = False
    prediction = judge_row["normalized_answer"] or shared_context["stage_a_vote"]
    score = score_prediction(shared_context["dataset"], prediction, shared_context["sample"].reference_answer)
    fallback_answer, _ = aggregate_with_confidence_tiebreak(candidates)
    return [judge_row], {
        "prediction": prediction,
        "score": score,
        "audit_tokens": float(judge_row.get("total_tokens") or 0.0),
        "wrong_overrule": prediction != fallback_answer and score < score_prediction(shared_context["dataset"], fallback_answer, shared_context["sample"].reference_answer),
        "minority_rescue": prediction != fallback_answer and score > score_prediction(shared_context["dataset"], fallback_answer, shared_context["sample"].reference_answer),
    }


def _run_local_auditing(
    shared_context: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    requested_message_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pair = select_audit_candidate_pair(candidates)
    fallback_answer, _ = aggregate_weighted_vote(candidates)
    fallback_score = score_prediction(shared_context["dataset"], fallback_answer, shared_context["sample"].reference_answer)
    if pair["skipped"]:
        return [], {
            "prediction": fallback_answer,
            "score": fallback_score,
            "audit_tokens": 0.0,
            "audit_status": "skipped_consensus",
            "audit_resolved": False,
            "audit_abstained": False,
            "wrong_overrule": False,
            "minority_rescue": False,
            "note": "consensus_after_debate",
        }

    messages = build_audit_messages(
        shared_context["sample"],
        build_prompt_packet(pair["candidate_a"]),
        build_prompt_packet(pair["candidate_b"]),
    )
    audit_row = _execute_turn(
        run_id=shared_context["run_id"],
        dataset=shared_context["dataset"],
        split_name=shared_context["split"],
        sample=shared_context["sample"],
        stage_name="audit",
        method_name="local_auditing",
        role="auditor",
        round_index=0,
        agent_id=0,
        visible_peer_count=2,
        messages=messages,
        backbone=shared_context["backbone"],
        provider=shared_context["provider"],
        cache=shared_context["cache"],
        limiter=shared_context["limiter"],
        temperature=0.0,
        top_p=1.0,
        max_output_tokens=shared_context["protocol"].max_output_tokens,
        seed=shared_context["experiment"].global_seed + 1999,
        question_preview=shared_context["question_preview"],
        output_mode=SCHEMA_AUDIT_VERDICT,
    )
    audit_row["input_includes_full_debate"] = False
    audit_row["pair_type"] = pair["pair_type"]
    validated = audit_row.get("validated_output", {}) if audit_row.get("output_status") == "ok" else {}
    decision = validated.get("decision")
    if decision == "resolve_for_a":
        chosen_candidate = pair["candidate_a"]
        prediction = normalize_prediction(
            shared_context["dataset"],
            str(validated.get("verified_answer") or chosen_candidate["final_answer"]),
        )
        audit_status = "resolved"
        audit_resolved = True
        audit_abstained = False
    elif decision == "resolve_for_b":
        chosen_candidate = pair["candidate_b"]
        prediction = normalize_prediction(
            shared_context["dataset"],
            str(validated.get("verified_answer") or chosen_candidate["final_answer"]),
        )
        audit_status = "resolved"
        audit_resolved = True
        audit_abstained = False
    else:
        prediction = fallback_answer
        audit_status = "abstain" if audit_row.get("output_status") == "ok" else "audit_error_fallback"
        audit_resolved = False
        audit_abstained = True
    score = score_prediction(shared_context["dataset"], prediction, shared_context["sample"].reference_answer)
    return [audit_row], {
        "prediction": prediction,
        "score": score,
        "audit_tokens": float(audit_row.get("total_tokens") or 0.0),
        "audit_status": audit_status,
        "audit_resolved": audit_resolved,
        "audit_abstained": audit_abstained,
        "wrong_overrule": prediction != fallback_answer and score < fallback_score,
        "minority_rescue": prediction != fallback_answer and score > fallback_score,
        "note": f"{pair['pair_type']}::{requested_message_mode}",
    }


def _build_prediction_row(
    shared_context: dict[str, Any],
    *,
    method_name: str,
    display_name: str,
    method_kind: str,
    prediction: str,
    score: float,
    stage_b_prediction: str,
    stage_b_score: float,
    message_mode: str | None,
    trigger_policy: str,
    aggregation_method: str,
    triggered: bool,
    early_exit: bool,
    communication_tokens: float,
    audit_tokens: float,
    prompt_tokens: float,
    completion_tokens: float,
    total_tokens: float,
    latency_ms: float,
    calls_per_question: int,
    stage_b_trace_hash_used: str | None,
    audit_status: str,
    audit_resolved: bool,
    audit_abstained: bool,
    wrong_overrule: bool,
    minority_rescue: bool,
    note: str | None,
    oracle_positive: bool | None = None,
    selected_trigger_policy: str | None = None,
    trigger_reason: str | None = None,
    route_value: float | None = None,
    route_cost: float | None = None,
    audit_verdict: str | None = None,
    drift_flag: bool | None = None,
) -> dict[str, Any]:
    if route_value is None:
        route_value = round(float(score) - float(shared_context["stage_a_score"]), 6)
    if route_cost is None:
        route_cost = round(float(communication_tokens) + float(audit_tokens), 6)
    if audit_verdict is None:
        audit_verdict = audit_status
    if trigger_reason is None:
        trigger_reason = note
    if drift_flag is None:
        drift_flag = bool(triggered and not early_exit and float(stage_b_score) < float(shared_context["stage_a_score"]))
    return {
        "run_id": shared_context["run_id"],
        "dataset": shared_context["dataset"],
        "split": shared_context["split"],
        "sample_id": shared_context["sample"].sample_id,
        "question_preview": shared_context["question_preview"],
        "method_name": method_name,
        "display_name": display_name,
        "method_kind": method_kind,
        "model_name": shared_context["backbone"].name,
        "prediction": prediction,
        "gold": shared_context["sample"].reference_answer,
        "score": score,
        "message_mode": message_mode,
        "trigger_policy": trigger_policy,
        "selected_trigger_policy": selected_trigger_policy,
        "aggregation_method": aggregation_method,
        "triggered": triggered,
        "early_exit": early_exit,
        "initial_disagreement": shared_context["initial_disagreement"],
        "oracle_positive": oracle_positive,
        "stage_a_prediction": shared_context["stage_a_vote"],
        "stage_a_score": shared_context["stage_a_score"],
        "stage_b_prediction": stage_b_prediction,
        "stage_b_score": stage_b_score,
        "stage_a_hash": shared_context["stage_a_trace_hash"],
        "stage_a_trace_hash": shared_context["stage_a_trace_hash"],
        "stage_b_trace_hash_used": stage_b_trace_hash_used,
        "prompt_tokens_per_question": prompt_tokens,
        "completion_tokens_per_question": completion_tokens,
        "total_tokens_per_question": total_tokens,
        "communication_tokens_per_question": communication_tokens,
        "audit_tokens_per_question": audit_tokens,
        "route_value": route_value,
        "route_cost": route_cost,
        "latency_ms_per_question": latency_ms,
        "calls_per_question": calls_per_question,
        "trigger_reason": trigger_reason,
        "audit_status": audit_status,
        "audit_verdict": audit_verdict,
        "audit_resolved": audit_resolved,
        "audit_abstained": audit_abstained,
        "abstain_flag": audit_abstained,
        "wrong_overrule": wrong_overrule,
        "minority_rescue": minority_rescue,
        "minority_rescue_flag": minority_rescue,
        "drift_flag": drift_flag,
        "note": note,
    }


def _build_metrics(prediction_rows: list[dict[str, Any]], variant_name: str) -> dict[str, Any]:
    summary: list[dict[str, Any]] = []
    method_order = {
        "content_ablation": ["mv_3", *MESSAGE_MODE_ORDER],
        "auditing_ablation": AGGREGATION_METHOD_ORDER,
        "sparc_v1": E2E_METHOD_ORDER,
    }[variant_name]
    for dataset in sorted({str(row["dataset"]) for row in prediction_rows} | {"overall"}):
        rows_for_dataset = prediction_rows if dataset == "overall" else [row for row in prediction_rows if row["dataset"] == dataset]
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in rows_for_dataset:
            grouped.setdefault((row["model_name"], row["method_name"]), []).append(row)
        for (model_name, method_name), rows in sorted(
            grouped.items(),
            key=lambda item: method_order.index(item[0][1]) if item[0][1] in method_order else 999,
        ):
            total_tokens_mean = _mean(float(row["total_tokens_per_question"]) for row in rows)
            row_payload = {
                "dataset": dataset,
                "model_name": model_name,
                "method_name": method_name,
                "display_name": rows[0]["display_name"],
                "method_kind": rows[0]["method_kind"],
                "question_count": len(rows),
                "accuracy_mean": _mean(float(row["score"]) for row in rows),
                "prompt_tokens_mean": _mean(float(row["prompt_tokens_per_question"]) for row in rows),
                "completion_tokens_mean": _mean(float(row["completion_tokens_per_question"]) for row in rows),
                "total_tokens_mean": total_tokens_mean,
                "communication_tokens_mean": _mean(float(row["communication_tokens_per_question"]) for row in rows),
                "audit_tokens_mean": _mean(float(row["audit_tokens_per_question"]) for row in rows),
                "latency_ms_mean": _mean(float(row["latency_ms_per_question"]) for row in rows),
                "calls_per_question_mean": _mean(float(row["calls_per_question"]) for row in rows),
                "acc_per_1k_tokens": round(_mean(float(row["score"]) for row in rows) / total_tokens_mean * 1000, 6) if total_tokens_mean else 0.0,
                "trigger_rate": _mean(1.0 if row.get("triggered") else 0.0 for row in rows),
                "early_exit_rate": _mean(1.0 if row.get("early_exit") else 0.0 for row in rows),
                "resolve_rate": _mean(1.0 if row.get("audit_resolved") else 0.0 for row in rows),
                "abstain_rate": _mean(1.0 if row.get("audit_abstained") else 0.0 for row in rows),
                "wrong_overrule_rate": _mean(1.0 if row.get("wrong_overrule") else 0.0 for row in rows),
                "minority_rescue_count": sum(1 for row in rows if row.get("minority_rescue")),
            }
            summary.append(row_payload)
    if variant_name == "content_ablation":
        _attach_content_compression_ratios(summary)
    return {"summary": summary}


def _build_diagnostics(
    prediction_rows: list[dict[str, Any]],
    trigger_selection: dict[str, Any],
    variant_name: str,
) -> dict[str, Any]:
    overall_rows = [row for row in _build_metrics(prediction_rows, variant_name)["summary"] if row["dataset"] == "overall"]
    recommendation = None
    if variant_name == "content_ablation":
        candidate_rows = [row for row in overall_rows if row["method_name"] in MESSAGE_MODE_ORDER]
        if candidate_rows:
            recommendation = _best_summary_row(candidate_rows)
    elif variant_name == "auditing_ablation":
        candidate_rows = [row for row in overall_rows if row["method_name"] in AGGREGATION_METHOD_ORDER]
        if candidate_rows:
            recommendation = _best_summary_row(candidate_rows)
    else:
        candidate_rows = [row for row in overall_rows if row["method_name"] in E2E_METHOD_ORDER]
        if candidate_rows:
            recommendation = _best_summary_row(candidate_rows)
    return {
        "variant_name": variant_name,
        "trigger_selection": trigger_selection,
        "recommended_next_default": recommendation,
    }


def _export_paper_summary(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "dataset",
        "model_name",
        "method_name",
        "accuracy_mean",
        "communication_tokens_mean",
        "total_tokens_mean",
        "calls_per_question_mean",
        "acc_per_1k_tokens",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "model_name": row["model_name"],
                    "method_name": row["method_name"],
                    "accuracy_mean": row["accuracy_mean"],
                    "communication_tokens_mean": row["communication_tokens_mean"],
                    "total_tokens_mean": row["total_tokens_mean"],
                    "calls_per_question_mean": row["calls_per_question_mean"],
                    "acc_per_1k_tokens": row["acc_per_1k_tokens"],
                }
            )


def _resolve_trigger_selection(experiment: SparcExperimentConfig, backbone) -> dict[str, Any]:
    if experiment.variant_name != "sparc_v1":
        return {"selected_policy": experiment.fixed_trigger_policy or "always_communicate", "reason": "not_applicable"}
    if experiment.trigger_reference is None:
        return {"selected_policy": "hybrid_trigger", "reason": "trigger_reference_missing"}
    return resolve_trigger_reference_selection(backbone=backbone, reference=experiment.trigger_reference)


def _attach_content_compression_ratios(summary_rows: list[dict[str, Any]]) -> None:
    by_key = {(row["dataset"], row["model_name"], row["method_name"]): row for row in summary_rows}
    for row in summary_rows:
        baseline = by_key.get((row["dataset"], row["model_name"], "full_cot"))
        if baseline and baseline["communication_tokens_mean"]:
            row["compression_ratio_vs_full_cot"] = round(
                1.0 - float(row["communication_tokens_mean"]) / float(baseline["communication_tokens_mean"]),
                6,
            )
        else:
            row["compression_ratio_vs_full_cot"] = None


def _best_summary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        rows,
        key=lambda row: (float(row["accuracy_mean"]), -float(row["total_tokens_mean"])),
    )


def _resolve_split_name(experiment: SparcExperimentConfig, phase_name: str, benchmark_slug: str) -> str:
    return resolve_phase_split_name(experiment, phase_name, benchmark_slug)


def _estimate_work(
    experiment: SparcExperimentConfig,
    phase_name: str,
    benchmarks,
    protocol: SparcProtocolConfig,
) -> tuple[int, int]:
    total_calls = 0
    total_predictions = 0
    for benchmark in benchmarks:
        split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
        sample_count = len(load_split_ids(benchmark.slug, split_name))
        if experiment.variant_name == "content_ablation":
            total_calls += sample_count * protocol.agent_count * (1 + len(experiment.message_modes))
            total_predictions += sample_count * (len(experiment.message_modes) + 1)
        elif experiment.variant_name == "auditing_ablation":
            total_calls += sample_count * (protocol.agent_count * 2 + 2)
            total_predictions += sample_count * len(experiment.aggregation_methods)
        else:
            total_calls += sample_count * (protocol.agent_count * 2 + 1)
            total_predictions += sample_count * len(E2E_METHOD_ORDER)
    return total_calls, total_predictions


def _prepare_run_paths(run_root: str | Path, experiment_name: str, phase_name: str, run_id: str) -> RunPaths:
    root = prepare_run_root(run_root, experiment_name, phase_name, run_id)
    return RunPaths(
        root=root,
        manifest=root / "manifest.json",
        stage_a_turns=root / "stage_a_turns.jsonl",
        message_packets=root / "message_packets.jsonl",
        belief_updates=root / "belief_updates.jsonl",
        audit_turns=root / "audit_turns.jsonl",
        final_predictions=root / "final_predictions.jsonl",
        metrics=root / "metrics.json",
        diagnostics=root / "diagnostics.json",
        progress=root / "progress.json",
        run_validation=root / "run_validation.json",
        report_markdown=root / "report.md",
        paper_summary=root / "paper_summary.csv",
    )


def _trace_hash(rows: list[dict[str, Any]], keys: list[str]) -> str:
    return stable_trace_hash(rows, keys)


def _question_preview(question: str, max_chars: int = 120) -> str:
    return build_question_preview(question, max_chars=max_chars)


def _execute_turn(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    stage_name: str,
    method_name: str,
    role: str,
    round_index: int,
    agent_id: int,
    visible_peer_count: int,
    messages: list[dict[str, str]],
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    seed: int,
    question_preview: str,
    output_mode: str,
) -> dict[str, Any]:
    def _request_executor(
        payload: dict[str, Any],
        active_provider: OpenAICompatibleProvider,
        active_limiter: SlidingWindowRateLimiter | None,
    ) -> dict[str, Any]:
        response_payload = execute_completion_request(
            active_provider,
            payload,
            limiter=active_limiter,
        )
        response_payload["sanitized_fallback_used"] = False
        if response_payload.get("request_error"):
            return _maybe_retry_with_sanitized_messages(
                provider=active_provider,
                limiter=limiter,
                backbone=backbone,
                original_messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_output_tokens=max_output_tokens,
                seed=seed,
                error_message=str(response_payload.get("request_error") or ""),
                error_http_status=response_payload.get("http_status"),
                error_provider_request_id=response_payload.get("provider_request_id"),
            )
        return response_payload

    result = execute_cached_turn(
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        seed=seed,
        schema_id=output_mode,
        request_executor=_request_executor,
    )
    answer_for_normalization = _answer_for_output_mode(result.validated_output, output_mode) if result.validated_output else ""
    row = {
        "run_id": run_id,
        "dataset": dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "question_preview": question_preview,
        "stage_name": stage_name,
        "method_name": method_name,
        "role": role,
        "round_index": round_index,
        "agent_id": agent_id,
        "visible_peer_count": visible_peer_count,
        "prompt_hash": result.prompt_hash,
        "prediction": normalize_prediction(dataset, answer_for_normalization) if answer_for_normalization else "",
        "normalized_answer": normalize_prediction(dataset, answer_for_normalization) if answer_for_normalization else "",
        "output_status": result.output_status,
        "prompt_tokens": float(result.usage.get("prompt_tokens") or 0.0),
        "completion_tokens": float(result.usage.get("completion_tokens") or 0.0),
        "total_tokens": float(result.usage.get("total_tokens") or 0.0),
        "latency_ms": float(result.response_payload.get("latency_ms") or 0.0),
        "cache_hit": result.cache_hit,
        "request_error": result.request_error,
        "payload": result.payload,
        "assistant_text": result.response_payload.get("assistant_text", ""),
        "provider_reasoning_text": result.response_payload.get("provider_reasoning_text", ""),
        "validated_output": result.validated_output,
        "sanitized_fallback_used": bool(result.response_payload.get("sanitized_fallback_used")),
    }
    if output_mode == SCHEMA_ANSWER_WITH_PROXY_SIGNALS_DELIBERATION:
        confidence_raw = result.validated_output.get("confidence_raw") if result.validated_output else None
        confidence_value, confidence_valid, confidence_source = normalize_confidence(confidence_raw)
        row.update(
            {
                "reasoning_trace": result.validated_output.get("reasoning_trace") if result.validated_output else None,
                "reasoning_sketch": result.validated_output.get("reasoning_trace") if result.validated_output else None,
                "claim_span": result.validated_output.get("claim_span") if result.validated_output else None,
                "confidence_raw": confidence_raw,
                "confidence_raw_display": confidence_raw if confidence_raw is not None else "",
                "confidence_value": confidence_value,
                "confidence_valid": confidence_valid,
                "confidence_source": confidence_source,
                "key_evidence": result.validated_output.get("key_evidence") if result.validated_output else None,
                "uncertain_point": result.validated_output.get("uncertain_point") if result.validated_output else None,
            }
        )
    elif output_mode == SCHEMA_BELIEF_UPDATE_DELTA:
        row.update(
            {
                "changed_answer": result.validated_output.get("changed_answer") if result.validated_output else None,
                "new_answer": result.validated_output.get("new_answer") if result.validated_output else None,
                "confidence_delta": result.validated_output.get("confidence_delta") if result.validated_output else None,
                "reason_for_change": result.validated_output.get("reason_for_change") if result.validated_output else None,
                "remaining_disagreement": result.validated_output.get("remaining_disagreement") if result.validated_output else None,
            }
        )
    elif output_mode == SCHEMA_AUDIT_VERDICT:
        row.update(
            {
                "decision": result.validated_output.get("decision") if result.validated_output else None,
                "verified_answer": result.validated_output.get("verified_answer") if result.validated_output else None,
                "rationale": result.validated_output.get("rationale") if result.validated_output else None,
            }
        )
    elif output_mode == SCHEMA_ANSWER_CORE:
        row["reasoning"] = result.validated_output.get("reasoning") if result.validated_output else None
    return row


def _answer_for_output_mode(validated_output: dict[str, Any], output_mode: str) -> str:
    if output_mode == SCHEMA_BELIEF_UPDATE_DELTA:
        return str(validated_output.get("new_answer") or "")
    if output_mode == SCHEMA_AUDIT_VERDICT:
        return str(validated_output.get("verified_answer") or "")
    return str(validated_output.get("final_answer") or "")


def _maybe_retry_with_sanitized_messages(
    *,
    provider: OpenAICompatibleProvider,
    limiter: SlidingWindowRateLimiter,
    backbone,
    original_messages: list[dict[str, str]],
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    seed: int,
    error_message: str,
    error_http_status: int | None,
    error_provider_request_id: str | None,
) -> dict[str, Any]:
    if "data_inspection_failed" not in error_message:
        return {
            "http_status": error_http_status,
            "assistant_text": "",
            "provider_reasoning_text": "",
            "usage_reported": None,
            "usage_estimated": None,
            "latency_ms": 0.0,
            "provider_request_id": error_provider_request_id,
            "request_error": error_message,
            "sanitized_fallback_used": False,
        }
    sanitized_messages = _sanitize_messages_for_provider(original_messages)
    fallback_payload = build_payload(
        config=backbone,
        messages=sanitized_messages,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        seed=seed,
    )
    response_payload = execute_completion_request(
        provider,
        fallback_payload,
        limiter=limiter,
    )
    response_payload["sanitized_fallback_used"] = True
    return response_payload


def _sanitize_messages_for_provider(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    sanitized: list[dict[str, str]] = []
    for message in messages:
        content = str(message.get("content", ""))
        ascii_content = content.encode("ascii", "ignore").decode("ascii")
        ascii_content = " ".join(ascii_content.split())
        sanitized.append(
            {
                "role": str(message.get("role", "user")),
                "content": ascii_content,
            }
        )
    return sanitized


def _sum_metric(rows: list[dict[str, Any]], key: str) -> float:
    return round(sum_metric(rows, key), 6)


def _mean(values) -> float:
    return safe_mean(values)


