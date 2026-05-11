"""SID-lite 主运行链路。

本模块把 SID-lite 的完整流程落地为可执行实验：
共享 Stage A、早退判定、压缩消息广播、belief update、题级聚合、
以及指标/诊断/报告产物生成。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
from hashlib import sha256
from pathlib import Path
import json
from typing import Any

from dotenv import load_dotenv

from experiment_core.foundation.artifacts import BufferedJsonlWriter
from experiment_core.foundation.cache import RequestCache, RequestCacheRouter, json_dump
from experiment_core.foundation.config import ResolvedModelConfig
from experiment_core.foundation.datasets import DatasetSample, load_split_ids, select_samples
from experiment_core.foundation.evaluation import normalize_prediction, score_prediction
from experiment_core.foundation.family_helpers import resolve_phase_split_name, safe_mean, stable_trace_hash, summarize_row_cost
from experiment_core.foundation.providers import OpenAICompatibleProvider
from experiment_core.foundation.rate_limits import SlidingWindowRateLimiter
from experiment_core.foundation.runner_common import (
    execute_cached_turn,
    prepare_run_root,
    run_indexed_batch,
)
from experiment_core.foundation.runtime import RunProgressTracker, build_run_id, finalize_run_outputs
from experiment_core.controls.selective_signals import normalize_confidence
from experiment_core.structured_outputs import (
    ARTIFACT_VERSION,
    SCHEMA_ANSWER_WITH_PROXY_SIGNALS_DELIBERATION,
    SCHEMA_BELIEF_UPDATE_DELTA,
    validate_or_recover_structured_output,
)
from experiment_core.foundation.workspace import default_cache_root, default_runs_root
from sid_lite.analytics import (
    build_diagnostics_payload as build_sid_diagnostics_payload,
    build_metrics_payload as build_sid_metrics_payload,
    write_paper_summary as write_sid_paper_summary,
)
from sid_lite.config import SidLiteExperimentConfig, SidLiteProtocolConfig, load_benchmarks, load_protocol_config, phase_metadata
from sid_lite.logic import (
    apply_belief_update,
    compression_ratio,
    decide_early_exit,
    majority_vote_with_counts,
    project_message_packet,
)
from sid_lite.prompting import build_belief_update_messages, build_solver_messages
from sid_lite.reporting import render_report, summarize_run
from sid_lite.validation import validate_run


@dataclass(frozen=True)
class RunPaths:
    """SID-lite 运行目录下的固定产物路径。"""

    root: Path
    manifest: Path
    stage_a_turns: Path
    message_packets: Path
    belief_updates: Path
    final_predictions: Path
    metrics: Path
    diagnostics: Path
    progress: Path
    run_summary: Path
    run_validation: Path
    report_markdown: Path
    paper_summary: Path


def run_experiment(
    experiment: SidLiteExperimentConfig,
    phase_name: str,
    backbone: ResolvedModelConfig,
    run_root: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> Path:
    """执行一个 SID-lite phase，并写出完整运行目录。"""
    load_dotenv(".env.local", override=False)
    run_root = run_root or default_runs_root("sid_lite")
    cache_root = cache_root or default_cache_root()
    protocol = load_protocol_config(experiment.protocol)
    benchmarks = load_benchmarks(experiment)
    phase = phase_metadata(experiment, phase_name)
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

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment.name,
        "description": experiment.description,
        "phase": phase_name,
        "phase_metadata": phase,
        "prompt_version": experiment.prompt_version,
        "artifact_version": ARTIFACT_VERSION,
        "backbone": asdict(backbone),
        "benchmarks": [asdict(item) for item in benchmarks],
        "protocol": asdict(protocol),
        "methods": experiment.methods,
        "max_concurrent_requests": experiment.max_concurrent_requests,
        "requests_per_minute_limit": experiment.requests_per_minute_limit,
        "tokens_per_minute_limit": experiment.tokens_per_minute_limit,
        "limitation_note": "SID-lite uses self-reported confidence and structured fields as black-box proxies; logits and attention are unavailable.",
        "total_planned_calls": total_calls,
        "total_planned_predictions": total_predictions,
    }
    run_paths.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    all_stage_a: list[dict[str, Any]] = []
    all_packets: list[dict[str, Any]] = []
    all_beliefs: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []

    try:
        with (
            run_paths.stage_a_turns.open("w", encoding="utf-8") as stage_a_handle,
            run_paths.message_packets.open("w", encoding="utf-8") as packet_handle,
            run_paths.belief_updates.open("w", encoding="utf-8") as belief_handle,
            run_paths.final_predictions.open("w", encoding="utf-8") as prediction_handle,
        ):
            stage_a_writer = BufferedJsonlWriter(stage_a_handle)
            packet_writer = BufferedJsonlWriter(packet_handle)
            belief_writer = BufferedJsonlWriter(belief_handle)
            prediction_writer = BufferedJsonlWriter(prediction_handle)
            for benchmark in benchmarks:
                cache = cache_router.for_request_target(
                    provider=backbone.provider,
                    request_model=backbone.model_id,
                    dataset=benchmark.slug,
                )
                split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
                samples = select_samples(benchmark, split_name)
                results = _run_sample_batch(
                    run_id=run_id,
                    dataset=benchmark.slug,
                    split_name=split_name,
                    samples=samples,
                    protocol=protocol,
                    backbone=backbone,
                    provider=provider,
                    cache=cache,
                    limiter=limiter,
                    global_seed=experiment.global_seed,
                    prompt_version=experiment.prompt_version,
                    max_concurrent_requests=experiment.max_concurrent_requests,
                )
                for _, stage_a_rows, packet_rows, belief_rows, prediction_rows in results:
                    for row in stage_a_rows:
                        stage_a_writer.write_row(row)
                        progress.record_call(row, method_key="stage_name")
                    for row in packet_rows:
                        packet_writer.write_row(row)
                    for row in belief_rows:
                        belief_writer.write_row(row)
                        progress.record_call(row, method_key="method_name")
                    for row in prediction_rows:
                        prediction_writer.write_row(row)
                        progress.record_predictions(1, str(row["dataset"]), str(row["method_name"]))
                    all_stage_a.extend(stage_a_rows)
                    all_packets.extend(packet_rows)
                    all_beliefs.extend(belief_rows)
                    all_predictions.extend(prediction_rows)

        metrics = build_sid_metrics_payload(all_predictions)
        diagnostics = build_sid_diagnostics_payload(all_predictions)
        run_paths.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        run_paths.diagnostics.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
        write_sid_paper_summary(run_paths.paper_summary, metrics)
        render_report(run_paths.root)
        run_paths.run_summary.write_text(json.dumps(summarize_run(run_paths.root), ensure_ascii=False, indent=2), encoding="utf-8")
        finalize_run_outputs(
            run_paths.root,
            validator=validate_run,
            validation_path=run_paths.run_validation,
        )
        progress.mark_completed()
        return run_paths.root
    finally:
        provider.close()
        cache_router.close()


def _run_sample_batch(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    samples: list[DatasetSample],
    protocol: SidLiteProtocolConfig,
    backbone: ResolvedModelConfig,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    global_seed: int,
    prompt_version: str,
    max_concurrent_requests: int,
) -> list[tuple[int, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]]:
    """样本级并发执行；单题内部保持 Stage A -> Stage B 顺序。"""
    worker = partial(
        _run_sample,
        run_id=run_id,
        dataset=dataset,
        split_name=split_name,
        protocol=protocol,
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        global_seed=global_seed,
        prompt_version=prompt_version,
    )
    return [
        (sample_index, *result)
        for sample_index, result in run_indexed_batch(
            samples,
            worker=worker,
            max_concurrent_requests=max_concurrent_requests,
        )
    ]


def _run_sample(
    sample: DatasetSample,
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    protocol: SidLiteProtocolConfig,
    backbone: ResolvedModelConfig,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    global_seed: int,
    prompt_version: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    stage_a_rows = []
    for agent_id in range(1, protocol.agent_count + 1):
        stage_a_rows.append(
            _execute_turn(
                run_id=run_id,
                dataset=dataset,
                split_name=split_name,
                sample=sample,
                stage_name="stage_a",
                method_name="shared_stage_a",
                round_index=0,
                agent_id=agent_id,
                role="solver",
                messages=build_solver_messages(sample, agent_id, prompt_version=prompt_version),
                output_mode="solver",
                backbone=backbone,
                provider=provider,
                cache=cache,
                limiter=limiter,
                temperature=protocol.initial_temperature,
                top_p=protocol.top_p,
                max_output_tokens=protocol.max_output_tokens,
                seed=global_seed + agent_id,
            )
        )
    stage_a_trace_hash = _trace_hash(stage_a_rows, ["agent_id", "normalized_answer", "validated_output"])
    for row in stage_a_rows:
        row["stage_a_trace_hash"] = stage_a_trace_hash

    full_packets = [
        _enrich_packet(run_id, dataset, split_name, sample.sample_id, "full", stage_a_trace_hash, project_message_packet(row, mode="full", token_cap=protocol.full_token_cap))
        for row in stage_a_rows
    ]
    compressed_packets = [
        _enrich_packet(run_id, dataset, split_name, sample.sample_id, "compressed", stage_a_trace_hash, project_message_packet(row, mode="compressed", token_cap=protocol.compressed_token_cap))
        for row in stage_a_rows
    ]
    full_beliefs = _run_belief_updates(
        run_id=run_id,
        dataset=dataset,
        split_name=split_name,
        sample=sample,
        packet_mode="full",
        packets=full_packets,
        protocol=protocol,
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        global_seed=global_seed,
        prompt_version=prompt_version,
    )
    compressed_beliefs = _run_belief_updates(
        run_id=run_id,
        dataset=dataset,
        split_name=split_name,
        sample=sample,
        packet_mode="compressed",
        packets=compressed_packets,
        protocol=protocol,
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        global_seed=global_seed,
        prompt_version=prompt_version,
    )
    packet_rows = full_packets + compressed_packets
    belief_rows = full_beliefs + compressed_beliefs
    predictions = _build_prediction_rows(
        run_id=run_id,
        dataset=dataset,
        split_name=split_name,
        sample=sample,
        backbone=backbone,
        stage_a_rows=stage_a_rows,
        full_packets=full_packets,
        compressed_packets=compressed_packets,
        full_beliefs=full_beliefs,
        compressed_beliefs=compressed_beliefs,
        protocol=protocol,
    )
    return stage_a_rows, packet_rows, belief_rows, predictions


def _run_belief_updates(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    packet_mode: str,
    packets: list[dict[str, Any]],
    protocol: SidLiteProtocolConfig,
    backbone: ResolvedModelConfig,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    global_seed: int,
    prompt_version: str,
) -> list[dict[str, Any]]:
    packet_by_agent = {int(packet["agent_id"]): packet for packet in packets}
    rows = []
    for recipient_id in range(1, protocol.agent_count + 1):
        peer_packets = [
            {
                "agent": f"agent_{packet['agent_id']}",
                "packet_mode": packet["packet_mode"],
                "packet_text": packet["packet_text"],
            }
            for packet in packets
            if int(packet["agent_id"]) != recipient_id
        ]
        rows.append(
            _execute_turn(
                run_id=run_id,
                dataset=dataset,
                split_name=split_name,
                sample=sample,
                stage_name="stage_b",
                method_name=f"shared_stage_b::{packet_mode}",
                round_index=1,
                agent_id=recipient_id,
                role="belief_update",
                messages=build_belief_update_messages(
                    sample,
                    recipient_id,
                    previous_packet=packet_by_agent[recipient_id],
                    peer_packets=peer_packets,
                    packet_mode=packet_mode,
                    prompt_version=prompt_version,
                ),
                output_mode="belief",
                backbone=backbone,
                provider=provider,
                cache=cache,
                limiter=limiter,
                temperature=protocol.debate_temperature,
                top_p=protocol.top_p,
                max_output_tokens=protocol.max_output_tokens,
                seed=global_seed + recipient_id + (1000 if packet_mode == "full" else 2000),
                extra_fields={"packet_mode": packet_mode, "visible_peer_count": len(peer_packets)},
            )
        )
    stage_b_trace_hash = _trace_hash(rows, ["agent_id", "validated_output", "packet_mode"])
    for row in rows:
        row["stage_b_trace_hash"] = stage_b_trace_hash
    return rows


def _build_prediction_rows(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    backbone: ResolvedModelConfig,
    stage_a_rows: list[dict[str, Any]],
    full_packets: list[dict[str, Any]],
    compressed_packets: list[dict[str, Any]],
    full_beliefs: list[dict[str, Any]],
    compressed_beliefs: list[dict[str, Any]],
    protocol: SidLiteProtocolConfig,
) -> list[dict[str, Any]]:
    stage_a_answers = [str(row.get("normalized_answer") or "") for row in stage_a_rows]
    stage_a_vote, stage_a_counts, stage_a_consensus = majority_vote_with_counts(stage_a_answers)
    stage_a_score = score_prediction(dataset, stage_a_vote, sample.reference_answer)
    full_candidates = [apply_belief_update(stage_a, belief) for stage_a, belief in zip(stage_a_rows, full_beliefs, strict=True)]
    compressed_candidates = [apply_belief_update(stage_a, belief) for stage_a, belief in zip(stage_a_rows, compressed_beliefs, strict=True)]
    full_vote, full_counts, full_consensus = majority_vote_with_counts([row["normalized_answer"] for row in full_candidates])
    compressed_vote, compressed_counts, compressed_consensus = majority_vote_with_counts([row["normalized_answer"] for row in compressed_candidates])
    full_score = score_prediction(dataset, full_vote, sample.reference_answer)
    compressed_score = score_prediction(dataset, compressed_vote, sample.reference_answer)
    decision = decide_early_exit(
        stage_a_rows,
        mean_conf_threshold=protocol.mean_conf_threshold,
        conf_spread_threshold=protocol.conf_spread_threshold,
    )
    full_comm_tokens = _broadcast_packet_tokens(full_packets, protocol.agent_count)
    compressed_comm_tokens = _broadcast_packet_tokens(compressed_packets, protocol.agent_count)
    ratio = compression_ratio(compressed_packets, full_packets)
    common = {
        "run_id": run_id,
        "dataset": dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "model_name": backbone.name,
        "gold": sample.reference_answer,
        "question_preview": sample.question[:160],
        "stage_a_prediction": stage_a_vote,
        "stage_a_score": stage_a_score,
        "stage_a_vote_counts": stage_a_counts,
        "initial_consensus": stage_a_consensus,
        "stage_a_trace_hash": stage_a_rows[0].get("stage_a_trace_hash"),
        "mean_confidence": decision.mean_confidence,
        "confidence_spread": decision.confidence_spread,
        "any_invalid_confidence": decision.any_invalid_confidence,
        "compression_ratio": ratio,
    }
    stage_a_cost = _cost(stage_a_rows)
    full_cost = _cost(stage_a_rows + full_beliefs)
    compressed_cost = _cost(stage_a_rows + compressed_beliefs)
    rows = [
        _prediction_row(
            common,
            method_name="mv_3",
            prediction=stage_a_vote,
            score=stage_a_score,
            method_kind="baseline",
            early_exit=True,
            triggered=False,
            trigger_reason="majority_vote_baseline",
            communication_tokens=0.0,
            costs=stage_a_cost,
            calls_per_question=len(stage_a_rows),
            stage_b_prediction=None,
            stage_b_score=None,
            stage_b_vote_counts={},
            stage_b_trace_hash_used=None,
            final_consensus=stage_a_consensus,
        ),
        _prediction_row(
            common,
            method_name="always_full",
            prediction=full_vote,
            score=full_score,
            method_kind="communication",
            early_exit=False,
            triggered=True,
            trigger_reason="always_full",
            communication_tokens=full_comm_tokens,
            costs=full_cost,
            calls_per_question=len(stage_a_rows) + len(full_beliefs),
            stage_b_prediction=full_vote,
            stage_b_score=full_score,
            stage_b_vote_counts=full_counts,
            stage_b_trace_hash_used=full_beliefs[0].get("stage_b_trace_hash"),
            final_consensus=full_consensus,
        ),
        _prediction_row(
            common,
            method_name="compression_only",
            prediction=compressed_vote,
            score=compressed_score,
            method_kind="communication",
            early_exit=False,
            triggered=True,
            trigger_reason="compression_only",
            communication_tokens=compressed_comm_tokens,
            costs=compressed_cost,
            calls_per_question=len(stage_a_rows) + len(compressed_beliefs),
            stage_b_prediction=compressed_vote,
            stage_b_score=compressed_score,
            stage_b_vote_counts=compressed_counts,
            stage_b_trace_hash_used=compressed_beliefs[0].get("stage_b_trace_hash"),
            final_consensus=compressed_consensus,
        ),
    ]
    if decision.early_exit:
        rows.append(
            _prediction_row(
                common,
                method_name="sid_lite",
                prediction=stage_a_vote,
                score=stage_a_score,
                method_kind="sid_lite",
                early_exit=True,
                triggered=False,
                trigger_reason=decision.reason,
                communication_tokens=0.0,
                costs=stage_a_cost,
                calls_per_question=len(stage_a_rows),
                stage_b_prediction=None,
                stage_b_score=None,
                stage_b_vote_counts={},
                stage_b_trace_hash_used=None,
                final_consensus=stage_a_consensus,
            )
        )
    else:
        rows.append(
            _prediction_row(
                common,
                method_name="sid_lite",
                prediction=compressed_vote,
                score=compressed_score,
                method_kind="sid_lite",
                early_exit=False,
                triggered=True,
                trigger_reason=decision.reason,
                communication_tokens=compressed_comm_tokens,
                costs=compressed_cost,
                calls_per_question=len(stage_a_rows) + len(compressed_beliefs),
                stage_b_prediction=compressed_vote,
                stage_b_score=compressed_score,
                stage_b_vote_counts=compressed_counts,
                stage_b_trace_hash_used=compressed_beliefs[0].get("stage_b_trace_hash"),
                final_consensus=compressed_consensus,
            )
        )
    return rows


def _prediction_row(
    common: dict[str, Any],
    *,
    method_name: str,
    prediction: str,
    score: float,
    method_kind: str,
    early_exit: bool,
    triggered: bool,
    trigger_reason: str,
    communication_tokens: float,
    costs: dict[str, float],
    calls_per_question: int,
    stage_b_prediction: str | None,
    stage_b_score: float | None,
    stage_b_vote_counts: dict[str, int],
    stage_b_trace_hash_used: str | None,
    final_consensus: bool,
) -> dict[str, Any]:
    corrected = float(common["stage_a_score"]) < 1.0 and score == 1.0
    harmed = float(common["stage_a_score"]) == 1.0 and score < 1.0
    return {
        **common,
        "method_name": method_name,
        "method_kind": method_kind,
        "prediction": prediction,
        "score": score,
        "stage_b_prediction": stage_b_prediction,
        "stage_b_score": stage_b_score,
        "stage_b_vote_counts": stage_b_vote_counts,
        "stage_b_trace_hash_used": stage_b_trace_hash_used,
        "final_consensus": final_consensus,
        "early_exit": early_exit,
        "triggered": triggered,
        "trigger_reason": trigger_reason,
        "communication_tokens_per_question": round(communication_tokens, 6),
        "prompt_tokens_per_question": costs["prompt_tokens"],
        "completion_tokens_per_question": costs["completion_tokens"],
        "total_tokens_per_question": costs["total_tokens"],
        "latency_ms_per_question": costs["latency_ms"],
        "calls_per_question": calls_per_question,
        "corrected_by_method": corrected,
        "harmed_by_method": harmed,
        "minority_rescue": corrected,
        "stage_a_hash": common.get("stage_a_trace_hash"),
        "minority_rescue_flag": corrected,
        "drift_flag": harmed,
    }


def _execute_turn(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    stage_name: str,
    method_name: str,
    round_index: int,
    agent_id: int,
    role: str,
    messages: list[dict[str, str]],
    output_mode: str,
    backbone: ResolvedModelConfig,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    seed: int,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
        validator=lambda raw_text, provider_reasoning_text: _validate_output(
            raw_text,
            output_mode,
            provider_reasoning_text=provider_reasoning_text,
        ),
    )
    final_answer = str(result.validated_output.get("final_answer") or result.validated_output.get("new_answer") or "")
    confidence_raw = result.validated_output.get("confidence_raw")
    confidence_value, confidence_valid, confidence_source = normalize_confidence(confidence_raw)
    row = {
        "run_id": run_id,
        "dataset": dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "stage_name": stage_name,
        "method_name": method_name,
        "round_index": round_index,
        "agent_id": agent_id,
        "role": role,
        "prompt_hash": result.prompt_hash,
        "output_status": result.output_status,
        "prediction": normalize_prediction(dataset, final_answer) if final_answer else "",
        "normalized_answer": normalize_prediction(dataset, final_answer) if final_answer else "",
        "score": score_prediction(dataset, final_answer, sample.reference_answer) if final_answer else 0.0,
        "confidence_raw": confidence_raw,
        "confidence_value": confidence_value,
        "confidence_valid": confidence_valid,
        "confidence_source": confidence_source,
        "reasoning_trace": str(result.validated_output.get("reasoning_trace") or ""),
        "claim_span": str(result.validated_output.get("claim_span") or ""),
        "key_evidence": str(result.validated_output.get("key_evidence") or ""),
        "uncertain_point": str(result.validated_output.get("uncertain_point") or ""),
        "prompt_tokens": float(result.usage.get("prompt_tokens") or 0.0),
        "completion_tokens": float(result.usage.get("completion_tokens") or 0.0),
        "total_tokens": float(result.usage.get("total_tokens") or 0.0),
        "latency_ms": float(result.response_payload.get("latency_ms") or 0.0),
        "cache_hit": result.cache_hit,
        "request_error": result.request_error,
        "assistant_text": result.response_payload.get("assistant_text", ""),
        "provider_reasoning_text": result.response_payload.get("provider_reasoning_text", ""),
        "validated_output": result.validated_output,
    }
    if extra_fields:
        row.update(extra_fields)
    return row


def _validate_output(raw_text: str, output_mode: str, *, provider_reasoning_text: str = "") -> dict[str, Any]:
    if output_mode == "solver":
        return validate_or_recover_structured_output(
            raw_text,
            SCHEMA_ANSWER_WITH_PROXY_SIGNALS_DELIBERATION,
            provider_reasoning_text=provider_reasoning_text,
        )
    if output_mode == "belief":
        return validate_or_recover_structured_output(
            raw_text,
            SCHEMA_BELIEF_UPDATE_DELTA,
            provider_reasoning_text=provider_reasoning_text,
        )
    raise ValueError(f"Unsupported SID-lite output mode: {output_mode}")


def _decode_json_object(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Assistant output must contain a JSON object.")
    payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Assistant output must be a JSON object.")
    return payload


def _require_textish(value: object, field_name: str) -> str:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{field_name} is required.")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _optional_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _enrich_packet(
    run_id: str,
    dataset: str,
    split_name: str,
    sample_id: str,
    packet_mode: str,
    stage_a_trace_hash: str,
    packet: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "dataset": dataset,
        "split": split_name,
        "sample_id": sample_id,
        "method_name": f"packet::{packet_mode}",
        "stage_a_trace_hash": stage_a_trace_hash,
        **packet,
    }


def _estimate_work(
    experiment: SidLiteExperimentConfig,
    phase_name: str,
    benchmarks,
    protocol: SidLiteProtocolConfig,
) -> tuple[int, int]:
    total_samples = 0
    for benchmark in benchmarks:
        split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
        total_samples += len(load_split_ids(benchmark.slug, split_name))
    calls_per_sample = protocol.agent_count * 3
    return total_samples * calls_per_sample, total_samples * len(experiment.methods)


def _resolve_split_name(experiment: SidLiteExperimentConfig, phase_name: str, benchmark_slug: str) -> str:
    return resolve_phase_split_name(experiment, phase_name, benchmark_slug)


def _prepare_run_paths(run_root: str | Path, experiment_name: str, phase_name: str, run_id: str) -> RunPaths:
    root = prepare_run_root(run_root, experiment_name, phase_name, run_id)
    return RunPaths(
        root=root,
        manifest=root / "manifest.json",
        stage_a_turns=root / "stage_a_turns.jsonl",
        message_packets=root / "message_packets.jsonl",
        belief_updates=root / "belief_updates.jsonl",
        final_predictions=root / "final_predictions.jsonl",
        metrics=root / "metrics.json",
        diagnostics=root / "diagnostics.json",
        progress=root / "progress.json",
        run_summary=root / "run_summary.json",
        run_validation=root / "run_validation.json",
        report_markdown=root / "report.md",
        paper_summary=root / "paper_summary.csv",
    )


def _broadcast_packet_tokens(packets: list[dict[str, Any]], agent_count: int) -> float:
    return float(sum(int(packet.get("approx_packet_tokens") or 0) for packet in packets) * max(0, agent_count - 1))


def _cost(rows: list[dict[str, Any]]) -> dict[str, float]:
    return summarize_row_cost(rows)


def _trace_hash(rows: list[dict[str, Any]], keys: list[str]) -> str:
    return stable_trace_hash(rows, keys)


def _mean(values) -> float:
    return safe_mean(values)

