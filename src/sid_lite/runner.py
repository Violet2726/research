"""SID-lite 主运行链路。

本模块把 SID-lite 的完整流程落地为可执行实验：
共享 Stage A、早退判定、压缩消息广播、belief update、题级聚合、
以及指标/诊断/报告产物生成。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
from hashlib import sha256
from pathlib import Path
import csv
import json
from typing import Any

from dotenv import load_dotenv

from experiment_core.cache import CachedResponse, RequestCache, RequestCacheRouter, build_request_cache_key, json_dump
from experiment_core.config import ResolvedModelConfig
from experiment_core.datasets import DatasetSample, load_split_ids, select_samples
from experiment_core.evaluation import normalize_prediction, score_prediction
from experiment_core.providers import OpenAICompatibleProvider, ProviderRequestError, build_payload, estimate_request_tokens
from experiment_core.rate_limits import SlidingWindowRateLimiter
from experiment_core.runtime import RunProgressTracker, build_run_id
from experiment_core.selective_signals import normalize_confidence
from experiment_core.structured_output import (
    ARTIFACT_VERSION,
    OUTPUT_MODE_SPARC_BELIEF_UPDATE,
    OUTPUT_MODE_SPARC_SOLVER,
    validate_or_recover_structured_output,
)
from experiment_core.workspace import default_cache_root, default_runs_root
from sid_lite.config import SidLiteExperimentConfig, SidLiteProtocolConfig, load_benchmarks, load_protocol_config, phase_metadata
from sid_lite.logic import (
    METHOD_ORDER,
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
    cache = cache_router.for_endpoint(
        provider=backbone.provider,
        base_url=backbone.base_url,
        chat_path=backbone.chat_path,
    )
    limiter = SlidingWindowRateLimiter(
        requests_per_minute=experiment.requests_per_minute_limit,
        tokens_per_minute=experiment.tokens_per_minute_limit,
    )
    run_id = build_run_id(experiment.name, phase_name, backbone.name)
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
            for benchmark in benchmarks:
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
                        stage_a_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        progress.record_call(row, method_key="stage_name")
                    stage_a_handle.flush()
                    for row in packet_rows:
                        packet_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    packet_handle.flush()
                    for row in belief_rows:
                        belief_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        progress.record_call(row, method_key="method_name")
                    belief_handle.flush()
                    for row in prediction_rows:
                        prediction_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        progress.record_predictions(1, str(row["dataset"]), str(row["method_name"]))
                    prediction_handle.flush()
                    all_stage_a.extend(stage_a_rows)
                    all_packets.extend(packet_rows)
                    all_beliefs.extend(belief_rows)
                    all_predictions.extend(prediction_rows)

        metrics = _build_metrics(all_predictions)
        diagnostics = _build_diagnostics(all_predictions)
        run_paths.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        run_paths.diagnostics.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_paper_summary(run_paths.paper_summary, metrics)
        render_report(run_paths.root)
        run_paths.run_summary.write_text(json.dumps(summarize_run(run_paths.root), ensure_ascii=False, indent=2), encoding="utf-8")
        run_paths.run_validation.write_text(json.dumps(validate_run(run_paths.root), ensure_ascii=False, indent=2), encoding="utf-8")
        progress.mark_completed()
        return run_paths.root
    finally:
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
    max_workers = max(1, min(max_concurrent_requests, len(samples) or 1))
    completed = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {executor.submit(worker, sample=sample): index for index, sample in enumerate(samples)}
        for future in as_completed(future_to_index):
            completed.append((future_to_index[future], *future.result()))
    completed.sort(key=lambda item: item[0])
    return completed


def _run_sample(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
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
    payload = build_payload(backbone, messages, temperature, top_p, max_output_tokens, seed)
    prompt_hash = _prompt_hash(messages)
    cache_key = build_request_cache_key(payload)
    cached = cache.get(cache_key)
    if cached is None:
        limiter.acquire(estimate_request_tokens(payload))
        try:
            response = provider.chat_completion(payload)
            response_payload = {
                "http_status": response.http_status,
                "assistant_text": response.assistant_text,
                "provider_reasoning_text": response.provider_reasoning_text,
                "usage_reported": response.usage_reported,
                "usage_estimated": response.usage_estimated,
                "latency_ms": response.latency_ms,
                "provider_request_id": response.provider_request_id,
                "request_error": None,
            }
            cache.put(
                CachedResponse(
                    cache_key=cache_key,
                    payload_json=json_dump(payload),
                    response_json=json_dump(response_payload),
                    http_status=response.http_status,
                    latency_ms=response.latency_ms,
                    provider_request_id=response.provider_request_id,
                )
            )
        except ProviderRequestError as exc:
            response_payload = {
                "http_status": exc.http_status,
                "assistant_text": "",
                "provider_reasoning_text": "",
                "usage_reported": None,
                "usage_estimated": None,
                "latency_ms": 0.0,
                "provider_request_id": exc.provider_request_id,
                "request_error": exc.message,
            }
        cache_hit = False
    else:
        response_payload = json.loads(cached.response_json)
        cache_hit = True

    request_error = response_payload.get("request_error")
    if request_error:
        validated_output: dict[str, Any] = {}
        output_status = "request_fail"
    else:
        try:
            validated_output = _validate_output(
                str(response_payload.get("assistant_text") or ""),
                output_mode,
                provider_reasoning_text=str(response_payload.get("provider_reasoning_text") or ""),
            )
            output_status = "ok"
        except Exception:
            validated_output = {}
            output_status = "schema_fail"

    final_answer = str(validated_output.get("final_answer") or validated_output.get("new_answer") or "")
    confidence_raw = validated_output.get("confidence_raw")
    confidence_value, confidence_valid, confidence_source = normalize_confidence(confidence_raw)
    usage = response_payload.get("usage_reported") or response_payload.get("usage_estimated") or {}
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
        "prompt_hash": prompt_hash,
        "output_status": output_status,
        "prediction": normalize_prediction(dataset, final_answer) if final_answer else "",
        "normalized_answer": normalize_prediction(dataset, final_answer) if final_answer else "",
        "score": score_prediction(dataset, final_answer, sample.reference_answer) if final_answer else 0.0,
        "confidence_raw": confidence_raw,
        "confidence_value": confidence_value,
        "confidence_valid": confidence_valid,
        "confidence_source": confidence_source,
        "reasoning_trace": str(validated_output.get("reasoning_trace") or ""),
        "claim_span": str(validated_output.get("claim_span") or ""),
        "key_evidence": str(validated_output.get("key_evidence") or ""),
        "uncertain_point": str(validated_output.get("uncertain_point") or ""),
        "prompt_tokens": float(usage.get("prompt_tokens") or 0.0),
        "completion_tokens": float(usage.get("completion_tokens") or 0.0),
        "total_tokens": float(usage.get("total_tokens") or 0.0),
        "latency_ms": float(response_payload.get("latency_ms") or 0.0),
        "cache_hit": cache_hit,
        "request_error": request_error,
        "assistant_text": response_payload.get("assistant_text", ""),
        "provider_reasoning_text": response_payload.get("provider_reasoning_text", ""),
        "validated_output": validated_output,
    }
    if extra_fields:
        row.update(extra_fields)
    return row


def _validate_output(raw_text: str, output_mode: str, *, provider_reasoning_text: str = "") -> dict[str, Any]:
    if output_mode == "solver":
        return validate_or_recover_structured_output(
            raw_text,
            OUTPUT_MODE_SPARC_SOLVER,
            provider_reasoning_text=provider_reasoning_text,
        )
    if output_mode == "belief":
        return validate_or_recover_structured_output(
            raw_text,
            OUTPUT_MODE_SPARC_BELIEF_UPDATE,
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


def _build_metrics(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: list[dict[str, Any]] = []
    datasets = sorted({row["dataset"] for row in prediction_rows})
    for dataset in [*datasets, "overall"]:
        rows_for_dataset = prediction_rows if dataset == "overall" else [row for row in prediction_rows if row["dataset"] == dataset]
        for method in METHOD_ORDER:
            rows = [row for row in rows_for_dataset if row["method_name"] == method]
            if not rows:
                continue
            accuracy = _mean(float(row["score"]) for row in rows)
            total_tokens = _mean(float(row["total_tokens_per_question"]) for row in rows)
            compression_values = [float(row["compression_ratio"]) for row in rows if row.get("compression_ratio") is not None]
            summary.append(
                {
                    "dataset": dataset,
                    "model_name": rows[0]["model_name"],
                    "method_name": method,
                    "method_kind": rows[0]["method_kind"],
                    "question_count": len(rows),
                    "accuracy_mean": round(accuracy, 6),
                    "prompt_tokens_mean": round(_mean(float(row["prompt_tokens_per_question"]) for row in rows), 6),
                    "completion_tokens_mean": round(_mean(float(row["completion_tokens_per_question"]) for row in rows), 6),
                    "total_tokens_mean": round(total_tokens, 6),
                    "communication_tokens_mean": round(_mean(float(row["communication_tokens_per_question"]) for row in rows), 6),
                    "latency_ms_mean": round(_mean(float(row["latency_ms_per_question"]) for row in rows), 6),
                    "calls_per_question_mean": round(_mean(float(row["calls_per_question"]) for row in rows), 6),
                    "acc_per_1k_tokens": round((accuracy / total_tokens * 1000) if total_tokens else 0.0, 6),
                    "early_exit_rate": round(_mean(1.0 if row.get("early_exit") else 0.0 for row in rows), 6),
                    "trigger_rate": round(_mean(1.0 if row.get("triggered") else 0.0 for row in rows), 6),
                    "compression_ratio_mean": round(sum(compression_values) / len(compression_values), 6) if compression_values else None,
                    "corrected_count": sum(1 for row in rows if row.get("corrected_by_method")),
                    "harmed_count": sum(1 for row in rows if row.get("harmed_by_method")),
                    "minority_rescue_count": sum(1 for row in rows if row.get("minority_rescue")),
                }
            )
    return {"summary": summary}


def _build_diagnostics(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    sid_rows = [row for row in prediction_rows if row.get("method_name") == "sid_lite"]
    return {
        "sid_early_exit_rate": round(_mean(1.0 if row.get("early_exit") else 0.0 for row in sid_rows), 6),
        "invalid_confidence_fail_open_count": sum(1 for row in sid_rows if row.get("trigger_reason") == "invalid_confidence_fail_open"),
        "black_box_proxy_note": "confidence_raw, claim_span, key_evidence, and uncertain_point approximate SID self signals.",
    }


def _write_paper_summary(path: Path, metrics: dict[str, Any]) -> None:
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
        for row in metrics.get("summary", []):
            writer.writerow({key: row.get(key) for key in fieldnames})


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
    phase = phase_metadata(experiment, phase_name)
    if "split_overrides" in phase:
        return str(phase["split_overrides"][benchmark_slug])
    return str(phase["split_suffix"])


def _prepare_run_paths(run_root: str | Path, experiment_name: str, phase_name: str, run_id: str) -> RunPaths:
    root = Path(run_root) / experiment_name / phase_name / run_id
    root.mkdir(parents=True, exist_ok=True)
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
    return {
        "prompt_tokens": round(sum(float(row.get("prompt_tokens") or 0.0) for row in rows), 6),
        "completion_tokens": round(sum(float(row.get("completion_tokens") or 0.0) for row in rows), 6),
        "total_tokens": round(sum(float(row.get("total_tokens") or 0.0) for row in rows), 6),
        "latency_ms": round(sum(float(row.get("latency_ms") or 0.0) for row in rows), 6),
    }


def _trace_hash(rows: list[dict[str, Any]], keys: list[str]) -> str:
    payload = [{key: row.get(key) for key in keys} for row in rows]
    return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _prompt_hash(messages: list[dict[str, str]]) -> str:
    return sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _mean(values) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return sum(materialized) / len(materialized)
