"""Free-MAD-lite 主运行链路。

本模块把 Free-MAD-lite 的完整流程落地为可执行实验：
共享 Stage A、单轮 anti-conformity debate、轨迹裁决器调用、
以及题级指标、诊断与报告产物生成。
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

from experiment_core.foundation.cache import CachedResponse, RequestCache, RequestCacheRouter, build_request_cache_key, json_dump
from experiment_core.foundation.config import ResolvedModelConfig
from experiment_core.foundation.datasets import DatasetSample, load_split_ids, select_samples
from experiment_core.foundation.evaluation import normalize_prediction, score_prediction
from experiment_core.foundation.providers import OpenAICompatibleProvider, ProviderRequestError, build_payload, estimate_request_tokens
from experiment_core.foundation.rate_limits import SlidingWindowRateLimiter
from experiment_core.foundation.runtime import RunProgressTracker, build_run_id
from experiment_core.controls.selective_signals import normalize_confidence
from experiment_core.foundation.structured_output import ARTIFACT_VERSION, OUTPUT_MODE_CORE, validate_or_recover_structured_output
from experiment_core.foundation.workspace import default_cache_root, default_runs_root
from free_mad_lite.config import (
    FreeMadLiteExperimentConfig,
    FreeMadLiteProtocolConfig,
    load_benchmarks,
    load_protocol_config,
    phase_metadata,
)
from free_mad_lite.logic import (
    METHOD_ORDER,
    build_trajectory_decision,
    deterministic_trajectory_fallback,
    majority_vote_with_counts,
    trajectory_hash,
)
from free_mad_lite.prompting import (
    anti_conformity_prompt_hash,
    build_debate_messages,
    build_initial_messages,
    build_trajectory_judge_messages,
)
from free_mad_lite.reporting import render_report, summarize_run
from free_mad_lite.validation import validate_run


@dataclass(frozen=True)
class RunPaths:
    """Free-MAD-lite 运行目录下的固定产物路径。"""

    root: Path
    manifest: Path
    agent_turns: Path
    debate_messages: Path
    trajectory_scores: Path
    final_predictions: Path
    metrics: Path
    diagnostics: Path
    progress: Path
    run_summary: Path
    run_validation: Path
    report_markdown: Path
    paper_summary: Path


def run_experiment(
    experiment: FreeMadLiteExperimentConfig,
    phase_name: str,
    backbone: ResolvedModelConfig,
    run_root: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> Path:
    """执行一个 Free-MAD-lite phase，并写出完整运行目录。"""
    load_dotenv(".env.local", override=False)
    run_root = run_root or default_runs_root("free_mad_lite")
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
        "anti_conformity_prompt_hash": anti_conformity_prompt_hash(),
        "max_concurrent_requests": experiment.max_concurrent_requests,
        "requests_per_minute_limit": experiment.requests_per_minute_limit,
        "tokens_per_minute_limit": experiment.tokens_per_minute_limit,
        "limitation_note": "Free-MAD-lite validates anti-conformity and LLM trajectory judging only; it is not full Free-MAD score-model reproduction.",
        "total_planned_calls": total_calls,
        "total_planned_predictions": total_predictions,
    }
    run_paths.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    all_turns: list[dict[str, Any]] = []
    all_debate_messages: list[dict[str, Any]] = []
    all_scores: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []

    try:
        with (
            run_paths.agent_turns.open("w", encoding="utf-8") as turn_handle,
            run_paths.debate_messages.open("w", encoding="utf-8") as debate_handle,
            run_paths.trajectory_scores.open("w", encoding="utf-8") as score_handle,
            run_paths.final_predictions.open("w", encoding="utf-8") as prediction_handle,
        ):
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
                for _, turn_rows, debate_rows, score_rows, prediction_rows in results:
                    for row in turn_rows:
                        turn_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        progress.record_call(row, method_key="method_name")
                    turn_handle.flush()
                    for row in debate_rows:
                        debate_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    debate_handle.flush()
                    for row in score_rows:
                        score_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    score_handle.flush()
                    for row in prediction_rows:
                        prediction_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        progress.record_predictions(1, str(row["dataset"]), str(row["method_name"]))
                    prediction_handle.flush()
                    all_turns.extend(turn_rows)
                    all_debate_messages.extend(debate_rows)
                    all_scores.extend(score_rows)
                    all_predictions.extend(prediction_rows)

        metrics = _build_metrics(all_predictions)
        diagnostics = _build_diagnostics(all_predictions, all_scores)
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
    protocol: FreeMadLiteProtocolConfig,
    backbone: ResolvedModelConfig,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    global_seed: int,
    prompt_version: str,
    max_concurrent_requests: int,
) -> list[tuple[int, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]]:
    """样本级并发执行；单题内部保持初始、辩论、裁决顺序。"""
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
    protocol: FreeMadLiteProtocolConfig,
    backbone: ResolvedModelConfig,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    global_seed: int,
    prompt_version: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    initial_rows = []
    for agent_id in range(1, protocol.agent_count + 1):
        initial_rows.append(
            _execute_turn(
                run_id=run_id,
                dataset=dataset,
                split_name=split_name,
                sample=sample,
                method_name="shared_initial",
                round_index=0,
                agent_id=agent_id,
                role="initial",
                messages=build_initial_messages(sample, agent_id, prompt_version=prompt_version),
                output_mode="agent",
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
    stage_a_trace_hash = _trace_hash(initial_rows, ["agent_id", "normalized_answer", "validated_output"])
    for row in initial_rows:
        row["stage_a_trace_hash"] = stage_a_trace_hash

    vanilla_rows, vanilla_messages = _run_debate_round(
        run_id=run_id,
        dataset=dataset,
        split_name=split_name,
        sample=sample,
        initial_rows=initial_rows,
        mode="vanilla",
        protocol=protocol,
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        global_seed=global_seed,
        prompt_version=prompt_version,
    )
    anti_rows, anti_messages = _run_debate_round(
        run_id=run_id,
        dataset=dataset,
        split_name=split_name,
        sample=sample,
        initial_rows=initial_rows,
        mode="anti_conformity",
        protocol=protocol,
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        global_seed=global_seed,
        prompt_version=prompt_version,
    )
    judge_row = _run_trajectory_judge(
        run_id=run_id,
        dataset=dataset,
        split_name=split_name,
        sample=sample,
        initial_rows=initial_rows,
        anti_rows=anti_rows,
        protocol=protocol,
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        global_seed=global_seed,
        prompt_version=prompt_version,
    )
    score_row, prediction_rows = _build_outputs(
        run_id=run_id,
        dataset=dataset,
        split_name=split_name,
        sample=sample,
        backbone=backbone,
        initial_rows=initial_rows,
        vanilla_rows=vanilla_rows,
        anti_rows=anti_rows,
        judge_row=judge_row,
        stage_a_trace_hash=stage_a_trace_hash,
    )
    turn_rows = initial_rows + vanilla_rows + anti_rows + [judge_row]
    return turn_rows, vanilla_messages + anti_messages, [score_row], prediction_rows


def _run_debate_round(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    initial_rows: list[dict[str, Any]],
    mode: str,
    protocol: FreeMadLiteProtocolConfig,
    backbone: ResolvedModelConfig,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    global_seed: int,
    prompt_version: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    messages_rows = []
    for recipient in initial_rows:
        recipient_id = int(recipient["agent_id"])
        peer_messages = []
        for sender in initial_rows:
            if int(sender["agent_id"]) == recipient_id:
                continue
            peer_messages.append(
                {
                    "agent": f"agent_{sender['agent_id']}",
                    "answer": str(sender.get("normalized_answer") or ""),
                    "reasoning": str(sender.get("reasoning") or ""),
                }
            )
            messages_rows.append(
                {
                    "run_id": run_id,
                    "dataset": dataset,
                    "split": split_name,
                    "sample_id": sample.sample_id,
                    "method_name": mode,
                    "round_index": 1,
                    "sender_agent_id": int(sender["agent_id"]),
                    "recipient_agent_id": recipient_id,
                    "sender_answer": str(sender.get("normalized_answer") or ""),
                    "sender_reasoning": str(sender.get("reasoning") or ""),
                }
            )
        rows.append(
            _execute_turn(
                run_id=run_id,
                dataset=dataset,
                split_name=split_name,
                sample=sample,
                method_name=f"{mode}_r1",
                round_index=1,
                agent_id=recipient_id,
                role=mode,
                messages=build_debate_messages(
                    sample,
                    recipient_id,
                    mode=mode,
                    previous_answer=str(recipient.get("normalized_answer") or ""),
                    previous_reasoning=str(recipient.get("reasoning") or ""),
                    peer_messages=peer_messages,
                    prompt_version=prompt_version,
                ),
                output_mode="agent",
                backbone=backbone,
                provider=provider,
                cache=cache,
                limiter=limiter,
                temperature=protocol.debate_temperature,
                top_p=protocol.top_p,
                max_output_tokens=protocol.max_output_tokens,
                seed=global_seed + recipient_id + (1000 if mode == "vanilla" else 2000),
                extra_fields={"debate_mode": mode, "visible_peer_count": len(peer_messages)},
            )
        )
    trace = _trace_hash(rows, ["agent_id", "normalized_answer", "validated_output", "debate_mode"])
    for row in rows:
        row["stage_b_trace_hash"] = trace
    return rows, messages_rows


def _run_trajectory_judge(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    initial_rows: list[dict[str, Any]],
    anti_rows: list[dict[str, Any]],
    protocol: FreeMadLiteProtocolConfig,
    backbone: ResolvedModelConfig,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    global_seed: int,
    prompt_version: str,
) -> dict[str, Any]:
    trajectories = [
        {
            "agent_id": int(initial["agent_id"]),
            "initial_answer": initial.get("normalized_answer"),
            "initial_reasoning": initial.get("reasoning"),
            "anti_answer": anti.get("normalized_answer"),
            "anti_reasoning": anti.get("reasoning"),
            "changed_answer": anti.get("changed_answer"),
        }
        for initial, anti in zip(initial_rows, anti_rows, strict=True)
    ]
    return _execute_turn(
        run_id=run_id,
        dataset=dataset,
        split_name=split_name,
        sample=sample,
        method_name="free_mad_lite_llm_trajectory",
        round_index=2,
        agent_id=0,
        role="trajectory_judge",
        messages=build_trajectory_judge_messages(sample, trajectories, prompt_version=prompt_version),
        output_mode="judge",
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        temperature=protocol.judge_temperature,
        top_p=protocol.top_p,
        max_output_tokens=protocol.judge_max_output_tokens,
        seed=global_seed + 9000,
        extra_fields={"visible_trajectory_count": len(trajectories)},
    )


def _build_outputs(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    backbone: ResolvedModelConfig,
    initial_rows: list[dict[str, Any]],
    vanilla_rows: list[dict[str, Any]],
    anti_rows: list[dict[str, Any]],
    judge_row: dict[str, Any],
    stage_a_trace_hash: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    initial_vote, initial_counts, initial_consensus = majority_vote_with_counts([row["normalized_answer"] for row in initial_rows])
    vanilla_vote, vanilla_counts, vanilla_consensus = majority_vote_with_counts([row["normalized_answer"] for row in vanilla_rows])
    anti_vote, anti_counts, anti_consensus = majority_vote_with_counts([row["normalized_answer"] for row in anti_rows])
    trajectory_decision = build_trajectory_decision(judge_row, initial_rows, anti_rows)
    trajectory_prediction = normalize_prediction(dataset, trajectory_decision.final_answer) if trajectory_decision.final_answer else ""
    trajectory_score = score_prediction(dataset, trajectory_prediction, sample.reference_answer) if trajectory_prediction else 0.0
    anti_hash = _trace_hash(anti_rows, ["agent_id", "normalized_answer", "validated_output", "debate_mode"])
    vanilla_hash = _trace_hash(vanilla_rows, ["agent_id", "normalized_answer", "validated_output", "debate_mode"])
    trajectory_content_hash = sha256(trajectory_hash(initial_rows + anti_rows).encode("utf-8")).hexdigest()
    score_row = {
        "run_id": run_id,
        "dataset": dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "method_name": "free_mad_lite_llm_trajectory",
        "output_status": judge_row.get("output_status"),
        "judge_fallback_used": trajectory_decision.fallback_used,
        "judge_fallback_reason": trajectory_decision.fallback_reason,
        "selected_agent_id": trajectory_decision.selected_agent_id,
        "prediction": trajectory_prediction,
        "score": trajectory_score,
        "rationale": trajectory_decision.rationale,
        "trajectory_hash": trajectory_content_hash,
        "judge_prompt_hash": judge_row.get("prompt_hash"),
    }
    common = {
        "run_id": run_id,
        "dataset": dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "model_name": backbone.name,
        "gold": sample.reference_answer,
        "question_preview": sample.question[:160],
        "stage_a_trace_hash": stage_a_trace_hash,
        "initial_vote_prediction": initial_vote,
        "initial_vote_score": score_prediction(dataset, initial_vote, sample.reference_answer),
        "initial_vote_counts": initial_counts,
        "initial_consensus": initial_consensus,
    }
    initial_cost = _cost(initial_rows)
    vanilla_cost = _cost(initial_rows + vanilla_rows)
    anti_cost = _cost(initial_rows + anti_rows)
    trajectory_cost = _cost(initial_rows + anti_rows + [judge_row])
    rows = [
        _prediction_row(
            common,
            method_name="mv_3_initial",
            method_kind="baseline",
            prediction=initial_vote,
            score=score_prediction(dataset, initial_vote, sample.reference_answer),
            vote_counts=initial_counts,
            final_consensus=initial_consensus,
            communication_tokens=0.0,
            costs=initial_cost,
            calls_per_question=len(initial_rows),
            stage_b_trace_hash_used=None,
            judge_fallback_used=False,
            changed_answer_rate=0.0,
        ),
        _prediction_row(
            common,
            method_name="vanilla_mad_r1_final_vote",
            method_kind="vanilla_debate",
            prediction=vanilla_vote,
            score=score_prediction(dataset, vanilla_vote, sample.reference_answer),
            vote_counts=vanilla_counts,
            final_consensus=vanilla_consensus,
            communication_tokens=_debate_message_tokens(initial_rows),
            costs=vanilla_cost,
            calls_per_question=len(initial_rows) + len(vanilla_rows),
            stage_b_trace_hash_used=vanilla_hash,
            judge_fallback_used=False,
            changed_answer_rate=_changed_answer_rate(vanilla_rows),
        ),
        _prediction_row(
            common,
            method_name="anti_conformity_final_vote",
            method_kind="anti_conformity",
            prediction=anti_vote,
            score=score_prediction(dataset, anti_vote, sample.reference_answer),
            vote_counts=anti_counts,
            final_consensus=anti_consensus,
            communication_tokens=_debate_message_tokens(initial_rows),
            costs=anti_cost,
            calls_per_question=len(initial_rows) + len(anti_rows),
            stage_b_trace_hash_used=anti_hash,
            judge_fallback_used=False,
            changed_answer_rate=_changed_answer_rate(anti_rows),
        ),
        _prediction_row(
            common,
            method_name="free_mad_lite_llm_trajectory",
            method_kind="trajectory_judge",
            prediction=trajectory_prediction,
            score=trajectory_score,
            vote_counts=anti_counts,
            final_consensus=anti_consensus,
            communication_tokens=_debate_message_tokens(initial_rows),
            costs=trajectory_cost,
            calls_per_question=len(initial_rows) + len(anti_rows) + 1,
            stage_b_trace_hash_used=anti_hash,
            judge_fallback_used=trajectory_decision.fallback_used,
            changed_answer_rate=_changed_answer_rate(anti_rows),
        ),
    ]
    return score_row, rows


def _prediction_row(
    common: dict[str, Any],
    *,
    method_name: str,
    method_kind: str,
    prediction: str,
    score: float,
    vote_counts: dict[str, int],
    final_consensus: bool,
    communication_tokens: float,
    costs: dict[str, float],
    calls_per_question: int,
    stage_b_trace_hash_used: str | None,
    judge_fallback_used: bool,
    changed_answer_rate: float,
) -> dict[str, Any]:
    initial_score = float(common["initial_vote_score"])
    corrected = initial_score < 1.0 and score == 1.0
    harmed = initial_score == 1.0 and score < 1.0
    return {
        **common,
        "method_name": method_name,
        "method_kind": method_kind,
        "prediction": prediction,
        "score": score,
        "vote_counts": vote_counts,
        "final_consensus": final_consensus,
        "communication_tokens_per_question": round(communication_tokens, 6),
        "prompt_tokens_per_question": costs["prompt_tokens"],
        "completion_tokens_per_question": costs["completion_tokens"],
        "total_tokens_per_question": costs["total_tokens"],
        "latency_ms_per_question": costs["latency_ms"],
        "calls_per_question": calls_per_question,
        "stage_b_trace_hash_used": stage_b_trace_hash_used,
        "judge_fallback_used": judge_fallback_used,
        "changed_answer_rate": changed_answer_rate,
        "corrected_by_method": corrected,
        "harmed_by_method": harmed,
        "minority_rescue": corrected and prediction != common["initial_vote_prediction"],
    }


def _execute_turn(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
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
    cache_key = build_request_cache_key(
        provider=backbone.provider,
        request_model=backbone.model_id,
        payload=payload,
    )
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

    final_answer = str(validated_output.get("final_answer") or "")
    confidence_raw = validated_output.get("confidence_raw")
    confidence_value, confidence_valid, confidence_source = normalize_confidence(confidence_raw)
    usage = response_payload.get("usage_reported") or response_payload.get("usage_estimated") or {}
    row = {
        "run_id": run_id,
        "dataset": dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "method_name": method_name,
        "round_index": round_index,
        "agent_id": agent_id,
        "role": role,
        "prompt_hash": prompt_hash,
        "output_status": output_status,
        "prediction": normalize_prediction(dataset, final_answer) if final_answer else "",
        "normalized_answer": normalize_prediction(dataset, final_answer) if final_answer else "",
        "score": score_prediction(dataset, final_answer, sample.reference_answer) if final_answer else 0.0,
        "reasoning": str(validated_output.get("reasoning") or ""),
        "changed_answer": bool(validated_output.get("changed_answer")) if "changed_answer" in validated_output else False,
        "confidence_raw": confidence_raw,
        "confidence_value": confidence_value,
        "confidence_valid": confidence_valid,
        "confidence_source": confidence_source,
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
    if output_mode == "agent":
        try:
            payload = _decode_json_object(raw_text)
            final_answer = _require_textish(payload.get("final_answer"), "final_answer")
            changed = payload.get("changed_answer")
            return {
                "final_answer": final_answer,
                "reasoning": _optional_text(payload.get("reasoning")),
                "changed_answer": changed if isinstance(changed, bool) else False,
                "confidence_raw": payload.get("confidence_raw"),
            }
        except Exception:
            recovered = validate_or_recover_structured_output(
                raw_text,
                OUTPUT_MODE_CORE,
                provider_reasoning_text=provider_reasoning_text,
            )
            return {
                "final_answer": recovered["final_answer"],
                "reasoning": _optional_text(recovered.get("reasoning")),
                "changed_answer": False,
                "confidence_raw": None,
            }
    payload = _decode_json_object(raw_text)
    if output_mode == "judge":
        final_answer = _require_textish(payload.get("final_answer"), "final_answer")
        return {
            "final_answer": final_answer,
            "selected_agent_id": _optional_int(payload.get("selected_agent_id")),
            "rationale": _optional_text(payload.get("rationale")) or "",
        }
    raise ValueError(f"Unsupported Free-MAD-lite output mode: {output_mode}")


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


def _optional_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(str(value).strip())
    except ValueError:
        return None


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
                    "judge_fallback_rate": round(_mean(1.0 if row.get("judge_fallback_used") else 0.0 for row in rows), 6),
                    "changed_answer_rate": round(_mean(float(row.get("changed_answer_rate") or 0.0) for row in rows), 6),
                    "corrected_count": sum(1 for row in rows if row.get("corrected_by_method")),
                    "harmed_count": sum(1 for row in rows if row.get("harmed_by_method")),
                    "minority_rescue_count": sum(1 for row in rows if row.get("minority_rescue")),
                }
            )
    return {"summary": summary}


def _build_diagnostics(prediction_rows: list[dict[str, Any]], score_rows: list[dict[str, Any]]) -> dict[str, Any]:
    trajectory_rows = [row for row in prediction_rows if row.get("method_name") == "free_mad_lite_llm_trajectory"]
    return {
        "judge_fallback_rate": round(_mean(1.0 if row.get("judge_fallback_used") else 0.0 for row in trajectory_rows), 6),
        "judge_fallback_count": sum(1 for row in score_rows if row.get("judge_fallback_used")),
        "trajectory_score_rows": len(score_rows),
        "anti_conformity_prompt_hash": anti_conformity_prompt_hash(),
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
    experiment: FreeMadLiteExperimentConfig,
    phase_name: str,
    benchmarks,
    protocol: FreeMadLiteProtocolConfig,
) -> tuple[int, int]:
    total_samples = 0
    for benchmark in benchmarks:
        split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
        total_samples += len(load_split_ids(benchmark.slug, split_name))
    calls_per_sample = protocol.agent_count * 3 + 1
    return total_samples * calls_per_sample, total_samples * len(experiment.methods)


def _resolve_split_name(experiment: FreeMadLiteExperimentConfig, phase_name: str, benchmark_slug: str) -> str:
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
        agent_turns=root / "agent_turns.jsonl",
        debate_messages=root / "debate_messages.jsonl",
        trajectory_scores=root / "trajectory_scores.jsonl",
        final_predictions=root / "final_predictions.jsonl",
        metrics=root / "metrics.json",
        diagnostics=root / "diagnostics.json",
        progress=root / "progress.json",
        run_summary=root / "run_summary.json",
        run_validation=root / "run_validation.json",
        report_markdown=root / "report.md",
        paper_summary=root / "paper_summary.csv",
    )


def _debate_message_tokens(initial_rows: list[dict[str, Any]]) -> float:
    # 这里统计可见 peer message 的近似通信量：每个 agent 的回答会被另外两个 agent 看到。
    text_tokens = 0
    for row in initial_rows:
        rendered = json.dumps(
            {
                "answer": row.get("normalized_answer"),
                "reasoning": row.get("reasoning"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        text_tokens += max(1, len(rendered) // 4)
    return float(text_tokens * max(0, len(initial_rows) - 1))


def _changed_answer_rate(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return round(sum(1 for row in rows if row.get("changed_answer")) / len(rows), 6)


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

