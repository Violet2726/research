"""多智能体实验主运行链路。

本模块把 Vanilla MAD 及其等预算控制方法组织成完整实验流程，
包括共享样本选择、setup 解析、agent turn 执行、debate 消息落盘、
题级投票聚合、成本拆分与最终报告/校验产物生成。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
from hashlib import sha256
from pathlib import Path
import json
from typing import Any

from dotenv import load_dotenv

from experiment_core.cache import CachedResponse, RequestCache, RequestCacheRouter, build_request_cache_key, json_dump
from experiment_core.datasets import DatasetSample, load_split_ids, select_samples
from experiment_core.evaluation import aggregate_majority, normalize_prediction, score_prediction
from experiment_core.no_comm_controls import run_no_comm_control_batch
from experiment_core.providers import OpenAICompatibleProvider, ProviderRequestError, build_payload, estimate_request_tokens
from experiment_core.rate_limits import SlidingWindowRateLimiter
from experiment_core.runtime import RunProgressTracker, build_run_id
from experiment_core.structured_output import (
    ARTIFACT_VERSION,
    OUTPUT_MODE_CORE,
    validate_or_recover_structured_output,
)
from experiment_core.workspace import default_cache_root, default_runs_root
from multi_agent.config import (
    ExperimentSetup,
    MultiAgentExperimentConfig,
    ProtocolConfig,
    RosterConfig,
    load_benchmarks,
    load_control_catalog,
    load_protocol_config,
    load_roster_config,
    phase_metadata,
)
from multi_agent.prompting import build_debate_messages, build_initial_messages
from multi_agent.reporting import report_debate_vs_vote, summarize_run
from multi_agent.validation import validate_run


@dataclass(frozen=True)
class RunPaths:
    """多智能体运行目录下的固定产物路径集合。"""

    root: Path
    manifest: Path
    agent_turns: Path
    debate_messages: Path
    final_predictions: Path
    metrics: Path
    cost_breakdown: Path
    debate_diagnostics: Path
    run_summary: Path
    run_validation: Path
    progress: Path


@dataclass(frozen=True)
class AgentTurnRecord:
    """单个 agent 在某一轮的执行记录。"""

    run_id: str
    dataset: str
    split: str
    sample_id: str
    method_name: str
    method_type: str
    round_index: int
    agent_id: int
    role: str
    prompt_hash: str
    prediction: str
    score: float | None
    output_status: str
    prompt_tokens: float
    completion_tokens: float
    total_tokens: float
    latency_ms: float
    cache_hit: bool
    request_error: str | None
    visible_peer_count: int
    payload: dict[str, Any]
    assistant_text: str
    provider_reasoning_text: str
    validated_output: dict[str, Any]


@dataclass(frozen=True)
class DebateMessageRecord:
    """一条显式 debate 可见消息。"""

    run_id: str
    dataset: str
    split: str
    sample_id: str
    method_name: str
    round_index: int
    sender_agent_id: int
    recipient_agent_id: int
    sender_answer: str
    sender_reasoning: str


@dataclass(frozen=True)
class FinalPredictionRecord:
    """某题在某种方法下的最终预测记录。"""

    run_id: str
    dataset: str
    split: str
    sample_id: str
    method_name: str
    method_type: str
    model_name: str
    prediction: str
    gold: str
    score: float
    initial_vote_prediction: str | None
    initial_vote_score: float | None
    initial_vote_counts: dict[str, int]
    initial_consensus: bool
    final_vote_prediction: str
    final_vote_score: float
    final_vote_counts: dict[str, int]
    prompt_tokens_per_question: float
    completion_tokens_per_question: float
    total_tokens_per_question: float
    latency_ms_per_question: float
    initial_prompt_tokens_per_question: float
    initial_completion_tokens_per_question: float
    initial_total_tokens_per_question: float
    initial_latency_ms_per_question: float
    debate_prompt_tokens_per_question: float
    debate_completion_tokens_per_question: float
    debate_total_tokens_per_question: float
    debate_latency_ms_per_question: float
    calls_per_question: int
    debate_rounds: int
    agent_count: int
    final_consensus: bool
    initial_disagreement: bool
    vote_flipped: bool
    corrected_by_debate: bool
    harmed_by_debate: bool
    unchanged_correct: bool
    unchanged_wrong: bool


def run_experiment(
    experiment: MultiAgentExperimentConfig,
    phase_name: str,
    backbone,
    run_root: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> Path:
    """执行一个多智能体 phase，并写出完整运行目录。"""
    load_dotenv(".env.local", override=False)
    run_root = run_root or default_runs_root("multi_agent")
    cache_root = cache_root or default_cache_root()
    benchmarks = load_benchmarks(experiment)
    phase = phase_metadata(experiment, phase_name)
    setups = _active_setups(experiment, phase_name)
    controls = load_control_catalog(experiment.control_catalog)
    matched_control_names = sorted({name for setup in setups for name in setup.matched_controls})
    provider = OpenAICompatibleProvider(backbone)
    cache_router = RequestCacheRouter(cache_root)
    cache = cache_router.for_request_target(
        provider=backbone.provider,
        request_model=backbone.model_id,
    )
    limiter = SlidingWindowRateLimiter(
        requests_per_minute=experiment.requests_per_minute_limit,
        tokens_per_minute=experiment.tokens_per_minute_limit,
    )
    run_id = build_run_id(experiment.name, phase_name, backbone.name)
    run_paths = _prepare_run_paths(run_root, experiment.name, phase_name, run_id)
    total_calls, total_predictions = _estimate_work(experiment, phase_name, benchmarks, setups, matched_control_names, controls)
    progress = RunProgressTracker(run_paths.progress, total_calls, total_predictions)

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "family_name": "multi_agent",
        "experiment_name": experiment.name,
        "phase_name": phase_name,
        "primary_model_ref": experiment.primary_model_ref,
        "resolved_model": asdict(backbone),
        "experiment": experiment.name,
        "description": experiment.description,
        "phase": phase_name,
        "phase_metadata": phase,
        "prompt_version": experiment.prompt_version,
        "artifact_version": ARTIFACT_VERSION,
        "backbone": asdict(backbone),
        "benchmarks": [asdict(item) for item in benchmarks],
        "setups": [
            {
                "name": setup.name,
                "protocol": asdict(load_protocol_config(setup.protocol)),
                "roster": asdict(load_roster_config(setup.roster)),
                "matched_controls": setup.matched_controls,
            }
            for setup in setups
        ],
        "control_methods": {name: asdict(controls[name]) for name in matched_control_names},
        "total_planned_calls": total_calls,
        "total_planned_predictions": total_predictions,
    }
    run_paths.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    all_turns: list[dict[str, Any]] = []
    debate_messages: list[dict[str, Any]] = []
    final_predictions: list[dict[str, Any]] = []

    with (
        run_paths.agent_turns.open("w", encoding="utf-8") as turn_handle,
        run_paths.debate_messages.open("w", encoding="utf-8") as debate_handle,
        run_paths.final_predictions.open("w", encoding="utf-8") as prediction_handle,
    ):
        for benchmark in benchmarks:
            split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
            samples = _load_selected_samples(benchmark, split_name)

            for setup in setups:
                protocol = load_protocol_config(setup.protocol)
                roster = load_roster_config(setup.roster)
                mad_results = _run_mad_setup_batch(
                    run_id=run_id,
                    benchmark_slug=benchmark.slug,
                    split_name=split_name,
                    samples=samples,
                    setup=setup,
                    protocol=protocol,
                    roster=roster,
                    backbone=backbone,
                    provider=provider,
                    cache=cache,
                    limiter=limiter,
                    global_seed=experiment.global_seed,
                    prompt_version=experiment.prompt_version,
                    max_concurrent_requests=experiment.max_concurrent_requests,
                )
                _write_sample_outputs(
                    sample_results=mad_results,
                    dataset_slug=benchmark.slug,
                    progress=progress,
                    turn_handle=turn_handle,
                    debate_handle=debate_handle,
                    prediction_handle=prediction_handle,
                    all_turns=all_turns,
                    debate_messages=debate_messages,
                    final_predictions=final_predictions,
                )

            for control_name in matched_control_names:
                method = controls[control_name]
                control_results = run_no_comm_control_batch(
                    run_id=run_id,
                    samples=samples,
                    control_name=control_name,
                    method=method,
                    benchmark_slug=benchmark.slug,
                    split_name=split_name,
                    backbone=backbone,
                    provider=provider,
                    cache=cache,
                    limiter=limiter,
                    global_seed=experiment.global_seed,
                    prompt_version=experiment.prompt_version,
                    max_concurrent_requests=experiment.max_concurrent_requests,
                    build_messages=build_initial_messages,
                    execute_turn=_execute_turn,
                    build_prediction_row=_build_control_prediction_row,
                )
                _write_sample_outputs(
                    sample_results=control_results,
                    dataset_slug=benchmark.slug,
                    progress=progress,
                    turn_handle=turn_handle,
                    debate_handle=debate_handle,
                    prediction_handle=prediction_handle,
                    all_turns=all_turns,
                    debate_messages=debate_messages,
                    final_predictions=final_predictions,
                )

    metrics = _build_metrics(final_predictions, experiment, setups)
    diagnostics = _build_debate_diagnostics(final_predictions)
    cost_breakdown = _build_cost_breakdown(all_turns)

    run_paths.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    run_paths.debate_diagnostics.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    run_paths.cost_breakdown.write_text(json.dumps(cost_breakdown, ensure_ascii=False, indent=2), encoding="utf-8")
    run_paths.run_summary.write_text(json.dumps(summarize_run(run_paths.root), ensure_ascii=False, indent=2), encoding="utf-8")
    report_debate_vs_vote(run_paths.root)
    run_paths.run_validation.write_text(json.dumps(validate_run(run_paths.root), ensure_ascii=False, indent=2), encoding="utf-8")
    progress.mark_completed()
    cache_router.close()
    return run_paths.root


def _run_mad_setup_batch(
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    samples: list[DatasetSample],
    setup: ExperimentSetup,
    protocol: ProtocolConfig,
    roster: RosterConfig,
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    global_seed: int,
    prompt_version: str,
    max_concurrent_requests: int,
) -> list[tuple[int, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]]:
    """并发执行同一 setup 下的全部样本。

    每个样本内部仍严格保持 Vanilla MAD 的回合顺序，只在样本之间做并发。
    """
    worker = partial(
        _run_mad_sample,
        run_id=run_id,
        benchmark_slug=benchmark_slug,
        split_name=split_name,
        setup=setup,
        protocol=protocol,
        roster=roster,
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        global_seed=global_seed,
        prompt_version=prompt_version,
    )
    max_workers = max(1, min(max_concurrent_requests, len(samples) or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(worker, sample=sample): sample_index
            for sample_index, sample in enumerate(samples)
        }
        completed: list[tuple[int, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]] = []
        for future in as_completed(future_to_index):
            sample_index = future_to_index[future]
            mad_turn_rows, debate_rows, prediction_row = future.result()
            completed.append((sample_index, mad_turn_rows, debate_rows, prediction_row))
    completed.sort(key=lambda item: item[0])
    return completed


def _write_sample_outputs(
    sample_results: list[tuple[int, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]],
    dataset_slug: str,
    progress: RunProgressTracker,
    turn_handle,
    debate_handle,
    prediction_handle,
    all_turns: list[dict[str, Any]],
    debate_messages: list[dict[str, Any]],
    final_predictions: list[dict[str, Any]],
) -> None:
    """把 worker 返回的样本结果按稳定顺序写盘，并同步更新进度。"""
    for _, turn_rows, debate_rows, prediction_row in sample_results:
        for row in turn_rows:
            turn_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            progress.record_call(row, method_key="method_name")
        for row in debate_rows:
            debate_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        prediction_handle.write(json.dumps(prediction_row, ensure_ascii=False) + "\n")
        progress.record_predictions(1, dataset_slug, prediction_row["method_name"])
        all_turns.extend(turn_rows)
        debate_messages.extend(debate_rows)
        final_predictions.append(prediction_row)


def _run_mad_sample(
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    sample: DatasetSample,
    setup: ExperimentSetup,
    protocol: ProtocolConfig,
    roster: RosterConfig,
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    global_seed: int,
    prompt_version: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """运行单个样本上的 Vanilla MAD 协议。"""
    turn_rows: list[dict[str, Any]] = []
    debate_rows: list[dict[str, Any]] = []

    initial_turns: list[dict[str, Any]] = []
    for agent_id in range(1, roster.agent_count + 1):
        messages = build_initial_messages(sample, agent_id, prompt_version=prompt_version)
        initial_turns.append(
            _execute_turn(
                run_id=run_id,
                dataset=benchmark_slug,
                split_name=split_name,
                sample=sample,
                method_name=setup.name,
                method_type="mad",
                round_index=0,
                agent_id=agent_id,
                role="initial",
                visible_peer_count=0,
                messages=messages,
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
    turn_rows.extend(initial_turns)
    previous_round = initial_turns

    for round_index in range(1, protocol.debate_rounds + 1):
        current_round: list[dict[str, Any]] = []
        for recipient_id in range(1, roster.agent_count + 1):
            recipient_previous = previous_round[recipient_id - 1]
            peer_messages: list[dict[str, str]] = []
            for sender in previous_round:
                if sender["agent_id"] == recipient_id:
                    continue
                peer_messages.append(
                    {
                        "agent": f"agent_{sender['agent_id']}",
                        "answer": str(sender["validated_output"].get("final_answer", "")).strip(),
                        "reasoning": str(sender["validated_output"].get("reasoning", "")).strip(),
                    }
                )
                debate_rows.append(
                    asdict(
                        DebateMessageRecord(
                            run_id=run_id,
                            dataset=benchmark_slug,
                            split=split_name,
                            sample_id=sample.sample_id,
                            method_name=setup.name,
                            round_index=round_index,
                            sender_agent_id=sender["agent_id"],
                            recipient_agent_id=recipient_id,
                            sender_answer=str(sender["validated_output"].get("final_answer", "")).strip(),
                            sender_reasoning=str(sender["validated_output"].get("reasoning", "")).strip(),
                        )
                    )
                )
            messages = build_debate_messages(
                sample=sample,
                agent_id=recipient_id,
                round_index=round_index,
                previous_reasoning=str(recipient_previous["validated_output"].get("reasoning", "")).strip(),
                previous_answer=str(recipient_previous["validated_output"].get("final_answer", "")).strip(),
                peer_messages=peer_messages,
                prompt_version=prompt_version,
            )
            current_round.append(
                _execute_turn(
                    run_id=run_id,
                    dataset=benchmark_slug,
                    split_name=split_name,
                    sample=sample,
                    method_name=setup.name,
                    method_type="mad",
                    round_index=round_index,
                    agent_id=recipient_id,
                    role="debate",
                    visible_peer_count=len(peer_messages),
                    messages=messages,
                    backbone=backbone,
                    provider=provider,
                    cache=cache,
                    limiter=limiter,
                    temperature=protocol.debate_temperature,
                    top_p=protocol.top_p,
                    max_output_tokens=protocol.max_output_tokens,
                    seed=global_seed + recipient_id + round_index * 100,
                )
            )
        turn_rows.extend(current_round)
        previous_round = current_round

    initial_answers = [row["normalized_answer"] for row in initial_turns]
    final_answers = [row["normalized_answer"] for row in previous_round]
    initial_vote, initial_vote_counts = aggregate_majority(initial_answers)
    final_vote, final_vote_counts = aggregate_majority(final_answers)
    initial_vote_score = score_prediction(benchmark_slug, initial_vote, sample.reference_answer)
    final_vote_score = score_prediction(benchmark_slug, final_vote, sample.reference_answer)
    initial_consensus = len(set(initial_answers)) == 1
    final_consensus = len(set(final_answers)) == 1
    initial_disagreement = len(set(initial_answers)) > 1
    initial_prompt_tokens = sum(float(row["prompt_tokens"]) for row in initial_turns)
    initial_completion_tokens = sum(float(row["completion_tokens"]) for row in initial_turns)
    initial_total_tokens = sum(float(row["total_tokens"]) for row in initial_turns)
    initial_latency = sum(float(row["latency_ms"]) for row in initial_turns)
    debate_turns = [row for row in turn_rows if row["role"] == "debate"]
    debate_prompt_tokens = sum(float(row["prompt_tokens"]) for row in debate_turns)
    debate_completion_tokens = sum(float(row["completion_tokens"]) for row in debate_turns)
    debate_total_tokens = sum(float(row["total_tokens"]) for row in debate_turns)
    debate_latency = sum(float(row["latency_ms"]) for row in debate_turns)
    question_prompt_tokens = initial_prompt_tokens + debate_prompt_tokens
    question_completion_tokens = initial_completion_tokens + debate_completion_tokens
    question_total_tokens = initial_total_tokens + debate_total_tokens
    question_latency = initial_latency + debate_latency
    corrected_by_debate = initial_vote_score < 1.0 and final_vote_score == 1.0
    harmed_by_debate = initial_vote_score == 1.0 and final_vote_score < 1.0
    unchanged_correct = initial_vote_score == 1.0 and final_vote_score == 1.0
    unchanged_wrong = initial_vote_score < 1.0 and final_vote_score < 1.0
    prediction_row = asdict(
        FinalPredictionRecord(
            run_id=run_id,
            dataset=benchmark_slug,
            split=split_name,
            sample_id=sample.sample_id,
            method_name=setup.name,
            method_type="mad",
            model_name=backbone.name,
            prediction=final_vote,
            gold=sample.reference_answer,
            score=final_vote_score,
            initial_vote_prediction=initial_vote,
            initial_vote_score=initial_vote_score,
            initial_vote_counts=initial_vote_counts,
            initial_consensus=initial_consensus,
            final_vote_prediction=final_vote,
            final_vote_score=final_vote_score,
            final_vote_counts=final_vote_counts,
            prompt_tokens_per_question=question_prompt_tokens,
            completion_tokens_per_question=question_completion_tokens,
            total_tokens_per_question=question_total_tokens,
            latency_ms_per_question=question_latency,
            initial_prompt_tokens_per_question=initial_prompt_tokens,
            initial_completion_tokens_per_question=initial_completion_tokens,
            initial_total_tokens_per_question=initial_total_tokens,
            initial_latency_ms_per_question=initial_latency,
            debate_prompt_tokens_per_question=debate_prompt_tokens,
            debate_completion_tokens_per_question=debate_completion_tokens,
            debate_total_tokens_per_question=debate_total_tokens,
            debate_latency_ms_per_question=debate_latency,
            calls_per_question=roster.agent_count * (1 + protocol.debate_rounds),
            debate_rounds=protocol.debate_rounds,
            agent_count=roster.agent_count,
            final_consensus=final_consensus,
            initial_disagreement=initial_disagreement,
            vote_flipped=initial_vote != final_vote,
            corrected_by_debate=corrected_by_debate,
            harmed_by_debate=harmed_by_debate,
            unchanged_correct=unchanged_correct,
            unchanged_wrong=unchanged_wrong,
        )
    )
    prediction_row["vote_counts"] = final_vote_counts
    return turn_rows, debate_rows, prediction_row


def _build_control_prediction_row(
    *,
    control_name: str,
    method,
    sample: DatasetSample,
    final_vote: str,
    final_score: float,
    vote_counts: dict[str, int],
    final_consensus: bool,
    turn_rows: list[dict[str, Any]],
    backbone,
    benchmark_slug: str,
    split_name: str,
    run_id: str,
) -> dict[str, Any]:
    """Build the final prediction row for a shared no-communication control."""
    prompt_tokens = sum(float(row["prompt_tokens"]) for row in turn_rows)
    completion_tokens = sum(float(row["completion_tokens"]) for row in turn_rows)
    total_tokens = sum(float(row["total_tokens"]) for row in turn_rows)
    latency_ms = sum(float(row["latency_ms"]) for row in turn_rows)
    prediction_row = asdict(
        FinalPredictionRecord(
            run_id=run_id,
            dataset=benchmark_slug,
            split=split_name,
            sample_id=sample.sample_id,
            method_name=control_name,
            method_type="control",
            model_name=backbone.name,
            prediction=final_vote,
            gold=sample.reference_answer,
            score=final_score,
            initial_vote_prediction=final_vote,
            initial_vote_score=final_score,
            initial_vote_counts=vote_counts,
            initial_consensus=final_consensus,
            final_vote_prediction=final_vote,
            final_vote_score=final_score,
            final_vote_counts=vote_counts,
            prompt_tokens_per_question=prompt_tokens,
            completion_tokens_per_question=completion_tokens,
            total_tokens_per_question=total_tokens,
            latency_ms_per_question=latency_ms,
            initial_prompt_tokens_per_question=prompt_tokens,
            initial_completion_tokens_per_question=completion_tokens,
            initial_total_tokens_per_question=total_tokens,
            initial_latency_ms_per_question=latency_ms,
            debate_prompt_tokens_per_question=0.0,
            debate_completion_tokens_per_question=0.0,
            debate_total_tokens_per_question=0.0,
            debate_latency_ms_per_question=0.0,
            calls_per_question=method.budget_calls,
            debate_rounds=0,
            agent_count=1 if method.family == "cot" else method.budget_calls,
            final_consensus=final_consensus,
            initial_disagreement=False,
            vote_flipped=False,
            corrected_by_debate=False,
            harmed_by_debate=False,
            unchanged_correct=final_score == 1.0,
            unchanged_wrong=final_score < 1.0,
        )
    )
    prediction_row["vote_counts"] = vote_counts
    return prediction_row


def _execute_turn(
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    method_name: str,
    method_type: str,
    round_index: int,
    agent_id: int,
    role: str,
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
) -> dict[str, Any]:
    """执行单次 agent turn，并统一返回日志行结构。"""
    payload = build_payload(
        config=backbone,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        seed=seed,
    )
    prompt_hash = _prompt_hash(messages)
    cache_key = build_request_cache_key(
        provider=backbone.provider,
        request_model=backbone.model_id,
        base_url=backbone.base_url,
        chat_path=backbone.chat_path,
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
        validated_output = {}
        output_status = "request_fail"
        final_answer = ""
    else:
        try:
            validated_output = validate_or_recover_structured_output(
                str(response_payload.get("assistant_text") or ""),
                OUTPUT_MODE_CORE,
                provider_reasoning_text=str(response_payload.get("provider_reasoning_text") or ""),
            )
            output_status = "ok"
            final_answer = validated_output["final_answer"]
        except Exception:
            validated_output = {}
            output_status = "schema_fail"
            final_answer = ""

    usage = response_payload.get("usage_reported") or response_payload.get("usage_estimated") or {}
    return asdict(
        AgentTurnRecord(
            run_id=run_id,
            dataset=dataset,
            split=split_name,
            sample_id=sample.sample_id,
            method_name=method_name,
            method_type=method_type,
            round_index=round_index,
            agent_id=agent_id,
            role=role,
            prompt_hash=prompt_hash,
            prediction=normalize_prediction(dataset, final_answer),
            score=None,
            output_status=output_status,
            prompt_tokens=float(usage.get("prompt_tokens") or 0.0),
            completion_tokens=float(usage.get("completion_tokens") or 0.0),
            total_tokens=float(usage.get("total_tokens") or 0.0),
            latency_ms=float(response_payload.get("latency_ms") or 0.0),
            cache_hit=cache_hit,
            request_error=request_error,
            visible_peer_count=visible_peer_count,
            payload=payload,
            assistant_text=response_payload.get("assistant_text", ""),
            provider_reasoning_text=response_payload.get("provider_reasoning_text", ""),
            validated_output=validated_output,
        )
    ) | {"normalized_answer": normalize_prediction(dataset, final_answer)}


def _build_metrics(
    prediction_rows: list[dict[str, Any]],
    experiment: MultiAgentExperimentConfig,
    setups: list[ExperimentSetup],
) -> dict[str, Any]:
    """把最终题级预测聚合成方法级 summary。"""
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in prediction_rows:
        key = (row["dataset"], row["model_name"], row["method_name"])
        grouped.setdefault(key, []).append(row)

    setup_map = {item.name: item for item in setups}
    summary: list[dict[str, Any]] = []
    for (dataset, model_name, method_name), rows in sorted(grouped.items()):
        accuracy = sum(float(row["score"]) for row in rows) / len(rows)
        total_tokens_mean = sum(float(row["total_tokens_per_question"]) for row in rows) / len(rows)
        row = {
            "dataset": dataset,
            "model_name": model_name,
            "method_name": method_name,
            "method_type": rows[0]["method_type"],
            "prediction_rows": len(rows),
            "accuracy_mean": accuracy,
            "prompt_tokens_mean": sum(float(item["prompt_tokens_per_question"]) for item in rows) / len(rows),
            "completion_tokens_mean": sum(float(item["completion_tokens_per_question"]) for item in rows) / len(rows),
            "total_tokens_mean": total_tokens_mean,
            "calls_per_question_mean": sum(float(item["calls_per_question"]) for item in rows) / len(rows),
            "latency_ms_mean": sum(float(item["latency_ms_per_question"]) for item in rows) / len(rows),
            "accuracy_per_1k_tokens": (accuracy / total_tokens_mean * 1000) if total_tokens_mean else 0.0,
            "debate_rounds": rows[0]["debate_rounds"],
            "agent_count": rows[0]["agent_count"],
        }
        if method_name in setup_map:
            controls = setup_map[method_name].matched_controls
            row["matched_vote_control"] = next((name for name in controls if name.startswith("mv_")), None)
        summary.append(row)

    by_lookup = {(row["dataset"], row["model_name"], row["method_name"]): row for row in summary}
    for row in summary:
        if row["method_type"] != "mad":
            continue
        vote_name = row.get("matched_vote_control")
        vote_row = by_lookup.get((row["dataset"], row["model_name"], vote_name)) if vote_name else None
        row["debate_gain_over_vote"] = round(row["accuracy_mean"] - vote_row["accuracy_mean"], 6) if vote_row else None
        row["token_overhead_vs_vote"] = (
            round((row["total_tokens_mean"] - vote_row["total_tokens_mean"]) / vote_row["total_tokens_mean"], 6)
            if vote_row and vote_row["total_tokens_mean"]
            else None
        )
    return {"summary": summary}


def _build_cost_breakdown(turn_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总初始回答、debate 和控制方法的 token 成本。"""
    grouped: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in turn_rows:
        key = (row["dataset"], row["method_name"], row["method_type"])
        bucket = grouped.setdefault(
            key,
            {
                "prompt_tokens": 0.0,
                "completion_tokens": 0.0,
                "total_tokens": 0.0,
                "latency_ms": 0.0,
                "turn_count": 0.0,
                "initial_tokens": 0.0,
                "debate_tokens": 0.0,
                "control_tokens": 0.0,
            },
        )
        total_tokens = float(row["total_tokens"])
        bucket["prompt_tokens"] += float(row["prompt_tokens"])
        bucket["completion_tokens"] += float(row["completion_tokens"])
        bucket["total_tokens"] += total_tokens
        bucket["latency_ms"] += float(row["latency_ms"])
        bucket["turn_count"] += 1
        if row["role"] == "initial":
            bucket["initial_tokens"] += total_tokens
        elif row["role"] == "debate":
            bucket["debate_tokens"] += total_tokens
        else:
            bucket["control_tokens"] += total_tokens

    rows = []
    for (dataset, method_name, method_type), bucket in sorted(grouped.items()):
        rows.append({"dataset": dataset, "method_name": method_name, "method_type": method_type} | bucket)
    return {"rows": rows}


def _build_debate_diagnostics(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """构建 debate 诊断指标。"""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in prediction_rows:
        if row["method_type"] != "mad":
            continue
        grouped.setdefault((row["dataset"], row["method_name"]), []).append(row)

    rows = []
    for (dataset, method_name), rows_for_key in sorted(grouped.items()):
        total = len(rows_for_key)
        rows.append(
            {
                "dataset": dataset,
                "method_name": method_name,
                "question_count": total,
                "initial_disagreement_rate": _ratio(sum(1 for row in rows_for_key if row["initial_disagreement"]), total),
                "post_debate_consensus_rate": _ratio(sum(1 for row in rows_for_key if row["final_consensus"]), total),
                "vote_flip_rate": _ratio(sum(1 for row in rows_for_key if row["vote_flipped"]), total),
                "wrong_consensus_rate": _ratio(
                    sum(1 for row in rows_for_key if row["final_consensus"] and float(row["score"]) < 1.0),
                    total,
                ),
            }
        )
    return {"rows": rows}


def _estimate_work(
    experiment: MultiAgentExperimentConfig,
    phase_name: str,
    benchmarks,
    setups: list[ExperimentSetup],
    matched_control_names: list[str],
    controls,
) -> tuple[int, int]:
    """估算本次多智能体运行的总调用量与总预测量。"""
    phase = phase_metadata(experiment, phase_name)
    total_calls = 0
    total_predictions = 0
    for benchmark in benchmarks:
        split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
        sample_count = len(load_split_ids(benchmark.slug, split_name))
        for setup in setups:
            protocol = load_protocol_config(setup.protocol)
            roster = load_roster_config(setup.roster)
            total_calls += sample_count * roster.agent_count * (1 + protocol.debate_rounds)
            total_predictions += sample_count
        for name in matched_control_names:
            total_calls += sample_count * controls[name].budget_calls
            total_predictions += sample_count
    return total_calls, total_predictions


def _active_setups(experiment: MultiAgentExperimentConfig, phase_name: str) -> list[ExperimentSetup]:
    """解析当前 phase 实际启用的 setup 列表。"""
    phase = phase_metadata(experiment, phase_name)
    requested = set(phase["setups"])
    available = {item.name: item for item in experiment.setups}
    missing = sorted(requested - set(available))
    if missing:
        raise RuntimeError(f"Unknown multi-agent setups for phase {phase_name}: {', '.join(missing)}")
    return [available[name] for name in phase["setups"]]


def _resolve_split_name(experiment: MultiAgentExperimentConfig, phase_name: str, benchmark_slug: str) -> str:
    """解析当前 benchmark 在该 phase 下对应的冻结 split 名称。"""
    phase = phase_metadata(experiment, phase_name)
    if "split_overrides" in phase:
        return phase["split_overrides"][benchmark_slug]
    return phase["split_suffix"]


def _load_selected_samples(benchmark, split_name: str) -> list[DatasetSample]:
    """按冻结 split 选择本轮要跑的样本。"""
    return select_samples(benchmark, split_name)


def _prepare_run_paths(run_root: str | Path, experiment_name: str, phase_name: str, run_id: str) -> RunPaths:
    """创建多智能体运行目录和固定产物路径。"""
    root = Path(run_root) / experiment_name / phase_name / run_id
    root.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        root=root,
        manifest=root / "manifest.json",
        agent_turns=root / "agent_turns.jsonl",
        debate_messages=root / "debate_messages.jsonl",
        final_predictions=root / "final_predictions.jsonl",
        metrics=root / "metrics.json",
        cost_breakdown=root / "cost_breakdown.json",
        debate_diagnostics=root / "debate_diagnostics.json",
        run_summary=root / "run_summary.json",
        run_validation=root / "run_validation.json",
        progress=root / "progress.json",
    )


def _prompt_hash(messages: list[dict[str, str]]) -> str:
    """对 prompt 内容做稳定哈希。"""
    return sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _ratio(numerator: int, denominator: int) -> float:
    """安全计算比例。"""
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)
