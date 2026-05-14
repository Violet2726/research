"""iMAD 的样本级执行、稳定性检测与指标聚合。"""

from __future__ import annotations

from dataclasses import asdict
from functools import partial
from hashlib import sha256
import math
import random
from typing import Any

from research_experiments.core.controls.no_comm_controls import run_no_comm_control_batch
from research_experiments.core.data.datasets import DatasetSample, load_split_ids, select_samples
from research_experiments.core.data.evaluation import aggregate_majority, normalize_prediction, score_prediction
from research_experiments.core.execution.artifacts import BufferedJsonlWriter
from research_experiments.core.execution.cache import RequestCache
from research_experiments.core.execution.providers import OpenAICompatibleProvider
from research_experiments.core.execution.rate_limits import SlidingWindowRateLimiter
from research_experiments.core.execution.runner_common import execute_cached_turn, run_indexed_batch
from research_experiments.core.controls.no_comm_controls import BuildPredictionRowFn
from research_experiments.core.structured_outputs import SCHEMA_ANSWER_CORE
from research_experiments.families.imad.config import DebateMethodSpec, ImadExperimentConfig, ProtocolConfig
from research_experiments.families.imad.prompts import build_debate_messages, build_initial_messages
from research_experiments.families.shared.common import resolve_phase_split_name


def _active_methods(experiment: ImadExperimentConfig) -> list[DebateMethodSpec]:
    """返回当前实验启用的 debate 方法列表。"""

    return list(experiment.methods)


def _resolve_split_name(experiment: ImadExperimentConfig, phase_name: str, benchmark_slug: str) -> str:
    """解析当前 benchmark 在该 phase 下对应的固定 split 名称。"""

    return resolve_phase_split_name(experiment, phase_name, benchmark_slug)


def _load_selected_samples(benchmark, split_name: str) -> list[DatasetSample]:
    """按固定 split 选择本轮样本。"""

    return select_samples(benchmark, split_name)


def _estimate_work(
    experiment: ImadExperimentConfig,
    phase_name: str,
    benchmarks,
    protocol: ProtocolConfig,
    methods: list[DebateMethodSpec],
    controls: dict[str, Any],
) -> tuple[int, int]:
    """估算本轮运行的总调用量与总预测量。"""

    total_calls = 0
    total_predictions = 0
    control_names = sorted({name for method in methods for name in method.matched_controls})
    for benchmark in benchmarks:
        split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
        sample_count = len(load_split_ids(benchmark.slug, split_name))
        for method in methods:
            total_calls += sample_count * protocol.agent_count * (1 + method.round_limit)
            total_predictions += sample_count
        for control_name in control_names:
            total_calls += sample_count * controls[control_name].budget_calls
            total_predictions += sample_count
    return total_calls, total_predictions


def _run_method_batch(
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    samples: list[DatasetSample],
    method: DebateMethodSpec,
    protocol: ProtocolConfig,
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    global_seed: int,
    prompt_version: str,
    max_concurrent_requests: int,
) -> list[tuple[int, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]]:
    """并发执行同一方法下的全部样本。"""

    worker = partial(
        _run_method_sample,
        run_id=run_id,
        benchmark_slug=benchmark_slug,
        split_name=split_name,
        method=method,
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


def _write_sample_outputs(
    *,
    sample_results: list[tuple[int, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]],
    dataset_slug: str,
    progress,
    turn_handle: BufferedJsonlWriter,
    debate_handle: BufferedJsonlWriter,
    round_handle: BufferedJsonlWriter,
    prediction_handle: BufferedJsonlWriter,
    all_turns: list[dict[str, Any]],
    all_round_diagnostics: list[dict[str, Any]],
    final_predictions: list[dict[str, Any]],
) -> None:
    """按稳定顺序落盘一批样本结果并更新进度。"""

    for _, turn_rows, debate_rows, round_rows, prediction_row in sample_results:
        for row in turn_rows:
            turn_handle.write_row(row)
            progress.record_call(row, method_key="method_name")
        for row in debate_rows:
            debate_handle.write_row(row)
        for row in round_rows:
            round_handle.write_row(row)
        prediction_handle.write_row(prediction_row)
        progress.record_predictions(1, dataset_slug, prediction_row["method_name"])
        all_turns.extend(turn_rows)
        all_round_diagnostics.extend(round_rows)
        final_predictions.append(prediction_row)


def _run_method_sample(
    sample: DatasetSample,
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    method: DebateMethodSpec,
    protocol: ProtocolConfig,
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    global_seed: int,
    prompt_version: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """运行单个样本上的固定轮数或自适应 iMAD 协议。"""

    turn_rows: list[dict[str, Any]] = []
    debate_rows: list[dict[str, Any]] = []
    round_diagnostics: list[dict[str, Any]] = []

    initial_turns: list[dict[str, Any]] = []
    for agent_id in range(1, protocol.agent_count + 1):
        messages = build_initial_messages(sample, agent_id, prompt_version=prompt_version)
        initial_turns.append(
            _execute_turn(
                run_id=run_id,
                dataset=benchmark_slug,
                split_name=split_name,
                sample=sample,
                method_name=method.name,
                method_type="mad",
                method_mode=method.mode,
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

    initial_answers = [row["normalized_answer"] for row in initial_turns]
    initial_vote, initial_vote_counts = aggregate_majority(initial_answers)
    initial_vote_score = score_prediction(benchmark_slug, initial_vote, sample.reference_answer)
    initial_consensus = len(set(initial_answers)) == 1
    initial_disagreement = len(set(initial_answers)) > 1

    previous_round = initial_turns
    previous_top_answer: str | None = None
    previous_posterior_samples: list[float] | None = None
    executed_round_count = 0
    stop_reason = ""
    stopped_early = False
    final_round_turns = initial_turns
    last_ks_statistic: float | None = None
    last_posterior_mean: float | None = None
    round_scores: dict[int, float | None] = {1: None, 2: None, 3: None}

    for round_index in range(1, method.round_limit + 1):
        current_round: list[dict[str, Any]] = []
        for recipient_id in range(1, protocol.agent_count + 1):
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
                    {
                        "run_id": run_id,
                        "dataset": benchmark_slug,
                        "split": split_name,
                        "sample_id": sample.sample_id,
                        "method_name": method.name,
                        "method_mode": method.mode,
                        "round_index": round_index,
                        "sender_agent_id": sender["agent_id"],
                        "recipient_agent_id": recipient_id,
                        "sender_answer": str(sender["validated_output"].get("final_answer", "")).strip(),
                        "sender_reasoning": str(sender["validated_output"].get("reasoning", "")).strip(),
                    }
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
                    method_name=method.name,
                    method_type="mad",
                    method_mode=method.mode,
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
        final_round_turns = current_round
        executed_round_count = round_index

        round_answers = [row["normalized_answer"] for row in current_round]
        round_vote, round_vote_counts = aggregate_majority(round_answers)
        round_score = score_prediction(benchmark_slug, round_vote, sample.reference_answer)
        round_scores[round_index] = round_score
        support_count = int(round_vote_counts.get(round_vote, 0))
        posterior_mean = _posterior_mean(support_count=support_count, agent_count=protocol.agent_count)
        posterior_samples = _beta_posterior_samples(
            support_count=support_count,
            agent_count=protocol.agent_count,
            sample_count=protocol.posterior_sample_count,
            seed=_posterior_seed(
                sample_id=sample.sample_id,
                method_name=method.name,
                round_index=round_index,
                global_seed=global_seed,
            ),
        )
        ks_statistic, stability_gate_passed = assess_stability_gate(
            previous_top_answer=previous_top_answer,
            current_top_answer=round_vote,
            previous_posterior_samples=previous_posterior_samples,
            current_posterior_samples=posterior_samples,
            posterior_mean=posterior_mean,
            ks_threshold=protocol.stability_ks_threshold,
            stable_posterior_mean_threshold=protocol.stable_posterior_mean_threshold,
        )
        round_diagnostics.append(
            {
                "run_id": run_id,
                "dataset": benchmark_slug,
                "split": split_name,
                "sample_id": sample.sample_id,
                "method_name": method.name,
                "method_mode": method.mode,
                "round_index": round_index,
                "round_limit": method.round_limit,
                "top_answer": round_vote,
                "support_count": support_count,
                "support_rate": round(support_count / protocol.agent_count, 6),
                "posterior_mean": posterior_mean,
                "ks_statistic": ks_statistic,
                "same_top_as_previous": previous_top_answer == round_vote if previous_top_answer is not None else False,
                "stability_gate_passed": stability_gate_passed,
                "round_score": round_score,
                "stop_triggered": False,
            }
        )
        last_ks_statistic = ks_statistic
        last_posterior_mean = posterior_mean
        previous_top_answer = round_vote
        previous_posterior_samples = posterior_samples

        if method.mode == "adaptive" and round_index >= 2 and stability_gate_passed:
            stopped_early = round_index < method.round_limit
            stop_reason = "stability_gate"
            round_diagnostics[-1]["stop_triggered"] = True
            break

    if not stop_reason:
        if method.mode == "fixed":
            stop_reason = f"fixed_round_limit_r{method.round_limit}"
        else:
            stop_reason = "max_rounds_reached"

    final_answers = [row["normalized_answer"] for row in final_round_turns]
    final_vote, final_vote_counts = aggregate_majority(final_answers)
    final_vote_score = score_prediction(benchmark_slug, final_vote, sample.reference_answer)
    final_consensus = len(set(final_answers)) == 1
    debate_turns = [row for row in turn_rows if row["role"] == "debate"]
    initial_prompt_tokens = sum(float(row["prompt_tokens"]) for row in initial_turns)
    initial_completion_tokens = sum(float(row["completion_tokens"]) for row in initial_turns)
    initial_total_tokens = sum(float(row["total_tokens"]) for row in initial_turns)
    initial_latency = sum(float(row["latency_ms"]) for row in initial_turns)
    debate_prompt_tokens = sum(float(row["prompt_tokens"]) for row in debate_turns)
    debate_completion_tokens = sum(float(row["completion_tokens"]) for row in debate_turns)
    debate_total_tokens = sum(float(row["total_tokens"]) for row in debate_turns)
    debate_latency = sum(float(row["latency_ms"]) for row in debate_turns)
    corrected_by_debate = initial_vote_score < 1.0 and final_vote_score == 1.0
    harmed_by_debate = initial_vote_score == 1.0 and final_vote_score < 1.0
    unchanged_correct = initial_vote_score == 1.0 and final_vote_score == 1.0
    unchanged_wrong = initial_vote_score < 1.0 and final_vote_score < 1.0

    prediction_row = {
        "run_id": run_id,
        "dataset": benchmark_slug,
        "split": split_name,
        "sample_id": sample.sample_id,
        "method_name": method.name,
        "method_type": "mad",
        "method_mode": method.mode,
        "configured_round_limit": method.round_limit,
        "model_name": backbone.name,
        "prediction": final_vote,
        "gold": sample.reference_answer,
        "score": final_vote_score,
        "initial_vote_prediction": initial_vote,
        "initial_vote_score": initial_vote_score,
        "initial_vote_counts": initial_vote_counts,
        "initial_consensus": initial_consensus,
        "final_vote_prediction": final_vote,
        "final_vote_score": final_vote_score,
        "final_vote_counts": final_vote_counts,
        "prompt_tokens_per_question": initial_prompt_tokens + debate_prompt_tokens,
        "completion_tokens_per_question": initial_completion_tokens + debate_completion_tokens,
        "total_tokens_per_question": initial_total_tokens + debate_total_tokens,
        "latency_ms_per_question": initial_latency + debate_latency,
        "initial_prompt_tokens_per_question": initial_prompt_tokens,
        "initial_completion_tokens_per_question": initial_completion_tokens,
        "initial_total_tokens_per_question": initial_total_tokens,
        "initial_latency_ms_per_question": initial_latency,
        "debate_prompt_tokens_per_question": debate_prompt_tokens,
        "debate_completion_tokens_per_question": debate_completion_tokens,
        "debate_total_tokens_per_question": debate_total_tokens,
        "debate_latency_ms_per_question": debate_latency,
        "calls_per_question": protocol.agent_count * (1 + executed_round_count),
        "debate_rounds": executed_round_count,
        "executed_round_count": executed_round_count,
        "agent_count": protocol.agent_count,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "ks_statistic_last": last_ks_statistic,
        "posterior_mean_last": last_posterior_mean,
        "round_1_score": round_scores[1],
        "round_2_score": round_scores[2],
        "round_3_score": round_scores[3],
        "final_consensus": final_consensus,
        "initial_disagreement": initial_disagreement,
        "vote_flipped": initial_vote != final_vote,
        "corrected_by_debate": corrected_by_debate,
        "harmed_by_debate": harmed_by_debate,
        "unchanged_correct": unchanged_correct,
        "unchanged_wrong": unchanged_wrong,
        "vote_counts": final_vote_counts,
    }
    return turn_rows, debate_rows, round_diagnostics, prediction_row


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
    """构造共享无通信控制组的最终预测行。"""

    prompt_tokens = sum(float(row["prompt_tokens"]) for row in turn_rows)
    completion_tokens = sum(float(row["completion_tokens"]) for row in turn_rows)
    total_tokens = sum(float(row["total_tokens"]) for row in turn_rows)
    latency_ms = sum(float(row["latency_ms"]) for row in turn_rows)
    return {
        "run_id": run_id,
        "dataset": benchmark_slug,
        "split": split_name,
        "sample_id": sample.sample_id,
        "method_name": control_name,
        "method_type": "control",
        "method_mode": "control",
        "configured_round_limit": 0,
        "model_name": backbone.name,
        "prediction": final_vote,
        "gold": sample.reference_answer,
        "score": final_score,
        "initial_vote_prediction": final_vote,
        "initial_vote_score": final_score,
        "initial_vote_counts": vote_counts,
        "initial_consensus": final_consensus,
        "final_vote_prediction": final_vote,
        "final_vote_score": final_score,
        "final_vote_counts": vote_counts,
        "prompt_tokens_per_question": prompt_tokens,
        "completion_tokens_per_question": completion_tokens,
        "total_tokens_per_question": total_tokens,
        "latency_ms_per_question": latency_ms,
        "initial_prompt_tokens_per_question": prompt_tokens,
        "initial_completion_tokens_per_question": completion_tokens,
        "initial_total_tokens_per_question": total_tokens,
        "initial_latency_ms_per_question": latency_ms,
        "debate_prompt_tokens_per_question": 0.0,
        "debate_completion_tokens_per_question": 0.0,
        "debate_total_tokens_per_question": 0.0,
        "debate_latency_ms_per_question": 0.0,
        "calls_per_question": method.budget_calls,
        "debate_rounds": 0,
        "executed_round_count": 0,
        "agent_count": 1 if method.family == "cot" else method.budget_calls,
        "stopped_early": False,
        "stop_reason": "control_no_debate",
        "ks_statistic_last": None,
        "posterior_mean_last": None,
        "round_1_score": None,
        "round_2_score": None,
        "round_3_score": None,
        "final_consensus": final_consensus,
        "initial_disagreement": False,
        "vote_flipped": False,
        "corrected_by_debate": False,
        "harmed_by_debate": False,
        "unchanged_correct": final_score == 1.0,
        "unchanged_wrong": final_score < 1.0,
        "vote_counts": vote_counts,
    }


def _execute_turn(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    method_name: str,
    method_type: str,
    method_mode: str | None = None,
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
    """执行单次 agent turn，并返回统一日志结构。"""

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
        schema_id=SCHEMA_ANSWER_CORE,
    )
    final_answer = str(result.validated_output.get("final_answer") or "")
    normalized = normalize_prediction(dataset, final_answer) if final_answer else ""
    turn_score = score_prediction(dataset, normalized, sample.reference_answer) if normalized else None
    row = {
        "run_id": run_id,
        "dataset": dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "method_name": method_name,
        "method_type": method_type,
        "method_mode": method_mode or ("control" if method_type == "control" else "fixed"),
        "round_index": round_index,
        "agent_id": agent_id,
        "role": role,
        "prompt_hash": result.prompt_hash,
        "prediction": normalized,
        "score": turn_score,
        "output_status": result.output_status,
        "prompt_tokens": float(result.usage.get("prompt_tokens") or 0.0),
        "completion_tokens": float(result.usage.get("completion_tokens") or 0.0),
        "total_tokens": float(result.usage.get("total_tokens") or 0.0),
        "latency_ms": float(result.response_payload.get("latency_ms") or 0.0),
        "cache_hit": result.cache_hit,
        "request_error": result.request_error,
        "visible_peer_count": visible_peer_count,
        "payload": result.payload,
        "assistant_text": result.response_payload.get("assistant_text", ""),
        "provider_reasoning_text": result.response_payload.get("provider_reasoning_text", ""),
        "validated_output": result.validated_output,
        "normalized_answer": normalized,
    }
    return row


def _build_metrics(
    prediction_rows: list[dict[str, Any]],
    methods: list[DebateMethodSpec],
) -> dict[str, Any]:
    """把题级预测聚合成 iMAD 的 summary 指标。"""

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in prediction_rows:
        key = (row["dataset"], row["model_name"], row["method_name"])
        grouped.setdefault(key, []).append(row)

    method_map = {method.name: method for method in methods}
    summary: list[dict[str, Any]] = []
    for (dataset, model_name, method_name), rows in sorted(grouped.items()):
        accuracy = _mean(float(row["score"]) for row in rows)
        total_tokens_mean = _mean(float(row["total_tokens_per_question"]) for row in rows)
        debate_total_tokens_mean = _mean(float(row["debate_total_tokens_per_question"]) for row in rows)
        method_spec = method_map.get(method_name)
        matched_vote_control = method_spec.matched_controls[0] if method_spec and method_spec.matched_controls else None
        method_mode = str(rows[0].get("method_mode") or "")
        summary.append(
            {
                "dataset": dataset,
                "model_name": model_name,
                "method_name": method_name,
                "method_type": rows[0]["method_type"],
                "method_mode": method_mode,
                "configured_round_limit": rows[0]["configured_round_limit"],
                "prediction_rows": len(rows),
                "question_count": len(rows),
                "accuracy_mean": accuracy,
                "prompt_tokens_mean": _mean(float(item["prompt_tokens_per_question"]) for item in rows),
                "completion_tokens_mean": _mean(float(item["completion_tokens_per_question"]) for item in rows),
                "total_tokens_mean": total_tokens_mean,
                "communication_tokens_mean": debate_total_tokens_mean,
                "calls_per_question_mean": _mean(float(item["calls_per_question"]) for item in rows),
                "latency_ms_mean": _mean(float(item["latency_ms_per_question"]) for item in rows),
                "accuracy_per_1k_tokens": (accuracy / total_tokens_mean * 1000) if total_tokens_mean else 0.0,
                "debate_rounds": _mean(float(item["debate_rounds"]) for item in rows),
                "executed_round_count_mean": _mean(float(item["executed_round_count"]) for item in rows),
                "stopped_early_rate": _mean(1.0 if item["stopped_early"] else 0.0 for item in rows),
                "stability_stop_rate": _mean(1.0 if item["stop_reason"] == "stability_gate" else 0.0 for item in rows),
                "ks_statistic_last_mean": _optional_mean(item.get("ks_statistic_last") for item in rows),
                "posterior_mean_last_mean": _optional_mean(item.get("posterior_mean_last") for item in rows),
                "matched_vote_control": matched_vote_control,
            }
        )

    direct_overall_keys = {
        (row["model_name"], row["method_name"])
        for row in summary
        if row["dataset"] == "overall"
    }
    grouped_overall: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in summary:
        if row["dataset"] == "overall":
            continue
        key = (row["model_name"], row["method_name"])
        if key in direct_overall_keys:
            continue
        grouped_overall.setdefault(key, []).append(row)
    for (model_name, method_name), rows in sorted(grouped_overall.items()):
        summary.append(
            {
                "dataset": "overall",
                "model_name": model_name,
                "method_name": method_name,
                "method_type": rows[0]["method_type"],
                "method_mode": rows[0]["method_mode"],
                "configured_round_limit": rows[0]["configured_round_limit"],
                "prediction_rows": sum(int(row["prediction_rows"]) for row in rows),
                "question_count": sum(int(row["question_count"]) for row in rows),
                "accuracy_mean": _mean(float(row["accuracy_mean"]) for row in rows),
                "prompt_tokens_mean": _mean(float(row["prompt_tokens_mean"]) for row in rows),
                "completion_tokens_mean": _mean(float(row["completion_tokens_mean"]) for row in rows),
                "total_tokens_mean": _mean(float(row["total_tokens_mean"]) for row in rows),
                "communication_tokens_mean": _mean(float(row["communication_tokens_mean"]) for row in rows),
                "calls_per_question_mean": _mean(float(row["calls_per_question_mean"]) for row in rows),
                "latency_ms_mean": _mean(float(row["latency_ms_mean"]) for row in rows),
                "accuracy_per_1k_tokens": _mean(float(row["accuracy_per_1k_tokens"]) for row in rows),
                "debate_rounds": _mean(float(row["debate_rounds"]) for row in rows),
                "executed_round_count_mean": _mean(float(row["executed_round_count_mean"]) for row in rows),
                "stopped_early_rate": _mean(float(row["stopped_early_rate"]) for row in rows),
                "stability_stop_rate": _mean(float(row["stability_stop_rate"]) for row in rows),
                "ks_statistic_last_mean": _optional_mean(row.get("ks_statistic_last_mean") for row in rows),
                "posterior_mean_last_mean": _optional_mean(row.get("posterior_mean_last_mean") for row in rows),
                "matched_vote_control": next((row.get("matched_vote_control") for row in rows if row.get("matched_vote_control")), None),
            }
        )

    by_lookup = {(row["dataset"], row["model_name"], row["method_name"]): row for row in summary}
    for row in summary:
        vote_name = row.get("matched_vote_control")
        vote_row = by_lookup.get((row["dataset"], row["model_name"], vote_name)) if vote_name else None
        row["debate_gain_over_vote"] = round(row["accuracy_mean"] - vote_row["accuracy_mean"], 6) if vote_row else None
        row["token_overhead_vs_vote"] = (
            round((row["total_tokens_mean"] - vote_row["total_tokens_mean"]) / vote_row["total_tokens_mean"], 6)
            if vote_row and vote_row["total_tokens_mean"]
            else None
        )
    return {"summary": summary}


def _build_stability_diagnostics(
    prediction_rows: list[dict[str, Any]],
    round_diagnostic_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """构建稳定性检测的汇总诊断。"""

    summary_grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in prediction_rows:
        if row["method_type"] != "mad":
            continue
        summary_grouped.setdefault((row["dataset"], row["method_name"]), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (dataset, method_name), rows in sorted(summary_grouped.items()):
        summary_rows.append(
            {
                "dataset": dataset,
                "method_name": method_name,
                "method_mode": rows[0]["method_mode"],
                "prediction_rows": len(rows),
                "executed_round_count_mean": _mean(float(row["executed_round_count"]) for row in rows),
                "stopped_early_rate": _mean(1.0 if row["stopped_early"] else 0.0 for row in rows),
                "stability_stop_rate": _mean(1.0 if row["stop_reason"] == "stability_gate" else 0.0 for row in rows),
                "max_round_reached_rate": _mean(1.0 if row["stop_reason"] == "max_rounds_reached" else 0.0 for row in rows),
                "ks_statistic_last_mean": _optional_mean(row.get("ks_statistic_last") for row in rows),
                "posterior_mean_last_mean": _optional_mean(row.get("posterior_mean_last") for row in rows),
            }
        )

    grouped_overall_summary: dict[str, list[dict[str, Any]]] = {}
    for row in summary_rows:
        if row["dataset"] == "overall":
            continue
        grouped_overall_summary.setdefault(str(row["method_name"]), []).append(row)
    for method_name, rows in sorted(grouped_overall_summary.items()):
        summary_rows.append(
            {
                "dataset": "overall",
                "method_name": method_name,
                "method_mode": rows[0]["method_mode"],
                "prediction_rows": sum(int(row["prediction_rows"]) for row in rows),
                "executed_round_count_mean": _mean(float(row["executed_round_count_mean"]) for row in rows),
                "stopped_early_rate": _mean(float(row["stopped_early_rate"]) for row in rows),
                "stability_stop_rate": _mean(float(row["stability_stop_rate"]) for row in rows),
                "max_round_reached_rate": _mean(float(row["max_round_reached_rate"]) for row in rows),
                "ks_statistic_last_mean": _optional_mean(row.get("ks_statistic_last_mean") for row in rows),
                "posterior_mean_last_mean": _optional_mean(row.get("posterior_mean_last_mean") for row in rows),
            }
        )

    round_grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in round_diagnostic_rows:
        key = (str(row["dataset"]), str(row["method_name"]), int(row["round_index"]))
        round_grouped.setdefault(key, []).append(row)

    round_rows: list[dict[str, Any]] = []
    for (dataset, method_name, round_index), rows in sorted(round_grouped.items()):
        round_rows.append(
            {
                "dataset": dataset,
                "method_name": method_name,
                "round_index": round_index,
                "sample_count": len(rows),
                "mean_support_rate": _mean(float(row["support_rate"]) for row in rows),
                "mean_posterior_mean": _mean(float(row["posterior_mean"]) for row in rows),
                "same_top_rate": _mean(1.0 if row["same_top_as_previous"] else 0.0 for row in rows),
                "stability_gate_pass_rate": _mean(1.0 if row["stability_gate_passed"] else 0.0 for row in rows),
                "mean_ks_statistic": _optional_mean(row.get("ks_statistic") for row in rows),
            }
        )
    grouped_overall_rounds: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in round_rows:
        if row["dataset"] == "overall":
            continue
        grouped_overall_rounds.setdefault((str(row["method_name"]), int(row["round_index"])), []).append(row)
    for (method_name, round_index), rows in sorted(grouped_overall_rounds.items()):
        round_rows.append(
            {
                "dataset": "overall",
                "method_name": method_name,
                "round_index": round_index,
                "sample_count": sum(int(row["sample_count"]) for row in rows),
                "mean_support_rate": _mean(float(row["mean_support_rate"]) for row in rows),
                "mean_posterior_mean": _mean(float(row["mean_posterior_mean"]) for row in rows),
                "same_top_rate": _mean(float(row["same_top_rate"]) for row in rows),
                "stability_gate_pass_rate": _mean(float(row["stability_gate_pass_rate"]) for row in rows),
                "mean_ks_statistic": _optional_mean(row.get("mean_ks_statistic") for row in rows),
            }
        )
    return {"summary_rows": summary_rows, "round_rows": round_rows}


def _build_cost_breakdown(turn_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总初始求解与 debate 的 token 成本。"""

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


def assess_stability_gate(
    *,
    previous_top_answer: str | None,
    current_top_answer: str,
    previous_posterior_samples: list[float] | None,
    current_posterior_samples: list[float],
    posterior_mean: float,
    ks_threshold: float,
    stable_posterior_mean_threshold: float,
) -> tuple[float | None, bool]:
    """判断当前轮是否满足自适应停止条件。"""

    if previous_top_answer is None or previous_posterior_samples is None:
        return None, False
    ks_statistic = _kolmogorov_smirnov_statistic(previous_posterior_samples, current_posterior_samples)
    stable = (
        previous_top_answer == current_top_answer
        and ks_statistic <= ks_threshold
        and posterior_mean >= stable_posterior_mean_threshold
    )
    return ks_statistic, stable


def _posterior_mean(*, support_count: int, agent_count: int) -> float:
    alpha = 1 + support_count
    beta = 1 + max(agent_count - support_count, 0)
    return round(alpha / (alpha + beta), 6)


def _beta_posterior_samples(*, support_count: int, agent_count: int, sample_count: int, seed: int) -> list[float]:
    alpha = 1 + support_count
    beta = 1 + max(agent_count - support_count, 0)
    rng = random.Random(seed)
    return [rng.betavariate(alpha, beta) for _ in range(max(1, sample_count))]


def _posterior_seed(*, sample_id: str, method_name: str, round_index: int, global_seed: int) -> int:
    payload = f"{sample_id}:{method_name}:{round_index}:{global_seed}".encode("utf-8")
    digest = sha256(payload).hexdigest()
    return int(digest[:12], 16)


def _kolmogorov_smirnov_statistic(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    left_sorted = sorted(left)
    right_sorted = sorted(right)
    values = sorted(set(left_sorted + right_sorted))
    left_index = 0
    right_index = 0
    left_total = len(left_sorted)
    right_total = len(right_sorted)
    statistic = 0.0
    for value in values:
        while left_index < left_total and left_sorted[left_index] <= value:
            left_index += 1
        while right_index < right_total and right_sorted[right_index] <= value:
            right_index += 1
        statistic = max(statistic, abs(left_index / left_total - right_index / right_total))
    return round(statistic, 6)


def _mean(values) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return round(sum(materialized) / len(materialized), 6)


def _optional_mean(values) -> float | None:
    materialized = [float(value) for value in values if value is not None]
    if not materialized:
        return None
    return round(sum(materialized) / len(materialized), 6)
