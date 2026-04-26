"""选择性通信实验主运行链路。

本模块围绕“共享前缀”组织整个实验：
先统一运行 Stage A，再为多种 trigger 策略复用同一份中间结果，
只在需要通信时才共享 Stage B，从而在保证对照公平的同时降低总成本。
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
from typing import Callable

from dotenv import load_dotenv

from experiment_core.cache import CachedResponse, RequestCache, json_dump
from experiment_core.datasets import DatasetSample, load_split_ids, select_samples
from experiment_core.evaluation import aggregate_majority, normalize_prediction, score_prediction
from experiment_core.providers import OpenAICompatibleProvider, ProviderRequestError, build_payload, estimate_request_tokens
from experiment_core.rate_limits import SlidingWindowRateLimiter
from experiment_core.runtime import RunProgressTracker, build_run_id
from experiment_core.selective_signals import (
    confidence_display,
    decide_trigger_from_policy,
    normalize_confidence,
    summarize_confidence_rows,
)
from experiment_core.methods import MethodConfig
from experiment_core.structured_output import (
    ARTIFACT_VERSION,
    OUTPUT_MODE_SELECTIVE_COMM,
    validate_structured_output,
)
from selective_comm.config import (
    SelectiveCommExperimentConfig,
    SharedDebateProtocolConfig,
    TriggerPolicyConfig,
    load_benchmarks,
    load_control_catalog,
    load_policies,
    load_protocol_config,
    phase_metadata,
)
from selective_comm.prompting import build_debate_messages, build_initial_messages
from selective_comm.reporting import render_trigger_report
from selective_comm.validation import validate_run


DISPLAY_NAME_MAP = {
    "always_communicate": "always",
    "disagreement_triggered": "disagreement",
    "confidence_triggered": "confidence",
    "hybrid_trigger": "hybrid",
}


@dataclass(frozen=True)
class RunPaths:
    """选择性通信运行目录下的固定产物路径集合。"""

    root: Path
    manifest: Path
    stage_a_turns: Path
    stage_b_turns: Path
    control_turns: Path
    trigger_decisions: Path
    policy_predictions: Path
    policy_metrics: Path
    policy_diagnostics: Path
    oracle_trigger_eval: Path
    progress: Path
    run_validation: Path
    trigger_report: Path


@dataclass(frozen=True)
class SampleResult:
    """单题运行产物。"""

    stage_a_turns: list[dict[str, Any]]
    stage_b_turns: list[dict[str, Any]]
    control_turns: list[dict[str, Any]]
    trigger_rows: list[dict[str, Any]]
    prediction_rows: list[dict[str, Any]]


def run_experiment(
    experiment: SelectiveCommExperimentConfig,
    phase_name: str,
    backbone,
    run_root: str | Path = "local/runs/selective_comm",
    cache_path: str | Path = "cache/selective_comm_requests.sqlite",
) -> Path:
    """执行一个选择性通信 phase，并写出完整运行目录。"""
    load_dotenv(".env.local", override=False)
    benchmarks = load_benchmarks(experiment)
    protocol = load_protocol_config(experiment.protocol)
    policies = load_policies(experiment.policy_configs)
    controls = load_control_catalog(experiment.control_catalog)
    provider = OpenAICompatibleProvider(backbone)
    cache = RequestCache(cache_path)
    limiter = SlidingWindowRateLimiter(
        requests_per_minute=experiment.requests_per_minute_limit,
        tokens_per_minute=experiment.tokens_per_minute_limit,
    )
    run_id = build_run_id(experiment.name, phase_name, backbone.name)
    run_paths = _prepare_run_paths(run_root, experiment.name, phase_name, run_id)
    total_calls, total_predictions = _estimate_work(experiment, phase_name, benchmarks, protocol, controls)
    progress = RunProgressTracker(run_paths.progress, total_calls, total_predictions)

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
        "primary_backbone": experiment.primary_backbone,
        "backbone": asdict(backbone),
        "benchmarks": [asdict(benchmark) for benchmark in benchmarks],
        "total_planned_calls": total_calls,
        "total_planned_predictions": total_predictions,
    }
    run_paths.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    all_stage_a_turns: list[dict[str, Any]] = []
    all_stage_b_turns: list[dict[str, Any]] = []
    all_control_turns: list[dict[str, Any]] = []
    all_trigger_rows: list[dict[str, Any]] = []
    all_prediction_rows: list[dict[str, Any]] = []

    with (
        run_paths.stage_a_turns.open("w", encoding="utf-8") as stage_a_handle,
        run_paths.stage_b_turns.open("w", encoding="utf-8") as stage_b_handle,
        run_paths.control_turns.open("w", encoding="utf-8") as control_handle,
        run_paths.trigger_decisions.open("w", encoding="utf-8") as trigger_handle,
        run_paths.policy_predictions.open("w", encoding="utf-8") as prediction_handle,
    ):
        for benchmark in benchmarks:
            split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
            samples = select_samples(benchmark, split_name)
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
                    stage_a_handle=stage_a_handle,
                    stage_b_handle=stage_b_handle,
                    control_handle=control_handle,
                    trigger_handle=trigger_handle,
                    prediction_handle=prediction_handle,
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
    render_trigger_report(run_paths.root, publish_dir="local/reports/selective_comm")
    run_paths.run_validation.write_text(json.dumps(validate_run(run_paths.root), ensure_ascii=False, indent=2), encoding="utf-8")
    progress.mark_completed()
    cache.close()
    return run_paths.root


def _run_sample_batch(
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    samples: list[DatasetSample],
    protocol: SharedDebateProtocolConfig,
    policies: list[TriggerPolicyConfig],
    controls: dict[str, MethodConfig],
    experiment: SelectiveCommExperimentConfig,
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    on_complete: Callable[[SampleResult], None] | None = None,
) -> None:
    """并发执行同一 benchmark 下的全部样本。"""
    worker = partial(
        _run_sample,
        run_id=run_id,
        benchmark_slug=benchmark_slug,
        split_name=split_name,
        protocol=protocol,
        policies=policies,
        controls=controls,
        experiment=experiment,
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
    )
    max_workers = max(1, min(experiment.max_concurrent_requests, len(samples) or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(worker, sample=sample): sample_index
            for sample_index, sample in enumerate(samples)
        }
        for future in as_completed(future_to_index):
            result = future.result()
            if on_complete is not None:
                on_complete(result)


def _write_sample_result(
    result: SampleResult,
    stage_a_handle,
    stage_b_handle,
    control_handle,
    trigger_handle,
    prediction_handle,
    progress: RunProgressTracker,
    all_stage_a_turns: list[dict[str, Any]],
    all_stage_b_turns: list[dict[str, Any]],
    all_control_turns: list[dict[str, Any]],
    all_trigger_rows: list[dict[str, Any]],
    all_prediction_rows: list[dict[str, Any]],
) -> None:
    """把单题结果立刻写盘并刷新进度。"""
    for row in result.stage_a_turns:
        stage_a_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        progress.record_call(row, method_key="stage_name")
    stage_a_handle.flush()
    for row in result.stage_b_turns:
        stage_b_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        progress.record_call(row, method_key="stage_name")
    stage_b_handle.flush()
    for row in result.control_turns:
        control_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        progress.record_call(row, method_key="method_name")
    control_handle.flush()
    for row in result.trigger_rows:
        trigger_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    trigger_handle.flush()
    for row in result.prediction_rows:
        prediction_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        progress.record_predictions(1, row["dataset"], row["method_name"])
    prediction_handle.flush()
    all_stage_a_turns.extend(result.stage_a_turns)
    all_stage_b_turns.extend(result.stage_b_turns)
    all_control_turns.extend(result.control_turns)
    all_trigger_rows.extend(result.trigger_rows)
    all_prediction_rows.extend(result.prediction_rows)


def _run_sample(
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    sample: DatasetSample,
    protocol: SharedDebateProtocolConfig,
    policies: list[TriggerPolicyConfig],
    controls: dict[str, MethodConfig],
    experiment: SelectiveCommExperimentConfig,
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
) -> SampleResult:
    """运行单题上的共享前缀、策略决策与控制方法。"""
    question_preview = _question_preview(sample.question)
    stage_a_turns: list[dict[str, Any]] = []
    for agent_id in range(1, protocol.agent_count + 1):
        messages = build_initial_messages(sample, agent_id, prompt_version=experiment.prompt_version)
        stage_a_turns.append(
            _execute_turn(
                run_id=run_id,
                dataset=benchmark_slug,
                split_name=split_name,
                sample=sample,
                stage_name="stage_a",
                method_name="shared_stage_a",
                role="initial",
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
            )
        )
    stage_a_trace_hash = _trace_hash(stage_a_turns)
    for row in stage_a_turns:
        row["stage_trace_hash"] = stage_a_trace_hash

    initial_answers = [row["normalized_answer"] for row in stage_a_turns]
    stage_a_vote, stage_a_vote_counts = aggregate_majority(initial_answers)
    stage_a_score = score_prediction(benchmark_slug, stage_a_vote, sample.reference_answer)
    initial_disagreement = len(set(initial_answers)) > 1

    confidence_summary = summarize_confidence_rows(stage_a_turns)
    invalid_confidence_agents = confidence_summary.invalid_agent_ids
    any_invalid_confidence = confidence_summary.any_invalid_confidence
    mean_confidence = confidence_summary.mean_confidence
    confidence_spread = confidence_summary.confidence_spread

    stage_a_prompt_tokens = sum(float(row["prompt_tokens"]) for row in stage_a_turns)
    stage_a_completion_tokens = sum(float(row["completion_tokens"]) for row in stage_a_turns)
    stage_a_total_tokens = sum(float(row["total_tokens"]) for row in stage_a_turns)
    stage_a_latency = sum(float(row["latency_ms"]) for row in stage_a_turns)

    stage_b_turns: list[dict[str, Any]] = []
    previous_round = stage_a_turns
    for round_index in range(1, protocol.debate_rounds + 1):
        current_round: list[dict[str, Any]] = []
        for recipient_id in range(1, protocol.agent_count + 1):
            recipient_previous = previous_round[recipient_id - 1]
            peer_messages = []
            for sender in previous_round:
                if sender["agent_id"] == recipient_id:
                    continue
                peer_messages.append(
                    {
                        "agent": f"agent_{sender['agent_id']}",
                        "answer": str(sender["validated_output"].get("final_answer", "")).strip(),
                        "confidence_raw": sender["confidence_raw_display"],
                        "reasoning": sender["reasoning"],
                    }
                )
            messages = build_debate_messages(
                sample=sample,
                agent_id=recipient_id,
                round_index=round_index,
                previous_reasoning=recipient_previous["reasoning"],
                previous_answer=str(recipient_previous["validated_output"].get("final_answer", "")).strip(),
                previous_confidence_raw=recipient_previous["confidence_raw_display"],
                peer_messages=peer_messages,
                prompt_version=experiment.prompt_version,
            )
            current_round.append(
                _execute_turn(
                    run_id=run_id,
                    dataset=benchmark_slug,
                    split_name=split_name,
                    sample=sample,
                    stage_name="stage_b",
                    method_name="shared_stage_b",
                    role="debate",
                    round_index=round_index,
                    agent_id=recipient_id,
                    visible_peer_count=len(peer_messages),
                    messages=messages,
                    backbone=backbone,
                    provider=provider,
                    cache=cache,
                    limiter=limiter,
                    temperature=protocol.debate_temperature,
                    top_p=protocol.top_p,
                    max_output_tokens=protocol.max_output_tokens,
                    seed=experiment.global_seed + recipient_id + round_index * 100,
                    question_preview=question_preview,
                )
            )
        stage_b_turns.extend(current_round)
        previous_round = current_round

    stage_b_trace_hash = _trace_hash(stage_b_turns)
    for row in stage_b_turns:
        row["stage_trace_hash"] = stage_b_trace_hash

    final_answers = [row["normalized_answer"] for row in previous_round]
    stage_b_vote, stage_b_vote_counts = aggregate_majority(final_answers)
    stage_b_score = score_prediction(benchmark_slug, stage_b_vote, sample.reference_answer)
    stage_b_prompt_tokens = sum(float(row["prompt_tokens"]) for row in stage_b_turns)
    stage_b_completion_tokens = sum(float(row["completion_tokens"]) for row in stage_b_turns)
    stage_b_total_tokens = sum(float(row["total_tokens"]) for row in stage_b_turns)
    stage_b_latency = sum(float(row["latency_ms"]) for row in stage_b_turns)
    beneficial_communication = stage_b_score > stage_a_score

    trigger_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for policy in policies:
        decision = decide_trigger_from_policy(
            policy=policy,
            initial_disagreement=initial_disagreement,
            mean_confidence=mean_confidence,
            confidence_spread=confidence_spread,
            any_invalid_confidence=any_invalid_confidence,
        )
        triggered = decision.triggered
        decision_reason = decision.decision_reason
        fail_open_applied = decision.fail_open_applied
        prediction = stage_b_vote if triggered else stage_a_vote
        score = stage_b_score if triggered else stage_a_score
        prompt_tokens = stage_a_prompt_tokens + (stage_b_prompt_tokens if triggered else 0.0)
        completion_tokens = stage_a_completion_tokens + (stage_b_completion_tokens if triggered else 0.0)
        total_tokens = stage_a_total_tokens + (stage_b_total_tokens if triggered else 0.0)
        latency_ms = stage_a_latency + (stage_b_latency if triggered else 0.0)
        communication_tokens = stage_b_total_tokens if triggered else 0.0
        communication_latency = stage_b_latency if triggered else 0.0
        calls_per_question = protocol.agent_count * (1 + protocol.debate_rounds) if triggered else protocol.agent_count
        trigger_row = {
            "run_id": run_id,
            "dataset": benchmark_slug,
            "split": split_name,
            "sample_id": sample.sample_id,
            "question_preview": question_preview,
            "policy_name": policy.policy_name,
            "display_name": DISPLAY_NAME_MAP.get(policy.policy_name, policy.policy_name),
            "trigger_type": policy.trigger_type,
            "triggered": triggered,
            "early_exit": not triggered,
            "fail_open_applied": fail_open_applied,
            "decision_reason": decision_reason,
            "initial_disagreement": initial_disagreement,
            "mean_confidence": mean_confidence,
            "confidence_spread": confidence_spread,
            "any_invalid_confidence": any_invalid_confidence,
            "invalid_confidence_agents": invalid_confidence_agents,
            "stage_a_trace_hash": stage_a_trace_hash,
            "stage_b_trace_hash": stage_b_trace_hash,
            "stage_b_trace_hash_used": stage_b_trace_hash if triggered else None,
            "mv_3_prediction": stage_a_vote,
            "always_prediction": stage_b_vote,
            "mv_3_score": stage_a_score,
            "always_score": stage_b_score,
            "oracle_positive": beneficial_communication,
        }
        trigger_rows.append(trigger_row)
        prediction_rows.append(
            {
                "run_id": run_id,
                "dataset": benchmark_slug,
                "split": split_name,
                "sample_id": sample.sample_id,
                "question_preview": question_preview,
                "method_name": policy.policy_name,
                "display_name": DISPLAY_NAME_MAP.get(policy.policy_name, policy.policy_name),
                "method_kind": "policy",
                "method_family": policy.trigger_type,
                "model_name": backbone.name,
                "prediction": prediction,
                "gold": sample.reference_answer,
                "score": score,
                "triggered": triggered,
                "early_exit": not triggered,
                "fail_open_applied": fail_open_applied,
                "decision_reason": decision_reason,
                "initial_disagreement": initial_disagreement,
                "mean_confidence": mean_confidence,
                "confidence_spread": confidence_spread,
                "any_invalid_confidence": any_invalid_confidence,
                "invalid_confidence_agents": invalid_confidence_agents,
                "stage_a_trace_hash": stage_a_trace_hash,
                "stage_b_trace_hash": stage_b_trace_hash,
                "stage_b_trace_hash_used": stage_b_trace_hash if triggered else None,
                "stage_a_prediction": stage_a_vote,
                "stage_b_prediction": stage_b_vote,
                "stage_a_score": stage_a_score,
                "stage_b_score": stage_b_score,
                "oracle_positive": beneficial_communication,
                "prompt_tokens_per_question": prompt_tokens,
                "completion_tokens_per_question": completion_tokens,
                "total_tokens_per_question": total_tokens,
                "communication_tokens_per_question": communication_tokens,
                "latency_ms_per_question": latency_ms,
                "communication_latency_ms_per_question": communication_latency,
                "calls_per_question": calls_per_question,
                "stage_a_tokens_per_question": stage_a_total_tokens,
                "stage_b_tokens_per_question": stage_b_total_tokens,
            }
        )

    prediction_rows.append(
        {
            "run_id": run_id,
            "dataset": benchmark_slug,
            "split": split_name,
            "sample_id": sample.sample_id,
            "question_preview": question_preview,
            "method_name": "mv_3",
            "display_name": "mv_3",
            "method_kind": "control",
            "method_family": "shared_majority_vote",
            "model_name": backbone.name,
            "prediction": stage_a_vote,
            "gold": sample.reference_answer,
            "score": stage_a_score,
            "triggered": False,
            "early_exit": True,
            "fail_open_applied": False,
            "decision_reason": "shared_stage_a_vote",
            "initial_disagreement": initial_disagreement,
            "mean_confidence": mean_confidence,
            "confidence_spread": confidence_spread,
            "any_invalid_confidence": any_invalid_confidence,
            "invalid_confidence_agents": invalid_confidence_agents,
            "stage_a_trace_hash": stage_a_trace_hash,
            "stage_b_trace_hash": stage_b_trace_hash,
            "stage_b_trace_hash_used": None,
            "stage_a_prediction": stage_a_vote,
            "stage_b_prediction": stage_b_vote,
            "stage_a_score": stage_a_score,
            "stage_b_score": stage_b_score,
            "oracle_positive": beneficial_communication,
            "prompt_tokens_per_question": stage_a_prompt_tokens,
            "completion_tokens_per_question": stage_a_completion_tokens,
            "total_tokens_per_question": stage_a_total_tokens,
            "communication_tokens_per_question": 0.0,
            "latency_ms_per_question": stage_a_latency,
            "communication_latency_ms_per_question": 0.0,
            "calls_per_question": protocol.agent_count,
            "stage_a_tokens_per_question": stage_a_total_tokens,
            "stage_b_tokens_per_question": stage_b_total_tokens,
        }
    )

    control_turns: list[dict[str, Any]] = []
    for control_name, control in sorted(controls.items()):
        if control.family == "shared_majority_vote":
            continue
        control_result = _run_control_method(
            run_id=run_id,
            benchmark_slug=benchmark_slug,
            split_name=split_name,
            sample=sample,
            control_name=control_name,
            control=control,
            experiment=experiment,
            backbone=backbone,
            provider=provider,
            cache=cache,
            limiter=limiter,
            question_preview=question_preview,
        )
        control_turns.extend(control_result["turn_rows"])
        prediction_rows.append(control_result["prediction_row"])

    return SampleResult(
        stage_a_turns=stage_a_turns,
        stage_b_turns=stage_b_turns,
        control_turns=control_turns,
        trigger_rows=trigger_rows,
        prediction_rows=prediction_rows,
    )


def _run_control_method(
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    sample: DatasetSample,
    control_name: str,
    control: MethodConfig,
    experiment: SelectiveCommExperimentConfig,
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    question_preview: str,
) -> dict[str, Any]:
    """运行一个独立控制方法。"""
    turn_rows: list[dict[str, Any]] = []
    for replicate_id in range(control.budget_calls):
        messages = build_initial_messages(sample, replicate_id + 1, prompt_version=experiment.prompt_version)
        seed = experiment.global_seed + replicate_id
        turn_rows.append(
            _execute_turn(
                run_id=run_id,
                dataset=benchmark_slug,
                split_name=split_name,
                sample=sample,
                stage_name="control",
                method_name=control_name,
                role="control",
                round_index=0,
                agent_id=replicate_id + 1,
                visible_peer_count=0,
                messages=messages,
                backbone=backbone,
                provider=provider,
                cache=cache,
                limiter=limiter,
                temperature=control.temperature,
                top_p=control.top_p,
                max_output_tokens=control.max_output_tokens,
                seed=seed,
                question_preview=question_preview,
            )
        )
    trace_hash = _trace_hash(turn_rows)
    for row in turn_rows:
        row["stage_trace_hash"] = trace_hash
    answers = [row["normalized_answer"] for row in turn_rows]
    prediction, vote_counts = aggregate_majority(answers)
    score = score_prediction(benchmark_slug, prediction, sample.reference_answer)
    prompt_tokens = sum(float(row["prompt_tokens"]) for row in turn_rows)
    completion_tokens = sum(float(row["completion_tokens"]) for row in turn_rows)
    total_tokens = sum(float(row["total_tokens"]) for row in turn_rows)
    latency_ms = sum(float(row["latency_ms"]) for row in turn_rows)
    return {
        "turn_rows": turn_rows,
        "prediction_row": {
            "run_id": run_id,
            "dataset": benchmark_slug,
            "split": split_name,
            "sample_id": sample.sample_id,
            "question_preview": question_preview,
            "method_name": control_name,
            "display_name": control_name,
            "method_kind": "control",
            "method_family": control.family,
            "model_name": backbone.name,
            "prediction": prediction,
            "gold": sample.reference_answer,
            "score": score,
            "triggered": False,
            "early_exit": False,
            "fail_open_applied": False,
            "decision_reason": "independent_control",
            "initial_disagreement": None,
            "mean_confidence": None,
            "confidence_spread": None,
            "any_invalid_confidence": None,
            "invalid_confidence_agents": [],
            "stage_a_trace_hash": None,
            "stage_b_trace_hash": None,
            "stage_b_trace_hash_used": None,
            "stage_a_prediction": None,
            "stage_b_prediction": None,
            "stage_a_score": None,
            "stage_b_score": None,
            "oracle_positive": None,
            "prompt_tokens_per_question": prompt_tokens,
            "completion_tokens_per_question": completion_tokens,
            "total_tokens_per_question": total_tokens,
            "communication_tokens_per_question": 0.0,
            "latency_ms_per_question": latency_ms,
            "communication_latency_ms_per_question": 0.0,
            "calls_per_question": control.budget_calls,
            "stage_a_tokens_per_question": 0.0,
            "stage_b_tokens_per_question": 0.0,
            "vote_counts": vote_counts,
        },
    }


def _execute_turn(
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
) -> dict[str, Any]:
    """执行单次 turn，并解析结构化字段。"""
    payload = build_payload(
        config=backbone,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        seed=seed,
    )
    prompt_hash = _prompt_hash(messages)
    cache_key = _cache_key(
        dataset=dataset,
        split_name=split_name,
        sample_id=sample.sample_id,
        stage_name=stage_name,
        method_name=method_name,
        round_index=round_index,
        agent_id=agent_id,
        prompt_hash=prompt_hash,
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
            validated_output = validate_structured_output(response_payload["assistant_text"], OUTPUT_MODE_SELECTIVE_COMM)
            output_status = "ok"
            final_answer = validated_output["final_answer"]
        except Exception:
            validated_output = {}
            output_status = "schema_fail"
            final_answer = ""

    usage = response_payload.get("usage_reported") or response_payload.get("usage_estimated") or {}
    reasoning = str(validated_output.get("reasoning", "")).strip() if validated_output else ""
    confidence_raw = validated_output.get("confidence_raw") if validated_output else None
    confidence_value, confidence_valid, confidence_source = normalize_confidence(confidence_raw)
    key_evidence = validated_output.get("key_evidence") if validated_output else None
    uncertainty = validated_output.get("uncertain_point") if validated_output else None
    return {
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
        "prompt_hash": prompt_hash,
        "prediction": normalize_prediction(dataset, final_answer),
        "normalized_answer": normalize_prediction(dataset, final_answer),
        "output_status": output_status,
        "prompt_tokens": float(usage.get("prompt_tokens") or 0.0),
        "completion_tokens": float(usage.get("completion_tokens") or 0.0),
        "total_tokens": float(usage.get("total_tokens") or 0.0),
        "latency_ms": float(response_payload.get("latency_ms") or 0.0),
        "cache_hit": cache_hit,
        "request_error": request_error,
        "payload": payload,
        "assistant_text": response_payload.get("assistant_text", ""),
        "provider_reasoning_text": response_payload.get("provider_reasoning_text", ""),
        "validated_output": validated_output,
        "reasoning": reasoning,
        "confidence_raw": confidence_raw,
        "confidence_raw_display": confidence_display(confidence_raw),
        "confidence_value": confidence_value,
        "confidence_valid": confidence_valid,
        "confidence_source": confidence_source,
        "key_evidence": key_evidence,
        "uncertain_point": uncertainty,
    }


def _build_metrics_payload(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """聚合方法级指标。"""
    summary: list[dict[str, Any]] = []
    for dataset in sorted({str(row["dataset"]) for row in prediction_rows} | {"overall"}):
        if dataset == "overall":
            rows_for_dataset = prediction_rows
        else:
            rows_for_dataset = [row for row in prediction_rows if row["dataset"] == dataset]
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in rows_for_dataset:
            grouped.setdefault((row["model_name"], row["method_name"]), []).append(row)
        for (model_name, method_name), rows in sorted(grouped.items()):
            question_count = len(rows)
            total_tokens_mean = _mean(float(row["total_tokens_per_question"]) for row in rows)
            row = {
                "dataset": dataset,
                "model_name": model_name,
                "method_name": method_name,
                "display_name": rows[0]["display_name"],
                "method_kind": rows[0]["method_kind"],
                "method_family": rows[0]["method_family"],
                "question_count": question_count,
                "accuracy_mean": _mean(float(item["score"]) for item in rows),
                "prompt_tokens_mean": _mean(float(item["prompt_tokens_per_question"]) for item in rows),
                "completion_tokens_mean": _mean(float(item["completion_tokens_per_question"]) for item in rows),
                "total_tokens_mean": total_tokens_mean,
                "communication_tokens_mean": _mean(float(item["communication_tokens_per_question"]) for item in rows),
                "latency_ms_mean": _mean(float(item["latency_ms_per_question"]) for item in rows),
                "calls_per_question_mean": _mean(float(item["calls_per_question"]) for item in rows),
                "acc_per_1k_tokens": (
                    round(_mean(float(item["score"]) for item in rows) / total_tokens_mean * 1000, 6)
                    if total_tokens_mean else 0.0
                ),
                "trigger_rate": _mean(1.0 if item.get("triggered") else 0.0 for item in rows) if rows[0]["method_kind"] == "policy" else 0.0,
                "early_exit_rate": _mean(1.0 if item.get("early_exit") else 0.0 for item in rows) if rows[0]["method_kind"] == "policy" else 0.0,
                "invalid_confidence_rate": _mean(1.0 if item.get("any_invalid_confidence") else 0.0 for item in rows if item.get("any_invalid_confidence") is not None),
            }
            summary.append(row)
    return {"summary": summary}


def _build_oracle_payload(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """构建 oracle trigger 评估数据。"""
    sample_groups: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in prediction_rows:
        key = (row["dataset"], row["sample_id"], row["model_name"])
        sample_groups.setdefault(key, {})[row["method_name"]] = row

    sample_rows: list[dict[str, Any]] = []
    policy_names = sorted(
        {
            row["method_name"]
            for row in prediction_rows
            if row["method_kind"] == "policy"
        }
    )
    for (dataset, sample_id, model_name), methods in sorted(sample_groups.items()):
        if "mv_3" not in methods or "always_communicate" not in methods:
            continue
        mv_3_row = methods["mv_3"]
        always_row = methods["always_communicate"]
        beneficial = float(always_row["score"]) > float(mv_3_row["score"])
        payload = {
            "dataset": dataset,
            "sample_id": sample_id,
            "model_name": model_name,
            "question_preview": mv_3_row["question_preview"],
            "mv_3_score": mv_3_row["score"],
            "always_score": always_row["score"],
            "beneficial_communication": beneficial,
            "initial_disagreement": always_row["initial_disagreement"],
            "mean_confidence": always_row["mean_confidence"],
            "confidence_spread": always_row["confidence_spread"],
            "any_invalid_confidence": always_row["any_invalid_confidence"],
            "policies": {},
        }
        for policy_name in policy_names:
            policy_row = methods.get(policy_name)
            if policy_row is None:
                continue
            payload["policies"][policy_name] = {
                "triggered": bool(policy_row["triggered"]),
                "score": float(policy_row["score"]),
                "prediction": policy_row["prediction"],
                "decision_reason": policy_row["decision_reason"],
            }
        sample_rows.append(payload)

    summary_rows: list[dict[str, Any]] = []
    for dataset in sorted({row["dataset"] for row in sample_rows} | {"overall"}):
        rows_for_dataset = sample_rows if dataset == "overall" else [row for row in sample_rows if row["dataset"] == dataset]
        total = len(rows_for_dataset)
        beneficial_count = sum(1 for row in rows_for_dataset if row["beneficial_communication"])
        for policy_name in policy_names:
            triggered_count = sum(1 for row in rows_for_dataset if row["policies"].get(policy_name, {}).get("triggered"))
            true_trigger_count = sum(
                1
                for row in rows_for_dataset
                if row["beneficial_communication"] and row["policies"].get(policy_name, {}).get("triggered")
            )
            false_trigger_count = sum(
                1
                for row in rows_for_dataset
                if (not row["beneficial_communication"]) and row["policies"].get(policy_name, {}).get("triggered")
            )
            missed_positive_count = sum(
                1
                for row in rows_for_dataset
                if row["beneficial_communication"] and (not row["policies"].get(policy_name, {}).get("triggered"))
            )
            summary_rows.append(
                {
                    "dataset": dataset,
                    "policy_name": policy_name,
                    "display_name": DISPLAY_NAME_MAP.get(policy_name, policy_name),
                    "question_count": total,
                    "beneficial_communication_count": beneficial_count,
                    "beneficial_communication_rate": _ratio(beneficial_count, total),
                    "trigger_count": triggered_count,
                    "precision": _ratio(true_trigger_count, triggered_count),
                    "recall": _ratio(true_trigger_count, beneficial_count),
                    "false_trigger_rate": _ratio(false_trigger_count, total),
                    "missed_beneficial_comm_rate": _ratio(missed_positive_count, beneficial_count),
                }
            )

    return {"sample_rows": sample_rows, "summary_rows": summary_rows}


def _build_policy_diagnostics(
    prediction_rows: list[dict[str, Any]],
    oracle_payload: dict[str, Any],
    stage_a_turns: list[dict[str, Any]],
    stage_b_turns: list[dict[str, Any]],
) -> dict[str, Any]:
    """构建策略诊断与共享前缀节省信息。"""
    oracle_rows = oracle_payload["summary_rows"]
    policy_rows: list[dict[str, Any]] = []
    metric_lookup = {
        (row["dataset"], row["method_name"]): row
        for row in _build_metrics_payload(prediction_rows)["summary"]
    }
    for row in oracle_rows:
        metric_row = metric_lookup.get((row["dataset"], row["policy_name"]))
        if metric_row is None:
            continue
        policy_rows.append(
            {
                "dataset": row["dataset"],
                "policy_name": row["policy_name"],
                "display_name": row["display_name"],
                "question_count": row["question_count"],
                "trigger_rate": metric_row["trigger_rate"],
                "early_exit_rate": metric_row["early_exit_rate"],
                "invalid_confidence_rate": metric_row["invalid_confidence_rate"],
                "precision": row["precision"],
                "recall": row["recall"],
                "false_trigger_rate": row["false_trigger_rate"],
                "missed_beneficial_comm_rate": row["missed_beneficial_comm_rate"],
                "accuracy_mean": metric_row["accuracy_mean"],
                "communication_tokens_mean": metric_row["communication_tokens_mean"],
                "total_tokens_mean": metric_row["total_tokens_mean"],
                "acc_per_1k_tokens": metric_row["acc_per_1k_tokens"],
            }
        )

    savings_rows: list[dict[str, Any]] = []
    stage_a_grouped = _group_stage_tokens(stage_a_turns)
    stage_b_grouped = _group_stage_tokens(stage_b_turns)
    trigger_methods = ["always_communicate", "disagreement_triggered", "confidence_triggered", "hybrid_trigger"]
    for dataset in sorted(set(stage_a_grouped) | {"overall"}):
        if dataset == "overall":
            shared_actual_tokens = sum(stage_a_grouped.values()) + sum(stage_b_grouped.values())
            naive_total_tokens = sum(
                float(row["total_tokens_per_question"])
                for row in prediction_rows
                if row["method_name"] in trigger_methods
            )
        else:
            shared_actual_tokens = stage_a_grouped.get(dataset, 0.0) + stage_b_grouped.get(dataset, 0.0)
            naive_total_tokens = sum(
                float(row["total_tokens_per_question"])
                for row in prediction_rows
                if row["dataset"] == dataset and row["method_name"] in trigger_methods
            )
        savings_rows.append(
            {
                "dataset": dataset,
                "shared_actual_tokens": shared_actual_tokens,
                "naive_independent_tokens": naive_total_tokens,
                "shared_prefix_savings_ratio": (
                    round(1 - shared_actual_tokens / naive_total_tokens, 6)
                    if naive_total_tokens
                    else 0.0
                ),
            }
        )

    recommendation = _select_next_default_policy(metric_lookup)
    return {
        "policy_rows": policy_rows,
        "shared_prefix_rows": savings_rows,
        "recommended_next_default_policy": recommendation,
    }


def _select_next_default_policy(metric_lookup: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    """按固定规则选择下一轮默认 trigger。"""
    always_row = metric_lookup.get(("overall", "always_communicate"))
    hybrid_row = metric_lookup.get(("overall", "hybrid_trigger"))
    if always_row is None or hybrid_row is None:
        return {
            "selected_policy": "disagreement_triggered",
            "reason": "missing_overall_rows",
        }
    accuracy_drop = float(always_row["accuracy_mean"]) - float(hybrid_row["accuracy_mean"])
    token_drop_ratio = 0.0
    if float(always_row["total_tokens_mean"]):
        token_drop_ratio = 1 - float(hybrid_row["total_tokens_mean"]) / float(always_row["total_tokens_mean"])
    if accuracy_drop <= 0.05 and token_drop_ratio >= 0.15:
        selected = "hybrid_trigger"
        rule_passed = True
    else:
        selected = "disagreement_triggered"
        rule_passed = False
    return {
        "selected_policy": selected,
        "accuracy_drop_vs_always": round(accuracy_drop, 6),
        "token_drop_ratio_vs_always": round(token_drop_ratio, 6),
        "rule_passed": rule_passed,
    }


def _group_stage_tokens(turn_rows: list[dict[str, Any]]) -> dict[str, float]:
    """按数据集汇总共享阶段 token。"""
    grouped: dict[str, float] = {}
    for row in turn_rows:
        grouped[row["dataset"]] = grouped.get(row["dataset"], 0.0) + float(row["total_tokens"])
    return grouped


def _resolve_split_name(experiment: SelectiveCommExperimentConfig, phase_name: str, benchmark_slug: str) -> str:
    """解析当前 benchmark 对应的冻结 split 名称。"""
    phase = phase_metadata(experiment, phase_name)
    if "split_overrides" in phase:
        return phase["split_overrides"][benchmark_slug]
    return phase["split_suffix"]


def _estimate_work(
    experiment: SelectiveCommExperimentConfig,
    phase_name: str,
    benchmarks,
    protocol: SharedDebateProtocolConfig,
    controls: dict[str, MethodConfig],
) -> tuple[int, int]:
    """估算总调用数与总预测数。"""
    policies = load_policies(experiment.policy_configs)
    total_calls = 0
    total_predictions = 0
    for benchmark in benchmarks:
        split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
        sample_count = len(load_split_ids(benchmark.slug, split_name))
        total_calls += sample_count * protocol.agent_count * (1 + protocol.debate_rounds)
        total_calls += sample_count * sum(method.budget_calls for method in controls.values())
        total_predictions += sample_count * (len(policies) + len(controls))
    return total_calls, total_predictions


def _prepare_run_paths(run_root: str | Path, experiment_name: str, phase_name: str, run_id: str) -> RunPaths:
    """创建运行目录和固定产物路径。"""
    root = Path(run_root) / experiment_name / phase_name / run_id
    root.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        root=root,
        manifest=root / "manifest.json",
        stage_a_turns=root / "stage_a_turns.jsonl",
        stage_b_turns=root / "stage_b_turns.jsonl",
        control_turns=root / "control_turns.jsonl",
        trigger_decisions=root / "trigger_decisions.jsonl",
        policy_predictions=root / "policy_predictions.jsonl",
        policy_metrics=root / "policy_metrics.json",
        policy_diagnostics=root / "policy_diagnostics.json",
        oracle_trigger_eval=root / "oracle_trigger_eval.json",
        progress=root / "progress.json",
        run_validation=root / "run_validation.json",
        trigger_report=root / "trigger_report.md",
    )


def _trace_hash(rows: list[dict[str, Any]]) -> str:
    """对共享阶段 turn 结果做稳定哈希。"""
    payload = [
        {
            "stage_name": row["stage_name"],
            "method_name": row["method_name"],
            "round_index": row["round_index"],
            "agent_id": row["agent_id"],
            "prompt_hash": row["prompt_hash"],
            "normalized_answer": row["normalized_answer"],
            "confidence_value": row["confidence_value"],
            "output_status": row["output_status"],
            "request_error": row["request_error"],
        }
        for row in rows
    ]
    return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _cache_key(
    dataset: str,
    split_name: str,
    sample_id: str,
    stage_name: str,
    method_name: str,
    round_index: int,
    agent_id: int,
    prompt_hash: str,
    payload: dict[str, Any],
) -> str:
    """构造单次调用的缓存键。"""
    fingerprint = {
        "dataset": dataset,
        "split_name": split_name,
        "sample_id": sample_id,
        "stage_name": stage_name,
        "method_name": method_name,
        "round_index": round_index,
        "agent_id": agent_id,
        "prompt_hash": prompt_hash,
        "payload": payload,
    }
    return sha256(json.dumps(fingerprint, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _prompt_hash(messages: list[dict[str, str]]) -> str:
    """对 prompt 内容做稳定哈希。"""
    return sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _question_preview(question: str, max_chars: int = 120) -> str:
    """生成稳定的问题预览。"""
    cleaned = " ".join(question.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3] + "..."


def _mean(values) -> float:
    """安全计算均值。"""
    materialized = list(values)
    if not materialized:
        return 0.0
    return round(sum(materialized) / len(materialized), 6)


def _ratio(numerator: int, denominator: int) -> float:
    """安全计算比例。"""
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)
