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
from research_experiments.families.selective_comm.run.io import RunPaths


DISPLAY_NAME_MAP = {
    "always_communicate": "always",
    "disagreement_triggered": "disagreement",
    "confidence_triggered": "confidence",
    "hybrid_trigger": "hybrid",
    "voc_trigger_v2": "voc_v2",
}



@dataclass(frozen=True)
class SampleResult:
    """单题运行产物。"""

    stage_a_turns: list[dict[str, Any]]
    stage_b_turns: list[dict[str, Any]]
    control_turns: list[dict[str, Any]]
    trigger_rows: list[dict[str, Any]]
    prediction_rows: list[dict[str, Any]]


@dataclass(frozen=True)
class ResumeSeedState:
    """从旧 run 中提取出的可安全复用结果。"""

    source_root: Path
    source_run_id: str
    completed_sample_keys: set[tuple[str, str]]
    stage_a_turns: list[dict[str, Any]]
    stage_b_turns: list[dict[str, Any]]
    control_turns: list[dict[str, Any]]
    trigger_rows: list[dict[str, Any]]
    prediction_rows: list[dict[str, Any]]
    initial_completed_calls: int
    initial_completed_predictions: int




def _write_seed_rows(
    resume_state: ResumeSeedState,
    *,
    stage_a_handle,
    stage_b_handle,
    control_handle,
    trigger_handle,
    prediction_handle,
) -> None:
    """把旧 run 中已完成样本的结果复制到新 run。"""
    for row in resume_state.stage_a_turns:
        stage_a_handle.write_row(row)
    for row in resume_state.stage_b_turns:
        stage_b_handle.write_row(row)
    for row in resume_state.control_turns:
        control_handle.write_row(row)
    for row in resume_state.trigger_rows:
        trigger_handle.write_row(row)
    for row in resume_state.prediction_rows:
        prediction_handle.write_row(row)


def _load_resume_seed_state(
    resume_root: Path,
    protocol: SharedDebateProtocolConfig,
    policies: list[TriggerPolicyConfig],
    controls: dict[str, MethodConfig],
) -> ResumeSeedState:
    """从旧 run 中抽取“完整且无 request_fail”的样本结果，供新 run 复用。"""
    manifest = _load_json_file(resume_root / "manifest.json")
    source_run_id = str(manifest.get("run_id") or resume_root.name)
    stage_a_turns = _load_jsonl_file(resume_root / "stage_a_turns.jsonl")
    stage_b_turns = _load_jsonl_file(resume_root / "stage_b_turns.jsonl")
    allowed_control_names = set(controls)
    allowed_policy_names = {policy.policy_name for policy in policies}
    allowed_prediction_methods = allowed_control_names | allowed_policy_names
    control_turns = [
        row
        for row in _load_jsonl_file(resume_root / "control_turns.jsonl")
        if str(row.get("method_name") or "") in allowed_control_names
    ]
    trigger_rows = [
        row
        for row in _load_jsonl_file(resume_root / "trigger_decisions.jsonl")
        if str(row.get("policy_name") or "") in allowed_policy_names
    ]
    prediction_rows = [
        row
        for row in _load_jsonl_file(resume_root / "policy_predictions.jsonl")
        if str(row.get("method_name") or "") in allowed_prediction_methods
    ]

    completed_sample_keys = _collect_completed_sample_keys(
        stage_a_turns=stage_a_turns,
        stage_b_turns=stage_b_turns,
        control_turns=control_turns,
        trigger_rows=trigger_rows,
        prediction_rows=prediction_rows,
        protocol=protocol,
        policies=policies,
        controls=controls,
    )

    def keep(row: dict[str, Any]) -> bool:
        return (str(row.get("dataset")), str(row.get("sample_id"))) in completed_sample_keys

    seeded_stage_a = [row for row in stage_a_turns if keep(row)]
    seeded_stage_b = [row for row in stage_b_turns if keep(row)]
    seeded_control = [row for row in control_turns if keep(row)]
    seeded_trigger = [row for row in trigger_rows if keep(row)]
    seeded_predictions = [row for row in prediction_rows if keep(row)]

    return ResumeSeedState(
        source_root=resume_root,
        source_run_id=source_run_id,
        completed_sample_keys=completed_sample_keys,
        stage_a_turns=seeded_stage_a,
        stage_b_turns=seeded_stage_b,
        control_turns=seeded_control,
        trigger_rows=seeded_trigger,
        prediction_rows=seeded_predictions,
        initial_completed_calls=len(seeded_stage_a) + len(seeded_stage_b) + len(seeded_control),
        initial_completed_predictions=len(seeded_predictions),
    )


def _collect_completed_sample_keys(
    *,
    stage_a_turns: list[dict[str, Any]],
    stage_b_turns: list[dict[str, Any]],
    control_turns: list[dict[str, Any]],
    trigger_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    protocol: SharedDebateProtocolConfig,
    policies: list[TriggerPolicyConfig],
    controls: dict[str, MethodConfig],
) -> set[tuple[str, str]]:
    """识别旧 run 中已经完整完成且没有 request_fail 的样本。"""
    expected_stage_a = protocol.agent_count
    expected_stage_b = protocol.agent_count * protocol.debate_rounds
    expected_control = sum(method.budget_calls for method in controls.values())
    expected_trigger = len(policies)
    expected_predictions = len(policies) + len(controls)

    stage_a_counts = _count_rows_by_sample(stage_a_turns)
    stage_b_counts = _count_rows_by_sample(stage_b_turns)
    control_counts = _count_rows_by_sample(control_turns)
    trigger_counts = _count_rows_by_sample(trigger_rows)
    prediction_counts = _count_rows_by_sample(prediction_rows)
    failed_keys = _non_ok_sample_keys(stage_a_turns + stage_b_turns + control_turns)

    candidate_keys = (
        set(stage_a_counts)
        | set(stage_b_counts)
        | set(control_counts)
        | set(trigger_counts)
        | set(prediction_counts)
    )

    completed: set[tuple[str, str]] = set()
    for key in candidate_keys:
        if key in failed_keys:
            continue
        if stage_a_counts.get(key, 0) != expected_stage_a:
            continue
        if stage_b_counts.get(key, 0) != expected_stage_b:
            continue
        if control_counts.get(key, 0) != expected_control:
            continue
        if trigger_counts.get(key, 0) != expected_trigger:
            continue
        if prediction_counts.get(key, 0) != expected_predictions:
            continue
        completed.add(key)
    return completed


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
        stage_a_handle.write_row(row)
        progress.record_call(row, method_key="stage_name")
    for row in result.stage_b_turns:
        stage_b_handle.write_row(row)
        progress.record_call(row, method_key="stage_name")
    for row in result.control_turns:
        control_handle.write_row(row)
        progress.record_call(row, method_key="method_name")
    for row in result.trigger_rows:
        trigger_handle.write_row(row)
    for row in result.prediction_rows:
        prediction_handle.write_row(row)
        progress.record_predictions(1, row["dataset"], row["method_name"])
    all_stage_a_turns.extend(result.stage_a_turns)
    all_stage_b_turns.extend(result.stage_b_turns)
    all_control_turns.extend(result.control_turns)
    all_trigger_rows.extend(result.trigger_rows)
    all_prediction_rows.extend(result.prediction_rows)


def _run_sample(
    sample: DatasetSample,
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
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

    confidence_summary = summarize_confidence_rows(stage_a_turns)
    divergence_summary = summarize_divergence_rows(stage_a_turns)
    invalid_confidence_agents = confidence_summary.invalid_agent_ids
    any_invalid_confidence = confidence_summary.any_invalid_confidence
    mean_confidence = confidence_summary.mean_confidence
    confidence_spread = confidence_summary.confidence_spread
    answer_unique_count = divergence_summary.answer_unique_count
    answer_divergence_score = divergence_summary.answer_divergence_score
    claim_similarity_mean = divergence_summary.claim_similarity_mean
    claim_divergence_score = divergence_summary.claim_divergence_score
    uncertainty_type_diversity_score = divergence_summary.uncertainty_type_diversity_score
    initial_disagreement = answer_divergence_score >= 0.5

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
                        "claim_span": sender["claim_span"] or "",
                        "uncertainty_type": sender["uncertainty_type"] or "",
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
                previous_claim_span=recipient_previous["claim_span"],
                previous_uncertainty_type=recipient_previous["uncertainty_type"],
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
            answer_divergence_score=answer_divergence_score,
            claim_divergence_score=claim_divergence_score,
            uncertainty_type_diversity_score=uncertainty_type_diversity_score,
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
            "answer_unique_count": answer_unique_count,
            "answer_divergence_score": answer_divergence_score,
            "claim_similarity_mean": claim_similarity_mean,
            "claim_divergence_score": claim_divergence_score,
            "uncertainty_type_diversity_score": uncertainty_type_diversity_score,
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
                "answer_unique_count": answer_unique_count,
                "answer_divergence_score": answer_divergence_score,
                "claim_similarity_mean": claim_similarity_mean,
                "claim_divergence_score": claim_divergence_score,
                "uncertainty_type_diversity_score": uncertainty_type_diversity_score,
                "mean_confidence": mean_confidence,
                "confidence_spread": confidence_spread,
                "any_invalid_confidence": any_invalid_confidence,
                "invalid_confidence_agents": invalid_confidence_agents,
                "stage_a_hash": stage_a_trace_hash,
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
                "trigger_reason": decision_reason,
                "drift_flag": triggered and stage_b_score < stage_a_score,
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
            "answer_unique_count": answer_unique_count,
            "answer_divergence_score": answer_divergence_score,
            "claim_similarity_mean": claim_similarity_mean,
            "claim_divergence_score": claim_divergence_score,
            "uncertainty_type_diversity_score": uncertainty_type_diversity_score,
            "mean_confidence": mean_confidence,
            "confidence_spread": confidence_spread,
            "any_invalid_confidence": any_invalid_confidence,
            "invalid_confidence_agents": invalid_confidence_agents,
            "stage_a_hash": stage_a_trace_hash,
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
            "trigger_reason": "shared_stage_a_vote",
            "drift_flag": False,
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
            "answer_unique_count": None,
            "answer_divergence_score": None,
            "claim_similarity_mean": None,
            "claim_divergence_score": None,
            "uncertainty_type_diversity_score": None,
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
        schema_id=SCHEMA_ANSWER_WITH_PROXY_SIGNALS_SELECTIVE,
        dataset=dataset,
        use_response_format=False,
    )
    final_answer = str(result.validated_output.get("final_answer") or "")
    prediction = normalize_prediction(dataset, final_answer) if final_answer else ""
    reasoning = str(result.validated_output.get("reasoning", "")).strip() if result.validated_output else ""
    confidence_raw = result.validated_output.get("confidence_raw") if result.validated_output else None
    confidence_value, confidence_valid, confidence_source = normalize_confidence(confidence_raw)
    claim_span = result.validated_output.get("claim_span") if result.validated_output else None
    uncertainty_type = result.validated_output.get("uncertainty_type") if result.validated_output else None
    key_evidence = result.validated_output.get("key_evidence") if result.validated_output else None
    uncertainty = result.validated_output.get("uncertain_point") if result.validated_output else None
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
        "prompt_hash": result.prompt_hash,
        "prediction": prediction,
        "normalized_answer": prediction,
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
        "reasoning": reasoning,
        "confidence_raw": confidence_raw,
        "confidence_raw_display": confidence_display(confidence_raw),
        "confidence_value": confidence_value,
        "confidence_valid": confidence_valid,
        "confidence_source": confidence_source,
        "claim_span": claim_span,
        "uncertainty_type": uncertainty_type,
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
        helpful = float(always_row["score"]) > float(mv_3_row["score"])
        harmful = float(always_row["score"]) < float(mv_3_row["score"])
        oracle_label = "helpful" if helpful else "harmful" if harmful else "neutral"
        payload = {
            "dataset": dataset,
            "sample_id": sample_id,
            "model_name": model_name,
            "question_preview": mv_3_row["question_preview"],
            "mv_3_score": mv_3_row["score"],
            "always_score": always_row["score"],
            "beneficial_communication": helpful,
            "oracle_label": oracle_label,
            "initial_disagreement": always_row["initial_disagreement"],
            "answer_unique_count": always_row.get("answer_unique_count"),
            "answer_divergence_score": always_row.get("answer_divergence_score"),
            "claim_similarity_mean": always_row.get("claim_similarity_mean"),
            "claim_divergence_score": always_row.get("claim_divergence_score"),
            "uncertainty_type_diversity_score": always_row.get("uncertainty_type_diversity_score"),
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
        helpful_count = sum(1 for row in rows_for_dataset if row["oracle_label"] == "helpful")
        harmful_count = sum(1 for row in rows_for_dataset if row["oracle_label"] == "harmful")
        neutral_count = sum(1 for row in rows_for_dataset if row["oracle_label"] == "neutral")
        for policy_name in policy_names:
            triggered_count = sum(1 for row in rows_for_dataset if row["policies"].get(policy_name, {}).get("triggered"))
            helpful_trigger_count = sum(
                1
                for row in rows_for_dataset
                if row["oracle_label"] == "helpful" and row["policies"].get(policy_name, {}).get("triggered")
            )
            harmful_trigger_count = sum(
                1
                for row in rows_for_dataset
                if row["oracle_label"] == "harmful" and row["policies"].get(policy_name, {}).get("triggered")
            )
            neutral_trigger_count = sum(
                1
                for row in rows_for_dataset
                if row["oracle_label"] == "neutral" and row["policies"].get(policy_name, {}).get("triggered")
            )
            missed_positive_count = sum(
                1
                for row in rows_for_dataset
                if row["oracle_label"] == "helpful" and (not row["policies"].get(policy_name, {}).get("triggered"))
            )
            summary_rows.append(
                {
                    "dataset": dataset,
                    "policy_name": policy_name,
                    "display_name": DISPLAY_NAME_MAP.get(policy_name, policy_name),
                    "question_count": total,
                    "beneficial_communication_count": helpful_count,
                    "beneficial_communication_rate": _ratio(helpful_count, total),
                    "helpful_count": helpful_count,
                    "harmful_count": harmful_count,
                    "neutral_count": neutral_count,
                    "trigger_count": triggered_count,
                    "helpful_trigger_count": helpful_trigger_count,
                    "harmful_trigger_count": harmful_trigger_count,
                    "neutral_trigger_count": neutral_trigger_count,
                    "precision": _ratio(helpful_trigger_count, triggered_count),
                    "recall": _ratio(helpful_trigger_count, helpful_count),
                    "false_trigger_rate": _ratio(harmful_trigger_count + neutral_trigger_count, total),
                    "missed_beneficial_comm_rate": _ratio(missed_positive_count, helpful_count),
                    "helpful_recall": _ratio(helpful_trigger_count, helpful_count),
                    "harmful_trigger_rate": _ratio(harmful_trigger_count, harmful_count),
                    "neutral_waste_rate": _ratio(neutral_trigger_count, triggered_count),
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
    voc_policy_rows: list[dict[str, Any]] = []
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
        voc_policy_rows.append(
            {
                "dataset": row["dataset"],
                "policy_name": row["policy_name"],
                "display_name": row["display_name"],
                "question_count": row["question_count"],
                "helpful_recall": row["helpful_recall"],
                "harmful_trigger_rate": row["harmful_trigger_rate"],
                "neutral_waste_rate": row["neutral_waste_rate"],
                "trigger_rate": metric_row["trigger_rate"],
                "communication_tokens_mean": metric_row["communication_tokens_mean"],
                "total_tokens_mean": metric_row["total_tokens_mean"],
            }
        )

    savings_rows: list[dict[str, Any]] = []
    stage_a_grouped = _group_stage_tokens(stage_a_turns)
    stage_b_grouped = _group_stage_tokens(stage_b_turns)
    trigger_methods = sorted(
        {
            str(row["method_name"])
            for row in prediction_rows
            if row.get("method_kind") == "policy"
        }
    )
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
        "voc_policy_rows": voc_policy_rows,
        "shared_prefix_rows": savings_rows,
        "recommended_next_default_policy": recommendation,
    }


def _select_next_default_policy(metric_lookup: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    """按固定规则选择下一轮默认 trigger。"""
    always_row = metric_lookup.get(("overall", "always_communicate"))
    preferred_row = metric_lookup.get(("overall", "voc_trigger_v2"))
    preferred_name = "voc_trigger_v2"
    if preferred_row is None:
        preferred_row = metric_lookup.get(("overall", "hybrid_trigger"))
        preferred_name = "hybrid_trigger"
    if always_row is None or preferred_row is None:
        return {
            "selected_policy": "disagreement_triggered",
            "reason": "missing_overall_rows",
        }
    accuracy_drop = float(always_row["accuracy_mean"]) - float(preferred_row["accuracy_mean"])
    token_drop_ratio = 0.0
    if float(always_row["total_tokens_mean"]):
        token_drop_ratio = 1 - float(preferred_row["total_tokens_mean"]) / float(always_row["total_tokens_mean"])
    if accuracy_drop <= 0.05 and token_drop_ratio >= 0.15:
        selected = preferred_name
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
    return resolve_phase_split_name(experiment, phase_name, benchmark_slug)


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
        sample_count = len(load_split_ids(benchmark.cache_namespace or benchmark.slug, split_name))
        total_calls += sample_count * protocol.agent_count * (1 + protocol.debate_rounds)
        total_calls += sample_count * sum(method.budget_calls for method in controls.values())
        total_predictions += sample_count * (len(policies) + len(controls))
    return total_calls, total_predictions




def _trace_hash(rows: list[dict[str, Any]]) -> str:
    """对共享阶段 turn 结果做稳定哈希。"""
    return stable_trace_hash(
        rows,
        [
            "stage_name",
            "method_name",
            "round_index",
            "agent_id",
            "prompt_hash",
            "normalized_answer",
            "confidence_value",
            "output_status",
            "request_error",
        ],
    )


def _question_preview(question: str, max_chars: int = 120) -> str:
    """生成稳定的问题预览。"""
    return build_question_preview(question, max_chars=max_chars)


def _mean(values) -> float:
    """安全计算均值。"""
    return safe_mean(values)


def _ratio(numerator: int, denominator: int) -> float:
    """安全计算比例。"""
    return safe_ratio(numerator, denominator)


def _count_rows_by_sample(rows: list[dict[str, Any]]) -> dict[tuple[str, str], int]:
    """按 `(dataset, sample_id)` 统计行数。"""
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (str(row.get("dataset")), str(row.get("sample_id")))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _non_ok_sample_keys(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    """收集包含非 `ok` 输出的样本键。"""
    return {
        (str(row.get("dataset")), str(row.get("sample_id")))
        for row in rows
        if row.get("output_status") not in {None, "ok"}
    }


def _load_jsonl_file(path: Path) -> list[dict[str, Any]]:
    """读取 UTF-8 JSONL 文件。"""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _load_json_file(path: Path) -> dict[str, Any]:
    """读取 UTF-8 JSON 文件。"""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


