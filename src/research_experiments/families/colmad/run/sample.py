"""ColMAD 的样本级执行、协议对照与指标汇总。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import json
from typing import Any

from research_experiments.core.data.datasets import DatasetSample, load_split_ids, select_samples
from research_experiments.core.data.evaluation import normalize_prediction, score_prediction
from research_experiments.core.execution.cache import RequestCache
from research_experiments.core.execution.providers import OpenAICompatibleProvider
from research_experiments.core.execution.rate_limits import SlidingWindowRateLimiter
from research_experiments.core.execution.runner_common import execute_cached_turn, run_indexed_batch
from research_experiments.families.colmad.config import ColmadExperimentConfig, ColmadMethodSpec, ProtocolConfig
from research_experiments.families.colmad.prompts import (
    build_debater_opening_messages,
    build_debater_reply_messages,
    build_judge_messages,
    build_single_agent_messages,
    validate_debater_output,
    validate_detector_output,
    validate_judge_output,
)
from research_experiments.families.shared.common import resolve_phase_split_name, safe_mean, safe_ratio, sum_metric


METHOD_ORDER = (
    "single_agent_detector",
    "copmad_competitive",
    "colmad_collaborative",
)

DEBATE_CALL_BUDGET = 5
SINGLE_CALL_BUDGET = 1


@dataclass(frozen=True)
class SampleResult:
    debate_rows: list[dict[str, Any]]
    judge_rows: list[dict[str, Any]]
    prediction_rows: list[dict[str, Any]]


def _active_methods(experiment: ColmadExperimentConfig) -> list[ColmadMethodSpec]:
    methods = {method.name: method for method in experiment.methods}
    return [methods[name] for name in METHOD_ORDER if name in methods]


def _resolve_split_name(experiment: ColmadExperimentConfig, phase_name: str, benchmark_slug: str) -> str:
    return resolve_phase_split_name(experiment, phase_name, benchmark_slug)


def _load_selected_samples(benchmark, split_name: str) -> list[DatasetSample]:
    return select_samples(benchmark, split_name)


def _estimate_work(
    experiment: ColmadExperimentConfig,
    phase_name: str,
    benchmarks,
    methods: list[ColmadMethodSpec],
) -> tuple[int, int]:
    total_calls = 0
    total_predictions = 0
    for benchmark in benchmarks:
        split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
        sample_count = len(load_split_ids(benchmark.cache_namespace or benchmark.slug, split_name))
        for method in methods:
            total_calls += sample_count * (_call_budget(method))
            total_predictions += sample_count
    return total_calls, total_predictions


def _run_sample_batch(
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    samples: list[DatasetSample],
    protocol: ProtocolConfig,
    experiment: ColmadExperimentConfig,
    backbone,
    provider,
    cache,
    limiter,
) -> list[SampleResult]:
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
    )
    return [result for _, result in run_indexed_batch(samples, worker=worker, max_concurrent_requests=experiment.max_concurrent_requests)]


def _run_sample(
    sample: DatasetSample,
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    protocol: ProtocolConfig,
    experiment: ColmadExperimentConfig,
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
) -> SampleResult:
    debate_rows: list[dict[str, Any]] = []
    judge_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    method_rows: list[dict[str, Any]] = []

    for method in _active_methods(experiment):
        if method.mode == "single_agent_detector":
            row, detector_turn = _run_single_agent_method(
                sample=sample,
                run_id=run_id,
                benchmark_slug=benchmark_slug,
                split_name=split_name,
                method=method,
                protocol=protocol,
                backbone=backbone,
                provider=provider,
                cache=cache,
                limiter=limiter,
                seed=experiment.global_seed + _stable_sample_seed(sample.sample_id),
            )
            debate_rows.append(detector_turn)
            method_rows.append(row)
            continue

        debate_protocol = "competitive" if method.mode == "copmad_competitive" else "collaborative"
        row, debate_chunk, judge_row = _run_debate_method(
            sample=sample,
            run_id=run_id,
            benchmark_slug=benchmark_slug,
            split_name=split_name,
            method=method,
            debate_protocol=debate_protocol,
            protocol=protocol,
            backbone=backbone,
            provider=provider,
            cache=cache,
            limiter=limiter,
            seed=experiment.global_seed + _stable_sample_seed(sample.sample_id),
        )
        debate_rows.extend(debate_chunk)
        judge_rows.append(judge_row)
        method_rows.append(row)

    verdict_map = {row["method_name"]: row["final_verdict"] for row in method_rows}
    for row in method_rows:
        enriched = dict(row)
        enriched["single_agent_verdict"] = verdict_map.get("single_agent_detector", "")
        enriched["copmad_verdict"] = verdict_map.get("copmad_competitive", "")
        enriched["colmad_verdict"] = verdict_map.get("colmad_collaborative", "")
        prediction_rows.append(enriched)
    return SampleResult(
        debate_rows=debate_rows,
        judge_rows=judge_rows,
        prediction_rows=prediction_rows,
    )


def _run_single_agent_method(
    *,
    sample: DatasetSample,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    method: ColmadMethodSpec,
    protocol: ProtocolConfig,
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    turn_row = _execute_structured_turn(
        run_id=run_id,
        dataset=benchmark_slug,
        split_name=split_name,
        sample=sample,
        method_name=method.name,
        debate_protocol="single_agent",
        role="detector",
        turn_stage="single_pass",
        round_index=0,
        messages=build_single_agent_messages(sample),
        validator=validate_detector_output,
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        temperature=protocol.opening_temperature,
        top_p=protocol.top_p,
        max_output_tokens=protocol.max_output_tokens,
        seed=seed,
    )
    payload = _coerce_debater_payload(turn_row)
    prediction_row = _build_prediction_row(
        run_id=run_id,
        dataset=benchmark_slug,
        split_name=split_name,
        sample=sample,
        method=method,
        debate_protocol="single_agent",
        opening_vote_verdict=payload["verdict"],
        opening_alice_verdict=payload["verdict"],
        opening_bob_verdict=payload["verdict"],
        final_verdict=payload["verdict"],
        judge_confidence=float(payload.get("confidence") or 0.5),
        judge_rationale=str(payload.get("rationale") or ""),
        observed_failure_modes=[],
        supportive_critique_observed=False,
        evidence_complementarity_observed=False,
        changed_after_debate=False,
        shift_direction="unchanged",
        judge_flip_after_debate=False,
        turn_rows=[turn_row],
    )
    return prediction_row, turn_row


def _run_debate_method(
    *,
    sample: DatasetSample,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    method: ColmadMethodSpec,
    debate_protocol: str,
    protocol: ProtocolConfig,
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    debate_rows: list[dict[str, Any]] = []
    opening_alice = _execute_structured_turn(
        run_id=run_id,
        dataset=benchmark_slug,
        split_name=split_name,
        sample=sample,
        method_name=method.name,
        debate_protocol=debate_protocol,
        role="debater_alice",
        turn_stage="opening",
        round_index=1,
        messages=build_debater_opening_messages(sample, debate_protocol=debate_protocol, debater_name="Alice"),
        validator=lambda assistant_text, provider_reasoning_text: validate_debater_output(
            assistant_text,
            provider_reasoning_text,
            debate_protocol=debate_protocol,
        ),
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        temperature=protocol.opening_temperature,
        top_p=protocol.top_p,
        max_output_tokens=protocol.max_output_tokens,
        seed=seed + 11,
    )
    debate_rows.append(opening_alice)
    opening_bob = _execute_structured_turn(
        run_id=run_id,
        dataset=benchmark_slug,
        split_name=split_name,
        sample=sample,
        method_name=method.name,
        debate_protocol=debate_protocol,
        role="debater_bob",
        turn_stage="opening",
        round_index=1,
        messages=build_debater_opening_messages(sample, debate_protocol=debate_protocol, debater_name="Bob"),
        validator=lambda assistant_text, provider_reasoning_text: validate_debater_output(
            assistant_text,
            provider_reasoning_text,
            debate_protocol=debate_protocol,
        ),
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        temperature=protocol.opening_temperature,
        top_p=protocol.top_p,
        max_output_tokens=protocol.max_output_tokens,
        seed=seed + 23,
    )
    debate_rows.append(opening_bob)

    alice_payload = _coerce_debater_payload(opening_alice)
    bob_payload = _coerce_debater_payload(opening_bob)

    reply_alice = _execute_structured_turn(
        run_id=run_id,
        dataset=benchmark_slug,
        split_name=split_name,
        sample=sample,
        method_name=method.name,
        debate_protocol=debate_protocol,
        role="debater_alice",
        turn_stage="reply",
        round_index=2,
        messages=build_debater_reply_messages(
            sample,
            debate_protocol=debate_protocol,
            debater_name="Alice",
            own_opening=alice_payload,
            peer_opening=bob_payload,
        ),
        validator=lambda assistant_text, provider_reasoning_text: validate_debater_output(
            assistant_text,
            provider_reasoning_text,
            debate_protocol=debate_protocol,
        ),
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        temperature=protocol.reply_temperature,
        top_p=protocol.top_p,
        max_output_tokens=protocol.max_output_tokens,
        seed=seed + 37,
    )
    debate_rows.append(reply_alice)
    reply_bob = _execute_structured_turn(
        run_id=run_id,
        dataset=benchmark_slug,
        split_name=split_name,
        sample=sample,
        method_name=method.name,
        debate_protocol=debate_protocol,
        role="debater_bob",
        turn_stage="reply",
        round_index=2,
        messages=build_debater_reply_messages(
            sample,
            debate_protocol=debate_protocol,
            debater_name="Bob",
            own_opening=bob_payload,
            peer_opening=alice_payload,
        ),
        validator=lambda assistant_text, provider_reasoning_text: validate_debater_output(
            assistant_text,
            provider_reasoning_text,
            debate_protocol=debate_protocol,
        ),
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        temperature=protocol.reply_temperature,
        top_p=protocol.top_p,
        max_output_tokens=protocol.max_output_tokens,
        seed=seed + 41,
    )
    debate_rows.append(reply_bob)

    judge_row = _execute_judge_turn(
        run_id=run_id,
        dataset=benchmark_slug,
        split_name=split_name,
        sample=sample,
        method_name=method.name,
        debate_protocol=debate_protocol,
        transcript_rows=debate_rows,
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        temperature=protocol.judge_temperature,
        top_p=protocol.top_p,
        max_output_tokens=protocol.max_output_tokens,
        seed=seed + 53,
    )
    judge_payload = _coerce_judge_payload(judge_row, debate_rows)
    opening_vote_verdict, _ = _aggregate_opening_verdict([alice_payload, bob_payload])
    opening_score = score_prediction(benchmark_slug, opening_vote_verdict, sample.reference_answer)
    final_score = score_prediction(benchmark_slug, judge_payload["final_verdict"], sample.reference_answer)
    changed_after_debate = judge_payload["final_verdict"] != opening_vote_verdict
    prediction_row = _build_prediction_row(
        run_id=run_id,
        dataset=benchmark_slug,
        split_name=split_name,
        sample=sample,
        method=method,
        debate_protocol=debate_protocol,
        opening_vote_verdict=opening_vote_verdict,
        opening_alice_verdict=alice_payload["verdict"],
        opening_bob_verdict=bob_payload["verdict"],
        final_verdict=judge_payload["final_verdict"],
        judge_confidence=float(judge_payload.get("confidence") or 0.5),
        judge_rationale=str(judge_payload.get("rationale") or ""),
        observed_failure_modes=list(judge_payload.get("observed_failure_modes") or []),
        supportive_critique_observed=bool(judge_payload.get("supportive_critique_observed")),
        evidence_complementarity_observed=bool(judge_payload.get("evidence_complementarity_observed")),
        changed_after_debate=changed_after_debate,
        shift_direction=_shift_direction(opening_score, final_score),
        judge_flip_after_debate=changed_after_debate,
        turn_rows=debate_rows + [judge_row],
    )
    return prediction_row, debate_rows, judge_row


def _execute_judge_turn(
    *,
    transcript_rows: list[dict[str, Any]],
    **kwargs,
) -> dict[str, Any]:
    return _execute_structured_turn(
        role="judge",
        turn_stage="judgment",
        round_index=3,
        messages=build_judge_messages(
            kwargs["sample"],
            debate_protocol=str(kwargs["debate_protocol"]),
            transcript_rows=transcript_rows,
        ),
        validator=validate_judge_output,
        **kwargs,
    )


def _execute_structured_turn(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    method_name: str,
    debate_protocol: str,
    role: str,
    turn_stage: str,
    round_index: int,
    messages: list[dict[str, str]],
    validator,
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    seed: int,
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
        validator=validator,
    )
    validated_output = result.validated_output or {}
    return {
        "run_id": run_id,
        "dataset": dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "task_name": str(sample.metadata.get("task_name") or ""),
        "candidate_response_model": str(sample.metadata.get("candidate_response_model") or "unknown"),
        "method_name": method_name,
        "method_type": "colmad",
        "debate_protocol": debate_protocol,
        "role": role,
        "turn_stage": turn_stage,
        "round_index": round_index,
        "prompt_hash": result.prompt_hash,
        "output_status": result.output_status,
        "prompt_tokens": float(result.usage.get("prompt_tokens") or 0.0),
        "completion_tokens": float(result.usage.get("completion_tokens") or 0.0),
        "total_tokens": float(result.usage.get("total_tokens") or 0.0),
        "latency_ms": float(result.response_payload.get("latency_ms") or 0.0),
        "cache_hit": result.cache_hit,
        "request_error": result.request_error,
        "assistant_text": result.response_payload.get("assistant_text", ""),
        "provider_reasoning_text": result.response_payload.get("provider_reasoning_text", ""),
        "validated_output": validated_output,
        "verdict": validated_output.get("verdict") or validated_output.get("final_verdict") or "",
        "confidence": float(validated_output.get("confidence") or 0.5),
        "observed_failure_modes": list(validated_output.get("observed_failure_modes") or []),
    }


def _coerce_debater_payload(turn_row: dict[str, Any]) -> dict[str, Any]:
    payload = turn_row.get("validated_output")
    if isinstance(payload, dict) and payload.get("verdict"):
        return dict(payload)
    assistant_text = str(turn_row.get("assistant_text") or "")
    return validate_detector_output(assistant_text, "")


def _coerce_judge_payload(turn_row: dict[str, Any], debate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    payload = turn_row.get("validated_output")
    if isinstance(payload, dict) and payload.get("final_verdict"):
        return dict(payload)
    opening_payloads = [
        _coerce_debater_payload(row)
        for row in debate_rows
        if row.get("turn_stage") == "opening"
    ]
    fallback_verdict, _ = _aggregate_opening_verdict(opening_payloads)
    return {
        "final_verdict": fallback_verdict,
        "confidence": 0.5,
        "rationale": str(turn_row.get("request_error") or "Judge fallback to opening vote."),
        "observed_failure_modes": [],
        "supportive_critique_observed": False,
        "evidence_complementarity_observed": False,
    }


def _aggregate_opening_verdict(payloads: list[dict[str, Any]]) -> tuple[str, float]:
    if not payloads:
        return "contains_no_error", 0.0
    verdict_groups: dict[str, list[float]] = {}
    for payload in payloads:
        verdict = str(payload.get("verdict") or "contains_no_error")
        verdict_groups.setdefault(verdict, []).append(float(payload.get("confidence") or 0.5))
    ranked = sorted(
        verdict_groups.items(),
        key=lambda item: (len(item[1]), safe_mean(item[1]), item[0] == "contains_error"),
        reverse=True,
    )
    verdict = ranked[0][0]
    agreement_ratio = safe_ratio(len(verdict_groups[verdict]), len(payloads))
    return verdict, agreement_ratio


def _shift_direction(previous_score: float, final_score: float) -> str:
    if final_score > previous_score:
        return "wrong_to_correct"
    if final_score < previous_score:
        return "correct_to_wrong"
    return "unchanged"


def _build_prediction_row(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    method: ColmadMethodSpec,
    debate_protocol: str,
    opening_vote_verdict: str,
    opening_alice_verdict: str,
    opening_bob_verdict: str,
    final_verdict: str,
    judge_confidence: float,
    judge_rationale: str,
    observed_failure_modes: list[str],
    supportive_critique_observed: bool,
    evidence_complementarity_observed: bool,
    changed_after_debate: bool,
    shift_direction: str,
    judge_flip_after_debate: bool,
    turn_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    opening_score = score_prediction(dataset, opening_vote_verdict, sample.reference_answer)
    final_score = score_prediction(dataset, final_verdict, sample.reference_answer)
    communication_tokens = sum(
        float(row.get("total_tokens") or 0.0)
        for row in turn_rows
        if row.get("turn_stage") in {"opening", "reply", "judgment"} and debate_protocol != "single_agent"
    )
    return {
        "run_id": run_id,
        "dataset": dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "task_name": str(sample.metadata.get("task_name") or ""),
        "candidate_response_model": str(sample.metadata.get("candidate_response_model") or "unknown"),
        "method_name": method.name,
        "method_type": "colmad",
        "method_mode": method.mode,
        "debate_protocol": debate_protocol,
        "opening_vote_verdict": opening_vote_verdict,
        "opening_alice_verdict": opening_alice_verdict,
        "opening_bob_verdict": opening_bob_verdict,
        "initial_verdict": opening_vote_verdict,
        "final_verdict": final_verdict,
        "prediction": normalize_prediction(dataset, final_verdict),
        "gold": sample.reference_answer,
        "score": final_score,
        "initial_score": opening_score,
        "changed_after_debate": changed_after_debate,
        "shift_direction": shift_direction,
        "judge_flip_after_debate": judge_flip_after_debate,
        "judge_confidence": judge_confidence,
        "judge_rationale": judge_rationale,
        "observed_failure_modes": list(observed_failure_modes),
        "supportive_critique_observed": supportive_critique_observed,
        "evidence_complementarity_observed": evidence_complementarity_observed,
        "competitive_hacking_flag": bool(observed_failure_modes),
        "correct_to_wrong_shift_flag": shift_direction == "correct_to_wrong",
        "wrong_to_correct_shift_flag": shift_direction == "wrong_to_correct",
        "correction_flag": final_score > opening_score,
        "degradation_flag": final_score < opening_score,
        "prompt_tokens_per_question": sum_metric(turn_rows, "prompt_tokens"),
        "completion_tokens_per_question": sum_metric(turn_rows, "completion_tokens"),
        "total_tokens_per_question": sum_metric(turn_rows, "total_tokens"),
        "communication_tokens_per_question": round(communication_tokens, 6),
        "latency_ms_per_question": sum_metric(turn_rows, "latency_ms"),
        "calls_per_question": len(turn_rows),
    }


def _build_metrics(
    prediction_rows: list[dict[str, Any]],
    methods: list[ColmadMethodSpec],
    *,
    model_name: str,
) -> dict[str, Any]:
    summary_rows: list[dict[str, Any]] = []
    datasets = sorted({row["dataset"] for row in prediction_rows})
    method_names = [method.name for method in methods]
    for dataset in [*datasets, "overall"]:
        dataset_rows = prediction_rows if dataset == "overall" else [row for row in prediction_rows if row["dataset"] == dataset]
        for method_name in method_names:
            rows = [row for row in dataset_rows if row["method_name"] == method_name]
            if not rows:
                continue
            summary_rows.append(_summarize_prediction_rows(rows, dataset=dataset, method_name=method_name, model_name=model_name))

    for dataset in [*datasets, "overall"]:
        scoped_rows = [row for row in summary_rows if row["dataset"] == dataset]
        single_row = next((row for row in scoped_rows if row["method_name"] == "single_agent_detector"), None)
        competitive_row = next((row for row in scoped_rows if row["method_name"] == "copmad_competitive"), None)
        for row in scoped_rows:
            row["gain_over_single_agent"] = (
                round(float(row["accuracy_mean"]) - float(single_row["accuracy_mean"]), 6)
                if single_row is not None and row["method_name"] != "single_agent_detector"
                else None
            )
            row["gain_over_competitive"] = (
                round(float(row["accuracy_mean"]) - float(competitive_row["accuracy_mean"]), 6)
                if competitive_row is not None and row["method_name"] != "copmad_competitive"
                else None
            )
            row["token_ratio_over_competitive"] = (
                round(float(row["total_tokens_mean"]) / float(competitive_row["total_tokens_mean"]), 6)
                if competitive_row is not None
                and float(competitive_row["total_tokens_mean"] or 0.0) > 0
                and row["method_name"] != "copmad_competitive"
                else None
            )
    return {
        "model_name": model_name,
        "summary": summary_rows,
    }


def _summarize_prediction_rows(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    method_name: str,
    model_name: str,
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "method_name": method_name,
        "display_name": method_name,
        "model_name": model_name,
        "question_count": len(rows),
        "accuracy_mean": safe_mean(float(row.get("score") or 0.0) for row in rows),
        "prompt_tokens_mean": safe_mean(float(row.get("prompt_tokens_per_question") or 0.0) for row in rows),
        "completion_tokens_mean": safe_mean(float(row.get("completion_tokens_per_question") or 0.0) for row in rows),
        "total_tokens_mean": safe_mean(float(row.get("total_tokens_per_question") or 0.0) for row in rows),
        "communication_tokens_mean": safe_mean(float(row.get("communication_tokens_per_question") or 0.0) for row in rows),
        "calls_per_question_mean": safe_mean(float(row.get("calls_per_question") or 0.0) for row in rows),
        "changed_after_debate_rate": safe_mean(1.0 if row.get("changed_after_debate") else 0.0 for row in rows),
        "judge_flip_after_debate_rate": safe_mean(1.0 if row.get("judge_flip_after_debate") else 0.0 for row in rows),
        "correct_to_wrong_shift_rate": safe_mean(1.0 if row.get("correct_to_wrong_shift_flag") else 0.0 for row in rows),
        "wrong_to_correct_shift_rate": safe_mean(1.0 if row.get("wrong_to_correct_shift_flag") else 0.0 for row in rows),
        "correction_rate": safe_mean(1.0 if row.get("correction_flag") else 0.0 for row in rows),
        "degradation_rate": safe_mean(1.0 if row.get("degradation_flag") else 0.0 for row in rows),
        "competitive_hacking_rate": safe_mean(1.0 if row.get("competitive_hacking_flag") else 0.0 for row in rows),
        "supportive_critique_rate": safe_mean(1.0 if row.get("supportive_critique_observed") else 0.0 for row in rows),
        "evidence_complementarity_rate": safe_mean(1.0 if row.get("evidence_complementarity_observed") else 0.0 for row in rows),
        "judge_confidence_mean": safe_mean(float(row.get("judge_confidence") or 0.0) for row in rows),
    }


def _build_protocol_diagnostics(
    prediction_rows: list[dict[str, Any]],
    judge_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summary_rows: list[dict[str, Any]] = []
    datasets = sorted({row["dataset"] for row in prediction_rows})
    method_names = sorted({row["method_name"] for row in prediction_rows})
    for dataset in [*datasets, "overall"]:
        scoped_predictions = prediction_rows if dataset == "overall" else [row for row in prediction_rows if row["dataset"] == dataset]
        scoped_judges = judge_rows if dataset == "overall" else [row for row in judge_rows if row["dataset"] == dataset]
        for method_name in method_names:
            method_predictions = [row for row in scoped_predictions if row["method_name"] == method_name]
            if not method_predictions:
                continue
            method_judges = [row for row in scoped_judges if row["method_name"] == method_name]
            failure_counts = {mode: 0 for mode in ("fake_evidence", "overconfident_claim", "fallacious_argument")}
            for row in method_judges:
                for mode in row.get("observed_failure_modes") or []:
                    if mode in failure_counts:
                        failure_counts[mode] += 1
            summary_rows.append(
                {
                    "dataset": dataset,
                    "method_name": method_name,
                    "competitive_hacking_rate": safe_mean(1.0 if row.get("competitive_hacking_flag") else 0.0 for row in method_predictions),
                    "supportive_critique_rate": safe_mean(1.0 if row.get("supportive_critique_observed") else 0.0 for row in method_predictions),
                    "correct_to_wrong_shift_rate": safe_mean(1.0 if row.get("correct_to_wrong_shift_flag") else 0.0 for row in method_predictions),
                    "wrong_to_correct_shift_rate": safe_mean(1.0 if row.get("wrong_to_correct_shift_flag") else 0.0 for row in method_predictions),
                    "judge_disagreement_rate": safe_mean(1.0 if row.get("judge_flip_after_debate") else 0.0 for row in method_predictions),
                    "evidence_complementarity_rate": safe_mean(1.0 if row.get("evidence_complementarity_observed") else 0.0 for row in method_predictions),
                    "fake_evidence_rate": safe_ratio(failure_counts["fake_evidence"], len(method_judges)),
                    "overconfident_claim_rate": safe_ratio(failure_counts["overconfident_claim"], len(method_judges)),
                    "fallacious_argument_rate": safe_ratio(failure_counts["fallacious_argument"], len(method_judges)),
                }
            )
    return {"summary_rows": summary_rows}


def _call_budget(method: ColmadMethodSpec) -> int:
    return SINGLE_CALL_BUDGET if method.mode == "single_agent_detector" else DEBATE_CALL_BUDGET


def _stable_sample_seed(sample_id: str) -> int:
    return sum(ord(char) for char in sample_id) % 100000
