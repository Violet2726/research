"""Table-Critic 的样本级执行、模板树维护与指标聚合。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import partial
import json
from pathlib import Path
from typing import Any

from research_experiments.core.data.datasets import DatasetSample, load_split_ids, select_samples
from research_experiments.core.data.evaluation import normalize_prediction, score_prediction
from research_experiments.core.execution.cache import RequestCache
from research_experiments.core.execution.providers import OpenAICompatibleProvider
from research_experiments.core.execution.rate_limits import SlidingWindowRateLimiter
from research_experiments.core.execution.runner_common import execute_cached_turn, run_indexed_batch
from research_experiments.families.shared.common import resolve_phase_split_name, safe_mean, safe_ratio
from research_experiments.families.table_critic.config import (
    ProtocolConfig,
    TableCriticExperimentConfig,
    TableCriticMethodSpec,
)
from research_experiments.families.table_critic.prompts import (
    TemplateHint,
    build_chain_of_table_messages,
    build_critic_messages,
    build_curator_messages,
    build_direct_messages,
    build_few_shot_messages,
    build_judge_messages,
    build_refiner_messages,
    validate_critic_output,
    validate_curator_output,
    validate_judge_output,
    validate_refiner_output_with_fallback,
    validate_reasoning_answer_output,
)


SINGLE_PASS_MODES = {"direct_qa", "few_shot_qa", "chain_of_table"}
REFINEMENT_MODES = {"critic_cot", "table_critic_paper"}


@dataclass(frozen=True)
class TemplateSummary:
    """模板树中的单条可复用批评摘要。"""

    template_id: str
    path: str
    pattern_summary: str
    reuse_hint: str
    template_title: str
    usage_count: int
    success_count: int


def _active_methods(experiment: TableCriticExperimentConfig) -> list[TableCriticMethodSpec]:
    return list(experiment.methods)


def _resolve_split_name(experiment: TableCriticExperimentConfig, phase_name: str, benchmark_slug: str) -> str:
    return resolve_phase_split_name(experiment, phase_name, benchmark_slug)


def _load_selected_samples(benchmark, split_name: str) -> list[DatasetSample]:
    return select_samples(benchmark, split_name)


def _estimate_work(
    experiment: TableCriticExperimentConfig,
    phase_name: str,
    benchmarks,
    protocol: ProtocolConfig,
    methods: list[TableCriticMethodSpec],
) -> tuple[int, int]:
    total_samples = 0
    for benchmark in benchmarks:
        split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
        total_samples += len(load_split_ids(benchmark.cache_namespace or benchmark.slug, split_name))

    total_calls = 0
    for method in methods:
        per_sample_calls = _planned_calls_per_sample(method, protocol)
        total_calls += total_samples * per_sample_calls
    total_predictions = total_samples * len(methods)
    return total_calls, total_predictions


def _planned_calls_per_sample(method: TableCriticMethodSpec, protocol: ProtocolConfig) -> int:
    if method.mode in SINGLE_PASS_MODES:
        return 1
    if method.mode == "critic_cot":
        return 4
    if method.mode == "table_critic_paper":
        return 1 + protocol.max_refine_rounds * (4 if protocol.use_curator else 3) + 1
    raise ValueError(f"Unsupported Table-Critic mode: {method.mode}")


def _run_simple_method_batch(
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    samples: list[DatasetSample],
    method: TableCriticMethodSpec,
    protocol: ProtocolConfig,
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    global_seed: int,
    max_concurrent_requests: int,
) -> list[tuple[int, dict[str, Any]]]:
    worker = partial(
        _run_simple_method_sample,
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
    )
    return run_indexed_batch(samples, worker=worker, max_concurrent_requests=max_concurrent_requests)


def _run_simple_method_sample(
    sample: DatasetSample,
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    method: TableCriticMethodSpec,
    protocol: ProtocolConfig,
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    global_seed: int,
) -> dict[str, Any]:
    turn_rows: list[dict[str, Any]] = []
    critic_rows: list[dict[str, Any]] = []
    refinement_rows: list[dict[str, Any]] = []

    initial_turn = _execute_reasoning_turn(
        run_id=run_id,
        dataset=benchmark_slug,
        split_name=split_name,
        sample=sample,
        method_name=method.name,
        role="solver",
        round_index=0,
        messages=_initial_messages_for_mode(sample, method.mode),
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        temperature=protocol.initial_temperature,
        top_p=protocol.top_p,
        max_output_tokens=protocol.max_output_tokens,
        seed=global_seed,
    )
    turn_rows.append(initial_turn)
    initial_reasoning = str(initial_turn["validated_output"].get("reasoning") or "")
    initial_answer = str(initial_turn["validated_output"].get("final_answer") or "")
    current_reasoning = initial_reasoning
    current_answer = initial_answer
    committed_reasoning = initial_reasoning
    committed_answer = initial_answer
    current_score = score_prediction(benchmark_slug, current_answer, sample.reference_answer) if current_answer else 0.0
    final_judge_payload = {"passed": True, "error_detected": False, "error_step": "None", "node_path": ["ROOT"], "rationale": "single pass"}
    observed_judge_payload = dict(final_judge_payload)
    critic_feedback = ""
    stopped_reason = "single_pass"
    refinement_round_count = 0

    if method.mode == "critic_cot":
        judge_turn = _execute_judge_turn(
            run_id=run_id,
            dataset=benchmark_slug,
            split_name=split_name,
            sample=sample,
            method_name=method.name,
            round_index=0,
            current_reasoning=current_reasoning,
            current_answer=current_answer,
            backbone=backbone,
            provider=provider,
            cache=cache,
            limiter=limiter,
            temperature=protocol.judge_temperature,
            top_p=protocol.top_p,
            max_output_tokens=protocol.max_output_tokens,
            seed=global_seed + 1,
        )
        turn_rows.append(judge_turn)
        final_judge_payload = _coerce_judge_payload(judge_turn)
        observed_judge_payload = dict(final_judge_payload)
        if final_judge_payload["passed"]:
            stopped_reason = "judge_passed_initial"
            committed_reasoning = current_reasoning
            committed_answer = current_answer
        else:
            critic_turn = _execute_critic_turn(
                run_id=run_id,
                dataset=benchmark_slug,
                split_name=split_name,
                sample=sample,
                method_name=method.name,
                round_index=1,
                current_reasoning=current_reasoning,
                current_answer=current_answer,
                judge_payload=final_judge_payload,
                template_hints=[],
                backbone=backbone,
                provider=provider,
                cache=cache,
                limiter=limiter,
                temperature=protocol.critic_temperature,
                top_p=protocol.top_p,
                max_output_tokens=protocol.max_output_tokens,
                seed=global_seed + 2,
            )
            turn_rows.append(critic_turn)
            critic_payload = _coerce_critic_payload(critic_turn)
            critic_feedback = str(critic_payload["critic_feedback"])
            refiner_turn = _execute_refiner_turn(
                run_id=run_id,
                dataset=benchmark_slug,
                split_name=split_name,
                sample=sample,
                method_name=method.name,
                round_index=1,
                current_reasoning=current_reasoning,
                current_answer=current_answer,
                judge_payload=final_judge_payload,
                critic_payload=critic_payload,
                template_hints=[],
                backbone=backbone,
                provider=provider,
                cache=cache,
                limiter=limiter,
                temperature=protocol.refiner_temperature,
                top_p=protocol.top_p,
                max_output_tokens=protocol.max_output_tokens,
                seed=global_seed + 3,
            )
            turn_rows.append(refiner_turn)
            refined_payload = _coerce_reasoning_payload(
                refiner_turn,
                previous_reasoning=current_reasoning,
                previous_answer=current_answer,
            )
            current_reasoning = str(refined_payload["reasoning"] or "")
            current_answer = _stabilize_answer_format_answer(
                previous_answer=current_answer,
                refined_answer=str(refined_payload["final_answer"] or ""),
                judge_payload=final_judge_payload,
            )
            refinement_round_count = 1
            stopped_reason = "single_refine_round"
            final_score = score_prediction(benchmark_slug, current_answer, sample.reference_answer) if current_answer else 0.0
            refinement_rows.append(
                _build_refinement_trace_row(
                    run_id=run_id,
                    dataset=benchmark_slug,
                    split_name=split_name,
                    sample=sample,
                    method=method,
                    round_index=1,
                    initial_answer=initial_answer,
                    previous_answer=initial_answer,
                    refined_answer=current_answer,
                    judge_payload=final_judge_payload,
                    critic_payload=critic_payload,
                    template_ids_used=[],
                    previous_score=current_score,
                    refined_score=final_score,
                    stopped_reason=stopped_reason,
                )
            )
            post_judge_turn = _execute_judge_turn(
                run_id=run_id,
                dataset=benchmark_slug,
                split_name=split_name,
                sample=sample,
                method_name=method.name,
                round_index=1,
                current_reasoning=current_reasoning,
                current_answer=current_answer,
                backbone=backbone,
                provider=provider,
                cache=cache,
                limiter=limiter,
                temperature=protocol.judge_temperature,
                top_p=protocol.top_p,
                max_output_tokens=protocol.max_output_tokens,
                seed=global_seed + 4,
            )
            turn_rows.append(post_judge_turn)
            final_judge_payload = _coerce_judge_payload(post_judge_turn)
            if final_judge_payload["passed"]:
                committed_reasoning = current_reasoning
                committed_answer = current_answer
            else:
                current_reasoning = committed_reasoning
                current_answer = committed_answer

    final_prediction = _build_prediction_row(
        run_id=run_id,
        dataset=benchmark_slug,
        split_name=split_name,
        sample=sample,
        method=method,
        turn_rows=turn_rows,
        initial_reasoning=initial_reasoning,
        initial_answer=initial_answer,
        final_reasoning=current_reasoning,
        final_answer=current_answer,
        final_judge_payload=observed_judge_payload,
        critic_feedback=critic_feedback,
        refinement_round_count=refinement_round_count,
        stopped_reason=stopped_reason,
        template_ids_used=[],
    )
    critic_rows.append(
        _build_critic_trace_row(
            run_id=run_id,
            dataset=benchmark_slug,
            split_name=split_name,
            sample=sample,
            method=method,
            final_judge_payload=observed_judge_payload,
            critic_feedback=critic_feedback,
            template_ids_used=[],
            refinement_round_count=refinement_round_count,
        )
    )
    if not refinement_rows:
        refinement_rows.append(
            _build_refinement_trace_row(
                run_id=run_id,
                dataset=benchmark_slug,
                split_name=split_name,
                sample=sample,
                method=method,
                round_index=0,
                initial_answer=initial_answer,
                previous_answer=initial_answer,
                refined_answer=current_answer,
                judge_payload=final_judge_payload,
                critic_payload={},
                template_ids_used=[],
                previous_score=score_prediction(benchmark_slug, initial_answer, sample.reference_answer) if initial_answer else 0.0,
                refined_score=float(final_prediction["score"]),
                stopped_reason=stopped_reason,
            )
        )
    return {
        "turn_rows": turn_rows,
        "critic_rows": critic_rows,
        "refinement_rows": refinement_rows,
        "prediction_row": final_prediction,
    }


def _run_table_critic_sequence(
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    samples: list[DatasetSample],
    method: TableCriticMethodSpec,
    protocol: ProtocolConfig,
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    global_seed: int,
    template_tree: dict[str, Any],
) -> list[tuple[int, dict[str, Any]]]:
    results: list[tuple[int, dict[str, Any]]] = []
    for sample_index, sample in enumerate(samples):
        payload = _run_table_critic_sample(
            sample,
            run_id=run_id,
            benchmark_slug=benchmark_slug,
            split_name=split_name,
            method=method,
            protocol=protocol,
            backbone=backbone,
            provider=provider,
            cache=cache,
            limiter=limiter,
            global_seed=global_seed + sample_index * 100,
            template_tree=template_tree,
        )
        results.append((sample_index, payload))
    return results


def _run_table_critic_sample(
    sample: DatasetSample,
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    method: TableCriticMethodSpec,
    protocol: ProtocolConfig,
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    global_seed: int,
    template_tree: dict[str, Any],
) -> dict[str, Any]:
    turn_rows: list[dict[str, Any]] = []
    critic_rows: list[dict[str, Any]] = []
    refinement_rows: list[dict[str, Any]] = []

    initial_turn = _execute_reasoning_turn(
        run_id=run_id,
        dataset=benchmark_slug,
        split_name=split_name,
        sample=sample,
        method_name=method.name,
        role="solver",
        round_index=0,
        messages=build_chain_of_table_messages(sample),
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        temperature=protocol.initial_temperature,
        top_p=protocol.top_p,
        max_output_tokens=protocol.max_output_tokens,
        seed=global_seed,
    )
    turn_rows.append(initial_turn)
    initial_reasoning = str(initial_turn["validated_output"].get("reasoning") or "")
    initial_answer = str(initial_turn["validated_output"].get("final_answer") or "")
    current_reasoning = initial_reasoning
    current_answer = initial_answer
    committed_reasoning = initial_reasoning
    committed_answer = initial_answer
    current_score = score_prediction(benchmark_slug, current_answer, sample.reference_answer) if current_answer else 0.0
    final_judge_payload = {"passed": False, "error_detected": True, "error_step": "Logical Error", "node_path": ["ROOT", "Final Query Error", "Logical Error"], "rationale": ""}
    observed_judge_payload: dict[str, Any] | None = None
    critic_feedback = ""
    template_ids_used: list[str] = []
    stopped_reason = "max_refine_rounds"
    refinement_round_count = 0

    for round_index in range(0, protocol.max_refine_rounds + 1):
        judge_turn = _execute_judge_turn(
            run_id=run_id,
            dataset=benchmark_slug,
            split_name=split_name,
            sample=sample,
            method_name=method.name,
            round_index=round_index,
            current_reasoning=current_reasoning,
            current_answer=current_answer,
            backbone=backbone,
            provider=provider,
            cache=cache,
            limiter=limiter,
            temperature=protocol.judge_temperature,
            top_p=protocol.top_p,
            max_output_tokens=protocol.max_output_tokens,
            seed=global_seed + round_index * 10 + 1,
        )
        turn_rows.append(judge_turn)
        final_judge_payload = _coerce_judge_payload(judge_turn)
        if final_judge_payload.get("error_detected"):
            observed_judge_payload = dict(final_judge_payload)
        if final_judge_payload["passed"]:
            stopped_reason = "judge_passed"
            committed_reasoning = current_reasoning
            committed_answer = current_answer
            break
        if round_index >= protocol.max_refine_rounds:
            stopped_reason = "max_refine_rounds"
            current_reasoning = committed_reasoning
            current_answer = committed_answer
            break

        template_hints = _lookup_template_hints(
            template_tree,
            final_judge_payload.get("node_path") or ["ROOT"],
            max_examples=protocol.max_template_examples,
        )
        template_ids_used.extend(hint.template_id for hint in template_hints)
        critic_turn = _execute_critic_turn(
            run_id=run_id,
            dataset=benchmark_slug,
            split_name=split_name,
            sample=sample,
            method_name=method.name,
            round_index=round_index + 1,
            current_reasoning=current_reasoning,
            current_answer=current_answer,
            judge_payload=final_judge_payload,
            template_hints=template_hints,
            backbone=backbone,
            provider=provider,
            cache=cache,
            limiter=limiter,
            temperature=protocol.critic_temperature,
            top_p=protocol.top_p,
            max_output_tokens=protocol.max_output_tokens,
            seed=global_seed + round_index * 10 + 2,
        )
        turn_rows.append(critic_turn)
        critic_payload = _coerce_critic_payload(critic_turn)
        critic_feedback = str(critic_payload["critic_feedback"])

        refiner_turn = _execute_refiner_turn(
            run_id=run_id,
            dataset=benchmark_slug,
            split_name=split_name,
            sample=sample,
            method_name=method.name,
            round_index=round_index + 1,
            current_reasoning=current_reasoning,
            current_answer=current_answer,
            judge_payload=final_judge_payload,
            critic_payload=critic_payload,
            template_hints=template_hints,
            backbone=backbone,
            provider=provider,
            cache=cache,
            limiter=limiter,
            temperature=protocol.refiner_temperature,
            top_p=protocol.top_p,
            max_output_tokens=protocol.max_output_tokens,
            seed=global_seed + round_index * 10 + 3,
        )
        turn_rows.append(refiner_turn)
        refined_payload = _coerce_reasoning_payload(
            refiner_turn,
            previous_reasoning=current_reasoning,
            previous_answer=current_answer,
        )
        refined_reasoning = str(refined_payload["reasoning"] or "")
        refined_answer = _stabilize_answer_format_answer(
            previous_answer=current_answer,
            refined_answer=str(refined_payload["final_answer"] or ""),
            judge_payload=final_judge_payload,
        )
        refined_score = score_prediction(benchmark_slug, refined_answer, sample.reference_answer) if refined_answer else 0.0
        improved = refined_score > current_score

        curator_payload: dict[str, Any] = {}
        if protocol.use_curator:
            curator_turn = _execute_curator_turn(
                run_id=run_id,
                dataset=benchmark_slug,
                split_name=split_name,
                sample=sample,
                method_name=method.name,
                round_index=round_index + 1,
                judge_payload=final_judge_payload,
                critic_payload=critic_payload,
                improved=improved,
                backbone=backbone,
                provider=provider,
                cache=cache,
                limiter=limiter,
                temperature=protocol.curator_temperature,
                top_p=protocol.top_p,
                max_output_tokens=protocol.max_output_tokens,
                seed=global_seed + round_index * 10 + 4,
            )
            turn_rows.append(curator_turn)
            curator_payload = _coerce_curator_payload(curator_turn)
            _update_template_tree(
                template_tree,
                final_judge_payload.get("node_path") or ["ROOT"],
                curator_payload,
                improved=improved,
                max_summaries_per_node=protocol.template_tree_max_summaries_per_node,
            )

        refinement_round_count = round_index + 1
        refinement_rows.append(
            _build_refinement_trace_row(
                run_id=run_id,
                dataset=benchmark_slug,
                split_name=split_name,
                sample=sample,
                method=method,
                round_index=round_index + 1,
                initial_answer=initial_answer,
                previous_answer=current_answer,
                refined_answer=refined_answer,
                judge_payload=final_judge_payload,
                critic_payload=critic_payload,
                template_ids_used=[hint.template_id for hint in template_hints],
                previous_score=current_score,
                refined_score=refined_score,
                stopped_reason="continue",
            )
        )

        if _normalize_trace_text(current_reasoning) == _normalize_trace_text(refined_reasoning) and _normalize_trace_text(current_answer) == _normalize_trace_text(refined_answer):
            current_reasoning = refined_reasoning
            current_answer = refined_answer
            current_score = refined_score
            stopped_reason = "stable_refinement"
            break

        current_reasoning = refined_reasoning
        current_answer = refined_answer
        current_score = refined_score

    observed_judge_payload = observed_judge_payload or dict(final_judge_payload)
    critic_rows.append(
        _build_critic_trace_row(
            run_id=run_id,
            dataset=benchmark_slug,
            split_name=split_name,
            sample=sample,
            method=method,
            final_judge_payload=observed_judge_payload,
            critic_feedback=critic_feedback,
            template_ids_used=_deduplicate_preserve_order(template_ids_used),
            refinement_round_count=refinement_round_count,
        )
    )
    prediction_row = _build_prediction_row(
        run_id=run_id,
        dataset=benchmark_slug,
        split_name=split_name,
        sample=sample,
        method=method,
        turn_rows=turn_rows,
        initial_reasoning=initial_reasoning,
        initial_answer=initial_answer,
        final_reasoning=current_reasoning,
        final_answer=current_answer,
        final_judge_payload=observed_judge_payload,
        critic_feedback=critic_feedback,
        refinement_round_count=refinement_round_count,
        stopped_reason=stopped_reason,
        template_ids_used=_deduplicate_preserve_order(template_ids_used),
    )
    if refinement_rows:
        refinement_rows[-1]["stopped_reason"] = stopped_reason
    else:
        refinement_rows.append(
            _build_refinement_trace_row(
                run_id=run_id,
                dataset=benchmark_slug,
                split_name=split_name,
                sample=sample,
                method=method,
                round_index=0,
                initial_answer=initial_answer,
                previous_answer=initial_answer,
                refined_answer=current_answer,
                judge_payload=final_judge_payload,
                critic_payload={},
                template_ids_used=[],
                previous_score=score_prediction(benchmark_slug, initial_answer, sample.reference_answer) if initial_answer else 0.0,
                refined_score=float(prediction_row["score"]),
                stopped_reason=stopped_reason,
            )
        )
    return {
        "turn_rows": turn_rows,
        "critic_rows": critic_rows,
        "refinement_rows": refinement_rows,
        "prediction_row": prediction_row,
    }


def _execute_reasoning_turn(**kwargs) -> dict[str, Any]:
    return _execute_structured_turn(validator=validate_reasoning_answer_output, **kwargs)


def _execute_judge_turn(
    *,
    current_reasoning: str,
    current_answer: str,
    sample: DatasetSample,
    **kwargs,
) -> dict[str, Any]:
    return _execute_structured_turn(
        role="judge",
        messages=build_judge_messages(sample, current_reasoning=current_reasoning, current_answer=current_answer),
        validator=validate_judge_output,
        sample=sample,
        **kwargs,
    )


def _execute_critic_turn(
    *,
    current_reasoning: str,
    current_answer: str,
    sample: DatasetSample,
    judge_payload: dict[str, Any],
    template_hints: list[TemplateHint],
    **kwargs,
) -> dict[str, Any]:
    return _execute_structured_turn(
        role="critic",
        messages=build_critic_messages(
            sample,
            current_reasoning=current_reasoning,
            current_answer=current_answer,
            judge_payload=judge_payload,
            template_hints=template_hints,
        ),
        validator=validate_critic_output,
        sample=sample,
        **kwargs,
    )


def _execute_refiner_turn(
    *,
    current_reasoning: str,
    current_answer: str,
    sample: DatasetSample,
    judge_payload: dict[str, Any],
    critic_payload: dict[str, Any],
    template_hints: list[TemplateHint],
    **kwargs,
) -> dict[str, Any]:
    return _execute_structured_turn(
        role="refiner",
        messages=build_refiner_messages(
            sample,
            current_reasoning=current_reasoning,
            current_answer=current_answer,
            judge_payload=judge_payload,
            critic_payload=critic_payload,
            template_hints=template_hints,
        ),
        validator=lambda assistant_text, provider_reasoning_text: validate_refiner_output_with_fallback(
            assistant_text,
            provider_reasoning_text,
            previous_reasoning=current_reasoning,
            previous_answer=current_answer,
        ),
        sample=sample,
        **kwargs,
    )


def _execute_curator_turn(
    *,
    sample: DatasetSample,
    judge_payload: dict[str, Any],
    critic_payload: dict[str, Any],
    improved: bool,
    **kwargs,
) -> dict[str, Any]:
    return _execute_structured_turn(
        role="curator",
        messages=build_curator_messages(
            sample,
            judge_payload=judge_payload,
            critic_payload=critic_payload,
            improved=improved,
        ),
        validator=validate_curator_output,
        sample=sample,
        **kwargs,
    )


def _execute_structured_turn(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    method_name: str,
    role: str,
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
    return {
        "run_id": run_id,
        "dataset": dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "table_id": str(sample.metadata.get("table_id") or ""),
        "question_type": str(sample.metadata.get("question_type") or ""),
        "method_name": method_name,
        "method_type": "table_critic",
        "role": role,
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
        "validated_output": result.validated_output,
    }


def _coerce_judge_payload(turn_row: dict[str, Any]) -> dict[str, Any]:
    payload = turn_row.get("validated_output")
    if isinstance(payload, dict) and "passed" in payload:
        return dict(payload)
    request_error = str(turn_row.get("request_error") or "").strip()
    rationale = request_error or "Judge output did not pass structured validation."
    return {
        "passed": False,
        "error_detected": True,
        "error_step": "Logical Error",
        "node_path": ["ROOT", "Final Query Error", "Logical Error"],
        "rationale": rationale,
    }


def _coerce_critic_payload(turn_row: dict[str, Any]) -> dict[str, Any]:
    payload = turn_row.get("validated_output")
    if isinstance(payload, dict) and payload.get("critic_feedback"):
        return dict(payload)
    request_error = str(turn_row.get("request_error") or "").strip()
    feedback = request_error or "Critic output did not pass structured validation."
    return {
        "critic_feedback": feedback,
        "conflicting_evidence": [],
        "repair_hint": "Re-evaluate the table evidence and revise the final answer conservatively.",
        "error_category": "fallback_critic",
    }


def _coerce_reasoning_payload(
    turn_row: dict[str, Any],
    *,
    previous_reasoning: str,
    previous_answer: str,
) -> dict[str, Any]:
    payload = turn_row.get("validated_output")
    if isinstance(payload, dict) and payload.get("final_answer"):
        return dict(payload)
    return {
        "reasoning": previous_reasoning,
        "final_answer": previous_answer,
    }


def _coerce_curator_payload(turn_row: dict[str, Any]) -> dict[str, Any]:
    payload = turn_row.get("validated_output")
    if isinstance(payload, dict) and payload.get("pattern_summary"):
        return dict(payload)
    return {
        "pattern_summary": "Structured curation output missing; keep a generic correction summary.",
        "reuse_hint": "Recheck the judge-critic mismatch before reusing this template.",
        "template_title": "Generic refinement fallback",
    }


def _stabilize_answer_format_answer(
    *,
    previous_answer: str,
    refined_answer: str,
    judge_payload: dict[str, Any],
) -> str:
    """`Answer Format Error` 只允许修正格式，不允许任意改写答案语义。"""

    if str(judge_payload.get("error_step") or "") != "Answer Format Error":
        return refined_answer
    normalized_previous = _normalize_trace_text(previous_answer)
    normalized_refined = _normalize_trace_text(refined_answer)
    if not normalized_previous:
        return refined_answer
    if not normalized_refined or normalized_previous != normalized_refined:
        return previous_answer
    return refined_answer


def _initial_messages_for_mode(sample: DatasetSample, mode: str) -> list[dict[str, str]]:
    if mode == "direct_qa":
        return build_direct_messages(sample)
    if mode == "few_shot_qa":
        return build_few_shot_messages(sample)
    if mode in {"chain_of_table", "critic_cot", "table_critic_paper"}:
        return build_chain_of_table_messages(sample)
    raise ValueError(f"Unsupported Table-Critic method mode: {mode}")


def _build_prediction_row(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    method: TableCriticMethodSpec,
    turn_rows: list[dict[str, Any]],
    initial_reasoning: str,
    initial_answer: str,
    final_reasoning: str,
    final_answer: str,
    final_judge_payload: dict[str, Any],
    critic_feedback: str,
    refinement_round_count: int,
    stopped_reason: str,
    template_ids_used: list[str],
) -> dict[str, Any]:
    initial_score = score_prediction(dataset, initial_answer, sample.reference_answer) if initial_answer else 0.0
    final_score = score_prediction(dataset, final_answer, sample.reference_answer) if final_answer else 0.0
    normalized_prediction = normalize_prediction(dataset, final_answer) if final_answer else ""
    answer_changed = _normalize_trace_text(initial_answer) != _normalize_trace_text(final_answer)
    return {
        "run_id": run_id,
        "dataset": dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "method_name": method.name,
        "method_type": "table_critic",
        "method_mode": method.mode,
        "model_name": "",
        "table_id": str(sample.metadata.get("table_id") or ""),
        "question_type": str(sample.metadata.get("question_type") or ""),
        "initial_reasoning": initial_reasoning,
        "initial_answer": initial_answer,
        "final_reasoning": final_reasoning,
        "final_answer": final_answer,
        "prediction": normalized_prediction or final_answer,
        "gold": sample.reference_answer,
        "score": final_score,
        "initial_score": initial_score,
        "judge_error_detected": bool(final_judge_payload.get("error_detected")),
        "judge_error_step": str(final_judge_payload.get("error_step") or "None"),
        "critic_feedback": critic_feedback,
        "refinement_round_count": refinement_round_count,
        "stopped_reason": stopped_reason,
        "template_ids_used": list(template_ids_used),
        "answer_changed": answer_changed,
        "correction_flag": final_score > initial_score,
        "degradation_flag": final_score < initial_score,
        "template_reused": bool(template_ids_used),
        "prompt_tokens_per_question": sum(float(row.get("prompt_tokens") or 0.0) for row in turn_rows),
        "completion_tokens_per_question": sum(float(row.get("completion_tokens") or 0.0) for row in turn_rows),
        "total_tokens_per_question": sum(float(row.get("total_tokens") or 0.0) for row in turn_rows),
        "latency_ms_per_question": sum(float(row.get("latency_ms") or 0.0) for row in turn_rows),
        "calls_per_question": len(turn_rows),
    }


def _build_critic_trace_row(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    method: TableCriticMethodSpec,
    final_judge_payload: dict[str, Any],
    critic_feedback: str,
    template_ids_used: list[str],
    refinement_round_count: int,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "dataset": dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "method_name": method.name,
        "table_id": str(sample.metadata.get("table_id") or ""),
        "question_type": str(sample.metadata.get("question_type") or ""),
        "judge_error_detected": bool(final_judge_payload.get("error_detected")),
        "judge_error_step": str(final_judge_payload.get("error_step") or "None"),
        "judge_node_path": list(final_judge_payload.get("node_path") or ["ROOT"]),
        "judge_rationale": str(final_judge_payload.get("rationale") or ""),
        "critic_feedback": critic_feedback,
        "template_ids_used": list(template_ids_used),
        "template_reuse_count": len(template_ids_used),
        "refinement_round_count": refinement_round_count,
    }


def _build_refinement_trace_row(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    method: TableCriticMethodSpec,
    round_index: int,
    initial_answer: str,
    previous_answer: str,
    refined_answer: str,
    judge_payload: dict[str, Any],
    critic_payload: dict[str, Any],
    template_ids_used: list[str],
    previous_score: float,
    refined_score: float,
    stopped_reason: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "dataset": dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "method_name": method.name,
        "table_id": str(sample.metadata.get("table_id") or ""),
        "question_type": str(sample.metadata.get("question_type") or ""),
        "round_index": round_index,
        "initial_answer": initial_answer,
        "previous_answer": previous_answer,
        "refined_answer": refined_answer,
        "judge_error_detected": bool(judge_payload.get("error_detected")),
        "judge_error_step": str(judge_payload.get("error_step") or "None"),
        "critic_feedback": str(critic_payload.get("critic_feedback") or ""),
        "template_ids_used": list(template_ids_used),
        "previous_score": previous_score,
        "refined_score": refined_score,
        "improved": refined_score > previous_score,
        "degraded": refined_score < previous_score,
        "stopped_reason": stopped_reason,
    }


def _seed_template_tree(protocol: ProtocolConfig) -> dict[str, Any]:
    tree = {
        "node_id": "node-root",
        "name": "ROOT",
        "path": "ROOT",
        "summaries": [],
        "children": {},
        "_next_template_index": 1,
    }
    for path in protocol.seed_template_paths:
        _ensure_template_node(tree, path.split("/"))
    return tree


def _lookup_template_hints(tree: dict[str, Any], node_path: list[str], *, max_examples: int) -> list[TemplateHint]:
    hints: list[TemplateHint] = []
    parts = node_path[:] if node_path else ["ROOT"]
    while parts:
        node = _find_template_node(tree, parts)
        if node is not None:
            for summary in node.get("summaries", []):
                hints.append(
                    TemplateHint(
                        template_id=summary["template_id"],
                        path=summary["path"],
                        pattern_summary=summary["pattern_summary"],
                        reuse_hint=summary["reuse_hint"],
                    )
                )
        if len(hints) >= max_examples:
            break
        parts = parts[:-1]
    return hints[:max_examples]


def _update_template_tree(
    tree: dict[str, Any],
    node_path: list[str],
    curator_payload: dict[str, Any],
    *,
    improved: bool,
    max_summaries_per_node: int,
) -> str:
    node = _ensure_template_node(tree, node_path or ["ROOT"])
    normalized_summary = _normalize_trace_text(str(curator_payload.get("pattern_summary") or ""))
    for summary in node["summaries"]:
        if _normalize_trace_text(summary["pattern_summary"]) == normalized_summary:
            summary["usage_count"] += 1
            if improved:
                summary["success_count"] += 1
            return summary["template_id"]

    template_id = f"tpl-{int(tree['_next_template_index']):05d}"
    tree["_next_template_index"] += 1
    node["summaries"].append(
        {
            "template_id": template_id,
            "path": node["path"],
            "pattern_summary": str(curator_payload["pattern_summary"]),
            "reuse_hint": str(curator_payload["reuse_hint"]),
            "template_title": str(curator_payload["template_title"]),
            "usage_count": 1,
            "success_count": 1 if improved else 0,
        }
    )
    node["summaries"] = node["summaries"][-max_summaries_per_node:]
    return template_id


def _ensure_template_node(tree: dict[str, Any], parts: list[str]) -> dict[str, Any]:
    current = tree
    normalized_parts = parts[:]
    if not normalized_parts or normalized_parts[0] != "ROOT":
        normalized_parts.insert(0, "ROOT")
    for part in normalized_parts[1:]:
        children = current.setdefault("children", {})
        if part not in children:
            parent_path = str(current["path"])
            children[part] = {
                "node_id": f"node-{len(children) + 1}-{part.lower().replace(' ', '-')}",
                "name": part,
                "path": f"{parent_path}/{part}",
                "summaries": [],
                "children": {},
            }
        current = children[part]
    return current


def _find_template_node(tree: dict[str, Any], parts: list[str]) -> dict[str, Any] | None:
    current = tree
    normalized_parts = parts[:]
    if not normalized_parts or normalized_parts[0] != "ROOT":
        normalized_parts.insert(0, "ROOT")
    for part in normalized_parts[1:]:
        current = current.get("children", {}).get(part)
        if current is None:
            return None
    return current


def _serialize_template_tree(tree: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": tree["name"],
        "path": tree["path"],
        "summaries": [
            {
                "template_id": item["template_id"],
                "path": item["path"],
                "template_title": item["template_title"],
                "pattern_summary": item["pattern_summary"],
                "reuse_hint": item["reuse_hint"],
                "usage_count": item["usage_count"],
                "success_count": item["success_count"],
            }
            for item in tree.get("summaries", [])
        ],
        "children": {name: _serialize_template_tree(node) for name, node in sorted(tree.get("children", {}).items())},
    }


def _build_metrics(
    prediction_rows: list[dict[str, Any]],
    methods: list[TableCriticMethodSpec],
    *,
    model_name: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    method_order = {method.name: index for index, method in enumerate(methods)}
    grouped = _group_prediction_rows(prediction_rows)
    for (dataset, method_name), items in grouped.items():
        rows.append(_summarize_prediction_group(dataset, method_name, items, model_name))
    overall_rows = _build_overall_rows(grouped, methods, model_name)
    rows.extend(overall_rows)

    chain_scores = {
        row["dataset"]: row["accuracy_mean"]
        for row in rows
        if row["method_name"] == "chain_of_table"
    }
    for row in rows:
        baseline = chain_scores.get(row["dataset"])
        row["gain_over_chain_of_table"] = round(float(row["accuracy_mean"]) - float(baseline), 6) if baseline is not None and row["method_name"] != "chain_of_table" else None

    rows.sort(key=lambda row: (row["dataset"], method_order.get(row["method_name"], 999), row["method_name"]))
    return {"summary": rows, "prediction_count": len(prediction_rows)}


def _group_prediction_rows(prediction_rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in prediction_rows:
        grouped.setdefault((str(row["dataset"]), str(row["method_name"])), []).append(row)
    return grouped


def _summarize_prediction_group(dataset: str, method_name: str, items: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    scores = [float(item.get("score") or 0.0) for item in items]
    prompt_tokens = [float(item.get("prompt_tokens_per_question") or 0.0) for item in items]
    completion_tokens = [float(item.get("completion_tokens_per_question") or 0.0) for item in items]
    total_tokens = [float(item.get("total_tokens_per_question") or 0.0) for item in items]
    latencies = [float(item.get("latency_ms_per_question") or 0.0) for item in items]
    calls = [float(item.get("calls_per_question") or 0.0) for item in items]
    refinement_rounds = [float(item.get("refinement_round_count") or 0.0) for item in items]
    judge_errors = sum(1 for item in items if item.get("judge_error_detected"))
    corrections = sum(1 for item in items if item.get("correction_flag"))
    degradations = sum(1 for item in items if item.get("degradation_flag"))
    template_reuse = sum(1 for item in items if item.get("template_reused"))
    answer_changed = sum(1 for item in items if item.get("answer_changed"))
    return {
        "dataset": dataset,
        "model_name": model_name,
        "method_name": method_name,
        "display_name": method_name,
        "prediction_rows": len(items),
        "question_count": len(items),
        "accuracy_mean": safe_mean(scores),
        "prompt_tokens_mean": safe_mean(prompt_tokens),
        "completion_tokens_mean": safe_mean(completion_tokens),
        "total_tokens_mean": safe_mean(total_tokens),
        "calls_per_question_mean": safe_mean(calls),
        "latency_ms_mean": safe_mean(latencies),
        "accuracy_per_1k_tokens": round(safe_mean(scores) / safe_mean(total_tokens) * 1000, 6) if safe_mean(total_tokens) else 0.0,
        "refinement_round_count_mean": safe_mean(refinement_rounds),
        "judge_error_detected_rate": safe_ratio(judge_errors, len(items)),
        "correction_rate": safe_ratio(corrections, len(items)),
        "degradation_rate": safe_ratio(degradations, len(items)),
        "template_reuse_rate": safe_ratio(template_reuse, len(items)),
        "changed_answer_rate": safe_ratio(answer_changed, len(items)),
    }


def _build_overall_rows(
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    methods: list[TableCriticMethodSpec],
    model_name: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in methods:
        collected: list[dict[str, Any]] = []
        for (dataset, method_name), items in grouped.items():
            if method_name == method.name and dataset != "overall":
                collected.extend(items)
        if not collected:
            continue
        row = _summarize_prediction_group("overall", method.name, collected, model_name)
        rows.append(row)
    return rows


def _build_error_analysis(
    prediction_rows: list[dict[str, Any]],
    critic_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summary_rows: list[dict[str, Any]] = []
    grouped_predictions = _group_prediction_rows(prediction_rows)
    critic_grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in critic_rows:
        critic_grouped.setdefault((str(row["dataset"]), str(row["method_name"])), []).append(row)

    for (dataset, method_name), rows in grouped_predictions.items():
        critic_items = critic_grouped.get((dataset, method_name), [])
        error_counter = Counter(str(item.get("judge_error_step") or "None") for item in critic_items)
        summary_rows.append(
            {
                "dataset": dataset,
                "method_name": method_name,
                "question_count": len(rows),
                "judge_error_detected_rate": safe_ratio(sum(1 for item in rows if item.get("judge_error_detected")), len(rows)),
                "correction_rate": safe_ratio(sum(1 for item in rows if item.get("correction_flag")), len(rows)),
                "degradation_rate": safe_ratio(sum(1 for item in rows if item.get("degradation_flag")), len(rows)),
                "template_reuse_rate": safe_ratio(sum(1 for item in rows if item.get("template_reused")), len(rows)),
                "top_error_step": error_counter.most_common(1)[0][0] if error_counter else "None",
            }
        )

    error_step_rows: list[dict[str, Any]] = []
    error_grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in critic_rows:
        key = (str(row["dataset"]), str(row["method_name"]), str(row.get("judge_error_step") or "None"))
        error_grouped.setdefault(key, []).append(row)
    for (dataset, method_name, error_step), rows in sorted(error_grouped.items()):
        error_step_rows.append(
            {
                "dataset": dataset,
                "method_name": method_name,
                "judge_error_step": error_step,
                "count": len(rows),
            }
        )
    return {"summary_rows": summary_rows, "error_step_rows": error_step_rows}


def _normalize_trace_text(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _deduplicate_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output
