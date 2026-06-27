"""CRED-V 样本级执行辅助。"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any

from research_experiments.core.data.datasets import DatasetSample, load_split_ids, select_samples
from research_experiments.core.data.evaluation import normalize_prediction, score_prediction
from research_experiments.core.execution.runner_common import iter_indexed_batch
from research_experiments.families.cred_v.algorithms import (
    aggregate_adaptive_candidate_search,
    aggregate_safe_verification,
    aggregate_stage_a_vote,
    aggregate_task_verification,
    build_router_decision,
    choice_permutation,
    evidence_quality,
    expansion_mode_for_dataset,
    map_shuffled_choice_answer,
    select_verification_targets,
)
from research_experiments.families.cred_v.config import (
    CRED_ACS_METHODS,
    CRED_COMM_METHODS,
    CRED_SAFE_VERIFY_METHODS,
    CRED_VERIFY_METHODS,
    CredVExperimentConfig,
    CredVProtocolConfig,
)
from research_experiments.families.cred_v.prompts import (
    AGENT_ROLES,
    build_choice_shuffle_solver_messages,
    build_hotpot_span_extractor_messages,
    build_math_symbolic_repair_messages,
    build_safe_hetero_verifier_messages,
    build_stage_a_messages,
    build_strategyqa_dual_polarity_messages,
    build_task_verifier_messages,
)
from research_experiments.family_runtime.common import resolve_phase_split_name, safe_mean, safe_ratio
from research_experiments.family_runtime.output_protocols import (
    FREE_TEXT_ANSWER_PROTOCOL_V1,
    execute_output_protocol_turn,
)


@dataclass(frozen=True)
class CredTurnRecord:
    run_id: str
    dataset: str
    split: str
    sample_id: str
    method_name: str
    method_type: str
    round_index: int
    agent_id: int
    role: str
    agent_role: str
    prompt_hash: str
    prediction: str
    normalized_answer: str
    score: float | None
    output_status: str
    prompt_tokens: float
    completion_tokens: float
    total_tokens: float
    latency_ms: float
    cache_hit: bool
    request_error: str | None
    request_status: str
    raw_finish_reason: str | None
    output_protocol: str
    protocol_parse_status: str
    protocol_parse_error: str | None
    reason_present: bool
    request_count: int
    cache_request_count: int
    network_request_count: int
    raw_prompt_tokens: float
    raw_completion_tokens: float
    raw_total_tokens: float
    raw_latency_ms: float
    visible_peer_count: int
    confidence_value: float | None
    key_evidence: str
    risk_level: str
    failure_risk: str
    evidence_quality: float
    payload: dict[str, Any]
    assistant_text: str
    provider_reasoning_text: str
    validated_output: dict[str, Any]


@dataclass(frozen=True)
class DebateMessageRecord:
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
    message_kind: str


@dataclass(frozen=True)
class CredVerifierRuntime:
    model_ref: str
    backbone: Any
    provider: Any
    cache: Any
    throttle: Any


@dataclass(frozen=True)
class CredPredictionRecord:
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
    corrected_by_verification: bool
    harmed_by_verification: bool
    unchanged_correct: bool
    unchanged_wrong: bool
    triggered: bool
    router_reasons: list[str]
    resolver: str
    survival_support: dict[str, float]
    verification_support: dict[str, float]
    oracle_candidate_correct: bool
    wrong_majority_some_correct: bool
    target_correct: bool | None
    safe_repair_applied: bool
    hetero_agreement_applied: bool
    expansion_call_count: int
    expansion_mode: str
    false_consensus_triggered: bool
    candidate_pool_oracle_correct: bool
    expansion_oracle_correct: bool
    expansion_validation_pass_count: int
    math_repair_applied: bool
    hotpot_span_repair_applied: bool
    choice_shuffle_agreement_count: int
    single_pro_promotion_blocked: bool
    protocol_failures_per_question: int
    reason_missing_turns_per_question: int


def run_cred_batch(
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    samples: list[DatasetSample],
    experiment: CredVExperimentConfig,
    protocol: CredVProtocolConfig,
    backbone,
    provider,
    cache,
    throttle,
    verifier_runtimes: list[CredVerifierRuntime] | None = None,
) -> Iterator[tuple[int, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]]:
    worker = partial(
        _run_cred_sample,
        run_id=run_id,
        benchmark_slug=benchmark_slug,
        split_name=split_name,
        experiment=experiment,
        protocol=protocol,
        backbone=backbone,
        provider=provider,
        cache=cache,
        throttle=throttle,
        verifier_runtimes=verifier_runtimes or [],
    )
    for sample_index, result in iter_indexed_batch(
        samples,
        worker=worker,
        max_concurrent_requests=experiment.max_concurrent_requests,
    ):
        yield (sample_index, *result)


def append_outputs(
    *,
    sample_results: Iterable[tuple[int, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]],
    dataset_slug: str,
    progress,
    turn_writer,
    debate_writer,
    router_writer,
    prediction_writer,
    all_turns: list[dict[str, Any]],
    all_debate_rows: list[dict[str, Any]],
    all_router_rows: list[dict[str, Any]],
    all_predictions: list[dict[str, Any]],
) -> None:
    for _, turn_rows, debate_rows, router_rows, prediction_rows in sample_results:
        for row in turn_rows:
            turn_writer.write_row(row)
            progress.record_call(row, method_key="method_name")
        for row in debate_rows:
            debate_writer.write_row(row)
        for row in router_rows:
            router_writer.write_row(row)
        for row in prediction_rows:
            prediction_writer.write_row(row)
            progress.record_predictions(1, dataset_slug, row["method_name"])
        all_turns.extend(turn_rows)
        all_debate_rows.extend(debate_rows)
        all_router_rows.extend(router_rows)
        all_predictions.extend(prediction_rows)


def _run_cred_sample(
    sample: DatasetSample,
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    experiment: CredVExperimentConfig,
    protocol: CredVProtocolConfig,
    backbone,
    provider,
    cache,
    throttle,
    verifier_runtimes: list[CredVerifierRuntime],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    stage_rows: list[dict[str, Any]] = []
    for index, agent_role in enumerate(AGENT_ROLES[: protocol.stage_a_agent_count], start=1):
        stage_rows.append(
            _execute_turn(
                run_id=run_id,
                dataset=benchmark_slug,
                split_name=split_name,
                sample=sample,
                method_name="cred_stage_a",
                method_type="mad",
                round_index=0,
                agent_id=index,
                role="initial",
                agent_role=agent_role,
                visible_peer_count=0,
                messages=build_stage_a_messages(
                    sample,
                    agent_id=index,
                    agent_role=agent_role,
                    output_protocol=experiment.cred_stage_a_output_protocol,
                ),
                backbone=backbone,
                provider=provider,
                cache=cache,
                throttle=throttle,
                temperature=protocol.stage_a_temperature,
                top_p=protocol.top_p,
                seed=experiment.global_seed + index - 1,
                output_protocol=experiment.cred_stage_a_output_protocol,
                max_tokens=_positive_token_cap(protocol.stage_a_max_tokens),
            )
        )

    vote_decision = aggregate_stage_a_vote(stage_rows)
    router = build_router_decision(stage_rows, protocol=protocol)
    method_set = set(experiment.cred_methods)
    verification_rows: list[dict[str, Any]] = []
    safe_verifier_rows: list[dict[str, Any]] = []
    expansion_rows: list[dict[str, Any]] = []
    debate_rows: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []

    if router.triggered and CRED_COMM_METHODS & method_set:
        targets = select_verification_targets(
            dataset=benchmark_slug,
            rows=stage_rows,
            leading_answer=vote_decision.final_answer,
            max_verifications=protocol.max_verifications,
        )
    if router.triggered and CRED_VERIFY_METHODS & method_set:
        for verification_index, target in enumerate(targets, start=1):
            verifier_row = _execute_turn(
                run_id=run_id,
                dataset=benchmark_slug,
                split_name=split_name,
                sample=sample,
                method_name="cred_task_verifier",
                method_type="verification",
                round_index=1,
                agent_id=100 + verification_index,
                role="verification",
                agent_role="task_verifier",
                visible_peer_count=len(stage_rows),
                messages=build_task_verifier_messages(
                    sample,
                    leading_answer=vote_decision.final_answer,
                    target_row=target,
                    stage_rows=stage_rows,
                ),
                backbone=backbone,
                provider=provider,
                cache=cache,
                throttle=throttle,
                temperature=protocol.verifier_temperature,
                top_p=protocol.top_p,
                seed=experiment.global_seed + 100 + verification_index,
                output_protocol=experiment.cred_verification_output_protocol,
                max_tokens=_positive_token_cap(protocol.verifier_max_tokens),
            )
            verification_rows.append(verifier_row)
            debate_rows.append(_debate_message_row(run_id, benchmark_slug, split_name, sample, verifier_row, "task_verification"))

    if (
        router.triggered
        and CRED_SAFE_VERIFY_METHODS & method_set
        and "hetero_verified" in set(protocol.verification_modes)
        and verifier_runtimes
    ):
        safe_runtimes = [
            runtime
            for runtime in verifier_runtimes
            if protocol.allow_same_model_promotion or str(runtime.backbone.name) != str(backbone.name)
        ]
        call_count = min(int(protocol.max_verification_calls), len(targets))
        call_count = min(call_count, len(safe_runtimes))
        for verification_index in range(1, call_count + 1):
            target = targets[verification_index - 1]
            runtime = safe_runtimes[(verification_index - 1) % len(safe_runtimes)]
            verifier_row = _execute_turn(
                run_id=run_id,
                dataset=benchmark_slug,
                split_name=split_name,
                sample=sample,
                method_name="cred_safe_hetero_verifier",
                method_type="verification",
                round_index=1,
                agent_id=400 + verification_index,
                role="verification",
                agent_role=f"safe_hetero_verifier:{runtime.model_ref}",
                visible_peer_count=len(stage_rows),
                messages=build_safe_hetero_verifier_messages(
                    sample,
                    leading_answer=vote_decision.final_answer,
                    target_row=target,
                    stage_rows=stage_rows,
                ),
                backbone=runtime.backbone,
                provider=runtime.provider,
                cache=runtime.cache,
                throttle=runtime.throttle,
                temperature=protocol.verifier_temperature,
                top_p=protocol.top_p,
                seed=experiment.global_seed + 400 + verification_index,
                output_protocol=experiment.cred_verification_output_protocol,
                max_tokens=_positive_token_cap(protocol.verifier_max_tokens),
            )
            safe_verifier_rows.append(verifier_row)
            debate_rows.append(_debate_message_row(run_id, benchmark_slug, split_name, sample, verifier_row, "safe_hetero_verification"))

    if router.triggered and CRED_ACS_METHODS & method_set and verifier_runtimes and int(protocol.max_expansion_calls) > 0:
        expansion_rows.extend(
            _run_acs_expansions(
                run_id=run_id,
                benchmark_slug=benchmark_slug,
                split_name=split_name,
                sample=sample,
                experiment=experiment,
                protocol=protocol,
                stage_rows=stage_rows,
                vote_decision=vote_decision,
                verifier_runtimes=verifier_runtimes,
            )
        )
        for row in expansion_rows:
            debate_rows.append(_debate_message_row(run_id, benchmark_slug, split_name, sample, row, "acs_expansion"))

    all_turns = [*stage_rows, *verification_rows, *safe_verifier_rows, *expansion_rows]
    router_row = _router_row(
        run_id=run_id,
        dataset=benchmark_slug,
        split_name=split_name,
        sample=sample,
        router=router,
        self_verification_count=len(verification_rows),
        hetero_verification_count=len(safe_verifier_rows),
        expansion_count=len(expansion_rows),
    )
    prediction_rows: list[dict[str, Any]] = []
    for method_name in experiment.cred_methods:
        decision = vote_decision
        method_turns = list(stage_rows)
        debate_turns: list[dict[str, Any]] = []
        if method_name in CRED_VERIFY_METHODS:
            decision = aggregate_task_verification(
                dataset=benchmark_slug,
                stage_rows=stage_rows,
                verifier_rows=verification_rows,
                stage_winner=vote_decision.final_answer,
                promotion_confidence_min=protocol.promotion_confidence_min,
                promotion_score_margin=protocol.promotion_score_margin,
                concrete_evidence_min_chars=protocol.concrete_evidence_min_chars,
            )
            debate_turns = list(verification_rows)
            method_turns = [*stage_rows, *debate_turns]
        elif method_name in CRED_SAFE_VERIFY_METHODS:
            decision = aggregate_safe_verification(
                dataset=benchmark_slug,
                question=sample.question,
                context=sample.prompt_context,
                stage_rows=stage_rows,
                verifier_rows=verification_rows,
                hetero_verifier_rows=safe_verifier_rows,
                stage_winner=vote_decision.final_answer,
                verification_modes=protocol.verification_modes,
                allow_same_model_promotion=protocol.allow_same_model_promotion,
                concrete_evidence_min_chars=protocol.concrete_evidence_min_chars,
                strong_majority_count=protocol.strong_majority_count,
            )
            debate_turns = list(safe_verifier_rows)
            method_turns = [*stage_rows, *debate_turns]
        elif method_name in CRED_ACS_METHODS:
            decision = aggregate_adaptive_candidate_search(
                dataset=benchmark_slug,
                question=sample.question,
                context=sample.prompt_context,
                stage_rows=stage_rows,
                expansion_rows=expansion_rows,
                stage_winner=vote_decision.final_answer,
                expansion_modes=protocol.expansion_modes,
                promotion_min_independent_support=protocol.promotion_min_independent_support,
                promotion_margin_min=protocol.promotion_margin_min,
                strong_majority_count=protocol.strong_majority_count,
            )
            debate_turns = list(expansion_rows)
            method_turns = [*stage_rows, *debate_turns]
        prediction_rows.append(
            _prediction_row(
                run_id=run_id,
                dataset=benchmark_slug,
                split_name=split_name,
                sample=sample,
                method_name=method_name,
                backbone_name=backbone.name,
                decision=decision,
                stage_decision=vote_decision,
                router=router,
                stage_rows=stage_rows,
                debate_rows=debate_turns,
                method_turns=method_turns,
                targets=targets,
                expansion_rows=expansion_rows,
            )
        )
    return all_turns, debate_rows, [router_row], prediction_rows


def _run_acs_expansions(
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    sample: DatasetSample,
    experiment: CredVExperimentConfig,
    protocol: CredVProtocolConfig,
    stage_rows: list[dict[str, Any]],
    vote_decision,
    verifier_runtimes: list[CredVerifierRuntime],
) -> list[dict[str, Any]]:
    mode = expansion_mode_for_dataset(benchmark_slug, protocol.expansion_modes)
    if not mode:
        return []
    allowed_refs = set(protocol.expansion_model_refs) or set(experiment.verifier_model_refs)
    runtimes = [runtime for runtime in verifier_runtimes if not allowed_refs or runtime.model_ref in allowed_refs]
    if not runtimes:
        return []
    max_calls = int(protocol.max_expansion_calls)
    if mode in {"math_symbolic_repair", "hotpot_span_extract"}:
        max_calls = min(max_calls, 1)
    elif mode == "strategyqa_dual_polarity":
        max_calls = min(max_calls, 3)
    elif mode == "mc_choice_shuffle":
        max_calls = min(max_calls, 3)
        if not _multiple_choice_options(sample):
            return []
    rows: list[dict[str, Any]] = []
    for index in range(1, max_calls + 1):
        runtime = runtimes[(index - 1) % len(runtimes)]
        permutation: list[int] = []
        shuffled_options: list[str] = []
        if mode == "math_symbolic_repair":
            messages = build_math_symbolic_repair_messages(
                sample,
                leading_answer=vote_decision.final_answer,
                stage_rows=stage_rows,
            )
        elif mode == "hotpot_span_extract":
            messages = build_hotpot_span_extractor_messages(
                sample,
                leading_answer=vote_decision.final_answer,
                stage_rows=stage_rows,
            )
        elif mode == "strategyqa_dual_polarity":
            messages = build_strategyqa_dual_polarity_messages(
                sample,
                variant_index=index,
                leading_answer=vote_decision.final_answer,
                stage_rows=stage_rows,
            )
        else:
            options = _multiple_choice_options(sample)
            permutation = choice_permutation(len(options), index)
            shuffled_options = [options[position] for position in permutation]
            messages = build_choice_shuffle_solver_messages(
                sample,
                shuffled_options=shuffled_options,
                shuffle_label=f"shuffle_{index}",
            )
        row = _execute_turn(
            run_id=run_id,
            dataset=benchmark_slug,
            split_name=split_name,
            sample=sample,
            method_name="cred_acs_expansion",
            method_type="expansion",
            round_index=1,
            agent_id=600 + index,
            role="expansion",
            agent_role=f"acs:{mode}:{runtime.model_ref}",
            visible_peer_count=len(stage_rows),
            messages=messages,
            backbone=runtime.backbone,
            provider=runtime.provider,
            cache=runtime.cache,
            throttle=runtime.throttle,
            temperature=protocol.verifier_temperature,
            top_p=protocol.top_p,
            seed=experiment.global_seed + 600 + index,
            output_protocol=experiment.cred_verification_output_protocol,
            max_tokens=_positive_token_cap(protocol.verifier_max_tokens),
        )
        _annotate_expansion_row(
            row,
            sample=sample,
            mode=mode,
            expansion_index=index,
            permutation=permutation,
            shuffled_options=shuffled_options,
        )
        rows.append(row)
    return rows


def _execute_turn(
    *,
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
    provider,
    cache,
    throttle,
    temperature: float,
    top_p: float,
    seed: int,
    agent_role: str = "",
    output_protocol: str = FREE_TEXT_ANSWER_PROTOCOL_V1,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    result = execute_output_protocol_turn(
        backbone=backbone,
        provider=provider,
        cache=cache,
        throttle=throttle,
        sample=sample,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        dataset=dataset,
        role=role,
        output_protocol=output_protocol,
        max_tokens=max_tokens,
    )
    final_answer = str(result.validated_output.get("final_answer") or result.validated_output.get("answer") or "")
    normalized = normalize_prediction(dataset, final_answer) if final_answer else ""
    confidence = result.validated_output.get("confidence")
    try:
        confidence_value = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence_value = None
    key_evidence = str(result.validated_output.get("key_evidence") or result.validated_output.get("evidence") or "")
    risk_level = str(result.validated_output.get("risk_level") or "none").strip().lower()
    failure_risk = str(result.validated_output.get("risk_summary") or "")
    row = asdict(
        CredTurnRecord(
            run_id=run_id,
            dataset=dataset,
            split=split_name,
            sample_id=sample.sample_id,
            method_name=method_name,
            method_type=method_type,
            round_index=round_index,
            agent_id=agent_id,
            role=role,
            agent_role=agent_role,
            prompt_hash=result.prompt_hash,
            prediction=normalized,
            normalized_answer=normalized,
            score=None,
            output_status=result.output_status,
            prompt_tokens=float(result.usage.get("prompt_tokens") or 0.0),
            completion_tokens=float(result.usage.get("completion_tokens") or 0.0),
            total_tokens=float(result.usage.get("total_tokens") or 0.0),
            latency_ms=float(result.response_payload.get("latency_ms") or 0.0),
            cache_hit=result.cache_hit,
            request_error=result.request_error,
            request_status=result.request_status,
            raw_finish_reason=result.raw_finish_reason,
            output_protocol=result.output_protocol,
            protocol_parse_status=result.protocol_parse_status,
            protocol_parse_error=result.protocol_parse_error,
            reason_present=result.reason_present,
            request_count=result.request_count,
            cache_request_count=result.cache_request_count,
            network_request_count=result.network_request_count,
            raw_prompt_tokens=float(result.usage.get("prompt_tokens") or 0.0),
            raw_completion_tokens=float(result.usage.get("completion_tokens") or 0.0),
            raw_total_tokens=float(result.usage.get("total_tokens") or 0.0),
            raw_latency_ms=float(result.response_payload.get("latency_ms") or 0.0),
            visible_peer_count=visible_peer_count,
            confidence_value=confidence_value,
            key_evidence=key_evidence,
            risk_level=risk_level if risk_level in {"none", "low", "medium", "high"} else "none",
            failure_risk=failure_risk,
            evidence_quality=0.0,
            payload=result.payload,
            assistant_text=result.response_payload.get("assistant_text", ""),
            provider_reasoning_text=result.response_payload.get("provider_reasoning_text", ""),
            validated_output=result.validated_output,
        )
    )
    row["evidence_quality"] = evidence_quality(row)
    return row


def _annotate_expansion_row(
    row: dict[str, Any],
    *,
    sample: DatasetSample,
    mode: str,
    expansion_index: int,
    permutation: list[int],
    shuffled_options: list[str],
) -> None:
    row["expansion_mode"] = mode
    row["expansion_index"] = expansion_index
    row["expansion_shuffle_permutation"] = permutation
    row["expansion_shuffled_options"] = shuffled_options
    row["raw_shuffled_answer"] = ""
    if mode == "mc_choice_shuffle":
        raw_answer = str(row.get("normalized_answer") or row.get("prediction") or "")
        mapped = map_shuffled_choice_answer(raw_answer, permutation)
        row["raw_shuffled_answer"] = raw_answer
        if mapped:
            row["prediction"] = mapped
            row["normalized_answer"] = mapped
            if isinstance(row.get("validated_output"), dict):
                row["validated_output"]["answer"] = mapped
                row["validated_output"]["final_answer"] = mapped
                row["validated_output"]["raw_shuffled_answer"] = raw_answer
                row["validated_output"]["shuffle_permutation"] = permutation
    row["expansion_validation_pass"] = _expansion_validation_pass(row, sample=sample, mode=mode)
    row["evidence_quality"] = evidence_quality(row)


def _expansion_validation_pass(row: dict[str, Any], *, sample: DatasetSample, mode: str) -> bool:
    if row.get("request_status") != "ok" or row.get("output_status") != "ok" or row.get("protocol_parse_status") == "failed":
        return False
    answer = str(row.get("normalized_answer") or row.get("prediction") or "").strip()
    if not answer or answer.lower() in {"unknown", "n/a", "none"}:
        return False
    if mode == "mc_choice_shuffle":
        options = _multiple_choice_options(sample)
        return bool(options) and answer in {chr(ord("A") + index) for index in range(len(options))}
    if mode == "strategyqa_dual_polarity":
        return answer.lower() in {"yes", "no"}
    if mode == "hotpot_span_extract":
        return _context_supports_answer_text(sample.prompt_context, answer)
    return True


def _multiple_choice_options(sample: DatasetSample) -> list[str]:
    raw_options = sample.metadata.get("options") or sample.metadata.get("choices") or []
    if not isinstance(raw_options, list):
        return []
    return [str(option).strip() for option in raw_options if str(option).strip()]


def _context_supports_answer_text(context: str, answer: str) -> bool:
    context_norm = _normalize_span_text(context)
    answer_norm = _normalize_span_text(answer)
    return bool(answer_norm and answer_norm in context_norm)


def _normalize_span_text(value: str) -> str:
    import re
    import unicodedata

    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text.lower()).strip()
    return re.sub(r"\s+", " ", text)


def _prediction_row(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    method_name: str,
    backbone_name: str,
    decision,
    stage_decision,
    router,
    stage_rows: list[dict[str, Any]],
    debate_rows: list[dict[str, Any]],
    method_turns: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    expansion_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    prediction = decision.final_answer
    score = score_prediction(dataset, prediction, sample.reference_answer)
    initial_score = score_prediction(dataset, stage_decision.final_answer, sample.reference_answer)
    initial_consensus = len(stage_decision.support) == 1 and bool(stage_decision.support)
    debate_tokens = _sum_field(debate_rows, "total_tokens")
    prompt_tokens = _sum_field(method_turns, "prompt_tokens")
    completion_tokens = _sum_field(method_turns, "completion_tokens")
    total_tokens = _sum_field(method_turns, "total_tokens")
    latency = _sum_field(method_turns, "latency_ms")
    corrected = initial_score < 1.0 and score == 1.0
    harmed = initial_score == 1.0 and score < 1.0
    oracle_candidate_correct = any(
        score_prediction(dataset, str(row.get("normalized_answer") or row.get("prediction") or ""), sample.reference_answer) == 1.0
        for row in stage_rows
    )
    expansion_oracle_correct = any(
        score_prediction(dataset, str(row.get("normalized_answer") or row.get("prediction") or ""), sample.reference_answer) == 1.0
        for row in expansion_rows
        if row.get("request_status") == "ok" and row.get("output_status") == "ok"
    )
    candidate_pool_oracle_correct = oracle_candidate_correct or expansion_oracle_correct
    target_correct = (
        None
        if not targets
        else any(
            score_prediction(dataset, str(row.get("normalized_answer") or row.get("prediction") or ""), sample.reference_answer) == 1.0
            for row in targets
        )
    )
    safe_repair_applied = str(decision.resolver).startswith("cred_verify_safe_deterministic") or str(decision.resolver).startswith(
        "cred_verify_safe_tool"
    )
    hetero_agreement_applied = str(decision.resolver) == "cred_verify_safe_hetero_promoted"
    math_repair_applied = str(decision.resolver) == "cred_acs_math_repair"
    hotpot_span_repair_applied = str(decision.resolver) == "cred_acs_hotpot_span_repair"
    single_pro_blocked = str(decision.resolver) == "cred_acs_single_pro_blocked"
    expansion_mode = str(next((row.get("expansion_mode") for row in expansion_rows if row.get("expansion_mode")), ""))
    expansion_validation_pass_count = sum(1 for row in expansion_rows if row.get("expansion_validation_pass") is True)
    choice_shuffle_agreement_count = _choice_shuffle_agreement_count(dataset, expansion_rows)
    row = asdict(
        CredPredictionRecord(
            run_id=run_id,
            dataset=dataset,
            split=split_name,
            sample_id=sample.sample_id,
            method_name=method_name,
            method_type="mad",
            model_name=backbone_name,
            prediction=prediction,
            gold=sample.reference_answer,
            score=score,
            initial_vote_prediction=stage_decision.final_answer,
            initial_vote_score=initial_score,
            initial_vote_counts={key: int(value) for key, value in stage_decision.support.items()},
            initial_consensus=initial_consensus,
            final_vote_prediction=prediction,
            final_vote_score=score,
            final_vote_counts={prediction: 1} if prediction else {},
            prompt_tokens_per_question=prompt_tokens,
            completion_tokens_per_question=completion_tokens,
            total_tokens_per_question=total_tokens,
            latency_ms_per_question=latency,
            initial_prompt_tokens_per_question=_sum_field(stage_rows, "prompt_tokens"),
            initial_completion_tokens_per_question=_sum_field(stage_rows, "completion_tokens"),
            initial_total_tokens_per_question=_sum_field(stage_rows, "total_tokens"),
            initial_latency_ms_per_question=_sum_field(stage_rows, "latency_ms"),
            debate_prompt_tokens_per_question=_sum_field(debate_rows, "prompt_tokens"),
            debate_completion_tokens_per_question=_sum_field(debate_rows, "completion_tokens"),
            debate_total_tokens_per_question=debate_tokens,
            debate_latency_ms_per_question=_sum_field(debate_rows, "latency_ms"),
            calls_per_question=len(method_turns),
            debate_rounds=1 if debate_rows else 0,
            agent_count=len(stage_rows),
            final_consensus=not decision.changed,
            initial_disagreement=not initial_consensus,
            vote_flipped=decision.changed,
            corrected_by_debate=corrected,
            harmed_by_debate=harmed,
            corrected_by_verification=corrected,
            harmed_by_verification=harmed,
            unchanged_correct=not decision.changed and score == 1.0,
            unchanged_wrong=not decision.changed and score < 1.0,
            triggered=router.triggered,
            router_reasons=list(router.reasons),
            resolver=decision.resolver,
            survival_support={key: round(float(value), 6) for key, value in decision.support.items()},
            verification_support={key: round(float(value), 6) for key, value in decision.support.items()},
            oracle_candidate_correct=oracle_candidate_correct,
            wrong_majority_some_correct=initial_score < 1.0 and oracle_candidate_correct,
            target_correct=target_correct,
            safe_repair_applied=safe_repair_applied,
            hetero_agreement_applied=hetero_agreement_applied,
            expansion_call_count=len(expansion_rows),
            expansion_mode=expansion_mode,
            false_consensus_triggered="false_consensus_probe" in set(router.reasons),
            candidate_pool_oracle_correct=candidate_pool_oracle_correct,
            expansion_oracle_correct=expansion_oracle_correct,
            expansion_validation_pass_count=expansion_validation_pass_count,
            math_repair_applied=math_repair_applied,
            hotpot_span_repair_applied=hotpot_span_repair_applied,
            choice_shuffle_agreement_count=choice_shuffle_agreement_count,
            single_pro_promotion_blocked=single_pro_blocked,
            protocol_failures_per_question=sum(1 for row in method_turns if row.get("protocol_parse_status") == "failed"),
            reason_missing_turns_per_question=sum(1 for row in method_turns if not row.get("reason_present")),
        )
    )
    row["vote_counts"] = row["initial_vote_counts"]
    return row


def _choice_shuffle_agreement_count(dataset: str, expansion_rows: list[dict[str, Any]]) -> int:
    if dataset not in {"gpqa_diamond", "mmlu", "mmlu_abstract_algebra", "mmlu_pro"}:
        return 0
    counts: dict[str, int] = {}
    for row in expansion_rows:
        if row.get("expansion_mode") != "mc_choice_shuffle" or row.get("expansion_validation_pass") is not True:
            continue
        answer = str(row.get("normalized_answer") or row.get("prediction") or "").strip()
        if answer:
            counts[answer] = counts.get(answer, 0) + 1
    return max(counts.values(), default=0)


def build_control_prediction_row(
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
    del method
    prompt_tokens = _sum_field(turn_rows, "prompt_tokens")
    completion_tokens = _sum_field(turn_rows, "completion_tokens")
    total_tokens = _sum_field(turn_rows, "total_tokens")
    latency_ms = _sum_field(turn_rows, "latency_ms")
    return {
        "run_id": run_id,
        "dataset": benchmark_slug,
        "split": split_name,
        "sample_id": sample.sample_id,
        "method_name": control_name,
        "method_type": "control",
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
        "calls_per_question": len(turn_rows),
        "debate_rounds": 0,
        "agent_count": len(turn_rows),
        "final_consensus": final_consensus,
        "initial_disagreement": False,
        "vote_flipped": False,
        "corrected_by_debate": False,
        "harmed_by_debate": False,
        "corrected_by_verification": False,
        "harmed_by_verification": False,
        "unchanged_correct": final_score == 1.0,
        "unchanged_wrong": final_score < 1.0,
        "triggered": False,
        "router_reasons": [],
        "resolver": "no_comm_control",
        "survival_support": {},
        "verification_support": {},
        "oracle_candidate_correct": final_score == 1.0,
        "wrong_majority_some_correct": False,
        "target_correct": None,
        "safe_repair_applied": False,
        "hetero_agreement_applied": False,
        "expansion_call_count": 0,
        "expansion_mode": "",
        "false_consensus_triggered": False,
        "candidate_pool_oracle_correct": final_score == 1.0,
        "expansion_oracle_correct": False,
        "expansion_validation_pass_count": 0,
        "math_repair_applied": False,
        "hotpot_span_repair_applied": False,
        "choice_shuffle_agreement_count": 0,
        "single_pro_promotion_blocked": False,
        "protocol_failures_per_question": sum(1 for row in turn_rows if row.get("protocol_parse_status") == "failed"),
        "reason_missing_turns_per_question": sum(1 for row in turn_rows if not row.get("reason_present")),
        "vote_counts": vote_counts,
    }


def build_metrics(
    prediction_rows: list[dict[str, Any]],
    *,
    dataset_order: list[str],
    method_order: list[str],
    control_names: list[str],
) -> dict[str, Any]:
    dataset_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in prediction_rows:
        grouped.setdefault((row["dataset"], row["model_name"], row["method_name"]), []).append(row)
    for (dataset, model_name, method_name), rows in grouped.items():
        dataset_rows.append(_summarize_prediction_rows(rows, dataset=dataset, model_name=model_name, method_name=method_name, aggregate_kind="dataset"))
    summary = [*dataset_rows, *_build_macro_rows(dataset_rows), *_build_micro_rows(prediction_rows)]
    _attach_comparison_fields(summary, control_names=control_names)
    summary.sort(key=lambda row: _summary_sort_key(row, dataset_order, method_order))
    return {"summary": summary}


def build_debate_diagnostics(
    prediction_rows: list[dict[str, Any]],
    *,
    dataset_order: list[str],
    method_order: list[str],
) -> dict[str, Any]:
    rows = [
        {
            "dataset": row["dataset"],
            "sample_id": row["sample_id"],
            "method_name": row["method_name"],
            "triggered": row.get("triggered"),
            "router_reasons": row.get("router_reasons"),
            "resolver": row.get("resolver"),
            "corrected_by_debate": row.get("corrected_by_debate"),
            "harmed_by_debate": row.get("harmed_by_debate"),
            "corrected_by_verification": row.get("corrected_by_verification"),
            "harmed_by_verification": row.get("harmed_by_verification"),
            "oracle_candidate_correct": row.get("oracle_candidate_correct"),
            "wrong_majority_some_correct": row.get("wrong_majority_some_correct"),
            "target_correct": row.get("target_correct"),
            "safe_repair_applied": row.get("safe_repair_applied"),
            "hetero_agreement_applied": row.get("hetero_agreement_applied"),
            "expansion_call_count": row.get("expansion_call_count"),
            "expansion_mode": row.get("expansion_mode"),
            "candidate_pool_oracle_correct": row.get("candidate_pool_oracle_correct"),
            "expansion_oracle_correct": row.get("expansion_oracle_correct"),
            "math_repair_applied": row.get("math_repair_applied"),
            "hotpot_span_repair_applied": row.get("hotpot_span_repair_applied"),
            "single_pro_promotion_blocked": row.get("single_pro_promotion_blocked"),
            "initial_vote_prediction": row.get("initial_vote_prediction"),
            "prediction": row.get("prediction"),
            "debate_tokens": row.get("debate_total_tokens_per_question"),
        }
        for row in prediction_rows
        if row.get("method_type") == "mad"
    ]
    summary_rows = [
        row for row in build_metrics(prediction_rows, dataset_order=dataset_order, method_order=method_order, control_names=[]).get("summary", [])
        if row.get("method_type") == "mad"
    ]
    return {"sample_rows": rows, "summary_rows": summary_rows}


def build_router_eval(router_rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in router_rows:
        grouped.setdefault(str(row.get("dataset") or ""), []).append(row)
    summary_rows = []
    for dataset, rows in grouped.items():
        count = len(rows)
        summary_rows.append(
            {
                "dataset": dataset,
                "question_count": count,
                "trigger_rate": safe_mean(1.0 if row.get("triggered") else 0.0 for row in rows),
                "avg_risk_count": safe_mean(float(row.get("risk_count") or 0.0) for row in rows),
                "avg_evidence_quality": safe_mean(float(row.get("evidence_quality_mean") or 0.0) for row in rows),
                "refutation_calls_mean": safe_mean(float(row.get("refutation_count") or 0.0) for row in rows),
                "self_verification_calls_mean": safe_mean(float(row.get("self_verification_count") or 0.0) for row in rows),
                "hetero_verification_calls_mean": safe_mean(float(row.get("hetero_verification_count") or 0.0) for row in rows),
                "expansion_calls_mean": safe_mean(float(row.get("expansion_count") or 0.0) for row in rows),
                "false_consensus_trigger_count": sum(1 for row in rows if row.get("trigger_bucket") == "false_consensus_probe"),
                "weak_split_trigger_count": sum(1 for row in rows if row.get("trigger_bucket") == "weak_split"),
                "clean_skip_count": sum(1 for row in rows if row.get("trigger_bucket") == "clean_skip"),
            }
        )
    if router_rows:
        summary_rows.append(
            {
                "dataset": "overall",
                "question_count": len(router_rows),
                "trigger_rate": safe_mean(1.0 if row.get("triggered") else 0.0 for row in router_rows),
                "avg_risk_count": safe_mean(float(row.get("risk_count") or 0.0) for row in router_rows),
                "avg_evidence_quality": safe_mean(float(row.get("evidence_quality_mean") or 0.0) for row in router_rows),
                "refutation_calls_mean": safe_mean(float(row.get("refutation_count") or 0.0) for row in router_rows),
                "self_verification_calls_mean": safe_mean(float(row.get("self_verification_count") or 0.0) for row in router_rows),
                "hetero_verification_calls_mean": safe_mean(float(row.get("hetero_verification_count") or 0.0) for row in router_rows),
                "expansion_calls_mean": safe_mean(float(row.get("expansion_count") or 0.0) for row in router_rows),
                "false_consensus_trigger_count": sum(1 for row in router_rows if row.get("trigger_bucket") == "false_consensus_probe"),
                "weak_split_trigger_count": sum(1 for row in router_rows if row.get("trigger_bucket") == "weak_split"),
                "clean_skip_count": sum(1 for row in router_rows if row.get("trigger_bucket") == "clean_skip"),
            }
        )
    return {"sample_rows": router_rows, "summary_rows": summary_rows}


def estimate_work(experiment: CredVExperimentConfig, phase_name: str, benchmarks, controls, protocol: CredVProtocolConfig) -> tuple[int, int]:
    total_calls = 0
    total_predictions = 0
    for benchmark in benchmarks:
        split_name = resolve_phase_split_name(experiment, phase_name, benchmark.slug)
        sample_count = len(load_split_ids(benchmark.cache_namespace or benchmark.slug, split_name))
        total_calls += sample_count * protocol.stage_a_agent_count
        if CRED_VERIFY_METHODS & set(experiment.cred_methods):
            total_calls += sample_count * protocol.max_verifications
        if CRED_SAFE_VERIFY_METHODS & set(experiment.cred_methods) and experiment.verifier_model_refs:
            total_calls += sample_count * protocol.max_verification_calls
        if CRED_ACS_METHODS & set(experiment.cred_methods) and experiment.verifier_model_refs:
            total_calls += sample_count * protocol.max_expansion_calls
        total_predictions += sample_count * len(experiment.cred_methods)
        for method_name in experiment.control_methods:
            total_calls += sample_count * controls[method_name].budget_calls
            total_predictions += sample_count
    return total_calls, total_predictions


def load_selected_samples(benchmark, split_name: str) -> list[DatasetSample]:
    return select_samples(benchmark, split_name)


def resolve_split_name(experiment: CredVExperimentConfig, phase_name: str, benchmark_slug: str) -> str:
    return resolve_phase_split_name(experiment, phase_name, benchmark_slug)


def _router_row(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    router,
    self_verification_count: int,
    hetero_verification_count: int,
    expansion_count: int,
) -> dict[str, Any]:
    verification_count = int(self_verification_count) + int(hetero_verification_count)
    return {
        "run_id": run_id,
        "dataset": dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "policy_name": "cred_acs_router_v1",
        "triggered": router.triggered,
        "trigger_bucket": router.trigger_bucket,
        "trigger_reasons": list(router.reasons),
        "leading_answer": router.leading_answer,
        "vote_counts": router.vote_counts,
        "risk_count": router.risk_count,
        "evidence_quality_mean": router.evidence_quality_mean,
        "leading_count": router.leading_count,
        "verification_count": verification_count,
        "self_verification_count": self_verification_count,
        "hetero_verification_count": hetero_verification_count,
        "expansion_count": expansion_count,
        "refutation_count": verification_count,
        "defense_count": 0,
        "judge_used": False,
    }


def _debate_message_row(
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    row: dict[str, Any],
    message_kind: str,
) -> dict[str, Any]:
    return asdict(
        DebateMessageRecord(
            run_id=run_id,
            dataset=dataset,
            split=split_name,
            sample_id=sample.sample_id,
            method_name=str(row.get("method_name") or ""),
            round_index=int(row.get("round_index") or 0),
            sender_agent_id=int(row.get("agent_id") or 0),
            recipient_agent_id=0,
            sender_answer=str(row.get("normalized_answer") or ""),
            sender_reasoning=str((row.get("validated_output") or {}).get("reasoning") or ""),
            message_kind=message_kind,
        )
    )


def _sum_field(rows: list[dict[str, Any]], field: str) -> float:
    return round(sum(float(row.get(field) or 0.0) for row in rows), 6)


def _positive_token_cap(value: int | None) -> int | None:
    if value is None or int(value) <= 0:
        return None
    return int(value)


def _summarize_prediction_rows(rows: list[dict[str, Any]], *, dataset: str, model_name: str, method_name: str, aggregate_kind: str) -> dict[str, Any]:
    question_count = len(rows)
    accuracy = safe_mean(float(row["score"]) for row in rows)
    total_tokens = safe_mean(float(row["total_tokens_per_question"]) for row in rows)
    initial_accuracy = safe_mean(_initial_vote_score(row) for row in rows)
    corrected_count = sum(1 for row in rows if row.get("corrected_by_debate"))
    harmed_count = sum(1 for row in rows if row.get("harmed_by_debate"))
    triggered_count = sum(1 for row in rows if row.get("triggered"))
    oracle_accuracy = safe_mean(1.0 if row.get("oracle_candidate_correct") else 0.0 for row in rows)
    candidate_pool_oracle_accuracy = safe_mean(1.0 if row.get("candidate_pool_oracle_correct") else 0.0 for row in rows)
    expansion_oracle_accuracy = safe_mean(1.0 if row.get("expansion_oracle_correct") else 0.0 for row in rows)
    target_rows = [row for row in rows if row.get("wrong_majority_some_correct") and row.get("target_correct") is not None]
    candidate_pool_target_rows = [row for row in rows if _initial_vote_score(row) < 1.0 and row.get("candidate_pool_oracle_correct")]
    promotion_events = corrected_count + harmed_count
    safe_repair_count = sum(1 for row in rows if row.get("safe_repair_applied"))
    hetero_agreement_count = sum(1 for row in rows if row.get("hetero_agreement_applied"))
    math_repair_count = sum(1 for row in rows if row.get("math_repair_applied"))
    hotpot_span_repair_count = sum(1 for row in rows if row.get("hotpot_span_repair_applied"))
    single_pro_blocked_count = sum(1 for row in rows if row.get("single_pro_promotion_blocked"))
    validator_pass_count = sum(int(row.get("expansion_validation_pass_count") or 0) for row in rows)
    choice_shuffle_agreement_count = sum(int(row.get("choice_shuffle_agreement_count") or 0) for row in rows)
    return {
        "dataset": dataset,
        "aggregate_kind": aggregate_kind,
        "model_name": model_name,
        "method_name": method_name,
        "method_type": rows[0]["method_type"],
        "question_count": question_count,
        "prediction_rows": question_count,
        "accuracy_mean": accuracy,
        "initial_vote_accuracy_mean": initial_accuracy,
        "oracle_accuracy_mean": oracle_accuracy,
        "oracle_gap": round(oracle_accuracy - initial_accuracy, 6),
        "candidate_pool_oracle_accuracy": candidate_pool_oracle_accuracy,
        "expansion_oracle_accuracy": expansion_oracle_accuracy,
        "expansion_oracle_gain": round(candidate_pool_oracle_accuracy - oracle_accuracy, 6),
        "target_precision_on_wrong_majority": safe_mean(1.0 if row.get("target_correct") else 0.0 for row in target_rows),
        "promotion_recall_on_wrong_majority": safe_ratio(corrected_count, len(candidate_pool_target_rows)),
        "debate_gain_over_initial_vote": round(accuracy - initial_accuracy, 6),
        "verification_gain_over_initial_vote": round(accuracy - initial_accuracy, 6),
        "prompt_tokens_mean": safe_mean(float(row["prompt_tokens_per_question"]) for row in rows),
        "completion_tokens_mean": safe_mean(float(row["completion_tokens_per_question"]) for row in rows),
        "total_tokens_mean": total_tokens,
        "communication_tokens_mean": safe_mean(float(row["debate_total_tokens_per_question"]) for row in rows),
        "latency_ms_mean": safe_mean(float(row["latency_ms_per_question"]) for row in rows),
        "calls_per_question_mean": safe_mean(float(row["calls_per_question"]) for row in rows),
        "protocol_failures_per_question_mean": safe_mean(float(row.get("protocol_failures_per_question") or 0.0) for row in rows),
        "reason_missing_turns_per_question_mean": safe_mean(float(row.get("reason_missing_turns_per_question") or 0.0) for row in rows),
        "accuracy_per_1k_tokens": round((accuracy / total_tokens * 1000) if total_tokens else 0.0, 6),
        "debate_rounds": safe_mean(float(row["debate_rounds"]) for row in rows),
        "agent_count": safe_mean(float(row["agent_count"]) for row in rows),
        "trigger_rate": safe_ratio(triggered_count, question_count),
        "expansion_trigger_rate": safe_mean(1.0 if int(row.get("expansion_call_count") or 0) > 0 else 0.0 for row in rows),
        "false_consensus_trigger_count": sum(1 for row in rows if row.get("false_consensus_triggered")),
        "corrected_count": corrected_count,
        "harmed_count": harmed_count,
        "verified_corrected_count": corrected_count,
        "verified_harmed_count": harmed_count,
        "promotion_precision": safe_ratio(corrected_count, promotion_events),
        "harm_per_correction": round((harmed_count / corrected_count) if corrected_count else float(harmed_count), 6),
        "safe_repair_count": safe_repair_count,
        "hetero_agreement_count": hetero_agreement_count,
        "math_repair_count": math_repair_count,
        "hotpot_span_repair_count": hotpot_span_repair_count,
        "validator_pass_count": validator_pass_count,
        "choice_shuffle_agreement_count": choice_shuffle_agreement_count,
        "single_pro_promotion_blocked_count": single_pro_blocked_count,
        "corrected_rate": safe_ratio(corrected_count, question_count),
        "harmed_rate": safe_ratio(harmed_count, question_count),
        "flip_rate": safe_mean(1.0 if row.get("vote_flipped") else 0.0 for row in rows),
        "initial_consensus_rate": safe_mean(1.0 if row.get("initial_consensus") else 0.0 for row in rows),
        "final_consensus_rate": safe_mean(1.0 if row.get("final_consensus") else 0.0 for row in rows),
    }


def _initial_vote_score(row: dict[str, Any]) -> float:
    value = row.get("initial_vote_score")
    if value is None:
        value = row["score"]
    return float(value)


def _build_macro_rows(dataset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in dataset_rows:
        grouped.setdefault((row["model_name"], row["method_name"]), []).append(row)
    rows = []
    for (model_name, method_name), items in grouped.items():
        rows.append(_macro_from_rows(items, dataset="overall", model_name=model_name, method_name=method_name, aggregate_kind="macro"))
    return rows


def _build_micro_rows(prediction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in prediction_rows:
        grouped.setdefault((row["model_name"], row["method_name"]), []).append(row)
    return [
        _summarize_prediction_rows(items, dataset="overall_micro", model_name=model_name, method_name=method_name, aggregate_kind="micro")
        for (model_name, method_name), items in grouped.items()
    ]


def _macro_from_rows(rows: list[dict[str, Any]], *, dataset: str, model_name: str, method_name: str, aggregate_kind: str) -> dict[str, Any]:
    accuracy = safe_mean(float(row["accuracy_mean"]) for row in rows)
    total_tokens = safe_mean(float(row["total_tokens_mean"]) for row in rows)
    initial_accuracy = safe_mean(float(row["initial_vote_accuracy_mean"]) for row in rows)
    oracle_accuracy = safe_mean(float(row.get("oracle_accuracy_mean") or 0.0) for row in rows)
    candidate_pool_oracle_accuracy = safe_mean(float(row.get("candidate_pool_oracle_accuracy") or 0.0) for row in rows)
    expansion_oracle_accuracy = safe_mean(float(row.get("expansion_oracle_accuracy") or 0.0) for row in rows)
    corrected_count = sum(int(row["corrected_count"]) for row in rows)
    harmed_count = sum(int(row["harmed_count"]) for row in rows)
    return {
        "dataset": dataset,
        "aggregate_kind": aggregate_kind,
        "model_name": model_name,
        "method_name": method_name,
        "method_type": rows[0]["method_type"],
        "question_count": sum(int(row["question_count"]) for row in rows),
        "prediction_rows": sum(int(row["prediction_rows"]) for row in rows),
        "accuracy_mean": accuracy,
        "initial_vote_accuracy_mean": initial_accuracy,
        "oracle_accuracy_mean": oracle_accuracy,
        "oracle_gap": round(oracle_accuracy - initial_accuracy, 6),
        "candidate_pool_oracle_accuracy": candidate_pool_oracle_accuracy,
        "expansion_oracle_accuracy": expansion_oracle_accuracy,
        "expansion_oracle_gain": round(candidate_pool_oracle_accuracy - oracle_accuracy, 6),
        "target_precision_on_wrong_majority": safe_mean(float(row.get("target_precision_on_wrong_majority") or 0.0) for row in rows),
        "promotion_recall_on_wrong_majority": safe_mean(float(row.get("promotion_recall_on_wrong_majority") or 0.0) for row in rows),
        "debate_gain_over_initial_vote": round(accuracy - initial_accuracy, 6),
        "verification_gain_over_initial_vote": round(accuracy - initial_accuracy, 6),
        "prompt_tokens_mean": safe_mean(float(row["prompt_tokens_mean"]) for row in rows),
        "completion_tokens_mean": safe_mean(float(row["completion_tokens_mean"]) for row in rows),
        "total_tokens_mean": total_tokens,
        "communication_tokens_mean": safe_mean(float(row["communication_tokens_mean"]) for row in rows),
        "latency_ms_mean": safe_mean(float(row["latency_ms_mean"]) for row in rows),
        "calls_per_question_mean": safe_mean(float(row["calls_per_question_mean"]) for row in rows),
        "protocol_failures_per_question_mean": safe_mean(float(row["protocol_failures_per_question_mean"]) for row in rows),
        "reason_missing_turns_per_question_mean": safe_mean(float(row["reason_missing_turns_per_question_mean"]) for row in rows),
        "accuracy_per_1k_tokens": round((accuracy / total_tokens * 1000) if total_tokens else 0.0, 6),
        "debate_rounds": safe_mean(float(row["debate_rounds"]) for row in rows),
        "agent_count": safe_mean(float(row["agent_count"]) for row in rows),
        "trigger_rate": safe_mean(float(row.get("trigger_rate") or 0.0) for row in rows),
        "expansion_trigger_rate": safe_mean(float(row.get("expansion_trigger_rate") or 0.0) for row in rows),
        "false_consensus_trigger_count": sum(int(row.get("false_consensus_trigger_count") or 0) for row in rows),
        "corrected_count": corrected_count,
        "harmed_count": harmed_count,
        "verified_corrected_count": corrected_count,
        "verified_harmed_count": harmed_count,
        "promotion_precision": safe_ratio(corrected_count, corrected_count + harmed_count),
        "harm_per_correction": round((harmed_count / corrected_count) if corrected_count else float(harmed_count), 6),
        "safe_repair_count": sum(int(row.get("safe_repair_count") or 0) for row in rows),
        "hetero_agreement_count": sum(int(row.get("hetero_agreement_count") or 0) for row in rows),
        "math_repair_count": sum(int(row.get("math_repair_count") or 0) for row in rows),
        "hotpot_span_repair_count": sum(int(row.get("hotpot_span_repair_count") or 0) for row in rows),
        "validator_pass_count": sum(int(row.get("validator_pass_count") or 0) for row in rows),
        "choice_shuffle_agreement_count": sum(int(row.get("choice_shuffle_agreement_count") or 0) for row in rows),
        "single_pro_promotion_blocked_count": sum(int(row.get("single_pro_promotion_blocked_count") or 0) for row in rows),
        "corrected_rate": safe_mean(float(row["corrected_rate"]) for row in rows),
        "harmed_rate": safe_mean(float(row["harmed_rate"]) for row in rows),
        "flip_rate": safe_mean(float(row["flip_rate"]) for row in rows),
        "initial_consensus_rate": safe_mean(float(row["initial_consensus_rate"]) for row in rows),
        "final_consensus_rate": safe_mean(float(row["final_consensus_rate"]) for row in rows),
    }


def _attach_comparison_fields(summary_rows: list[dict[str, Any]], *, control_names: list[str]) -> None:
    control_set = set(control_names)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in summary_rows:
        grouped.setdefault((str(row["dataset"]), str(row["model_name"])), []).append(row)
    for rows in grouped.values():
        cot_row = next((row for row in rows if row["method_name"] == "cot_1"), None)
        controls = [row for row in rows if row["method_name"] in control_set]
        best_no_comm = max(controls, key=lambda row: float(row["accuracy_mean"])) if controls else None
        for row in rows:
            row["accuracy_delta_vs_cot_1"] = _delta_against(row, cot_row, "accuracy_mean")
            row["token_ratio_vs_cot_1"] = _ratio_against(row, cot_row, "total_tokens_mean")
            row["calls_ratio_vs_cot_1"] = _ratio_against(row, cot_row, "calls_per_question_mean")
            row["best_no_comm_method"] = None if best_no_comm is None else best_no_comm["method_name"]
            row["best_no_comm_accuracy"] = None if best_no_comm is None else best_no_comm["accuracy_mean"]
            row["accuracy_delta_vs_best_no_comm"] = _delta_against(row, best_no_comm, "accuracy_mean")
            row["token_ratio_vs_best_no_comm"] = _ratio_against(row, best_no_comm, "total_tokens_mean")
            row["calls_ratio_vs_best_no_comm"] = _ratio_against(row, best_no_comm, "calls_per_question_mean")


def _delta_against(row: dict[str, Any], reference: dict[str, Any] | None, key: str) -> float | None:
    if reference is None:
        return None
    return round(float(row[key]) - float(reference[key]), 6)


def _ratio_against(row: dict[str, Any], reference: dict[str, Any] | None, key: str) -> float | None:
    if reference is None or not float(reference[key]):
        return None
    return round(float(row[key]) / float(reference[key]), 6)


def _summary_sort_key(row: dict[str, Any], dataset_order: list[str], method_order: list[str]) -> tuple[int, int, str]:
    dataset_rank = {name: index for index, name in enumerate(dataset_order)}
    method_rank = {name: index for index, name in enumerate(method_order)}
    special_rank = {"overall": len(dataset_order), "overall_micro": len(dataset_order) + 1}
    return (
        special_rank.get(row["dataset"], dataset_rank.get(row["dataset"], len(dataset_order) + 2)),
        method_rank.get(row["method_name"], 999),
        str(row["model_name"]),
    )
