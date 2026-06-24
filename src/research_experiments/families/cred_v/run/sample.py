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
    aggregate_stage_a_vote,
    aggregate_survival,
    build_router_decision,
    evidence_quality,
    select_refutation_targets,
)
from research_experiments.families.cred_v.config import (
    CRED_DEBATE_METHODS,
    CredMadExperimentConfig,
    CredMadProtocolConfig,
)
from research_experiments.families.cred_v.prompts import (
    AGENT_ROLES,
    build_defense_messages,
    build_judge_messages,
    build_refutation_messages,
    build_stage_a_messages,
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
    unchanged_correct: bool
    unchanged_wrong: bool
    triggered: bool
    router_reasons: list[str]
    resolver: str
    survival_support: dict[str, float]
    protocol_failures_per_question: int
    reason_missing_turns_per_question: int


def run_cred_batch(
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    samples: list[DatasetSample],
    experiment: CredMadExperimentConfig,
    protocol: CredMadProtocolConfig,
    backbone,
    provider,
    cache,
    throttle,
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
    experiment: CredMadExperimentConfig,
    protocol: CredMadProtocolConfig,
    backbone,
    provider,
    cache,
    throttle,
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
    refutation_rows: list[dict[str, Any]] = []
    defense_rows: list[dict[str, Any]] = []
    judge_row: dict[str, Any] | None = None
    debate_rows: list[dict[str, Any]] = []

    if router.triggered and CRED_DEBATE_METHODS & set(experiment.cred_methods):
        targets = select_refutation_targets(
            dataset=benchmark_slug,
            rows=stage_rows,
            leading_answer=vote_decision.final_answer,
            max_refutations=protocol.max_refutations,
        )
        for refutation_index, target in enumerate(targets, start=1):
            refutation_row = _execute_turn(
                run_id=run_id,
                dataset=benchmark_slug,
                split_name=split_name,
                sample=sample,
                method_name="cred_refutation",
                method_type="mad",
                round_index=1,
                agent_id=100 + refutation_index,
                role="refutation",
                agent_role="refuter",
                visible_peer_count=len(stage_rows),
                messages=build_refutation_messages(
                    sample,
                    leading_answer=vote_decision.final_answer,
                    target_row=target,
                    stage_rows=stage_rows,
                ),
                backbone=backbone,
                provider=provider,
                cache=cache,
                throttle=throttle,
                temperature=protocol.debate_temperature,
                top_p=protocol.top_p,
                seed=experiment.global_seed + 100 + refutation_index,
                output_protocol=experiment.cred_debate_output_protocol,
                max_tokens=_positive_token_cap(protocol.refutation_max_tokens),
            )
            refutation_rows.append(refutation_row)
            debate_rows.append(_debate_message_row(run_id, benchmark_slug, split_name, sample, refutation_row, "refutation"))
            defense_row = _execute_turn(
                run_id=run_id,
                dataset=benchmark_slug,
                split_name=split_name,
                sample=sample,
                method_name="cred_defense",
                method_type="mad",
                round_index=1,
                agent_id=200 + refutation_index,
                role="defense",
                agent_role="defender",
                visible_peer_count=len(stage_rows) + 1,
                messages=build_defense_messages(
                    sample,
                    leading_answer=vote_decision.final_answer,
                    refutation_row=refutation_row,
                    stage_rows=stage_rows,
                ),
                backbone=backbone,
                provider=provider,
                cache=cache,
                throttle=throttle,
                temperature=protocol.debate_temperature,
                top_p=protocol.top_p,
                seed=experiment.global_seed + 200 + refutation_index,
                output_protocol=experiment.cred_debate_output_protocol,
                max_tokens=_positive_token_cap(protocol.defense_max_tokens),
            )
            defense_rows.append(defense_row)
            debate_rows.append(_debate_message_row(run_id, benchmark_slug, split_name, sample, defense_row, "defense"))
        if refutation_rows or defense_rows:
            judge_row = _execute_turn(
                run_id=run_id,
                dataset=benchmark_slug,
                split_name=split_name,
                sample=sample,
                method_name="cred_judge",
                method_type="mad",
                round_index=2,
                agent_id=300,
                role="judge",
                agent_role="judge",
                visible_peer_count=len(stage_rows) + len(refutation_rows) + len(defense_rows),
                messages=build_judge_messages(
                    sample,
                    leading_answer=vote_decision.final_answer,
                    stage_rows=stage_rows,
                    refutation_rows=refutation_rows,
                    defense_rows=defense_rows,
                ),
                backbone=backbone,
                provider=provider,
                cache=cache,
                throttle=throttle,
                temperature=protocol.judge_temperature,
                top_p=protocol.top_p,
                seed=experiment.global_seed + 300,
                output_protocol=experiment.cred_debate_output_protocol,
                max_tokens=_positive_token_cap(protocol.judge_max_tokens),
            )
            debate_rows.append(_debate_message_row(run_id, benchmark_slug, split_name, sample, judge_row, "judge"))

    all_turns = [*stage_rows, *refutation_rows, *defense_rows, *([judge_row] if judge_row is not None else [])]
    router_row = _router_row(
        run_id=run_id,
        dataset=benchmark_slug,
        split_name=split_name,
        sample=sample,
        router=router,
        refutation_count=len(refutation_rows),
        defense_count=len(defense_rows),
        judge_used=judge_row is not None,
    )
    prediction_rows: list[dict[str, Any]] = []
    for method_name in experiment.cred_methods:
        decision = vote_decision
        method_turns = list(stage_rows)
        debate_turns: list[dict[str, Any]] = []
        if method_name in CRED_DEBATE_METHODS:
            decision = aggregate_survival(
                dataset=benchmark_slug,
                stage_rows=stage_rows,
                refutation_rows=refutation_rows,
                defense_rows=defense_rows,
                judge_row=judge_row,
                stage_winner=vote_decision.final_answer,
                survival_override_margin=protocol.locked_override_margin,
                concrete_evidence_min_chars=protocol.concrete_evidence_min_chars,
                locked=True,
            )
            debate_turns = [*refutation_rows, *defense_rows, *([judge_row] if judge_row is not None else [])]
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
            )
        )
    return all_turns, debate_rows, [router_row], prediction_rows


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
            unchanged_correct=not decision.changed and score == 1.0,
            unchanged_wrong=not decision.changed and score < 1.0,
            triggered=router.triggered,
            router_reasons=list(router.reasons),
            resolver=decision.resolver,
            survival_support={key: round(float(value), 6) for key, value in decision.support.items()},
            protocol_failures_per_question=sum(1 for row in method_turns if row.get("protocol_parse_status") == "failed"),
            reason_missing_turns_per_question=sum(1 for row in method_turns if not row.get("reason_present")),
        )
    )
    row["vote_counts"] = row["initial_vote_counts"]
    return row


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
        "unchanged_correct": final_score == 1.0,
        "unchanged_wrong": final_score < 1.0,
        "triggered": False,
        "router_reasons": [],
        "resolver": "no_comm_control",
        "survival_support": {},
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
            }
        )
    return {"sample_rows": router_rows, "summary_rows": summary_rows}


def estimate_work(experiment: CredMadExperimentConfig, phase_name: str, benchmarks, controls, protocol: CredMadProtocolConfig) -> tuple[int, int]:
    total_calls = 0
    total_predictions = 0
    for benchmark in benchmarks:
        split_name = resolve_phase_split_name(experiment, phase_name, benchmark.slug)
        sample_count = len(load_split_ids(benchmark.cache_namespace or benchmark.slug, split_name))
        total_calls += sample_count * protocol.stage_a_agent_count
        if CRED_DEBATE_METHODS & set(experiment.cred_methods):
            total_calls += sample_count * (protocol.max_refutations * 2 + 1)
        total_predictions += sample_count * len(experiment.cred_methods)
        for method_name in experiment.control_methods:
            total_calls += sample_count * controls[method_name].budget_calls
            total_predictions += sample_count
    return total_calls, total_predictions


def load_selected_samples(benchmark, split_name: str) -> list[DatasetSample]:
    return select_samples(benchmark, split_name)


def resolve_split_name(experiment: CredMadExperimentConfig, phase_name: str, benchmark_slug: str) -> str:
    return resolve_phase_split_name(experiment, phase_name, benchmark_slug)


def _router_row(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    router,
    refutation_count: int,
    defense_count: int,
    judge_used: bool,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "dataset": dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "policy_name": "cred_v_router_v1",
        "triggered": router.triggered,
        "trigger_reasons": list(router.reasons),
        "leading_answer": router.leading_answer,
        "vote_counts": router.vote_counts,
        "risk_count": router.risk_count,
        "evidence_quality_mean": router.evidence_quality_mean,
        "refutation_count": refutation_count,
        "defense_count": defense_count,
        "judge_used": judge_used,
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
        "debate_gain_over_initial_vote": round(accuracy - initial_accuracy, 6),
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
        "corrected_count": corrected_count,
        "harmed_count": harmed_count,
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
    return {
        "dataset": dataset,
        "aggregate_kind": aggregate_kind,
        "model_name": model_name,
        "method_name": method_name,
        "method_type": rows[0]["method_type"],
        "question_count": sum(int(row["question_count"]) for row in rows),
        "prediction_rows": sum(int(row["prediction_rows"]) for row in rows),
        "accuracy_mean": accuracy,
        "initial_vote_accuracy_mean": safe_mean(float(row["initial_vote_accuracy_mean"]) for row in rows),
        "debate_gain_over_initial_vote": round(accuracy - safe_mean(float(row["initial_vote_accuracy_mean"]) for row in rows), 6),
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
        "corrected_count": sum(int(row["corrected_count"]) for row in rows),
        "harmed_count": sum(int(row["harmed_count"]) for row in rows),
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
