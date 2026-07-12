"""RCTA-MAD 独立样本执行链路。"""

from __future__ import annotations

import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

from research_experiments.core.controls.control_prompts import build_cot_messages
from research_experiments.core.data.datasets import DatasetSample, select_samples
from research_experiments.core.data.evaluation import normalize_prediction, score_prediction
from research_experiments.core.execution.runner_common import execute_cached_request, iter_indexed_batch
from research_experiments.families.risk_controlled_trace_mad.algorithms import (
    build_feature_vector,
    build_trace_board,
    majority_with_anchor_fallback,
    stage_decision,
)
from research_experiments.families.risk_controlled_trace_mad.certificates import (
    CERTIFICATE_TYPES,
    verify_certificate,
)
from research_experiments.families.risk_controlled_trace_mad.config import (
    RctaExperimentConfig,
    RctaProtocolConfig,
)
from research_experiments.families.risk_controlled_trace_mad.prompts import (
    RCTA_SCHEMA_VERSION,
    build_debate_update_messages,
    build_synthesis_messages,
)
from research_experiments.families.risk_controlled_trace_mad.router import RiskRouter
from research_experiments.family_runtime.common import resolve_phase_split_name
from research_experiments.family_runtime.free_text_protocol import FREE_TEXT_ANSWER_PROTOCOL_V1
from research_experiments.family_runtime.output_protocols import execute_output_protocol_turn


def resolve_split_name(experiment: RctaExperimentConfig, phase_name: str, dataset: str) -> str:
    return resolve_phase_split_name(experiment, phase_name, dataset)


def load_selected_samples(benchmark, split_name: str) -> list[DatasetSample]:
    return select_samples(benchmark, split_name)


def run_batch(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    samples: list[DatasetSample],
    experiment: RctaExperimentConfig,
    protocol: RctaProtocolConfig,
    active_methods: list[str],
    backbone,
    provider,
    cache,
    throttle,
    router: RiskRouter | None,
) -> Iterator[tuple[int, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]]:
    worker = partial(
        run_sample,
        run_id=run_id,
        dataset=dataset,
        split_name=split_name,
        experiment=experiment,
        protocol=protocol,
        active_methods=active_methods,
        backbone=backbone,
        provider=provider,
        cache=cache,
        throttle=throttle,
        router=router,
    )
    for index, result in iter_indexed_batch(samples, worker=worker, max_concurrent_requests=experiment.max_concurrent_requests):
        yield index, *result


def run_sample(
    sample: DatasetSample,
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    experiment: RctaExperimentConfig,
    protocol: RctaProtocolConfig,
    active_methods: list[str],
    backbone,
    provider,
    cache,
    throttle,
    router: RiskRouter | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    stage_rows = _run_stage_pool(
        run_id=run_id,
        dataset=dataset,
        split_name=split_name,
        sample=sample,
        experiment=experiment,
        protocol=protocol,
        backbone=backbone,
        provider=provider,
        cache=cache,
        throttle=throttle,
    )
    first_five = stage_rows[:5]
    stage = stage_decision(first_five)
    anchor_score = score_prediction(dataset, stage.anchor_answer, sample.reference_answer)
    all_turns = list(stage_rows)
    debate_rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []

    for control_name in experiment.control_methods:
        count = int(control_name.rsplit("_", 1)[1]) if control_name.startswith("sc_") else 1
        predictions.append(_prediction_from_votes(
            run_id=run_id, dataset=dataset, split_name=split_name, sample=sample, backbone_name=backbone.name,
            method_name=control_name, method_type="control", rows=stage_rows[:count], anchor=stage.anchor_answer,
        ))

    if "adaptive_sc_9" in active_methods:
        adaptive_rows = stage_rows if stage.triggered else first_five
        predictions.append(_prediction_from_votes(
            run_id=run_id, dataset=dataset, split_name=split_name, sample=sample, backbone_name=backbone.name,
            method_name="adaptive_sc_9", method_type="rcta", rows=adaptive_rows, anchor=stage.anchor_answer,
            initial_rows=first_five, additional_rows=stage_rows[5:] if stage.triggered else [],
        ))

    synthesis_turn: dict[str, Any] | None = None
    synthesis: dict[str, Any] = {}
    certificate = {"status": "unsupported", "certificate_type": "unsupported", "detail": "not_triggered"}
    feature_vector: dict[str, float] | None = None
    board = ""
    board_counts: dict[str, int] = {}
    synthesis_methods = {"gsa_trace_1", "rcta_certificate_shadow_1", "rcta_1", "rcta_no_certificate", "rcta_existing_only"}
    if stage.triggered and synthesis_methods.intersection(active_methods):
        board, board_counts = build_trace_board(
            first_five, seed=experiment.global_seed, sample_id=sample.sample_id,
            trace_max_chars=protocol.trace_max_chars, board_max_chars=protocol.board_max_chars,
        )
        synthesis_turn = _execute_synthesis_turn(
            run_id=run_id, dataset=dataset, split_name=split_name, sample=sample, board=board,
            backbone=backbone, provider=provider, cache=cache, throttle=throttle,
            temperature=protocol.synthesis_temperature, top_p=protocol.top_p,
            seed=experiment.global_seed + 20_000, max_tokens=protocol.synthesis_max_tokens,
            reasoning_word_limit=protocol.reasoning_word_limit,
        )
        all_turns.append(synthesis_turn)
        synthesis = dict(synthesis_turn.get("validated_output") or {})
        synthesis_answer = str(synthesis.get("final_answer") or "")
        if synthesis_answer:
            synthesis["final_answer"] = normalize_prediction(dataset, synthesis_answer)
        certificate = verify_certificate(
            question=sample.question,
            final_answer=str(synthesis.get("final_answer") or ""),
            certificate_type=str(synthesis.get("certificate_type") or "unsupported"),
            payload=dict(synthesis.get("certificate_payload") or {}),
        )
        feature_vector = build_feature_vector(first_five, synthesis, certificate)
        debate_rows.append({
            "run_id": run_id, "dataset": dataset, "split": split_name, "sample_id": sample.sample_id,
            "method_name": "rcta_trace_synthesizer", "round_index": 1, "sender_agent_id": 1,
            "recipient_agent_id": 0, "message_kind": "trace_synthesis", "sender_answer": synthesis.get("final_answer", ""),
            "sender_reasoning": synthesis.get("reasoning_summary", ""),
        })

    if "gsa_trace_1" in active_methods:
        predictions.append(_rcta_prediction(
            run_id=run_id, dataset=dataset, split_name=split_name, sample=sample, backbone_name=backbone.name,
            method_name="gsa_trace_1", stage_rows=first_five, stage=stage, anchor_score=anchor_score,
            synthesis_turn=synthesis_turn, synthesis=synthesis, certificate=certificate, feature_vector=feature_vector,
            accept=bool(synthesis.get("final_answer")), resolver="gsa_trace_always_accept", board=board, board_counts=board_counts,
        ))

    for method_name in [name for name in active_methods if name in {"rcta_certificate_shadow_1", "rcta_1", "rcta_no_certificate", "rcta_existing_only"}]:
        router_result: dict[str, Any] = {"accept": False, "risk_score": None, "p_gain": None, "p_harm": None}
        if router is not None and feature_vector is not None:
            router_result = router.score(feature_vector, without_certificate=method_name == "rcta_no_certificate")
        accept = bool(router_result["accept"])
        if method_name == "rcta_certificate_shadow_1":
            accept = False
        if method_name == "rcta_existing_only" and str(synthesis.get("final_answer") or "") not in stage.vote_counts:
            accept = False
        predictions.append(_rcta_prediction(
            run_id=run_id, dataset=dataset, split_name=split_name, sample=sample, backbone_name=backbone.name,
            method_name=method_name, stage_rows=first_five, stage=stage, anchor_score=anchor_score,
            synthesis_turn=synthesis_turn, synthesis=synthesis, certificate=certificate, feature_vector=feature_vector,
            accept=accept, resolver=("rcta_risk_accept" if accept else "rcta_anchor_fallback"), board=board,
            board_counts=board_counts, router_result=router_result,
        ))

    for method_name, confidence_mode in (("mad_5a_r1", False), ("confidence_mad_5a_r1", True)):
        if method_name not in active_methods:
            continue
        updates = _run_debate_round(
            run_id=run_id, dataset=dataset, split_name=split_name, sample=sample, stage_rows=first_five,
            method_name=method_name, confidence_mode=confidence_mode, experiment=experiment, protocol=protocol,
            backbone=backbone, provider=provider, cache=cache, throttle=throttle,
        ) if stage.triggered else []
        all_turns.extend(updates)
        method_rows = updates or first_five
        prediction = _prediction_from_votes(
            run_id=run_id, dataset=dataset, split_name=split_name, sample=sample, backbone_name=backbone.name,
            method_name=method_name, method_type="mad", rows=method_rows, anchor=stage.anchor_answer,
            initial_rows=first_five, additional_rows=updates,
        )
        predictions.append(prediction)
        debate_rows.extend(_debate_message_rows(run_id, dataset, split_name, sample, method_name, updates))

    router_row = {
        "run_id": run_id, "dataset": dataset, "split": split_name, "sample_id": sample.sample_id,
        "policy_name": "rcta_global_risk_v1", "triggered": stage.triggered, "anchor_answer": stage.anchor_answer,
        "anchor_score": anchor_score, "vote_counts": stage.vote_counts, "disagreement_pattern": stage.disagreement_pattern,
        "feature_vector": feature_vector, "certificate": certificate,
    }
    return all_turns, debate_rows, [router_row], predictions


def _run_stage_pool(*, run_id, dataset, split_name, sample, experiment, protocol, backbone, provider, cache, throttle) -> list[dict[str, Any]]:
    requests = []
    for agent_id in range(1, protocol.sc_ceiling_candidates + 1):
        requests.append({
            "run_id": run_id, "dataset": dataset, "split_name": split_name, "sample": sample,
            "method_name": "rcta_stage_a_shared", "method_type": "rcta_stage_a", "round_index": 0,
            "agent_id": agent_id, "role": "initial", "messages": build_cot_messages(sample, agent_id, experiment.control_prompt_version),
            "backbone": backbone, "provider": provider, "cache": cache, "throttle": throttle,
            "temperature": protocol.stage_a_temperature, "top_p": protocol.top_p,
            "seed": experiment.global_seed + agent_id - 1, "max_tokens": None,
        })
    with ThreadPoolExecutor(max_workers=protocol.sc_ceiling_candidates) as executor:
        return list(executor.map(lambda values: _execute_free_text_turn(**values), requests))


def _execute_free_text_turn(
    *, run_id: str, dataset: str, split_name: str, sample: DatasetSample, method_name: str,
    method_type: str, round_index: int, agent_id: int, role: str, messages: list[dict[str, str]],
    backbone, provider, cache, throttle, temperature: float, top_p: float, seed: int,
    max_tokens: int | None,
) -> dict[str, Any]:
    result = execute_output_protocol_turn(
        backbone=backbone, provider=provider, cache=cache, throttle=throttle, sample=sample,
        messages=messages, temperature=temperature, top_p=top_p, seed=seed, dataset=dataset,
        role=role, output_protocol=FREE_TEXT_ANSWER_PROTOCOL_V1, max_tokens=max_tokens,
    )
    if result.output_status != "ok":
        cache.delete(result.cache_key)
    answer = str(result.validated_output.get("final_answer") or "")
    normalized = normalize_prediction(dataset, answer) if answer else ""
    response = result.response_payload
    return {
        "run_id": run_id, "dataset": dataset, "split": split_name, "sample_id": sample.sample_id,
        "method_name": method_name, "method_type": method_type, "round_index": round_index,
        "agent_id": agent_id, "role": role, "prompt_hash": result.prompt_hash, "cache_key": result.cache_key,
        "prediction": normalized, "normalized_answer": normalized, "score": None,
        "output_status": result.output_status, "prompt_tokens": float(result.usage.get("prompt_tokens") or 0),
        "completion_tokens": float(result.usage.get("completion_tokens") or 0),
        "total_tokens": float(result.usage.get("total_tokens") or 0),
        "latency_ms": float(response.get("latency_ms") or 0), "cache_hit": result.cache_hit,
        "request_error": result.request_error, "request_status": result.request_status,
        "raw_finish_reason": result.raw_finish_reason, "output_protocol": result.output_protocol,
        "protocol_parse_status": result.protocol_parse_status, "protocol_parse_error": result.protocol_parse_error,
        "reason_present": result.reason_present, "request_count": result.request_count,
        "cache_request_count": result.cache_request_count, "network_request_count": result.network_request_count,
        "logical_call_count": 1, "network_attempt_count": int(response.get("network_attempt_count") or (0 if result.cache_hit else 1)),
        "request_started_at": response.get("request_started_at"), "payload": result.payload,
        "request_started_at_events": list(response.get("request_started_at_events") or []),
        "assistant_text": str(response.get("assistant_text") or ""),
        "provider_reasoning_text": str(response.get("provider_reasoning_text") or ""),
        "validated_output": result.validated_output,
    }


def parse_synthesis_output(raw_text: str, *, reasoning_word_limit: int = 120) -> dict[str, Any]:
    text = str(raw_text or "").strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("RCTA synthesis output must be a JSON object")
    required = {"reasoning_summary", "final_answer", "source_trace_ids", "decisive_claim", "certificate_type", "certificate_payload"}
    if not required.issubset(payload):
        raise ValueError("Missing RCTA synthesis field(s): " + ", ".join(sorted(required - set(payload))))
    for key in ("reasoning_summary", "final_answer", "decisive_claim", "certificate_type"):
        if key == "final_answer" and isinstance(payload[key], (int, float, bool)):
            payload[key] = str(payload[key])
        if not isinstance(payload[key], str) or not payload[key].strip():
            raise ValueError(f"RCTA synthesis field {key} must be non-empty text")
    reasoning_words = payload["reasoning_summary"].split()
    if len(reasoning_words) > reasoning_word_limit:
        payload["reasoning_summary"] = " ".join(reasoning_words[:reasoning_word_limit])
    trace_ids = payload["source_trace_ids"]
    if not isinstance(trace_ids, list) or not all(isinstance(item, str) and item in {f"T{i}" for i in range(1, 6)} for item in trace_ids):
        raise ValueError("source_trace_ids must contain only T1..T5")
    if payload["certificate_type"] not in CERTIFICATE_TYPES or not isinstance(payload["certificate_payload"], dict):
        raise ValueError("invalid certificate contract")
    return {key: payload[key] for key in required}


def _execute_synthesis_turn(
    *, run_id, dataset, split_name, sample, board, backbone, provider, cache, throttle,
    temperature, top_p, seed, max_tokens, reasoning_word_limit,
) -> dict[str, Any]:
    messages = build_synthesis_messages(sample, board)
    validated: dict[str, Any] = {}
    parse_error = None
    request = None
    response: dict[str, Any] = {}
    aggregate_usage = {"prompt_tokens": 0.0, "completion_tokens": 0.0, "total_tokens": 0.0}
    aggregate_latency = 0.0
    aggregate_network_attempts = 0
    aggregate_cache_requests = 0
    aggregate_request_events: list[str] = []
    protocol_normalization_flags: list[str] = []
    protocol_attempts = 0
    for protocol_attempt in range(1, 3):
        protocol_attempts = protocol_attempt
        request = execute_cached_request(
            backbone=backbone, provider=provider, cache=cache, throttle=throttle, messages=messages,
            temperature=temperature, top_p=top_p, seed=seed, use_response_format=True, max_tokens=max_tokens,
        )
        response = dict(request.response_payload)
        for key in aggregate_usage:
            aggregate_usage[key] += float(request.usage.get(key) or 0.0)
        aggregate_latency += float(response.get("latency_ms") or 0.0)
        aggregate_network_attempts += 0 if request.cache_hit else int(response.get("network_attempt_count") or 1)
        aggregate_cache_requests += int(request.cache_hit)
        if not request.cache_hit:
            response_events = response.get("request_started_at_events")
            if isinstance(response_events, list):
                aggregate_request_events.extend(str(value) for value in response_events if value)
            elif response.get("request_started_at"):
                aggregate_request_events.append(str(response["request_started_at"]))
        if request.request_error or str(response.get("finish_reason") or "") in {"length", "content_filter"}:
            break
        try:
            raw_text = str(response.get("assistant_text") or "")
            try:
                raw_payload = json.loads(raw_text)
            except (json.JSONDecodeError, TypeError):
                raw_payload = {}
            if isinstance(raw_payload, dict):
                if isinstance(raw_payload.get("final_answer"), (int, float, bool)):
                    protocol_normalization_flags.append("final_answer_json_scalar_to_text")
                summary = raw_payload.get("reasoning_summary")
                if isinstance(summary, str) and len(summary.split()) > reasoning_word_limit:
                    protocol_normalization_flags.append("reasoning_summary_truncated_to_word_limit")
            validated = parse_synthesis_output(raw_text, reasoning_word_limit=reasoning_word_limit)
            break
        except Exception as exc:
            parse_error = str(exc)
            cache.delete(request.cache_key)
            if protocol_attempt == 2:
                break
    assert request is not None
    if not validated and not request.request_error:
        cache.delete(request.cache_key)
    status = "request_fail" if request.request_error else ("ok" if validated else "protocol_fail")
    answer = str(validated.get("final_answer") or "")
    return {
        "run_id": run_id, "dataset": dataset, "split": split_name, "sample_id": sample.sample_id,
        "method_name": "rcta_trace_synthesizer", "method_type": "rcta_synthesis", "round_index": 1,
        "agent_id": 1, "role": "trace_synthesizer", "prompt_hash": request.prompt_hash, "cache_key": request.cache_key,
        "prediction": answer, "normalized_answer": answer, "score": None, "output_status": status,
        "prompt_tokens": aggregate_usage["prompt_tokens"], "completion_tokens": aggregate_usage["completion_tokens"],
        "total_tokens": aggregate_usage["total_tokens"], "latency_ms": aggregate_latency,
        "cache_hit": request.cache_hit, "request_error": request.request_error,
        "request_status": "request_fail" if request.request_error else "ok",
        "raw_finish_reason": response.get("finish_reason"), "output_protocol": RCTA_SCHEMA_VERSION,
        "protocol_parse_status": "not_attempted" if request.request_error else ("ok" if validated else "failed"),
        "protocol_parse_error": parse_error, "reason_present": bool(validated.get("reasoning_summary")),
        "request_count": aggregate_cache_requests + aggregate_network_attempts,
        "cache_request_count": aggregate_cache_requests, "network_request_count": aggregate_network_attempts,
        "logical_call_count": 1, "network_attempt_count": aggregate_network_attempts,
        "protocol_attempt_count": protocol_attempts,
        "protocol_normalization_flags": sorted(set(protocol_normalization_flags)),
        "request_started_at": response.get("request_started_at"), "payload": request.payload,
        "request_started_at_events": aggregate_request_events,
        "assistant_text": str(response.get("assistant_text") or ""),
        "provider_reasoning_text": str(response.get("provider_reasoning_text") or ""), "validated_output": validated,
    }


def _run_debate_round(
    *, run_id, dataset, split_name, sample, stage_rows, method_name, confidence_mode,
    experiment, protocol, backbone, provider, cache, throttle,
) -> list[dict[str, Any]]:
    requests = []
    for index, own in enumerate(stage_rows, start=1):
        peers = [row for row in stage_rows if row is not own]
        requests.append({
            "run_id": run_id, "dataset": dataset, "split_name": split_name, "sample": sample,
            "method_name": method_name, "method_type": "mad_update", "round_index": 1, "agent_id": index,
            "role": "debate_update", "messages": build_debate_update_messages(sample, own, peers, confidence_mode=confidence_mode),
            "backbone": backbone, "provider": provider, "cache": cache, "throttle": throttle,
            "temperature": protocol.debate_temperature, "top_p": protocol.top_p,
            "seed": experiment.global_seed + 30_000 + index + (100 if confidence_mode else 0),
            "max_tokens": protocol.debate_max_tokens,
        })
    with ThreadPoolExecutor(max_workers=5) as executor:
        return list(executor.map(lambda values: _execute_free_text_turn(**values), requests))


def _prediction_from_votes(
    *, run_id, dataset, split_name, sample, backbone_name, method_name, method_type, rows, anchor,
    initial_rows=None, additional_rows=None,
) -> dict[str, Any]:
    initial = list(initial_rows or rows)
    additional = list(additional_rows or [])
    initial_stage = stage_decision(initial[:5])
    prediction, counts, resolver = majority_with_anchor_fallback(list(rows), anchor)
    initial_answer = initial_stage.anchor_answer or anchor
    score = score_prediction(dataset, prediction, sample.reference_answer)
    initial_score = score_prediction(dataset, initial_answer, sample.reference_answer)
    return _base_prediction(
        run_id=run_id, dataset=dataset, split_name=split_name, sample=sample, backbone_name=backbone_name,
        method_name=method_name, method_type=method_type, prediction=prediction, score=score,
        initial_answer=initial_answer, initial_score=initial_score, vote_counts=counts,
        initial_counts=initial_stage.vote_counts, initial_rows=initial, additional_rows=additional,
        triggered=initial_stage.triggered, resolver=resolver, override=prediction != initial_answer,
    )


def _rcta_prediction(
    *, run_id, dataset, split_name, sample, backbone_name, method_name, stage_rows, stage, anchor_score,
    synthesis_turn, synthesis, certificate, feature_vector, accept, resolver, board, board_counts,
    router_result=None,
) -> dict[str, Any]:
    synthesis_answer = str(synthesis.get("final_answer") or "")
    valid_accept = bool(accept and synthesis_answer and synthesis_turn and synthesis_turn.get("output_status") == "ok")
    prediction = synthesis_answer if valid_accept else stage.anchor_answer
    score = score_prediction(dataset, prediction, sample.reference_answer)
    turns = [*stage_rows, *([synthesis_turn] if synthesis_turn is not None else [])]
    base = _base_prediction(
        run_id=run_id, dataset=dataset, split_name=split_name, sample=sample, backbone_name=backbone_name,
        method_name=method_name, method_type="rcta", prediction=prediction, score=score,
        initial_answer=stage.anchor_answer, initial_score=anchor_score, vote_counts={prediction: 1},
        initial_counts=stage.vote_counts, initial_rows=stage_rows,
        additional_rows=[synthesis_turn] if synthesis_turn is not None else [],
        triggered=stage.triggered, resolver=resolver if valid_accept else "rcta_anchor_fallback",
        override=valid_accept and prediction != stage.anchor_answer,
    )
    synthesis_score = score_prediction(dataset, synthesis_answer, sample.reference_answer) if synthesis_answer else 0.0
    base.update({
        "synthesis_answer": synthesis_answer,
        "synthesis_score": synthesis_score,
        "synthesis_existing_candidate": synthesis_answer in stage.vote_counts,
        "synthesis_source_trace_ids": list(synthesis.get("source_trace_ids") or []),
        "decisive_claim": synthesis.get("decisive_claim"),
        "certificate": certificate,
        "feature_version": "rcta_router_features_v1",
        "feature_vector": feature_vector,
        "router_result": router_result or {},
        "risk_score": (router_result or {}).get("risk_score"),
        "p_gain": (router_result or {}).get("p_gain"),
        "p_harm": (router_result or {}).get("p_harm"),
        "board_char_count": len(board),
        "trace_char_counts": board_counts,
        "candidate_answers": sorted(stage.vote_counts),
        "candidate_oracle_correct": any(score_prediction(dataset, answer, sample.reference_answer) == 1.0 for answer in stage.vote_counts),
        "candidate_board_all_candidates_visible": all(f"Trace T{index}" in board for index in range(1, 6)) if board else None,
        "protocol_normalization_flags": list((synthesis_turn or {}).get("protocol_normalization_flags") or []),
        "logical_calls_per_question": 5 + int(synthesis_turn is not None),
        "network_attempts_per_question": sum(int(row.get("network_attempt_count") or 0) for row in turns),
    })
    return base


def _base_prediction(
    *, run_id, dataset, split_name, sample, backbone_name, method_name, method_type, prediction, score,
    initial_answer, initial_score, vote_counts, initial_counts, initial_rows, additional_rows,
    triggered, resolver, override,
) -> dict[str, Any]:
    turns = [*initial_rows, *additional_rows]
    return {
        "run_id": run_id, "dataset": dataset, "split": split_name, "sample_id": sample.sample_id,
        "task": sample.metadata.get("task"), "method_name": method_name, "method_type": method_type,
        "model_name": backbone_name, "prediction": prediction, "gold": sample.reference_answer, "score": score,
        "initial_vote_prediction": initial_answer, "initial_vote_score": initial_score,
        "initial_vote_counts": initial_counts, "initial_consensus": len(initial_counts) <= 1,
        "final_vote_prediction": prediction, "final_vote_score": score, "final_vote_counts": vote_counts,
        "prompt_tokens_per_question": _sum(turns, "prompt_tokens"),
        "completion_tokens_per_question": _sum(turns, "completion_tokens"),
        "total_tokens_per_question": _sum(turns, "total_tokens"),
        "latency_ms_per_question": _sum(turns, "latency_ms"),
        "initial_prompt_tokens_per_question": _sum(initial_rows, "prompt_tokens"),
        "initial_completion_tokens_per_question": _sum(initial_rows, "completion_tokens"),
        "initial_total_tokens_per_question": _sum(initial_rows, "total_tokens"),
        "initial_latency_ms_per_question": _sum(initial_rows, "latency_ms"),
        "debate_prompt_tokens_per_question": _sum(additional_rows, "prompt_tokens"),
        "debate_completion_tokens_per_question": _sum(additional_rows, "completion_tokens"),
        "debate_total_tokens_per_question": _sum(additional_rows, "total_tokens"),
        "debate_latency_ms_per_question": _sum(additional_rows, "latency_ms"),
        "calls_per_question": len(turns), "logical_calls_per_question": len(turns),
        "network_attempts_per_question": sum(int(row.get("network_attempt_count") or 0) for row in turns),
        "debate_rounds": 1 if additional_rows else 0, "agent_count": len(initial_rows),
        "final_consensus": len(vote_counts) <= 1, "initial_disagreement": triggered,
        "vote_flipped": override, "corrected_by_debate": initial_score < 1.0 and score == 1.0,
        "harmed_by_debate": initial_score == 1.0 and score < 1.0,
        "unchanged_correct": not override and score == 1.0, "unchanged_wrong": not override and score < 1.0,
        "triggered": triggered, "router_reasons": ["answer_disagreement"] if triggered else [],
        "resolver": resolver, "protocol_failures_per_question": sum(row.get("protocol_parse_status") == "failed" for row in turns),
        "reason_missing_turns_per_question": sum(not row.get("reason_present") for row in turns),
        "vote_counts": initial_counts, "override_accepted": override,
        "stage_a_prompt_hashes": [str(row.get("prompt_hash") or "") for row in initial_rows[:5]],
    }


def _debate_message_rows(run_id, dataset, split_name, sample, method_name, turns):
    return [{
        "run_id": run_id, "dataset": dataset, "split": split_name, "sample_id": sample.sample_id,
        "method_name": method_name, "round_index": 1, "sender_agent_id": row.get("agent_id"),
        "recipient_agent_id": 0, "message_kind": "one_round_update", "sender_answer": row.get("normalized_answer", ""),
        "sender_reasoning": row.get("assistant_text", ""),
    } for row in turns]


def _sum(rows: list[dict[str, Any]], key: str) -> float:
    return float(sum(float(row.get(key) or 0.0) for row in rows if row is not None))


def estimate_work(experiment: RctaExperimentConfig, phase_name: str, benchmarks, active_methods: list[str]) -> tuple[int, int]:
    sample_total = 0
    for benchmark in benchmarks:
        sample_total += len(load_selected_samples(benchmark, resolve_split_name(experiment, phase_name, benchmark.slug)))
    upper_calls = 9
    if {"gsa_trace_1", "rcta_certificate_shadow_1", "rcta_1", "rcta_no_certificate", "rcta_existing_only"}.intersection(active_methods):
        upper_calls += 1
    upper_calls += 5 * int("mad_5a_r1" in active_methods)
    upper_calls += 5 * int("confidence_mad_5a_r1" in active_methods)
    return sample_total * upper_calls, sample_total * (len(experiment.control_methods) + len(active_methods))
