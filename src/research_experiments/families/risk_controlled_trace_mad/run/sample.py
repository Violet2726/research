"""EVF-MAD 的异构样本执行链路。"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any

from research_experiments.core.controls.control_prompts import build_cot_messages
from research_experiments.core.data.datasets import DatasetSample, select_samples
from research_experiments.core.data.evaluation import canonicalize_answer, normalize_prediction, score_prediction
from research_experiments.core.execution.runner_common import execute_cached_request, iter_indexed_batch
from research_experiments.families.risk_controlled_trace_mad.algorithms import (
    StageDecision,
    build_candidate_board,
    decide_override,
    deterministic_challenger_fallback,
    majority_with_anchor_fallback,
    normalized_answer,
    stage_decision,
)
from research_experiments.families.risk_controlled_trace_mad.certificates import EVF_TEST_TYPES, verify_evidence
from research_experiments.families.risk_controlled_trace_mad.config import (
    EvfProtocolConfig,
    MadInnovationExperimentConfig,
)
from research_experiments.families.risk_controlled_trace_mad.prompts import (
    EVF_AUDIT_SCHEMA_VERSION,
    EVF_SELECTOR_SCHEMA_VERSION,
    build_audit_messages,
    build_cross_exam_messages,
    build_selector_messages,
)
from research_experiments.family_runtime.common import resolve_phase_split_name
from research_experiments.family_runtime.free_text_protocol import FREE_TEXT_ANSWER_PROTOCOL_V1
from research_experiments.family_runtime.output_protocols import execute_output_protocol_turn


@dataclass(frozen=True)
class ModelEndpoint:
    lineage: str
    backbone: Any
    provider: Any
    cache: Any
    throttle: Any


def resolve_split_name(experiment: MadInnovationExperimentConfig, phase_name: str, dataset: str) -> str:
    return resolve_phase_split_name(experiment, phase_name, dataset)


def load_selected_samples(benchmark, split_name: str) -> list[DatasetSample]:
    return select_samples(benchmark, split_name)


def run_batch(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    samples: list[DatasetSample],
    experiment: MadInnovationExperimentConfig,
    protocol: EvfProtocolConfig,
    active_methods: list[str],
    qwen: ModelEndpoint,
    mimo: ModelEndpoint,
    max_concurrent_samples: int,
) -> Iterator[tuple[int, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]]:
    worker = partial(
        run_sample,
        run_id=run_id,
        dataset=dataset,
        split_name=split_name,
        experiment=experiment,
        protocol=protocol,
        active_methods=active_methods,
        qwen=qwen,
        mimo=mimo,
    )
    for index, result in iter_indexed_batch(samples, worker=worker, max_concurrent_requests=max_concurrent_samples):
        yield index, *result


def run_sample(
    sample: DatasetSample,
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    experiment: MadInnovationExperimentConfig,
    protocol: EvfProtocolConfig,
    active_methods: list[str],
    qwen: ModelEndpoint,
    mimo: ModelEndpoint,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    qwen_rows, mimo_rows = _run_stage_pools(
        run_id=run_id,
        dataset=dataset,
        split_name=split_name,
        sample=sample,
        experiment=experiment,
        protocol=protocol,
        qwen=qwen,
        mimo=mimo,
    )
    mixed_rows = [*qwen_rows[:3], *mimo_rows[:2]]
    stage = stage_decision(mixed_rows, qwen_rows=qwen_rows[:3])
    turns = [*qwen_rows, *mimo_rows]
    messages: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    model_name = f"{qwen.backbone.name}+{mimo.backbone.name}"

    row_sets = {
        "cot_1": qwen_rows[:1],
        "qwen_sc_5": qwen_rows[:5],
        "qwen_sc_9": qwen_rows[:9],
        "mimo_sc_5": mimo_rows[:5],
        "mimo_sc_9": mimo_rows[:9],
        "heterogeneous_mv_5": mixed_rows,
    }
    for method in active_methods:
        if method in row_sets:
            method_rows = row_sets[method]
            method_stage = stage_decision(method_rows, qwen_rows=qwen_rows[:3])
            predictions.append(
                _vote_prediction(
                    run_id=run_id,
                    dataset=dataset,
                    split_name=split_name,
                    sample=sample,
                    model_name=model_name,
                    method_name=method,
                    rows=method_rows,
                    anchor=method_stage.anchor_answer,
                    initial_rows=mixed_rows,
                    initial_stage=stage,
                )
            )

    if "hcp_mad_budget10" in active_methods:
        hcp_rows = [*mixed_rows, *qwen_rows[3:6], *mimo_rows[2:4]]
        predictions.append(
            _vote_prediction(
                run_id=run_id,
                dataset=dataset,
                split_name=split_name,
                sample=sample,
                model_name=model_name,
                method_name="hcp_mad_budget10",
                rows=hcp_rows,
                anchor=stage.anchor_answer,
                initial_rows=mixed_rows,
                initial_stage=stage,
            )
        )

    if "minority_sentinel_reproduction" in active_methods:
        answer, reason = _minority_sentinel_nonofficial(mixed_rows, stage)
        predictions.append(
            _prediction(
                run_id=run_id,
                dataset=dataset,
                split_name=split_name,
                sample=sample,
                model_name=model_name,
                method_name="minority_sentinel_reproduction",
                prediction=answer,
                stage=stage,
                initial_rows=mixed_rows,
                additional_rows=[],
                resolver=reason,
                method_metadata={"reproduction_status": "non_official_rule_reproduction"},
            )
        )

    if "heterogeneous_gsa_1" in active_methods and stage.triggered:
        gsa_turn = _run_gsa(run_id, dataset, split_name, sample, mixed_rows, experiment, protocol, qwen)
        turns.append(gsa_turn)
        gsa_answer = normalized_answer(gsa_turn) or stage.anchor_answer
        predictions.append(
            _prediction(
                run_id=run_id,
                dataset=dataset,
                split_name=split_name,
                sample=sample,
                model_name=model_name,
                method_name="heterogeneous_gsa_1",
                prediction=gsa_answer,
                stage=stage,
                initial_rows=mixed_rows,
                additional_rows=[gsa_turn],
                resolver="heterogeneous_gsa",
                method_metadata={"novel_answer": gsa_answer not in stage.vote_counts},
            )
        )
    elif "heterogeneous_gsa_1" in active_methods:
        predictions.append(
            _prediction(
                run_id=run_id,
                dataset=dataset,
                split_name=split_name,
                sample=sample,
                model_name=model_name,
                method_name="heterogeneous_gsa_1",
                prediction=stage.anchor_answer,
                stage=stage,
                initial_rows=mixed_rows,
                additional_rows=[],
                resolver="unanimous_no_trigger",
            )
        )

    if "mad_5a_r1" in active_methods:
        debate_turns = (
            _run_mad_round(
                run_id,
                dataset,
                split_name,
                sample,
                mixed_rows,
                experiment,
                protocol,
                qwen,
                mimo,
            )
            if stage.triggered
            else []
        )
        turns.extend(debate_turns)
        answer, _, resolver = majority_with_anchor_fallback(debate_turns or mixed_rows, stage.anchor_answer)
        predictions.append(
            _prediction(
                run_id=run_id,
                dataset=dataset,
                split_name=split_name,
                sample=sample,
                model_name=model_name,
                method_name="mad_5a_r1",
                prediction=answer,
                stage=stage,
                initial_rows=mixed_rows,
                additional_rows=debate_turns,
                resolver=resolver,
            )
        )

    if "evf_mad_1" in active_methods:
        evf_turns, evf_messages, evf_decision, evf_prediction = _run_evf(
            run_id=run_id,
            dataset=dataset,
            split_name=split_name,
            sample=sample,
            experiment=experiment,
            protocol=protocol,
            mixed_rows=mixed_rows,
            stage=stage,
            qwen=qwen,
            mimo=mimo,
            model_name=model_name,
        )
        turns.extend(evf_turns)
        messages.extend(evf_messages)
        decisions.append(evf_decision)
        predictions.append(evf_prediction)

    return turns, messages, decisions, predictions


def _run_stage_pools(*, run_id, dataset, split_name, sample, experiment, protocol, qwen, mimo):
    requests: list[tuple[ModelEndpoint, int, int]] = []
    for index in range(protocol.stage_qwen_candidates):
        requests.append((qwen, index + 1, experiment.global_seed + index))
    for index in range(protocol.stage_mimo_candidates):
        requests.append((mimo, index + 1, experiment.global_seed + 100 + index))

    def execute(values):
        endpoint, agent_id, seed = values
        return _execute_free_text_turn(
            run_id=run_id,
            dataset=dataset,
            split_name=split_name,
            sample=sample,
            method_name="evf_stage_a_shared",
            method_type="evf_stage_a",
            round_index=0,
            agent_id=agent_id,
            role=f"{endpoint.lineage}_solver",
            messages=build_cot_messages(sample, agent_id, "single_agent_free_text_v1"),
            endpoint=endpoint,
            temperature=protocol.stage_temperature,
            top_p=protocol.top_p,
            seed=seed,
            max_tokens=None,
        )

    with ThreadPoolExecutor(max_workers=len(requests)) as executor:
        rows = list(executor.map(execute, requests))
    return rows[: protocol.stage_qwen_candidates], rows[protocol.stage_qwen_candidates :]


def _execute_free_text_turn(
    *,
    run_id,
    dataset,
    split_name,
    sample,
    method_name,
    method_type,
    round_index,
    agent_id,
    role,
    messages,
    endpoint,
    temperature,
    top_p,
    seed,
    max_tokens,
) -> dict[str, Any]:
    results = []
    for _attempt in range(2):
        result = execute_output_protocol_turn(
            backbone=endpoint.backbone,
            provider=endpoint.provider,
            cache=endpoint.cache,
            throttle=endpoint.throttle,
            sample=sample,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            dataset=dataset,
            role=role,
            output_protocol=FREE_TEXT_ANSWER_PROTOCOL_V1,
            max_tokens=max_tokens,
        )
        results.append(result)
        finish_reason = str(result.raw_finish_reason or "")
        if result.output_status == "ok" or finish_reason == "content_filter" or result.request_error:
            break
        endpoint.cache.delete(result.cache_key)
    result = results[-1]
    finish_reason = str(result.raw_finish_reason or "")
    if result.output_status != "ok":
        endpoint.cache.delete(result.cache_key)
    abstention = finish_reason == "content_filter"
    answer = str(result.validated_output.get("final_answer") or "")
    canonical = canonicalize_answer(sample, answer) if answer and result.output_status == "ok" else None
    normalized = (
        canonical.key
        if canonical is not None and canonical.valid
        else (normalize_prediction(dataset, answer) if answer and result.output_status == "ok" and dataset != "bbeh" else "")
    )
    usage = {
        key: sum(float(item.usage.get(key) or 0) for item in results)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    response = result.response_payload
    events = [event for item in results for event in list(item.response_payload.get("request_started_at_events") or [])]
    network_attempts = sum(
        0 if item.cache_hit else int(item.response_payload.get("network_attempt_count") or 1) for item in results
    )
    return {
        "run_id": run_id,
        "dataset": dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "method_name": method_name,
        "method_type": method_type,
        "round_index": round_index,
        "agent_id": agent_id,
        "role": role,
        "model_lineage": endpoint.lineage,
        "model_name": endpoint.backbone.name,
        "prompt_hash": result.prompt_hash,
        "cache_key": result.cache_key,
        "prediction": normalized,
        "normalized_answer": normalized,
        "answer_class_key": canonical.key if canonical is not None else "",
        "canonicalization_status": "valid" if canonical is not None and canonical.valid else "invalid",
        "canonicalization_invalid_reason": canonical.invalid_reason if canonical is not None else "empty_answer",
        "score": None,
        "output_status": "abstain" if abstention else result.output_status,
        "provider_abstention": abstention,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "latency_ms": sum(float(item.response_payload.get("latency_ms") or 0) for item in results),
        "cache_hit": all(item.cache_hit for item in results),
        "request_error": result.request_error,
        "request_status": result.request_status,
        "raw_finish_reason": result.raw_finish_reason,
        "output_protocol": result.output_protocol,
        "protocol_parse_status": ("abstain" if abstention else result.protocol_parse_status),
        "protocol_parse_error": result.protocol_parse_error,
        "reason_present": result.reason_present,
        "request_count": len(results) + network_attempts,
        "cache_request_count": sum(item.cache_hit for item in results),
        "network_request_count": network_attempts,
        "logical_call_count": 1,
        "network_attempt_count": network_attempts,
        "protocol_attempt_count": len(results),
        "request_started_at": response.get("request_started_at"),
        "request_started_at_events": events,
        "payload": result.payload,
        "assistant_text": str(response.get("assistant_text") or ""),
        "provider_reasoning_text": str(response.get("provider_reasoning_text") or ""),
        "validated_output": result.validated_output,
    }


def _run_evf(*, run_id, dataset, split_name, sample, experiment, protocol, mixed_rows, stage, qwen, mimo, model_name):
    turns: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    anchor_score = (
        score_prediction(dataset, stage.anchor_answer, sample.reference_answer, sample=sample)
        if stage.anchor_answer
        else 0.0
    )
    if stage.valid_trace_count < protocol.minimum_valid_stage_traces:
        prediction = _prediction(
            run_id=run_id,
            dataset=dataset,
            split_name=split_name,
            sample=sample,
            model_name=model_name,
            method_name="evf_mad_1",
            prediction=stage.anchor_answer,
            stage=stage,
            initial_rows=mixed_rows,
            additional_rows=[],
            resolver="insufficient_valid_stage_traces",
            method_metadata={"evf_gate_passed": False, "evf_gate_reasons": ["insufficient_valid_stage_traces"]},
        )
        return (
            turns,
            messages,
            _decision_row(
                run_id,
                dataset,
                split_name,
                sample,
                stage,
                anchor_score,
                "",
                [],
                False,
                ["insufficient_valid_stage_traces"],
            ),
            prediction,
        )
    if not stage.triggered:
        prediction = _prediction(
            run_id=run_id,
            dataset=dataset,
            split_name=split_name,
            sample=sample,
            model_name=model_name,
            method_name="evf_mad_1",
            prediction=stage.anchor_answer,
            stage=stage,
            initial_rows=mixed_rows,
            additional_rows=[],
            resolver="unanimous_no_trigger",
            method_metadata={"evf_gate_passed": False, "evf_gate_reasons": ["unanimous"]},
        )
        return (
            turns,
            messages,
            _decision_row(run_id, dataset, split_name, sample, stage, anchor_score, "", [], False, ["unanimous"]),
            prediction,
        )

    selector_board, selector_map, selector_counts = build_candidate_board(
        mixed_rows,
        seed=experiment.global_seed,
        sample_id=sample.sample_id,
        purpose="selector",
        trace_max_chars=protocol.trace_max_chars,
        board_max_chars=protocol.board_max_chars,
    )
    anchor_label = next(label for label, answer in selector_map.items() if answer == stage.anchor_answer)
    selector_turn = _execute_json_turn(
        run_id=run_id,
        dataset=dataset,
        split_name=split_name,
        sample=sample,
        method_name="evf_challenger_selector",
        role="challenger_selector",
        messages=build_selector_messages(sample, selector_board, anchor_label=anchor_label),
        endpoint=qwen,
        temperature=protocol.selector_temperature,
        top_p=protocol.top_p,
        seed=experiment.global_seed + 20_000,
        max_tokens=protocol.selector_max_tokens,
        schema_version=EVF_SELECTOR_SCHEMA_VERSION,
        parser=lambda raw: _parse_selector(raw, selector_map, stage.anchor_answer),
    )
    turns.append(selector_turn)
    challenger = str((selector_turn.get("validated_output") or {}).get("challenger_answer") or "")
    if not challenger:
        challenger = deterministic_challenger_fallback(stage, seed=experiment.global_seed, sample_id=sample.sample_id)

    audits: list[dict[str, Any]] = []
    for endpoint, purpose, offset in ((qwen, "qwen-audit", 30_000), (mimo, "mimo-audit", 40_000)):
        board, mapping, _ = build_candidate_board(
            mixed_rows,
            seed=experiment.global_seed,
            sample_id=sample.sample_id,
            purpose=purpose,
            trace_max_chars=protocol.trace_max_chars,
            board_max_chars=protocol.board_max_chars,
            restrict_answers={stage.anchor_answer, challenger},
        )
        turn = _execute_json_turn(
            run_id=run_id,
            dataset=dataset,
            split_name=split_name,
            sample=sample,
            method_name="evf_symmetric_audit",
            role=purpose,
            messages=build_audit_messages(sample, board),
            endpoint=endpoint,
            temperature=protocol.audit_temperature,
            top_p=protocol.top_p,
            seed=experiment.global_seed + offset,
            max_tokens=protocol.audit_max_tokens,
            schema_version=EVF_AUDIT_SCHEMA_VERSION,
            parser=lambda raw, mapping=mapping: _parse_audit(raw, mapping, sample.question),
        )
        turns.append(turn)
        if turn.get("output_status") == "ok":
            audits.append(dict(turn["validated_output"]))

    accepted, reasons = decide_override(
        anchor=stage.anchor_answer,
        challenger=challenger,
        audits=audits,
        challenger_required_passes=protocol.challenger_required_passes,
        anchor_required_falsifications=protocol.anchor_required_falsifications,
    )
    if (
        not accepted
        and len(turns) == 3
        and any(evidence.get("status") == "pass" for audit in audits for evidence in audit.get("evidence_results", []))
    ):
        cross_audits: list[dict[str, Any]] = []
        opposing = " | ".join(str(audit.get("decisive_claim") or "") for audit in audits)
        for endpoint, assigned, purpose, offset in (
            (qwen, stage.anchor_answer, "qwen-cross", 50_000),
            (mimo, challenger, "mimo-cross", 60_000),
        ):
            board, mapping, _ = build_candidate_board(
                mixed_rows,
                seed=experiment.global_seed,
                sample_id=sample.sample_id,
                purpose=purpose,
                trace_max_chars=protocol.trace_max_chars,
                board_max_chars=protocol.board_max_chars,
                restrict_answers={stage.anchor_answer, challenger},
            )
            assigned_label = next(label for label, answer in mapping.items() if answer == assigned)
            turn = _execute_json_turn(
                run_id=run_id,
                dataset=dataset,
                split_name=split_name,
                sample=sample,
                method_name="evf_cross_exam",
                role=purpose,
                messages=build_cross_exam_messages(
                    sample, board, assigned_label=assigned_label, opposing_claim=opposing
                ),
                endpoint=endpoint,
                temperature=protocol.cross_exam_temperature,
                top_p=protocol.top_p,
                seed=experiment.global_seed + offset,
                max_tokens=protocol.cross_exam_max_tokens,
                schema_version=EVF_AUDIT_SCHEMA_VERSION,
                parser=lambda raw, mapping=mapping: _parse_audit(raw, mapping, sample.question),
            )
            turns.append(turn)
            if turn.get("output_status") == "ok":
                cross_audits.append(dict(turn["validated_output"]))
        if len(cross_audits) == 2:
            audits = cross_audits
            accepted, reasons = decide_override(
                anchor=stage.anchor_answer,
                challenger=challenger,
                audits=audits,
                challenger_required_passes=protocol.challenger_required_passes,
                anchor_required_falsifications=protocol.anchor_required_falsifications,
            )

    prediction_answer = challenger if accepted else stage.anchor_answer
    prediction = _prediction(
        run_id=run_id,
        dataset=dataset,
        split_name=split_name,
        sample=sample,
        model_name=model_name,
        method_name="evf_mad_1",
        prediction=prediction_answer,
        stage=stage,
        initial_rows=mixed_rows,
        additional_rows=turns,
        resolver="evf_executable_override" if accepted else "evf_anchor_fallback",
        method_metadata={
            "evf_gate_passed": accepted,
            "evf_gate_reasons": reasons,
            "challenger_answer": challenger,
            "audit_count": len(audits),
            "evidence_results": [item for audit in audits for item in audit.get("evidence_results", [])],
            "candidate_answers": sorted(stage.vote_counts),
            "candidate_oracle_correct": any(
                score_prediction(dataset, answer, sample.reference_answer, sample=sample) == 1.0
                for answer in stage.vote_counts
            ),
            "selector_board_char_count": len(selector_board),
            "selector_trace_char_counts": selector_counts,
            "novel_answer": prediction_answer not in stage.vote_counts,
        },
    )
    messages.extend(
        {
            "run_id": run_id,
            "dataset": dataset,
            "split": split_name,
            "sample_id": sample.sample_id,
            "method_name": row["method_name"],
            "round_index": row["round_index"],
            "sender_agent_id": row["agent_id"],
            "recipient_agent_id": 0,
            "message_kind": row["role"],
            "sender_answer": row.get("normalized_answer", ""),
            "sender_reasoning": row.get("assistant_text", ""),
        }
        for row in turns
    )
    decision = _decision_row(
        run_id, dataset, split_name, sample, stage, anchor_score, challenger, audits, accepted, reasons
    )
    return turns, messages, decision, prediction


def _execute_json_turn(
    *,
    run_id,
    dataset,
    split_name,
    sample,
    method_name,
    role,
    messages,
    endpoint,
    temperature,
    top_p,
    seed,
    max_tokens,
    schema_version,
    parser: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    requests = []
    validated: dict[str, Any] = {}
    parse_error = None
    for _ in range(2):
        request = execute_cached_request(
            backbone=endpoint.backbone,
            provider=endpoint.provider,
            cache=endpoint.cache,
            throttle=endpoint.throttle,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            use_response_format=True,
            max_tokens=max_tokens,
        )
        requests.append(request)
        response = request.response_payload
        finish = str(response.get("finish_reason") or "")
        if request.request_error or finish in {"length", "content_filter"}:
            endpoint.cache.delete(request.cache_key)
            break
        try:
            validated = parser(str(response.get("assistant_text") or ""))
            break
        except Exception as exc:
            parse_error = str(exc)
            endpoint.cache.delete(request.cache_key)
    request = requests[-1]
    response = request.response_payload
    if not validated:
        endpoint.cache.delete(request.cache_key)
    finish = str(response.get("finish_reason") or "")
    abstention = finish == "content_filter"
    status = (
        "request_fail"
        if request.request_error
        else ("abstain" if abstention else ("ok" if validated else "protocol_fail"))
    )
    usage = {
        key: sum(float(item.usage.get(key) or 0) for item in requests)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    events = [
        event for item in requests for event in list(item.response_payload.get("request_started_at_events") or [])
    ]
    network_attempts = sum(
        0 if item.cache_hit else int(item.response_payload.get("network_attempt_count") or 1) for item in requests
    )
    answer = str(
        validated.get("final_answer") or validated.get("challenger_answer") or validated.get("preferred_answer") or ""
    )
    return {
        "run_id": run_id,
        "dataset": dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "method_name": method_name,
        "method_type": "evf_verification",
        "round_index": 1,
        "agent_id": 1,
        "role": role,
        "model_lineage": endpoint.lineage,
        "model_name": endpoint.backbone.name,
        "prompt_hash": request.prompt_hash,
        "cache_key": request.cache_key,
        "prediction": answer,
        "normalized_answer": answer,
        "score": None,
        "output_status": status,
        "provider_abstention": abstention,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "latency_ms": sum(float(item.response_payload.get("latency_ms") or 0) for item in requests),
        "cache_hit": all(item.cache_hit for item in requests),
        "request_error": request.request_error,
        "request_status": "request_fail" if request.request_error else "ok",
        "raw_finish_reason": response.get("finish_reason"),
        "output_protocol": schema_version,
        "protocol_parse_status": "ok" if validated else ("abstain" if abstention else "failed"),
        "protocol_parse_error": parse_error,
        "reason_present": bool(validated.get("decisive_claim") or validated.get("decisive_difference")),
        "request_count": len(requests) + network_attempts,
        "cache_request_count": sum(item.cache_hit for item in requests),
        "network_request_count": network_attempts,
        "logical_call_count": 1,
        "network_attempt_count": network_attempts,
        "protocol_attempt_count": len(requests),
        "request_started_at": response.get("request_started_at"),
        "request_started_at_events": events,
        "payload": request.payload,
        "assistant_text": str(response.get("assistant_text") or ""),
        "provider_reasoning_text": str(response.get("provider_reasoning_text") or ""),
        "validated_output": validated,
    }


def _parse_selector(raw_text: str, mapping: dict[str, str], anchor: str) -> dict[str, Any]:
    payload = _json_object(raw_text)
    label = str(payload.get("challenger_label") or "")
    difference = str(payload.get("decisive_difference") or "").strip()
    if label not in mapping or mapping[label] == anchor or not difference:
        raise ValueError("selector must choose one existing non-anchor label with a decisive difference")
    return {"challenger_label": label, "challenger_answer": mapping[label], "decisive_difference": difference[:1000]}


def _parse_audit(raw_text: str, mapping: dict[str, str], question: str) -> dict[str, Any]:
    payload = _json_object(raw_text)
    preferred_label = str(payload.get("preferred_label") or "")
    if preferred_label not in mapping:
        raise ValueError("audit preferred_label is not on the board")
    decisive = str(payload.get("decisive_claim") or "").strip()
    evidence_items = payload.get("evidence")
    if not decisive or not isinstance(evidence_items, list) or len(evidence_items) > 10:
        raise ValueError("audit evidence contract is invalid")
    results = []
    for raw in evidence_items:
        if not isinstance(raw, dict):
            raise ValueError("audit evidence item must be an object")
        label = str(raw.get("target_label") or "")
        if label not in mapping:
            raise ValueError("evidence target is not on the board")
        test_type = str(raw.get("test_type") or "unsupported")
        if test_type not in EVF_TEST_TYPES:
            raise ValueError("evidence test_type is invalid")
        contract = {
            "target_answer": mapping[label],
            "claim_kind": str(raw.get("claim_kind") or ""),
            "test_type": test_type,
            "payload": raw.get("payload") or {},
        }
        results.append(verify_evidence(question=question, target_answer=mapping[label], evidence=contract))
    return {
        "preferred_label": preferred_label,
        "preferred_answer": mapping[preferred_label],
        "decisive_claim": decisive[:1500],
        "evidence_results": results,
    }


def _json_object(raw_text: str) -> dict[str, Any]:
    text = str(raw_text or "").strip()
    if text.startswith("```") and text.endswith("```"):
        text = "\n".join(text.splitlines()[1:-1]).strip()
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("JSON output must be an object")
    return payload


def _run_gsa(run_id, dataset, split_name, sample, rows, experiment, protocol, qwen):
    board, mapping, _ = build_candidate_board(
        rows,
        seed=experiment.global_seed,
        sample_id=sample.sample_id,
        purpose="gsa",
        trace_max_chars=protocol.trace_max_chars,
        board_max_chars=protocol.board_max_chars,
    )
    prompt = [
        {
            "role": "system",
            "content": "Synthesize the best final answer from the anonymous responses. Return JSON only.",
        },
        {
            "role": "user",
            "content": f'Question:\n{sample.question}\n\nResponses:\n{board}\n\nReturn {{"final_answer":"...","reasoning_summary":"..."}}',
        },
    ]
    return _execute_json_turn(
        run_id=run_id,
        dataset=dataset,
        split_name=split_name,
        sample=sample,
        method_name="heterogeneous_gsa_synthesis",
        role="gsa_synthesis",
        messages=prompt,
        endpoint=qwen,
        temperature=0.7,
        top_p=protocol.top_p,
        seed=experiment.global_seed + 70_000,
        max_tokens=protocol.audit_max_tokens,
        schema_version="heterogeneous_gsa_v1",
        parser=lambda raw: _parse_gsa(raw, dataset),
    )


def _parse_gsa(raw: str, dataset: str) -> dict[str, Any]:
    payload = _json_object(raw)
    answer = str(payload.get("final_answer") or "").strip()
    reasoning = str(payload.get("reasoning_summary") or "").strip()
    if not answer or not reasoning:
        raise ValueError("GSA output is incomplete")
    return {"final_answer": normalize_prediction(dataset, answer), "reasoning_summary": reasoning[:1500]}


def _run_mad_round(run_id, dataset, split_name, sample, rows, experiment, protocol, qwen, mimo):
    requests = []
    for index, own in enumerate(rows):
        peers = [row for row in rows if row is not own]
        peer_text = "\n\n".join(
            f"Peer answer: {normalized_answer(row)}\n{str(row.get('assistant_text') or '')[:1000]}" for row in peers
        )
        prompt = [
            {
                "role": "system",
                "content": "Re-solve independently after checking peer arguments. Return REASONING then FINAL_ANSWER.",
            },
            {
                "role": "user",
                "content": f"Question:\n{sample.question}\n\nYour answer: {normalized_answer(own)}\nPeers:\n{peer_text}",
            },
        ]
        endpoint = qwen if own.get("model_lineage") == "qwen" else mimo
        requests.append((endpoint, index + 1, prompt))

    def execute(values):
        endpoint, agent_id, prompt = values
        return _execute_free_text_turn(
            run_id=run_id,
            dataset=dataset,
            split_name=split_name,
            sample=sample,
            method_name="mad_5a_r1_update",
            method_type="mad_update",
            round_index=1,
            agent_id=agent_id,
            role="debate_update",
            messages=prompt,
            endpoint=endpoint,
            temperature=0.7,
            top_p=protocol.top_p,
            seed=experiment.global_seed + 80_000 + agent_id,
            max_tokens=protocol.cross_exam_max_tokens,
        )

    with ThreadPoolExecutor(max_workers=5) as executor:
        return list(executor.map(execute, requests))


def _minority_sentinel_nonofficial(rows: list[dict[str, Any]], stage: StageDecision) -> tuple[str, str]:
    support: dict[str, set[str]] = {}
    for row in rows:
        answer = normalized_answer(row)
        if answer:
            support.setdefault(answer, set()).add(str(row.get("model_lineage") or ""))
    challengers = [
        answer
        for answer in stage.vote_counts
        if answer != stage.anchor_answer and support.get(answer) == {"qwen", "mimo"}
    ]
    anchor_lineages = support.get(stage.anchor_answer, set())
    if len(challengers) == 1 and len(anchor_lineages) == 1:
        return challengers[0], "nonofficial_minority_sentinel_cross_lineage_flip"
    return stage.anchor_answer, "nonofficial_minority_sentinel_anchor_fallback"


def _vote_prediction(
    *, run_id, dataset, split_name, sample, model_name, method_name, rows, anchor, initial_rows, initial_stage
):
    answer, _, resolver = majority_with_anchor_fallback(rows, anchor)
    return _prediction(
        run_id=run_id,
        dataset=dataset,
        split_name=split_name,
        sample=sample,
        model_name=model_name,
        method_name=method_name,
        prediction=answer,
        stage=initial_stage,
        initial_rows=initial_rows,
        additional_rows=[row for row in rows if row not in initial_rows],
        resolver=resolver,
        logical_rows=rows,
    )


def _prediction(
    *,
    run_id,
    dataset,
    split_name,
    sample,
    model_name,
    method_name,
    prediction,
    stage,
    initial_rows,
    additional_rows,
    resolver,
    method_metadata=None,
    logical_rows=None,
):
    score = score_prediction(dataset, prediction, sample.reference_answer, sample=sample) if prediction else 0.0
    initial_score = (
        score_prediction(dataset, stage.anchor_answer, sample.reference_answer, sample=sample)
        if stage.anchor_answer
        else 0.0
    )
    rows = list(logical_rows or [*initial_rows, *additional_rows])
    override = bool(prediction and prediction != stage.anchor_answer)
    abstentions = sum(bool(row.get("provider_abstention")) for row in rows)
    payload = {
        "run_id": run_id,
        "dataset": dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "task": sample.metadata.get("task"),
        "method_name": method_name,
        "method_type": "mad_innovation",
        "model_name": model_name,
        "prediction": prediction,
        "gold": sample.reference_answer,
        "score": score,
        "initial_vote_prediction": stage.anchor_answer,
        "initial_vote_score": initial_score,
        "initial_vote_counts": stage.vote_counts,
        "initial_consensus": not stage.triggered,
        "final_vote_prediction": prediction,
        "final_vote_score": score,
        "final_vote_counts": {prediction: 1} if prediction else {},
        "prompt_tokens_per_question": _sum(rows, "prompt_tokens"),
        "completion_tokens_per_question": _sum(rows, "completion_tokens"),
        "total_tokens_per_question": _sum(rows, "total_tokens"),
        "latency_ms_per_question": _sum(rows, "latency_ms"),
        "initial_prompt_tokens_per_question": _sum(initial_rows, "prompt_tokens"),
        "initial_completion_tokens_per_question": _sum(initial_rows, "completion_tokens"),
        "initial_total_tokens_per_question": _sum(initial_rows, "total_tokens"),
        "initial_latency_ms_per_question": _sum(initial_rows, "latency_ms"),
        "debate_prompt_tokens_per_question": _sum(additional_rows, "prompt_tokens"),
        "debate_completion_tokens_per_question": _sum(additional_rows, "completion_tokens"),
        "debate_total_tokens_per_question": _sum(additional_rows, "total_tokens"),
        "debate_latency_ms_per_question": _sum(additional_rows, "latency_ms"),
        "calls_per_question": len(rows),
        "logical_calls_per_question": len(rows),
        "network_attempts_per_question": sum(int(row.get("network_attempt_count") or 0) for row in rows),
        "valid_trajectories_per_question": sum(bool(normalized_answer(row)) for row in initial_rows),
        "provider_abstentions_per_question": abstentions,
        "debate_rounds": int(bool(additional_rows)),
        "agent_count": len(initial_rows),
        "final_consensus": True,
        "initial_disagreement": stage.triggered,
        "vote_flipped": override,
        "corrected_by_debate": initial_score < 1 and score == 1,
        "harmed_by_debate": initial_score == 1 and score < 1,
        "unchanged_correct": not override and score == 1,
        "unchanged_wrong": not override and score < 1,
        "triggered": stage.triggered,
        "router_reasons": ["answer_disagreement"] if stage.triggered else [],
        "resolver": resolver,
        "protocol_failures_per_question": sum(row.get("protocol_parse_status") == "failed" for row in rows),
        "request_failures_per_question": sum(row.get("output_status") == "request_fail" for row in rows),
        "reason_missing_turns_per_question": sum(
            row.get("protocol_parse_status") == "ok" and not row.get("reason_present") for row in rows
        ),
        "vote_counts": stage.vote_counts,
        "override_accepted": override,
        "stage_a_prompt_hashes": [str(row.get("prompt_hash") or "") for row in initial_rows],
    }
    payload.update(method_metadata or {})
    return payload


def _decision_row(run_id, dataset, split_name, sample, stage, anchor_score, challenger, audits, accepted, reasons):
    return {
        "run_id": run_id,
        "dataset": dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "policy_name": "evf_executable_falsification_v4",
        "triggered": stage.triggered,
        "anchor_answer": stage.anchor_answer,
        "anchor_score": anchor_score,
        "vote_counts": stage.vote_counts,
        "disagreement_pattern": stage.disagreement_pattern,
        "anchor_resolver": stage.resolver,
        "valid_trace_count": stage.valid_trace_count,
        "challenger_answer": challenger,
        "audits": audits,
        "override_accepted": accepted,
        "gate_reasons": reasons,
    }


def _sum(rows, key):
    return float(sum(float(row.get(key) or 0) for row in rows if row is not None))


def estimate_work(
    experiment: MadInnovationExperimentConfig, phase_name: str, benchmarks, active_methods: list[str]
) -> tuple[int, int]:
    sample_total = sum(
        len(load_selected_samples(benchmark, resolve_split_name(experiment, phase_name, benchmark.slug)))
        for benchmark in benchmarks
    )
    upper_calls = 18
    upper_calls += 1 if "heterogeneous_gsa_1" in active_methods else 0
    upper_calls += 5 if "mad_5a_r1" in active_methods else 0
    upper_calls += 5 if "evf_mad_1" in active_methods else 0
    return sample_total * upper_calls, sample_total * len(active_methods)
