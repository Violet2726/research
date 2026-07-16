"""共享 Stage-A 调用的 DGCR 样本级执行。"""

from __future__ import annotations

import json
from typing import Any

from research_experiments.core.controls.control_prompts import build_cot_messages
from research_experiments.core.data.datasets import DatasetSample
from research_experiments.core.data.evaluation import canonicalize_answer, score_prediction
from research_experiments.core.execution.runner_common import execute_cached_request
from research_experiments.family_runtime.free_text_protocol import FREE_TEXT_ANSWER_PROTOCOL_V1
from research_experiments.family_runtime.output_protocols import execute_output_protocol_turn
from research_experiments.families.disagreement_guided_crux_reconstruction.algorithms import (
    build_panel_labels,
    build_stage_decision,
    decide_override,
    panel_successes,
    validate_crux_span,
)
from research_experiments.families.disagreement_guided_crux_reconstruction.prompts import (
    build_panel_messages,
    build_proposer_messages,
)


def run_dgcr_sample(
    sample: DatasetSample,
    *,
    run_id: str,
    split_name: str,
    experiment,
    protocol,
    endpoint,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    stage_rows = [
        _answer_turn(
            sample,
            run_id=run_id,
            split_name=split_name,
            endpoint=endpoint,
            method_name="dgcr_stage_a_shared",
            role="stage_a_solver",
            agent_id=index,
            seed=42_000 + index,
            max_tokens=protocol.solver_max_tokens,
        )
        for index in range(1, protocol.stage_candidates + 1)
    ]
    stage = build_stage_decision(stage_rows, seed=experiment.global_seed, sample_id=sample.sample_id)
    physical_rows = list(stage_rows)
    resample_rows: list[dict[str, Any]] = []
    proposer_row: dict[str, Any] | None = None
    panel_rows: list[dict[str, Any]] = []
    span = None
    panel_results = []
    label_mappings: list[dict[str, str]] = []

    if stage.triggered:
        resample_rows = [
            _answer_turn(
                sample,
                run_id=run_id,
                split_name=split_name,
                endpoint=endpoint,
                method_name="dgcr_adaptive_resample_shared",
                role="independent_resample",
                agent_id=index,
                seed=45_000 + index,
                max_tokens=protocol.solver_max_tokens,
            )
            for index in range(1, protocol.resample_candidates + 1)
        ]
        proposer_labels = build_panel_labels(
            stage.candidates, seed=experiment.global_seed, sample_id=sample.sample_id, panel_index=0
        )
        proposer_row, proposer_payload = _json_turn(
            sample,
            run_id=run_id,
            split_name=split_name,
            endpoint=endpoint,
            method_name="dgcr_crux_proposer",
            role="crux_proposer",
            agent_id=1,
            seed=43_000,
            max_tokens=protocol.role_max_tokens,
            messages=build_proposer_messages(sample, label_to_key=proposer_labels),
        )
        if proposer_payload is not None:
            span = validate_crux_span(
                sample.question,
                start_char=proposer_payload.get("start_char", -1),
                end_char=proposer_payload.get("end_char", -1),
            )
        if span is not None:
            for panel_index in range(1, protocol.panel_count + 1):
                labels = build_panel_labels(
                    stage.candidates,
                    seed=experiment.global_seed,
                    sample_id=sample.sample_id,
                    panel_index=panel_index,
                )
                panel_row, panel_payload = _json_turn(
                    sample,
                    run_id=run_id,
                    split_name=split_name,
                    endpoint=endpoint,
                    method_name="dgcr_reconstruction_panel",
                    role="reconstruction_panel",
                    agent_id=panel_index,
                    seed=44_000 + panel_index,
                    max_tokens=protocol.role_max_tokens,
                    messages=build_panel_messages(sample, masked_question=span.masked_question, label_to_key=labels),
                )
                panel_row["label_to_answer_class_key"] = labels
                panel_rows.append(panel_row)
                label_mappings.append(labels)
                reconstructions = panel_payload.get("reconstructions") if isinstance(panel_payload, dict) else None
                panel_results.append(
                    panel_successes(reconstructions, label_to_key=labels, span=span)
                    if isinstance(reconstructions, dict)
                    else None
                )
        else:
            panel_results = [None, None]
        physical_rows.extend([*resample_rows, proposer_row, *panel_rows])

    adaptive = build_stage_decision(
        [*stage_rows, *resample_rows], seed=experiment.global_seed, sample_id=sample.sample_id
    )
    adaptive_answer = adaptive.anchor_answer or stage.anchor_answer
    dgcr_answer, override, resolver = decide_override(stage, panel_results) if stage.triggered else (
        stage.anchor_answer,
        False,
        "no_answer_class_disagreement",
    )
    sc5_score = _score(sample, stage.anchor_answer)
    adaptive_score = _score(sample, adaptive_answer)
    dgcr_score = _score(sample, dgcr_answer)
    candidate_oracle = any(_score(sample, candidate.answer) == 1.0 for candidate in stage.candidates)
    router = {
        "run_id": run_id,
        "dataset": sample.dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "triggered": stage.triggered,
        "valid_stage_answers": stage.valid_count,
        "anchor_answer": stage.anchor_answer,
        "answer_class_vote_counts": stage.vote_counts,
        "candidate_count": len(stage.candidates),
        "candidate_oracle_correct": candidate_oracle,
        "crux_valid": span is not None,
        "crux_start_char": span.start_char if span is not None else None,
        "crux_end_char": span.end_char if span is not None else None,
        "crux_sha256": _sha256(span.hidden_text) if span is not None else None,
        "panel_label_mappings": label_mappings,
        "panel_successes": panel_results,
        "skipped_panel_calls": max(0, protocol.panel_count - len(panel_rows)) if stage.triggered else 0,
        "override_accepted": override,
        "resolver": resolver,
    }
    predictions = [
        _prediction(
            sample, run_id, split_name, "sc_5", stage.anchor_answer, sc5_score, stage.anchor_answer, sc5_score,
            stage, stage_rows, [], False, "stage_a_majority", candidate_oracle,
        ),
        _prediction(
            sample, run_id, split_name, "adaptive_sc_8", adaptive_answer, adaptive_score, stage.anchor_answer, sc5_score,
            stage, [*stage_rows, *resample_rows], resample_rows, adaptive_answer != stage.anchor_answer,
            "adaptive_answer_class_majority" if stage.triggered else "no_answer_class_disagreement", candidate_oracle,
        ),
        _prediction(
            sample, run_id, split_name, "dgcr", dgcr_answer, dgcr_score, stage.anchor_answer, sc5_score,
            stage, [*stage_rows, *([proposer_row] if proposer_row is not None else []), *panel_rows],
            [*([proposer_row] if proposer_row is not None else []), *panel_rows],
            override, resolver, candidate_oracle,
        ),
    ]
    return [row for row in physical_rows if row is not None], router, predictions


def _answer_turn(sample: DatasetSample, *, run_id: str, split_name: str, endpoint, method_name: str, role: str, agent_id: int, seed: int, max_tokens: int) -> dict[str, Any]:
    result = execute_output_protocol_turn(
        backbone=endpoint.backbone, provider=endpoint.provider, cache=endpoint.cache, throttle=endpoint.throttle,
        sample=sample, messages=build_cot_messages(sample, agent_id, "single_agent_free_text_v1"),
        temperature=0.7, top_p=1.0, seed=seed, dataset=sample.dataset, role=role,
        output_protocol=FREE_TEXT_ANSWER_PROTOCOL_V1, max_tokens=max_tokens,
    )
    raw_answer = str(result.validated_output.get("final_answer") or "")
    canonical = canonicalize_answer(sample, raw_answer) if result.output_status == "ok" else None
    row = _turn_base(
        run_id, sample, split_name, method_name, role, agent_id, seed, result.payload, result.response_payload,
        result.cache_hit, result.request_error, result.raw_finish_reason, result.usage,
        "ok" if canonical is not None and canonical.valid else "failed",
        {"final_answer": raw_answer},
        canonical.key if canonical is not None and canonical.valid else "",
        canonical.invalid_reason if canonical is not None else "request_or_protocol_failure",
    )
    row["cache_namespace"] = endpoint.cache_namespace
    return row


def _json_turn(sample: DatasetSample, *, run_id: str, split_name: str, endpoint, method_name: str, role: str, agent_id: int, seed: int, max_tokens: int, messages: list[dict[str, str]]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    request = execute_cached_request(
        backbone=endpoint.backbone, provider=endpoint.provider, cache=endpoint.cache, throttle=endpoint.throttle,
        messages=messages, temperature=0.7, top_p=1.0, seed=seed, use_response_format=True, max_tokens=max_tokens,
    )
    parsed: dict[str, Any] | None = None
    error = request.request_error
    if not error:
        try:
            candidate = json.loads(str(request.response_payload.get("assistant_text") or ""))
            if not isinstance(candidate, dict):
                raise ValueError("JSON output must be an object")
            parsed = candidate
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            error = str(exc)
            endpoint.cache.delete(request.cache_key)
    row = _turn_base(
        run_id, sample, split_name, method_name, role, agent_id, seed, request.payload, request.response_payload,
        request.cache_hit, request.request_error, request.response_payload.get("finish_reason"), request.usage,
        "ok" if parsed is not None else "failed", parsed or {}, "", error,
    )
    row["cache_namespace"] = endpoint.cache_namespace
    return row, parsed


def _turn_base(run_id, sample, split_name, method_name, role, agent_id, seed, payload, response, cache_hit, request_error, finish_reason, usage, parse_status, validated, answer_key, invalid_reason):
    attempts = 0 if cache_hit else max(1, int(response.get("network_attempt_count") or 1))
    reported_usage = dict(response.get("usage_reported") or {})
    details = reported_usage.get("completion_tokens_details") or {}
    return {
        "run_id": run_id, "dataset": sample.dataset, "split": split_name, "sample_id": sample.sample_id,
        "task": sample.metadata.get("task"), "method_name": method_name, "role": role, "agent_id": agent_id,
        "request_seed": seed, "payload": payload, "cache_namespace": None,
        "request_source": "dgcr_confirmation_cache",
        "prompt_hash": _sha256(json.dumps(payload.get("messages") or [], ensure_ascii=False, sort_keys=True)),
        "cache_hit": cache_hit, "request_error": request_error, "request_status": "request_fail" if request_error else "ok",
        "raw_finish_reason": finish_reason, "network_attempt_count": attempts, "network_request_count": attempts,
        "request_count": max(1, attempts), "cache_request_count": int(cache_hit),
        "prompt_tokens": float(usage.get("prompt_tokens") or 0), "completion_tokens": float(usage.get("completion_tokens") or 0),
        "total_tokens": float(usage.get("total_tokens") or 0), "reasoning_tokens": reported_usage.get("reasoning_tokens", details.get("reasoning_tokens")),
        "usage_source": response.get("usage_source"), "usage_reported": reported_usage,
        "actual_prompt_tokens": reported_usage.get("prompt_tokens"),
        "actual_completion_tokens": reported_usage.get("completion_tokens"),
        "actual_total_tokens": reported_usage.get("total_tokens"),
        "latency_ms": float(response.get("latency_ms") or 0), "assistant_text": str(response.get("assistant_text") or ""),
        "provider_reasoning_text": str(response.get("provider_reasoning_text") or ""), "validated_output": validated,
        "protocol_parse_status": parse_status, "protocol_parse_error": invalid_reason,
        "prediction": answer_key, "normalized_answer": answer_key, "answer_class_key": answer_key,
        "canonicalization_invalid_reason": invalid_reason,
    }


def _prediction(sample, run_id, split_name, method_name, prediction, score, initial, initial_score, stage, logical_rows, intervention_rows, override, resolver, candidate_oracle):
    planned_intervention_calls = 3 if stage.triggered and method_name in {"adaptive_sc_8", "dgcr"} else 0
    logical_calls = max(len(logical_rows), 5 + planned_intervention_calls)
    return {
        "run_id": run_id, "dataset": sample.dataset, "split": split_name, "sample_id": sample.sample_id,
        "task": sample.metadata.get("task"), "method_name": method_name, "prediction": prediction,
        "gold": sample.reference_answer, "score": score, "initial_vote_prediction": initial,
        "initial_vote_score": initial_score, "initial_answer_class_key": stage.anchor_key,
        "initial_vote_counts": stage.vote_counts, "candidate_oracle_correct": candidate_oracle,
        "triggered": stage.triggered, "override_accepted": override, "vote_flipped": override,
        "corrected_by_debate": override and initial_score < 1 and score == 1,
        "harmed_by_debate": override and initial_score == 1 and score < 1,
        "resolver": resolver, "calls_per_question": logical_calls,
        "logical_calls_per_question": logical_calls,
        "actual_executed_calls_per_question": len(logical_rows),
        "total_tokens_per_question": sum(float(row.get("total_tokens") or 0) for row in logical_rows),
        "completion_tokens_per_question": sum(float(row.get("completion_tokens") or 0) for row in logical_rows),
        "network_attempts_per_question": sum(int(row.get("network_attempt_count") or 0) for row in logical_rows),
        "intervention_calls_per_question": planned_intervention_calls,
        "actual_intervention_calls_per_question": len(intervention_rows),
    }


def _score(sample: DatasetSample, answer: str) -> float:
    return score_prediction(sample.dataset, answer, sample.reference_answer, sample=sample) if answer else 0.0


def _sha256(value: str) -> str:
    import hashlib
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()
