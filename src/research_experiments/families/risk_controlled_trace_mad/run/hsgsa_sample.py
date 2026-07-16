"""H-SGSA v5 同质支持度盲化裁决的逐样本执行。

All reported methods are counterfactual views over one shared physical call
graph: five initial solves, then (only on answer-class disagreement) three
fresh solves and three independently permuted blind reviews.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

from research_experiments.core.controls.control_prompts import build_cot_messages
from research_experiments.core.data.datasets import DatasetSample
from research_experiments.core.data.evaluation import answer_class_key, normalize_prediction, score_prediction
from research_experiments.core.execution.runner_common import execute_cached_request, iter_indexed_batch
from research_experiments.families.risk_controlled_trace_mad.algorithms import (
    build_support_blind_board,
    class_majority,
    homogeneous_stage_decision,
    normalized_answer,
    reviewer_selected_key,
)
from research_experiments.families.risk_controlled_trace_mad.config import (
    HsgsaProtocolConfig,
    MadInnovationExperimentConfig,
)
from research_experiments.families.risk_controlled_trace_mad.prompts import (
    HSGSA_REVIEW_SCHEMA_VERSION,
    build_blind_reviewer_messages,
)
from research_experiments.families.risk_controlled_trace_mad.run.sample import (
    ModelEndpoint,
    _execute_free_text_turn,
)


class NetworkAttemptLimitExceeded(RuntimeError):
    pass


class NetworkAttemptBudget:
    """Thread-safe hard cap that reserves the worst case before a live call."""

    def __init__(self, limit: int) -> None:
        self.limit = int(limit)
        self.actual = 0
        self._reserved = 0
        self._lock = threading.Lock()

    def reserve(self, maximum: int) -> int:
        with self._lock:
            if self.actual + self._reserved + maximum > self.limit:
                raise NetworkAttemptLimitExceeded(
                    f"network-attempt hard stop: actual={self.actual}, reserved={self._reserved}, "
                    f"requested_reserve={maximum}, limit={self.limit}"
                )
            self._reserved += maximum
        return maximum

    def settle(self, reservation: int, actual: int) -> None:
        with self._lock:
            self._reserved -= reservation
            self.actual += int(actual)
            if self.actual > self.limit:
                raise NetworkAttemptLimitExceeded(
                    f"network-attempt hard stop exceeded: actual={self.actual}, limit={self.limit}"
                )


def run_hsgsa_batch(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    samples: list[DatasetSample],
    experiment: MadInnovationExperimentConfig,
    protocol: HsgsaProtocolConfig,
    active_methods: list[str],
    endpoint: ModelEndpoint,
    max_concurrent_samples: int,
    network_budget: NetworkAttemptBudget,
) -> Iterator[tuple[int, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]]:
    worker = partial(
        run_hsgsa_sample,
        run_id=run_id,
        dataset=dataset,
        split_name=split_name,
        experiment=experiment,
        protocol=protocol,
        active_methods=active_methods,
        endpoint=endpoint,
        network_budget=network_budget,
    )
    yield from (
        (index, *result)
        for index, result in iter_indexed_batch(
            samples, worker=worker, max_concurrent_requests=max_concurrent_samples
        )
    )


def run_hsgsa_sample(
    sample: DatasetSample,
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    experiment: MadInnovationExperimentConfig,
    protocol: HsgsaProtocolConfig,
    active_methods: list[str],
    endpoint: ModelEndpoint,
    network_budget: NetworkAttemptBudget,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    stage_rows = _solve_rows(
        count=protocol.stage_candidates,
        seed_offset=0,
        method_name="hsgsa_stage_a_shared",
        method_type="homogeneous_stage_a",
        role="homogeneous_solver",
        run_id=run_id,
        dataset=dataset,
        split_name=split_name,
        sample=sample,
        experiment=experiment,
        protocol=protocol,
        endpoint=endpoint,
        network_budget=network_budget,
    )
    stage = homogeneous_stage_decision(
        stage_rows, dataset=dataset, seed=experiment.global_seed, sample_id=sample.sample_id
    )
    triggered = stage.triggered and stage.valid_trace_count >= protocol.minimum_valid_stage_traces
    resample_rows: list[dict[str, Any]] = []
    reviewer_rows: list[dict[str, Any]] = []
    reviewer_metadata: list[dict[str, Any]] = []
    if triggered:
        with ThreadPoolExecutor(max_workers=2) as executor:
            resample_future = executor.submit(
                _solve_rows,
                count=protocol.resample_candidates,
                seed_offset=10_000,
                method_name="hsgsa_resample_shared",
                method_type="homogeneous_resample",
                role="independent_resample",
                run_id=run_id,
                dataset=dataset,
                split_name=split_name,
                sample=sample,
                experiment=experiment,
                protocol=protocol,
                endpoint=endpoint,
                network_budget=network_budget,
            )
            review_future = executor.submit(
                _review_rows,
                stage_rows=stage_rows,
                run_id=run_id,
                dataset=dataset,
                split_name=split_name,
                sample=sample,
                experiment=experiment,
                protocol=protocol,
                endpoint=endpoint,
                network_budget=network_budget,
            )
            resample_rows = resample_future.result()
            reviewer_rows, reviewer_metadata = review_future.result()

    physical_rows = [*stage_rows, *resample_rows, *reviewer_rows]
    predictions: list[dict[str, Any]] = []
    candidates = set(stage.vote_counts)

    method_answers: dict[str, tuple[str, str, list[dict[str, Any]], str]] = {}
    cot = homogeneous_stage_decision(
        stage_rows[:1], dataset=dataset, seed=experiment.global_seed, sample_id=sample.sample_id, purpose="cot1"
    )
    sc3 = homogeneous_stage_decision(
        stage_rows[:3], dataset=dataset, seed=experiment.global_seed, sample_id=sample.sample_id, purpose="sc3"
    )
    method_answers["cot_1"] = (cot.anchor_key, cot.anchor_answer, stage_rows[:1], cot.resolver)
    method_answers["sc_3"] = (sc3.anchor_key, sc3.anchor_answer, stage_rows[:3], sc3.resolver)
    method_answers["sc_5"] = (stage.anchor_key, stage.anchor_answer, stage_rows, stage.resolver)

    if triggered:
        adaptive_key, adaptive_answer, _, adaptive_resolver = class_majority(
            [*stage_rows, *resample_rows],
            dataset=dataset,
            seed=experiment.global_seed,
            sample_id=sample.sample_id,
            purpose="adaptive_sc8",
            fallback_key=stage.anchor_key,
            fallback_answer=stage.anchor_answer,
        )
        conditional_key, conditional_answer, _, conditional_resolver = class_majority(
            resample_rows,
            dataset=dataset,
            seed=experiment.global_seed,
            sample_id=sample.sample_id,
            purpose="conditional_resample3",
            fallback_key=stage.anchor_key,
            fallback_answer=stage.anchor_answer,
        )
    else:
        adaptive_key = conditional_key = stage.anchor_key
        adaptive_answer = conditional_answer = stage.anchor_answer
        adaptive_resolver = conditional_resolver = "no_answer_class_disagreement"
    method_answers["adaptive_sc_8"] = (
        adaptive_key,
        adaptive_answer,
        [*stage_rows, *resample_rows],
        adaptive_resolver,
    )
    method_answers["conditional_resample_3"] = (
        conditional_key,
        conditional_answer,
        [*stage_rows, *resample_rows],
        conditional_resolver,
    )

    for method, required, used_reviewers in (
        ("blind_gsa_1", 1, reviewer_rows[:1]),
        ("blind_gsa_quorum_3", protocol.quorum_size, reviewer_rows),
        ("hsgsa_unanimous_3", protocol.unanimity_size, reviewer_rows),
    ):
        if triggered:
            selected_key, resolver = reviewer_selected_key(
                used_reviewers, anchor_key=stage.anchor_key, candidate_keys=candidates, required=required
            )
        else:
            selected_key, resolver = stage.anchor_key, "no_answer_class_disagreement"
        selected_answer = stage.answer_by_key.get(selected_key, stage.anchor_answer)
        method_answers[method] = (selected_key, selected_answer, [*stage_rows, *used_reviewers], resolver)

    for method in active_methods:
        if method not in method_answers:
            continue
        key, answer, logical_rows, resolver = method_answers[method]
        predictions.append(
            _hsgsa_prediction(
                run_id=run_id,
                dataset=dataset,
                split_name=split_name,
                sample=sample,
                method_name=method,
                model_name=endpoint.backbone.name,
                final_key=key,
                prediction=answer,
                stage=stage,
                stage_rows=stage_rows,
                resample_rows=resample_rows,
                reviewer_rows=reviewer_rows,
                reviewer_metadata=reviewer_metadata,
                logical_rows=logical_rows,
                physical_rows=physical_rows,
                triggered=triggered,
                resolver=resolver,
            )
        )

    messages = [_message_row(row) for row in reviewer_rows]
    decision = {
        "run_id": run_id,
        "dataset": dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "policy_name": "homogeneous_support_blind_sgsa_v5",
        "triggered": triggered,
        "anchor_answer": stage.anchor_answer,
        "anchor_answer_class_key": stage.anchor_key,
        "answer_class_vote_counts": stage.vote_counts,
        "disagreement_pattern": stage.disagreement_pattern,
        "anchor_resolver": stage.resolver,
        "valid_trace_count": stage.valid_trace_count,
        "reviewers": reviewer_metadata,
        "override_accepted": bool(
            predictions
            and next(
                (row.get("override_accepted") for row in predictions if row["method_name"] == "hsgsa_unanimous_3"),
                False,
            )
        ),
    }
    return physical_rows, messages, [decision], predictions


def _solve_rows(
    *, count, seed_offset, method_name, method_type, role, run_id, dataset, split_name, sample,
    experiment, protocol, endpoint, network_budget
):
    def execute(index: int) -> dict[str, Any]:
        reservation = network_budget.reserve(10)
        try:
            row = _execute_free_text_turn(
                run_id=run_id,
                dataset=dataset,
                split_name=split_name,
                sample=sample,
                method_name=method_name,
                method_type=method_type,
                round_index=0 if seed_offset == 0 else 1,
                agent_id=index + 1,
                role=role,
                messages=build_cot_messages(sample, index + 1, "single_agent_free_text_v1"),
                endpoint=endpoint,
                temperature=protocol.stage_temperature,
                top_p=protocol.top_p,
                seed=experiment.global_seed + seed_offset + index,
                max_tokens=protocol.stage_max_tokens,
            )
        except Exception:
            network_budget.settle(reservation, 0)
            raise
        network_budget.settle(reservation, int(row.get("network_attempt_count") or 0))
        answer = normalized_answer(row)
        row["answer_class_key"] = answer_class_key(dataset, answer) if answer else ""
        row["stage_name"] = "stage_a" if seed_offset == 0 else "resample"
        return row

    with ThreadPoolExecutor(max_workers=count) as executor:
        return list(executor.map(execute, range(count)))


def _review_rows(
    *, stage_rows, run_id, dataset, split_name, sample, experiment, protocol, endpoint, network_budget
):
    prepared = []
    for index in range(protocol.reviewer_count):
        board, label_to_key, label_to_answer, hashes = build_support_blind_board(
            stage_rows,
            dataset=dataset,
            seed=experiment.global_seed,
            sample_id=sample.sample_id,
            reviewer_index=index + 1,
            trace_max_chars=protocol.trace_max_chars,
            board_max_chars=protocol.board_max_chars,
        )
        prepared.append((index, board, label_to_key, label_to_answer, hashes))

    def execute(values):
        index, board, label_to_key, label_to_answer, hashes = values
        row = _execute_reviewer_turn(
            run_id=run_id,
            dataset=dataset,
            split_name=split_name,
            sample=sample,
            reviewer_index=index + 1,
            messages=build_blind_reviewer_messages(sample, board),
            label_to_key=label_to_key,
            label_to_answer=label_to_answer,
            endpoint=endpoint,
            temperature=protocol.reviewer_temperature,
            top_p=protocol.top_p,
            seed=experiment.global_seed + 20_000 + index,
            max_tokens=protocol.reviewer_max_tokens,
            network_budget=network_budget,
        )
        metadata = {
            "reviewer_index": index + 1,
            "label_to_answer_class_key": label_to_key,
            "representative_trace_hashes": hashes,
            "board_hash": row.get("prompt_hash"),
            "pick": (row.get("validated_output") or {}).get("pick"),
            "picked_answer_class_key": (row.get("validated_output") or {}).get("picked_answer_class_key"),
            "output_status": row.get("output_status"),
        }
        return row, metadata

    with ThreadPoolExecutor(max_workers=protocol.reviewer_count) as executor:
        results = list(executor.map(execute, prepared))
    return [row for row, _ in results], [metadata for _, metadata in results]


def parse_blind_reviewer_output(
    raw_text: str, *, label_to_key: dict[str, str], label_to_answer: dict[str, str], dataset: str
) -> dict[str, Any]:
    pick_match = re.search(r"(?im)^\s*PICK\s*:\s*(.+?)\s*$", str(raw_text or ""))
    if not pick_match:
        raise ValueError("missing PICK line")
    raw_pick = pick_match.group(1).strip().upper()
    if raw_pick == "ABSTAIN":
        pick = "ABSTAIN"
    else:
        # Extract a single uppercase letter from variants like "A", "Candidate A", "(A)", "Option A"
        letter_match = re.search(r"\b([A-Z])\b", raw_pick)
        pick = letter_match.group(1) if letter_match else raw_pick
    final_match = re.search(r"(?im)^\s*FINAL_ANSWER\s*:\s*(.*?)\s*$", str(raw_text or ""))
    final_answer = str(final_match.group(1) if final_match else "").strip()
    if pick == "ABSTAIN":
        return {
            "pick": "ABSTAIN",
            "picked_answer_class_key": "",
            "picked_answer": "",
            "generated_final_answer": normalize_prediction(dataset, final_answer) if final_answer else "",
        }
    if pick not in label_to_key:
        # Label hallucinated or out of range — treat as abstention
        return {
            "pick": "ABSTAIN",
            "picked_answer_class_key": "",
            "picked_answer": "",
            "generated_final_answer": normalize_prediction(dataset, final_answer) if final_answer else "",
        }
    return {
        "pick": pick,
        "picked_answer_class_key": label_to_key[pick],
        "picked_answer": label_to_answer[pick],
        "generated_final_answer": normalize_prediction(dataset, final_answer) if final_answer else "",
    }


def _execute_reviewer_turn(
    *, run_id, dataset, split_name, sample, reviewer_index, messages, label_to_key, label_to_answer,
    endpoint, temperature, top_p, seed, max_tokens, network_budget
):
    reservation = network_budget.reserve(5)
    try:
        request = execute_cached_request(
            backbone=endpoint.backbone,
            provider=endpoint.provider,
            cache=endpoint.cache,
            throttle=endpoint.throttle,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            use_response_format=False,
            max_tokens=max_tokens,
        )
    except Exception:
        network_budget.settle(reservation, 0)
        raise
    response = request.response_payload
    network_attempts = 0 if request.cache_hit else int(response.get("network_attempt_count") or 1)
    network_budget.settle(reservation, network_attempts)
    raw_text = str(response.get("assistant_text") or "")
    validated: dict[str, Any] = {}
    parse_error = None
    if not request.request_error and str(response.get("finish_reason") or "") != "content_filter":
        try:
            validated = parse_blind_reviewer_output(
                raw_text, label_to_key=label_to_key, label_to_answer=label_to_answer, dataset=dataset
            )
        except ValueError as exc:
            parse_error = str(exc)
            endpoint.cache.delete(request.cache_key)
    abstention = str(response.get("finish_reason") or "") == "content_filter"
    status = "request_fail" if request.request_error else ("abstain" if abstention or not validated else "ok")
    usage = request.usage
    picked_answer = str(validated.get("picked_answer") or "")
    return {
        "run_id": run_id,
        "dataset": dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "method_name": "hsgsa_blind_reviewer_shared",
        "method_type": "support_blind_review",
        "round_index": 1,
        "agent_id": reviewer_index,
        "role": "blind_reviewer",
        "model_lineage": endpoint.lineage,
        "model_name": endpoint.backbone.name,
        "prompt_hash": request.prompt_hash,
        "cache_key": request.cache_key,
        "prediction": picked_answer,
        "normalized_answer": picked_answer,
        "answer_class_key": str(validated.get("picked_answer_class_key") or ""),
        "score": None,
        "output_status": status,
        "provider_abstention": abstention,
        "prompt_tokens": float(usage.get("prompt_tokens") or 0),
        "completion_tokens": float(usage.get("completion_tokens") or 0),
        "total_tokens": float(usage.get("total_tokens") or 0),
        "latency_ms": float(response.get("latency_ms") or 0),
        "cache_hit": request.cache_hit,
        "request_error": request.request_error,
        "request_status": "request_fail" if request.request_error else "ok",
        "raw_finish_reason": response.get("finish_reason"),
        "output_protocol": HSGSA_REVIEW_SCHEMA_VERSION,
        "protocol_parse_status": "ok" if validated else ("abstain" if abstention else "failed"),
        "protocol_parse_error": parse_error,
        "reason_present": bool(
            re.sub(r"(?im)^\s*(PICK|FINAL_ANSWER)\s*:.*$", "", raw_text).strip()
        ),
        "request_count": 1,
        "cache_request_count": int(request.cache_hit),
        "network_request_count": network_attempts,
        "logical_call_count": 1,
        "network_attempt_count": network_attempts,
        "protocol_attempt_count": 1,
        "request_started_at": response.get("request_started_at"),
        "request_started_at_events": list(response.get("request_started_at_events") or []),
        "payload": request.payload,
        "assistant_text": raw_text,
        "provider_reasoning_text": str(response.get("provider_reasoning_text") or ""),
        "validated_output": validated,
    }


def _hsgsa_prediction(
    *, run_id, dataset, split_name, sample, method_name, model_name, final_key, prediction, stage,
    stage_rows, resample_rows, reviewer_rows, reviewer_metadata, logical_rows, physical_rows, triggered, resolver
):
    score = score_prediction(dataset, prediction, sample.reference_answer) if prediction else 0.0
    initial_score = score_prediction(dataset, stage.anchor_answer, sample.reference_answer) if stage.anchor_answer else 0.0
    override = bool(final_key and final_key != stage.anchor_key)
    reviewer_picks = [str((row.get("validated_output") or {}).get("pick") or "ABSTAIN") for row in reviewer_rows]
    generated_answers = [
        str((row.get("validated_output") or {}).get("generated_final_answer") or "") for row in reviewer_rows
    ]
    return {
        "run_id": run_id,
        "dataset": dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "task": sample.metadata.get("task"),
        "method_name": method_name,
        "method_type": "homogeneous_support_blind_sgsa",
        "model_name": model_name,
        "prediction": prediction,
        "answer_class_key": final_key,
        "gold": sample.reference_answer,
        "score": score,
        "initial_vote_prediction": stage.anchor_answer,
        "initial_answer_class_key": stage.anchor_key,
        "initial_vote_score": initial_score,
        "initial_vote_counts": stage.vote_counts,
        "initial_consensus": not stage.triggered,
        "final_vote_prediction": prediction,
        "final_vote_score": score,
        "final_vote_counts": {final_key: 1} if final_key else {},
        "prompt_tokens_per_question": _sum(logical_rows, "prompt_tokens"),
        "completion_tokens_per_question": _sum(logical_rows, "completion_tokens"),
        "total_tokens_per_question": _sum(logical_rows, "total_tokens"),
        "latency_ms_per_question": _sum(logical_rows, "latency_ms"),
        "stage_a_prompt_tokens": _sum(stage_rows, "prompt_tokens"),
        "stage_a_completion_tokens": _sum(stage_rows, "completion_tokens"),
        "stage_a_total_tokens": _sum(stage_rows, "total_tokens"),
        "resample_prompt_tokens": _sum(resample_rows, "prompt_tokens"),
        "resample_completion_tokens": _sum(resample_rows, "completion_tokens"),
        "resample_total_tokens": _sum(resample_rows, "total_tokens"),
        "reviewer_prompt_tokens": _sum(reviewer_rows, "prompt_tokens"),
        "reviewer_completion_tokens": _sum(reviewer_rows, "completion_tokens"),
        "reviewer_total_tokens": _sum(reviewer_rows, "total_tokens"),
        "calls_per_question": len(logical_rows),
        "logical_calls_per_question": len(logical_rows),
        "network_attempts_per_question": sum(int(row.get("network_attempt_count") or 0) for row in logical_rows),
        "shared_physical_logical_calls_per_question": len(physical_rows),
        "shared_physical_network_attempts_per_question": sum(
            int(row.get("network_attempt_count") or 0) for row in physical_rows
        ),
        "shared_physical_request_failures_per_question": sum(
            row.get("output_status") == "request_fail" for row in physical_rows
        ),
        "shared_physical_protocol_failures_per_question": sum(
            row.get("protocol_parse_status") == "failed" for row in physical_rows
        ),
        "valid_trajectories_per_question": stage.valid_trace_count,
        "provider_abstentions_per_question": sum(bool(row.get("provider_abstention")) for row in logical_rows),
        "protocol_failures_per_question": sum(row.get("protocol_parse_status") == "failed" for row in logical_rows),
        "request_failures_per_question": sum(row.get("output_status") == "request_fail" for row in logical_rows),
        "reviewer_calls_per_question": len(reviewer_rows),
        "reviewer_valid_picks_per_question": sum(row.get("output_status") == "ok" for row in reviewer_rows),
        "reviewer_protocol_failures_per_question": sum(
            row.get("protocol_parse_status") == "failed" for row in reviewer_rows
        ),
        "initial_disagreement": stage.triggered,
        "triggered": triggered,
        "router_reasons": ["answer_class_disagreement"] if triggered else [],
        "resolver": resolver,
        "vote_flipped": override,
        "override_accepted": override,
        "corrected_by_debate": initial_score < 1 and score == 1,
        "harmed_by_debate": initial_score == 1 and score < 1,
        "unchanged_correct": not override and score == 1,
        "unchanged_wrong": not override and score < 1,
        "candidate_answers": sorted(stage.answer_by_key.values()),
        "candidate_answer_class_keys": sorted(stage.vote_counts),
        "candidate_oracle_correct": any(
            score_prediction(dataset, answer, sample.reference_answer) == 1.0
            for answer in stage.answer_by_key.values()
        ),
        "reviewer_picks": reviewer_picks,
        "reviewer_label_mappings": [row["label_to_answer_class_key"] for row in reviewer_metadata],
        "representative_trace_hashes": [row["representative_trace_hashes"] for row in reviewer_metadata],
        "reviewer_generated_answers_shadow": generated_answers,
        "novel_answer": final_key not in stage.vote_counts,
        "stage_a_prompt_hashes": [str(row.get("prompt_hash") or "") for row in stage_rows],
    }


def _message_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": row["run_id"],
        "dataset": row["dataset"],
        "split": row["split"],
        "sample_id": row["sample_id"],
        "method_name": row["method_name"],
        "round_index": row["round_index"],
        "sender_agent_id": row["agent_id"],
        "recipient_agent_id": 0,
        "message_kind": "blind_review",
        "sender_answer": row.get("normalized_answer", ""),
        "sender_reasoning": row.get("assistant_text", ""),
    }


def _sum(rows: list[dict[str, Any]], key: str) -> float:
    return float(sum(float(row.get(key) or 0) for row in rows))


def estimate_hsgsa_work(
    experiment: MadInnovationExperimentConfig, phase_name: str, benchmarks, active_methods: list[str]
) -> tuple[int, int]:
    from research_experiments.families.risk_controlled_trace_mad.run.sample import (
        load_selected_samples,
        resolve_split_name,
    )

    sample_total = sum(
        len(load_selected_samples(benchmark, resolve_split_name(experiment, phase_name, benchmark.slug)))
        for benchmark in benchmarks
    )
    return sample_total * 11, sample_total * len(active_methods)
