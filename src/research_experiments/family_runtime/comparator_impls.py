"""共享 comparator 实现层。"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.core.data.evaluation import aggregate_majority, score_prediction
from research_experiments.family_runtime.vanilla_mad_prompting import (
    CONSISTENT_FREE_TEXT_PROMPT_VERSION,
    SUPPORTED_SHARED_PROMPT_VERSIONS,
)
from research_experiments.family_runtime.vanilla_mad_prompting import (
    build_debate_messages as build_standard_mad_debate_messages,
)
from research_experiments.family_runtime.vanilla_mad_prompting import (
    build_initial_messages as build_standard_mad_initial_messages,
)

ExecuteVanillaMadTurnFn = Callable[..., dict[str, Any]]
BuildDebateRowFn = Callable[[dict[str, Any], int, int], dict[str, Any]]


def assert_standard_vanilla_mad_prompt_version(prompt_version: str, *, method_name: str) -> None:
    """约束标准 MAD comparator 只能使用共享 controlled prompt 版本。"""
    if prompt_version not in SUPPORTED_SHARED_PROMPT_VERSIONS:
        raise ValueError(
            f"Standard vanilla MAD comparator {method_name} must use prompt_version="
            f"one of {SUPPORTED_SHARED_PROMPT_VERSIONS}, got {prompt_version}."
        )


def build_stage_a_mv3_prediction(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    question_preview: str,
    model_name: str,
    stage_a_turns: list[dict[str, Any]],
    stage_a_vote: str,
    stage_a_score: float,
    stage_a_trace_hash: str | None,
    vote_counts: dict[str, int],
    method_kind: str,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造统一的 Stage A 无通信投票复用 `mv_3`。"""

    prompt_tokens = sum(float(row["prompt_tokens"]) for row in stage_a_turns)
    completion_tokens = sum(float(row["completion_tokens"]) for row in stage_a_turns)
    total_tokens = sum(float(row["total_tokens"]) for row in stage_a_turns)
    latency_ms = sum(float(row["latency_ms"]) for row in stage_a_turns)
    row = {
        "run_id": run_id,
        "dataset": dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "question_preview": question_preview,
        "method_name": "mv_3",
        "display_name": "mv_3",
        "method_kind": method_kind,
        "method_family": "shared_majority_vote",
        "model_name": model_name,
        "prediction": stage_a_vote,
        "gold": sample.reference_answer,
        "score": stage_a_score,
        "prompt_tokens_per_question": prompt_tokens,
        "completion_tokens_per_question": completion_tokens,
        "total_tokens_per_question": total_tokens,
        "communication_tokens_per_question": 0.0,
        "latency_ms_per_question": latency_ms,
        "communication_latency_ms_per_question": 0.0,
        "calls_per_question": float(len(stage_a_turns)),
        "stage_a_prediction": stage_a_vote,
        "stage_a_score": stage_a_score,
        "stage_a_trace_hash": stage_a_trace_hash,
        "stage_a_tokens_per_question": total_tokens,
        "vote_counts": vote_counts,
    }
    if extra_fields:
        row.update(extra_fields)
    return row


def run_shared_vanilla_mad_rounds(
    *,
    sample: DatasetSample,
    run_id: str,
    dataset: str,
    split_name: str,
    method_name: str,
    agent_count: int,
    debate_rounds: int,
    initial_temperature: float,
    debate_temperature: float,
    top_p: float,
    global_seed: int,
    execute_turn: ExecuteVanillaMadTurnFn,
    build_debate_row: BuildDebateRowFn,
    prompt_version: str = CONSISTENT_FREE_TEXT_PROMPT_VERSION,
    initial_turns: list[dict[str, Any]] | None = None,
    include_initial_turns_in_output: bool = True,
) -> dict[str, Any]:
    """运行共享 vanilla MAD 固定轮数核心。"""
    assert_standard_vanilla_mad_prompt_version(prompt_version, method_name=method_name)

    generated_initial_turns: list[dict[str, Any]] = []
    if initial_turns is None:
        for agent_id in range(1, agent_count + 1):
            messages = build_standard_mad_initial_messages(sample, agent_id, prompt_version=prompt_version)
            generated_initial_turns.append(
                execute_turn(
                    round_index=0,
                    agent_id=agent_id,
                    role="initial",
                    visible_peer_count=0,
                    messages=messages,
                    temperature=initial_temperature,
                    top_p=top_p,
                    seed=global_seed + agent_id - 1,
                    prompt_version=prompt_version,
                )
            )
        initial_turns = generated_initial_turns

    turn_rows: list[dict[str, Any]] = list(initial_turns) if include_initial_turns_in_output else []
    debate_rows: list[dict[str, Any]] = []
    previous_round = list(initial_turns)
    final_round_turns = list(initial_turns)
    stage_a_vote, stage_a_vote_counts = aggregate_majority(row["normalized_answer"] for row in initial_turns)

    for round_index in range(1, debate_rounds + 1):
        current_round: list[dict[str, Any]] = []
        for recipient_id in range(1, agent_count + 1):
            recipient_previous = previous_round[recipient_id - 1]
            peer_messages: list[dict[str, str]] = []
            for sender in previous_round:
                if int(sender["agent_id"]) == recipient_id:
                    continue
                peer_messages.append(
                    {
                        "agent": f"agent_{sender['agent_id']}",
                        "answer": str(sender["validated_output"].get("final_answer", "")).strip(),
                        "reasoning": _turn_reasoning_for_prompt(sender),
                        "response_text": "",
                    }
                )
                debate_rows.append(build_debate_row(sender, recipient_id, round_index))
            messages = build_standard_mad_debate_messages(
                sample=sample,
                agent_id=recipient_id,
                round_index=round_index,
                previous_reasoning=_turn_reasoning_for_prompt(recipient_previous),
                previous_answer=str(recipient_previous["validated_output"].get("final_answer", "")).strip(),
                peer_messages=peer_messages,
                prompt_version=prompt_version,
                stage_a_majority_answer=stage_a_vote,
                stage_a_vote_counts=stage_a_vote_counts,
            )
            current_round.append(
                execute_turn(
                    round_index=round_index,
                    agent_id=recipient_id,
                    role="debate",
                    visible_peer_count=len(peer_messages),
                    messages=messages,
                    temperature=debate_temperature,
                    top_p=top_p,
                    seed=global_seed + recipient_id + round_index * 100,
                    prompt_version=prompt_version,
                )
            )
        turn_rows.extend(current_round)
        previous_round = current_round
        final_round_turns = current_round

    summary = summarize_shared_vanilla_mad_turn_rows(
        turn_rows=turn_rows,
        dataset=dataset,
        gold=sample.reference_answer,
        debate_rounds=debate_rounds,
        agent_count=agent_count,
    )
    return {
        "turn_rows": turn_rows,
        "debate_rows": debate_rows,
        "initial_turns": list(initial_turns),
        "final_round_turns": final_round_turns,
        **summary,
    }


def build_shared_vanilla_mad_prediction(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    method_name: str,
    method_type: str,
    model_name: str,
    result: dict[str, Any],
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """把共享 vanilla MAD 结果转换为标准题级预测行。"""

    row = {
        "run_id": run_id,
        "dataset": dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "method_name": method_name,
        "method_type": method_type,
        "model_name": model_name,
        "prediction": result["final_vote_prediction"],
        "gold": sample.reference_answer,
        "score": result["final_vote_score"],
        "initial_vote_prediction": result["initial_vote_prediction"],
        "initial_vote_score": result["initial_vote_score"],
        "initial_vote_counts": result["initial_vote_counts"],
        "initial_consensus": result["initial_consensus"],
        "stage_a_majority_prediction": result["initial_vote_prediction"],
        "stage_a_majority_score": result["initial_vote_score"],
        "stage_a_majority_counts": result["initial_vote_counts"],
        "debate_proposed_prediction": result["debate_proposed_prediction"],
        "debate_proposed_score": result["debate_proposed_score"],
        "debate_proposed_counts": result["debate_proposed_counts"],
        "debate_override_applied": result["debate_override_applied"],
        "debate_override_reason": result["debate_override_reason"],
        "final_vote_prediction": result["final_vote_prediction"],
        "final_vote_score": result["final_vote_score"],
        "final_vote_counts": result["final_vote_counts"],
        "prompt_tokens_per_question": result["prompt_tokens_per_question"],
        "completion_tokens_per_question": result["completion_tokens_per_question"],
        "total_tokens_per_question": result["total_tokens_per_question"],
        "latency_ms_per_question": result["latency_ms_per_question"],
        "initial_prompt_tokens_per_question": result["initial_prompt_tokens_per_question"],
        "initial_completion_tokens_per_question": result["initial_completion_tokens_per_question"],
        "initial_total_tokens_per_question": result["initial_total_tokens_per_question"],
        "initial_latency_ms_per_question": result["initial_latency_ms_per_question"],
        "debate_prompt_tokens_per_question": result["debate_prompt_tokens_per_question"],
        "debate_completion_tokens_per_question": result["debate_completion_tokens_per_question"],
        "debate_total_tokens_per_question": result["debate_total_tokens_per_question"],
        "debate_latency_ms_per_question": result["debate_latency_ms_per_question"],
        "calls_per_question": result["calls_per_question"],
        "debate_rounds": result["debate_rounds"],
        "agent_count": result["agent_count"],
        "final_consensus": result["final_consensus"],
        "initial_disagreement": result["initial_disagreement"],
        "vote_flipped": result["vote_flipped"],
        "corrected_by_debate": result["corrected_by_debate"],
        "harmed_by_debate": result["harmed_by_debate"],
        "unchanged_correct": result["unchanged_correct"],
        "unchanged_wrong": result["unchanged_wrong"],
        "vote_counts": result["final_vote_counts"],
        "protocol_failures_per_question": result.get("protocol_failures_per_question", 0),
        "reason_missing_turns_per_question": result.get("reason_missing_turns_per_question", 0),
    }
    if extra_fields:
        row.update(extra_fields)
    return row


def summarize_shared_vanilla_mad_turn_rows(
    *,
    turn_rows: list[dict[str, Any]],
    dataset: str,
    gold: str,
    debate_rounds: int,
    agent_count: int,
) -> dict[str, Any]:
    initial_turns = [row for row in turn_rows if row.get("role") == "initial"]
    debate_turns = [row for row in turn_rows if row.get("role") == "debate"]
    final_round_turns = (
        [row for row in debate_turns if int(row.get("round_index") or 0) == debate_rounds]
        if debate_rounds > 0
        else list(initial_turns)
    )

    initial_answers = [row["normalized_answer"] for row in initial_turns]
    debate_proposed_answers = [row["normalized_answer"] for row in final_round_turns]
    initial_vote, initial_vote_counts = aggregate_majority(initial_answers)
    debate_proposed_vote, debate_proposed_counts = aggregate_majority(debate_proposed_answers)
    final_vote, final_vote_counts, override_applied, override_reason = _resolve_conservative_mad_final_vote(
        initial_turns=initial_turns,
        final_round_turns=final_round_turns,
        initial_vote=initial_vote,
        initial_vote_counts=initial_vote_counts,
        debate_proposed_vote=debate_proposed_vote,
        debate_proposed_counts=debate_proposed_counts,
        debate_rounds=debate_rounds,
        agent_count=agent_count,
    )
    initial_vote_score = score_prediction(dataset, initial_vote, gold)
    debate_proposed_score = score_prediction(dataset, debate_proposed_vote, gold)
    final_vote_score = score_prediction(dataset, final_vote, gold)
    initial_consensus = len(set(initial_answers)) == 1
    final_consensus = len(final_vote_counts) == 1
    initial_disagreement = len(set(initial_answers)) > 1

    initial_prompt_tokens = sum(float(row["prompt_tokens"]) for row in initial_turns)
    initial_completion_tokens = sum(float(row["completion_tokens"]) for row in initial_turns)
    initial_total_tokens = sum(float(row.get("raw_total_tokens") or row["total_tokens"]) for row in initial_turns)
    initial_latency = sum(float(row.get("raw_latency_ms") or row["latency_ms"]) for row in initial_turns)
    debate_prompt_tokens = sum(float(row["prompt_tokens"]) for row in debate_turns)
    debate_completion_tokens = sum(float(row["completion_tokens"]) for row in debate_turns)
    debate_total_tokens = sum(float(row.get("raw_total_tokens") or row["total_tokens"]) for row in debate_turns)
    debate_latency = sum(float(row.get("raw_latency_ms") or row["latency_ms"]) for row in debate_turns)
    total_prompt_tokens = sum(float(row["prompt_tokens"]) for row in turn_rows)
    total_completion_tokens = sum(float(row["completion_tokens"]) for row in turn_rows)
    total_tokens = sum(float(row["total_tokens"]) for row in turn_rows)
    total_latency = sum(float(row["latency_ms"]) for row in turn_rows)
    return {
        "initial_vote_prediction": initial_vote,
        "initial_vote_score": initial_vote_score,
        "initial_vote_counts": initial_vote_counts,
        "initial_consensus": initial_consensus,
        "debate_proposed_prediction": debate_proposed_vote,
        "debate_proposed_score": debate_proposed_score,
        "debate_proposed_counts": debate_proposed_counts,
        "debate_override_applied": override_applied,
        "debate_override_reason": override_reason,
        "final_vote_prediction": final_vote,
        "final_vote_score": final_vote_score,
        "final_vote_counts": final_vote_counts,
        "final_consensus": final_consensus,
        "initial_disagreement": initial_disagreement,
        "prompt_tokens_per_question": total_prompt_tokens,
        "completion_tokens_per_question": total_completion_tokens,
        "total_tokens_per_question": total_tokens,
        "latency_ms_per_question": total_latency,
        "initial_prompt_tokens_per_question": initial_prompt_tokens,
        "initial_completion_tokens_per_question": initial_completion_tokens,
        "initial_total_tokens_per_question": initial_total_tokens,
        "initial_latency_ms_per_question": initial_latency,
        "debate_prompt_tokens_per_question": debate_prompt_tokens,
        "debate_completion_tokens_per_question": debate_completion_tokens,
        "debate_total_tokens_per_question": debate_total_tokens,
        "debate_latency_ms_per_question": debate_latency,
        "calls_per_question": sum(float(row.get("request_count") or 1.0) for row in turn_rows),
        "debate_rounds": debate_rounds,
        "agent_count": agent_count,
        "vote_flipped": override_applied and initial_vote != final_vote,
        "corrected_by_debate": initial_vote_score < 1.0 and final_vote_score == 1.0,
        "harmed_by_debate": initial_vote_score == 1.0 and final_vote_score < 1.0,
        "unchanged_correct": initial_vote_score == 1.0 and final_vote_score == 1.0,
        "unchanged_wrong": initial_vote_score < 1.0 and final_vote_score < 1.0,
        "protocol_failures_per_question": sum(1 for row in turn_rows if row.get("protocol_parse_status") == "failed"),
        "reason_missing_turns_per_question": sum(1 for row in turn_rows if not row.get("reason_present")),
    }


def _resolve_conservative_mad_final_vote(
    *,
    initial_turns: list[dict[str, Any]],
    final_round_turns: list[dict[str, Any]],
    initial_vote: str,
    initial_vote_counts: dict[str, int],
    debate_proposed_vote: str,
    debate_proposed_counts: dict[str, int],
    debate_rounds: int,
    agent_count: int,
) -> tuple[str, dict[str, int], bool, str]:
    if debate_rounds <= 0:
        return initial_vote, initial_vote_counts, False, "keep_stage_a_majority_no_debate_rounds"
    if debate_proposed_vote == initial_vote:
        return initial_vote, initial_vote_counts, False, "keep_stage_a_majority_no_debate_change"

    rejection_reason = _conservative_override_rejection_reason(
        initial_turns=initial_turns,
        final_round_turns=final_round_turns,
        initial_vote=initial_vote,
        initial_vote_counts=initial_vote_counts,
        debate_proposed_vote=debate_proposed_vote,
        debate_proposed_counts=debate_proposed_counts,
        agent_count=agent_count,
    )
    if rejection_reason is not None:
        return initial_vote, initial_vote_counts, False, rejection_reason
    return (
        debate_proposed_vote,
        debate_proposed_counts,
        True,
        "accepted_debate_majority_with_majority_error_certificate",
    )


def _conservative_override_rejection_reason(
    *,
    initial_turns: list[dict[str, Any]],
    final_round_turns: list[dict[str, Any]],
    initial_vote: str,
    initial_vote_counts: dict[str, int],
    debate_proposed_vote: str,
    debate_proposed_counts: dict[str, int],
    agent_count: int,
) -> str | None:
    if not str(debate_proposed_vote or "").strip():
        return "keep_stage_a_majority_empty_debate_candidate"

    candidate_support = int(debate_proposed_counts.get(debate_proposed_vote) or 0)
    if candidate_support < (agent_count // 2 + 1):
        return "keep_stage_a_majority_insufficient_debate_majority"

    candidate_rows = [
        row for row in final_round_turns if str(row.get("normalized_answer") or "").strip() == debate_proposed_vote
    ]
    if not candidate_rows:
        return "keep_stage_a_majority_empty_debate_candidate"
    if any(row.get("protocol_parse_status") == "failed" for row in candidate_rows):
        return "keep_stage_a_majority_protocol_failure"
    if any((row.get("validated_output") or {}).get("format_warning") for row in candidate_rows):
        return "keep_stage_a_majority_invalid_task_format"
    if any(not _turn_reasoning_for_prompt(row) for row in candidate_rows):
        return "keep_stage_a_majority_missing_reasoning"

    initial_by_agent = {_agent_id(row): str(row.get("normalized_answer") or "").strip() for row in initial_turns}
    switcher_rows = [
        row
        for row in candidate_rows
        if initial_by_agent.get(_agent_id(row)) == initial_vote
    ]
    initial_majority_support = int(initial_vote_counts.get(initial_vote) or 0)
    required_switchers = max(1, initial_majority_support // 2 + 1)
    if len(switcher_rows) < required_switchers:
        return "keep_stage_a_majority_insufficient_stage_a_switchers"
    certified_switchers = [row for row in switcher_rows if _has_majority_error_certificate(row)]
    if len(certified_switchers) < required_switchers:
        return "keep_stage_a_majority_missing_majority_error_certificate"
    return None


def _agent_id(row: dict[str, Any]) -> int:
    try:
        return int(row.get("agent_id") or 0)
    except (TypeError, ValueError):
        return 0


def _has_majority_error_certificate(row: dict[str, Any]) -> bool:
    value = str(
        row.get("majority_error")
        or (row.get("validated_output") or {}).get("majority_error")
        or ""
    )
    normalized = re.sub(r"[^a-z0-9]+", " ", value.strip().lower()).strip()
    return normalized not in {
        "",
        "none",
        "no",
        "n a",
        "na",
        "null",
        "unknown",
        "not applicable",
        "no material error",
        "no concrete error",
    }


def build_shared_output_protocol_diagnostics(
    turn_rows: list[dict[str, Any]],
    *,
    dataset_order: list[str],
    method_order: list[str],
) -> dict[str, Any]:
    dataset_rank = {name: index for index, name in enumerate(dataset_order)}
    method_rank = {name: index for index, name in enumerate(method_order)}
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in turn_rows:
        grouped.setdefault((str(row.get("dataset") or ""), str(row.get("method_name") or "")), []).append(row)

    rows: list[dict[str, Any]] = []
    for (dataset, method_name), items in grouped.items():
        rows.append(_output_protocol_diagnostic_row(dataset, method_name, items))

    overall_grouped: dict[str, list[dict[str, Any]]] = {}
    for row in turn_rows:
        overall_grouped.setdefault(str(row.get("method_name") or ""), []).append(row)
    for method_name, items in overall_grouped.items():
        rows.append(_output_protocol_diagnostic_row("overall", method_name, items))

    def _sort_key(row: dict[str, Any]) -> tuple[int, int]:
        dataset_idx = dataset_rank.get(str(row["dataset"]), len(dataset_order))
        if row["dataset"] == "overall":
            dataset_idx = len(dataset_order) + 1
        return dataset_idx, method_rank.get(str(row["method_name"]), 999)

    rows.sort(key=_sort_key)
    return {"rows": rows}


def _output_protocol_diagnostic_row(
    dataset: str,
    method_name: str,
    turn_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    turn_count = len(turn_rows)
    protocol_failures = sum(1 for row in turn_rows if row.get("protocol_parse_status") == "failed")
    reason_missing = sum(1 for row in turn_rows if not row.get("reason_present"))
    return {
        "dataset": dataset,
        "method_name": method_name,
        "turn_count": turn_count,
        "request_failure_count": sum(1 for row in turn_rows if row.get("request_status") == "request_fail"),
        "protocol_failure_count": protocol_failures,
        "protocol_failure_rate": _ratio_count(protocol_failures, turn_count),
        "reason_missing_count": reason_missing,
        "reason_missing_rate": _ratio_count(reason_missing, turn_count),
    }


def _turn_reasoning_for_prompt(turn_row: dict[str, Any]) -> str:
    return str(turn_row["validated_output"].get("reasoning", "")).strip()


def _ratio_count(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)
