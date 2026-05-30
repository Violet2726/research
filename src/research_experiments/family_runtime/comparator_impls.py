"""共享 comparator 实现层。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.core.data.evaluation import aggregate_majority, score_prediction
from research_experiments.family_runtime.vanilla_mad_prompting import (
    CONTROLLED_PROMPT_VERSION,
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
    if prompt_version != CONTROLLED_PROMPT_VERSION:
        raise ValueError(
            f"Standard vanilla MAD comparator {method_name} must use prompt_version="
            f"{CONTROLLED_PROMPT_VERSION}, got {prompt_version}."
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
    max_output_tokens: int,
    global_seed: int,
    execute_turn: ExecuteVanillaMadTurnFn,
    build_debate_row: BuildDebateRowFn,
    prompt_version: str = CONTROLLED_PROMPT_VERSION,
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
                    max_output_tokens=max_output_tokens,
                    seed=global_seed + agent_id,
                )
            )
        initial_turns = generated_initial_turns

    turn_rows: list[dict[str, Any]] = list(initial_turns) if include_initial_turns_in_output else []
    debate_rows: list[dict[str, Any]] = []
    previous_round = list(initial_turns)
    final_round_turns = list(initial_turns)

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
                        "reasoning": str(sender["validated_output"].get("reasoning", "")).strip(),
                    }
                )
                debate_rows.append(build_debate_row(sender, recipient_id, round_index))
            messages = build_standard_mad_debate_messages(
                sample=sample,
                agent_id=recipient_id,
                round_index=round_index,
                previous_reasoning=str(recipient_previous["validated_output"].get("reasoning", "")).strip(),
                previous_answer=str(recipient_previous["validated_output"].get("final_answer", "")).strip(),
                peer_messages=peer_messages,
                prompt_version=prompt_version,
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
                    max_output_tokens=max_output_tokens,
                    seed=global_seed + recipient_id + round_index * 100,
                )
            )
        turn_rows.extend(current_round)
        previous_round = current_round
        final_round_turns = current_round

    initial_answers = [row["normalized_answer"] for row in initial_turns]
    final_answers = [row["normalized_answer"] for row in final_round_turns]
    initial_vote, initial_vote_counts = aggregate_majority(initial_answers)
    final_vote, final_vote_counts = aggregate_majority(final_answers)
    initial_vote_score = score_prediction(dataset, initial_vote, sample.reference_answer)
    final_vote_score = score_prediction(dataset, final_vote, sample.reference_answer)
    initial_consensus = len(set(initial_answers)) == 1
    final_consensus = len(set(final_answers)) == 1
    initial_disagreement = len(set(initial_answers)) > 1
    debate_turns = [row for row in turn_rows if row["role"] == "debate"]
    initial_prompt_tokens = sum(float(row["prompt_tokens"]) for row in initial_turns)
    initial_completion_tokens = sum(float(row["completion_tokens"]) for row in initial_turns)
    initial_total_tokens = sum(float(row["total_tokens"]) for row in initial_turns)
    initial_latency = sum(float(row["latency_ms"]) for row in initial_turns)
    debate_prompt_tokens = sum(float(row["prompt_tokens"]) for row in debate_turns)
    debate_completion_tokens = sum(float(row["completion_tokens"]) for row in debate_turns)
    debate_total_tokens = sum(float(row["total_tokens"]) for row in debate_turns)
    debate_latency = sum(float(row["latency_ms"]) for row in debate_turns)
    return {
        "turn_rows": turn_rows,
        "debate_rows": debate_rows,
        "initial_turns": list(initial_turns),
        "final_round_turns": final_round_turns,
        "initial_vote_prediction": initial_vote,
        "initial_vote_score": initial_vote_score,
        "initial_vote_counts": initial_vote_counts,
        "initial_consensus": initial_consensus,
        "final_vote_prediction": final_vote,
        "final_vote_score": final_vote_score,
        "final_vote_counts": final_vote_counts,
        "final_consensus": final_consensus,
        "initial_disagreement": initial_disagreement,
        "prompt_tokens_per_question": initial_prompt_tokens + debate_prompt_tokens,
        "completion_tokens_per_question": initial_completion_tokens + debate_completion_tokens,
        "total_tokens_per_question": initial_total_tokens + debate_total_tokens,
        "latency_ms_per_question": initial_latency + debate_latency,
        "initial_prompt_tokens_per_question": initial_prompt_tokens,
        "initial_completion_tokens_per_question": initial_completion_tokens,
        "initial_total_tokens_per_question": initial_total_tokens,
        "initial_latency_ms_per_question": initial_latency,
        "debate_prompt_tokens_per_question": debate_prompt_tokens,
        "debate_completion_tokens_per_question": debate_completion_tokens,
        "debate_total_tokens_per_question": debate_total_tokens,
        "debate_latency_ms_per_question": debate_latency,
        "calls_per_question": agent_count * (1 + debate_rounds),
        "debate_rounds": debate_rounds,
        "agent_count": agent_count,
        "vote_flipped": initial_vote != final_vote,
        "corrected_by_debate": initial_vote_score < 1.0 and final_vote_score == 1.0,
        "harmed_by_debate": initial_vote_score == 1.0 and final_vote_score < 1.0,
        "unchanged_correct": initial_vote_score == 1.0 and final_vote_score == 1.0,
        "unchanged_wrong": initial_vote_score < 1.0 and final_vote_score < 1.0,
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
    }
    if extra_fields:
        row.update(extra_fields)
    return row

