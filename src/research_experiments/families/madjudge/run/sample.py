"""MADJudge 实验的样本级执行逻辑。

实现论文中的辩论流程：
- Phase 1：初始响应生成
- Phase 2：多轮辩论（含 Beta-Binomial KS 稳定性检测）
- Phase 3：Majority Vote 聚合
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any

from research_experiments.core.data.datasets import DatasetSample, load_split_ids, select_samples
from research_experiments.core.data.evaluation import normalize_prediction, score_prediction
from research_experiments.core.execution.runner_common import execute_cached_turn, run_indexed_batch
from research_experiments.families.madjudge.algorithms import (
    BetaBinomialParams,
    aggregate_majority_vote,
    check_stability,
    check_stability_batch,
    compute_majority_count,
)
from research_experiments.families.madjudge.config import (
    ExperimentSetup,
    MadJudgeExperimentConfig,
    ProtocolConfig,
    RosterConfig,
    load_protocol_config,
    load_roster_config,
)
from research_experiments.families.madjudge.prompts import (
    build_debate_messages,
    build_initial_messages,
)
from research_experiments.families.shared.common import resolve_phase_split_name, safe_mean, safe_ratio


@dataclass(frozen=True)
class AgentTurnRecord:
    """单个智能体在某一轮的执行记录。"""

    run_id: str
    dataset: str
    split: str
    sample_id: str
    method_name: str
    method_type: str
    round_index: int
    agent_id: int
    role: str
    prompt_hash: str
    prediction: str
    output_status: str
    prompt_tokens: float
    completion_tokens: float
    total_tokens: float
    latency_ms: float
    cache_hit: bool
    request_error: str | None
    visible_peer_count: int
    payload: dict[str, Any]
    assistant_text: str
    provider_reasoning_text: str
    validated_output: dict[str, Any]


@dataclass(frozen=True)
class DebateMessageRecord:
    """一条显式 debate 可见消息。"""

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


@dataclass(frozen=True)
class FinalPredictionRecord:
    """某题在某种方法下的最终预测记录。"""

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
    initial_vote_prediction: str
    initial_vote_score: float
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
    actual_debate_rounds: int
    agent_count: int
    final_consensus: bool
    initial_disagreement: bool
    vote_flipped: bool
    corrected_by_debate: bool
    harmed_by_debate: bool
    unchanged_correct: bool
    unchanged_wrong: bool
    # 稳定性检测相关
    ks_statistic_final: float
    stable_rounds: int


def _run_madjudge_sample(
    sample: DatasetSample,
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    setup: ExperimentSetup,
    protocol: ProtocolConfig,
    roster: RosterConfig,
    backbone,
    provider,
    cache,
    limiter,
    global_seed: int,
    prompt_version: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """运行单个样本上的 MADJudge 协议。"""
    turn_rows: list[dict[str, Any]] = []
    debate_rows: list[dict[str, Any]] = []
    method_type = protocol.method_type

    # 构建 agent_id → profile 查找表
    profile_by_id: dict[int, Any] = {p.agent_id: p for p in roster.agents}

    # Phase 1：初始响应生成
    initial_turns: list[dict[str, Any]] = []
    for agent_id in range(1, roster.agent_count + 1):
        profile = profile_by_id.get(agent_id)
        persona = profile.persona_instruction if profile else ""
        agent_temp = (
            profile.temperature_override
            if profile and profile.temperature_override is not None
            else protocol.temperature
        )
        messages = build_initial_messages(
            sample, agent_id, prompt_version=prompt_version, persona_instruction=persona,
        )
        initial_turns.append(
            _execute_turn(
                run_id=run_id,
                dataset=benchmark_slug,
                split_name=split_name,
                sample=sample,
                method_name=setup.name,
                method_type=method_type,
                round_index=0,
                agent_id=agent_id,
                role="initial",
                visible_peer_count=0,
                messages=messages,
                backbone=backbone,
                provider=provider,
                cache=cache,
                limiter=limiter,
                temperature=agent_temp,
                top_p=protocol.top_p,
                max_output_tokens=protocol.max_output_tokens,
                seed=global_seed + agent_id,
            )
        )
    turn_rows.extend(initial_turns)

    # Phase 2：多轮辩论（含 Beta-Binomial KS 稳定性检测）
    previous_round = initial_turns
    round_history: list[list[dict[str, Any]]] = []
    actual_debate_rounds = 0
    consecutive_stable = 0
    previous_params: BetaBinomialParams | None = None
    final_ks_statistic = 1.0

    for round_index in range(1, protocol.max_debate_rounds + 1):
        current_round: list[dict[str, Any]] = []

        # 每个智能体进行辩论
        for recipient_id in range(1, roster.agent_count + 1):
            recipient_previous = previous_round[recipient_id - 1]
            profile = profile_by_id.get(recipient_id)
            persona = profile.persona_instruction if profile else ""
            agent_temp = (
                profile.temperature_override
                if profile and profile.temperature_override is not None
                else protocol.temperature
            )

            # 构造对等消息
            peer_messages: list[dict[str, str]] = []
            for sender in previous_round:
                if sender["agent_id"] == recipient_id:
                    continue
                peer_messages.append({
                    "agent": f"agent_{sender['agent_id']}",
                    "answer": str(sender["validated_output"].get("final_answer", "")).strip(),
                    "reasoning": str(sender["validated_output"].get("reasoning", "")).strip(),
                })
                debate_rows.append(
                    asdict(DebateMessageRecord(
                        run_id=run_id,
                        dataset=benchmark_slug,
                        split=split_name,
                        sample_id=sample.sample_id,
                        method_name=setup.name,
                        round_index=round_index,
                        sender_agent_id=sender["agent_id"],
                        recipient_agent_id=recipient_id,
                        sender_answer=str(sender["validated_output"].get("final_answer", "")).strip(),
                        sender_reasoning=str(sender["validated_output"].get("reasoning", "")).strip(),
                    ))
                )

            # 构造辩论消息
            messages = build_debate_messages(
                sample=sample,
                agent_id=recipient_id,
                round_index=round_index,
                previous_reasoning=str(recipient_previous["validated_output"].get("reasoning", "")).strip(),
                previous_answer=str(recipient_previous["validated_output"].get("final_answer", "")).strip(),
                peer_messages=peer_messages,
                prompt_version=prompt_version,
                persona_instruction=persona,
            )

            current_round.append(
                _execute_turn(
                    run_id=run_id,
                    dataset=benchmark_slug,
                    split_name=split_name,
                    sample=sample,
                    method_name=setup.name,
                    method_type=method_type,
                    round_index=round_index,
                    agent_id=recipient_id,
                    role="debate",
                    visible_peer_count=len(peer_messages),
                    messages=messages,
                    backbone=backbone,
                    provider=provider,
                    cache=cache,
                    limiter=limiter,
                    temperature=agent_temp,
                    top_p=protocol.top_p,
                    max_output_tokens=protocol.max_output_tokens,
                    seed=global_seed + recipient_id + round_index * 100,
                )
            )

        turn_rows.extend(current_round)
        actual_debate_rounds = round_index

        # 构造本轮数据
        current_round_data = []
        for turn in current_round:
            answer = str(turn["validated_output"].get("final_answer", "")).strip()
            current_round_data.append({
                "agent_id": turn["agent_id"],
                "answer": answer,
            })

        # 检查稳定性（论文 Section 5.3）
        stability_state, current_params = check_stability(
            round_history=round_history,
            current_round_data=current_round_data,
            ks_threshold=protocol.stability.ks_threshold,
            consecutive_stable_required=protocol.stability.consecutive_stable_required,
            previous_params=previous_params,
            consecutive_stable_count=consecutive_stable,
        )

        final_ks_statistic = stability_state.ks_statistic
        consecutive_stable = stability_state.consecutive_stable_rounds

        # 更新轮次历史
        round_history.append(current_round_data)
        previous_params = current_params
        previous_round = current_round

        # 检查是否应该停止（连续稳定 2 轮）
        if consecutive_stable >= protocol.stability.consecutive_stable_required:
            break

        # 检查共识（所有 agent 答案一致）
        answers = [str(t["validated_output"].get("final_answer", "")).strip() for t in current_round]
        if len(set(answers)) == 1:
            break

    # Phase 3：Majority Vote 聚合
    initial_answers = [str(row["validated_output"].get("final_answer", "")).strip() for row in initial_turns]
    final_answers = [str(row["validated_output"].get("final_answer", "")).strip() for row in previous_round]

    # 检查初始共识
    initial_consensus = len(set(initial_answers)) == 1

    # Majority Vote 聚合
    initial_vote, initial_vote_counts = aggregate_majority_vote(initial_answers)
    final_vote, final_vote_counts = aggregate_majority_vote(final_answers)

    # 计算分数
    initial_vote_score = score_prediction(benchmark_slug, initial_vote, sample.reference_answer)
    final_vote_score = score_prediction(benchmark_slug, final_vote, sample.reference_answer)

    # 计算 token 和延迟
    initial_prompt_tokens = sum(float(row["prompt_tokens"]) for row in initial_turns)
    initial_completion_tokens = sum(float(row["completion_tokens"]) for row in initial_turns)
    initial_total_tokens = sum(float(row["total_tokens"]) for row in initial_turns)
    initial_latency = sum(float(row["latency_ms"]) for row in initial_turns)

    debate_turns = [row for row in turn_rows if row["role"] == "debate"]
    debate_prompt_tokens = sum(float(row["prompt_tokens"]) for row in debate_turns)
    debate_completion_tokens = sum(float(row["completion_tokens"]) for row in debate_turns)
    debate_total_tokens = sum(float(row["total_tokens"]) for row in debate_turns)
    debate_latency = sum(float(row["latency_ms"]) for row in debate_turns)

    question_prompt_tokens = initial_prompt_tokens + debate_prompt_tokens
    question_completion_tokens = initial_completion_tokens + debate_completion_tokens
    question_total_tokens = initial_total_tokens + debate_total_tokens
    question_latency = initial_latency + debate_latency

    # 计算辩论效果
    corrected_by_debate = initial_vote_score < 1.0 and final_vote_score == 1.0
    harmed_by_debate = initial_vote_score == 1.0 and final_vote_score < 1.0
    unchanged_correct = initial_vote_score == 1.0 and final_vote_score == 1.0
    unchanged_wrong = initial_vote_score < 1.0 and final_vote_score < 1.0

    prediction_row = asdict(FinalPredictionRecord(
        run_id=run_id,
        dataset=benchmark_slug,
        split=split_name,
        sample_id=sample.sample_id,
        method_name=setup.name,
        method_type=method_type,
        model_name=backbone.name,
        prediction=final_vote,
        gold=sample.reference_answer,
        score=final_vote_score,
        initial_vote_prediction=initial_vote,
        initial_vote_score=initial_vote_score,
        initial_vote_counts=initial_vote_counts,
        initial_consensus=initial_consensus,
        final_vote_prediction=final_vote,
        final_vote_score=final_vote_score,
        final_vote_counts=final_vote_counts,
        prompt_tokens_per_question=question_prompt_tokens,
        completion_tokens_per_question=question_completion_tokens,
        total_tokens_per_question=question_total_tokens,
        latency_ms_per_question=question_latency,
        initial_prompt_tokens_per_question=initial_prompt_tokens,
        initial_completion_tokens_per_question=initial_completion_tokens,
        initial_total_tokens_per_question=initial_total_tokens,
        initial_latency_ms_per_question=initial_latency,
        debate_prompt_tokens_per_question=debate_prompt_tokens,
        debate_completion_tokens_per_question=debate_completion_tokens,
        debate_total_tokens_per_question=debate_total_tokens,
        debate_latency_ms_per_question=debate_latency,
        calls_per_question=roster.agent_count * (1 + actual_debate_rounds),
        actual_debate_rounds=actual_debate_rounds,
        agent_count=roster.agent_count,
        final_consensus=len(set(final_answers)) == 1,
        initial_disagreement=not initial_consensus,
        vote_flipped=initial_vote != final_vote,
        corrected_by_debate=corrected_by_debate,
        harmed_by_debate=harmed_by_debate,
        unchanged_correct=unchanged_correct,
        unchanged_wrong=unchanged_wrong,
        ks_statistic_final=final_ks_statistic,
        stable_rounds=consecutive_stable,
    ))
    prediction_row["vote_counts"] = final_vote_counts
    return turn_rows, debate_rows, prediction_row


# ── 逐轮跨题批次执行（论文 Section 5.3 的正确实现）───────────────────────────


@dataclass(frozen=True)
class SampleState:
    """单个样本在辩论过程中的状态。"""

    sample: DatasetSample
    initial_turns: list[dict[str, Any]]
    current_round_turns: list[dict[str, Any]]
    turn_rows: list[dict[str, Any]]
    debate_rows: list[dict[str, Any]]
    actual_debate_rounds: int
    stopped: bool = False
    stop_reason: str = ""


def _run_initial_round_for_samples(
    samples: list[DatasetSample],
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    setup: ExperimentSetup,
    protocol: ProtocolConfig,
    roster: RosterConfig,
    backbone,
    provider,
    cache,
    limiter,
    global_seed: int,
    prompt_version: str,
    max_concurrent_requests: int,
) -> list[SampleState]:
    """为所有样本并行运行初始轮（round 0）。"""
    from functools import partial

    from research_experiments.core.execution.runner_common import run_indexed_batch

    def _run_initial_single(sample: DatasetSample) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        turn_rows: list[dict[str, Any]] = []
        profile_by_id: dict[int, Any] = {p.agent_id: p for p in roster.agents}
        initial_turns: list[dict[str, Any]] = []

        for agent_id in range(1, roster.agent_count + 1):
            profile = profile_by_id.get(agent_id)
            persona = profile.persona_instruction if profile else ""
            agent_temp = (
                profile.temperature_override
                if profile and profile.temperature_override is not None
                else protocol.temperature
            )
            messages = build_initial_messages(
                sample, agent_id, prompt_version=prompt_version, persona_instruction=persona,
            )
            initial_turns.append(
                _execute_turn(
                    run_id=run_id,
                    dataset=benchmark_slug,
                    split_name=split_name,
                    sample=sample,
                    method_name=setup.name,
                    method_type=protocol.method_type,
                    round_index=0,
                    agent_id=agent_id,
                    role="initial",
                    visible_peer_count=0,
                    messages=messages,
                    backbone=backbone,
                    provider=provider,
                    cache=cache,
                    limiter=limiter,
                    temperature=agent_temp,
                    top_p=protocol.top_p,
                    max_output_tokens=protocol.max_output_tokens,
                    seed=global_seed + agent_id,
                )
            )
        turn_rows.extend(initial_turns)
        return turn_rows, initial_turns

    worker = partial(
        _run_initial_single,
    )
    results = list(run_indexed_batch(samples, worker=worker, max_concurrent_requests=max_concurrent_requests))

    states: list[SampleState] = []
    for idx, (turn_rows, initial_turns) in results:
        sample = samples[idx]
        # 检查初始共识
        initial_answers = [str(t["validated_output"].get("final_answer", "")).strip() for t in initial_turns]
        consensus = len(set(initial_answers)) == 1
        states.append(SampleState(
            sample=sample,
            initial_turns=initial_turns,
            current_round_turns=initial_turns,
            turn_rows=list(turn_rows),
            debate_rows=[],
            actual_debate_rounds=0,
            stopped=consensus,
            stop_reason="initial_consensus" if consensus else "",
        ))
    return states


def _run_debate_round_for_samples(
    states: list[SampleState],
    round_index: int,
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    setup: ExperimentSetup,
    protocol: ProtocolConfig,
    roster: RosterConfig,
    backbone,
    provider,
    cache,
    limiter,
    global_seed: int,
    prompt_version: str,
    max_concurrent_requests: int,
) -> list[SampleState]:
    """为所有未停止的样本运行一轮辩论。"""
    from functools import partial

    from research_experiments.core.execution.runner_common import run_indexed_batch

    # 只处理未停止的样本
    active_indices = [i for i, s in enumerate(states) if not s.stopped]
    if not active_indices:
        return states

    def _run_debate_single(idx: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        state = states[idx]
        sample = state.sample
        previous_round = state.current_round_turns
        turn_rows: list[dict[str, Any]] = []
        debate_rows: list[dict[str, Any]] = []
        profile_by_id: dict[int, Any] = {p.agent_id: p for p in roster.agents}

        current_round: list[dict[str, Any]] = []
        for recipient_id in range(1, roster.agent_count + 1):
            recipient_previous = previous_round[recipient_id - 1]
            profile = profile_by_id.get(recipient_id)
            persona = profile.persona_instruction if profile else ""
            agent_temp = (
                profile.temperature_override
                if profile and profile.temperature_override is not None
                else protocol.temperature
            )

            peer_messages: list[dict[str, str]] = []
            for sender in previous_round:
                if sender["agent_id"] == recipient_id:
                    continue
                peer_messages.append({
                    "agent": f"agent_{sender['agent_id']}",
                    "answer": str(sender["validated_output"].get("final_answer", "")).strip(),
                    "reasoning": str(sender["validated_output"].get("reasoning", "")).strip(),
                })
                debate_rows.append(
                    asdict(DebateMessageRecord(
                        run_id=run_id,
                        dataset=benchmark_slug,
                        split=split_name,
                        sample_id=sample.sample_id,
                        method_name=setup.name,
                        round_index=round_index,
                        sender_agent_id=sender["agent_id"],
                        recipient_agent_id=recipient_id,
                        sender_answer=str(sender["validated_output"].get("final_answer", "")).strip(),
                        sender_reasoning=str(sender["validated_output"].get("reasoning", "")).strip(),
                    ))
                )

            messages = build_debate_messages(
                sample=sample,
                agent_id=recipient_id,
                round_index=round_index,
                previous_reasoning=str(recipient_previous["validated_output"].get("reasoning", "")).strip(),
                previous_answer=str(recipient_previous["validated_output"].get("final_answer", "")).strip(),
                peer_messages=peer_messages,
                prompt_version=prompt_version,
                persona_instruction=persona,
            )

            current_round.append(
                _execute_turn(
                    run_id=run_id,
                    dataset=benchmark_slug,
                    split_name=split_name,
                    sample=sample,
                    method_name=setup.name,
                    method_type=protocol.method_type,
                    round_index=round_index,
                    agent_id=recipient_id,
                    role="debate",
                    visible_peer_count=len(peer_messages),
                    messages=messages,
                    backbone=backbone,
                    provider=provider,
                    cache=cache,
                    limiter=limiter,
                    temperature=agent_temp,
                    top_p=protocol.top_p,
                    max_output_tokens=protocol.max_output_tokens,
                    seed=global_seed + recipient_id + round_index * 100,
                )
            )
        turn_rows.extend(current_round)
        return turn_rows, debate_rows, current_round

    worker = partial(_run_debate_single)
    results = list(run_indexed_batch(
        active_indices, worker=worker, max_concurrent_requests=max_concurrent_requests,
    ))

    # 更新状态
    new_states = list(states)
    for idx_in_active, (turn_rows, debate_rows, current_round) in results:
        real_idx = active_indices[idx_in_active]
        old_state = states[real_idx]

        # 检查本轮共识
        current_answers = [str(t["validated_output"].get("final_answer", "")).strip() for t in current_round]
        consensus = len(set(current_answers)) == 1

        new_states[real_idx] = SampleState(
            sample=old_state.sample,
            initial_turns=old_state.initial_turns,
            current_round_turns=current_round,
            turn_rows=old_state.turn_rows + list(turn_rows),
            debate_rows=old_state.debate_rows + list(debate_rows),
            actual_debate_rounds=round_index,
            stopped=consensus,
            stop_reason="consensus" if consensus else "",
        )
    return new_states


def _finalize_samples(
    states: list[SampleState],
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    setup: ExperimentSetup,
    roster: RosterConfig,
    backbone,
    ks_statistic: float,
    stable_rounds: int,
) -> list[tuple[int, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]]:
    """为所有样本生成最终预测记录。"""
    results: list[tuple[int, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]] = []

    for idx, state in enumerate(states):
        sample = state.sample
        initial_turns = state.initial_turns
        previous_round = state.current_round_turns

        initial_answers = [str(row["validated_output"].get("final_answer", "")).strip() for row in initial_turns]
        final_answers = [str(row["validated_output"].get("final_answer", "")).strip() for row in previous_round]

        initial_consensus = len(set(initial_answers)) == 1
        initial_vote, initial_vote_counts = aggregate_majority_vote(initial_answers)
        final_vote, final_vote_counts = aggregate_majority_vote(final_answers)

        initial_vote_score = score_prediction(benchmark_slug, initial_vote, sample.reference_answer)
        final_vote_score = score_prediction(benchmark_slug, final_vote, sample.reference_answer)

        initial_prompt_tokens = sum(float(row["prompt_tokens"]) for row in initial_turns)
        initial_completion_tokens = sum(float(row["completion_tokens"]) for row in initial_turns)
        initial_total_tokens = sum(float(row["total_tokens"]) for row in initial_turns)
        initial_latency = sum(float(row["latency_ms"]) for row in initial_turns)

        debate_turns = [row for row in state.turn_rows if row["role"] == "debate"]
        debate_prompt_tokens = sum(float(row["prompt_tokens"]) for row in debate_turns)
        debate_completion_tokens = sum(float(row["completion_tokens"]) for row in debate_turns)
        debate_total_tokens = sum(float(row["total_tokens"]) for row in debate_turns)
        debate_latency = sum(float(row["latency_ms"]) for row in debate_turns)

        question_prompt_tokens = initial_prompt_tokens + debate_prompt_tokens
        question_completion_tokens = initial_completion_tokens + debate_completion_tokens
        question_total_tokens = initial_total_tokens + debate_total_tokens
        question_latency = initial_latency + debate_latency

        corrected_by_debate = initial_vote_score < 1.0 and final_vote_score == 1.0
        harmed_by_debate = initial_vote_score == 1.0 and final_vote_score < 1.0
        unchanged_correct = initial_vote_score == 1.0 and final_vote_score == 1.0
        unchanged_wrong = initial_vote_score < 1.0 and final_vote_score < 1.0

        prediction_row = asdict(FinalPredictionRecord(
            run_id=run_id,
            dataset=benchmark_slug,
            split=split_name,
            sample_id=sample.sample_id,
            method_name=setup.name,
            method_type="madjudge",
            model_name=backbone.name,
            prediction=final_vote,
            gold=sample.reference_answer,
            score=final_vote_score,
            initial_vote_prediction=initial_vote,
            initial_vote_score=initial_vote_score,
            initial_vote_counts=initial_vote_counts,
            initial_consensus=initial_consensus,
            final_vote_prediction=final_vote,
            final_vote_score=final_vote_score,
            final_vote_counts=final_vote_counts,
            prompt_tokens_per_question=question_prompt_tokens,
            completion_tokens_per_question=question_completion_tokens,
            total_tokens_per_question=question_total_tokens,
            latency_ms_per_question=question_latency,
            initial_prompt_tokens_per_question=initial_prompt_tokens,
            initial_completion_tokens_per_question=initial_completion_tokens,
            initial_total_tokens_per_question=initial_total_tokens,
            initial_latency_ms_per_question=initial_latency,
            debate_prompt_tokens_per_question=debate_prompt_tokens,
            debate_completion_tokens_per_question=debate_completion_tokens,
            debate_total_tokens_per_question=debate_total_tokens,
            debate_latency_ms_per_question=debate_latency,
            calls_per_question=roster.agent_count * (1 + state.actual_debate_rounds),
            actual_debate_rounds=state.actual_debate_rounds,
            agent_count=roster.agent_count,
            final_consensus=len(set(final_answers)) == 1,
            initial_disagreement=not initial_consensus,
            vote_flipped=initial_vote != final_vote,
            corrected_by_debate=corrected_by_debate,
            harmed_by_debate=harmed_by_debate,
            unchanged_correct=unchanged_correct,
            unchanged_wrong=unchanged_wrong,
            ks_statistic_final=ks_statistic,
            stable_rounds=stable_rounds,
        ))
        prediction_row["vote_counts"] = final_vote_counts
        results.append((idx, state.turn_rows, state.debate_rows, prediction_row))

    return results


def _run_madjudge_batch_round_by_round(
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    samples: list[DatasetSample],
    setup: ExperimentSetup,
    protocol: ProtocolConfig,
    roster: RosterConfig,
    backbone,
    provider,
    cache,
    limiter,
    global_seed: int,
    prompt_version: str,
    max_concurrent_requests: int,
) -> list[tuple[int, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]]:
    """逐轮跨题批次执行 MADJudge 辩论（论文 Section 5.3 的正确实现）。

    论文的 Beta-Binomial 模型跨所有题目聚合观测：
    每轮结束后，收集所有题目的 majority count，拟合分布，比较相邻轮次差异。
    """

    # Phase 1：初始轮
    print(f"[MADJudge] Running initial round for {len(samples)} samples...", flush=True)
    states = _run_initial_round_for_samples(
        samples,
        run_id=run_id,
        benchmark_slug=benchmark_slug,
        split_name=split_name,
        setup=setup,
        protocol=protocol,
        roster=roster,
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        global_seed=global_seed,
        prompt_version=prompt_version,
        max_concurrent_requests=max_concurrent_requests,
    )

    # Phase 2：逐轮辩论
    k = roster.agent_count
    consecutive_stable = 0
    previous_params: BetaBinomialParams | None = None
    final_ks_statistic = 1.0
    actual_debate_rounds = 0

    for round_index in range(1, protocol.max_debate_rounds + 1):
        # 检查是否所有样本都已停止
        active_count = sum(1 for s in states if not s.stopped)
        if active_count == 0:
            print("[MADJudge] All samples reached consensus, stopping.", flush=True)
            break

        print(f"[MADJudge] Running debate round {round_index} ({active_count} active samples)...", flush=True)
        states = _run_debate_round_for_samples(
            states,
            round_index,
            run_id=run_id,
            benchmark_slug=benchmark_slug,
            split_name=split_name,
            setup=setup,
            protocol=protocol,
            roster=roster,
            backbone=backbone,
            provider=provider,
            cache=cache,
            limiter=limiter,
            global_seed=global_seed,
            prompt_version=prompt_version,
            max_concurrent_requests=max_concurrent_requests,
        )
        actual_debate_rounds = round_index

        # 跨题聚合：收集所有活跃样本的 majority count
        majority_counts: list[int] = []
        for state in states:
            if state.actual_debate_rounds == round_index:
                # 本轮有新数据的样本
                answers = [str(t["validated_output"].get("final_answer", "")).strip() for t in state.current_round_turns]
                majority_counts.append(compute_majority_count(answers))

        if not majority_counts:
            continue

        # 跨题稳定性检测
        stability_state, current_params = check_stability_batch(
            round_majority_counts=majority_counts,
            k=k,
            ks_threshold=protocol.stability.ks_threshold,
            consecutive_stable_required=protocol.stability.consecutive_stable_required,
            previous_params=previous_params,
            consecutive_stable_count=consecutive_stable,
        )

        final_ks_statistic = stability_state.ks_statistic
        consecutive_stable = stability_state.consecutive_stable_rounds
        previous_params = current_params

        print(f"[MADJudge] Round {round_index}: KS={final_ks_statistic:.4f}, stable={consecutive_stable}/{protocol.stability.consecutive_stable_required}", flush=True)

        # 检查是否应该停止
        if stability_state.is_stable:
            print(f"[MADJudge] Stability reached after {round_index} rounds.", flush=True)
            break

    # Phase 3：Finalize
    return _finalize_samples(
        states,
        run_id=run_id,
        benchmark_slug=benchmark_slug,
        split_name=split_name,
        setup=setup,
        roster=roster,
        backbone=backbone,
        ks_statistic=final_ks_statistic,
        stable_rounds=consecutive_stable,
    )


def _execute_turn(
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
    limiter,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    seed: int,
) -> dict[str, Any]:
    """执行单次 agent turn。"""
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
        validator=_validate_madjudge_output,
        dataset=dataset,
        use_response_format=False,
    )
    final_answer = str(result.validated_output.get("final_answer") or "")
    normalized = normalize_prediction(dataset, final_answer) if final_answer else ""
    return asdict(
        AgentTurnRecord(
            run_id=run_id,
            dataset=dataset,
            split=split_name,
            sample_id=sample.sample_id,
            method_name=method_name,
            method_type=method_type,
            round_index=round_index,
            agent_id=agent_id,
            role=role,
            prompt_hash=result.prompt_hash,
            prediction=normalized,
            output_status=result.output_status,
            prompt_tokens=float(result.usage.get("prompt_tokens") or 0.0),
            completion_tokens=float(result.usage.get("completion_tokens") or 0.0),
            total_tokens=float(result.usage.get("total_tokens") or 0.0),
            latency_ms=float(result.response_payload.get("latency_ms") or 0.0),
            cache_hit=result.cache_hit,
            request_error=result.request_error,
            visible_peer_count=visible_peer_count,
            payload=result.payload,
            assistant_text=result.response_payload.get("assistant_text", ""),
            provider_reasoning_text=result.response_payload.get("provider_reasoning_text", ""),
            validated_output=result.validated_output,
        )
    ) | {"normalized_answer": normalized}


def _build_metrics(
    prediction_rows: list[dict[str, Any]],
    experiment: MadJudgeExperimentConfig,
    setups: list[ExperimentSetup],
) -> dict[str, Any]:
    """把最终题级预测聚合成方法级 summary。"""
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in prediction_rows:
        key = (row["dataset"], row["model_name"], row["method_name"])
        grouped.setdefault(key, []).append(row)

    setup_map = {item.name: item for item in setups}
    summary: list[dict[str, Any]] = []
    for (dataset, model_name, method_name), rows in sorted(grouped.items()):
        accuracy = sum(float(row["score"]) for row in rows) / len(rows)
        total_tokens_mean = sum(float(row["total_tokens_per_question"]) for row in rows) / len(rows)
        row = {
            "dataset": dataset,
            "model_name": model_name,
            "method_name": method_name,
            "method_type": rows[0]["method_type"],
            "prediction_rows": len(rows),
            "accuracy_mean": accuracy,
            "prompt_tokens_mean": sum(float(item["prompt_tokens_per_question"]) for item in rows) / len(rows),
            "completion_tokens_mean": sum(float(item["completion_tokens_per_question"]) for item in rows) / len(rows),
            "total_tokens_mean": total_tokens_mean,
            "calls_per_question_mean": sum(float(item["calls_per_question"]) for item in rows) / len(rows),
            "latency_ms_mean": sum(float(item["latency_ms_per_question"]) for item in rows) / len(rows),
            "accuracy_per_1k_tokens": (accuracy / total_tokens_mean * 1000) if total_tokens_mean else 0.0,
            "actual_debate_rounds_mean": sum(float(item["actual_debate_rounds"]) for item in rows) / len(rows),
            "agent_count": rows[0]["agent_count"],
            "ks_statistic_mean": safe_mean(float(item.get("ks_statistic_final", 1.0)) for item in rows),
            "stable_rounds_mean": safe_mean(float(item.get("stable_rounds", 0)) for item in rows),
        }
        if method_name in setup_map:
            controls = setup_map[method_name].matched_controls
            row["matched_vote_control"] = next((name for name in controls if name.startswith("mv_")), None)
        summary.append(row)

    by_lookup = {(row["dataset"], row["model_name"], row["method_name"]): row for row in summary}
    for row in summary:
        if row["method_type"] != "madjudge":
            continue
        vote_name = row.get("matched_vote_control")
        vote_row = by_lookup.get((row["dataset"], row["model_name"], vote_name)) if vote_name else None
        row["debate_gain_over_vote"] = round(row["accuracy_mean"] - vote_row["accuracy_mean"], 6) if vote_row else None
        row["token_overhead_vs_vote"] = (
            round((row["total_tokens_mean"] - vote_row["total_tokens_mean"]) / vote_row["total_tokens_mean"], 6)
            if vote_row and vote_row["total_tokens_mean"]
            else None
        )
    return {"summary": summary}


def _build_cost_breakdown(turn_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总初始回答和 debate 的 token 成本。"""
    grouped: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in turn_rows:
        key = (row["dataset"], row["method_name"], row["method_type"])
        bucket = grouped.setdefault(
            key,
            {
                "prompt_tokens": 0.0,
                "completion_tokens": 0.0,
                "total_tokens": 0.0,
                "latency_ms": 0.0,
                "turn_count": 0.0,
                "initial_tokens": 0.0,
                "debate_tokens": 0.0,
                "control_tokens": 0.0,
            },
        )
        total_tokens = float(row["total_tokens"])
        bucket["prompt_tokens"] += float(row["prompt_tokens"])
        bucket["completion_tokens"] += float(row["completion_tokens"])
        bucket["total_tokens"] += total_tokens
        bucket["latency_ms"] += float(row["latency_ms"])
        bucket["turn_count"] += 1
        if row["role"] == "initial":
            bucket["initial_tokens"] += total_tokens
        elif row["role"] == "debate":
            bucket["debate_tokens"] += total_tokens
        else:
            bucket["control_tokens"] += total_tokens

    rows = []
    for (dataset, method_name, method_type), bucket in sorted(grouped.items()):
        rows.append({"dataset": dataset, "method_name": method_name, "method_type": method_type} | bucket)
    return {"rows": rows}


def _build_debate_diagnostics(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """构建 debate 诊断指标。"""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in prediction_rows:
        if row["method_type"] != "madjudge":
            continue
        grouped.setdefault((row["dataset"], row["method_name"]), []).append(row)

    rows = []
    for (dataset, method_name), rows_for_key in sorted(grouped.items()):
        total = len(rows_for_key)
        rows.append({
            "dataset": dataset,
            "method_name": method_name,
            "question_count": total,
            "initial_disagreement_rate": safe_ratio(sum(1 for row in rows_for_key if row["initial_disagreement"]), total),
            "post_debate_consensus_rate": safe_ratio(sum(1 for row in rows_for_key if row["final_consensus"]), total),
            "vote_flip_rate": safe_ratio(sum(1 for row in rows_for_key if row["vote_flipped"]), total),
            "wrong_consensus_rate": safe_ratio(
                sum(1 for row in rows_for_key if row["final_consensus"] and float(row["score"]) < 1.0),
                total,
            ),
            "avg_debate_rounds": safe_mean(float(row["actual_debate_rounds"]) for row in rows_for_key),
            "ks_statistic_mean": safe_mean(float(row.get("ks_statistic_final", 1.0)) for row in rows_for_key),
            "stable_rounds_mean": safe_mean(float(row.get("stable_rounds", 0)) for row in rows_for_key),
        })
    return {"rows": rows}


def _estimate_work(
    experiment: MadJudgeExperimentConfig,
    phase_name: str,
    benchmarks,
    setups: list[ExperimentSetup],
    matched_control_names: list[str],
    controls,
) -> tuple[int, int]:
    """估算本次运行的总调用量与总预测量。"""
    from research_experiments.families.madjudge.config import phase_metadata

    phase = phase_metadata(experiment, phase_name)
    total_calls = 0
    total_predictions = 0
    for benchmark in benchmarks:
        split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
        sample_count = len(load_split_ids(benchmark.cache_namespace or benchmark.slug, split_name))
        for setup in setups:
            protocol = load_protocol_config(setup.protocol)
            roster = load_roster_config(setup.roster)
            total_calls += sample_count * roster.agent_count * (1 + protocol.max_debate_rounds)
            total_predictions += sample_count
        for name in matched_control_names:
            total_calls += sample_count * controls[name].budget_calls
            total_predictions += sample_count
    return total_calls, total_predictions


def _active_setups(experiment: MadJudgeExperimentConfig, phase_name: str) -> list[ExperimentSetup]:
    """解析当前 phase 实际启用的 setup 列表。"""
    from research_experiments.families.madjudge.config import phase_metadata

    phase = phase_metadata(experiment, phase_name)
    requested = set(phase["setups"])
    available = {item.name: item for item in experiment.setups}
    missing = sorted(requested - set(available))
    if missing:
        raise RuntimeError(f"Unknown MADJudge setups for phase {phase_name}: {', '.join(missing)}")
    return [available[name] for name in phase["setups"]]


def _resolve_split_name(experiment: MadJudgeExperimentConfig, phase_name: str, benchmark_slug: str) -> str:
    """解析当前 benchmark 在该 phase 下对应的冻结 split 名称。"""
    return resolve_phase_split_name(experiment, phase_name, benchmark_slug)


def _load_selected_samples(benchmark, split_name: str) -> list[DatasetSample]:
    """按冻结 split 选择本轮要跑的样本。"""
    return select_samples(benchmark, split_name)


def _validate_madjudge_output(assistant_text: str, provider_reasoning_text: str) -> dict[str, Any]:
    """解析 ##Answer/##Explanation 标记格式，带 JSON fallback。"""
    text = str(assistant_text or "").strip() or str(provider_reasoning_text or "").strip()
    if not text:
        raise ValueError("Model output is empty.")

    # 尝试从 JSON 外壳中提取内嵌的 ##Answer 格式
    inner_text = _extract_inner_text(text)

    # 优先解析标记格式
    answer_match = re.search(r"##Answer\s*\n?\s*(.*?)(?=\s*##|\s*$)", inner_text, re.DOTALL | re.IGNORECASE)
    explanation_match = re.search(r"##Explanation\s*\n?\s*(.*?)(?=\s*##|\s*$)", inner_text, re.DOTALL | re.IGNORECASE)

    if answer_match:
        validated = {
            "final_answer": answer_match.group(1).strip(),
            "reasoning": explanation_match.group(1).strip() if explanation_match else "",
        }
        return validated

    # JSON fallback
    payload = _try_parse_json(text)
    if payload is None:
        raise ValueError("Failed to parse model output: neither ##Answer format nor JSON found.")

    if not isinstance(payload, dict):
        raise ValueError("Model output is not a JSON object.")

    final_answer = str(
        payload.get("final_answer") or payload.get("answer") or payload.get("prediction") or ""
    ).strip()
    reasoning = str(
        payload.get("reasoning") or payload.get("explanation") or ""
    ).strip()

    if not final_answer:
        list_val = payload.get("list") or payload.get("answers") or payload.get("titles")
        if isinstance(list_val, list):
            final_answer = ", ".join(str(item).strip() for item in list_val if str(item).strip())

    if not final_answer:
        raise ValueError("Could not extract answer from model output.")

    return {
        "final_answer": final_answer,
        "reasoning": reasoning,
    }


def _extract_inner_text(text: str) -> str:
    """如果 text 是 JSON 且包含内嵌的 ##Answer 格式，提取内部文本。"""
    if not text.startswith("{"):
        return text
    try:
        payload = json.loads(text)
    except Exception:
        return text
    if not isinstance(payload, dict):
        return text
    for key in ("answer", "response", "text", "content", "output"):
        val = payload.get(key)
        if isinstance(val, str) and "##Answer" in val:
            return val
    return text


def _try_parse_json(text: str) -> dict[str, Any] | None:
    """尝试多种方式解析 JSON。"""
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except Exception:
        pass
    match = re.search(r'\{[^{}]*(?:"final_answer"|"answer"|"list"|"explanation")[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, dict):
                return result
        except Exception:
            pass
    return None


def _run_madjudge_batch(
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    samples: list[DatasetSample],
    setup: ExperimentSetup,
    protocol: ProtocolConfig,
    roster: RosterConfig,
    backbone,
    provider,
    cache,
    limiter,
    global_seed: int,
    prompt_version: str,
    max_concurrent_requests: int,
) -> list[tuple[int, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]]:
    """并发执行同一 setup 下的全部样本。"""
    worker = partial(
        _run_madjudge_sample,
        run_id=run_id,
        benchmark_slug=benchmark_slug,
        split_name=split_name,
        setup=setup,
        protocol=protocol,
        roster=roster,
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        global_seed=global_seed,
        prompt_version=prompt_version,
    )
    return [
        (sample_index, *result)
        for sample_index, result in run_indexed_batch(
            samples,
            worker=worker,
            max_concurrent_requests=max_concurrent_requests,
        )
    ]


def _write_sample_outputs(
    sample_results: list[tuple[int, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]],
    dataset_slug: str,
    progress,
    turn_handle,
    debate_handle,
    prediction_handle,
    all_turns: list[dict[str, Any]],
    debate_messages: list[dict[str, Any]],
    final_predictions: list[dict[str, Any]],
) -> None:
    """把 worker 返回的样本结果按稳定顺序写盘，并同步更新进度。"""
    for _, turn_rows, debate_rows, prediction_row in sample_results:
        for row in turn_rows:
            turn_handle.write_row(row)
            progress.record_call(row, method_key="method_name")
        for row in debate_rows:
            debate_handle.write_row(row)
        prediction_handle.write_row(prediction_row)
        progress.record_predictions(1, dataset_slug, prediction_row["method_name"])
        all_turns.extend(turn_rows)
        debate_messages.extend(debate_rows)
        final_predictions.append(prediction_row)


def _build_control_prediction_row(
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
    """构造共享无通信对照组的最终预测行。"""
    prompt_tokens = sum(float(row["prompt_tokens"]) for row in turn_rows)
    completion_tokens = sum(float(row["completion_tokens"]) for row in turn_rows)
    total_tokens = sum(float(row["total_tokens"]) for row in turn_rows)
    latency_ms = sum(float(row["latency_ms"]) for row in turn_rows)
    return asdict(FinalPredictionRecord(
        run_id=run_id,
        dataset=benchmark_slug,
        split=split_name,
        sample_id=sample.sample_id,
        method_name=control_name,
        method_type="control",
        model_name=backbone.name,
        prediction=final_vote,
        gold=sample.reference_answer,
        score=final_score,
        initial_vote_prediction=final_vote,
        initial_vote_score=final_score,
        initial_vote_counts=vote_counts,
        initial_consensus=final_consensus,
        final_vote_prediction=final_vote,
        final_vote_score=final_score,
        final_vote_counts=vote_counts,
        prompt_tokens_per_question=prompt_tokens,
        completion_tokens_per_question=completion_tokens,
        total_tokens_per_question=total_tokens,
        latency_ms_per_question=latency_ms,
        initial_prompt_tokens_per_question=prompt_tokens,
        initial_completion_tokens_per_question=completion_tokens,
        initial_total_tokens_per_question=total_tokens,
        initial_latency_ms_per_question=latency_ms,
        debate_prompt_tokens_per_question=0.0,
        debate_completion_tokens_per_question=0.0,
        debate_total_tokens_per_question=0.0,
        debate_latency_ms_per_question=0.0,
        calls_per_question=method.budget_calls,
        actual_debate_rounds=0,
        agent_count=1 if method.family == "cot" else method.budget_calls,
        final_consensus=final_consensus,
        initial_disagreement=False,
        vote_flipped=False,
        corrected_by_debate=False,
        harmed_by_debate=False,
        unchanged_correct=final_score == 1.0,
        unchanged_wrong=final_score < 1.0,
        ks_statistic_final=0.0,
        stable_rounds=0,
    )) | {"vote_counts": vote_counts}
