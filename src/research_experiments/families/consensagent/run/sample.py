"""CONSENSAGENT 实验的样本级执行逻辑。

本模块实现论文中的四阶段流程：
- Phase 1：初始响应生成（含置信度分数）
- Phase 2：多轮辩论（含触发机制检测）
- Phase 4：团队答案生成（置信度加权聚合）
"""

from __future__ import annotations

import json
import re
from contextlib import suppress
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any

from research_experiments.core.data.datasets import DatasetSample, load_split_ids, select_samples
from research_experiments.core.data.evaluation import aggregate_majority, normalize_prediction, score_prediction
from research_experiments.core.execution.runner_common import execute_cached_turn, run_indexed_batch
from research_experiments.families.consensagent.algorithms import (
    TriggerState,
    aggregate_weighted_answer,
    check_triggers,
    compute_consistency_score,
    compute_sycophancy_rate,
)
from research_experiments.families.consensagent.config import (
    ConsensagentExperimentConfig,
    ExperimentSetup,
    ProtocolConfig,
    RosterConfig,
    load_protocol_config,
    load_roster_config,
)
from research_experiments.families.consensagent.prompts import (
    build_debate_messages,
    build_initial_messages,
    build_optimizer_messages,
)
from research_experiments.families.shared.common import resolve_phase_split_name, safe_mean, safe_ratio
from research_experiments.families.shared.comparator_impls import (
    build_shared_vanilla_mad_prediction,
    run_shared_vanilla_mad_rounds,
)


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
    confidence: float
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
    sender_confidence: float


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
    weighted_prediction: str
    weighted_score: float
    weighted_vote_counts: dict[str, int]
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
    # 触发机制相关
    trigger_type: str | None
    trigger_round: int | None
    sycophancy_rate: float
    # 一致性分数
    initial_consistency_score: float
    final_consistency_score: float


def _run_consensagent_sample(
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
    """运行单个样本上的 CONSENSAGENT 协议。"""
    if protocol.method_type == "mad":
        shared_result = run_shared_vanilla_mad_rounds(
            sample=sample,
            run_id=run_id,
            dataset=benchmark_slug,
            split_name=split_name,
            method_name=setup.name,
            agent_count=roster.agent_count,
            debate_rounds=protocol.max_debate_rounds,
            initial_temperature=protocol.initial_temperature,
            debate_temperature=protocol.debate_temperature,
            top_p=protocol.top_p,
            max_output_tokens=protocol.max_output_tokens,
            global_seed=global_seed,
            prompt_version=prompt_version,
            execute_turn=lambda **kwargs: _execute_turn(
                run_id=run_id,
                dataset=benchmark_slug,
                split_name=split_name,
                sample=sample,
                method_name=setup.name,
                method_type=protocol.method_type,
                backbone=backbone,
                provider=provider,
                cache=cache,
                limiter=limiter,
                **kwargs,
            ),
            build_debate_row=lambda sender, recipient_id, round_index: asdict(
                DebateMessageRecord(
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
                    sender_confidence=float(sender["validated_output"].get("confidence", 0.5)),
                )
            ),
        )

        initial_answers = [
            str(row["validated_output"].get("final_answer", "")).strip()
            for row in shared_result["initial_turns"]
        ]
        final_answers = [
            str(row["validated_output"].get("final_answer", "")).strip()
            for row in shared_result["final_round_turns"]
        ]
        initial_consistency = compute_consistency_score(initial_answers)
        final_consistency = compute_consistency_score(final_answers)
        prediction_row = build_shared_vanilla_mad_prediction(
            run_id=run_id,
            dataset=benchmark_slug,
            split_name=split_name,
            sample=sample,
            method_name=setup.name,
            method_type=protocol.method_type,
            model_name=backbone.name,
            result=shared_result,
            extra_fields={
                "weighted_prediction": shared_result["final_vote_prediction"],
                "weighted_score": shared_result["final_vote_score"],
                "weighted_vote_counts": shared_result["final_vote_counts"],
                "actual_debate_rounds": protocol.max_debate_rounds,
                "trigger_type": None,
                "trigger_round": None,
                "sycophancy_rate": 0.0,
                "initial_consistency_score": initial_consistency,
                "final_consistency_score": final_consistency,
            },
        )
        return shared_result["turn_rows"], shared_result["debate_rows"], prediction_row

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
            else protocol.initial_temperature
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

    # Phase 2：多轮辩论（含触发机制）
    # 论文：达成共识即终止辩论（除非检测到谄媚）
    previous_round = initial_turns
    round_history: list[list[dict[str, Any]]] = []
    debate_memory: list[dict[str, Any]] = []
    trigger_state = TriggerState()
    actual_debate_rounds = 0

    # 检查初始共识
    initial_answers = [str(t["validated_output"].get("final_answer", "")).strip() for t in initial_turns]
    initial_consistency = compute_consistency_score(initial_answers)

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
                else protocol.debate_temperature
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
                    "confidence": float(sender["validated_output"].get("confidence", 0.5)),
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
                        sender_confidence=float(sender["validated_output"].get("confidence", 0.5)),
                    ))
                )

            # 构造辩论消息
            messages = build_debate_messages(
                sample=sample,
                agent_id=recipient_id,
                round_index=round_index,
                previous_reasoning=str(recipient_previous["validated_output"].get("reasoning", "")).strip(),
                previous_answer=str(recipient_previous["validated_output"].get("final_answer", "")).strip(),
                previous_confidence=float(recipient_previous["validated_output"].get("confidence", 0.5)),
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
        round_answers_for_memory = []
        for i, turn in enumerate(current_round):
            answer = str(turn["validated_output"].get("final_answer", "")).strip()
            confidence = float(turn["validated_output"].get("confidence", 0.5))
            current_round_data.append({
                "agent_id": turn["agent_id"],
                "answer": answer,
                "confidence": confidence,
                "previous_answer": str(previous_round[i]["validated_output"].get("final_answer", "")).strip(),
            })
            round_answers_for_memory.append({
                "agent_id": turn["agent_id"],
                "answer": answer,
                "confidence": confidence,
            })
        debate_memory.append({"round": round_index, "answers": round_answers_for_memory})

        # 检查触发条件
        trigger_state = check_triggers(
            round_history=round_history,
            current_round_answers=current_round_data,
            stagnation_threshold=protocol.trigger.stagnation_threshold,
            sycophancy_consistency_threshold=protocol.trigger.sycophancy_consistency_threshold,
            check_sycophancy_on_consensus=protocol.trigger.check_sycophancy_on_consensus,
        )

        # 更新轮次历史
        round_history.append(current_round_data)

        # 触发条件：停滞或谄媚 → 提前结束
        if trigger_state.stagnation_triggered or trigger_state.sycophancy_triggered:
            break

        # 论文优化：达成共识且无谄媚 → 提前结束
        current_answers = [str(t["validated_output"].get("final_answer", "")).strip() for t in current_round]
        current_consistency = compute_consistency_score(current_answers)
        if current_consistency.is_consensus and not trigger_state.sycophancy_triggered:
            break

        previous_round = current_round

    # Phase 3：Prompt 优化（论文完整版使用微调 GPT-4o，这里用 LLM in-context learning）
    phase3_triggered = trigger_state.stagnation_triggered or trigger_state.sycophancy_triggered
    optimized_prompt = ""
    phase3_turn_rows: list[dict[str, Any]] = []

    if phase3_triggered and protocol.phase3.enabled:
        # 构建优化器消息
        optimizer_messages = build_optimizer_messages(
            sample=sample,
            debate_history=debate_memory,
            trigger_type=trigger_state.trigger_type or "unknown",
            prompt_version="consensagent_paper_v1",
        )

        # 调用 LLM 生成优化 prompt
        optimizer_result = _execute_turn(
            run_id=run_id,
            dataset=benchmark_slug,
            split_name=split_name,
            sample=sample,
            method_name=setup.name,
            method_type=method_type,
            round_index=-1,  # optimizer round
            agent_id=0,
            role="optimizer",
            visible_peer_count=0,
            messages=optimizer_messages,
            backbone=backbone,
            provider=provider,
            cache=cache,
            limiter=limiter,
            temperature=protocol.phase3.optimizer_temperature,
            top_p=protocol.top_p,
            max_output_tokens=protocol.phase3.max_optimizer_output_tokens,
            seed=global_seed,
        )
        phase3_turn_rows.append(optimizer_result)
        optimized_prompt = str(optimizer_result["validated_output"].get("final_answer", "")).strip()

        # 用优化后的 prompt 跑 post_optimization_rounds 轮辩论
        if optimized_prompt and protocol.phase3.post_optimization_rounds > 0:
            _opt_round_data: list[dict[str, Any]] = []
            for _ in range(protocol.phase3.post_optimization_rounds):
                opt_round: list[dict[str, Any]] = []
                for recipient_id in range(1, roster.agent_count + 1):
                    profile = profile_by_id.get(recipient_id)
                    persona = profile.persona_instruction if profile else ""
                    agent_temp = (
                        profile.temperature_override
                        if profile and profile.temperature_override is not None
                        else protocol.debate_temperature
                    )
                    recipient_prev = previous_round[recipient_id - 1]

                    peer_msgs: list[dict[str, str]] = []
                    for sender in previous_round:
                        if sender["agent_id"] == recipient_id:
                            continue
                        peer_msgs.append({
                            "agent": f"Agent {sender['agent_id']}",
                            "answer": str(sender["validated_output"].get("final_answer", "")).strip(),
                            "reasoning": str(sender["validated_output"].get("reasoning", "")).strip(),
                            "confidence": float(sender["validated_output"].get("confidence", 0.5)),
                        })

                    # 构造优化后的辩论消息（用 optimized_prompt 替换原始问题）
                    opt_messages = [
                        {"role": "system", "content": _optimized_system_prompt(persona)},
                        {"role": "user", "content": _format_optimized_debate_prompt(
                            sample=sample,
                            agent_id=recipient_id,
                            optimized_prompt=optimized_prompt,
                            previous_answer=str(recipient_prev["validated_output"].get("final_answer", "")).strip(),
                            previous_reasoning=str(recipient_prev["validated_output"].get("reasoning", "")).strip(),
                            previous_confidence=float(recipient_prev["validated_output"].get("confidence", 0.5)),
                            peer_messages=peer_msgs,
                        )},
                    ]

                    opt_round.append(_execute_turn(
                        run_id=run_id,
                        dataset=benchmark_slug,
                        split_name=split_name,
                        sample=sample,
                        method_name=setup.name,
                        method_type=method_type,
                        round_index=actual_debate_rounds + 1,
                        agent_id=recipient_id,
                        role="debate_optimized",
                        visible_peer_count=len(peer_msgs),
                        messages=opt_messages,
                        backbone=backbone,
                        provider=provider,
                        cache=cache,
                        limiter=limiter,
                        temperature=agent_temp,
                        top_p=protocol.top_p,
                        max_output_tokens=protocol.max_output_tokens,
                        seed=global_seed + recipient_id + (actual_debate_rounds + 1) * 100,
                    ))
                phase3_turn_rows.extend(opt_round)
                _opt_round_data = [
                    {
                        "agent_id": t["agent_id"],
                        "answer": str(t["validated_output"].get("final_answer", "")).strip(),
                        "confidence": float(t["validated_output"].get("confidence", 0.5)),
                        "previous_answer": "",
                    }
                    for t in opt_round
                ]
            # 用优化后的结果更新 previous_round
            if opt_round:
                previous_round = opt_round

    # Phase 4：团队答案生成
    initial_answers = [str(row["validated_output"].get("final_answer", "")).strip() for row in initial_turns]

    final_answers = [str(row["validated_output"].get("final_answer", "")).strip() for row in previous_round]

    # 计算一致性分数
    initial_consistency = compute_consistency_score(initial_answers)
    final_consistency = compute_consistency_score(final_answers)

    # 计算谄媚率
    sycophancy_rate = compute_sycophancy_rate(
        round_history,
        consistency_threshold=protocol.trigger.sycophancy_consistency_threshold,
    )

    # 使用置信度加权聚合
    agent_answers_for_weighting = []
    for turn in previous_round:
        agent_answers_for_weighting.append({
            "agent_id": turn["agent_id"],
            "answer": str(turn["validated_output"].get("final_answer", "")).strip(),
            "reasoning": str(turn["validated_output"].get("reasoning", "")).strip(),
            "confidence": float(turn["validated_output"].get("confidence", 0.5)),
        })
    weighted_prediction, weighted_counts = aggregate_weighted_answer(agent_answers_for_weighting)

    # 使用多数投票聚合（用于对比）
    initial_vote, initial_vote_counts = aggregate_majority(initial_answers)
    final_vote, final_vote_counts = aggregate_majority(final_answers)

    # 计算分数
    initial_vote_score = score_prediction(benchmark_slug, initial_vote, sample.reference_answer)
    final_vote_score = score_prediction(benchmark_slug, final_vote, sample.reference_answer)
    weighted_score = score_prediction(benchmark_slug, weighted_prediction, sample.reference_answer)

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
    corrected_by_debate = initial_vote_score < 1.0 and weighted_score == 1.0
    harmed_by_debate = initial_vote_score == 1.0 and weighted_score < 1.0
    unchanged_correct = initial_vote_score == 1.0 and weighted_score == 1.0
    unchanged_wrong = initial_vote_score < 1.0 and weighted_score < 1.0

    prediction_row = asdict(FinalPredictionRecord(
        run_id=run_id,
        dataset=benchmark_slug,
        split=split_name,
        sample_id=sample.sample_id,
        method_name=setup.name,
        method_type=method_type,
        model_name=backbone.name,
        prediction=weighted_prediction,
        gold=sample.reference_answer,
        score=weighted_score,
        initial_vote_prediction=initial_vote,
        initial_vote_score=initial_vote_score,
        initial_vote_counts=initial_vote_counts,
        initial_consensus=initial_consistency.is_consensus,
        final_vote_prediction=final_vote,
        final_vote_score=final_vote_score,
        final_vote_counts=final_vote_counts,
        weighted_prediction=weighted_prediction,
        weighted_score=weighted_score,
        weighted_vote_counts=weighted_counts,
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
        final_consensus=final_consistency.is_consensus,
        initial_disagreement=not initial_consistency.is_consensus,
        vote_flipped=initial_vote != weighted_prediction,
        corrected_by_debate=corrected_by_debate,
        harmed_by_debate=harmed_by_debate,
        unchanged_correct=unchanged_correct,
        unchanged_wrong=unchanged_wrong,
        trigger_type=trigger_state.trigger_type,
        trigger_round=trigger_state.trigger_round,
        sycophancy_rate=sycophancy_rate,
        initial_consistency_score=initial_consistency.score,
        final_consistency_score=final_consistency.score,
    ))
    prediction_row["vote_counts"] = weighted_counts
    return turn_rows, debate_rows, prediction_row


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
    """执行单次 agent turn，并统一返回日志行结构。"""
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
        validator=_validate_consensagent_output,
        dataset=dataset,
        use_response_format=False,
    )
    final_answer = str(result.validated_output.get("final_answer") or "")
    normalized = normalize_prediction(dataset, final_answer) if final_answer else ""
    confidence = float(result.validated_output.get("confidence", 0.5))
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
            confidence=confidence,
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
    experiment: ConsensagentExperimentConfig,
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
            "sycophancy_rate_mean": sum(float(item["sycophancy_rate"]) for item in rows) / len(rows),
            "initial_consistency_score_mean": sum(float(item["initial_consistency_score"]) for item in rows) / len(rows),
            "final_consistency_score_mean": sum(float(item["final_consistency_score"]) for item in rows) / len(rows),
            "trigger_rate": sum(1 for item in rows if item["trigger_type"] is not None) / len(rows),
        }
        if method_name in setup_map:
            controls = setup_map[method_name].matched_controls
            row["matched_vote_control"] = next((name for name in controls if name.startswith("mv_")), None)
        summary.append(row)

    by_lookup = {(row["dataset"], row["model_name"], row["method_name"]): row for row in summary}
    for row in summary:
        if row["method_type"] not in ("consensagent", "mad"):
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
    """汇总初始回答、debate 和控制方法的 token 成本。"""
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
        if row["method_type"] not in ("consensagent", "mad"):
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
            "sycophancy_rate_mean": safe_mean(float(row["sycophancy_rate"]) for row in rows_for_key),
            "trigger_rate": safe_ratio(sum(1 for row in rows_for_key if row["trigger_type"] is not None), total),
            "avg_debate_rounds": safe_mean(float(row["actual_debate_rounds"]) for row in rows_for_key),
        })
    return {"rows": rows}


def _estimate_work(
    experiment: ConsensagentExperimentConfig,
    phase_name: str,
    benchmarks,
    setups: list[ExperimentSetup],
    matched_control_names: list[str],
    controls,
) -> tuple[int, int]:
    """估算本次运行的总调用量与总预测量。"""
    from research_experiments.families.consensagent.config import phase_metadata

    phase_metadata(experiment, phase_name)
    total_calls = 0
    total_predictions = 0
    for benchmark in benchmarks:
        split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
        sample_count = len(load_split_ids(benchmark.cache_namespace or benchmark.slug, split_name))
        for setup in setups:
            protocol = load_protocol_config(setup.protocol)
            roster = load_roster_config(setup.roster)
            # 估算：初始轮 + 最大辩论轮数
            total_calls += sample_count * roster.agent_count * (1 + protocol.max_debate_rounds)
            total_predictions += sample_count
        for name in matched_control_names:
            total_calls += sample_count * controls[name].budget_calls
            total_predictions += sample_count
    return total_calls, total_predictions


def _active_setups(experiment: ConsensagentExperimentConfig, phase_name: str) -> list[ExperimentSetup]:
    """解析当前 phase 实际启用的 setup 列表。"""
    from research_experiments.families.consensagent.config import phase_metadata

    phase = phase_metadata(experiment, phase_name)
    requested = set(phase["setups"])
    available = {item.name: item for item in experiment.setups}
    missing = sorted(requested - set(available))
    if missing:
        raise RuntimeError(f"Unknown CONSENSAGENT setups for phase {phase_name}: {', '.join(missing)}")
    return [available[name] for name in phase["setups"]]


def _resolve_split_name(experiment: ConsensagentExperimentConfig, phase_name: str, benchmark_slug: str) -> str:
    """解析当前 benchmark 在该 phase 下对应的冻结 split 名称。"""
    return resolve_phase_split_name(experiment, phase_name, benchmark_slug)


def _load_selected_samples(benchmark, split_name: str) -> list[DatasetSample]:
    """按冻结 split 选择本轮要跑的样本。"""
    return select_samples(benchmark, split_name)


def _validate_consensagent_output(assistant_text: str, provider_reasoning_text: str) -> dict[str, Any]:
    """解析 JSON 格式输出。"""
    text = str(assistant_text or "").strip() or str(provider_reasoning_text or "").strip()
    if not text:
        raise ValueError("Model output is empty.")

    payload = _try_parse_json(text)
    if payload is None or not isinstance(payload, dict):
        raise ValueError("Failed to parse model output as JSON.")

    final_answer = str(
        payload.get("final_answer") or payload.get("answer") or payload.get("prediction") or ""
    ).strip()
    reasoning = str(
        payload.get("reasoning") or payload.get("explanation") or ""
    ).strip()
    confidence = 0.5
    with suppress(ValueError, TypeError):
        confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.5))))

    if not final_answer:
        list_val = payload.get("list") or payload.get("answers") or payload.get("titles")
        if isinstance(list_val, list):
            final_answer = ", ".join(str(item).strip() for item in list_val if str(item).strip())

    if not final_answer:
        raise ValueError("Could not extract answer from JSON output.")

    return {"final_answer": final_answer, "reasoning": reasoning, "confidence": confidence}


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


def _run_consensagent_batch(
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
        _run_consensagent_sample,
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


def _ratio(numerator: int, denominator: int) -> float:
    """安全计算比例。"""
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)


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
        weighted_prediction=final_vote,
        weighted_score=final_score,
        weighted_vote_counts=vote_counts,
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
        trigger_type=None,
        trigger_round=None,
        sycophancy_rate=0.0,
        initial_consistency_score=1.0,
        final_consistency_score=1.0,
    )) | {"vote_counts": vote_counts}


# ── Phase 3 辅助函数 ─────────────────────────────────────────────────────


def _optimized_system_prompt(persona_instruction: str = "") -> str:
    """Phase 3 优化后重辩论的系统提示。"""
    base = persona_instruction or "You are a reasoning agent in a multi-agent debate experiment."
    return (
        f"{base} "
        "The debate prompt has been refined to help you reach better answers. "
        "Consider the refined instructions carefully. "
        "Provide honest, well-calibrated confidence scores between 0.0 and 1.0."
    )


def _format_optimized_debate_prompt(
    *,
    sample,
    agent_id: int,
    optimized_prompt: str,
    previous_answer: str,
    previous_reasoning: str,
    previous_confidence: float,
    peer_messages: list[dict[str, Any]],
) -> str:
    """构造 Phase 3 优化后的辩论用户 prompt。"""
    context_block = f"\nContext:\n{sample.prompt_context}" if sample.prompt_context else ""
    peer_block = "\n".join(
        f"Agent {m['agent']} said the answer is {m['answer']} "
        f"and their explanation is {m['reasoning']} with confidence {m['confidence']:.2f}"
        for m in peer_messages
    ) if peer_messages else "(No peer messages)"

    return (
        f"{optimized_prompt}{context_block}\n\n"
        f"Your previous answer: {previous_answer}\n"
        f"Your previous reasoning: {previous_reasoning}\n"
        f"Your previous confidence: {previous_confidence:.2f}\n\n"
        f"Other agents' responses:\n{peer_block}\n\n"
        f"Update your response if the refined prompt leads you to a different conclusion.\n"
        "Return exactly one JSON object with keys \"final_answer\", \"reasoning\", and \"confidence\". Return JSON only."
    )
