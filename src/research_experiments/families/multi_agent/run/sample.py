"""多智能体实验主运行链路。

本模块把 Vanilla MAD 及其等预算控制方法组织成完整实验流程，
包括共享样本选择、setup 解析、agent turn 执行、debate 消息落盘、
题级投票聚合、成本拆分与最终报告/校验产物生成。
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any

from research_experiments.core.data.datasets import DatasetSample, load_split_ids, select_samples
from research_experiments.core.data.evaluation import normalize_prediction
from research_experiments.core.execution.cache import RequestCache
from research_experiments.core.execution.providers import OpenAICompatibleProvider
from research_experiments.core.execution.rate_limits import RequestThrottle
from research_experiments.core.execution.runner_common import (
    iter_indexed_batch,
)
from research_experiments.core.execution.runtime import RunProgressTracker
from research_experiments.families.multi_agent.config import (
    ExperimentSetup,
    MultiAgentExperimentConfig,
    ProtocolConfig,
    RosterConfig,
    load_protocol_config,
    load_roster_config,
)
from research_experiments.family_runtime.common import resolve_phase_split_name
from research_experiments.family_runtime.comparator_impls import (
    build_shared_output_protocol_diagnostics,
    build_shared_vanilla_mad_prediction,
    run_shared_vanilla_mad_rounds,
)
from research_experiments.family_runtime.config_helpers import phase_metadata
from research_experiments.family_runtime.output_protocols import (
    FREE_TEXT_ANSWER_PROTOCOL_V1,
    execute_output_protocol_turn,
)
from research_experiments.family_runtime.vanilla_mad_prompting import (
    CONSISTENT_FREE_TEXT_PROMPT_VERSION,
    prompt_version_uses_json_response_format,
)


@dataclass(frozen=True)
class AgentTurnRecord:
    """单个 agent 在某一轮的执行记录。"""

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
    protocol_failures_per_question: int
    reason_missing_turns_per_question: int


def _run_mad_setup_batch(
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    samples: list[DatasetSample],
    setup: ExperimentSetup,
    protocol: ProtocolConfig,
    roster: RosterConfig,
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    throttle: RequestThrottle,
    global_seed: int,
    prompt_version: str,
    initial_output_protocol: str,
    debate_output_protocol: str,
    max_concurrent_requests: int,
) -> Iterator[tuple[int, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]]:
    """并发执行同一 setup 下的全部样本。

    每个样本内部仍严格保持 Vanilla MAD 的回合顺序，只在样本之间做并发。
    """
    worker = partial(
        _run_mad_sample,
        run_id=run_id,
        benchmark_slug=benchmark_slug,
        split_name=split_name,
        setup=setup,
        protocol=protocol,
        roster=roster,
        backbone=backbone,
        provider=provider,
        cache=cache,
        throttle=throttle,
        global_seed=global_seed,
        prompt_version=prompt_version,
        initial_output_protocol=initial_output_protocol,
        debate_output_protocol=debate_output_protocol,
    )
    for sample_index, result in iter_indexed_batch(
        samples,
        worker=worker,
        max_concurrent_requests=max_concurrent_requests,
    ):
        yield (sample_index, *result)


def _write_sample_outputs(
    sample_results: Iterable[tuple[int, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]],
    dataset_slug: str,
    progress: RunProgressTracker,
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


def _run_mad_sample(
    sample: DatasetSample,
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    setup: ExperimentSetup,
    protocol: ProtocolConfig,
    roster: RosterConfig,
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    throttle: RequestThrottle,
    global_seed: int,
    prompt_version: str,
    initial_output_protocol: str,
    debate_output_protocol: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """运行单个样本上的 Vanilla MAD 协议。"""
    shared_result = run_shared_vanilla_mad_rounds(
        sample=sample,
        run_id=run_id,
        dataset=benchmark_slug,
        split_name=split_name,
        method_name=setup.name,
        agent_count=roster.agent_count,
        debate_rounds=protocol.debate_rounds,
        initial_temperature=protocol.initial_temperature,
        debate_temperature=protocol.debate_temperature,
        top_p=protocol.top_p,
        global_seed=global_seed,
        prompt_version=prompt_version,
        execute_turn=lambda **kwargs: _execute_turn(
            run_id=run_id,
            dataset=benchmark_slug,
            split_name=split_name,
            sample=sample,
            method_name=setup.name,
            method_type="mad",
            backbone=backbone,
            provider=provider,
            cache=cache,
            throttle=throttle,
            initial_output_protocol=initial_output_protocol,
            debate_output_protocol=debate_output_protocol,
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
            )
        ),
    )
    prediction_row = build_shared_vanilla_mad_prediction(
        run_id=run_id,
        dataset=benchmark_slug,
        split_name=split_name,
        sample=sample,
        method_name=setup.name,
        method_type="mad",
        model_name=backbone.name,
        result=shared_result,
    )
    return shared_result["turn_rows"], shared_result["debate_rows"], prediction_row


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
    prediction_row = asdict(
        FinalPredictionRecord(
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
            debate_rounds=0,
            agent_count=1 if method.family == "cot" else method.budget_calls,
            final_consensus=final_consensus,
            initial_disagreement=False,
            vote_flipped=False,
            corrected_by_debate=False,
            harmed_by_debate=False,
            unchanged_correct=final_score == 1.0,
            unchanged_wrong=final_score < 1.0,
            protocol_failures_per_question=sum(
                1 for row in turn_rows if row.get("protocol_parse_status") == "failed"
            ),
            reason_missing_turns_per_question=sum(1 for row in turn_rows if not row.get("reason_present")),
        )
    )
    prediction_row["vote_counts"] = vote_counts
    return prediction_row


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
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    throttle: RequestThrottle,
    temperature: float,
    top_p: float,
    seed: int,
    prompt_version: str = CONSISTENT_FREE_TEXT_PROMPT_VERSION,
    initial_output_protocol: str = FREE_TEXT_ANSWER_PROTOCOL_V1,
    debate_output_protocol: str = FREE_TEXT_ANSWER_PROTOCOL_V1,
) -> dict[str, Any]:
    """执行单次 agent turn，并统一返回日志行结构。"""
    output_protocol = debate_output_protocol if role == "debate" else initial_output_protocol
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
            payload=result.payload,
            assistant_text=result.response_payload.get("assistant_text", ""),
            provider_reasoning_text=result.response_payload.get("provider_reasoning_text", ""),
            validated_output=result.validated_output,
        )
    ) | {"normalized_answer": normalized}


def _build_metrics(
    prediction_rows: list[dict[str, Any]],
    experiment: MultiAgentExperimentConfig,
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
            "protocol_failures_per_question_mean": sum(
                float(item.get("protocol_failures_per_question") or 0.0) for item in rows
            )
            / len(rows),
            "reason_missing_turns_per_question_mean": sum(
                float(item.get("reason_missing_turns_per_question") or 0.0) for item in rows
            )
            / len(rows),
            "latency_ms_mean": sum(float(item["latency_ms_per_question"]) for item in rows) / len(rows),
            "accuracy_per_1k_tokens": (accuracy / total_tokens_mean * 1000) if total_tokens_mean else 0.0,
            "debate_rounds": rows[0]["debate_rounds"],
            "agent_count": rows[0]["agent_count"],
        }
        if method_name in setup_map:
            controls = setup_map[method_name].matched_controls
            row["matched_vote_control"] = next((name for name in controls if name.startswith("mv_")), None)
        summary.append(row)

    by_lookup = {(row["dataset"], row["model_name"], row["method_name"]): row for row in summary}
    for row in summary:
        if row["method_type"] != "mad":
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
        raw_total_tokens = float(row.get("raw_total_tokens") or total_tokens)
        bucket["prompt_tokens"] += float(row["prompt_tokens"])
        bucket["completion_tokens"] += float(row["completion_tokens"])
        bucket["total_tokens"] += total_tokens
        bucket["latency_ms"] += float(row["latency_ms"])
        bucket["turn_count"] += 1
        if row["role"] == "initial":
            bucket["initial_tokens"] += raw_total_tokens
        elif row["role"] == "debate":
            bucket["debate_tokens"] += raw_total_tokens
        else:
            bucket["control_tokens"] += raw_total_tokens

    rows = []
    for (dataset, method_name, method_type), bucket in sorted(grouped.items()):
        rows.append({"dataset": dataset, "method_name": method_name, "method_type": method_type} | bucket)
    return {"rows": rows}


def _build_debate_diagnostics(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """构建 debate 诊断指标。"""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in prediction_rows:
        if row["method_type"] != "mad":
            continue
        grouped.setdefault((row["dataset"], row["method_name"]), []).append(row)

    rows = []
    for (dataset, method_name), rows_for_key in sorted(grouped.items()):
        total = len(rows_for_key)
        rows.append(
            {
                "dataset": dataset,
                "method_name": method_name,
                "question_count": total,
                "initial_disagreement_rate": _ratio(
                    sum(1 for row in rows_for_key if row["initial_disagreement"]), total
                ),
                "post_debate_consensus_rate": _ratio(sum(1 for row in rows_for_key if row["final_consensus"]), total),
                "vote_flip_rate": _ratio(sum(1 for row in rows_for_key if row["vote_flipped"]), total),
                "wrong_consensus_rate": _ratio(
                    sum(1 for row in rows_for_key if row["final_consensus"] and float(row["score"]) < 1.0),
                    total,
                ),
            }
        )
    return {"rows": rows}


def _build_output_protocol_diagnostics(
    turn_rows: list[dict[str, Any]],
    *,
    dataset_order: list[str],
    method_order: list[str],
) -> dict[str, Any]:
    return build_shared_output_protocol_diagnostics(
        turn_rows,
        dataset_order=dataset_order,
        method_order=method_order,
    )


def _estimate_work(
    experiment: MultiAgentExperimentConfig,
    phase_name: str,
    benchmarks,
    setups: list[ExperimentSetup],
    matched_control_names: list[str],
    controls,
) -> tuple[int, int]:
    """估算本次多智能体运行的总调用量与总预测量。"""
    phase_metadata(experiment, phase_name)
    total_calls = 0
    total_predictions = 0
    for benchmark in benchmarks:
        split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
        sample_count = len(load_split_ids(benchmark.cache_namespace or benchmark.slug, split_name))
        for setup in setups:
            protocol = load_protocol_config(setup.protocol)
            roster = load_roster_config(setup.roster)
            total_calls += sample_count * roster.agent_count * (1 + protocol.debate_rounds)
            total_predictions += sample_count
        for name in matched_control_names:
            total_calls += sample_count * controls[name].budget_calls
            total_predictions += sample_count
    return total_calls, total_predictions


def _active_setups(experiment: MultiAgentExperimentConfig, phase_name: str) -> list[ExperimentSetup]:
    """解析当前 phase 实际启用的 setup 列表。"""
    phase = phase_metadata(experiment, phase_name)
    requested = set(phase["setups"])
    available = {item.name: item for item in experiment.setups}
    missing = sorted(requested - set(available))
    if missing:
        raise RuntimeError(f"Unknown multi-agent setups for phase {phase_name}: {', '.join(missing)}")
    return [available[name] for name in phase["setups"]]


def _resolve_split_name(experiment: MultiAgentExperimentConfig, phase_name: str, benchmark_slug: str) -> str:
    """解析当前 benchmark 在该 phase 下对应的冻结 split 名称。"""
    return resolve_phase_split_name(experiment, phase_name, benchmark_slug)


def _load_selected_samples(benchmark, split_name: str) -> list[DatasetSample]:
    """按冻结 split 选择本轮要跑的样本。"""
    return select_samples(benchmark, split_name)


def _ratio(numerator: int, denominator: int) -> float:
    """安全计算比例。"""
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)
