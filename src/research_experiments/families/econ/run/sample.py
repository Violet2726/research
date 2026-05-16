"""ECON family 的题级执行、聚合与指标构造。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from research_experiments.core.data.datasets import DatasetSample, load_split_ids, select_samples
from research_experiments.core.data.evaluation import normalize_prediction, score_prediction
from research_experiments.core.execution.runner_common import execute_cached_turn, run_indexed_batch
from research_experiments.core.structured_outputs import (
    SCHEMA_ANSWER_WITH_PROXY_SIGNALS_BUDGET,
    SCHEMA_BELIEF_UPDATE_DELTA,
)
from research_experiments.families.econ.algorithms import (
    ACTION_ORDER,
    aggregate_confidence_weighted,
    aggregate_majority_with_counts,
    apply_belief_answer_safeguard,
    build_belief_state,
    pick_action_packets,
)
from research_experiments.families.econ.config import EconExperimentConfig, EconMethodSpec, ProtocolConfig
from research_experiments.families.econ.prompts import (
    build_agent_messages,
    build_belief_update_messages,
    build_single_agent_messages,
)
from research_experiments.families.shared.common import build_question_preview, resolve_phase_split_name, safe_mean, sum_metric


METHOD_ORDER = (
    "single_agent_cot",
    "vote_mv3",
    "econ_full_comm_r1",
    "econ_bne_main",
)

DISPLAY_NAME = {
    "single_agent_cot": "single_agent_cot",
    "vote_mv3": "vote_mv3",
    "econ_full_comm_r1": "econ_full_comm_r1",
    "econ_bne_main": "econ_bne_main",
}


@dataclass(frozen=True)
class SampleResult:
    """单题运行产生的全部中间产物与最终结果。"""

    turn_rows: list[dict[str, Any]]
    belief_rows: list[dict[str, Any]]
    equilibrium_rows: list[dict[str, Any]]
    communication_rows: list[dict[str, Any]]
    prediction_rows: list[dict[str, Any]]


def _active_methods(experiment: EconExperimentConfig) -> list[EconMethodSpec]:
    """返回按稳定顺序排列的生效方法。"""

    methods = {method.name: method for method in experiment.methods}
    return [methods[name] for name in METHOD_ORDER if name in methods]


def _estimate_work(
    experiment: EconExperimentConfig,
    phase_name: str,
    benchmarks,
    protocol: ProtocolConfig,
    methods: list[EconMethodSpec],
) -> tuple[int, int]:
    """估算总调用数与总预测数，用于进度展示。"""

    total_calls = 0
    total_predictions = 0
    for benchmark in benchmarks:
        split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
        sample_count = len(load_split_ids(benchmark.cache_namespace or benchmark.slug, split_name))
        # 上界：single_agent 1 次 + stage_a 3 次 + full_comm 3 次 + econ_bne 3 次。
        total_calls += sample_count * (1 + protocol.agent_count + protocol.agent_count + protocol.agent_count)
        total_predictions += sample_count * len(methods)
    return total_calls, total_predictions


def _resolve_split_name(experiment: EconExperimentConfig, phase_name: str, benchmark_slug: str) -> str:
    """解析某个 benchmark 在当前 phase 下使用的冻结 split 名称。"""

    return resolve_phase_split_name(experiment, phase_name, benchmark_slug)


def _load_selected_samples(benchmark, split_name: str) -> list[DatasetSample]:
    """按冻结 split 选出本轮样本。"""

    return select_samples(benchmark, split_name)


def _run_sample_batch(
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    samples: list[DatasetSample],
    protocol: ProtocolConfig,
    experiment: EconExperimentConfig,
    backbone,
    provider,
    cache,
    limiter,
) -> list[SampleResult]:
    """并发执行同一数据集下的一整批样本，并保持原始样本顺序。"""

    worker = partial(
        _run_sample,
        run_id=run_id,
        benchmark_slug=benchmark_slug,
        split_name=split_name,
        protocol=protocol,
        experiment=experiment,
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
    )
    return [
        result
        for _, result in run_indexed_batch(
            samples,
            worker=worker,
            max_concurrent_requests=experiment.max_concurrent_requests,
        )
    ]


def _run_sample(
    sample: DatasetSample,
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    protocol: ProtocolConfig,
    experiment: EconExperimentConfig,
    backbone,
    provider,
    cache,
    limiter,
) -> SampleResult:
    """执行单题上的全部 ECON 方法。"""

    question_preview = build_question_preview(sample.question)
    sample_seed = experiment.global_seed + _stable_sample_seed(sample.sample_id)
    turn_rows: list[dict[str, Any]] = []
    belief_rows: list[dict[str, Any]] = []
    equilibrium_rows: list[dict[str, Any]] = []
    communication_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    single_turn = _execute_turn(
        run_id=run_id,
        dataset=benchmark_slug,
        split_name=split_name,
        sample=sample,
        question_preview=question_preview,
        method_name="single_agent_cot",
        round_index=0,
        agent_id=1,
        role="single_agent",
        visible_peer_count=0,
        messages=build_single_agent_messages(sample, prompt_version=experiment.prompt_version),
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        temperature=protocol.initial_temperature,
        top_p=protocol.top_p,
        max_output_tokens=protocol.max_output_tokens,
        seed=sample_seed,
        output_mode=SCHEMA_ANSWER_WITH_PROXY_SIGNALS_BUDGET,
    )
    turn_rows.append(single_turn)
    prediction_rows.append(
        _build_single_agent_prediction_row(
            single_turn=single_turn,
            sample=sample,
            run_id=run_id,
            benchmark_slug=benchmark_slug,
            split_name=split_name,
            backbone=backbone,
        )
    )

    stage_a_rows: list[dict[str, Any]] = []
    for agent_id in range(1, protocol.agent_count + 1):
        row = _execute_turn(
            run_id=run_id,
            dataset=benchmark_slug,
            split_name=split_name,
            sample=sample,
            question_preview=question_preview,
            method_name="shared_stage_a",
            round_index=0,
            agent_id=agent_id,
            role="stage_a_solver",
            visible_peer_count=0,
            messages=build_agent_messages(sample, agent_id, prompt_version=experiment.prompt_version),
            backbone=backbone,
            provider=provider,
            cache=cache,
            limiter=limiter,
            temperature=protocol.initial_temperature,
            top_p=protocol.top_p,
            max_output_tokens=protocol.max_output_tokens,
            seed=sample_seed + agent_id,
            output_mode=SCHEMA_ANSWER_WITH_PROXY_SIGNALS_BUDGET,
        )
        stage_a_rows.append(row)
    turn_rows.extend(stage_a_rows)

    initial_vote_answer, initial_vote_counts = aggregate_majority_with_counts(stage_a_rows)
    initial_vote_score = score_prediction(benchmark_slug, initial_vote_answer, sample.reference_answer)
    prediction_rows.append(
        _build_vote_prediction_row(
            stage_a_rows=stage_a_rows,
            sample=sample,
            run_id=run_id,
            benchmark_slug=benchmark_slug,
            split_name=split_name,
            backbone=backbone,
            initial_vote_answer=initial_vote_answer,
            initial_vote_score=initial_vote_score,
            initial_vote_counts=initial_vote_counts,
        )
    )

    belief_state = build_belief_state(stage_a_rows, protocol=protocol)
    belief_rows.append(
        {
            "run_id": run_id,
            "dataset": benchmark_slug,
            "split": split_name,
            "sample_id": sample.sample_id,
            "method_name": "econ_bne_main",
            **{key: value for key, value in belief_state.items() if key not in {"packet_lookup", "sample_packet_rows"}},
        }
    )
    equilibrium_rows.append(
        {
            "run_id": run_id,
            "dataset": benchmark_slug,
            "split": split_name,
            "sample_id": sample.sample_id,
            "method_name": "econ_bne_main",
            "selected_action": belief_state["selected_action"],
            "action_scores": belief_state["action_scores"],
            "belief_score": next(
                float(row["belief_score"])
                for row in belief_state["action_scores"]
                if row["action"] == belief_state["selected_action"]
            ),
            "expected_gain": next(
                float(row["expected_gain"])
                for row in belief_state["action_scores"]
                if row["action"] == belief_state["selected_action"]
            ),
            "communication_cost": next(
                float(row["communication_cost"])
                for row in belief_state["action_scores"]
                if row["action"] == belief_state["selected_action"]
            ),
        }
    )

    full_packets_by_agent = pick_action_packets(
        stage_a_rows,
        belief_state["packet_lookup"],
        selected_action="econ_full_comm_r1",
    )
    full_turn_rows, full_comm_rows, full_prediction = _run_coordination_method(
        method_name="econ_full_comm_r1",
        sample=sample,
        run_id=run_id,
        benchmark_slug=benchmark_slug,
        split_name=split_name,
        backbone=backbone,
        protocol=protocol,
        initial_vote_answer=initial_vote_answer,
        initial_vote_score=initial_vote_score,
        initial_vote_counts=initial_vote_counts,
        stage_a_rows=stage_a_rows,
        selected_action="query_all_peers",
        selected_action_score=None,
        packets_by_agent=full_packets_by_agent,
        provider=provider,
        cache=cache,
        limiter=limiter,
        experiment=experiment,
        question_preview=question_preview,
        sample_seed=sample_seed,
        coordination_mode="full_comm_r1",
    )
    turn_rows.extend(full_turn_rows)
    communication_rows.extend(full_comm_rows)
    prediction_rows.append(full_prediction)

    bne_packets_by_agent = pick_action_packets(
        stage_a_rows,
        belief_state["packet_lookup"],
        selected_action=str(belief_state["selected_action"]),
    )
    bne_turn_rows, bne_comm_rows, bne_prediction = _run_coordination_method(
        method_name="econ_bne_main",
        sample=sample,
        run_id=run_id,
        benchmark_slug=benchmark_slug,
        split_name=split_name,
        backbone=backbone,
        protocol=protocol,
        initial_vote_answer=initial_vote_answer,
        initial_vote_score=initial_vote_score,
        initial_vote_counts=initial_vote_counts,
        stage_a_rows=stage_a_rows,
        selected_action=str(belief_state["selected_action"]),
        selected_action_score=next(
            row for row in belief_state["action_scores"] if row["action"] == belief_state["selected_action"]
        ),
        packets_by_agent=bne_packets_by_agent,
        provider=provider,
        cache=cache,
        limiter=limiter,
        experiment=experiment,
        question_preview=question_preview,
        sample_seed=sample_seed,
        coordination_mode="belief_equilibrium",
    )
    turn_rows.extend(bne_turn_rows)
    communication_rows.extend(bne_comm_rows)
    prediction_rows.append(bne_prediction)
    return SampleResult(
        turn_rows=turn_rows,
        belief_rows=belief_rows,
        equilibrium_rows=equilibrium_rows,
        communication_rows=communication_rows,
        prediction_rows=prediction_rows,
    )


def _run_coordination_method(
    *,
    method_name: str,
    sample: DatasetSample,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    backbone,
    protocol: ProtocolConfig,
    initial_vote_answer: str,
    initial_vote_score: float,
    initial_vote_counts: dict[str, int],
    stage_a_rows: list[dict[str, Any]],
    selected_action: str,
    selected_action_score: dict[str, Any] | None,
    packets_by_agent: dict[int, list[dict[str, Any]]],
    provider,
    cache,
    limiter,
    experiment: EconExperimentConfig,
    question_preview: str,
    sample_seed: int,
    coordination_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """执行一条协调方法，并返回 belief update 行、通信 trace 和题级预测。"""

    belief_turn_rows: list[dict[str, Any]] = []
    communication_rows: list[dict[str, Any]] = []
    if selected_action in {"keep_local", "adopt_vote"}:
        updated_rows = stage_a_rows
    else:
        updated_rows = []
        for stage_a_row in stage_a_rows:
            agent_id = int(stage_a_row["agent_id"])
            selected_peer_packets = packets_by_agent.get(agent_id, [])
            belief_row = _execute_turn(
                run_id=run_id,
                dataset=benchmark_slug,
                split_name=split_name,
                sample=sample,
                question_preview=question_preview,
                method_name=method_name,
                round_index=1,
                agent_id=agent_id,
                role="belief_update",
                visible_peer_count=len(selected_peer_packets),
                messages=build_belief_update_messages(
                    sample,
                    agent_id,
                    previous_answer=str(stage_a_row.get("normalized_answer") or ""),
                    previous_reasoning_trace=str(stage_a_row.get("reasoning_trace") or ""),
                    previous_confidence_raw=stage_a_row.get("confidence_raw"),
                    selected_action=selected_action,
                    selected_peer_packets=selected_peer_packets,
                    prompt_version=experiment.prompt_version,
                ),
                backbone=backbone,
                provider=provider,
                cache=cache,
                limiter=limiter,
                temperature=protocol.belief_temperature,
                top_p=protocol.top_p,
                max_output_tokens=protocol.max_output_tokens,
                seed=sample_seed + 100 + agent_id,
                output_mode=SCHEMA_BELIEF_UPDATE_DELTA,
            )
            belief_turn_rows.append(belief_row)
            updated_row = apply_belief_answer_safeguard(
                stage_a_row=stage_a_row,
                belief_row=belief_row,
                selected_peer_packets=selected_peer_packets,
            )
            updated_rows.append(updated_row)
            communication_rows.append(
                {
                    "run_id": run_id,
                    "dataset": benchmark_slug,
                    "split": split_name,
                    "sample_id": sample.sample_id,
                    "method_name": method_name,
                    "agent_id": agent_id,
                    "selected_action": selected_action,
                    "selected_peer_agent_ids": [int(packet["agent_id"]) for packet in selected_peer_packets],
                    "selected_peer_count": len(selected_peer_packets),
                    "selected_peer_packet_tokens": sum(int(packet["approx_packet_tokens"]) for packet in selected_peer_packets),
                    "belief_output_status": belief_row["output_status"],
                    "changed_answer": updated_row["changed_answer"],
                }
            )

    if selected_action == "keep_local":
        best_row = max(
            updated_rows,
            key=lambda row: (
                float(row.get("confidence_value") or 0.0),
                -int(row.get("agent_id") or 10**6),
            ),
        )
        final_answer = str(best_row.get("normalized_answer") or "")
        final_support = {final_answer: 1.0} if final_answer else {}
    elif selected_action == "adopt_vote":
        final_answer = initial_vote_answer
        final_support = {key: float(value) for key, value in initial_vote_counts.items()}
    else:
        final_answer, final_support = aggregate_confidence_weighted(updated_rows)

    final_score = score_prediction(benchmark_slug, final_answer, sample.reference_answer)
    stage_a_tokens = sum(float(row.get("total_tokens") or 0.0) for row in stage_a_rows)
    belief_tokens = sum(float(row.get("total_tokens") or 0.0) for row in belief_turn_rows)
    communication_tokens = belief_tokens
    total_tokens = stage_a_tokens + belief_tokens
    final_prediction = {
        "run_id": run_id,
        "dataset": benchmark_slug,
        "split": split_name,
        "sample_id": sample.sample_id,
        "question_preview": question_preview,
        "model_name": backbone.name,
        "method_name": method_name,
        "display_name": DISPLAY_NAME[method_name],
        "method_type": "coordination",
        "coordination_mode": coordination_mode,
        "gold_answer": sample.reference_answer,
        "initial_answer": initial_vote_answer,
        "final_answer": final_answer,
        "prediction": normalize_prediction(benchmark_slug, final_answer) if final_answer else "",
        "normalized_answer": normalize_prediction(benchmark_slug, final_answer) if final_answer else "",
        "score": final_score,
        "initial_vote_score": initial_vote_score,
        "communication_score": final_score,
        "selected_action": selected_action,
        "belief_score": None if selected_action_score is None else float(selected_action_score["belief_score"]),
        "expected_gain": None if selected_action_score is None else float(selected_action_score["expected_gain"]),
        "communication_cost": None if selected_action_score is None else float(selected_action_score["communication_cost"]),
        "changed_after_coordination": final_answer != initial_vote_answer,
        "prompt_tokens_per_question": round(
            sum(float(row.get("prompt_tokens") or 0.0) for row in stage_a_rows + belief_turn_rows),
            6,
        ),
        "completion_tokens_per_question": round(
            sum(float(row.get("completion_tokens") or 0.0) for row in stage_a_rows + belief_turn_rows),
            6,
        ),
        "total_tokens_per_question": round(total_tokens, 6),
        "communication_tokens_per_question": round(communication_tokens, 6),
        "latency_ms_per_question": round(
            sum(float(row.get("latency_ms") or 0.0) for row in stage_a_rows + belief_turn_rows),
            6,
        ),
        "calls_per_question": len(stage_a_rows) + len(belief_turn_rows),
        "correction_flag": bool(initial_vote_score < final_score),
        "degradation_flag": bool(initial_vote_score > final_score),
        "selected_peer_count_mean": round(safe_mean(row["selected_peer_count"] for row in communication_rows), 6) if communication_rows else 0.0,
    }
    return belief_turn_rows, communication_rows, final_prediction


def _build_single_agent_prediction_row(
    *,
    single_turn: dict[str, Any],
    sample: DatasetSample,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    backbone,
) -> dict[str, Any]:
    """构造单智能体题级预测。"""

    final_answer = str(single_turn.get("normalized_answer") or "")
    score = score_prediction(benchmark_slug, final_answer, sample.reference_answer)
    return {
        "run_id": run_id,
        "dataset": benchmark_slug,
        "split": split_name,
        "sample_id": sample.sample_id,
        "question_preview": str(single_turn.get("question_preview") or ""),
        "model_name": backbone.name,
        "method_name": "single_agent_cot",
        "display_name": DISPLAY_NAME["single_agent_cot"],
        "method_type": "single_agent",
        "coordination_mode": "none",
        "gold_answer": sample.reference_answer,
        "initial_answer": final_answer,
        "final_answer": final_answer,
        "prediction": final_answer,
        "normalized_answer": final_answer,
        "score": score,
        "initial_vote_score": score,
        "communication_score": score,
        "selected_action": "none",
        "belief_score": None,
        "expected_gain": None,
        "communication_cost": 0.0,
        "changed_after_coordination": False,
        "prompt_tokens_per_question": round(float(single_turn.get("prompt_tokens") or 0.0), 6),
        "completion_tokens_per_question": round(float(single_turn.get("completion_tokens") or 0.0), 6),
        "total_tokens_per_question": round(float(single_turn.get("total_tokens") or 0.0), 6),
        "communication_tokens_per_question": 0.0,
        "latency_ms_per_question": round(float(single_turn.get("latency_ms") or 0.0), 6),
        "calls_per_question": 1,
        "correction_flag": False,
        "degradation_flag": False,
        "selected_peer_count_mean": 0.0,
    }


def _build_vote_prediction_row(
    *,
    stage_a_rows: list[dict[str, Any]],
    sample: DatasetSample,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    backbone,
    initial_vote_answer: str,
    initial_vote_score: float,
    initial_vote_counts: dict[str, int],
) -> dict[str, Any]:
    """构造 vote_mv3 题级预测。"""

    total_tokens = sum(float(row.get("total_tokens") or 0.0) for row in stage_a_rows)
    return {
        "run_id": run_id,
        "dataset": benchmark_slug,
        "split": split_name,
        "sample_id": sample.sample_id,
        "question_preview": build_question_preview(sample.question),
        "model_name": backbone.name,
        "method_name": "vote_mv3",
        "display_name": DISPLAY_NAME["vote_mv3"],
        "method_type": "no_comm_vote",
        "coordination_mode": "vote_only",
        "gold_answer": sample.reference_answer,
        "initial_answer": initial_vote_answer,
        "final_answer": initial_vote_answer,
        "prediction": normalize_prediction(benchmark_slug, initial_vote_answer) if initial_vote_answer else "",
        "normalized_answer": normalize_prediction(benchmark_slug, initial_vote_answer) if initial_vote_answer else "",
        "score": initial_vote_score,
        "initial_vote_score": initial_vote_score,
        "communication_score": initial_vote_score,
        "selected_action": "adopt_vote",
        "belief_score": None,
        "expected_gain": None,
        "communication_cost": 0.0,
        "changed_after_coordination": False,
        "prompt_tokens_per_question": round(sum(float(row.get("prompt_tokens") or 0.0) for row in stage_a_rows), 6),
        "completion_tokens_per_question": round(sum(float(row.get("completion_tokens") or 0.0) for row in stage_a_rows), 6),
        "total_tokens_per_question": round(total_tokens, 6),
        "communication_tokens_per_question": 0.0,
        "latency_ms_per_question": round(sum(float(row.get("latency_ms") or 0.0) for row in stage_a_rows), 6),
        "calls_per_question": len(stage_a_rows),
        "correction_flag": False,
        "degradation_flag": False,
        "selected_peer_count_mean": 0.0,
        "vote_counts": initial_vote_counts,
    }


def _write_sample_outputs(
    *,
    sample_results: list[SampleResult],
    dataset_slug: str,
    progress,
    turn_handle,
    belief_handle,
    equilibrium_handle,
    communication_handle,
    prediction_handle,
    all_turns: list[dict[str, Any]],
    all_belief_rows: list[dict[str, Any]],
    all_equilibrium_rows: list[dict[str, Any]],
    all_communication_rows: list[dict[str, Any]],
    final_predictions: list[dict[str, Any]],
) -> None:
    """把题级结果稳定写盘，并同步更新内存聚合与进度快照。"""

    for result in sample_results:
        for row in result.turn_rows:
            turn_handle.write_row(row)
            progress.record_call(row)
        for row in result.belief_rows:
            belief_handle.write_row(row)
        for row in result.equilibrium_rows:
            equilibrium_handle.write_row(row)
        for row in result.communication_rows:
            communication_handle.write_row(row)
        for row in result.prediction_rows:
            prediction_handle.write_row(row)
            progress.record_predictions(1, dataset_slug, str(row["method_name"]))
        all_turns.extend(result.turn_rows)
        all_belief_rows.extend(result.belief_rows)
        all_equilibrium_rows.extend(result.equilibrium_rows)
        all_communication_rows.extend(result.communication_rows)
        final_predictions.extend(result.prediction_rows)


def _build_metrics(
    prediction_rows: list[dict[str, Any]],
    methods: list[EconMethodSpec],
    *,
    model_name: str,
) -> dict[str, Any]:
    """从题级预测构建 summary 指标表。"""

    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    datasets = sorted({str(row["dataset"]) for row in prediction_rows})
    method_names = [method.name for method in methods]
    for dataset in datasets:
        for method_name in method_names:
            subset = [row for row in prediction_rows if row["dataset"] == dataset and row["method_name"] == method_name]
            if not subset:
                continue
            rows.append(_summarize_prediction_rows(subset, dataset=dataset, method_name=method_name, model_name=model_name))
            grouped[(dataset, method_name)] = subset
    for method_name in method_names:
        subset = [row for row in prediction_rows if row["method_name"] == method_name]
        if not subset:
            continue
        rows.append(_summarize_prediction_rows(subset, dataset="overall", method_name=method_name, model_name=model_name))

    by_dataset_method = {(row["dataset"], row["method_name"]): row for row in rows}
    for row in rows:
        vote_row = by_dataset_method.get((row["dataset"], "vote_mv3"))
        full_comm_row = by_dataset_method.get((row["dataset"], "econ_full_comm_r1"))
        if vote_row is not None and row["method_name"] != "vote_mv3":
            row["gain_over_vote_mv3"] = round(float(row["accuracy_mean"]) - float(vote_row["accuracy_mean"]), 6)
        else:
            row["gain_over_vote_mv3"] = None
        if full_comm_row is not None and row["method_name"] != "econ_full_comm_r1":
            denominator = float(full_comm_row["total_tokens_mean"] or 0.0)
            row["token_ratio_over_full_comm"] = round(float(row["total_tokens_mean"]) / denominator, 6) if denominator > 0 else None
        else:
            row["token_ratio_over_full_comm"] = None
    return {"summary": rows, "prediction_count": len(prediction_rows)}


def _summarize_prediction_rows(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    method_name: str,
    model_name: str,
) -> dict[str, Any]:
    """把一组题级预测汇总成一条 summary 行。"""

    action_counts = {action: sum(1 for row in rows if row.get("selected_action") == action) for action in ACTION_ORDER}
    return {
        "dataset": dataset,
        "model_name": model_name,
        "method_name": method_name,
        "display_name": DISPLAY_NAME.get(method_name, method_name),
        "accuracy_mean": round(safe_mean(float(row.get("score") or 0.0) for row in rows), 6),
        "prompt_tokens_mean": round(safe_mean(float(row.get("prompt_tokens_per_question") or 0.0) for row in rows), 6),
        "completion_tokens_mean": round(safe_mean(float(row.get("completion_tokens_per_question") or 0.0) for row in rows), 6),
        "total_tokens_mean": round(safe_mean(float(row.get("total_tokens_per_question") or 0.0) for row in rows), 6),
        "communication_tokens_mean": round(safe_mean(float(row.get("communication_tokens_per_question") or 0.0) for row in rows), 6),
        "latency_ms_mean": round(safe_mean(float(row.get("latency_ms_per_question") or 0.0) for row in rows), 6),
        "calls_per_question_mean": round(safe_mean(float(row.get("calls_per_question") or 0.0) for row in rows), 6),
        "accuracy_per_1k_tokens": _accuracy_per_1k_tokens(rows),
        "belief_score_mean": round(safe_mean(float(row.get("belief_score") or 0.0) for row in rows if row.get("belief_score") is not None), 6),
        "expected_gain_mean": round(safe_mean(float(row.get("expected_gain") or 0.0) for row in rows if row.get("expected_gain") is not None), 6),
        "communication_cost_mean": round(safe_mean(float(row.get("communication_cost") or 0.0) for row in rows if row.get("communication_cost") is not None), 6),
        "changed_after_coordination_rate": round(safe_mean(1.0 if row.get("changed_after_coordination") else 0.0 for row in rows), 6),
        "correction_rate": round(safe_mean(1.0 if row.get("correction_flag") else 0.0 for row in rows), 6),
        "degradation_rate": round(safe_mean(1.0 if row.get("degradation_flag") else 0.0 for row in rows), 6),
        "keep_local_rate": round(action_counts["keep_local"] / len(rows), 6),
        "adopt_vote_rate": round(action_counts["adopt_vote"] / len(rows), 6),
        "query_best_peer_rate": round(action_counts["query_best_peer"] / len(rows), 6),
        "query_two_peers_rate": round(action_counts["query_two_peers"] / len(rows), 6),
        "selected_peer_count_mean": round(safe_mean(float(row.get("selected_peer_count_mean") or 0.0) for row in rows), 6),
        "question_count": len(rows),
    }


def _accuracy_per_1k_tokens(rows: list[dict[str, Any]]) -> float:
    accuracy = safe_mean(float(row.get("score") or 0.0) for row in rows)
    total_tokens = safe_mean(float(row.get("total_tokens_per_question") or 0.0) for row in rows)
    if total_tokens <= 0:
        return 0.0
    return round(accuracy / (total_tokens / 1000.0), 6)


def _execute_turn(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    question_preview: str,
    method_name: str,
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
    output_mode: str,
) -> dict[str, Any]:
    """执行单次 solver 或 belief update 调用，并整理成统一日志结构。"""

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
        schema_id=output_mode,
        dataset=dataset,
    )
    if output_mode == SCHEMA_BELIEF_UPDATE_DELTA:
        answer_for_normalization = str(result.validated_output.get("new_answer") or "")
    else:
        answer_for_normalization = str(result.validated_output.get("final_answer") or "")
    row = {
        "run_id": run_id,
        "dataset": dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "question_preview": question_preview,
        "track_name": "same_context",
        "view_kind": "same_context",
        "method_name": method_name,
        "role": role,
        "round_index": round_index,
        "agent_id": agent_id,
        "visible_peer_count": visible_peer_count,
        "prompt_hash": result.prompt_hash,
        "prediction": normalize_prediction(dataset, answer_for_normalization) if answer_for_normalization else "",
        "normalized_answer": normalize_prediction(dataset, answer_for_normalization) if answer_for_normalization else "",
        "output_status": result.output_status,
        "prompt_tokens": float(result.usage.get("prompt_tokens") or 0.0),
        "completion_tokens": float(result.usage.get("completion_tokens") or 0.0),
        "total_tokens": float(result.usage.get("total_tokens") or 0.0),
        "latency_ms": float(result.response_payload.get("latency_ms") or 0.0),
        "cache_hit": result.cache_hit,
        "request_error": result.request_error,
        "payload": result.payload,
        "assistant_text": result.response_payload.get("assistant_text", ""),
        "provider_reasoning_text": result.response_payload.get("provider_reasoning_text", ""),
        "validated_output": result.validated_output,
    }
    if output_mode == SCHEMA_ANSWER_WITH_PROXY_SIGNALS_BUDGET:
        confidence_raw = result.validated_output.get("confidence_raw") if result.validated_output else None
        confidence_value = _coerce_confidence_value(confidence_raw)
        row.update(
            {
                "final_answer": result.validated_output.get("final_answer") if result.validated_output else None,
                "reasoning_trace": result.validated_output.get("reasoning_trace") if result.validated_output else None,
                "claim_span": result.validated_output.get("claim_span") if result.validated_output else None,
                "key_evidence": result.validated_output.get("key_evidence") if result.validated_output else None,
                "keyword_clues": result.validated_output.get("keyword_clues") if result.validated_output else [],
                "confidence_raw": confidence_raw,
                "confidence_value": confidence_value,
                "confidence_valid": confidence_value is not None,
                "uncertain_point": result.validated_output.get("uncertain_point") if result.validated_output else None,
            }
        )
    else:
        row.update(
            {
                "changed_answer": result.validated_output.get("changed_answer") if result.validated_output else None,
                "new_answer": result.validated_output.get("new_answer") if result.validated_output else None,
                "confidence_delta": result.validated_output.get("confidence_delta") if result.validated_output else None,
                "reason_for_change": result.validated_output.get("reason_for_change") if result.validated_output else None,
                "remaining_disagreement": result.validated_output.get("remaining_disagreement") if result.validated_output else None,
            }
        )
    return row


def _coerce_confidence_value(confidence_raw: object) -> float | None:
    """把 schema 中的置信度字段归一化到 [0, 1]。"""

    if confidence_raw is None:
        return None
    if isinstance(confidence_raw, (int, float)):
        value = float(confidence_raw)
    else:
        text = str(confidence_raw).strip()
        if not text:
            return None
        if text.endswith("%"):
            try:
                value = float(text[:-1].strip()) / 100.0
            except ValueError:
                return None
        else:
            try:
                value = float(text)
            except ValueError:
                return None
    if value > 1.0 and value <= 100.0:
        value = value / 100.0
    return round(min(1.0, max(0.0, value)), 6)


def _stable_sample_seed(sample_id: str) -> int:
    """从样本 ID 派生稳定种子，避免全局随机漂移。"""

    return sum(ord(char) for char in sample_id)

