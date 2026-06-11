"""无通信对照组的共享执行辅助。

本模块统一维护 cot / sc / mv 等无通信对照的 prompt 选择、seed 规则与单样本执行链路。
批处理入口按样本完成顺序流式产出，调用方可以在每个样本完成后立即写盘并刷新进度。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import Any

from research_experiments.core.controls.control_prompts import (
    build_cot_messages,
    build_mv_messages,
)
from research_experiments.core.data.datasets import DatasetSample
from research_experiments.core.data.evaluation import aggregate_majority, score_prediction
from research_experiments.core.execution.runner_common import iter_indexed_batch

ExecuteTurnFn = Callable[..., dict[str, Any]]
BuildPredictionRowFn = Callable[..., dict[str, Any]]
ControlSampleResult = tuple[int, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]


def resolve_unified_control_message_builder(
    method_family: str,
) -> Callable[[DatasetSample, int, str | None], list[dict[str, str]]]:
    """按共享 no-comm comparator 口径解析标准 prompt 构建器。"""

    family = str(method_family or "").strip().lower()
    if family in ("cot", "chain_of_thought", "self_consistency"):
        return build_cot_messages
    if family in ("majority_vote", "mv"):
        return build_mv_messages
    return build_cot_messages


def resolve_unified_control_seed(*, global_seed: int, method_family: str, replicate_id: int) -> int:
    """按共享 no-comm comparator 规则生成单次采样 seed。"""

    if str(method_family or "").strip().lower() == "cot":
        return global_seed
    return global_seed + replicate_id


def run_unified_control_batch(
    *,
    samples: Iterable[DatasetSample],
    control_name: str,
    method,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    backbone,
    provider,
    cache,
    throttle,
    global_seed: int,
    max_concurrent_requests: int,
    execute_turn: ExecuteTurnFn,
    build_prediction_row: BuildPredictionRowFn,
) -> Iterator[ControlSampleResult]:
    """流式执行无通信对照，确保跨实验家族公平复用同一 prompt 与 seed 规则。"""

    build_messages = resolve_unified_control_message_builder(str(getattr(method, "family", "") or ""))
    def worker(sample: DatasetSample) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return run_unified_control_sample(
            run_id=run_id,
            benchmark_slug=benchmark_slug,
            split_name=split_name,
            sample=sample,
            control_name=control_name,
            method=method,
            backbone=backbone,
            provider=provider,
            cache=cache,
            throttle=throttle,
            global_seed=global_seed,
            build_messages=build_messages,
            execute_turn=execute_turn,
            build_prediction_row=build_prediction_row,
        )
    for sample_index, result in iter_indexed_batch(
        samples,
        worker=worker,
        max_concurrent_requests=max_concurrent_requests,
    ):
        turn_rows, prediction_row = result
        yield sample_index, turn_rows, [], prediction_row


def run_unified_control_sample(
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    sample: DatasetSample,
    control_name: str,
    method,
    backbone,
    provider,
    cache,
    throttle,
    global_seed: int,
    execute_turn: ExecuteTurnFn,
    build_prediction_row: BuildPredictionRowFn,
    build_messages: Callable[[DatasetSample, int, str | None], list[dict[str, str]]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """执行单题无通信对照，返回调用轨迹与最终预测。"""

    turn_rows: list[dict[str, Any]] = []
    resolved_build_messages = build_messages or resolve_unified_control_message_builder(
        str(getattr(method, "family", "") or "")
    )
    for replicate_id in range(method.budget_calls):
        messages = resolved_build_messages(sample, replicate_id + 1, None)
        seed = resolve_unified_control_seed(
            global_seed=global_seed,
            method_family=str(getattr(method, "family", "") or ""),
            replicate_id=replicate_id,
        )
        turn_rows.append(
            execute_turn(
                run_id=run_id,
                dataset=benchmark_slug,
                split_name=split_name,
                sample=sample,
                method_name=control_name,
                method_type="control",
                round_index=0,
                agent_id=replicate_id + 1,
                role="control",
                visible_peer_count=0,
                messages=messages,
                backbone=backbone,
                provider=provider,
                cache=cache,
                throttle=throttle,
                temperature=method.temperature,
                top_p=method.top_p,
                seed=seed,
            )
        )

    answers = [row["normalized_answer"] for row in turn_rows]
    final_vote, vote_counts = aggregate_majority(answers)
    final_score = score_prediction(benchmark_slug, final_vote, sample.reference_answer)
    final_consensus = len(set(answers)) == 1
    prediction_row = build_prediction_row(
        control_name=control_name,
        method=method,
        sample=sample,
        final_vote=final_vote,
        final_score=final_score,
        vote_counts=vote_counts,
        final_consensus=final_consensus,
        turn_rows=turn_rows,
        backbone=backbone,
        benchmark_slug=benchmark_slug,
        split_name=split_name,
        run_id=run_id,
    )
    return turn_rows, prediction_row
