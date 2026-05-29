"""无通信对照组的共享执行辅助。

本模块把"等预算、单求解器、彼此不通信"的对照执行链路统一下沉，
让 multi_agent 等实验家族可以直接复用 cot / sc / mv 对照，
而不必在各自目录里重复维护一套运行时细节。
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Any

from research_experiments.core.controls.control_prompts import (
    build_cot_messages,
    build_mv_messages,
)
from research_experiments.core.data.datasets import DatasetSample
from research_experiments.core.data.evaluation import aggregate_majority, score_prediction

ExecuteTurnFn = Callable[..., dict[str, Any]]
BuildPredictionRowFn = Callable[..., dict[str, Any]]


def run_unified_control_batch(
    *,
    samples: list[DatasetSample],
    control_name: str,
    method,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    backbone,
    provider,
    cache,
    limiter,
    global_seed: int,
    max_concurrent_requests: int,
    execute_turn: ExecuteTurnFn,
    build_prediction_row: BuildPredictionRowFn,
) -> list[tuple[int, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]]:
    """使用统一 prompt 执行无通信对照，确保跨实验公平对比。

    此函数自动根据 method.family 选择对应的统一 prompt 模板，
    不依赖任何实验家族的 build_messages。
    """
    # 根据方法族选择统一的 prompt 构建函数
    family = str(getattr(method, "family", "") or "").strip().lower()
    if family in ("cot", "chain_of_thought", "self_consistency"):
        # self_consistency 使用与 cot 相同的 prompt，多次采样后做多数投票
        build_messages = build_cot_messages
    elif family in ("majority_vote", "mv"):
        build_messages = build_mv_messages
    else:
        # 对于未知的 method family，默认使用 CoT prompt
        build_messages = build_cot_messages

    worker = partial(
        _run_control_sample,
        run_id=run_id,
        benchmark_slug=benchmark_slug,
        split_name=split_name,
        control_name=control_name,
        method=method,
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        global_seed=global_seed,
        build_messages=build_messages,
        execute_turn=execute_turn,
        build_prediction_row=build_prediction_row,
    )
    max_workers = max(1, min(max_concurrent_requests, len(samples) or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(worker, sample=sample): sample_index
            for sample_index, sample in enumerate(samples)
        }
        completed: list[tuple[int, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]] = []
        for future in as_completed(future_to_index):
            sample_index = future_to_index[future]
            turn_rows, prediction_row = future.result()
            completed.append((sample_index, turn_rows, [], prediction_row))
    completed.sort(key=lambda item: item[0])
    return completed


def _run_control_sample(
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
    limiter,
    global_seed: int,
    build_messages: Callable[[DatasetSample, int, str | None], list[dict[str, str]]],
    execute_turn: ExecuteTurnFn,
    build_prediction_row: BuildPredictionRowFn,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """执行单题无通信对照，返回调用轨迹与最终预测。"""
    turn_rows: list[dict[str, Any]] = []
    for replicate_id in range(method.budget_calls):
        messages = build_messages(sample, replicate_id + 1, None)
        seed = global_seed if method.family == "cot" else global_seed + replicate_id
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
                limiter=limiter,
                temperature=method.temperature,
                top_p=method.top_p,
                max_output_tokens=method.max_output_tokens,
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
