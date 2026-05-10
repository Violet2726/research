"""`budget_comm` 实验主运行链路。

本模块把 DALA-lite 风格预算通信实验落成完整的可执行流程：
1. 校准每个数据集的轮次预算；
2. 运行共享 Stage A，得到可压缩候选；
3. 分别执行多种预算决策方法；
4. 在 Stage B 中基于中标消息包做 belief update；
5. 汇总指标、诊断与论文摘要产物。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from hashlib import sha256
from pathlib import Path
import csv
import json
import math
import re
from typing import Any

from dotenv import load_dotenv

from budget_comm.config import (
    AuctionPolicyConfig,
    BudgetCommExperimentConfig,
    BudgetProtocolConfig,
    ContextViewConfig,
    load_auction_policy_config,
    load_benchmarks,
    load_context_view_config,
    load_protocol_config,
    phase_metadata,
)
from budget_comm.dataset_views import build_context_views, serialize_view_row
from budget_comm.logic import (
    METHOD_ORDER,
    PACKET_MODE_ORDER,
    apply_belief_update,
    build_all_to_all_full_decision,
    build_budget_confidence_decision,
    build_budget_random_decision,
    build_dala_lite_decision,
    build_shared_candidate_features,
    evaluate_full_dala_gate,
)
from budget_comm.prompting import build_belief_update_messages, build_solver_messages
from experiment_core.foundation.artifacts import BufferedJsonlWriter
from experiment_core.foundation.cache import RequestCache, RequestCacheRouter, build_request_cache_key, cache_successful_response, json_dump
from experiment_core.foundation.datasets import DatasetSample, load_split_ids, select_samples
from experiment_core.foundation.evaluation import aggregate_majority, normalize_prediction, score_prediction
from experiment_core.foundation.providers import OpenAICompatibleProvider, build_payload, execute_completion_request
from experiment_core.foundation.rate_limits import SlidingWindowRateLimiter
from experiment_core.foundation.runtime import RunProgressTracker, build_run_id, finalize_run_outputs
from experiment_core.controls.selective_signals import confidence_display, normalize_confidence
from experiment_core.foundation.structured_output import (
    ARTIFACT_VERSION,
    OUTPUT_MODE_BUDGET_BELIEF_UPDATE,
    OUTPUT_MODE_BUDGET_SOLVER,
    validate_or_recover_structured_output,
)
from experiment_core.foundation.workspace import default_cache_root, default_runs_root


@dataclass(frozen=True)
class RunPaths:
    """`budget_comm` 运行目录下的固定产物路径集合。"""

    root: Path
    manifest: Path
    sample_views: Path
    stage_a_turns: Path
    candidate_packets: Path
    auction_decisions: Path
    belief_updates: Path
    final_predictions: Path
    metrics: Path
    budget_diagnostics: Path
    progress: Path
    run_validation: Path
    report_markdown: Path
    paper_summary: Path


@dataclass(frozen=True)
class SampleResult:
    """单题运行产生的全部中间产物与最终结果。"""

    sample_views: list[dict[str, Any]]
    stage_a_turns: list[dict[str, Any]]
    candidate_packets: list[dict[str, Any]]
    auction_decisions: list[dict[str, Any]]
    belief_updates: list[dict[str, Any]]
    prediction_rows: list[dict[str, Any]]


def run_experiment(
    experiment: BudgetCommExperimentConfig,
    phase_name: str,
    backbone,
    run_root: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> Path:
    """执行一个 `budget_comm` phase，并写出完整运行目录。

    这是整条实验线的总调度入口，负责准备共享依赖、校准预算、并发跑样本、
    落盘中间日志、汇总最终指标，并生成报告与校验结果。
    """
    from budget_comm.reporting import render_report
    from budget_comm.validation import validate_run

    load_dotenv(".env.local", override=False)
    run_root = run_root or default_runs_root("budget_comm")
    cache_root = cache_root or default_cache_root()
    benchmarks = load_benchmarks(experiment)
    protocol = load_protocol_config(experiment.protocol)
    auction_policy = load_auction_policy_config(experiment.auction_policy)
    context_view_config = load_context_view_config(experiment.context_view)
    provider = OpenAICompatibleProvider(backbone)
    cache_router = RequestCacheRouter(cache_root)
    limiter = SlidingWindowRateLimiter(
        requests_per_minute=experiment.requests_per_minute_limit,
        tokens_per_minute=experiment.tokens_per_minute_limit,
    )

    # 先固定每个 benchmark 的 split，再一次性加载本轮样本，
    # 避免不同方法在运行途中出现数据选择漂移。
    benchmark_to_split = {
        benchmark.slug: _resolve_split_name(experiment, phase_name, benchmark.slug)
        for benchmark in benchmarks
    }
    benchmark_to_samples = {
        benchmark.slug: select_samples(benchmark, benchmark_to_split[benchmark.slug])
        for benchmark in benchmarks
    }
    # 预算不是拍脑袋给定，而是先用 calibration 样本跑 all_to_all_full，
    # 再按配置比例冻结成每个数据集的 round budget。
    calibration = _calibrate_budgets(
        experiment=experiment,
        benchmarks=benchmarks,
        benchmark_to_samples=benchmark_to_samples,
        benchmark_to_split=benchmark_to_split,
        protocol=protocol,
        auction_policy=auction_policy,
        context_view_config=context_view_config,
        backbone=backbone,
        provider=provider,
        cache_router=cache_router,
        limiter=limiter,
    )

    run_id = build_run_id(backbone.name)
    run_paths = _prepare_run_paths(run_root, experiment.name, phase_name, run_id)
    total_calls, total_predictions = _estimate_work(experiment, phase_name, benchmarks, protocol)
    progress = RunProgressTracker(run_paths.progress, total_calls, total_predictions)

    # manifest 记录“本次真正落地使用了什么配置与预算”，
    # 是后续分析、复现与审计的第一入口。
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment.name,
        "description": experiment.description,
        "phase": phase_name,
        "phase_metadata": phase_metadata(experiment, phase_name),
        "prompt_version": experiment.prompt_version,
        "artifact_version": ARTIFACT_VERSION,
        "global_seed": experiment.global_seed,
        "max_concurrent_requests": experiment.max_concurrent_requests,
        "requests_per_minute_limit": experiment.requests_per_minute_limit,
        "tokens_per_minute_limit": experiment.tokens_per_minute_limit,
        "family_name": "budget_comm",
        "experiment_name": experiment.name,
        "phase_name": phase_name,
        "primary_model_ref": experiment.primary_model_ref,
        "resolved_model": asdict(backbone),
        "protocol": asdict(protocol),
        "auction_policy": asdict(auction_policy),
        "context_view": asdict(context_view_config),
        "calibration": calibration,
        "benchmarks": [asdict(benchmark) for benchmark in benchmarks],
        "method_order": METHOD_ORDER,
        "packet_mode_order": PACKET_MODE_ORDER,
        "total_planned_calls": total_calls,
        "total_planned_predictions": total_predictions,
    }
    run_paths.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    all_sample_views: list[dict[str, Any]] = []
    all_stage_a_turns: list[dict[str, Any]] = []
    all_candidate_packets: list[dict[str, Any]] = []
    all_auction_decisions: list[dict[str, Any]] = []
    all_belief_updates: list[dict[str, Any]] = []
    all_prediction_rows: list[dict[str, Any]] = []

    with (
        run_paths.sample_views.open("w", encoding="utf-8") as sample_view_handle,
        run_paths.stage_a_turns.open("w", encoding="utf-8") as stage_a_handle,
        run_paths.candidate_packets.open("w", encoding="utf-8") as candidate_handle,
        run_paths.auction_decisions.open("w", encoding="utf-8") as auction_handle,
        run_paths.belief_updates.open("w", encoding="utf-8") as belief_handle,
        run_paths.final_predictions.open("w", encoding="utf-8") as prediction_handle,
    ):
        sample_view_writer = BufferedJsonlWriter(sample_view_handle)
        stage_a_writer = BufferedJsonlWriter(stage_a_handle)
        candidate_writer = BufferedJsonlWriter(candidate_handle)
        auction_writer = BufferedJsonlWriter(auction_handle)
        belief_writer = BufferedJsonlWriter(belief_handle)
        prediction_writer = BufferedJsonlWriter(prediction_handle)
        for benchmark in benchmarks:
            cache = cache_router.for_request_target(
                provider=backbone.provider,
                request_model=backbone.model_id,
                dataset=benchmark.slug,
            )
            split_name = benchmark_to_split[benchmark.slug]
            round_budget_tokens = int(calibration["datasets"][benchmark.slug]["round_budget_tokens"])
            sample_results = _run_sample_batch(
                run_id=run_id,
                benchmark_slug=benchmark.slug,
                split_name=split_name,
                samples=benchmark_to_samples[benchmark.slug],
                protocol=protocol,
                auction_policy=auction_policy,
                context_view_config=context_view_config,
                round_budget_tokens=round_budget_tokens,
                experiment=experiment,
                backbone=backbone,
                provider=provider,
                cache=cache,
                limiter=limiter,
            )
            _write_sample_results(
                sample_results=sample_results,
                dataset_slug=benchmark.slug,
                progress=progress,
                sample_view_handle=sample_view_writer,
                stage_a_handle=stage_a_writer,
                candidate_handle=candidate_writer,
                auction_handle=auction_writer,
                belief_handle=belief_writer,
                prediction_handle=prediction_writer,
                all_sample_views=all_sample_views,
                all_stage_a_turns=all_stage_a_turns,
                all_candidate_packets=all_candidate_packets,
                all_auction_decisions=all_auction_decisions,
                all_belief_updates=all_belief_updates,
                all_prediction_rows=all_prediction_rows,
            )

    metrics_payload = _build_metrics(all_prediction_rows)
    diagnostics_payload = _build_budget_diagnostics(metrics_payload, calibration, context_view_config.track_name)
    run_paths.metrics.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    run_paths.budget_diagnostics.write_text(json.dumps(diagnostics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _export_paper_summary(run_paths.paper_summary, metrics_payload["summary"])
    render_report(run_paths.root)
    finalize_run_outputs(
        run_paths.root,
        validator=validate_run,
        validation_path=run_paths.run_validation,
    )
    progress.mark_completed()
    provider.close()
    cache_router.close()
    return run_paths.root


def _calibrate_budgets(
    *,
    experiment: BudgetCommExperimentConfig,
    benchmarks,
    benchmark_to_samples: dict[str, list[DatasetSample]],
    benchmark_to_split: dict[str, str],
    protocol: BudgetProtocolConfig,
    auction_policy: AuctionPolicyConfig,
    context_view_config: ContextViewConfig,
    backbone,
    provider: OpenAICompatibleProvider,
    cache_router: RequestCacheRouter,
    limiter: SlidingWindowRateLimiter,
) -> dict[str, Any]:
    """先做少量 `all_to_all_full` 校准，再冻结每个数据集的预算。

    当前实现使用校准样本上的通信 token 中位数，再乘以 `calibration_fraction`
    作为该数据集后续所有方法共享的 `round_budget_tokens`。
    """
    datasets_payload: dict[str, dict[str, Any]] = {}
    for benchmark in benchmarks:
        cache = cache_router.for_request_target(
            provider=backbone.provider,
            request_model=backbone.model_id,
            dataset=benchmark.slug,
        )
        calibration_samples = benchmark_to_samples[benchmark.slug][: experiment.calibration_sample_size]
        communication_tokens: list[int] = []
        total_tokens: list[float] = []
        for sample in calibration_samples:
            prepared = _prepare_shared_context(
                run_id="calibration",
                benchmark_slug=benchmark.slug,
                split_name=benchmark_to_split[benchmark.slug],
                sample=sample,
                protocol=protocol,
                auction_policy=auction_policy,
                context_view_config=context_view_config,
                experiment=experiment,
                backbone=backbone,
                provider=provider,
                cache=cache,
                limiter=limiter,
            )
            all_to_all = build_all_to_all_full_decision(prepared["shared_candidates"])
            _, prediction_row = _run_stage_b_method(
                shared_context=prepared,
                decision=all_to_all,
                method_name="all_to_all_full",
                round_budget_tokens=None,
            )
            communication_tokens.append(int(prediction_row["communication_tokens_per_question"]))
            total_tokens.append(float(prediction_row["total_tokens_per_question"]))
        p50_tokens = _median_floor(communication_tokens)
        round_budget_tokens = int(math.floor(auction_policy.calibration_fraction * p50_tokens))
        datasets_payload[benchmark.slug] = {
            "split_name": benchmark_to_split[benchmark.slug],
            "track_name": context_view_config.track_name,
            "sample_count": len(calibration_samples),
            "communication_tokens": communication_tokens,
            "total_tokens": [round(value, 6) for value in total_tokens],
            "p50_all_to_all_full_communication_tokens": p50_tokens,
            "round_budget_tokens": round_budget_tokens,
            "episode_budget_tokens": round_budget_tokens,
            "calibration_fraction": auction_policy.calibration_fraction,
        }
    return {"datasets": datasets_payload}


def _run_sample_batch(
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    samples: list[DatasetSample],
    protocol: BudgetProtocolConfig,
    auction_policy: AuctionPolicyConfig,
    context_view_config: ContextViewConfig,
    round_budget_tokens: int,
    experiment: BudgetCommExperimentConfig,
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
) -> list[SampleResult]:
    """并发执行同一数据集下的一整批样本，并保持原始样本顺序。"""
    worker = partial(
        _run_sample,
        run_id=run_id,
        benchmark_slug=benchmark_slug,
        split_name=split_name,
        protocol=protocol,
        auction_policy=auction_policy,
        context_view_config=context_view_config,
        round_budget_tokens=round_budget_tokens,
        experiment=experiment,
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
    )
    max_workers = max(1, min(experiment.max_concurrent_requests, len(samples) or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(worker, sample=sample): index
            for index, sample in enumerate(samples)
        }
        completed: list[tuple[int, SampleResult]] = []
        for future in as_completed(future_to_index):
            completed.append((future_to_index[future], future.result()))
    completed.sort(key=lambda item: item[0])
    return [result for _, result in completed]


def _write_sample_results(
    *,
    sample_results: list[SampleResult],
    dataset_slug: str,
    progress: RunProgressTracker,
    sample_view_handle,
    stage_a_handle,
    candidate_handle,
    auction_handle,
    belief_handle,
    prediction_handle,
    all_sample_views: list[dict[str, Any]],
    all_stage_a_turns: list[dict[str, Any]],
    all_candidate_packets: list[dict[str, Any]],
    all_auction_decisions: list[dict[str, Any]],
    all_belief_updates: list[dict[str, Any]],
    all_prediction_rows: list[dict[str, Any]],
) -> None:
    """把单题结果稳定写盘，并同步更新内存聚合与进度快照。"""
    for result in sample_results:
        for row in result.sample_views:
            sample_view_handle.write_row(row)
        for row in result.stage_a_turns:
            stage_a_handle.write_row(row)
            progress.record_call(row, method_key="stage_name")
        for row in result.candidate_packets:
            candidate_handle.write_row(row)
        for row in result.auction_decisions:
            auction_handle.write_row(row)
        for row in result.belief_updates:
            belief_handle.write_row(row)
            progress.record_call(row, method_key="method_name")
        for row in result.prediction_rows:
            prediction_handle.write_row(row)
            progress.record_predictions(1, dataset_slug, str(row["method_name"]))
        all_sample_views.extend(result.sample_views)
        all_stage_a_turns.extend(result.stage_a_turns)
        all_candidate_packets.extend(result.candidate_packets)
        all_auction_decisions.extend(result.auction_decisions)
        all_belief_updates.extend(result.belief_updates)
        all_prediction_rows.extend(result.prediction_rows)


def _run_sample(
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    sample: DatasetSample,
    protocol: BudgetProtocolConfig,
    auction_policy: AuctionPolicyConfig,
    context_view_config: ContextViewConfig,
    round_budget_tokens: int,
    experiment: BudgetCommExperimentConfig,
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
) -> SampleResult:
    """执行单题上的全部预算通信方法。

    执行顺序固定为：
    1. 共享 Stage A；
    2. 无通信基线 `mv_3`；
    3. `all_to_all_full`；
    4. `budget_random`；
    5. `budget_confidence`；
    6. `dala_lite`。
    """
    shared_context = _prepare_shared_context(
        run_id=run_id,
        benchmark_slug=benchmark_slug,
        split_name=split_name,
        sample=sample,
        protocol=protocol,
        auction_policy=auction_policy,
        context_view_config=context_view_config,
        experiment=experiment,
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
    )
    candidate_rows: list[dict[str, Any]] = []
    auction_rows: list[dict[str, Any]] = []
    belief_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    # `mv_3` 直接复用 Stage A 投票结果，作为无通信基线。
    prediction_rows.append(_build_mv3_prediction(shared_context))

    all_to_all = build_all_to_all_full_decision(shared_context["shared_candidates"])
    candidate_rows.extend(_enrich_candidate_rows(shared_context, all_to_all["candidate_rows"]))
    auction_rows.append(_build_auction_row(shared_context, all_to_all, round_budget_tokens=None))
    method_belief_rows, prediction_row = _run_stage_b_method(
        shared_context=shared_context,
        decision=all_to_all,
        method_name="all_to_all_full",
        round_budget_tokens=None,
    )
    belief_rows.extend(method_belief_rows)
    prediction_rows.append(prediction_row)

    budget_random = build_budget_random_decision(
        shared_candidates=shared_context["shared_candidates"],
        round_budget_tokens=round_budget_tokens,
        seed=experiment.global_seed + _stable_sample_seed(sample.sample_id),
    )
    candidate_rows.extend(_enrich_candidate_rows(shared_context, budget_random["candidate_rows"]))
    auction_rows.append(_build_auction_row(shared_context, budget_random, round_budget_tokens=round_budget_tokens))
    method_belief_rows, prediction_row = _run_stage_b_method(
        shared_context=shared_context,
        decision=budget_random,
        method_name="budget_random",
        round_budget_tokens=round_budget_tokens,
    )
    belief_rows.extend(method_belief_rows)
    prediction_rows.append(prediction_row)

    budget_confidence = build_budget_confidence_decision(
        shared_candidates=shared_context["shared_candidates"],
        round_budget_tokens=round_budget_tokens,
    )
    candidate_rows.extend(_enrich_candidate_rows(shared_context, budget_confidence["candidate_rows"]))
    auction_rows.append(_build_auction_row(shared_context, budget_confidence, round_budget_tokens=round_budget_tokens))
    method_belief_rows, prediction_row = _run_stage_b_method(
        shared_context=shared_context,
        decision=budget_confidence,
        method_name="budget_confidence",
        round_budget_tokens=round_budget_tokens,
    )
    belief_rows.extend(method_belief_rows)
    prediction_rows.append(prediction_row)

    dala_lite = build_dala_lite_decision(
        shared_candidates=shared_context["shared_candidates"],
        round_budget_tokens=round_budget_tokens,
        positive_density_threshold=auction_policy.positive_density_threshold,
    )
    candidate_rows.extend(_enrich_candidate_rows(shared_context, dala_lite["candidate_rows"]))
    auction_rows.append(_build_auction_row(shared_context, dala_lite, round_budget_tokens=round_budget_tokens))
    method_belief_rows, prediction_row = _run_stage_b_method(
        shared_context=shared_context,
        decision=dala_lite,
        method_name="dala_lite",
        round_budget_tokens=round_budget_tokens,
    )
    belief_rows.extend(method_belief_rows)
    prediction_rows.append(prediction_row)

    return SampleResult(
        sample_views=shared_context["sample_view_rows"],
        stage_a_turns=shared_context["stage_a_turns"],
        candidate_packets=candidate_rows,
        auction_decisions=auction_rows,
        belief_updates=belief_rows,
        prediction_rows=prediction_rows,
    )


def _prepare_shared_context(
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    sample: DatasetSample,
    protocol: BudgetProtocolConfig,
    auction_policy: AuctionPolicyConfig,
    context_view_config: ContextViewConfig,
    experiment: BudgetCommExperimentConfig,
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
) -> dict[str, Any]:
    """准备单题共享上下文。

    这里会完成三件事：
    1. 按 track 生成 agent 视图；
    2. 跑共享 Stage A solver；
    3. 从 Stage A 输出中提炼可供所有方法复用的候选特征。
    """
    question_preview = _question_preview(sample.question)
    context_views = build_context_views(sample, context_view_config, agent_count=protocol.agent_count)
    sample_view_rows = [
        serialize_view_row(
            run_id=run_id,
            split_name=split_name,
            question_preview=question_preview,
            view=view,
        )
        for view in context_views
    ]
    stage_a_turns: list[dict[str, Any]] = []
    for view in context_views:
        # Stage A 是所有方法共享的前缀，因此它的输出必须完整落盘，
        # 后续所有预算方法都从这里出发，保证对照公平。
        messages = build_solver_messages(sample, view, prompt_version=experiment.prompt_version)
        stage_a_turn = _execute_turn(
            run_id=run_id,
            dataset=benchmark_slug,
            split_name=split_name,
            sample=sample,
            question_preview=question_preview,
            stage_name="stage_a",
            method_name="shared_stage_a",
            round_index=0,
            agent_id=view.agent_id,
            role="solver",
            visible_peer_count=0,
            track_name=context_view_config.track_name,
            view_kind=view.view_kind,
            messages=messages,
            backbone=backbone,
            provider=provider,
            cache=cache,
            limiter=limiter,
            temperature=protocol.initial_temperature,
            top_p=protocol.top_p,
            max_output_tokens=protocol.max_output_tokens,
            seed=experiment.global_seed + view.agent_id,
            output_mode=OUTPUT_MODE_BUDGET_SOLVER,
        )
        stage_a_turn["context_view_hash"] = view.view_context_hash
        stage_a_turn["includes_full_context"] = view.includes_full_context
        stage_a_turn["shard_titles"] = view.shard_titles
        stage_a_turns.append(stage_a_turn)
    stage_a_trace_hash = _trace_hash(stage_a_turns)
    for row in stage_a_turns:
        row["stage_a_trace_hash"] = stage_a_trace_hash

    shared_candidates = build_shared_candidate_features(stage_a_turns, auction_policy)
    stage_a_answers = [row["normalized_answer"] for row in stage_a_turns]
    stage_a_vote, stage_a_vote_counts = aggregate_majority(stage_a_answers)
    stage_a_score = score_prediction(benchmark_slug, stage_a_vote, sample.reference_answer)
    return {
        "run_id": run_id,
        "dataset": benchmark_slug,
        "split": split_name,
        "sample": sample,
        "question_preview": question_preview,
        "track_name": context_view_config.track_name,
        "context_views": context_views,
        "sample_view_rows": sample_view_rows,
        "protocol": protocol,
        "auction_policy": auction_policy,
        "experiment": experiment,
        "backbone": backbone,
        "provider": provider,
        "cache": cache,
        "limiter": limiter,
        "stage_a_turns": stage_a_turns,
        "stage_a_trace_hash": stage_a_trace_hash,
        "stage_a_vote": stage_a_vote,
        "stage_a_vote_counts": stage_a_vote_counts,
        "stage_a_score": stage_a_score,
        "shared_candidates": shared_candidates,
    }


def _run_stage_b_method(
    *,
    shared_context: dict[str, Any],
    decision: dict[str, Any],
    method_name: str,
    round_budget_tokens: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """执行某个方法的一轮 Stage B，并返回 belief rows 与题级预测。

    每个 agent 会看到“预算筛选后、且不包含自己”的 peer packets，
    再基于这些消息决定是否修正答案与置信度。
    """
    stage_a_turns = shared_context["stage_a_turns"]
    context_views = shared_context["context_views"]
    belief_rows: list[dict[str, Any]] = []
    post_debate_candidates: list[dict[str, Any]] = []
    selected_rows = [row for row in decision["candidate_rows"] if row.get("is_winner")]
    for view, stage_a_row in zip(context_views, stage_a_turns, strict=False):
        visible_peer_packets = [
            {
                "agent": f"agent_{row['agent_id']}",
                "packet_mode": row["selected_mode"],
                "packet_text": row["selected_packet_text"],
            }
            for row in selected_rows
            if int(row["agent_id"]) != int(view.agent_id)
        ]
        messages = build_belief_update_messages(
            shared_context["sample"],
            view,
            previous_answer=str(stage_a_row.get("normalized_answer") or ""),
            previous_reasoning_trace=str(stage_a_row.get("reasoning_trace") or ""),
            previous_confidence_raw=stage_a_row.get("confidence_raw_display"),
            selected_peer_packets=visible_peer_packets,
            method_name=method_name,
            round_budget_tokens=round_budget_tokens,
            prompt_version=shared_context["experiment"].prompt_version,
        )
        belief_row = _execute_turn(
            run_id=shared_context["run_id"],
            dataset=shared_context["dataset"],
            split_name=shared_context["split"],
            sample=shared_context["sample"],
            question_preview=shared_context["question_preview"],
            stage_name="stage_b",
            method_name=method_name,
            round_index=1,
            agent_id=view.agent_id,
            role="belief_update",
            visible_peer_count=len(visible_peer_packets),
            track_name=shared_context["track_name"],
            view_kind=view.view_kind,
            messages=messages,
            backbone=shared_context["backbone"],
            provider=shared_context["provider"],
            cache=shared_context["cache"],
            limiter=shared_context["limiter"],
            temperature=shared_context["protocol"].debate_temperature,
            top_p=shared_context["protocol"].top_p,
            max_output_tokens=shared_context["protocol"].max_output_tokens,
            seed=shared_context["experiment"].global_seed + view.agent_id + 100,
            output_mode=OUTPUT_MODE_BUDGET_BELIEF_UPDATE,
        )
        belief_row["selected_peer_agent_ids"] = sorted(int(row["agent_id"]) for row in selected_rows if int(row["agent_id"]) != int(view.agent_id))
        belief_row["selected_peer_packet_modes"] = {
            str(row["agent_id"]): row["selected_mode"]
            for row in selected_rows
            if int(row["agent_id"]) != int(view.agent_id)
        }
        belief_rows.append(belief_row)
        post_debate_candidates.append(apply_belief_update(stage_a_row=stage_a_row, belief_row=belief_row))

    stage_b_trace_hash = _trace_hash(belief_rows)
    for row in belief_rows:
        row["stage_b_trace_hash"] = stage_b_trace_hash

    # Stage B 结束后再按题级多数票聚合，得到该方法在本题上的最终表现。
    post_answers = [normalize_prediction(shared_context["dataset"], str(candidate["final_answer"])) for candidate in post_debate_candidates]
    post_vote, post_vote_counts = aggregate_majority(post_answers)
    post_score = score_prediction(shared_context["dataset"], post_vote, shared_context["sample"].reference_answer)
    stage_a_score = float(shared_context["stage_a_score"])
    full_count = sum(1 for row in decision["candidate_rows"] if row["selected_mode"] == "full")
    summary_count = sum(1 for row in decision["candidate_rows"] if row["selected_mode"] == "summary")
    keywords_count = sum(1 for row in decision["candidate_rows"] if row["selected_mode"] == "keywords")
    silence_count = sum(1 for row in decision["candidate_rows"] if row["selected_mode"] == "silence")
    prediction_row = {
        "run_id": shared_context["run_id"],
        "dataset": shared_context["dataset"],
        "split": shared_context["split"],
        "sample_id": shared_context["sample"].sample_id,
        "question_preview": shared_context["question_preview"],
        "track_name": shared_context["track_name"],
        "method_name": method_name,
        "method_kind": "budget_method",
        "display_name": method_name,
        "model_name": shared_context["backbone"].name,
        "prediction": post_vote,
        "gold": shared_context["sample"].reference_answer,
        "score": post_score,
        "stage_a_prediction": shared_context["stage_a_vote"],
        "stage_a_score": stage_a_score,
        "stage_a_vote_counts": shared_context["stage_a_vote_counts"],
        "post_vote_counts": post_vote_counts,
        "prompt_tokens_per_question": round(_sum_metric(stage_a_turns + belief_rows, "prompt_tokens"), 6),
        "completion_tokens_per_question": round(_sum_metric(stage_a_turns + belief_rows, "completion_tokens"), 6),
        "total_tokens_per_question": round(_sum_metric(stage_a_turns + belief_rows, "total_tokens"), 6),
        "communication_tokens_per_question": float(decision["total_cost"]),
        "latency_ms_per_question": round(_sum_metric(stage_a_turns + belief_rows, "latency_ms"), 6),
        "calls_per_question": len(stage_a_turns) + len(belief_rows),
        "round_budget_tokens": round_budget_tokens,
        "episode_budget_tokens": round_budget_tokens,
        "budget_utilization": decision["budget_utilization"],
        "winner_set_size": len(decision["winner_agent_ids"]),
        "winner_agent_ids": decision["winner_agent_ids"],
        "winner_modes": decision["winner_modes"],
        "full_count": full_count,
        "summary_count": summary_count,
        "keywords_count": keywords_count,
        "silence_count": silence_count,
        "full_ratio": round(full_count / shared_context["protocol"].agent_count, 6),
        "summary_ratio": round(summary_count / shared_context["protocol"].agent_count, 6),
        "keywords_ratio": round(keywords_count / shared_context["protocol"].agent_count, 6),
        "silence_ratio": round(silence_count / shared_context["protocol"].agent_count, 6),
        "corrected_by_method": stage_a_score < 1.0 and post_score == 1.0,
        "harmed_by_method": stage_a_score == 1.0 and post_score < 1.0,
        "selection_rule": decision["selection_rule"],
        "stage_a_hash": shared_context["stage_a_trace_hash"],
        "stage_a_trace_hash": shared_context["stage_a_trace_hash"],
        "stage_b_trace_hash_used": stage_b_trace_hash,
        "route_cost": float(decision["total_cost"]),
        "route_value": round(post_score - stage_a_score, 6),
        "drift_flag": post_score < stage_a_score,
        "note": None,
    }
    return belief_rows, prediction_row


def _build_mv3_prediction(shared_context: dict[str, Any]) -> dict[str, Any]:
    """构造无通信基线 `mv_3`。"""
    agent_count = int(shared_context["protocol"].agent_count)
    return {
        "run_id": shared_context["run_id"],
        "dataset": shared_context["dataset"],
        "split": shared_context["split"],
        "sample_id": shared_context["sample"].sample_id,
        "question_preview": shared_context["question_preview"],
        "track_name": shared_context["track_name"],
        "method_name": "mv_3",
        "method_kind": "baseline",
        "display_name": "mv_3",
        "model_name": shared_context["backbone"].name,
        "prediction": shared_context["stage_a_vote"],
        "gold": shared_context["sample"].reference_answer,
        "score": float(shared_context["stage_a_score"]),
        "stage_a_prediction": shared_context["stage_a_vote"],
        "stage_a_score": float(shared_context["stage_a_score"]),
        "stage_a_vote_counts": shared_context["stage_a_vote_counts"],
        "post_vote_counts": shared_context["stage_a_vote_counts"],
        "prompt_tokens_per_question": round(_sum_metric(shared_context["stage_a_turns"], "prompt_tokens"), 6),
        "completion_tokens_per_question": round(_sum_metric(shared_context["stage_a_turns"], "completion_tokens"), 6),
        "total_tokens_per_question": round(_sum_metric(shared_context["stage_a_turns"], "total_tokens"), 6),
        "communication_tokens_per_question": 0.0,
        "latency_ms_per_question": round(_sum_metric(shared_context["stage_a_turns"], "latency_ms"), 6),
        "calls_per_question": len(shared_context["stage_a_turns"]),
        "round_budget_tokens": None,
        "episode_budget_tokens": None,
        "budget_utilization": None,
        "winner_set_size": 0,
        "winner_agent_ids": [],
        "winner_modes": {},
        "full_count": 0,
        "summary_count": 0,
        "keywords_count": 0,
        "silence_count": agent_count,
        "full_ratio": 0.0,
        "summary_ratio": 0.0,
        "keywords_ratio": 0.0,
        "silence_ratio": 1.0,
        "corrected_by_method": False,
        "harmed_by_method": False,
        "selection_rule": "stage_a_majority_vote",
        "stage_a_hash": shared_context["stage_a_trace_hash"],
        "stage_a_trace_hash": shared_context["stage_a_trace_hash"],
        "stage_b_trace_hash_used": None,
        "route_cost": 0.0,
        "route_value": 0.0,
        "drift_flag": False,
        "note": "no_communication_baseline",
    }


def _enrich_candidate_rows(shared_context: dict[str, Any], candidate_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """为候选消息包行补齐样本级上下文字段。"""
    enriched: list[dict[str, Any]] = []
    for row in candidate_rows:
        enriched.append(
            {
                "run_id": shared_context["run_id"],
                "dataset": shared_context["dataset"],
                "split": shared_context["split"],
                "sample_id": shared_context["sample"].sample_id,
                "question_preview": shared_context["question_preview"],
                "track_name": shared_context["track_name"],
            }
            | row
        )
    return enriched


def _build_auction_row(shared_context: dict[str, Any], decision: dict[str, Any], *, round_budget_tokens: int | None) -> dict[str, Any]:
    """构建方法级拍卖 / 预算决策诊断行。"""
    candidate_score_lookup = {str(row["agent_id"]): row["selection_score"] for row in decision["candidate_rows"]}
    candidate_cost_lookup = {str(row["agent_id"]): row["candidate_cost"] for row in decision["candidate_rows"]}
    candidate_density_lookup = {str(row["agent_id"]): row["density_score"] for row in decision["candidate_rows"]}
    dala_tier_lookup = {str(row["agent_id"]): row["dala_assigned_mode"] for row in decision["candidate_rows"]}
    positive_density_agent_ids = [
        int(row["agent_id"])
        for row in decision["candidate_rows"]
        if float(row["density_score"]) > shared_context["auction_policy"].positive_density_threshold
    ]
    return {
        "run_id": shared_context["run_id"],
        "dataset": shared_context["dataset"],
        "split": shared_context["split"],
        "sample_id": shared_context["sample"].sample_id,
        "question_preview": shared_context["question_preview"],
        "track_name": shared_context["track_name"],
        "method_name": decision["method_name"],
        "selection_rule": decision["selection_rule"],
        "round_budget_tokens": round_budget_tokens,
        "episode_budget_tokens": round_budget_tokens,
        "winner_agent_ids": decision["winner_agent_ids"],
        "winner_modes": decision["winner_modes"],
        "total_score": decision["total_score"],
        "total_cost": decision["total_cost"],
        "budget_utilization": decision["budget_utilization"],
        "candidate_scores": candidate_score_lookup,
        "candidate_costs": candidate_cost_lookup,
        "candidate_density_scores": candidate_density_lookup,
        "candidate_dala_tiers": dala_tier_lookup,
        "positive_density_agent_ids": positive_density_agent_ids,
        "vcg_payments": decision["vcg_payments"],
    }


def _build_metrics(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """把题级结果聚合成方法级 summary。"""
    summary: list[dict[str, Any]] = []
    for dataset in sorted({str(row["dataset"]) for row in prediction_rows} | {"overall"}):
        rows_for_dataset = prediction_rows if dataset == "overall" else [row for row in prediction_rows if row["dataset"] == dataset]
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in rows_for_dataset:
            grouped.setdefault((row["model_name"], row["method_name"]), []).append(row)
        for (model_name, method_name), rows in sorted(grouped.items()):
            total_tokens_mean = _mean(float(row["total_tokens_per_question"]) for row in rows)
            summary.append(
                {
                    "dataset": dataset,
                    "track_name": rows[0]["track_name"],
                    "model_name": model_name,
                    "method_name": method_name,
                    "display_name": rows[0]["display_name"],
                    "method_kind": rows[0]["method_kind"],
                    "question_count": len(rows),
                    "accuracy_mean": _mean(float(row["score"]) for row in rows),
                    "prompt_tokens_mean": _mean(float(row["prompt_tokens_per_question"]) for row in rows),
                    "completion_tokens_mean": _mean(float(row["completion_tokens_per_question"]) for row in rows),
                    "total_tokens_mean": total_tokens_mean,
                    "communication_tokens_mean": _mean(float(row["communication_tokens_per_question"]) for row in rows),
                    "latency_ms_mean": _mean(float(row["latency_ms_per_question"]) for row in rows),
                    "calls_per_question_mean": _mean(float(row["calls_per_question"]) for row in rows),
                    "acc_per_1k_tokens": round(_mean(float(row["score"]) for row in rows) / total_tokens_mean * 1000, 6) if total_tokens_mean else 0.0,
                    "winner_set_size_mean": _mean(float(row["winner_set_size"]) for row in rows),
                    "budget_utilization_mean": _mean(float(row["budget_utilization"] or 0.0) for row in rows),
                    "full_ratio_mean": _mean(float(row["full_ratio"]) for row in rows),
                    "summary_ratio_mean": _mean(float(row["summary_ratio"]) for row in rows),
                    "keywords_ratio_mean": _mean(float(row["keywords_ratio"]) for row in rows),
                    "silence_ratio_mean": _mean(float(row["silence_ratio"]) for row in rows),
                    "corrected_count": sum(1 for row in rows if row["corrected_by_method"]),
                    "harmed_count": sum(1 for row in rows if row["harmed_by_method"]),
                }
            )
    return {"summary": summary}


def _build_budget_diagnostics(
    metrics_payload: dict[str, Any],
    calibration: dict[str, Any],
    track_name: str,
) -> dict[str, Any]:
    """构建预算诊断与 full DALA gate。"""
    overall_rows = [row for row in metrics_payload["summary"] if row["dataset"] == "overall"]
    return {
        "track_name": track_name,
        "calibration": calibration["datasets"],
        "full_dala_gate": evaluate_full_dala_gate(overall_rows),
    }


def _export_paper_summary(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    """导出面向论文整理的轻量 CSV 摘要。"""
    fieldnames = [
        "dataset",
        "track_name",
        "model_name",
        "method_name",
        "accuracy_mean",
        "communication_tokens_mean",
        "total_tokens_mean",
        "calls_per_question_mean",
        "acc_per_1k_tokens",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _prepare_run_paths(run_root: str | Path, experiment_name: str, phase_name: str, run_id: str) -> RunPaths:
    """创建运行目录，并返回其中所有固定产物路径。"""
    root = Path(run_root) / experiment_name / phase_name / run_id
    root.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        root=root,
        manifest=root / "manifest.json",
        sample_views=root / "sample_views.jsonl",
        stage_a_turns=root / "stage_a_turns.jsonl",
        candidate_packets=root / "candidate_packets.jsonl",
        auction_decisions=root / "auction_decisions.jsonl",
        belief_updates=root / "belief_updates.jsonl",
        final_predictions=root / "final_predictions.jsonl",
        metrics=root / "metrics.json",
        budget_diagnostics=root / "budget_diagnostics.json",
        progress=root / "progress.json",
        run_validation=root / "run_validation.json",
        report_markdown=root / "report.md",
        paper_summary=root / "paper_summary.csv",
    )


def _estimate_work(
    experiment: BudgetCommExperimentConfig,
    phase_name: str,
    benchmarks,
    protocol: BudgetProtocolConfig,
) -> tuple[int, int]:
    """估算总调用数与总预测数，用于进度展示。"""
    total_calls = 0
    total_predictions = 0
    for benchmark in benchmarks:
        split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
        sample_count = len(load_split_ids(benchmark.slug, split_name))
        total_calls += sample_count * protocol.agent_count * (1 + 4 * protocol.debate_rounds)
        total_predictions += sample_count * len(METHOD_ORDER)
    return total_calls, total_predictions


def _resolve_split_name(experiment: BudgetCommExperimentConfig, phase_name: str, benchmark_slug: str) -> str:
    """解析某个 benchmark 在当前 phase 下使用的冻结 split 名称。"""
    phase = phase_metadata(experiment, phase_name)
    if "split_overrides" in phase:
        return str(phase["split_overrides"][benchmark_slug])
    return str(phase["split_suffix"])


def _execute_turn(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    question_preview: str,
    stage_name: str,
    method_name: str,
    round_index: int,
    agent_id: int,
    role: str,
    visible_peer_count: int,
    track_name: str,
    view_kind: str,
    messages: list[dict[str, str]],
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    seed: int,
    output_mode: str,
) -> dict[str, Any]:
    """执行单次 Stage A 或 Stage B 调用，并整理成统一日志结构。

    这里会统一处理缓存、限流、provider 请求、结构化校验、轻量修复与字段展开，
    让上层阶段逻辑只关心“这一轮产生了什么记录”。
    """
    payload = build_payload(
        config=backbone,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        seed=seed,
    )
    prompt_hash = _prompt_hash(messages)
    cache_key = build_request_cache_key(
        provider=backbone.provider,
        request_model=backbone.model_id,
        payload=payload,
    )
    cached = cache.get(cache_key)
    if cached is None:
        # 只有真正发网路请求时才占用限流配额；缓存命中不计入。
        response_payload = execute_completion_request(
            provider,
            payload,
            limiter=limiter,
        )
        cache_hit = False
    else:
        response_payload = json.loads(cached.response_json)
        cache_hit = True

    request_error = response_payload.get("request_error")
    if request_error:
        validated_output = {}
        output_status = "request_fail"
        answer_for_normalization = ""
    else:
        try:
            validated_output = validate_or_recover_structured_output(
                str(response_payload.get("assistant_text") or ""),
                output_mode,  # type: ignore[arg-type]
                provider_reasoning_text=str(response_payload.get("provider_reasoning_text") or ""),
            )
            output_status = "ok"
            if output_mode == OUTPUT_MODE_BUDGET_BELIEF_UPDATE:
                answer_for_normalization = str(validated_output.get("new_answer") or "")
            else:
                answer_for_normalization = str(validated_output.get("final_answer") or "")
            if not cache_hit:
                cache_successful_response(
                    cache,
                    cache_key=cache_key,
                    payload=payload,
                    response_payload=response_payload,
                )
        except Exception:
            # `budget_comm` 对截断 JSON 做一次保守修复，
            # 尽量把“轻微格式问题”与“真正逻辑错误”区分开来。
            validated_output = {}
            output_status = "schema_fail"
            answer_for_normalization = ""

    usage = response_payload.get("usage_reported") or response_payload.get("usage_estimated") or {}
    row = {
        "run_id": run_id,
        "dataset": dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "question_preview": question_preview,
        "track_name": track_name,
        "view_kind": view_kind,
        "stage_name": stage_name,
        "method_name": method_name,
        "role": role,
        "round_index": round_index,
        "agent_id": agent_id,
        "visible_peer_count": visible_peer_count,
        "prompt_hash": prompt_hash,
        "prediction": normalize_prediction(dataset, answer_for_normalization) if answer_for_normalization else "",
        "normalized_answer": normalize_prediction(dataset, answer_for_normalization) if answer_for_normalization else "",
        "output_status": output_status,
        "prompt_tokens": float(usage.get("prompt_tokens") or 0.0),
        "completion_tokens": float(usage.get("completion_tokens") or 0.0),
        "total_tokens": float(usage.get("total_tokens") or 0.0),
        "latency_ms": float(response_payload.get("latency_ms") or 0.0),
        "cache_hit": cache_hit,
        "request_error": request_error,
        "payload": payload,
        "assistant_text": response_payload.get("assistant_text", ""),
        "provider_reasoning_text": response_payload.get("provider_reasoning_text", ""),
        "validated_output": validated_output,
    }
    if output_mode == OUTPUT_MODE_BUDGET_SOLVER:
        # Stage A 需要额外展开证据、关键词与置信度归一化字段，
        # 供后续 density 计算与消息压缩复用。
        confidence_raw = validated_output.get("confidence_raw") if validated_output else None
        confidence_value, confidence_valid, confidence_source = normalize_confidence(confidence_raw)
        row.update(
            {
                "reasoning_trace": validated_output.get("reasoning_trace") if validated_output else None,
                "claim_span": validated_output.get("claim_span") if validated_output else None,
                "key_evidence": validated_output.get("key_evidence") if validated_output else None,
                "keyword_clues": validated_output.get("keyword_clues") if validated_output else [],
                "confidence_raw": confidence_raw,
                "confidence_raw_display": confidence_display(confidence_raw),
                "confidence_value": confidence_value,
                "confidence_valid": confidence_valid,
                "confidence_source": confidence_source,
                "uncertain_point": validated_output.get("uncertain_point") if validated_output else None,
            }
        )
    else:
        row.update(
            {
                "changed_answer": validated_output.get("changed_answer") if validated_output else None,
                "new_answer": validated_output.get("new_answer") if validated_output else None,
                "confidence_delta": validated_output.get("confidence_delta") if validated_output else None,
                "reason_for_change": validated_output.get("reason_for_change") if validated_output else None,
                "remaining_disagreement": validated_output.get("remaining_disagreement") if validated_output else None,
            }
        )
    return row


def _trace_hash(rows: list[dict[str, Any]]) -> str:
    """为一组阶段日志生成稳定 trace 哈希。"""
    payload = [
        {
            "stage_name": row["stage_name"],
            "method_name": row["method_name"],
            "round_index": row["round_index"],
            "agent_id": row["agent_id"],
            "prompt_hash": row["prompt_hash"],
            "normalized_answer": row["normalized_answer"],
            "output_status": row["output_status"],
        }
        for row in rows
    ]
    return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _prompt_hash(messages: list[dict[str, str]]) -> str:
    """为提示词内容生成稳定哈希。"""
    return sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _question_preview(question: str, max_chars: int = 120) -> str:
    """生成用于日志与报告展示的短问题预览。"""
    cleaned = " ".join(question.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3] + "..."


def _stable_sample_seed(sample_id: str) -> int:
    """从样本 ID 派生稳定种子，避免全局随机漂移。"""
    return sum(ord(char) for char in sample_id)


def _repair_budget_output(raw_text: str, output_mode: str) -> dict[str, Any] | None:
    """对被截断的 budget JSON 做轻量修复。"""
    if output_mode == OUTPUT_MODE_BUDGET_SOLVER:
        return _repair_budget_solver_output(raw_text)
    if output_mode == OUTPUT_MODE_BUDGET_BELIEF_UPDATE:
        return _repair_budget_belief_update_output(raw_text)
    return None


def _repair_budget_solver_output(raw_text: str) -> dict[str, Any] | None:
    """尝试从破损的 Stage A 输出中恢复最小可用字段。"""
    final_answer = _extract_json_string_field(raw_text, "final_answer")
    if final_answer is None and re.search(r'"final_answer"\s*:\s*""', raw_text):
        final_answer = "unknown"
    if not final_answer:
        return None
    keyword_clues = _extract_json_string_list(raw_text, "keyword_clues")
    if not keyword_clues:
        keyword_clues = [final_answer]
    confidence_raw = _extract_json_number_field(raw_text, "confidence_raw")
    return {
        "final_answer": final_answer,
        "reasoning_trace": _extract_json_string_field(raw_text, "reasoning_trace"),
        "claim_span": _extract_json_string_field(raw_text, "claim_span"),
        "key_evidence": _extract_json_string_field(raw_text, "key_evidence"),
        "keyword_clues": keyword_clues,
        "confidence_raw": confidence_raw if confidence_raw is not None else 0.5,
        "uncertain_point": _extract_json_string_field(raw_text, "uncertain_point"),
    }


def _repair_budget_belief_update_output(raw_text: str) -> dict[str, Any] | None:
    """尝试从破损的 belief update 输出中恢复最小可用字段。"""
    changed_answer = _extract_json_bool_field(raw_text, "changed_answer")
    if changed_answer is None:
        return None
    new_answer = _extract_json_string_field(raw_text, "new_answer")
    if changed_answer and not new_answer:
        return None
    return {
        "changed_answer": changed_answer,
        "new_answer": new_answer,
        "confidence_delta": _extract_json_number_field(raw_text, "confidence_delta"),
        "reason_for_change": _extract_json_string_field(raw_text, "reason_for_change"),
        "remaining_disagreement": _extract_json_string_field(raw_text, "remaining_disagreement"),
    }


def _extract_json_string_field(raw_text: str, field_name: str) -> str | None:
    """用正则从原始文本中提取 JSON 字符串字段。"""
    pattern = rf'"{re.escape(field_name)}"\s*:\s*"((?:\\.|[^"\\])*)"'
    match = re.search(pattern, raw_text, flags=re.DOTALL)
    if not match:
        return None
    return bytes(match.group(1), "utf-8").decode("unicode_escape").strip() or None


def _extract_json_string_list(raw_text: str, field_name: str) -> list[str]:
    """用正则从原始文本中提取 JSON 字符串列表字段。"""
    pattern = rf'"{re.escape(field_name)}"\s*:\s*\[(.*?)\]'
    match = re.search(pattern, raw_text, flags=re.DOTALL)
    if not match:
        return []
    list_body = match.group(1)
    items = re.findall(r'"((?:\\.|[^"\\])*)"', list_body)
    return [bytes(item, "utf-8").decode("unicode_escape").strip() for item in items if bytes(item, "utf-8").decode("unicode_escape").strip()]


def _extract_json_number_field(raw_text: str, field_name: str) -> float | None:
    """用正则从原始文本中提取 JSON 数值字段。"""
    pattern = rf'"{re.escape(field_name)}"\s*:\s*(-?\d+(?:\.\d+)?)'
    match = re.search(pattern, raw_text)
    if not match:
        return None
    return float(match.group(1))


def _extract_json_bool_field(raw_text: str, field_name: str) -> bool | None:
    """用正则从原始文本中提取 JSON 布尔字段。"""
    pattern = rf'"{re.escape(field_name)}"\s*:\s*(true|false)'
    match = re.search(pattern, raw_text, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).lower() == "true"


def _sum_metric(rows: list[dict[str, Any]], key: str) -> float:
    """对一组日志行中的某个数值字段求和。"""
    return sum(float(row.get(key) or 0.0) for row in rows)


def _mean(values) -> float:
    """安全计算均值。"""
    materialized = list(values)
    if not materialized:
        return 0.0
    return round(sum(materialized) / len(materialized), 6)


def _median_floor(values: list[int]) -> int:
    """计算整型列表的下取整中位数。"""
    if not values:
        return 0
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return int(ordered[midpoint])
    return int((ordered[midpoint - 1] + ordered[midpoint]) // 2)

