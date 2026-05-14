"""`budget_comm` 实验主运行链路。

本模块把 DALA-lite 风格预算通信实验落成完整的可执行流程：
1. 校准每个数据集的轮次预算；
2. 运行共享 Stage A，得到可压缩候选；
3. 分别执行多种预算决策方法；
4. 在 Stage B 中基于中标消息包做 belief update；
5. 汇总指标、诊断与论文摘要产物。
"""

from __future__ import annotations

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

from research_experiments.families.budget_comm.config import (
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
from research_experiments.families.budget_comm.dataset_views import build_context_views, serialize_view_row
from research_experiments.families.budget_comm.algorithms import (
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
from research_experiments.families.budget_comm.prompts import build_belief_update_messages, build_solver_messages
from research_experiments.core.execution.artifacts import BufferedJsonlWriter
from research_experiments.core.execution.cache import RequestCache, RequestCacheRouter, json_dump
from research_experiments.core.data.datasets import DatasetSample, load_split_ids, select_samples
from research_experiments.core.data.evaluation import aggregate_majority, normalize_prediction, score_prediction
from research_experiments.families.shared.common import build_question_preview, resolve_phase_split_name, safe_mean, stable_trace_hash, sum_metric
from research_experiments.core.execution.providers import OpenAICompatibleProvider
from research_experiments.core.execution.rate_limits import SlidingWindowRateLimiter
from research_experiments.core.execution.runner_common import (
    execute_cached_turn,
    prepare_run_root,
    run_indexed_batch,
)
from research_experiments.core.execution.runtime import RunProgressTracker, build_run_id, finalize_run_outputs
from research_experiments.core.controls.selective_signals import confidence_display, normalize_confidence
from research_experiments.core.structured_outputs import (
    ARTIFACT_VERSION,
    SCHEMA_ANSWER_WITH_PROXY_SIGNALS_BUDGET,
    SCHEMA_BELIEF_UPDATE_DELTA,
)
from research_experiments.workspace.layout import default_cache_root, default_runs_root

from research_experiments.families.budget_comm.run.io import _prepare_run_paths
from research_experiments.families.budget_comm.run.sample import (
    _build_budget_diagnostics,
    _build_metrics,
    _calibrate_budgets,
    _estimate_work,
    _export_paper_summary,
    _resolve_split_name,
    _run_sample_batch,
    _write_sample_results,
)

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
    from research_experiments.families.budget_comm.run.report import render_report
    from research_experiments.families.budget_comm.run.validate import validate_run

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
                dataset=benchmark.cache_namespace or benchmark.slug,
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
