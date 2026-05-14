"""单智能体实验主运行链路。

本模块负责把实验规格转换成真实的 API 调用计划，并在一次运行中完成：
phase 解析、样本展开、并发请求、缓存复用、解析兜底、指标聚合、报告导出
与运行后校验。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
import json
from statistics import mean, pstdev
from typing import Any

from dotenv import load_dotenv

from research_experiments.core.execution.artifacts import BufferedJsonlWriter
from research_experiments.core.execution.cache import RequestCache, RequestCacheRouter, build_request_cache_key, cache_successful_response
from research_experiments.core.config import (
    BenchmarkConfig,
    ResolvedModelConfig,
)
from research_experiments.core.data.datasets import (
    DatasetSample,
    generate_split_manifests as core_generate_split_manifests,
    load_split_ids,
    select_samples,
)
from research_experiments.core.data.evaluation import aggregate_majority, normalize_prediction, score_prediction
from research_experiments.families.shared.common import resolve_phase_split_name
from research_experiments.families.shared.method_catalog import MethodConfig, load_method_catalog
from research_experiments.core.execution.providers import OpenAICompatibleProvider, build_payload, execute_completion_request
from research_experiments.core.execution.rate_limits import SlidingWindowRateLimiter
from research_experiments.core.execution.runner_common import (
    execute_cached_turn,
    prepare_run_root,
    prompt_hash as build_prompt_hash,
    run_indexed_batch,
)
from research_experiments.core.execution.runtime import RunProgressTracker, build_run_id, finalize_run_outputs
from research_experiments.core.structured_outputs import (
    ARTIFACT_VERSION,
    SCHEMA_ANSWER_CORE,
    validate_or_recover_structured_output,
)
from research_experiments.workspace.layout import (
    default_cache_root,
    default_reports_root,
    default_runs_root,
)
from research_experiments.families.single_agent.config import (
    ExperimentConfig,
    phase_metadata,
    required_benchmark_tags,
    required_model_tags,
)
from research_experiments.families.single_agent.prompts import build_messages
from research_experiments.families.single_agent.run.report import export_paper_tables, render_report, summarize_run
from research_experiments.families.single_agent.run.validate import validate_run

from research_experiments.families.single_agent.run.io import _prepare_run_paths
from research_experiments.families.single_agent.run.sample import (
    _aggregate_metrics,
    _benchmark_is_allowed,
    _ensure_run_has_eligible_work,
    _estimate_run_work,
    _model_is_allowed,
    _phase_methods,
    _resolve_split_name,
    _run_method_batch,
    _write_leaderboard,
)

def run_experiment(
    experiment: ExperimentConfig,
    phase_name: str,
    models: list[ResolvedModelConfig],
    benchmarks: list[BenchmarkConfig],
    run_root: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> Path:
    """执行一个 `single_agent` phase，并产出完整运行目录。

    它会把“模型 × 数据集 × 方法 × rerun”的笛卡尔积展开成具体调用计划，
    然后负责并发执行、缓存复用、题级聚合、指标汇总与产物落盘。
    """
    load_dotenv(".env.local", override=False)
    run_root = run_root or default_runs_root("single_agent")
    cache_root = cache_root or default_cache_root()
    phase = phase_metadata(experiment, phase_name)
    method_catalog = load_method_catalog(experiment.method_catalog)
    methods = _phase_methods(experiment, phase_name, method_catalog)
    primary_model = models[0] if models else None
    run_id = build_run_id(primary_model.name if primary_model is not None else "")
    run_paths = _prepare_run_paths(run_root, experiment.name, phase_name, run_id)
    cache_router = RequestCacheRouter(cache_root)
    rate_limiter = SlidingWindowRateLimiter(
        requests_per_minute=experiment.requests_per_minute_limit,
        tokens_per_minute=experiment.tokens_per_minute_limit,
    )
    total_planned_calls, total_planned_predictions = _estimate_run_work(
        experiment=experiment,
        phase_name=phase_name,
        models=models,
        benchmarks=benchmarks,
        method_catalog=method_catalog,
    )
    _ensure_run_has_eligible_work(experiment, phase_name, models, benchmarks)
    progress = RunProgressTracker(
        progress_path=run_paths.progress,
        total_planned_calls=total_planned_calls,
        total_planned_predictions=total_planned_predictions,
    )

    # manifest 记录的是“这次运行最终使用了什么配置”，它是最重要的审计入口。
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "family_name": "single_agent",
        "experiment_name": experiment.name,
        "phase_name": phase_name,
        "primary_model_ref": experiment.primary_model_ref,
        "resolved_model": asdict(primary_model) if primary_model is not None else None,
        "experiment": experiment.name,
        "phase": phase_name,
        "description": experiment.description,
        "prompt_version": experiment.prompt_version,
        "artifact_version": ARTIFACT_VERSION,
        "reruns_per_method": experiment.reruns_per_method,
        "max_concurrent_requests": experiment.max_concurrent_requests,
        "requests_per_minute_limit": experiment.requests_per_minute_limit,
        "tokens_per_minute_limit": experiment.tokens_per_minute_limit,
        "models": [asdict(model) for model in models],
        "benchmarks": [asdict(benchmark) for benchmark in benchmarks],
        "phase_metadata": phase,
        "total_planned_calls": total_planned_calls,
        "total_planned_predictions": total_planned_predictions,
    }
    run_paths.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    all_predictions: list[dict[str, Any]] = []
    with (
        run_paths.raw_responses.open("w", encoding="utf-8") as raw_handle,
        run_paths.predictions.open("w", encoding="utf-8") as pred_handle,
    ):
        raw_writer = BufferedJsonlWriter(raw_handle)
        prediction_writer = BufferedJsonlWriter(pred_handle)
        for model in models:
            if not _model_is_allowed(experiment, phase_name, model):
                continue
            provider = OpenAICompatibleProvider(model)
            for benchmark in benchmarks:
                if not _benchmark_is_allowed(experiment, phase_name, model, benchmark.slug):
                    continue
                cache = cache_router.for_request_target(
                    provider=model.provider,
                    request_model=model.model_id,
                    dataset=benchmark.cache_namespace or benchmark.slug,
                )
                split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
                selected_samples = select_samples(benchmark, split_name)

                for method in methods:
                    # CoT 是单次确定性求解；其余方法通过多次调用形成同预算对比。
                    reruns = (
                        1 if method.family == "cot"
                        else int(phase.get("reruns_override", experiment.reruns_per_method))
                    )
                    for rerun_index in range(reruns):
                        batch_predictions = _run_method_batch(
                            run_id=run_id,
                            phase_name=phase_name,
                            experiment=experiment,
                            benchmark=benchmark,
                            model=model,
                            method=method,
                            samples=selected_samples,
                            provider=provider,
                            cache=cache,
                            rate_limiter=rate_limiter,
                            progress=progress,
                            rerun_index=rerun_index,
                            raw_writer=raw_writer,
                        )
                        for record in batch_predictions:
                            prediction_writer.write_row(record)
                        all_predictions.extend(batch_predictions)
            provider.close()

    metrics_payload = _aggregate_metrics(all_predictions)
    run_paths.metrics.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_leaderboard(Path(default_reports_root("single_agent")) / "leaderboard.csv", all_predictions)
    run_paths.run_summary.write_text(
        json.dumps(summarize_run(run_paths.root), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    render_report(run_paths.root)
    export_paper_tables(run_paths.root, run_paths.paper_tables)
    finalize_run_outputs(
        run_paths.root,
        validator=validate_run,
        validation_path=run_paths.run_validation,
    )
    progress.mark_completed()
    cache_router.close()
    return run_paths.root
