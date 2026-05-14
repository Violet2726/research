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
from research_experiments.families.single_agent.run.io import RunPaths




@dataclass(frozen=True)
class CallSpec:
    """单次 API 调用的完整执行规格。

    该结构把一次调用需要的实验上下文全部固定下来，便于并发执行、
    计算缓存键，以及在原始日志中保留足够的追溯信息。
    """

    run_id: str
    dataset: str
    split_name: str
    sample_id: str
    sample_order: int
    method_name: str
    method_family: str
    rerun_index: int
    replicate_id: int
    agent_id: int | None
    model_name: str
    model_id: str
    provider_name: str
    base_url: str
    prompt_hash: str
    payload: dict[str, Any]
    cache_key: str
    backbone: ResolvedModelConfig | None = None
    messages: list[dict[str, str]] | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = None
    seed: int | None = None


def generate_split_manifests(benchmark_configs: list[BenchmarkConfig], output_dir: str | Path) -> list[Path]:
    """按 benchmark 配置生成冻结后的 split 清单。"""
    return core_generate_split_manifests(benchmark_configs, output_dir)




def _run_method_batch(
    run_id: str,
    phase_name: str,
    experiment: ExperimentConfig,
    benchmark: BenchmarkConfig,
    model: ResolvedModelConfig,
    method: MethodConfig,
    samples: list[DatasetSample],
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    rate_limiter: SlidingWindowRateLimiter,
    progress: RunProgressTracker,
    rerun_index: int,
    raw_writer: BufferedJsonlWriter,
) -> list[dict[str, Any]]:
    """执行一个模型-数据集-方法-重跑组合下的一整批样本。"""
    split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
    call_specs: list[CallSpec] = []

    for sample_order, sample in enumerate(samples):
        messages = build_messages(sample, method.family, prompt_version=experiment.prompt_version)
        prompt_hash = build_prompt_hash(messages)
        for replicate_id in range(method.budget_calls):
            # SC 与 MV 共享同一 budget_calls 语义，只在后续聚合方式上区分。
            seed = (
                experiment.global_seed + rerun_index + replicate_id
                if method.family != "cot"
                else experiment.global_seed
            )
            payload = build_payload(
                config=model,
                messages=messages,
                temperature=method.temperature,
                top_p=method.top_p,
                max_output_tokens=method.max_output_tokens,
                seed=seed,
            )
            cache_key = build_request_cache_key(
                provider=model.provider,
                request_model=model.model_id,
                payload=payload,
            )
            call_specs.append(
                CallSpec(
                    run_id=run_id,
                    dataset=benchmark.cache_namespace or benchmark.slug,
                    split_name=split_name,
                    sample_id=sample.sample_id,
                    sample_order=sample_order,
                    method_name=method.name,
                    method_family=method.family,
                    rerun_index=rerun_index,
                    replicate_id=replicate_id,
                    agent_id=None,
                    model_name=model.name,
                    model_id=model.model_id,
                    provider_name=model.provider,
                    base_url=model.base_url,
                    prompt_hash=prompt_hash,
                    payload=payload,
                    cache_key=cache_key,
                    backbone=model,
                    messages=messages,
                    temperature=method.temperature,
                    top_p=method.top_p,
                    max_output_tokens=method.max_output_tokens,
                    seed=seed,
                )
            )

    worker = partial(
        _execute_call,
        provider=provider,
        cache=cache,
        rate_limiter=rate_limiter,
    )
    call_logs: list[dict[str, Any]] = []
    for _, call_log in run_indexed_batch(
        call_specs,
        worker=worker,
        max_concurrent_requests=experiment.max_concurrent_requests,
    ):
        call_logs.append(call_log)
        raw_writer.write_row(call_log)
        progress.record_call(call_log)

    grouped_logs: dict[str, list[dict[str, Any]]] = {}
    for call_log in call_logs:
        grouped_logs.setdefault(call_log["sample_id"], []).append(call_log)

    predictions: list[dict[str, Any]] = []
    for sample in samples:
        # 题级预测以“同一题的多次调用”为单位聚合，方便后续做方法级统计。
        per_call_logs = sorted(grouped_logs[sample.sample_id], key=lambda row: row["replicate_id"])
        votes = [log["normalized_answer"] for log in per_call_logs]
        request_failure_count = sum(1 for log in per_call_logs if log.get("request_error"))
        aggregate_answer, vote_counts = aggregate_majority(votes)
        question_prompt_tokens = _sum_usage(per_call_logs, "prompt_tokens")
        question_completion_tokens = _sum_usage(per_call_logs, "completion_tokens")
        question_total_tokens = _sum_usage(per_call_logs, "total_tokens")
        question_latency = sum(float(log.get("latency_ms") or 0.0) for log in per_call_logs)
        predictions.append(
            {
                "run_id": run_id,
                "dataset": benchmark.slug,
                "split": split_name,
                "sample_id": sample.sample_id,
                "method_name": method.name,
                "method_family": method.family,
                "model_name": model.name,
                "model_id": model.model_id,
                "rerun_index": rerun_index,
                "prediction": aggregate_answer,
                "gold": sample.reference_answer,
                "score": score_prediction(benchmark.slug, aggregate_answer, sample.reference_answer),
                "vote_counts": vote_counts,
                "request_failure_count": request_failure_count,
                "prompt_tokens_per_question": question_prompt_tokens,
                "completion_tokens_per_question": question_completion_tokens,
                "total_tokens_per_question": question_total_tokens,
                "latency_ms_per_question": question_latency,
                "calls_per_question": method.budget_calls,
            }
        )

    progress.record_predictions(len(predictions), benchmark.slug, method.name)
    return predictions


def _execute_call(
    spec: CallSpec,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    rate_limiter: SlidingWindowRateLimiter,
) -> dict[str, Any]:
    """执行一次实际调用，必要时命中缓存，并整理成统一日志结构。"""
    if (
        spec.backbone is not None
        and spec.messages is not None
        and spec.temperature is not None
        and spec.top_p is not None
        and spec.max_output_tokens is not None
    ):
        result = execute_cached_turn(
            backbone=spec.backbone,
            provider=provider,
            cache=cache,
            limiter=rate_limiter,
            messages=spec.messages,
            temperature=spec.temperature,
            top_p=spec.top_p,
            max_output_tokens=spec.max_output_tokens,
            seed=spec.seed,
            schema_id=SCHEMA_ANSWER_CORE,
        )
    else:
        cached = cache.get(spec.cache_key)
        if cached is None:
            response_payload = execute_completion_request(
                provider,
                spec.payload,
                limiter=rate_limiter,
            )
            cache_hit = False
        else:
            response_payload = json.loads(cached.response_json)
            cache_hit = True

        request_error = response_payload.get("request_error")
        if request_error:
            validated_output = {}
            output_status = "request_fail"
        else:
            try:
                validated_output = validate_or_recover_structured_output(
                    str(response_payload.get("assistant_text") or ""),
                    SCHEMA_ANSWER_CORE,
                    provider_reasoning_text=str(response_payload.get("provider_reasoning_text") or ""),
                )
                output_status = "ok"
                if not cache_hit:
                    cache_successful_response(
                        cache,
                        cache_key=spec.cache_key,
                        payload=spec.payload,
                        response_payload=response_payload,
                    )
            except Exception:
                validated_output = {}
                output_status = "schema_fail"

        usage = response_payload.get("usage_reported") or response_payload.get("usage_estimated") or {}
        result = type("CompatTurnResult", (), {
            "prompt_hash": spec.prompt_hash,
            "response_payload": response_payload,
            "validated_output": validated_output,
            "output_status": output_status,
            "usage": usage,
            "request_error": str(request_error) if request_error else None,
            "cache_hit": cache_hit,
            "payload": spec.payload,
        })()

    final_answer = str(result.validated_output.get("final_answer") or "")
    normalized_answer = normalize_prediction(spec.dataset, final_answer) if final_answer else ""
    return {
        "run_id": spec.run_id,
        "dataset": spec.dataset,
        "split": spec.split_name,
        "sample_id": spec.sample_id,
        "sample_order": spec.sample_order,
        "method_name": spec.method_name,
        "method_family": spec.method_family,
        "replicate_id": spec.replicate_id,
        "agent_id": spec.agent_id,
        "rerun_index": spec.rerun_index,
        "model_name": spec.model_name,
        "model_id": spec.model_id,
        "provider": spec.provider_name,
        "base_url": spec.base_url,
        "prompt_hash": result.prompt_hash,
        "assistant_text": result.response_payload.get("assistant_text", ""),
        "provider_reasoning_text": result.response_payload.get("provider_reasoning_text", ""),
        "validated_output": result.validated_output,
        "normalized_answer": normalized_answer,
        "output_status": result.output_status,
        "usage_reported": result.response_payload.get("usage_reported"),
        "usage_estimated": result.response_payload.get("usage_estimated"),
        "usage_source": result.response_payload.get("usage_source"),
        "latency_ms": result.response_payload.get("latency_ms"),
        "http_status": result.response_payload.get("http_status"),
        "provider_request_id": result.response_payload.get("provider_request_id"),
        "request_error": result.request_error,
        "cache_hit": result.cache_hit,
    }


def _aggregate_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """把题级预测聚合成论文与报告使用的 summary 指标。"""
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in predictions:
        key = (record["dataset"], record["model_name"], record["method_name"])
        grouped.setdefault(key, []).append(record)

    summary: list[dict[str, Any]] = []
    for (dataset, model_name, method_name), rows in grouped.items():
        scores = [float(row["score"]) for row in rows]
        prompt_tokens = [float(row["prompt_tokens_per_question"]) for row in rows]
        completion_tokens = [float(row["completion_tokens_per_question"]) for row in rows]
        total_tokens = [float(row["total_tokens_per_question"]) for row in rows]
        calls = [float(row["calls_per_question"]) for row in rows]
        latencies = [float(row["latency_ms_per_question"]) for row in rows]
        request_failures = [int(row.get("request_failure_count", 0)) for row in rows]

        rerun_scores: dict[int, list[float]] = {}
        for row in rows:
            rerun_scores.setdefault(int(row["rerun_index"]), []).append(float(row["score"]))
        rerun_means = [mean(values) for _, values in sorted(rerun_scores.items())]
        accuracy_mean = mean(rerun_means) if rerun_means else 0.0
        accuracy_std = pstdev(rerun_means) if len(rerun_means) > 1 else 0.0

        avg_total_tokens = mean(total_tokens) if total_tokens else 0.0
        summary.append(
            {
                "dataset": dataset,
                "model_name": model_name,
                "method_name": method_name,
                "prediction_rows": len(rows),
                "questions_per_rerun": int(len(rows) / len(rerun_means)) if rerun_means else 0,
                "rerun_count": len(rerun_means),
                "accuracy_mean": accuracy_mean,
                "accuracy_std": accuracy_std,
                "prompt_tokens_mean": mean(prompt_tokens) if prompt_tokens else 0.0,
                "completion_tokens_mean": mean(completion_tokens) if completion_tokens else 0.0,
                "total_tokens_mean": avg_total_tokens,
                "calls_per_question_mean": mean(calls) if calls else 0.0,
                "latency_ms_mean": mean(latencies) if latencies else 0.0,
                "acc_per_1k_tokens": (accuracy_mean / avg_total_tokens * 1000) if avg_total_tokens else 0.0,
                "rows_with_request_failures": sum(1 for value in request_failures if value > 0),
                "request_failures_total": sum(request_failures),
                "row_score_std": pstdev(scores) if len(scores) > 1 else 0.0,
                "rerun_accuracy_min": min(rerun_means) if rerun_means else 0.0,
                "rerun_accuracy_max": max(rerun_means) if rerun_means else 0.0,
            }
        )

    summary.sort(key=lambda row: (row["dataset"], row["model_name"], row["method_name"]))
    return {"summary": summary, "prediction_count": len(predictions)}


def _write_leaderboard(path: Path, predictions: list[dict[str, Any]]) -> None:
    """把 summary 指标导出为全局 leaderboard CSV。"""
    summary = _aggregate_metrics(predictions)["summary"]
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "dataset",
        "model_name",
        "method_name",
        "prediction_rows",
        "questions_per_rerun",
        "rerun_count",
        "accuracy_mean",
        "accuracy_std",
        "prompt_tokens_mean",
        "completion_tokens_mean",
        "total_tokens_mean",
        "calls_per_question_mean",
        "latency_ms_mean",
        "acc_per_1k_tokens",
        "rows_with_request_failures",
        "request_failures_total",
        "rerun_accuracy_min",
        "rerun_accuracy_max",
    ]
    lines = [",".join(headers)]
    for row in summary:
        lines.append(",".join(str(row[field]) for field in headers))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")




def _resolve_split_name(experiment: ExperimentConfig, phase_name: str, benchmark_slug: str) -> str:
    """解析某个 benchmark 在当前 phase 下实际使用的 split 名称。"""
    return resolve_phase_split_name(experiment, phase_name, benchmark_slug)


def _model_is_allowed(experiment: ExperimentConfig, phase_name: str, model: ResolvedModelConfig) -> bool:
    """检查模型是否满足 experiment / phase 级标签约束。"""
    required_tags = set(required_model_tags(experiment, phase_name))
    return required_tags.issubset(set(model.tags))


def _benchmark_is_allowed(
    experiment: ExperimentConfig,
    phase_name: str,
    model: ResolvedModelConfig,
    benchmark_slug: str,
) -> bool:
    """检查模型是否能在当前 phase 下运行指定 benchmark。"""
    phase = phase_metadata(experiment, phase_name)
    benchmark_filter = phase.get("benchmark_filter")
    if benchmark_filter is not None and benchmark_slug not in set(benchmark_filter):
        return False
    required_tags_for_benchmark = set(required_benchmark_tags(experiment, phase_name, benchmark_slug))
    return required_tags_for_benchmark.issubset(set(model.tags))


def _phase_methods(experiment: ExperimentConfig, phase_name: str, method_catalog: dict[str, MethodConfig]) -> list[MethodConfig]:
    """把 phase 中声明的方法名解析成完整方法配置，并在缺失时立即报错。"""
    method_names = experiment.raw["phases"][phase_name]["methods"]
    missing = [name for name in method_names if name not in method_catalog]
    if missing:
        raise RuntimeError(
            f"Experiment {experiment.name} phase {phase_name} references undefined methods: {', '.join(missing)}"
        )
    return [method_catalog[name] for name in method_names]


def _sum_usage(call_logs: list[dict[str, Any]], key: str) -> float:
    """优先使用 provider 上报 usage，不足时退回估算值。"""
    values = []
    for log in call_logs:
        usage = log["usage_reported"] or log["usage_estimated"] or {}
        value = usage.get(key)
        if value is not None:
            values.append(float(value))
    return sum(values)


def _estimate_run_work(
    experiment: ExperimentConfig,
    phase_name: str,
    models: list[ResolvedModelConfig],
    benchmarks: list[BenchmarkConfig],
    method_catalog: dict[str, MethodConfig],
) -> tuple[int, int]:
    """估算本次运行总调用数与总预测数，用于进度展示。"""
    methods = _phase_methods(experiment, phase_name, method_catalog)
    total_calls = 0
    total_predictions = 0

    for model in models:
        if not _model_is_allowed(experiment, phase_name, model):
            continue
        for benchmark in benchmarks:
            if not _benchmark_is_allowed(experiment, phase_name, model, benchmark.slug):
                continue
            split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
            split_size = len(load_split_ids(benchmark.cache_namespace or benchmark.slug, split_name))
            for method in methods:
                reruns = (
                    1 if method.family == "cot"
                    else int(phase_metadata(experiment, phase_name).get("reruns_override", experiment.reruns_per_method))
                )
                total_calls += split_size * method.budget_calls * reruns
                total_predictions += split_size * reruns

    return total_calls, total_predictions


def _ensure_run_has_eligible_work(
    experiment: ExperimentConfig,
    phase_name: str,
    models: list[ResolvedModelConfig],
    benchmarks: list[BenchmarkConfig],
) -> None:
    """在真正发请求前做 fail-fast 校验。

    如果模型缺少必需标签，或者当前 phase 下没有任何 benchmark 可运行，
    这里会直接报错，避免消耗无意义的 API 配额。
    """
    for model in models:
        if not _model_is_allowed(experiment, phase_name, model):
            required_tags = ", ".join(required_model_tags(experiment, phase_name))
            raise RuntimeError(
                f"Model {model.name} is missing required tags for phase {phase_name}. "
                f"Required tags: [{required_tags}] | model tags: [{', '.join(model.tags)}]"
            )
        eligible_benchmarks = [
            benchmark.slug
            for benchmark in benchmarks
            if _benchmark_is_allowed(experiment, phase_name, model, benchmark.slug)
        ]
        if eligible_benchmarks:
            return

        benchmark_requirements = {
            benchmark.slug: required_benchmark_tags(experiment, phase_name, benchmark.slug)
            for benchmark in benchmarks
        }
        raise RuntimeError(
            f"Model {model.name} is not eligible for any benchmark in phase {phase_name}. "
            f"Benchmark tag requirements: {json.dumps(benchmark_requirements, ensure_ascii=False)} | "
            f"model tags: [{', '.join(model.tags)}]"
        )


