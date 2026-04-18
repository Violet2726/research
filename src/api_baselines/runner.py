"""实验主运行链路。

该模块负责把实验规格转换成真实的 API 调用计划，并在一次运行中完成：
phase 解析、样本展开、并发请求、缓存复用、解析兜底、指标汇总、
报告导出与运行后校验。
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
from hashlib import sha256
from pathlib import Path
import json
import random
from statistics import mean, pstdev
import threading
import time
from typing import Any

from dotenv import load_dotenv

from api_baselines.cache import CachedResponse, RequestCache, json_dump
from api_baselines.config import (
    BenchmarkConfig,
    ExperimentConfig,
    MethodConfig,
    ResolvedModelConfig,
    load_method_catalog,
    phase_metadata,
    required_benchmark_tags,
    required_model_tags,
)
from api_baselines.datasets import DatasetSample, load_samples
from api_baselines.evaluation import aggregate_majority, normalize_prediction, score_prediction
from api_baselines.fallbacks import extract_fallback_answer
from api_baselines.parsing import parse_model_output
from api_baselines.prompting import build_messages
from api_baselines.providers import (
    OpenAICompatibleProvider,
    ProviderRequestError,
    build_payload,
    estimate_request_tokens,
)
from api_baselines.rate_limits import SlidingWindowRateLimiter
from api_baselines.reporting import budget_fairness_check, export_paper_tables, summarize_run
from api_baselines.validation import validate_run
from experiment_common.runtime import RunProgressTracker, build_run_id, load_split_ids, select_samples


@dataclass(frozen=True)
class RunPaths:
    """单次运行目录下各类产物文件的固定路径集合。"""

    root: Path
    manifest: Path
    raw_responses: Path
    predictions: Path
    metrics: Path
    run_summary: Path
    budget_fairness: Path
    paper_tables: Path
    run_validation: Path
    progress: Path


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


def generate_split_manifests(benchmark_configs: list[BenchmarkConfig], output_dir: str | Path) -> list[Path]:
    """按 benchmark 配置生成冻结后的 split 清单。"""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    for config in benchmark_configs:
        samples = load_samples(config)
        indexed_ids = [sample.sample_id for sample in samples]
        shuffled = indexed_ids[:]
        random.Random(config.random_seed).shuffle(shuffled)

        split_specs = [
            (f"{config.slug}-smoke20_seed42.json", shuffled[: config.smoke_size]),
            (f"{config.slug}-pilot100_seed42.json", shuffled[: min(config.pilot_size, len(shuffled))]),
        ]
        if config.slug == "strategyqa":
            split_specs.append((f"{config.slug}-dev_full_229_seed42.json", indexed_ids[:]))
        else:
            split_specs.append((f"{config.slug}-dev300_seed42.json", shuffled[: min(config.main_size, len(shuffled))]))

        for filename, sample_ids in split_specs:
            path = output / filename
            payload = {
                "dataset": config.slug,
                "source_split": config.source_split,
                "sample_count": len(sample_ids),
                "sample_ids": sample_ids,
                "random_seed": config.random_seed,
            }
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            created.append(path)

    return created


def run_experiment(
    experiment: ExperimentConfig,
    phase_name: str,
    models: list[ResolvedModelConfig],
    benchmarks: list[BenchmarkConfig],
    run_root: str | Path = "runs",
    cache_path: str | Path = "cache/requests.sqlite",
) -> Path:
    """执行一个 experiment phase，并产出完整运行目录。"""
    load_dotenv(".env.local", override=False)
    phase = phase_metadata(experiment, phase_name)
    method_catalog = load_method_catalog(experiment.method_catalog)
    methods = _phase_methods(experiment, phase_name, method_catalog)
    run_id = build_run_id(experiment.name, phase_name)
    run_paths = _prepare_run_paths(run_root, run_id)
    cache = RequestCache(cache_path)
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
        "experiment": experiment.name,
        "phase": phase_name,
        "description": experiment.description,
        "prompt_version": experiment.prompt_version,
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
    with run_paths.raw_responses.open("w", encoding="utf-8") as raw_handle, run_paths.predictions.open("w", encoding="utf-8") as pred_handle:
        for model in models:
            if not _model_is_allowed(experiment, phase_name, model):
                continue
            provider = OpenAICompatibleProvider(model)
            for benchmark in benchmarks:
                if not _benchmark_is_allowed(experiment, phase_name, model, benchmark.slug):
                    continue
                split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
                selected_samples = select_samples(benchmark, split_name)

                for method in methods:
                    # CoT 是单次确定性求解；其余方法通过多次调用形成同预算对比。
                    reruns = 1 if method.family == "cot" else int(phase.get("reruns_override", experiment.reruns_per_method))
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
                            raw_handle=raw_handle,
                        )
                        for record in batch_predictions:
                            pred_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        pred_handle.flush()
                        all_predictions.extend(batch_predictions)

    metrics_payload = _aggregate_metrics(all_predictions)
    run_paths.metrics.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_leaderboard(Path("reports/leaderboard.csv"), all_predictions)
    run_paths.run_summary.write_text(
        json.dumps(summarize_run(run_paths.root), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    run_paths.budget_fairness.write_text(
        json.dumps(budget_fairness_check(run_paths.root), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    export_paper_tables(run_paths.root, run_paths.paper_tables)
    run_paths.run_validation.write_text(
        json.dumps(validate_run(run_paths.root), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    progress.mark_completed()
    cache.close()
    return run_paths.root


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
    raw_handle,
) -> list[dict[str, Any]]:
    """执行一个模型-数据集-方法-重跑组合下的整批样本。"""
    split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
    call_specs: list[CallSpec] = []

    for sample_order, sample in enumerate(samples):
        messages = build_messages(sample, method.family)
        prompt_hash = _prompt_hash(messages)
        for replicate_id in range(method.budget_calls):
            # SC 与 MV 共享同一 budget_calls 语义，只在后续聚合方式上区分。
            payload = build_payload(
                config=model,
                messages=messages,
                temperature=method.temperature,
                top_p=method.top_p,
                max_output_tokens=method.max_output_tokens,
                seed=experiment.global_seed + rerun_index + replicate_id if method.family != "cot" else experiment.global_seed,
            )
            cache_key = _cache_key(
                dataset=benchmark.slug,
                sample_id=sample.sample_id,
                split_name=split_name,
                method_name=method.name,
                replicate_id=replicate_id,
                model_name=model.name,
                rerun_index=rerun_index,
                prompt_hash=prompt_hash,
                payload=payload,
            )
            call_specs.append(
                CallSpec(
                    run_id=run_id,
                    dataset=benchmark.slug,
                    split_name=split_name,
                    sample_id=sample.sample_id,
                    sample_order=sample_order,
                    method_name=method.name,
                    method_family=method.family,
                    rerun_index=rerun_index,
                    replicate_id=replicate_id,
                    agent_id=replicate_id if method.family == "majority_vote" else None,
                    model_name=model.name,
                    model_id=model.model_id,
                    provider_name=model.provider,
                    base_url=model.base_url,
                    prompt_hash=prompt_hash,
                    payload=payload,
                    cache_key=cache_key,
                )
            )

    worker = partial(
        _execute_call,
        provider=provider,
        cache=cache,
        rate_limiter=rate_limiter,
    )
    max_workers = max(1, min(experiment.max_concurrent_requests, len(call_specs) or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_spec = {executor.submit(worker, spec): spec for spec in call_specs}
        call_logs: list[dict[str, Any]] = []
        for future in as_completed(future_to_spec):
            call_log = future.result()
            call_logs.append(call_log)
            raw_handle.write(json.dumps(call_log, ensure_ascii=False) + "\n")
            raw_handle.flush()
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
                "method": method.name,
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
    """执行一次实际调用，必要时命中缓存，并把响应整理成统一日志行。"""
    cached = cache.get(spec.cache_key)

    if cached is None:
        # 只有真正发网路请求时才占用限流配额；缓存命中不计入。
        rate_limiter.acquire(estimate_request_tokens(spec.payload))
        try:
            provider_response = provider.chat_completion(spec.payload)
            response_payload = {
                "http_status": provider_response.http_status,
                "raw_payload": provider_response.raw_payload,
                "raw_text": provider_response.raw_text,
                "finish_reason": provider_response.finish_reason,
                "usage_reported": provider_response.usage_reported,
                "usage_estimated": provider_response.usage_estimated,
                "usage_source": provider_response.usage_source,
                "latency_ms": provider_response.latency_ms,
                "provider_request_id": provider_response.provider_request_id,
                "response_id": provider_response.response_id,
                "request_error": None,
            }
            cache.put(
                CachedResponse(
                    cache_key=spec.cache_key,
                    payload_json=json_dump(spec.payload),
                    response_json=json_dump(response_payload),
                    http_status=provider_response.http_status,
                    latency_ms=provider_response.latency_ms,
                    provider_request_id=provider_response.provider_request_id,
                )
            )
        except ProviderRequestError as exc:
            response_payload = {
                "http_status": exc.http_status,
                "raw_payload": {"error": exc.message},
                "raw_text": "",
                "finish_reason": None,
                "usage_reported": None,
                "usage_estimated": None,
                "usage_source": "missing",
                "latency_ms": 0.0,
                "provider_request_id": exc.provider_request_id,
                "response_id": None,
                "request_error": exc.message,
            }
        cache_hit = False
    else:
        response_payload = json.loads(cached.response_json)
        cache_hit = True

    request_error = response_payload.get("request_error")
    if request_error:
        parsed = {}
        parse_status = "request_fail"
        final_answer = ""
    else:
        try:
            parsed, parse_status = parse_model_output(response_payload["raw_text"])
            final_answer = str(parsed.get("final_answer", "")).strip()
        except Exception:
            # 当模型没返回合法 JSON 时，再按数据集规则做保守兜底。
            fallback = extract_fallback_answer(spec.dataset, response_payload["raw_text"])
            if fallback is None:
                parsed = {}
                parse_status = "parse_fail"
                final_answer = ""
            else:
                parsed, parse_status = fallback
                final_answer = str(parsed.get("final_answer", "")).strip()

    normalized_answer = normalize_prediction(spec.dataset, final_answer)
    return {
        "run_id": spec.run_id,
        "dataset": spec.dataset,
        "split": spec.split_name,
        "sample_id": spec.sample_id,
        "sample_order": spec.sample_order,
        "method": spec.method_name,
        "method_family": spec.method_family,
        "replicate_id": spec.replicate_id,
        "agent_id": spec.agent_id,
        "rerun_index": spec.rerun_index,
        "model_name": spec.model_name,
        "model_id": spec.model_id,
        "provider": spec.provider_name,
        "base_url": spec.base_url,
        "prompt_hash": spec.prompt_hash,
        "raw_response": response_payload["raw_text"],
        "parsed_output": parsed,
        "parsed_answer": final_answer,
        "normalized_answer": normalized_answer,
        "parse_status": parse_status,
        "usage_reported": response_payload.get("usage_reported"),
        "usage_estimated": response_payload.get("usage_estimated"),
        "usage_source": response_payload.get("usage_source"),
        "latency_ms": response_payload.get("latency_ms"),
        "http_status": response_payload.get("http_status"),
        "provider_request_id": response_payload.get("provider_request_id"),
        "request_error": request_error,
        "cache_hit": cache_hit,
    }


def _aggregate_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """把题级预测聚合成论文与报告使用的 summary 指标。"""
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in predictions:
        key = (record["dataset"], record["model_name"], record["method"])
        grouped.setdefault(key, []).append(record)

    summary: list[dict[str, Any]] = []
    for (dataset, model_name, method), rows in grouped.items():
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
                "method": method,
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

    summary.sort(key=lambda row: (row["dataset"], row["model_name"], row["method"]))
    return {"summary": summary, "prediction_count": len(predictions)}


def _write_leaderboard(path: Path, predictions: list[dict[str, Any]]) -> None:
    """把 summary 指标导出成全局 leaderboard CSV。"""
    summary = _aggregate_metrics(predictions)["summary"]
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "dataset",
        "model_name",
        "method",
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


def _prepare_run_paths(run_root: str | Path, run_id: str) -> RunPaths:
    """创建运行目录，并返回其中所有固定产物路径。"""
    root = Path(run_root) / run_id
    root.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        root=root,
        manifest=root / "manifest.json",
        raw_responses=root / "raw_responses.jsonl",
        predictions=root / "predictions.jsonl",
        metrics=root / "metrics.json",
        run_summary=root / "run_summary.json",
        budget_fairness=root / "budget_fairness.json",
        paper_tables=root / "paper_tables.md",
        run_validation=root / "run_validation.json",
        progress=root / "progress.json",
    )


def _resolve_split_name(experiment: ExperimentConfig, phase_name: str, benchmark_slug: str) -> str:
    """解析某个 benchmark 在当前 phase 下实际使用的 split 名称。"""
    phase = phase_metadata(experiment, phase_name)
    if "split_overrides" in phase:
        return phase["split_overrides"][benchmark_slug]
    return phase["split_suffix"]


def _model_is_allowed(experiment: ExperimentConfig, phase_name: str, model: ResolvedModelConfig) -> bool:
    """检查模型是否满足 experiment/phase 级标签约束。"""
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


def _cache_key(
    dataset: str,
    sample_id: str,
    split_name: str,
    method_name: str,
    replicate_id: int,
    model_name: str,
    rerun_index: int,
    prompt_hash: str,
    payload: dict[str, Any],
) -> str:
    """构造稳定缓存键。

    只要 prompt、payload、方法、模型或重跑索引变化，缓存键就会变化。
    """
    fingerprint = {
        "dataset": dataset,
        "sample_id": sample_id,
        "split_name": split_name,
        "method_name": method_name,
        "replicate_id": replicate_id,
        "model_name": model_name,
        "rerun_index": rerun_index,
        "prompt_hash": prompt_hash,
        "payload": payload,
    }
    return sha256(json.dumps(fingerprint, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _prompt_hash(messages: list[dict[str, str]]) -> str:
    """对提示词内容做稳定哈希，便于事后验证公平性。"""
    return sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


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
            split_size = len(load_split_ids(benchmark.slug, split_name))
            for method in methods:
                reruns = 1 if method.family == "cot" else int(phase_metadata(experiment, phase_name).get("reruns_override", experiment.reruns_per_method))
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
    这里会直接抛错，避免消耗无意义的 API 配额。
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
