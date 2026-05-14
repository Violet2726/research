"""DoG family 的顶层实验执行入口。"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
import json

from dotenv import load_dotenv

from research_experiments.core.execution.artifacts import BufferedJsonlWriter
from research_experiments.core.execution.cache import RequestCacheRouter
from research_experiments.core.execution.providers import OpenAICompatibleProvider
from research_experiments.core.execution.rate_limits import SlidingWindowRateLimiter
from research_experiments.core.execution.runtime import RunProgressTracker, build_run_id, finalize_run_outputs
from research_experiments.workspace.layout import default_cache_root, default_runs_root
from research_experiments.families.dog_graph.config import (
    DogGraphExperimentConfig,
    PaperProtocolConfig,
    StaticProtocolConfig,
    load_benchmarks,
    load_protocol_config,
)
from research_experiments.families.dog_graph.paper_backend import build_backend_for_benchmark
from research_experiments.families.dog_graph.run.io import _prepare_run_paths
from research_experiments.families.dog_graph.run.report import render_report, summarize_run
from research_experiments.families.dog_graph.run.validate import validate_run
from research_experiments.families.dog_graph.run import paper as paper_run
from research_experiments.families.dog_graph.run import sample as static_run


def run_experiment(
    experiment: DogGraphExperimentConfig,
    phase_name: str,
    backbone,
    run_root: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> Path:
    """执行一个 DoG phase，并写出完整运行目录。"""

    load_dotenv(".env.local", override=False)
    run_root = run_root or default_runs_root("dog_graph")
    cache_root = cache_root or default_cache_root()
    benchmarks = load_benchmarks(experiment)
    methods = _active_methods(experiment)
    protocol = load_protocol_config(experiment.protocol)
    provider = OpenAICompatibleProvider(backbone)
    cache_router = RequestCacheRouter(cache_root)
    limiter = SlidingWindowRateLimiter(
        requests_per_minute=experiment.requests_per_minute_limit,
        tokens_per_minute=experiment.tokens_per_minute_limit,
    )
    run_id = build_run_id(backbone.name)
    run_paths = _prepare_run_paths(run_root, experiment.name, phase_name, run_id)

    selected_batches: list[tuple[object, str, list[object]]] = []
    total_calls = 0
    total_predictions = 0
    for benchmark in benchmarks:
        split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
        samples = _load_selected_samples(experiment, benchmark, split_name)
        selected_batches.append((benchmark, split_name, samples))
        sample_count = len(samples)
        total_predictions += sample_count * len(methods)
        if experiment.experiment_kind == "paper":
            protocol = _expect_paper_protocol(protocol)
            total_calls += sum(
                paper_run._planned_calls_per_sample(method, sample, protocol)
                for sample in samples
                for method in methods
            )
        else:
            total_calls += sample_count * sum(static_run._planned_calls_per_sample(method) for method in methods)

    progress = RunProgressTracker(run_paths.progress, total_calls, total_predictions)
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "family_name": "dog_graph",
        "experiment_name": experiment.name,
        "phase_name": phase_name,
        "primary_model_ref": experiment.primary_model_ref,
        "resolved_model": asdict(backbone),
        "description": experiment.description,
        "experiment_kind": experiment.experiment_kind,
        "phase_metadata": dict(experiment.raw["phases"][phase_name]),
        "prompt_version": experiment.prompt_version,
        "protocol": _serialize_protocol(protocol),
        "methods": [asdict(method) for method in methods],
        "benchmarks": [asdict(item) for item in benchmarks],
        "total_planned_calls": total_calls,
        "total_planned_predictions": total_predictions,
    }
    run_paths.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    all_turn_rows: list[dict[str, object]] = []
    all_graph_rows: list[dict[str, object]] = []
    final_predictions: list[dict[str, object]] = []
    retrieval_rows: list[dict[str, object]] = []
    relation_rows: list[dict[str, object]] = []
    simplification_rows: list[dict[str, object]] = []
    answer_attempt_rows: list[dict[str, object]] = []

    backends: dict[str, object] = {}
    if experiment.experiment_kind == "paper":
        paper_protocol = _expect_paper_protocol(protocol)
        backends = {
            benchmark.slug: build_backend_for_benchmark(
                benchmark,
                freebase_sparql_url=paper_protocol.freebase_sparql_url,
                freebase_backend_mode=paper_protocol.freebase_backend_mode,
            )
            for benchmark in benchmarks
        }

    try:
        with (
            run_paths.agent_turns.open("w", encoding="utf-8") as turn_handle_raw,
            run_paths.debate_messages.open("w", encoding="utf-8") as debate_handle_raw,
            run_paths.graph_trace.open("w", encoding="utf-8") as graph_handle_raw,
            run_paths.retrieval_trace.open("w", encoding="utf-8") as retrieval_handle_raw,
            run_paths.relation_selection_trace.open("w", encoding="utf-8") as relation_handle_raw,
            run_paths.simplification_trace.open("w", encoding="utf-8") as simplification_handle_raw,
            run_paths.answer_attempt_trace.open("w", encoding="utf-8") as answer_handle_raw,
            run_paths.final_predictions.open("w", encoding="utf-8") as prediction_handle_raw,
        ):
            turn_handle = BufferedJsonlWriter(turn_handle_raw)
            debate_handle = BufferedJsonlWriter(debate_handle_raw)
            graph_handle = BufferedJsonlWriter(graph_handle_raw)
            retrieval_handle = BufferedJsonlWriter(retrieval_handle_raw)
            relation_handle = BufferedJsonlWriter(relation_handle_raw)
            simplification_handle = BufferedJsonlWriter(simplification_handle_raw)
            answer_handle = BufferedJsonlWriter(answer_handle_raw)
            prediction_handle = BufferedJsonlWriter(prediction_handle_raw)

            for benchmark, split_name, samples in selected_batches:
                cache = cache_router.for_request_target(
                    provider=backbone.provider,
                    request_model=backbone.model_id,
                    dataset=_cache_dataset_key(experiment.experiment_kind, benchmark),
                )
                for method in methods:
                    if experiment.experiment_kind == "paper":
                        paper_protocol = _expect_paper_protocol(protocol)
                        method_results = paper_run._run_method_batch(
                            run_id=run_id,
                            benchmark_slug=benchmark.slug,
                            split_name=split_name,
                            samples=samples,
                            method=method,
                            protocol=paper_protocol,
                            backend=backends[benchmark.slug],
                            backbone=backbone,
                            provider=provider,
                            cache=cache,
                            limiter=limiter,
                            global_seed=experiment.global_seed,
                            max_concurrent_requests=experiment.max_concurrent_requests,
                        )
                        paper_run._write_method_outputs(
                            batch_results=method_results,
                            dataset_slug=benchmark.slug,
                            progress=progress,
                            turn_handle=turn_handle,
                            debate_handle=debate_handle,
                            graph_handle=graph_handle,
                            prediction_handle=prediction_handle,
                            retrieval_handle=retrieval_handle,
                            relation_handle=relation_handle,
                            simplification_handle=simplification_handle,
                            answer_attempt_handle=answer_handle,
                            all_turn_rows=all_turn_rows,
                            all_graph_rows=all_graph_rows,
                            retrieval_rows=retrieval_rows,
                            relation_rows=relation_rows,
                            simplification_rows=simplification_rows,
                            answer_attempt_rows=answer_attempt_rows,
                            final_predictions=final_predictions,
                        )
                    else:
                        static_protocol = _expect_static_protocol(protocol)
                        method_results = static_run._run_method_batch(
                            run_id=run_id,
                            benchmark_slug=benchmark.slug,
                            split_name=split_name,
                            samples=samples,
                            method=method,
                            protocol=static_protocol,
                            backbone=backbone,
                            provider=provider,
                            cache=cache,
                            limiter=limiter,
                            global_seed=experiment.global_seed,
                            prompt_version=experiment.prompt_version,
                            max_concurrent_requests=experiment.max_concurrent_requests,
                        )
                        static_run._write_sample_outputs(
                            sample_results=method_results,
                            dataset_slug=benchmark.slug,
                            progress=progress,
                            turn_handle=turn_handle,
                            debate_handle=debate_handle,
                            graph_handle=graph_handle,
                            prediction_handle=prediction_handle,
                            all_turns=all_turn_rows,
                            all_graph_trace_rows=all_graph_rows,
                            final_predictions=final_predictions,
                        )

        if experiment.experiment_kind == "paper":
            metrics = paper_run._build_metrics(final_predictions, methods)
            graph_diagnostics = paper_run._build_graph_diagnostics(
                final_predictions,
                retrieval_rows,
                relation_rows,
                simplification_rows,
                answer_attempt_rows,
            )
            cost_breakdown = paper_run._build_cost_breakdown(all_turn_rows)
        else:
            metrics = static_run._build_metrics(final_predictions, methods)
            graph_diagnostics = static_run._build_graph_diagnostics(final_predictions, all_graph_rows)
            cost_breakdown = static_run._build_cost_breakdown(all_turn_rows)

        run_paths.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        run_paths.graph_diagnostics.write_text(json.dumps(graph_diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
        run_paths.cost_breakdown.write_text(json.dumps(cost_breakdown, ensure_ascii=False, indent=2), encoding="utf-8")
        run_paths.run_summary.write_text(json.dumps(summarize_run(run_paths.root), ensure_ascii=False, indent=2), encoding="utf-8")
        render_report(run_paths.root)
        finalize_run_outputs(
            run_paths.root,
            validator=validate_run,
            validation_path=run_paths.run_validation,
        )
        progress.mark_completed()
        return run_paths.root
    finally:
        provider.close()
        cache_router.close()
        for backend in backends.values():
            close = getattr(backend, "close", None)
            if callable(close):
                close()


def _active_methods(experiment: DogGraphExperimentConfig):
    if experiment.experiment_kind == "paper":
        return paper_run._active_methods(experiment)
    return static_run._active_methods(experiment)


def _resolve_split_name(experiment: DogGraphExperimentConfig, phase_name: str, benchmark_slug: str) -> str:
    if experiment.experiment_kind == "paper":
        return paper_run._resolve_split_name(experiment, phase_name, benchmark_slug)
    return static_run._resolve_split_name(experiment, phase_name, benchmark_slug)


def _load_selected_samples(experiment: DogGraphExperimentConfig, benchmark, split_name: str):
    if experiment.experiment_kind == "paper":
        return paper_run._load_selected_samples(benchmark, split_name)
    return static_run._load_selected_samples(benchmark, split_name)


def _expect_paper_protocol(protocol) -> PaperProtocolConfig:
    if not isinstance(protocol, PaperProtocolConfig):
        raise TypeError("Expected PaperProtocolConfig for DoG paper experiment.")
    return protocol


def _expect_static_protocol(protocol) -> StaticProtocolConfig:
    if not isinstance(protocol, StaticProtocolConfig):
        raise TypeError("Expected StaticProtocolConfig for DoG static experiment.")
    return protocol


def _serialize_protocol(protocol) -> dict[str, object]:
    if is_dataclass(protocol):
        return asdict(protocol)
    return dict(protocol)


def _cache_dataset_key(experiment_kind: str, benchmark) -> str:
    """为 DoG 运行解析真正的缓存分库键。

    `dog_graph_main` 的 benchmark slug 带有 `dog_` 前缀，是为了和仓库里已有
    的 benchmark 配置及 split manifest 解耦；但 cache 分库应该按“真实数据集”
    归档，而不是按实验线名字归档。
    """

    if experiment_kind != "paper":
        return benchmark.cache_namespace or benchmark.slug
    return benchmark.cache_namespace or benchmark.slug
