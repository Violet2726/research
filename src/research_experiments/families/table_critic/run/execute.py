"""Table-Critic family 的顶层实验执行入口。"""

from __future__ import annotations

from dataclasses import asdict
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
from research_experiments.families.shared.config_loading import load_benchmarks
from research_experiments.families.table_critic.config import ProtocolConfig, TableCriticExperimentConfig, load_protocol_config
from research_experiments.families.table_critic.run.io import _prepare_run_paths
from research_experiments.families.table_critic.run.report import render_report, summarize_run
from research_experiments.families.table_critic.run.sample import (
    _active_methods,
    _build_error_analysis,
    _build_metrics,
    _estimate_work,
    _load_selected_samples,
    _resolve_split_name,
    _run_simple_method_batch,
    _run_table_critic_sequence,
    _seed_template_tree,
    _serialize_template_tree,
)
from research_experiments.families.table_critic.run.validate import validate_run


def run_experiment(
    experiment: TableCriticExperimentConfig,
    phase_name: str,
    backbone,
    run_root: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> Path:
    """执行一个 Table-Critic phase，并写出完整运行目录。"""

    load_dotenv(".env.local", override=False)
    run_root = run_root or default_runs_root("table_critic")
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
    total_calls, total_predictions = _estimate_work(experiment, phase_name, benchmarks, protocol, methods)
    progress = RunProgressTracker(
        run_paths.progress,
        total_calls,
        total_predictions,
        planned_calls_are_upper_bound=True,
    )

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "family_name": "table_critic",
        "experiment_name": experiment.name,
        "experiment_kind": experiment.experiment_kind,
        "phase_name": phase_name,
        "primary_model_ref": experiment.primary_model_ref,
        "resolved_model": asdict(backbone),
        "description": experiment.description,
        "prompt_version": experiment.prompt_version,
        "protocol": asdict(protocol),
        "methods": [asdict(method) for method in methods],
        "benchmarks": [asdict(item) for item in benchmarks],
        "total_planned_calls": total_calls,
        "total_planned_predictions": total_predictions,
    }
    run_paths.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    all_turn_rows: list[dict[str, object]] = []
    critic_rows: list[dict[str, object]] = []
    refinement_rows: list[dict[str, object]] = []
    final_predictions: list[dict[str, object]] = []
    dataset_trees: dict[str, dict[str, Any]] = {}

    with (
        run_paths.agent_turns.open("w", encoding="utf-8") as turn_handle_raw,
        run_paths.critic_trace.open("w", encoding="utf-8") as critic_handle_raw,
        run_paths.refinement_trace.open("w", encoding="utf-8") as refinement_handle_raw,
        run_paths.final_predictions.open("w", encoding="utf-8") as prediction_handle_raw,
    ):
        turn_handle = BufferedJsonlWriter(turn_handle_raw)
        critic_handle = BufferedJsonlWriter(critic_handle_raw)
        refinement_handle = BufferedJsonlWriter(refinement_handle_raw)
        prediction_handle = BufferedJsonlWriter(prediction_handle_raw)

        for benchmark in benchmarks:
            cache = cache_router.for_request_target(
                provider=backbone.provider,
                request_model=backbone.model_id,
                dataset=benchmark.cache_namespace or benchmark.slug,
            )
            split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
            samples = _load_selected_samples(benchmark, split_name)
            dataset_tree = _seed_template_tree(protocol)
            dataset_trees[benchmark.slug] = dataset_tree

            for method in methods:
                if method.mode == "table_critic_paper":
                    batch_results = _run_table_critic_sequence(
                        run_id=run_id,
                        benchmark_slug=benchmark.slug,
                        split_name=split_name,
                        samples=samples,
                        method=method,
                        protocol=protocol,
                        backbone=backbone,
                        provider=provider,
                        cache=cache,
                        limiter=limiter,
                        global_seed=experiment.global_seed,
                        template_tree=dataset_tree,
                    )
                else:
                    batch_results = _run_simple_method_batch(
                        run_id=run_id,
                        benchmark_slug=benchmark.slug,
                        split_name=split_name,
                        samples=samples,
                        method=method,
                        protocol=protocol,
                        backbone=backbone,
                        provider=provider,
                        cache=cache,
                        limiter=limiter,
                        global_seed=experiment.global_seed,
                        max_concurrent_requests=experiment.max_concurrent_requests,
                    )

                for _, payload in batch_results:
                    for row in payload["turn_rows"]:
                        row["model_name"] = backbone.name
                        turn_handle.write_row(row)
                        progress.record_call(row, method_key="method_name")
                    for row in payload["critic_rows"]:
                        critic_handle.write_row(row)
                    for row in payload["refinement_rows"]:
                        refinement_handle.write_row(row)
                    prediction_row = dict(payload["prediction_row"])
                    prediction_row["model_name"] = backbone.name
                    prediction_handle.write_row(prediction_row)
                    progress.record_predictions(1, benchmark.slug, method.name)

                    all_turn_rows.extend(payload["turn_rows"])
                    critic_rows.extend(payload["critic_rows"])
                    refinement_rows.extend(payload["refinement_rows"])
                    final_predictions.append(prediction_row)

    metrics = _build_metrics(final_predictions, methods, model_name=backbone.name)
    error_analysis = _build_error_analysis(final_predictions, critic_rows)
    serialized_tree = {
        "datasets": {
            dataset: _serialize_template_tree(tree)
            for dataset, tree in sorted(dataset_trees.items())
        }
    }

    run_paths.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    run_paths.template_tree.write_text(json.dumps(serialized_tree, ensure_ascii=False, indent=2), encoding="utf-8")
    run_paths.error_analysis.write_text(json.dumps(error_analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    run_paths.run_summary.write_text(json.dumps(summarize_run(run_paths.root), ensure_ascii=False, indent=2), encoding="utf-8")
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

