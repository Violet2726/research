"""A-SMAD 实验执行入口。

本模块负责一次完整 run 的资源准备、样本调度、产物写入、报告渲染和最终校验。
样本级模型调用与聚合细节下沉到 `sample.py`，这里保持运行编排职责。
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from dotenv import load_dotenv

from research_experiments.core.config import BenchmarkConfig
from research_experiments.core.data.datasets import select_samples
from research_experiments.core.execution.artifacts import BufferedJsonlWriter
from research_experiments.core.execution.cache import RequestCacheRouter
from research_experiments.core.execution.providers import OpenAICompatibleProvider
from research_experiments.core.execution.rate_limits import RequestThrottle
from research_experiments.core.execution.runtime import RunProgressTracker, build_run_id, finalize_run_outputs
from research_experiments.core.io import read_json, read_jsonl, write_json, write_jsonl
from research_experiments.core.structured_outputs import ARTIFACT_VERSION
from research_experiments.families.adaptive_sparse_mad.config import (
    ADAPTIVE_POLICY_METHODS,
    AdaptiveSparseMadExperimentConfig,
    AdaptiveSparseMadProtocolConfig,
    load_control_catalog,
    load_protocol_config,
)
from research_experiments.families.adaptive_sparse_mad.run.report import render_report
from research_experiments.families.adaptive_sparse_mad.run.sample import (
    append_sample_result,
    build_metrics_payload,
    build_policy_diagnostics,
    build_router_eval_payload,
    build_stage_a_error_bucket_payload,
    build_stage_a_resolver_breakdown_payload,
    build_stage_a_solver_contribution_payload,
    estimate_work,
    refresh_prediction_rows_for_run,
    run_sample_batch,
    summarize_run,
)
from research_experiments.families.adaptive_sparse_mad.run.validate import validate_run
from research_experiments.family_runtime.config_helpers import load_benchmarks, phase_metadata
from research_experiments.family_runtime.layout import prepare_registered_run_layout
from research_experiments.family_runtime.manifest import finalize_family_manifest
from research_experiments.workspace.layout import default_cache_root, default_runs_root
from research_experiments.workspace.run_archives import pack_run_artifacts


def run_experiment(
    experiment: AdaptiveSparseMadExperimentConfig,
    phase_name: str,
    backbone,
    run_root: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> Path:
    """执行指定实验 phase，写入全部 run 产物并返回运行目录。"""
    load_dotenv(".env.local", override=False)
    run_root = run_root or default_runs_root("adaptive_sparse_mad")
    cache_root = cache_root or default_cache_root()
    benchmarks = load_benchmarks(experiment)
    protocol = load_protocol_config(experiment.protocol)
    controls = load_control_catalog(experiment.control_catalog)
    provider = OpenAICompatibleProvider(backbone)
    cache_router = RequestCacheRouter(cache_root)
    throttle = RequestThrottle.for_model(
        backbone,
        max_concurrent_requests=experiment.max_concurrent_requests,
        requests_per_minute=experiment.requests_per_minute_limit,
    )
    run_id = build_run_id(backbone.name)
    run_paths = prepare_registered_run_layout(
        "adaptive_sparse_mad",
        run_root,
        experiment.name,
        phase_name,
        run_id,
    )
    total_calls, total_predictions = estimate_work(experiment, phase_name, benchmarks, controls)
    progress = RunProgressTracker(
        run_paths.progress,
        total_calls,
        total_predictions,
        planned_calls_are_upper_bound=any(
            method_name in ADAPTIVE_POLICY_METHODS for method_name in experiment.aggregate_methods
        ),
        target_network_rpm=experiment.requests_per_minute_limit,
        rate_limit_snapshot_provider=throttle.snapshot,
    )

    manifest = finalize_family_manifest(
        {
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "experiment": experiment.name,
            "description": experiment.description,
            "phase": phase_name,
            "phase_metadata": phase_metadata(experiment, phase_name),
            "protocol": {
                "agent_count": protocol.agent_count,
                "top_p": protocol.top_p,
                "stage_a_temperature": protocol.stage_a_temperature,
                "consensus_confidence_threshold": protocol.consensus_confidence_threshold,
                "majority_confidence_threshold": protocol.majority_confidence_threshold,
                "majority_margin_threshold": protocol.majority_margin_threshold,
                "debate_rounds": protocol.debate_rounds,
                "debate_temperature": protocol.debate_temperature
                if protocol.debate_temperature is not None
                else protocol.stage_a_temperature,
                "debate_trigger_mode": protocol.debate_trigger_mode,
            },
            "controls": {name: method.__dict__ for name, method in controls.items()},
            "aggregate_methods": list(experiment.aggregate_methods),
            "max_adaptive_addon_calls": experiment.max_adaptive_addon_calls,
            "prompt_version": experiment.prompt_version,
            "stage_a_prompt_version": experiment.stage_a_prompt_version,
            "adaptive_prompt_version": experiment.adaptive_prompt_version,
            "artifact_version": ARTIFACT_VERSION,
            "global_seed": experiment.global_seed,
            "max_concurrent_requests": experiment.max_concurrent_requests,
            "requests_per_minute_limit": experiment.requests_per_minute_limit,
            "family_name": "adaptive_sparse_mad",
            "experiment_name": experiment.name,
            "phase_name": phase_name,
            "primary_model_ref": experiment.primary_model_ref,
            "resolved_model": backbone.__dict__,
            "benchmarks": [benchmark.__dict__ for benchmark in benchmarks],
            "total_planned_calls": total_calls,
            "total_planned_predictions": total_predictions,
        },
        family_name="adaptive_sparse_mad",
    )
    run_paths.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # 运行期间保留内存副本，便于结束后一次性生成指标和诊断视图。
    all_stage_a_turns: list[dict[str, object]] = []
    all_control_turns: list[dict[str, object]] = []
    all_debate_rows: list[dict[str, object]] = []
    all_router_rows: list[dict[str, object]] = []
    all_prediction_rows: list[dict[str, object]] = []

    try:
        with (
            run_paths.stage_a_turns.open("w", encoding="utf-8") as stage_a_handle,
            run_paths.control_turns.open("w", encoding="utf-8") as control_handle,
            run_paths.debate_messages.open("w", encoding="utf-8") as debate_handle,
            run_paths.router_decisions.open("w", encoding="utf-8") as router_handle,
            run_paths.predictions.open("w", encoding="utf-8") as prediction_handle,
        ):
            stage_a_writer = BufferedJsonlWriter(stage_a_handle)
            control_writer = BufferedJsonlWriter(control_handle)
            debate_writer = BufferedJsonlWriter(debate_handle)
            router_writer = BufferedJsonlWriter(router_handle)
            prediction_writer = BufferedJsonlWriter(prediction_handle)

            for benchmark in benchmarks:
                cache = cache_router.for_request_target(
                    provider=backbone.provider,
                    request_model=backbone.model_id,
                    dataset=benchmark.slug,
                )
                split_name = phase_metadata(experiment, phase_name)["split_overrides"][benchmark.slug]
                samples = select_samples(benchmark, split_name)
                run_sample_batch(
                    run_id=run_id,
                    benchmark_slug=benchmark.slug,
                    split_name=split_name,
                    samples=samples,
                    protocol=protocol,
                    controls=controls,
                    experiment=experiment,
                    backbone=backbone,
                    provider=provider,
                    cache=cache,
                    throttle=throttle,
                    on_complete=partial(
                        append_sample_result,
                        stage_a_handle=stage_a_writer,
                        control_handle=control_writer,
                        debate_handle=debate_writer,
                        router_handle=router_writer,
                        prediction_handle=prediction_writer,
                        progress=progress,
                        all_stage_a_turns=all_stage_a_turns,
                        all_control_turns=all_control_turns,
                        all_debate_rows=all_debate_rows,
                        all_router_rows=all_router_rows,
                        all_prediction_rows=all_prediction_rows,
                    ),
                )

        metrics_payload = build_metrics_payload(list(all_prediction_rows))
        router_eval_payload = build_router_eval_payload(list(all_router_rows))
        diagnostics_payload = build_policy_diagnostics(list(all_prediction_rows), router_eval_payload)
        stage_a_resolver_breakdown = build_stage_a_resolver_breakdown_payload(
            list(all_stage_a_turns),
            list(all_prediction_rows),
        )
        stage_a_error_buckets = build_stage_a_error_bucket_payload(
            list(all_stage_a_turns),
            list(all_prediction_rows),
        )
        stage_a_solver_contributions = build_stage_a_solver_contribution_payload(list(all_stage_a_turns))
        run_paths.metrics.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        run_paths.router_eval.write_text(json.dumps(router_eval_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        run_paths.policy_diagnostics.write_text(json.dumps(diagnostics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        run_paths.diagnostic_path("stage_a_resolver_breakdown.json").write_text(
            json.dumps(stage_a_resolver_breakdown, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        run_paths.diagnostic_path("stage_a_error_buckets.json").write_text(
            json.dumps(stage_a_error_buckets, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        run_paths.diagnostic_path("stage_a_solver_contributions.json").write_text(
            json.dumps(stage_a_solver_contributions, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        run_paths.run_summary.write_text(json.dumps(summarize_run(run_paths.root), ensure_ascii=False, indent=2), encoding="utf-8")
        render_report(run_paths.root)
        finalize_run_outputs(
            run_paths.root,
            validator=validate_run,
            validation_path=run_paths.validation,
        )
        progress.mark_completed()
        return run_paths.root
    finally:
        progress.close()
        provider.close()
        cache_router.close()


def refresh_stage_a_only_run_artifacts(run_dir: str | Path) -> Path:
    """使用当前代码重算已完成 run 的 Stage A 聚合、诊断和报告产物。"""
    root = Path(run_dir)
    manifest = read_json(root / "manifest.json")
    protocol_payload = dict(manifest.get("protocol") or {})
    protocol = AdaptiveSparseMadProtocolConfig(
        agent_count=int(protocol_payload.get("agent_count") or 3),
        top_p=float(protocol_payload.get("top_p") or 1.0),
        stage_a_temperature=float(protocol_payload.get("stage_a_temperature") or 0.7),
        consensus_confidence_threshold=float(protocol_payload.get("consensus_confidence_threshold") or 0.65),
        majority_confidence_threshold=float(protocol_payload.get("majority_confidence_threshold") or 0.6),
        majority_margin_threshold=float(protocol_payload.get("majority_margin_threshold") or 0.25),
        debate_rounds=int(protocol_payload.get("debate_rounds") or 1),
        debate_temperature=float(
            protocol_payload.get("debate_temperature")
            or protocol_payload.get("stage_a_temperature")
            or 0.7
        ),
        debate_trigger_mode=str(protocol_payload.get("debate_trigger_mode") or "adaptive_gate"),
    )
    model_name = str((manifest.get("resolved_model") or {}).get("name") or manifest.get("primary_model_ref") or "unknown_model")
    stage_a_rows = read_jsonl(root / "turns" / "stage_a_turns.jsonl")
    prediction_rows = read_jsonl(root / "views" / "predictions.jsonl")
    router_rows = read_jsonl(root / "turns" / "router_decisions.jsonl")

    benchmarks = [BenchmarkConfig(**benchmark_payload) for benchmark_payload in manifest.get("benchmarks", [])]
    split_overrides = dict((manifest.get("phase_metadata") or {}).get("split_overrides") or {})
    sample_lookup: dict[tuple[str, str], object] = {}
    for benchmark in benchmarks:
        split_name = str(split_overrides.get(benchmark.slug) or "")
        if not split_name:
            continue
        for sample in select_samples(benchmark, split_name):
            sample_lookup[(benchmark.slug, sample.sample_id)] = sample

    # 刷新逻辑只重放已有 turn 记录，不重新发起模型请求。
    refreshed_predictions, refreshed_router_rows = refresh_prediction_rows_for_run(
        stage_a_rows,
        prediction_rows,
        router_rows,
        sample_lookup=sample_lookup,
        protocol=protocol,
        model_name=model_name,
        prompt_version=str(manifest.get("stage_a_prompt_version") or manifest.get("prompt_version") or ""),
    )
    metrics_payload = build_metrics_payload(refreshed_predictions)
    router_eval_payload = build_router_eval_payload(refreshed_router_rows)
    diagnostics_payload = build_policy_diagnostics(refreshed_predictions, router_eval_payload)
    stage_a_resolver_breakdown = build_stage_a_resolver_breakdown_payload(stage_a_rows, refreshed_predictions)
    stage_a_error_buckets = build_stage_a_error_bucket_payload(stage_a_rows, refreshed_predictions)
    stage_a_solver_contributions = build_stage_a_solver_contribution_payload(stage_a_rows)

    write_jsonl(root / "views" / "predictions.jsonl", refreshed_predictions)
    write_jsonl(root / "turns" / "router_decisions.jsonl", refreshed_router_rows)
    write_json(root / "views" / "metrics.json", metrics_payload)
    write_json(root / "diagnostics" / "router_eval.json", router_eval_payload)
    write_json(root / "diagnostics" / "policy_diagnostics.json", diagnostics_payload)
    write_json(root / "diagnostics" / "stage_a_resolver_breakdown.json", stage_a_resolver_breakdown)
    write_json(root / "diagnostics" / "stage_a_error_buckets.json", stage_a_error_buckets)
    write_json(root / "diagnostics" / "stage_a_solver_contributions.json", stage_a_solver_contributions)
    write_json(root / "views" / "run_summary.json", summarize_run(root))
    render_report(root)
    pack_run_artifacts(root)
    validation = validate_run(root)
    write_json(root / "run_validation.json", validation)
    return root
