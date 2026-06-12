"""多智能体实验主运行链路。

本模块把 Vanilla MAD 及其等预算控制方法组织成完整实验流程，
包括共享样本选择、setup 解析、agent turn 执行、debate 消息落盘、
题级投票聚合、成本拆分与最终报告/校验产物生成。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from research_experiments.core.config import BenchmarkConfig, resolve_model_ref
from research_experiments.core.data.datasets import select_samples
from research_experiments.core.data.evaluation import normalize_prediction
from research_experiments.core.controls.no_comm_controls import run_unified_control_batch
from research_experiments.core.execution.artifacts import BufferedJsonlWriter
from research_experiments.core.execution.cache import RequestCacheRouter
from research_experiments.core.io import read_json, read_jsonl, write_json, write_jsonl
from research_experiments.core.execution.providers import OpenAICompatibleProvider
from research_experiments.core.execution.rate_limits import RequestThrottle
from research_experiments.core.execution.runtime import RunProgressTracker, build_run_id, finalize_run_outputs
from research_experiments.core.structured_outputs import ARTIFACT_VERSION
from research_experiments.families.multi_agent.config import (
    MultiAgentExperimentConfig,
    load_control_catalog,
    load_protocol_config,
    load_roster_config,
)
from research_experiments.families.multi_agent.run.report import render_report, summarize_run
from research_experiments.families.multi_agent.run.sample import (
    _active_setups,
    _build_answer_contract_diagnostics,
    _build_control_prediction_row,
    _build_cost_breakdown,
    _build_debate_diagnostics,
    _build_metrics,
    _estimate_work,
    _execute_turn,
    _load_selected_samples,
    _resolve_split_name,
    _run_mad_setup_batch,
    _write_sample_outputs,
)
from research_experiments.families.multi_agent.run.validate import validate_run
from research_experiments.families.registry import get_family_registration
from research_experiments.family_runtime.answer_contracts import refresh_answer_contract_turn
from research_experiments.family_runtime.artifact_index import named_turn_record_paths, resolve_run_artifact_index
from research_experiments.family_runtime.comparator_impls import (
    build_shared_vanilla_mad_prediction,
    summarize_shared_vanilla_mad_turn_rows,
)
from research_experiments.family_runtime.config_helpers import load_benchmarks, phase_metadata
from research_experiments.family_runtime.layout import prepare_registered_run_layout
from research_experiments.family_runtime.manifest import finalize_family_manifest
from research_experiments.workspace.layout import default_cache_root, default_runs_root


def run_experiment(
    experiment: MultiAgentExperimentConfig,
    phase_name: str,
    backbone,
    run_root: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> Path:
    """执行一个多智能体 phase，并写出完整运行目录。"""
    load_dotenv(".env.local", override=False)
    run_root = run_root or default_runs_root("multi_agent")
    cache_root = cache_root or default_cache_root()
    benchmarks = load_benchmarks(experiment)
    phase = phase_metadata(experiment, phase_name)
    setups = _active_setups(experiment, phase_name)
    controls = load_control_catalog(experiment.control_catalog)
    matched_control_names = sorted({name for setup in setups for name in setup.matched_controls})
    provider = OpenAICompatibleProvider(backbone)
    cache_router = RequestCacheRouter(cache_root)
    throttle = RequestThrottle.for_model(
        backbone,
        max_concurrent_requests=experiment.max_concurrent_requests,
        requests_per_minute=experiment.requests_per_minute_limit,
    )
    run_id = build_run_id(backbone.name)
    run_paths = prepare_registered_run_layout('multi_agent', run_root, experiment.name, phase_name, run_id)
    total_calls, total_predictions = _estimate_work(experiment, phase_name, benchmarks, setups, matched_control_names, controls)
    progress = RunProgressTracker(
        run_paths.progress,
        total_calls,
        total_predictions,
        target_network_rpm=experiment.requests_per_minute_limit,
        rate_limit_snapshot_provider=throttle.snapshot,
    )

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "family_name": "multi_agent",
        "experiment_name": experiment.name,
        "phase_name": phase_name,
        "primary_model_ref": experiment.primary_model_ref,
        "resolved_model": asdict(backbone),
        "experiment": experiment.name,
        "description": experiment.description,
        "phase": phase_name,
        "phase_metadata": phase,
        "control_prompt_version": experiment.control_prompt_version,
        "control_answer_contract": experiment.control_answer_contract,
        "mad_prompt_version": experiment.mad_prompt_version,
        "mad_answer_contract": experiment.mad_answer_contract,
        "artifact_version": ARTIFACT_VERSION,
        "backbone": asdict(backbone),
        "benchmarks": [asdict(item) for item in benchmarks],
        "setups": [
            {
                "name": setup.name,
                "protocol": asdict(load_protocol_config(setup.protocol)),
                "roster": asdict(load_roster_config(setup.roster)),
                "matched_controls": setup.matched_controls,
            }
            for setup in setups
        ],
        "control_methods": {name: asdict(controls[name]) for name in matched_control_names},
        "total_planned_calls": total_calls,
        "total_planned_predictions": total_predictions,
    }
    manifest = finalize_family_manifest(manifest, family_name="multi_agent")
    run_paths.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    all_turns: list[dict[str, Any]] = []
    debate_messages: list[dict[str, Any]] = []
    final_predictions: list[dict[str, Any]] = []

    try:
        with (
            run_paths.agent_turns.open("w", encoding="utf-8") as turn_handle,
            run_paths.debate_messages.open("w", encoding="utf-8") as debate_handle,
            run_paths.final_predictions.open("w", encoding="utf-8") as prediction_handle,
        ):
            turn_writer = BufferedJsonlWriter(turn_handle)
            debate_writer = BufferedJsonlWriter(debate_handle)
            prediction_writer = BufferedJsonlWriter(prediction_handle)
            for benchmark in benchmarks:
                cache = cache_router.for_request_target(
                    provider=backbone.provider,
                    request_model=backbone.model_id,
                    dataset=benchmark.slug,
                )
                split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
                samples = _load_selected_samples(benchmark, split_name)

                for setup in setups:
                    protocol = load_protocol_config(setup.protocol)
                    roster = load_roster_config(setup.roster)
                    mad_results = _run_mad_setup_batch(
                        run_id=run_id,
                        benchmark_slug=benchmark.slug,
                        split_name=split_name,
                        samples=samples,
                        setup=setup,
                        protocol=protocol,
                        roster=roster,
                        backbone=backbone,
                        provider=provider,
                        cache=cache,
                        throttle=throttle,
                        global_seed=experiment.global_seed,
                        prompt_version=experiment.mad_prompt_version,
                        answer_contract=experiment.mad_answer_contract,
                        max_concurrent_requests=experiment.max_concurrent_requests,
                    )
                    _write_sample_outputs(
                        sample_results=mad_results,
                        dataset_slug=benchmark.slug,
                        progress=progress,
                        turn_handle=turn_writer,
                        debate_handle=debate_writer,
                        prediction_handle=prediction_writer,
                        all_turns=all_turns,
                        debate_messages=debate_messages,
                        final_predictions=final_predictions,
                    )

                for control_name in matched_control_names:
                    method = controls[control_name]
                    control_results = run_unified_control_batch(
                        run_id=run_id,
                        samples=samples,
                        control_name=control_name,
                        method=method,
                        benchmark_slug=benchmark.slug,
                        split_name=split_name,
                        backbone=backbone,
                        provider=provider,
                        cache=cache,
                        throttle=throttle,
                        global_seed=experiment.global_seed,
                        max_concurrent_requests=experiment.max_concurrent_requests,
                        execute_turn=_execute_turn,
                        build_prediction_row=_build_control_prediction_row,
                        prompt_version=experiment.control_prompt_version,
                    )
                    _write_sample_outputs(
                        sample_results=control_results,
                        dataset_slug=benchmark.slug,
                        progress=progress,
                        turn_handle=turn_writer,
                        debate_handle=debate_writer,
                        prediction_handle=prediction_writer,
                        all_turns=all_turns,
                        debate_messages=debate_messages,
                        final_predictions=final_predictions,
                    )

        metrics = _build_metrics(final_predictions, experiment, setups)
        diagnostics = _build_debate_diagnostics(final_predictions)
        cost_breakdown = _build_cost_breakdown(all_turns)
        answer_contract_diagnostics = _build_answer_contract_diagnostics(
            all_turns,
            dataset_order=[benchmark.slug for benchmark in benchmarks],
            method_order=[setup.name for setup in setups] + matched_control_names,
        )

        run_paths.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        run_paths.debate_diagnostics.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
        run_paths.cost_breakdown.write_text(json.dumps(cost_breakdown, ensure_ascii=False, indent=2), encoding="utf-8")
        run_paths.diagnostic_path("answer_contract_diagnostics.json").write_text(
            json.dumps(answer_contract_diagnostics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
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
        progress.close()
        provider.close()
        cache_router.close()


def refresh_run_artifacts(
    run_dir: str | Path,
    *,
    cache_root: str | Path | None = None,
) -> Path:
    del cache_root
    root = Path(run_dir)
    index = resolve_run_artifact_index(root, family_name="multi_agent")
    manifest = read_json(index.manifest_path)
    turn_paths = named_turn_record_paths(root, family_name="multi_agent")
    turn_rows = read_jsonl(turn_paths["agent_turns.jsonl"])
    prediction_rows = read_jsonl(index.prediction_records_path)

    answer_contract = str(manifest.get("mad_answer_contract") or manifest.get("control_answer_contract") or "")
    manifest["artifact_schema"] = get_family_registration("multi_agent").artifact_schema.to_manifest_payload()
    setup_map = {
        str(item["name"]): {
            "debate_rounds": int(((item.get("protocol") or {}).get("debate_rounds") or 0)),
            "agent_count": int(((item.get("roster") or {}).get("agent_count") or 1)),
        }
        for item in manifest.get("setups", [])
    }
    sample_lookup = _sample_lookup_from_manifest(manifest)
    matched_control_names = sorted({name for item in manifest.get("setups", []) for name in item.get("matched_controls", [])})

    backbone = resolve_model_ref(str(manifest.get("primary_model_ref") or ""))
    try:
        refreshed_turn_rows: list[dict[str, Any]] = []
        for row in turn_rows:
            refreshed = refresh_answer_contract_turn(
                row=row,
                sample=sample_lookup.get((str(row.get("dataset") or ""), str(row.get("sample_id") or ""))),
                backbone=backbone,
                provider=None,
                cache=None,
                throttle=None,
                answer_contract=answer_contract,
            )
            refreshed_turn_rows.append(_merge_refreshed_turn_row(row, refreshed))

        refreshed_predictions = _refresh_prediction_rows(
            prediction_rows,
            refreshed_turn_rows,
            sample_lookup=sample_lookup,
            setup_map=setup_map,
            backbone_name=str(((manifest.get("backbone") or {}).get("name")) or manifest.get("primary_model_ref") or ""),
        )

        metrics = _build_metrics(
            refreshed_predictions,
            type("RefreshExperiment", (), {"answer_contract": answer_contract})(),
            [type("Setup", (), {"name": name, "matched_controls": matched_control_names})() for name in setup_map],
        )
        diagnostics = _build_debate_diagnostics(refreshed_predictions)
        cost_breakdown = _build_cost_breakdown(refreshed_turn_rows)
        answer_contract_diagnostics = _build_answer_contract_diagnostics(
            refreshed_turn_rows,
            dataset_order=[benchmark["slug"] for benchmark in manifest.get("benchmarks", [])],
            method_order=[*setup_map.keys(), *matched_control_names],
        )

        write_jsonl(turn_paths["agent_turns.jsonl"], refreshed_turn_rows)
        write_jsonl(index.prediction_records_path, refreshed_predictions)
        write_json(index.manifest_path, manifest)
        write_json(index.metrics_view_path, metrics)
        write_json(root / "diagnostics" / "cost_breakdown.json", cost_breakdown)
        write_json(root / "diagnostics" / "debate_diagnostics.json", diagnostics)
        write_json(root / "diagnostics" / "answer_contract_diagnostics.json", answer_contract_diagnostics)
        write_json(index.run_summary_path, summarize_run(root))
        render_report(root)
        finalize_run_outputs(
            root,
            validator=validate_run,
            validation_path=index.validation_path,
        )
        return root
    finally:
        pass


def _sample_lookup_from_manifest(manifest: dict[str, Any]) -> dict[tuple[str, str], object]:
    benchmarks = [BenchmarkConfig(**payload) for payload in manifest.get("benchmarks", [])]
    phase_meta = dict(manifest.get("phase_metadata") or {})
    split_overrides = dict(phase_meta.get("split_overrides") or {})
    split_suffix = str(phase_meta.get("split_suffix") or "")
    lookup: dict[tuple[str, str], object] = {}
    for benchmark in benchmarks:
        split_name = str(split_overrides.get(benchmark.slug) or split_suffix)
        if not split_name:
            continue
        for sample in select_samples(benchmark, split_name):
            lookup[(benchmark.slug, sample.sample_id)] = sample
    return lookup


def _refresh_prediction_rows(
    prediction_rows: list[dict[str, Any]],
    turn_rows: list[dict[str, Any]],
    *,
    sample_lookup: dict[tuple[str, str], object],
    setup_map: dict[str, dict[str, int]],
    backbone_name: str,
) -> list[dict[str, Any]]:
    grouped_turns: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in turn_rows:
        grouped_turns.setdefault(
            (str(row.get("dataset") or ""), str(row.get("sample_id") or ""), str(row.get("method_name") or "")),
            [],
        ).append(row)

    refreshed: list[dict[str, Any]] = []
    for row in prediction_rows:
        method_name = str(row.get("method_name") or "")
        if method_name not in setup_map:
            refreshed.append(dict(row))
            continue
        sample = sample_lookup.get((str(row.get("dataset") or ""), str(row.get("sample_id") or "")))
        if sample is None:
            refreshed.append(dict(row))
            continue
        result = summarize_shared_vanilla_mad_turn_rows(
            turn_rows=grouped_turns.get((str(row.get("dataset") or ""), str(row.get("sample_id") or ""), method_name), []),
            dataset=str(row.get("dataset") or ""),
            gold=sample.reference_answer,
            debate_rounds=int(setup_map[method_name]["debate_rounds"]),
            agent_count=int(setup_map[method_name]["agent_count"]),
        )
        refreshed.append(
            build_shared_vanilla_mad_prediction(
                run_id=str(row.get("run_id") or ""),
                dataset=str(row.get("dataset") or ""),
                split_name=str(row.get("split") or ""),
                sample=sample,
                method_name=method_name,
                method_type="mad",
                model_name=str(row.get("model_name") or backbone_name),
                result=result,
            )
        )
    return refreshed


def _merge_refreshed_turn_row(row: dict[str, Any], refreshed) -> dict[str, Any]:
    merged = dict(row)
    final_answer = str(refreshed.validated_output.get("final_answer") or "")
    prediction = normalize_prediction(str(row.get("dataset") or ""), final_answer) if final_answer else ""
    merged.update(
        {
            "prediction": prediction,
            "normalized_answer": prediction,
            "output_status": refreshed.output_status,
            "prompt_tokens": float(refreshed.usage.get("prompt_tokens") or 0.0),
            "completion_tokens": float(refreshed.usage.get("completion_tokens") or 0.0),
            "total_tokens": float(refreshed.usage.get("total_tokens") or 0.0),
            "latency_ms": float(refreshed.response_payload.get("latency_ms") or 0.0),
            "cache_hit": refreshed.cache_hit,
            "request_error": refreshed.request_error,
            "request_status": refreshed.request_status,
            "raw_finish_reason": refreshed.raw_finish_reason,
            "answer_contract_status": refreshed.answer_contract_status,
            "answer_contract_source": refreshed.answer_contract_source,
            "answer_contract_error": refreshed.answer_contract_error,
            "answer_field_consistent": refreshed.answer_field_consistent,
            "reasoning_present": refreshed.reasoning_present,
            "json_parse_mode": refreshed.json_parse_mode,
            "request_count": refreshed.request_count,
            "cache_request_count": refreshed.cache_request_count,
            "network_request_count": refreshed.network_request_count,
            "raw_prompt_tokens": float(refreshed.usage.get("prompt_tokens") or 0.0),
            "raw_completion_tokens": float(refreshed.usage.get("completion_tokens") or 0.0),
            "raw_total_tokens": float(refreshed.usage.get("total_tokens") or 0.0),
            "raw_latency_ms": float(refreshed.response_payload.get("latency_ms") or 0.0),
            "payload": refreshed.payload,
            "assistant_text": refreshed.response_payload.get("assistant_text", ""),
            "provider_reasoning_text": refreshed.response_payload.get("provider_reasoning_text", ""),
            "validated_output": refreshed.validated_output,
        }
    )
    return merged
