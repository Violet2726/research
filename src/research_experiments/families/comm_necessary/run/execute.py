"""`comm_necessary` 实验主运行链路。

本模块把 HotpotQA 通信必要性实验落成完整流程：
构造 split-context 视图、运行不同消息强度的方法、聚合答案与 supporting facts，
并导出联合指标、官方预测文件与报告产物。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
from hashlib import sha256
from pathlib import Path
import csv
import json
from typing import Any

from dotenv import load_dotenv

from research_experiments.families.comm_necessary.config import (
    CommNecessaryExperimentConfig,
    CommNecessaryProtocolConfig,
    load_benchmarks,
    load_protocol_config,
    phase_metadata,
)
from research_experiments.families.comm_necessary.dataset_views import HotpotView, build_hotpot_views, serialize_view_row
from research_experiments.families.comm_necessary.algorithms import (
    METHOD_ORDER,
    aggregate_supporting_facts,
    approximate_token_count,
    build_packet,
    gold_supporting_facts,
    majority_vote_with_counts,
    normalize_supporting_facts,
    official_prediction_payload,
    score_hotpot_prediction,
    support_facts_to_jsonable,
)
from research_experiments.families.comm_necessary.prompts import build_belief_update_messages, build_solver_messages
from research_experiments.core.foundation.artifacts import BufferedJsonlWriter
from research_experiments.core.foundation.cache import RequestCache, RequestCacheRouter, json_dump
from research_experiments.core.foundation.config import ResolvedModelConfig
from research_experiments.core.foundation.datasets import DatasetSample, load_split_ids, select_samples
from research_experiments.core.foundation.evaluation import normalize_prediction
from research_experiments.core.foundation.family_helpers import resolve_phase_split_name
from research_experiments.core.foundation.providers import OpenAICompatibleProvider, estimate_request_tokens
from research_experiments.core.foundation.rate_limits import SlidingWindowRateLimiter
from research_experiments.core.foundation.runner_common import (
    execute_cached_turn,
    prepare_run_root,
    run_indexed_batch,
)
from research_experiments.core.foundation.runtime import RunProgressTracker, build_run_id, finalize_run_outputs
from research_experiments.core.structured_outputs import (
    SCHEMA_SPLIT_CONTEXT_BELIEF,
    SCHEMA_SPLIT_CONTEXT_SOLVER,
    validate_or_recover_structured_output,
)
from research_experiments.core.foundation.workspace import default_cache_root, default_runs_root


@dataclass(frozen=True)
class RunPaths:
    """comm_necessary 运行目录下的固定产物路径。"""

    root: Path
    manifest: Path
    sample_views: Path
    stage_a_turns: Path
    message_packets: Path
    stage_b_turns: Path
    final_predictions: Path
    hotpot_predictions: Path
    metrics: Path
    diagnostics: Path
    progress: Path
    run_validation: Path
    report_markdown: Path
    paper_summary: Path


@dataclass(frozen=True)
class SampleResult:
    """单题运行产物。"""

    sample_views: list[dict[str, Any]]
    stage_a_turns: list[dict[str, Any]]
    message_packets: list[dict[str, Any]]
    stage_b_turns: list[dict[str, Any]]
    final_predictions: list[dict[str, Any]]


def run_experiment(
    experiment: CommNecessaryExperimentConfig,
    phase_name: str,
    backbone: ResolvedModelConfig,
    run_root: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> Path:
    """执行一个 comm_necessary phase，并写出完整运行目录。"""
    from research_experiments.families.comm_necessary.run.report import render_report
    from research_experiments.families.comm_necessary.run.validate import validate_run

    load_dotenv(".env.local", override=False)
    run_root = run_root or default_runs_root("comm_necessary")
    cache_root = cache_root or default_cache_root()
    protocol = load_protocol_config(experiment.protocol)
    benchmarks = load_benchmarks(experiment)
    provider = OpenAICompatibleProvider(backbone)
    cache_router = RequestCacheRouter(cache_root)
    limiter = SlidingWindowRateLimiter(
        requests_per_minute=experiment.requests_per_minute_limit,
        tokens_per_minute=experiment.tokens_per_minute_limit,
    )

    run_id = build_run_id(backbone.name)
    paths = _prepare_run_paths(run_root, experiment.name, phase_name, run_id)
    total_calls, total_predictions = _estimate_work(experiment, phase_name, benchmarks, protocol)
    progress = RunProgressTracker(paths.progress, total_calls, total_predictions)

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment.name,
        "description": experiment.description,
        "phase": phase_name,
        "phase_metadata": phase_metadata(experiment, phase_name),
        "prompt_version": experiment.prompt_version,
        "backbone": asdict(backbone),
        "benchmarks": [asdict(item) for item in benchmarks],
        "protocol": asdict(protocol),
        "methods": experiment.methods,
        "global_seed": experiment.global_seed,
        "max_concurrent_requests": experiment.max_concurrent_requests,
        "requests_per_minute_limit": experiment.requests_per_minute_limit,
        "tokens_per_minute_limit": experiment.tokens_per_minute_limit,
        "total_planned_calls": total_calls,
        "total_planned_predictions": total_predictions,
        "source_note": "HotpotQA distractor provides answer and sentence-level supporting facts; AgentsNet is reserved for later topology experiments.",
    }
    paths.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    all_views: list[dict[str, Any]] = []
    all_stage_a: list[dict[str, Any]] = []
    all_packets: list[dict[str, Any]] = []
    all_stage_b: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []

    try:
        with (
            paths.sample_views.open("w", encoding="utf-8") as views_handle,
            paths.stage_a_turns.open("w", encoding="utf-8") as stage_a_handle,
            paths.message_packets.open("w", encoding="utf-8") as packets_handle,
            paths.stage_b_turns.open("w", encoding="utf-8") as stage_b_handle,
            paths.final_predictions.open("w", encoding="utf-8") as predictions_handle,
        ):
            views_writer = BufferedJsonlWriter(views_handle)
            stage_a_writer = BufferedJsonlWriter(stage_a_handle)
            packets_writer = BufferedJsonlWriter(packets_handle)
            stage_b_writer = BufferedJsonlWriter(stage_b_handle)
            predictions_writer = BufferedJsonlWriter(predictions_handle)
            for benchmark in benchmarks:
                cache = cache_router.for_request_target(
                    provider=backbone.provider,
                    request_model=backbone.model_id,
                    dataset=benchmark.slug,
                )
                split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
                samples = select_samples(benchmark, split_name)
                results = _run_sample_batch(
                    run_id=run_id,
                    dataset=benchmark.slug,
                    split_name=split_name,
                    samples=samples,
                    protocol=protocol,
                    backbone=backbone,
                    provider=provider,
                    cache=cache,
                    limiter=limiter,
                    global_seed=experiment.global_seed,
                    prompt_version=experiment.prompt_version,
                    max_concurrent_requests=experiment.max_concurrent_requests,
                )
                for result in results:
                    for row in result.sample_views:
                        views_writer.write_row(row)
                    for row in result.stage_a_turns:
                        stage_a_writer.write_row(row)
                        progress.record_call(row, method_key="stage_name")
                    for row in result.message_packets:
                        packets_writer.write_row(row)
                    for row in result.stage_b_turns:
                        stage_b_writer.write_row(row)
                        progress.record_call(row, method_key="method_name")
                    for row in result.final_predictions:
                        predictions_writer.write_row(row)
                        progress.record_predictions(1, str(row["dataset"]), str(row["method_name"]))
                    all_views.extend(result.sample_views)
                    all_stage_a.extend(result.stage_a_turns)
                    all_packets.extend(result.message_packets)
                    all_stage_b.extend(result.stage_b_turns)
                    all_predictions.extend(result.final_predictions)

        metrics = _build_metrics(all_predictions)
        diagnostics = _build_diagnostics(all_predictions, all_views)
        paths.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        paths.diagnostics.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_hotpot_predictions(paths.hotpot_predictions, all_predictions)
        _write_paper_summary(paths.paper_summary, metrics)
        render_report(paths.root)
        finalize_run_outputs(
            paths.root,
            validator=validate_run,
            validation_path=paths.run_validation,
        )
        progress.mark_completed()
        return paths.root
    finally:
        provider.close()
        cache_router.close()


def _run_sample_batch(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    samples: list[DatasetSample],
    protocol: CommNecessaryProtocolConfig,
    backbone: ResolvedModelConfig,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    global_seed: int,
    prompt_version: str,
    max_concurrent_requests: int,
) -> list[SampleResult]:
    """样本级并发；单题内部保持 Stage A -> packet -> Stage B 顺序。"""
    worker = partial(
        _run_sample,
        run_id=run_id,
        dataset=dataset,
        split_name=split_name,
        protocol=protocol,
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        global_seed=global_seed,
        prompt_version=prompt_version,
    )
    return [
        result
        for _, result in run_indexed_batch(
            samples,
            worker=worker,
            max_concurrent_requests=max_concurrent_requests,
        )
    ]


def _run_sample(
    sample: DatasetSample,
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    protocol: CommNecessaryProtocolConfig,
    backbone: ResolvedModelConfig,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    global_seed: int,
    prompt_version: str,
) -> SampleResult:
    views = build_hotpot_views(sample)
    split_views = [view for view in views if view.agent_id in {1, 2, 3}]
    full_view = next(view for view in views if view.view_kind == "full_context")
    view_rows = [serialize_view_row(run_id=run_id, split_name=split_name, view=view) for view in views]

    stage_a_rows: list[dict[str, Any]] = []
    full_context_row = _execute_turn(
        run_id=run_id,
        dataset=dataset,
        split_name=split_name,
        sample=sample,
        view=full_view,
        stage_name="full_context",
        method_name="full_context_single",
        round_index=0,
        role="full_context_solver",
        messages=build_solver_messages(sample, full_view, prompt_version=prompt_version),
        output_mode="solver",
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        temperature=protocol.initial_temperature,
        top_p=protocol.top_p,
        max_output_tokens=protocol.max_output_tokens,
        seed=global_seed + _stable_sample_seed(sample.sample_id),
    )
    stage_a_rows.append(full_context_row)

    split_stage_a: list[dict[str, Any]] = []
    for view in split_views:
        row = _execute_turn(
            run_id=run_id,
            dataset=dataset,
            split_name=split_name,
            sample=sample,
            view=view,
            stage_name="stage_a",
            method_name="shared_split_stage_a",
            round_index=0,
            role="split_solver",
            messages=build_solver_messages(sample, view, prompt_version=prompt_version),
            output_mode="solver",
            backbone=backbone,
            provider=provider,
            cache=cache,
            limiter=limiter,
            temperature=protocol.initial_temperature,
            top_p=protocol.top_p,
            max_output_tokens=protocol.max_output_tokens,
            seed=global_seed + _stable_sample_seed(sample.sample_id) + int(view.agent_id),
        )
        split_stage_a.append(row)
    split_trace_hash = _trace_hash(split_stage_a, ["agent_id", "normalized_answer", "supporting_facts", "output_status"])
    full_trace_hash = _trace_hash([full_context_row], ["agent_id", "normalized_answer", "supporting_facts", "output_status"])
    for row in split_stage_a:
        row["stage_a_trace_hash"] = split_trace_hash
    full_context_row["stage_a_trace_hash"] = full_trace_hash
    stage_a_rows.extend(split_stage_a)

    packets = _build_message_packets(
        run_id=run_id,
        dataset=dataset,
        split_name=split_name,
        sample_id=sample.sample_id,
        stage_a_trace_hash=split_trace_hash,
        split_stage_a=split_stage_a,
        protocol=protocol,
    )
    stage_b_rows: list[dict[str, Any]] = []
    for method_name, packet_mode in [
        ("answer_only_exchange", "answer_only"),
        ("evidence_exchange", "evidence"),
        ("full_packet_exchange", "full_packet"),
    ]:
        method_packets = [packet for packet in packets if packet["method_name"] == method_name]
        stage_b_rows.extend(
            _run_belief_updates(
                run_id=run_id,
                dataset=dataset,
                split_name=split_name,
                sample=sample,
                split_views=split_views,
                split_stage_a=split_stage_a,
                packets=method_packets,
                method_name=method_name,
                packet_mode=packet_mode,
                protocol=protocol,
                backbone=backbone,
                provider=provider,
                cache=cache,
                limiter=limiter,
                global_seed=global_seed,
                prompt_version=prompt_version,
            )
        )

    prediction_rows = _build_prediction_rows(
        run_id=run_id,
        dataset=dataset,
        split_name=split_name,
        sample=sample,
        backbone=backbone,
        full_context_row=full_context_row,
        split_stage_a=split_stage_a,
        packets=packets,
        stage_b_rows=stage_b_rows,
        protocol=protocol,
    )
    return SampleResult(
        sample_views=view_rows,
        stage_a_turns=stage_a_rows,
        message_packets=packets,
        stage_b_turns=stage_b_rows,
        final_predictions=prediction_rows,
    )


def _build_message_packets(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample_id: str,
    stage_a_trace_hash: str,
    split_stage_a: list[dict[str, Any]],
    protocol: CommNecessaryProtocolConfig,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = [
        ("answer_only_exchange", "answer_only", protocol.answer_only_token_cap),
        ("evidence_exchange", "evidence", protocol.evidence_token_cap),
        ("full_packet_exchange", "full_packet", protocol.full_packet_token_cap),
    ]
    for method_name, packet_mode, token_cap in specs:
        for turn in split_stage_a:
            packet = build_packet(turn, packet_mode=packet_mode, token_cap=token_cap)
            rows.append(
                {
                    "run_id": run_id,
                    "dataset": dataset,
                    "split": split_name,
                    "sample_id": sample_id,
                    "method_name": method_name,
                    "stage_a_trace_hash": stage_a_trace_hash,
                    **packet,
                }
            )
    return rows


def _run_belief_updates(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    split_views: list[HotpotView],
    split_stage_a: list[dict[str, Any]],
    packets: list[dict[str, Any]],
    method_name: str,
    packet_mode: str,
    protocol: CommNecessaryProtocolConfig,
    backbone: ResolvedModelConfig,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    global_seed: int,
    prompt_version: str,
) -> list[dict[str, Any]]:
    stage_a_by_agent = {int(row["agent_id"]): row for row in split_stage_a}
    packet_by_agent = {int(row["agent_id"]): row for row in packets}
    rows: list[dict[str, Any]] = []
    for view in split_views:
        recipient_id = int(view.agent_id)
        peer_packets = [
            {
                "agent": f"agent_{packet['agent_id']}",
                "packet_mode": packet["packet_mode"],
                "packet_text": packet["packet_text"],
            }
            for packet in packets
            if int(packet["agent_id"]) != recipient_id
        ]
        previous = stage_a_by_agent[recipient_id]
        row = _execute_turn(
                run_id=run_id,
                dataset=dataset,
                split_name=split_name,
                sample=sample,
                view=view,
                stage_name="stage_b",
                method_name=method_name,
                round_index=1,
                role="belief_update",
                messages=build_belief_update_messages(
                    sample,
                    view,
                    previous_answer=str(previous.get("normalized_answer") or ""),
                    previous_reasoning_trace=str(previous.get("reasoning_trace") or ""),
                    previous_evidence_summary=str(previous.get("evidence_summary") or ""),
                    peer_packets=peer_packets,
                    method_name=method_name,
                    prompt_version=prompt_version,
                ),
                output_mode="belief",
                backbone=backbone,
                provider=provider,
                cache=cache,
                limiter=limiter,
                temperature=protocol.update_temperature,
                top_p=protocol.top_p,
                max_output_tokens=protocol.max_output_tokens,
                seed=global_seed + _stable_sample_seed(sample.sample_id) + recipient_id + _method_seed_offset(method_name),
                extra_fields={
                    "packet_mode": packet_mode,
                    "visible_peer_count": len(peer_packets),
                    "own_packet_text": packet_by_agent[recipient_id]["packet_text"],
                },
            )
        _apply_belief_answer_fallback(row, previous)
        rows.append(row)
    trace_hash = _trace_hash(rows, ["agent_id", "normalized_answer", "supporting_facts", "output_status", "method_name"])
    for row in rows:
        row["stage_b_trace_hash"] = trace_hash
    return rows


def _build_prediction_rows(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    backbone: ResolvedModelConfig,
    full_context_row: dict[str, Any],
    split_stage_a: list[dict[str, Any]],
    packets: list[dict[str, Any]],
    stage_b_rows: list[dict[str, Any]],
    protocol: CommNecessaryProtocolConfig,
) -> list[dict[str, Any]]:
    gold_facts = gold_supporting_facts(sample.metadata)
    split_vote, split_counts, split_consensus = majority_vote_with_counts([str(row.get("normalized_answer") or "") for row in split_stage_a])
    split_support = aggregate_supporting_facts(split_stage_a, split_vote)
    stage_a_trace_hash = split_stage_a[0].get("stage_a_trace_hash")
    baseline_scores = score_hotpot_prediction(
        predicted_answer=split_vote,
        gold_answer=sample.reference_answer,
        predicted_supporting_facts=split_support,
        gold_supporting_facts=gold_facts,
    )
    rows = [
        _prediction_row(
            run_id=run_id,
            dataset=dataset,
            split_name=split_name,
            sample=sample,
            backbone=backbone,
            method_name="full_context_single",
            method_kind="upper_reference",
            prediction=str(full_context_row.get("normalized_answer") or ""),
            raw_prediction=str(full_context_row.get("final_answer_raw") or ""),
            supporting_facts=normalize_supporting_facts(full_context_row.get("supporting_facts")),
            vote_counts={str(full_context_row.get("normalized_answer") or ""): 1},
            final_consensus=True,
            communication_tokens=0.0,
            costs=_cost([full_context_row]),
            calls_per_question=1,
            stage_a_trace_hash=full_context_row.get("stage_a_trace_hash"),
            stage_b_trace_hash_used=None,
            baseline_answer_em=baseline_scores.answer_em,
            gold_facts=gold_facts,
        ),
        _prediction_row(
            run_id=run_id,
            dataset=dataset,
            split_name=split_name,
            sample=sample,
            backbone=backbone,
            method_name="split_no_comm_mv3",
            method_kind="no_communication",
            prediction=split_vote,
            raw_prediction=split_vote,
            supporting_facts=split_support,
            vote_counts=split_counts,
            final_consensus=split_consensus,
            communication_tokens=0.0,
            costs=_cost(split_stage_a),
            calls_per_question=len(split_stage_a),
            stage_a_trace_hash=stage_a_trace_hash,
            stage_b_trace_hash_used=None,
            baseline_answer_em=baseline_scores.answer_em,
            gold_facts=gold_facts,
        ),
    ]
    for method_name in ["answer_only_exchange", "evidence_exchange", "full_packet_exchange"]:
        method_stage_b = [row for row in stage_b_rows if row["method_name"] == method_name]
        method_vote, method_counts, method_consensus = majority_vote_with_counts([str(row.get("normalized_answer") or "") for row in method_stage_b])
        method_support = aggregate_supporting_facts(method_stage_b, method_vote)
        method_packets = [packet for packet in packets if packet["method_name"] == method_name]
        rows.append(
            _prediction_row(
                run_id=run_id,
                dataset=dataset,
                split_name=split_name,
                sample=sample,
                backbone=backbone,
                method_name=method_name,
                method_kind="communication",
                prediction=method_vote,
                raw_prediction=method_vote,
                supporting_facts=method_support,
                vote_counts=method_counts,
                final_consensus=method_consensus,
                communication_tokens=_broadcast_packet_tokens(method_packets, protocol.agent_count),
                costs=_cost(split_stage_a + method_stage_b),
                calls_per_question=len(split_stage_a) + len(method_stage_b),
                stage_a_trace_hash=stage_a_trace_hash,
                stage_b_trace_hash_used=method_stage_b[0].get("stage_b_trace_hash") if method_stage_b else None,
                baseline_answer_em=baseline_scores.answer_em,
                gold_facts=gold_facts,
            )
        )
    return rows


def _apply_belief_answer_fallback(row: dict[str, Any], previous_row: dict[str, Any]) -> None:
    """Keep the prior belief when Stage B returns no grounded answer.

    Belief-update methods are defined as revisions over an existing Stage A
    answer. When the model emits a structurally valid JSON object but leaves
    `final_answer` empty, the least-assumptive interpretation is "no grounded
    revision beyond the previous belief", not a hard schema failure.
    """
    if str(row.get("output_status") or "") != "ok":
        return
    if str(row.get("normalized_answer") or ""):
        return

    previous_answer = str(previous_row.get("normalized_answer") or "")
    previous_raw_answer = str(previous_row.get("final_answer_raw") or previous_answer)
    previous_supporting_facts = support_facts_to_jsonable(
        normalize_supporting_facts(previous_row.get("supporting_facts"))
    )

    row["changed_answer"] = False
    if previous_answer:
        row["prediction"] = previous_answer
        row["normalized_answer"] = previous_answer
        row["final_answer_raw"] = previous_raw_answer
        row["supporting_facts"] = previous_supporting_facts
        row["belief_fallback"] = "kept_previous_answer_after_empty_belief_output"
        validated_output = row.get("validated_output")
        if isinstance(validated_output, dict):
            validated_output["changed_answer"] = False
            validated_output["final_answer"] = previous_raw_answer
            if not validated_output.get("supporting_facts"):
                validated_output["supporting_facts"] = previous_supporting_facts
    else:
        row["belief_fallback"] = "abstained_without_prior_answer"


def _prediction_row(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    backbone: ResolvedModelConfig,
    method_name: str,
    method_kind: str,
    prediction: str,
    raw_prediction: str,
    supporting_facts: list[tuple[str, int]],
    vote_counts: dict[str, int],
    final_consensus: bool,
    communication_tokens: float,
    costs: dict[str, float],
    calls_per_question: int,
    stage_a_trace_hash: object,
    stage_b_trace_hash_used: object,
    baseline_answer_em: float,
    gold_facts: list[tuple[str, int]],
) -> dict[str, Any]:
    scores = score_hotpot_prediction(
        predicted_answer=prediction,
        gold_answer=sample.reference_answer,
        predicted_supporting_facts=supporting_facts,
        gold_supporting_facts=gold_facts,
    )
    corrected = baseline_answer_em < 1.0 and scores.answer_em == 1.0
    harmed = baseline_answer_em == 1.0 and scores.answer_em < 1.0
    return {
        "run_id": run_id,
        "dataset": dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "model_name": backbone.name,
        "method_name": method_name,
        "method_kind": method_kind,
        "gold": sample.reference_answer,
        "gold_supporting_facts": support_facts_to_jsonable(gold_facts),
        "question_preview": sample.question[:160],
        "prediction": prediction,
        "prediction_raw": raw_prediction,
        "supporting_facts": support_facts_to_jsonable(supporting_facts),
        "vote_counts": vote_counts,
        "final_consensus": final_consensus,
        "answer_em": round(scores.answer_em, 6),
        "answer_f1": round(scores.answer_f1, 6),
        "supporting_em": round(scores.supporting_em, 6),
        "supporting_f1": round(scores.supporting_f1, 6),
        "joint_em": round(scores.joint_em, 6),
        "joint_f1": round(scores.joint_f1, 6),
        "support_title_recall": round(scores.support_title_recall, 6),
        "support_fact_recall": round(scores.support_fact_recall, 6),
        "score": round(scores.answer_em, 6),
        "communication_tokens_per_question": round(communication_tokens, 6),
        "prompt_tokens_per_question": costs["prompt_tokens"],
        "completion_tokens_per_question": costs["completion_tokens"],
        "total_tokens_per_question": costs["total_tokens"],
        "latency_ms_per_question": costs["latency_ms"],
        "calls_per_question": calls_per_question,
        "stage_a_hash": stage_a_trace_hash,
        "stage_a_trace_hash": stage_a_trace_hash,
        "stage_b_trace_hash_used": stage_b_trace_hash_used,
        "corrected_by_method": corrected,
        "harmed_by_method": harmed,
        "drift_flag": harmed,
    }


def _execute_turn(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    view: HotpotView,
    stage_name: str,
    method_name: str,
    round_index: int,
    role: str,
    messages: list[dict[str, str]],
    output_mode: str,
    backbone: ResolvedModelConfig,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    seed: int,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request_started_at = datetime.now(timezone.utc).isoformat()

    def _response_hook(payload: dict[str, Any], response_payload: dict[str, Any]) -> None:
        response_payload["request_started_at"] = request_started_at
        response_payload["estimated_request_tokens"] = estimate_request_tokens(payload)

    result = execute_cached_turn(
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        seed=seed,
        validator=lambda raw_text, provider_reasoning_text: _validate_output(
            raw_text,
            output_mode,
            provider_reasoning_text=provider_reasoning_text,
        ),
        response_hook=_response_hook,
    )

    final_answer = str(result.validated_output.get("final_answer") or "")
    supporting_facts = normalize_supporting_facts(result.validated_output.get("supporting_facts"))
    row = {
        "run_id": run_id,
        "dataset": dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "stage_name": stage_name,
        "method_name": method_name,
        "round_index": round_index,
        "agent_id": view.agent_id,
        "role": role,
        "view_kind": view.view_kind,
        "includes_full_context": view.includes_full_context,
        "prompt_hash": result.prompt_hash,
        "output_status": result.output_status,
        "prediction": normalize_prediction(dataset, final_answer) if final_answer else "",
        "normalized_answer": normalize_prediction(dataset, final_answer) if final_answer else "",
        "final_answer_raw": final_answer,
        "reasoning_trace": str(result.validated_output.get("reasoning_trace") or ""),
        "evidence_summary": str(result.validated_output.get("evidence_summary") or ""),
        "supporting_facts": support_facts_to_jsonable(supporting_facts),
        "changed_answer": bool(result.validated_output.get("changed_answer")) if "changed_answer" in result.validated_output else False,
        "confidence_raw": result.validated_output.get("confidence_raw"),
        "prompt_tokens": float(result.usage.get("prompt_tokens") or 0.0),
        "completion_tokens": float(result.usage.get("completion_tokens") or 0.0),
        "total_tokens": float(result.usage.get("total_tokens") or 0.0),
        "latency_ms": float(result.response_payload.get("latency_ms") or 0.0),
        "cache_hit": result.cache_hit,
        "request_started_at": result.response_payload.get("request_started_at"),
        "estimated_request_tokens": int(result.response_payload.get("estimated_request_tokens") or estimate_request_tokens(result.payload)),
        "request_error": result.request_error,
        "assistant_text": result.response_payload.get("assistant_text", ""),
        "provider_reasoning_text": result.response_payload.get("provider_reasoning_text", ""),
        "validated_output": result.validated_output,
    }
    if extra_fields:
        row.update(extra_fields)
    return row


def _validate_output(raw_text: str, output_mode: str, *, provider_reasoning_text: str = "") -> dict[str, Any]:
    """对模型 JSON 做宽容解析，降低小样本烟测中的格式噪声。"""
    if output_mode not in {"solver", "belief"}:
        raise ValueError(f"Unsupported output_mode: {output_mode}")
    structured_mode = SCHEMA_SPLIT_CONTEXT_SOLVER if output_mode == "solver" else SCHEMA_SPLIT_CONTEXT_BELIEF
    payload = validate_or_recover_structured_output(
        raw_text,
        structured_mode,
        provider_reasoning_text=provider_reasoning_text,
    )
    return {
        "changed_answer": payload.get("changed_answer") if isinstance(payload.get("changed_answer"), bool) else False,
        "final_answer": _textish(payload.get("final_answer")),
        "reasoning_trace": _textish(payload.get("reasoning_trace") or payload.get("reasoning")),
        "evidence_summary": _textish(payload.get("evidence_summary") or payload.get("key_evidence")),
        "supporting_facts": support_facts_to_jsonable(normalize_supporting_facts(payload.get("supporting_facts"))),
        "confidence_raw": _normalize_confidence_raw(payload.get("confidence_raw")),
    }


def _decode_json_object(raw_text: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Assistant output must contain a JSON object.")
    payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Assistant output must be a JSON object.")
    return payload


def _textish(value: object) -> str:
    if value is None or isinstance(value, bool):
        return ""
    return str(value).strip()


def _normalize_confidence_raw(value: object) -> float | str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return max(0.0, min(1.0, float(value)))
    normalized = str(value).strip()
    if not normalized:
        return None
    try:
        numeric = float(normalized.rstrip("%"))
    except ValueError:
        return None
    if normalized.endswith("%"):
        numeric = numeric / 100
    return max(0.0, min(1.0, numeric))


def _build_metrics(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """聚合题级预测成论文表格指标。"""
    summary: list[dict[str, Any]] = []
    datasets = sorted({str(row["dataset"]) for row in prediction_rows})
    for dataset in [*datasets, "overall"]:
        rows_for_dataset = prediction_rows if dataset == "overall" else [row for row in prediction_rows if row["dataset"] == dataset]
        for method in METHOD_ORDER:
            rows = [row for row in rows_for_dataset if row["method_name"] == method]
            if not rows:
                continue
            total_tokens = _mean(float(row["total_tokens_per_question"]) for row in rows)
            summary.append(
                {
                    "dataset": dataset,
                    "model_name": rows[0]["model_name"],
                    "method_name": method,
                    "method_kind": rows[0]["method_kind"],
                    "question_count": len(rows),
                    "accuracy_mean": round(_mean(float(row["answer_em"]) for row in rows), 6),
                    "answer_em_mean": round(_mean(float(row["answer_em"]) for row in rows), 6),
                    "answer_f1_mean": round(_mean(float(row["answer_f1"]) for row in rows), 6),
                    "supporting_em_mean": round(_mean(float(row["supporting_em"]) for row in rows), 6),
                    "supporting_f1_mean": round(_mean(float(row["supporting_f1"]) for row in rows), 6),
                    "joint_em_mean": round(_mean(float(row["joint_em"]) for row in rows), 6),
                    "joint_f1_mean": round(_mean(float(row["joint_f1"]) for row in rows), 6),
                    "support_title_recall_mean": round(_mean(float(row["support_title_recall"]) for row in rows), 6),
                    "support_fact_recall_mean": round(_mean(float(row["support_fact_recall"]) for row in rows), 6),
                    "communication_tokens_mean": round(_mean(float(row["communication_tokens_per_question"]) for row in rows), 6),
                    "prompt_tokens_mean": round(_mean(float(row["prompt_tokens_per_question"]) for row in rows), 6),
                    "completion_tokens_mean": round(_mean(float(row["completion_tokens_per_question"]) for row in rows), 6),
                    "total_tokens_mean": round(total_tokens, 6),
                    "latency_ms_mean": round(_mean(float(row["latency_ms_per_question"]) for row in rows), 6),
                    "calls_per_question_mean": round(_mean(float(row["calls_per_question"]) for row in rows), 6),
                    "acc_per_1k_tokens": round((_mean(float(row["answer_em"]) for row in rows) / total_tokens * 1000) if total_tokens else 0.0, 6),
                    "corrected_count": sum(1 for row in rows if row.get("corrected_by_method")),
                    "harmed_count": sum(1 for row in rows if row.get("harmed_by_method")),
                }
            )
    return {"summary": summary}


def _build_diagnostics(prediction_rows: list[dict[str, Any]], sample_views: list[dict[str, Any]]) -> dict[str, Any]:
    """构建通信必要性诊断和关键 delta。"""
    overall = {
        row["method_name"]: row
        for row in _build_metrics(prediction_rows)["summary"]
        if row.get("dataset") == "overall"
    }
    delta_pairs = [
        ("evidence_exchange", "split_no_comm_mv3"),
        ("full_packet_exchange", "answer_only_exchange"),
        ("full_context_single", "evidence_exchange"),
    ]
    deltas = []
    for left, right in delta_pairs:
        if left not in overall or right not in overall:
            continue
        left_row = overall[left]
        right_row = overall[right]
        deltas.append(
            {
                "comparison": f"{left} - {right}",
                "answer_em_delta": round(float(left_row["answer_em_mean"]) - float(right_row["answer_em_mean"]), 6),
                "supporting_f1_delta": round(float(left_row["supporting_f1_mean"]) - float(right_row["supporting_f1_mean"]), 6),
                "joint_f1_delta": round(float(left_row["joint_f1_mean"]) - float(right_row["joint_f1_mean"]), 6),
                "communication_tokens_delta": round(float(left_row["communication_tokens_mean"]) - float(right_row["communication_tokens_mean"]), 6),
            }
        )
    split_views = [row for row in sample_views if int(row.get("agent_id") or -1) in {1, 2, 3}]
    return {
        "count20_note": "Small-sample result for engineering validation and directional evidence only.",
        "key_deltas": deltas,
        "split_view_count": len(split_views),
        "full_context_view_count": sum(1 for row in sample_views if row.get("view_kind") == "full_context"),
    }


def _write_hotpot_predictions(output_dir: Path, prediction_rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for method in METHOD_ORDER:
        rows = [row for row in prediction_rows if row["method_name"] == method]
        if not rows:
            continue
        payload = official_prediction_payload(rows)
        (output_dir / f"{method}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_paper_summary(path: Path, metrics: dict[str, Any]) -> None:
    fieldnames = [
        "dataset",
        "model_name",
        "method_name",
        "answer_em_mean",
        "answer_f1_mean",
        "supporting_f1_mean",
        "joint_f1_mean",
        "communication_tokens_mean",
        "total_tokens_mean",
        "calls_per_question_mean",
        "acc_per_1k_tokens",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics.get("summary", []):
            writer.writerow({field: row.get(field) for field in fieldnames})


def _prepare_run_paths(run_root: str | Path, experiment_name: str, phase_name: str, run_id: str) -> RunPaths:
    root = prepare_run_root(run_root, experiment_name, phase_name, run_id)
    return RunPaths(
        root=root,
        manifest=root / "manifest.json",
        sample_views=root / "sample_views.jsonl",
        stage_a_turns=root / "stage_a_turns.jsonl",
        message_packets=root / "message_packets.jsonl",
        stage_b_turns=root / "stage_b_turns.jsonl",
        final_predictions=root / "final_predictions.jsonl",
        hotpot_predictions=root / "hotpot_predictions",
        metrics=root / "metrics.json",
        diagnostics=root / "diagnostics.json",
        progress=root / "progress.json",
        run_validation=root / "run_validation.json",
        report_markdown=root / "report.md",
        paper_summary=root / "paper_summary.csv",
    )


def _estimate_work(
    experiment: CommNecessaryExperimentConfig,
    phase_name: str,
    benchmarks,
    protocol: CommNecessaryProtocolConfig,
) -> tuple[int, int]:
    sample_count = 0
    for benchmark in benchmarks:
        split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
        sample_count += len(load_split_ids(benchmark.slug, split_name))
    calls_per_sample = 1 + protocol.agent_count + protocol.agent_count * 3
    return sample_count * calls_per_sample, sample_count * len(METHOD_ORDER)


def _resolve_split_name(experiment: CommNecessaryExperimentConfig, phase_name: str, benchmark_slug: str) -> str:
    return resolve_phase_split_name(experiment, phase_name, benchmark_slug)


def _cost(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "prompt_tokens": round(sum(float(row.get("prompt_tokens") or 0.0) for row in rows), 6),
        "completion_tokens": round(sum(float(row.get("completion_tokens") or 0.0) for row in rows), 6),
        "total_tokens": round(sum(float(row.get("total_tokens") or 0.0) for row in rows), 6),
        "latency_ms": round(sum(float(row.get("latency_ms") or 0.0) for row in rows), 6),
    }


def _broadcast_packet_tokens(packets: list[dict[str, Any]], agent_count: int) -> float:
    return float(sum(int(packet.get("approx_packet_tokens") or 0) for packet in packets) * max(0, agent_count - 1))


def _trace_hash(rows: list[dict[str, Any]], keys: list[str]) -> str:
    payload = [{key: row.get(key) for key in keys} for row in rows]
    return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _stable_sample_seed(sample_id: str) -> int:
    return sum(ord(char) for char in sample_id)


def _method_seed_offset(method_name: str) -> int:
    return {"answer_only_exchange": 1000, "evidence_exchange": 2000, "full_packet_exchange": 3000}.get(method_name, 0)


def _mean(values) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return sum(materialized) / len(materialized)



