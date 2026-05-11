"""CUE 实验的主运行链路。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import partial
from hashlib import sha256
from pathlib import Path
import json
from typing import Any

from dotenv import load_dotenv

from cue.config import (
    CueExperimentConfig,
    CuePolicyConfig,
    CueProtocolConfig,
    load_benchmarks,
    load_control_catalog,
    load_policies,
    load_protocol_config,
    phase_metadata,
)
from cue.logic import (
    aggregate_weighted_vote,
    aggregate_with_confidence_tiebreak,
    apply_belief_update,
    build_conflict_object,
    build_peer_packet,
    build_prompt_candidate,
    compute_utility,
    decide_policy_trigger,
    select_audit_candidate_pair,
    summarize_cue_signals,
)
from cue.prompting import build_audit_messages, build_communication_messages, build_solver_messages
from cue.reporting import render_report
from cue.validation import validate_run
from experiment_core.foundation.artifacts import BufferedJsonlWriter
from experiment_core.foundation.cache import RequestCache, RequestCacheRouter, json_dump
from experiment_core.foundation.datasets import DatasetSample, select_samples
from experiment_core.foundation.evaluation import aggregate_majority as eval_aggregate_majority
from experiment_core.foundation.evaluation import normalize_prediction, score_prediction
from experiment_core.foundation.providers import OpenAICompatibleProvider
from experiment_core.foundation.rate_limits import SlidingWindowRateLimiter
from experiment_core.foundation.runner_common import (
    execute_cached_turn,
    prepare_run_root,
    run_indexed_batch,
)
from experiment_core.foundation.runtime import RunProgressTracker, build_run_id, finalize_run_outputs
from experiment_core.controls.selective_signals import confidence_display, normalize_confidence
from experiment_core.foundation.structured_output import (
    ARTIFACT_VERSION,
    OUTPUT_MODE_CUE_AUDIT,
    OUTPUT_MODE_CUE_BELIEF_UPDATE,
    OUTPUT_MODE_CUE_SOLVER,
    validate_or_recover_structured_output,
)
from experiment_core.foundation.workspace import default_cache_root, default_runs_root
from experiment_core.foundation.methods import MethodConfig


DISPLAY_NAME_MAP = {
    "cue_v1": "cue_v1",
    "always_communicate": "always",
    "disagreement_triggered": "disagreement",
    "consensus_freeze": "consensus_freeze",
}


@dataclass(frozen=True)
class RunPaths:
    """CUE 运行目录下的固定产物路径集合。"""

    root: Path
    manifest: Path
    stage_a_turns: Path
    communication_turns: Path
    audit_turns: Path
    control_turns: Path
    policy_predictions: Path
    policy_metrics: Path
    policy_diagnostics: Path
    oracle_trigger_eval: Path
    progress: Path
    run_validation: Path
    cue_report: Path


@dataclass(frozen=True)
class SampleResult:
    """单题运行结束后沉淀的中间与最终产物。"""

    stage_a_turns: list[dict[str, Any]]
    communication_turns: list[dict[str, Any]]
    audit_turns: list[dict[str, Any]]
    control_turns: list[dict[str, Any]]
    prediction_rows: list[dict[str, Any]]


@dataclass(frozen=True)
class SharedCommunicationResult:
    """多个策略复用的共享通信结果。"""

    communication_turns: list[dict[str, Any]]
    audit_turns: list[dict[str, Any]]
    communication_trace_hash: str
    communication_prediction: str
    communication_score: float
    communication_tokens: float
    communication_latency: float
    communication_calls: int
    audit_decision: str | None
    audited_prediction: str
    audited_score: float
    audit_tokens: float
    audit_latency: float
    audit_calls: int


def _run_sample_batch(
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    samples: list[DatasetSample],
    protocol: CueProtocolConfig,
    policies: list[CuePolicyConfig],
    controls: dict[str, MethodConfig],
    experiment: CueExperimentConfig,
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    on_complete=None,
) -> None:
    worker = partial(
        _run_sample,
        run_id=run_id,
        benchmark_slug=benchmark_slug,
        split_name=split_name,
        protocol=protocol,
        policies=policies,
        controls=controls,
        experiment=experiment,
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
    )
    for _, result in run_indexed_batch(
        samples,
        worker=worker,
        max_concurrent_requests=experiment.max_concurrent_requests,
    ):
        if on_complete is not None:
            on_complete(result)


def _write_sample_result(
    result: SampleResult,
    *,
    stage_a_handle,
    communication_handle,
    audit_handle,
    control_handle,
    prediction_handle,
    progress: RunProgressTracker,
    all_stage_a_turns: list[dict[str, Any]],
    all_communication_turns: list[dict[str, Any]],
    all_audit_turns: list[dict[str, Any]],
    all_control_turns: list[dict[str, Any]],
    all_prediction_rows: list[dict[str, Any]],
) -> None:
    _write_rows(stage_a_handle, result.stage_a_turns, progress)
    _write_rows(communication_handle, result.communication_turns, progress)
    _write_rows(audit_handle, result.audit_turns, progress)
    _write_rows(control_handle, result.control_turns, progress)
    _write_predictions(prediction_handle, result.prediction_rows, progress)
    all_stage_a_turns.extend(result.stage_a_turns)
    all_communication_turns.extend(result.communication_turns)
    all_audit_turns.extend(result.audit_turns)
    all_control_turns.extend(result.control_turns)
    all_prediction_rows.extend(result.prediction_rows)


def run_experiment(
    experiment: CueExperimentConfig,
    phase_name: str,
    backbone,
    run_root: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> Path:
    """执行一个 CUE 实验 phase，并返回运行目录。"""
    load_dotenv(".env.local", override=False)
    run_root = run_root or default_runs_root("cue")
    cache_root = cache_root or default_cache_root()
    benchmarks = load_benchmarks(experiment)
    protocol = load_protocol_config(experiment.protocol)
    policies = load_policies(experiment.policy_configs)
    controls = load_control_catalog(experiment.control_catalog)
    provider = OpenAICompatibleProvider(backbone)
    cache_router = RequestCacheRouter(cache_root)
    limiter = SlidingWindowRateLimiter(
        requests_per_minute=experiment.requests_per_minute_limit,
        tokens_per_minute=experiment.tokens_per_minute_limit,
    )
    run_id = build_run_id(backbone.name)
    run_paths = _prepare_run_paths(run_root, experiment.name, phase_name, run_id)
    total_calls, total_predictions = _estimate_work(experiment, phase_name, benchmarks, protocol, controls, policies)
    progress = RunProgressTracker(run_paths.progress, total_calls, total_predictions)

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment.name,
        "description": experiment.description,
        "phase": phase_name,
        "phase_metadata": phase_metadata(experiment, phase_name),
        "protocol": asdict(protocol),
        "policies": [asdict(policy) for policy in policies],
        "controls": {name: asdict(control) for name, control in sorted(controls.items())},
        "prompt_version": experiment.prompt_version,
        "artifact_version": ARTIFACT_VERSION,
        "global_seed": experiment.global_seed,
        "max_concurrent_requests": experiment.max_concurrent_requests,
        "requests_per_minute_limit": experiment.requests_per_minute_limit,
        "tokens_per_minute_limit": experiment.tokens_per_minute_limit,
        "family_name": "cue",
        "experiment_name": experiment.name,
        "phase_name": phase_name,
        "primary_model_ref": experiment.primary_model_ref,
        "resolved_model": asdict(backbone),
        "benchmarks": [asdict(benchmark) for benchmark in benchmarks],
        "total_planned_calls": total_calls,
        "total_planned_predictions": total_predictions,
    }
    run_paths.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    all_stage_a_turns: list[dict[str, Any]] = []
    all_communication_turns: list[dict[str, Any]] = []
    all_audit_turns: list[dict[str, Any]] = []
    all_control_turns: list[dict[str, Any]] = []
    all_prediction_rows: list[dict[str, Any]] = []

    with (
        run_paths.stage_a_turns.open("w", encoding="utf-8") as stage_a_handle,
        run_paths.communication_turns.open("w", encoding="utf-8") as communication_handle,
        run_paths.audit_turns.open("w", encoding="utf-8") as audit_handle,
        run_paths.control_turns.open("w", encoding="utf-8") as control_handle,
        run_paths.policy_predictions.open("w", encoding="utf-8") as prediction_handle,
    ):
        stage_a_writer = BufferedJsonlWriter(stage_a_handle)
        communication_writer = BufferedJsonlWriter(communication_handle)
        audit_writer = BufferedJsonlWriter(audit_handle)
        control_writer = BufferedJsonlWriter(control_handle)
        prediction_writer = BufferedJsonlWriter(prediction_handle)
        for benchmark in benchmarks:
            cache = cache_router.for_request_target(
                provider=backbone.provider,
                request_model=backbone.model_id,
                dataset=benchmark.slug,
            )
            split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
            samples = select_samples(benchmark, split_name)
            print(
                f"[cue] start dataset={benchmark.slug} split={split_name} sample_count={len(samples)}",
                flush=True,
            )
            _run_sample_batch(
                run_id=run_id,
                benchmark_slug=benchmark.slug,
                split_name=split_name,
                samples=samples,
                protocol=protocol,
                policies=policies,
                controls=controls,
                experiment=experiment,
                backbone=backbone,
                provider=provider,
                cache=cache,
                limiter=limiter,
                on_complete=partial(
                    _write_sample_result,
                    stage_a_handle=stage_a_writer,
                    communication_handle=communication_writer,
                    audit_handle=audit_writer,
                    control_handle=control_writer,
                    prediction_handle=prediction_writer,
                    progress=progress,
                    all_stage_a_turns=all_stage_a_turns,
                    all_communication_turns=all_communication_turns,
                    all_audit_turns=all_audit_turns,
                    all_control_turns=all_control_turns,
                    all_prediction_rows=all_prediction_rows,
                ),
            )

    metrics_payload = _build_metrics_payload(all_prediction_rows)
    oracle_payload = _build_oracle_payload(all_prediction_rows)
    diagnostics_payload = _build_policy_diagnostics(all_prediction_rows, oracle_payload)
    run_paths.policy_metrics.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    run_paths.oracle_trigger_eval.write_text(json.dumps(oracle_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    run_paths.policy_diagnostics.write_text(json.dumps(diagnostics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
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


def _run_sample(
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    sample: DatasetSample,
    protocol: CueProtocolConfig,
    policies: list[CuePolicyConfig],
    controls: dict[str, MethodConfig],
    experiment: CueExperimentConfig,
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
) -> SampleResult:
    question_preview = _question_preview(sample.question)
    stage_a_specs = [
        {
            "run_id": run_id,
            "dataset": benchmark_slug,
            "split_name": split_name,
            "sample": sample,
            "stage_name": "stage_a",
            "method_name": "shared_stage_a",
            "role": "solver",
            "round_index": 0,
            "agent_id": agent_id,
            "visible_peer_count": 0,
            "messages": build_solver_messages(sample, agent_id, prompt_version=experiment.prompt_version),
            "backbone": backbone,
            "provider": provider,
            "cache": cache,
            "limiter": limiter,
            "temperature": protocol.initial_temperature,
            "top_p": protocol.top_p,
            "max_output_tokens": protocol.max_output_tokens,
            "seed": experiment.global_seed + agent_id,
            "question_preview": question_preview,
            "output_mode": OUTPUT_MODE_CUE_SOLVER,
        }
        for agent_id in range(1, protocol.agent_count + 1)
    ]
    stage_a_turns = _execute_turn_batch(stage_a_specs, max_workers=min(protocol.agent_count, experiment.max_concurrent_requests))
    stage_a_trace_hash = _trace_hash(stage_a_turns)
    for row in stage_a_turns:
        row["stage_trace_hash"] = stage_a_trace_hash

    stage_a_answers = [row["normalized_answer"] for row in stage_a_turns if row.get("normalized_answer")]
    stage_a_vote_norm, vote_counts = eval_aggregate_majority(stage_a_answers)
    stage_a_vote = stage_a_vote_norm
    stage_a_score = score_prediction(benchmark_slug, stage_a_vote, sample.reference_answer)
    signals = summarize_cue_signals(stage_a_turns, protocol.message_token_cap)
    conflict_object = build_conflict_object(stage_a_turns, signals, protocol.message_token_cap)

    prediction_rows: list[dict[str, Any]] = []
    communication_turns: list[dict[str, Any]] = []
    audit_turns: list[dict[str, Any]] = []
    policy_decisions: list[dict[str, Any]] = []

    for policy in policies:
        utility = compute_utility(signals, conflict_object, policy)
        triggered, decision_reason = decide_policy_trigger(policy, signals, utility)
        policy_decisions.append(
            {
                "policy": policy,
                "utility": utility,
                "triggered": triggered,
                "decision_reason": decision_reason,
            }
        )

    triggered_policies = [item for item in policy_decisions if item["triggered"]]
    shared_result: SharedCommunicationResult | None = None
    if triggered_policies:
        shared_result = _run_shared_communication(
            run_id=run_id,
            benchmark_slug=benchmark_slug,
            split_name=split_name,
            sample=sample,
            protocol=protocol,
            experiment=experiment,
            backbone=backbone,
            provider=provider,
            cache=cache,
            limiter=limiter,
            question_preview=question_preview,
            stage_a_turns=stage_a_turns,
            stage_a_vote=stage_a_vote,
            conflict_object=conflict_object,
            run_audit=any(bool(item["policy"].enable_audit) for item in triggered_policies),
        )
        communication_turns.extend(shared_result.communication_turns)
        audit_turns.extend(shared_result.audit_turns)

    for item in policy_decisions:
        policy = item["policy"]
        utility = item["utility"]
        triggered = item["triggered"]
        decision_reason = item["decision_reason"]
        final_prediction = stage_a_vote
        final_score = stage_a_score
        communication_tokens = 0.0
        communication_latency = 0.0
        calls_per_question = float(protocol.agent_count)
        audit_decision = None
        communication_trace_hash = None
        if triggered:
            assert shared_result is not None
            communication_trace_hash = shared_result.communication_trace_hash
            if bool(policy.enable_audit):
                final_prediction = shared_result.audited_prediction
                final_score = shared_result.audited_score
                communication_tokens = shared_result.communication_tokens + shared_result.audit_tokens
                communication_latency = shared_result.communication_latency + shared_result.audit_latency
                calls_per_question = float(protocol.agent_count + shared_result.communication_calls + shared_result.audit_calls)
                audit_decision = shared_result.audit_decision
                extra_turns = shared_result.communication_turns + shared_result.audit_turns
            else:
                final_prediction = shared_result.communication_prediction
                final_score = shared_result.communication_score
                communication_tokens = shared_result.communication_tokens
                communication_latency = shared_result.communication_latency
                calls_per_question = float(protocol.agent_count + shared_result.communication_calls)
                extra_turns = shared_result.communication_turns
        else:
            extra_turns = []
        prompt_tokens = sum(float(row["prompt_tokens"]) for row in stage_a_turns + extra_turns)
        completion_tokens = sum(float(row["completion_tokens"]) for row in stage_a_turns + extra_turns)
        total_tokens = sum(float(row["total_tokens"]) for row in stage_a_turns + extra_turns)
        latency_ms = sum(float(row["latency_ms"]) for row in stage_a_turns + extra_turns)
        prediction_rows.append(
            {
                "run_id": run_id,
                "dataset": benchmark_slug,
                "split": split_name,
                "sample_id": sample.sample_id,
                "question_preview": question_preview,
                "method_name": policy.policy_name,
                "display_name": DISPLAY_NAME_MAP.get(policy.policy_name, policy.policy_name),
                "method_kind": "policy",
                "method_family": "cue",
                "model_name": backbone.name,
                "prediction": final_prediction,
                "gold": sample.reference_answer,
                "score": final_score,
                "triggered": triggered,
                "early_exit": not triggered,
                "fail_open_applied": False,
                "decision_reason": decision_reason,
                "initial_disagreement": signals["initial_disagreement"],
                "answer_entropy": signals["answer_entropy"],
                "mean_confidence": signals["mean_confidence"],
                "confidence_spread": signals["confidence_spread"],
                "claim_conflict_rate": signals["claim_conflict_rate"],
                "evidence_gap": signals["evidence_gap"],
                "fragile_consensus": signals["fragile_consensus"],
                "format_conflict_risk": signals["format_conflict_risk"],
                "majority_pressure_risk": signals["majority_pressure_risk"],
                "any_invalid_confidence": signals["any_invalid_confidence"],
                "stage_a_trace_hash": stage_a_trace_hash,
                "communication_trace_hash": communication_trace_hash,
                "communication_trace_hash_used": communication_trace_hash if triggered else None,
                "stage_a_prediction": stage_a_vote,
                "stage_a_score": stage_a_score,
                "conflict_type": conflict_object.conflict_type,
                "message_type": conflict_object.message_type,
                "correction_potential": utility.correction_potential,
                "resolvability": utility.resolvability,
                "collapse_risk": utility.collapse_risk,
                "normalized_cost": utility.normalized_cost,
                "utility_score": utility.utility,
                "audit_decision": audit_decision,
                "prompt_tokens_per_question": prompt_tokens,
                "completion_tokens_per_question": completion_tokens,
                "total_tokens_per_question": total_tokens,
                "communication_tokens_per_question": communication_tokens,
                "latency_ms_per_question": latency_ms,
                "communication_latency_ms_per_question": communication_latency,
                "calls_per_question": calls_per_question,
                "stage_a_tokens_per_question": sum(float(row["total_tokens"]) for row in stage_a_turns),
                "vote_counts": vote_counts,
                "oracle_positive": None,
            }
        )

    prediction_rows.append(
        {
            "run_id": run_id,
            "dataset": benchmark_slug,
            "split": split_name,
            "sample_id": sample.sample_id,
            "question_preview": question_preview,
            "method_name": "mv_3",
            "display_name": "mv_3",
            "method_kind": "control",
            "method_family": "shared_majority_vote",
            "model_name": backbone.name,
            "prediction": stage_a_vote,
            "gold": sample.reference_answer,
            "score": stage_a_score,
            "triggered": False,
            "early_exit": True,
            "fail_open_applied": False,
            "decision_reason": "shared_stage_a_vote",
            "initial_disagreement": signals["initial_disagreement"],
            "answer_entropy": signals["answer_entropy"],
            "mean_confidence": signals["mean_confidence"],
            "confidence_spread": signals["confidence_spread"],
            "claim_conflict_rate": signals["claim_conflict_rate"],
            "evidence_gap": signals["evidence_gap"],
            "fragile_consensus": signals["fragile_consensus"],
            "format_conflict_risk": signals["format_conflict_risk"],
            "majority_pressure_risk": signals["majority_pressure_risk"],
            "any_invalid_confidence": signals["any_invalid_confidence"],
            "stage_a_trace_hash": stage_a_trace_hash,
            "communication_trace_hash": None,
            "communication_trace_hash_used": None,
            "stage_a_prediction": stage_a_vote,
            "stage_a_score": stage_a_score,
            "conflict_type": conflict_object.conflict_type,
            "message_type": conflict_object.message_type,
            "correction_potential": None,
            "resolvability": None,
            "collapse_risk": None,
            "normalized_cost": None,
            "utility_score": None,
            "audit_decision": None,
            "prompt_tokens_per_question": sum(float(row["prompt_tokens"]) for row in stage_a_turns),
            "completion_tokens_per_question": sum(float(row["completion_tokens"]) for row in stage_a_turns),
            "total_tokens_per_question": sum(float(row["total_tokens"]) for row in stage_a_turns),
            "communication_tokens_per_question": 0.0,
            "latency_ms_per_question": sum(float(row["latency_ms"]) for row in stage_a_turns),
            "communication_latency_ms_per_question": 0.0,
            "calls_per_question": float(protocol.agent_count),
            "stage_a_tokens_per_question": sum(float(row["total_tokens"]) for row in stage_a_turns),
            "vote_counts": vote_counts,
            "oracle_positive": None,
        }
    )

    control_turns: list[dict[str, Any]] = []
    for control_name, control in sorted(controls.items()):
        if control.family == "shared_majority_vote":
            continue
        control_result = _run_control_method(
            run_id=run_id,
            benchmark_slug=benchmark_slug,
            split_name=split_name,
            sample=sample,
            control_name=control_name,
            control=control,
            experiment=experiment,
            backbone=backbone,
            provider=provider,
            cache=cache,
            limiter=limiter,
            question_preview=question_preview,
        )
        control_turns.extend(control_result["turn_rows"])
        prediction_rows.append(control_result["prediction_row"])

    return SampleResult(
        stage_a_turns=stage_a_turns,
        communication_turns=communication_turns,
        audit_turns=audit_turns,
        control_turns=control_turns,
        prediction_rows=prediction_rows,
    )


def _run_shared_communication(
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    sample: DatasetSample,
    protocol: CueProtocolConfig,
    experiment: CueExperimentConfig,
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    question_preview: str,
    stage_a_turns: list[dict[str, Any]],
    stage_a_vote: str,
    conflict_object,
    run_audit: bool,
) -> SharedCommunicationResult:
    communication_specs: list[dict[str, Any]] = []
    for stage_a_row in stage_a_turns:
        peer_packets = []
        for peer_row in stage_a_turns:
            if peer_row["agent_id"] == stage_a_row["agent_id"]:
                continue
            packet = build_peer_packet(peer_row, conflict_object.message_type, protocol.message_token_cap)
            peer_packets.append(
                {
                    "agent": f"agent_{peer_row['agent_id']}",
                    "packet_text": packet["packet_text"],
                }
            )
        previous_packet = {
            "final_answer": stage_a_row.get("final_answer"),
            "confidence": stage_a_row.get("confidence_value"),
            "reasoning_sketch": stage_a_row.get("reasoning_sketch"),
            "uncertain_point": stage_a_row.get("uncertain_point"),
            "top_claims": stage_a_row.get("top_claims"),
            "evidence_items": stage_a_row.get("evidence_items"),
            "counter_answer": stage_a_row.get("counter_answer"),
        }
        messages = build_communication_messages(
            sample=sample,
            agent_id=int(stage_a_row["agent_id"]),
            round_index=1,
            previous_packet=previous_packet,
            peer_packets=peer_packets,
            conflict_object=asdict(conflict_object),
            message_type=conflict_object.message_type,
            prompt_version=experiment.prompt_version,
        )
        communication_specs.append(
            {
                "run_id": run_id,
                "dataset": benchmark_slug,
                "split_name": split_name,
                "sample": sample,
                "stage_name": "communication",
                "method_name": "shared_communication",
                "role": "belief_update",
                "round_index": 1,
                "agent_id": int(stage_a_row["agent_id"]),
                "visible_peer_count": len(peer_packets),
                "messages": messages,
                "backbone": backbone,
                "provider": provider,
                "cache": cache,
                "limiter": limiter,
                "temperature": protocol.debate_temperature,
                "top_p": protocol.top_p,
                "max_output_tokens": protocol.max_output_tokens,
                "seed": experiment.global_seed + 100 + int(stage_a_row["agent_id"]),
                "question_preview": question_preview,
                "output_mode": OUTPUT_MODE_CUE_BELIEF_UPDATE,
            }
        )
    communication_turns = _execute_turn_batch(
        communication_specs,
        max_workers=min(protocol.agent_count, experiment.max_concurrent_requests),
    )
    communication_trace_hash = _trace_hash(communication_turns)
    for row in communication_turns:
        row["stage_trace_hash"] = communication_trace_hash
    revised_candidates = [
        apply_belief_update(stage_a_row=stage_a_row, belief_row=belief_row, conflict_object=conflict_object)
        for stage_a_row, belief_row in zip(stage_a_turns, communication_turns, strict=False)
    ]
    winner_norm, _ = aggregate_weighted_vote(revised_candidates)
    if not winner_norm:
        winner_norm, _ = aggregate_with_confidence_tiebreak(revised_candidates)
    communication_prediction = winner_norm or stage_a_vote
    audit_turns: list[dict[str, Any]] = []
    audit_decision = None
    audited_prediction = communication_prediction
    if run_audit:
        pair = select_audit_candidate_pair(revised_candidates)
        if not pair["skipped"]:
            audit_row = _execute_turn(
                run_id=run_id,
                dataset=benchmark_slug,
                split_name=split_name,
                sample=sample,
                stage_name="audit",
                method_name="shared_audit",
                role="auditor",
                round_index=1,
                agent_id=0,
                visible_peer_count=2,
                messages=build_audit_messages(
                    sample=sample,
                    candidate_a=build_prompt_candidate(pair["candidate_a"]),
                    candidate_b=build_prompt_candidate(pair["candidate_b"]),
                    conflict_object=asdict(conflict_object),
                    prompt_version=experiment.prompt_version,
                ),
                backbone=backbone,
                provider=provider,
                cache=cache,
                limiter=limiter,
                temperature=0.0,
                top_p=1.0,
                max_output_tokens=protocol.audit_token_cap,
                seed=experiment.global_seed + 999,
                question_preview=question_preview,
                output_mode=OUTPUT_MODE_CUE_AUDIT,
            )
            audit_turns.append(audit_row)
            if audit_row["output_status"] == "ok":
                audit_decision = str(audit_row["validated_output"].get("decision"))
                if audit_decision == "resolve_for_a":
                    audited_prediction = str(pair["candidate_a"].get("final_answer") or pair["candidate_a"].get("normalized_answer") or audited_prediction)
                elif audit_decision == "resolve_for_b":
                    audited_prediction = str(pair["candidate_b"].get("final_answer") or pair["candidate_b"].get("normalized_answer") or audited_prediction)
    communication_score = score_prediction(benchmark_slug, communication_prediction, sample.reference_answer)
    audited_score = score_prediction(benchmark_slug, audited_prediction, sample.reference_answer)
    communication_tokens = sum(float(row["total_tokens"]) for row in communication_turns)
    audit_tokens = sum(float(row["total_tokens"]) for row in audit_turns)
    communication_latency = sum(float(row["latency_ms"]) for row in communication_turns)
    audit_latency = sum(float(row["latency_ms"]) for row in audit_turns)
    return SharedCommunicationResult(
        communication_turns=communication_turns,
        audit_turns=audit_turns,
        communication_trace_hash=communication_trace_hash,
        communication_prediction=communication_prediction,
        communication_score=communication_score,
        communication_tokens=communication_tokens,
        communication_latency=communication_latency,
        communication_calls=len(communication_turns),
        audit_decision=audit_decision,
        audited_prediction=audited_prediction,
        audited_score=audited_score,
        audit_tokens=audit_tokens,
        audit_latency=audit_latency,
        audit_calls=len(audit_turns),
    )


def _run_control_method(
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    sample: DatasetSample,
    control_name: str,
    control: MethodConfig,
    experiment: CueExperimentConfig,
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    question_preview: str,
) -> dict[str, Any]:
    turn_specs = [
        {
            "run_id": run_id,
            "dataset": benchmark_slug,
            "split_name": split_name,
            "sample": sample,
            "stage_name": "control",
            "method_name": control_name,
            "role": "control",
            "round_index": 0,
            "agent_id": replicate_id + 1,
            "visible_peer_count": 0,
            "messages": build_solver_messages(sample, replicate_id + 1, prompt_version=experiment.prompt_version),
            "backbone": backbone,
            "provider": provider,
            "cache": cache,
            "limiter": limiter,
            "temperature": control.temperature,
            "top_p": control.top_p,
            "max_output_tokens": control.max_output_tokens,
            "seed": experiment.global_seed + 1000 + replicate_id,
            "question_preview": question_preview,
            "output_mode": OUTPUT_MODE_CUE_SOLVER,
        }
        for replicate_id in range(control.budget_calls)
    ]
    turn_rows = _execute_turn_batch(
        turn_specs,
        max_workers=min(max(1, control.budget_calls), experiment.max_concurrent_requests),
    )
    trace_hash = _trace_hash(turn_rows)
    for row in turn_rows:
        row["stage_trace_hash"] = trace_hash
    answers = [row["normalized_answer"] for row in turn_rows if row.get("normalized_answer")]
    prediction, vote_counts = eval_aggregate_majority(answers)
    score = score_prediction(benchmark_slug, prediction, sample.reference_answer)
    prompt_tokens = sum(float(row["prompt_tokens"]) for row in turn_rows)
    completion_tokens = sum(float(row["completion_tokens"]) for row in turn_rows)
    total_tokens = sum(float(row["total_tokens"]) for row in turn_rows)
    latency_ms = sum(float(row["latency_ms"]) for row in turn_rows)
    return {
        "turn_rows": turn_rows,
        "prediction_row": {
            "run_id": run_id,
            "dataset": benchmark_slug,
            "split": split_name,
            "sample_id": sample.sample_id,
            "question_preview": question_preview,
            "method_name": control_name,
            "display_name": control_name,
            "method_kind": "control",
            "method_family": control.family,
            "model_name": backbone.name,
            "prediction": prediction,
            "gold": sample.reference_answer,
            "score": score,
            "triggered": False,
            "early_exit": False,
            "fail_open_applied": False,
            "decision_reason": "independent_control",
            "initial_disagreement": None,
            "answer_entropy": None,
            "mean_confidence": None,
            "confidence_spread": None,
            "claim_conflict_rate": None,
            "evidence_gap": None,
            "fragile_consensus": None,
            "format_conflict_risk": None,
            "majority_pressure_risk": None,
            "any_invalid_confidence": None,
            "stage_a_trace_hash": None,
            "communication_trace_hash": None,
            "communication_trace_hash_used": None,
            "stage_a_prediction": None,
            "stage_a_score": None,
            "conflict_type": None,
            "message_type": None,
            "correction_potential": None,
            "resolvability": None,
            "collapse_risk": None,
            "normalized_cost": None,
            "utility_score": None,
            "audit_decision": None,
            "prompt_tokens_per_question": prompt_tokens,
            "completion_tokens_per_question": completion_tokens,
            "total_tokens_per_question": total_tokens,
            "communication_tokens_per_question": 0.0,
            "latency_ms_per_question": latency_ms,
            "communication_latency_ms_per_question": 0.0,
            "calls_per_question": float(control.budget_calls),
            "stage_a_tokens_per_question": 0.0,
            "vote_counts": vote_counts,
            "oracle_positive": None,
        },
    }


def _execute_turn(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    stage_name: str,
    method_name: str,
    role: str,
    round_index: int,
    agent_id: int,
    visible_peer_count: int,
    messages: list[dict[str, str]],
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    seed: int,
    question_preview: str,
    output_mode: str,
) -> dict[str, Any]:
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
        output_mode=output_mode,
    )
    if output_mode == OUTPUT_MODE_CUE_SOLVER:
        final_answer = str(result.validated_output.get("final_answer") or "")
    elif output_mode == OUTPUT_MODE_CUE_BELIEF_UPDATE:
        final_answer = str(result.validated_output.get("new_answer") or "")
    elif output_mode == OUTPUT_MODE_CUE_AUDIT:
        final_answer = str(result.validated_output.get("verified_answer") or "")
    else:
        final_answer = ""

    confidence_raw = result.validated_output.get("confidence") if output_mode == OUTPUT_MODE_CUE_SOLVER and result.validated_output else None
    confidence_value, confidence_valid, confidence_source = normalize_confidence(confidence_raw)
    top_claims = result.validated_output.get("top_claims", []) if output_mode == OUTPUT_MODE_CUE_SOLVER and result.validated_output else []
    evidence_items = result.validated_output.get("evidence_items", []) if output_mode == OUTPUT_MODE_CUE_SOLVER and result.validated_output else []
    reasoning_sketch = result.validated_output.get("reasoning_sketch") if output_mode == OUTPUT_MODE_CUE_SOLVER and result.validated_output else None
    uncertain_point = result.validated_output.get("uncertain_point") if output_mode == OUTPUT_MODE_CUE_SOLVER and result.validated_output else None
    counter_answer = result.validated_output.get("counter_answer") if output_mode == OUTPUT_MODE_CUE_SOLVER and result.validated_output else None
    return {
        "run_id": run_id,
        "dataset": dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "question_preview": question_preview,
        "stage_name": stage_name,
        "method_name": method_name,
        "role": role,
        "round_index": round_index,
        "agent_id": agent_id,
        "visible_peer_count": visible_peer_count,
        "prompt_hash": result.prompt_hash,
        "prediction": normalize_prediction(dataset, final_answer) if final_answer else "",
        "normalized_answer": normalize_prediction(dataset, final_answer) if final_answer else "",
        "final_answer": final_answer,
        "output_status": result.output_status,
        "prompt_tokens": float(result.usage.get("prompt_tokens") or 0.0),
        "completion_tokens": float(result.usage.get("completion_tokens") or 0.0),
        "total_tokens": float(result.usage.get("total_tokens") or 0.0),
        "latency_ms": float(result.response_payload.get("latency_ms") or 0.0),
        "cache_hit": result.cache_hit,
        "request_error": result.request_error,
        "payload": result.payload,
        "assistant_text": result.response_payload.get("assistant_text", ""),
        "provider_reasoning_text": result.response_payload.get("provider_reasoning_text", ""),
        "validated_output": result.validated_output,
        "confidence_raw": confidence_raw,
        "confidence_raw_display": confidence_display(confidence_raw),
        "confidence_value": confidence_value,
        "confidence_valid": confidence_valid,
        "confidence_source": confidence_source,
        "reasoning_sketch": reasoning_sketch,
        "uncertain_point": uncertain_point,
        "top_claims": top_claims,
        "evidence_items": evidence_items,
        "counter_answer": counter_answer,
    }


def _build_metrics_payload(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
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
                    "model_name": model_name,
                    "method_name": method_name,
                    "display_name": rows[0]["display_name"],
                    "method_kind": rows[0]["method_kind"],
                    "method_family": rows[0]["method_family"],
                    "question_count": len(rows),
                    "accuracy_mean": _mean(float(item["score"]) for item in rows),
                    "prompt_tokens_mean": _mean(float(item["prompt_tokens_per_question"]) for item in rows),
                    "completion_tokens_mean": _mean(float(item["completion_tokens_per_question"]) for item in rows),
                    "total_tokens_mean": total_tokens_mean,
                    "communication_tokens_mean": _mean(float(item["communication_tokens_per_question"]) for item in rows),
                    "latency_ms_mean": _mean(float(item["latency_ms_per_question"]) for item in rows),
                    "calls_per_question_mean": _mean(float(item["calls_per_question"]) for item in rows),
                    "acc_per_1k_tokens": round(_mean(float(item["score"]) for item in rows) / total_tokens_mean * 1000, 6) if total_tokens_mean else 0.0,
                    "trigger_rate": _mean(1.0 if item.get("triggered") else 0.0 for item in rows) if rows[0]["method_kind"] == "policy" else 0.0,
                    "early_exit_rate": _mean(1.0 if item.get("early_exit") else 0.0 for item in rows) if rows[0]["method_kind"] == "policy" else 0.0,
                }
            )
    return {"summary": summary}


def _build_oracle_payload(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    sample_groups: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
    for row in prediction_rows:
        key = (row["dataset"], row["sample_id"], row["model_name"])
        sample_groups.setdefault(key, {})[row["method_name"]] = row
    sample_rows: list[dict[str, Any]] = []
    policy_names = sorted({row["method_name"] for row in prediction_rows if row["method_kind"] == "policy"})
    for (dataset, sample_id, model_name), methods in sorted(sample_groups.items()):
        if "mv_3" not in methods or "always_communicate" not in methods:
            continue
        mv_3_row = methods["mv_3"]
        always_row = methods["always_communicate"]
        helpful = float(always_row["score"]) > float(mv_3_row["score"])
        harmful = float(always_row["score"]) < float(mv_3_row["score"])
        oracle_label = "helpful" if helpful else "harmful" if harmful else "neutral"
        payload = {
            "dataset": dataset,
            "sample_id": sample_id,
            "model_name": model_name,
            "question_preview": mv_3_row["question_preview"],
            "mv_3_score": mv_3_row["score"],
            "always_score": always_row["score"],
            "beneficial_communication": helpful,
            "oracle_label": oracle_label,
            "policies": {},
        }
        for policy_name in policy_names:
            policy_row = methods.get(policy_name)
            if policy_row is None:
                continue
            payload["policies"][policy_name] = {
                "triggered": bool(policy_row["triggered"]),
                "score": float(policy_row["score"]),
                "prediction": policy_row["prediction"],
                "decision_reason": policy_row["decision_reason"],
                "utility_score": policy_row.get("utility_score"),
            }
        sample_rows.append(payload)

    summary_rows: list[dict[str, Any]] = []
    for dataset in sorted({row["dataset"] for row in sample_rows} | {"overall"}):
        rows_for_dataset = sample_rows if dataset == "overall" else [row for row in sample_rows if row["dataset"] == dataset]
        total = len(rows_for_dataset)
        helpful_count = sum(1 for row in rows_for_dataset if row["oracle_label"] == "helpful")
        harmful_count = sum(1 for row in rows_for_dataset if row["oracle_label"] == "harmful")
        neutral_count = sum(1 for row in rows_for_dataset if row["oracle_label"] == "neutral")
        for policy_name in policy_names:
            triggered_count = sum(1 for row in rows_for_dataset if row["policies"].get(policy_name, {}).get("triggered"))
            helpful_trigger_count = sum(1 for row in rows_for_dataset if row["oracle_label"] == "helpful" and row["policies"].get(policy_name, {}).get("triggered"))
            harmful_trigger_count = sum(1 for row in rows_for_dataset if row["oracle_label"] == "harmful" and row["policies"].get(policy_name, {}).get("triggered"))
            neutral_trigger_count = sum(1 for row in rows_for_dataset if row["oracle_label"] == "neutral" and row["policies"].get(policy_name, {}).get("triggered"))
            missed_positive_count = sum(1 for row in rows_for_dataset if row["oracle_label"] == "helpful" and not row["policies"].get(policy_name, {}).get("triggered"))
            summary_rows.append(
                {
                    "dataset": dataset,
                    "policy_name": policy_name,
                    "display_name": DISPLAY_NAME_MAP.get(policy_name, policy_name),
                    "question_count": total,
                    "helpful_count": helpful_count,
                    "harmful_count": harmful_count,
                    "neutral_count": neutral_count,
                    "trigger_count": triggered_count,
                    "helpful_trigger_count": helpful_trigger_count,
                    "harmful_trigger_count": harmful_trigger_count,
                    "neutral_trigger_count": neutral_trigger_count,
                    "precision": _ratio(helpful_trigger_count, triggered_count),
                    "recall": _ratio(helpful_trigger_count, helpful_count),
                    "false_trigger_rate": _ratio(harmful_trigger_count + neutral_trigger_count, total),
                    "missed_beneficial_comm_rate": _ratio(missed_positive_count, helpful_count),
                }
            )
    return {"sample_rows": sample_rows, "summary_rows": summary_rows}


def _build_policy_diagnostics(prediction_rows: list[dict[str, Any]], oracle_payload: dict[str, Any]) -> dict[str, Any]:
    metric_lookup = {(row["dataset"], row["method_name"]): row for row in _build_metrics_payload(prediction_rows)["summary"]}
    policy_rows: list[dict[str, Any]] = []
    for row in oracle_payload["summary_rows"]:
        metric_row = metric_lookup.get((row["dataset"], row["policy_name"]))
        if metric_row is None:
            continue
        policy_rows.append(
            {
                "dataset": row["dataset"],
                "policy_name": row["policy_name"],
                "display_name": row["display_name"],
                "question_count": row["question_count"],
                "trigger_rate": metric_row["trigger_rate"],
                "early_exit_rate": metric_row["early_exit_rate"],
                "precision": row["precision"],
                "recall": row["recall"],
                "false_trigger_rate": row["false_trigger_rate"],
                "missed_beneficial_comm_rate": row["missed_beneficial_comm_rate"],
                "accuracy_mean": metric_row["accuracy_mean"],
                "communication_tokens_mean": metric_row["communication_tokens_mean"],
                "total_tokens_mean": metric_row["total_tokens_mean"],
                "acc_per_1k_tokens": metric_row["acc_per_1k_tokens"],
            }
        )
    recommendation = _select_next_default_policy(metric_lookup)
    return {"policy_rows": policy_rows, "recommended_next_default_policy": recommendation}


def _select_next_default_policy(metric_lookup: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    always_row = metric_lookup.get(("overall", "always_communicate"))
    cue_row = metric_lookup.get(("overall", "cue_v1"))
    if always_row is None or cue_row is None:
        return {"selected_policy": "disagreement_triggered", "reason": "missing_overall_rows"}
    accuracy_drop = float(always_row["accuracy_mean"]) - float(cue_row["accuracy_mean"])
    token_drop_ratio = 1 - float(cue_row["total_tokens_mean"]) / float(always_row["total_tokens_mean"]) if float(always_row["total_tokens_mean"]) else 0.0
    selected = "cue_v1" if accuracy_drop <= 0.05 and token_drop_ratio >= 0.10 else "disagreement_triggered"
    return {
        "selected_policy": selected,
        "accuracy_drop_vs_always": round(accuracy_drop, 6),
        "token_drop_ratio_vs_always": round(token_drop_ratio, 6),
        "rule_passed": selected == "cue_v1",
    }


def _resolve_split_name(experiment: CueExperimentConfig, phase_name: str, benchmark_slug: str) -> str:
    phase = phase_metadata(experiment, phase_name)
    if "split_overrides" in phase:
        return phase["split_overrides"][benchmark_slug]
    return phase["split_suffix"]


def _estimate_work(
    experiment: CueExperimentConfig,
    phase_name: str,
    benchmarks,
    protocol: CueProtocolConfig,
    controls: dict[str, MethodConfig],
    policies: list[CuePolicyConfig],
) -> tuple[int, int]:
    total_calls = 0
    total_predictions = 0
    for benchmark in benchmarks:
        split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
        sample_count = len(select_samples(benchmark, split_name))
        total_calls += sample_count * (protocol.agent_count * 2 + 1)
        total_calls += sample_count * sum(method.budget_calls for method in controls.values())
        total_predictions += sample_count * (len(policies) + len(controls) + 1)
    return total_calls, total_predictions


def _prepare_run_paths(run_root: str | Path, experiment_name: str, phase_name: str, run_id: str) -> RunPaths:
    root = prepare_run_root(run_root, experiment_name, phase_name, run_id)
    return RunPaths(
        root=root,
        manifest=root / "manifest.json",
        stage_a_turns=root / "stage_a_turns.jsonl",
        communication_turns=root / "communication_turns.jsonl",
        audit_turns=root / "audit_turns.jsonl",
        control_turns=root / "control_turns.jsonl",
        policy_predictions=root / "policy_predictions.jsonl",
        policy_metrics=root / "policy_metrics.json",
        policy_diagnostics=root / "policy_diagnostics.json",
        oracle_trigger_eval=root / "oracle_trigger_eval.json",
        progress=root / "progress.json",
        run_validation=root / "run_validation.json",
        cue_report=root / "report.md",
    )


def _write_rows(handle, rows: list[dict[str, Any]], progress: RunProgressTracker) -> None:
    for row in rows:
        handle.write_row(row)
        progress.record_call(row)


def _write_predictions(handle, rows: list[dict[str, Any]], progress: RunProgressTracker) -> None:
    for row in rows:
        handle.write_row(row)
    if rows:
        progress.record_predictions(len(rows), str(rows[0].get("dataset")), str(rows[0].get("method_name")))


def _execute_turn_batch(turn_specs: list[dict[str, Any]], *, max_workers: int) -> list[dict[str, Any]]:
    if not turn_specs:
        return []
    # CUE 的样本级外层已经并发；这里顺序执行，避免再嵌套一层线程池。
    return [_execute_turn(**spec) for spec in turn_specs]


    return [_execute_turn(**spec) for spec in turn_specs]


def _trace_hash(rows: list[dict[str, Any]]) -> str:
    payload = [
        {
            "stage_name": row["stage_name"],
            "method_name": row["method_name"],
            "round_index": row["round_index"],
            "agent_id": row["agent_id"],
            "prompt_hash": row["prompt_hash"],
            "normalized_answer": row["normalized_answer"],
            "output_status": row["output_status"],
            "request_error": row["request_error"],
        }
        for row in rows
    ]
    return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _question_preview(question: str, max_chars: int = 120) -> str:
    cleaned = " ".join(question.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3] + "..."


def _mean(values) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return round(sum(materialized) / len(materialized), 6)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)

