"""baseline_compare 的样本级执行与聚合辅助。"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any

from research_experiments.core.data.datasets import DatasetSample, load_split_ids, select_samples
from research_experiments.core.data.evaluation import normalize_prediction
from research_experiments.core.execution.cache import RequestCache
from research_experiments.core.execution.providers import OpenAICompatibleProvider
from research_experiments.core.execution.rate_limits import RequestThrottle
from research_experiments.core.execution.runner_common import execute_cached_turn, iter_indexed_batch
from research_experiments.core.execution.runtime import RunProgressTracker
from research_experiments.core.structured_outputs import SCHEMA_ANSWER_CORE
from research_experiments.families.baseline_compare.config import (
    BaselineCompareExperimentConfig,
    ExperimentSetup,
    ProtocolConfig,
    RosterConfig,
    load_protocol_config,
    load_roster_config,
)
from research_experiments.family_runtime.common import resolve_phase_split_name, safe_mean, safe_ratio
from research_experiments.family_runtime.comparator_impls import (
    build_shared_vanilla_mad_prediction,
    run_shared_vanilla_mad_rounds,
)
from research_experiments.family_runtime.config_helpers import phase_metadata
from research_experiments.family_runtime.vanilla_mad_prompting import (
    CONTROLLED_PROMPT_VERSION,
    PAPER_PROMPT_VERSION,
    prompt_version_uses_json_response_format,
)


@dataclass(frozen=True)
class AgentTurnRecord:
    run_id: str
    dataset: str
    split: str
    sample_id: str
    method_name: str
    method_type: str
    round_index: int
    agent_id: int
    role: str
    prompt_hash: str
    prediction: str
    score: float | None
    output_status: str
    prompt_tokens: float
    completion_tokens: float
    total_tokens: float
    latency_ms: float
    cache_hit: bool
    request_error: str | None
    visible_peer_count: int
    payload: dict[str, Any]
    assistant_text: str
    provider_reasoning_text: str
    validated_output: dict[str, Any]


@dataclass(frozen=True)
class DebateMessageRecord:
    run_id: str
    dataset: str
    split: str
    sample_id: str
    method_name: str
    round_index: int
    sender_agent_id: int
    recipient_agent_id: int
    sender_answer: str
    sender_reasoning: str


@dataclass(frozen=True)
class FinalPredictionRecord:
    run_id: str
    dataset: str
    split: str
    sample_id: str
    method_name: str
    method_type: str
    model_name: str
    prediction: str
    gold: str
    score: float
    initial_vote_prediction: str | None
    initial_vote_score: float | None
    initial_vote_counts: dict[str, int]
    initial_consensus: bool
    final_vote_prediction: str
    final_vote_score: float
    final_vote_counts: dict[str, int]
    prompt_tokens_per_question: float
    completion_tokens_per_question: float
    total_tokens_per_question: float
    latency_ms_per_question: float
    initial_prompt_tokens_per_question: float
    initial_completion_tokens_per_question: float
    initial_total_tokens_per_question: float
    initial_latency_ms_per_question: float
    debate_prompt_tokens_per_question: float
    debate_completion_tokens_per_question: float
    debate_total_tokens_per_question: float
    debate_latency_ms_per_question: float
    calls_per_question: int
    debate_rounds: int
    agent_count: int
    final_consensus: bool
    initial_disagreement: bool
    vote_flipped: bool
    corrected_by_debate: bool
    harmed_by_debate: bool
    unchanged_correct: bool
    unchanged_wrong: bool


def _run_mad_setup_batch(
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    samples: list[DatasetSample],
    setup: ExperimentSetup,
    protocol: ProtocolConfig,
    roster: RosterConfig,
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    throttle: RequestThrottle,
    global_seed: int,
    prompt_version: str,
    max_concurrent_requests: int,
) -> Iterator[tuple[int, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]]:
    worker = partial(
        _run_mad_sample,
        run_id=run_id,
        benchmark_slug=benchmark_slug,
        split_name=split_name,
        setup=setup,
        protocol=protocol,
        roster=roster,
        backbone=backbone,
        provider=provider,
        cache=cache,
        throttle=throttle,
        global_seed=global_seed,
        prompt_version=prompt_version,
    )
    for sample_index, result in iter_indexed_batch(
        samples,
        worker=worker,
        max_concurrent_requests=max_concurrent_requests,
    ):
        yield (sample_index, *result)


def _write_sample_outputs(
    sample_results: Iterable[tuple[int, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]],
    dataset_slug: str,
    progress: RunProgressTracker,
    turn_handle,
    debate_handle,
    prediction_handle,
    all_turns: list[dict[str, Any]],
    debate_messages: list[dict[str, Any]],
    final_predictions: list[dict[str, Any]],
) -> None:
    for _, turn_rows, debate_rows, prediction_row in sample_results:
        for row in turn_rows:
            turn_handle.write_row(row)
            progress.record_call(row, method_key="method_name")
        for row in debate_rows:
            debate_handle.write_row(row)
        prediction_handle.write_row(prediction_row)
        progress.record_predictions(1, dataset_slug, prediction_row["method_name"])
        all_turns.extend(turn_rows)
        debate_messages.extend(debate_rows)
        final_predictions.append(prediction_row)


def _run_mad_sample(
    sample: DatasetSample,
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    setup: ExperimentSetup,
    protocol: ProtocolConfig,
    roster: RosterConfig,
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    throttle: RequestThrottle,
    global_seed: int,
    prompt_version: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    shared_result = run_shared_vanilla_mad_rounds(
        sample=sample,
        run_id=run_id,
        dataset=benchmark_slug,
        split_name=split_name,
        method_name=setup.name,
        agent_count=roster.agent_count,
        debate_rounds=protocol.debate_rounds,
        initial_temperature=protocol.initial_temperature,
        debate_temperature=protocol.debate_temperature,
        top_p=protocol.top_p,
        global_seed=global_seed,
        prompt_version=prompt_version,
        execute_turn=lambda **kwargs: _execute_turn(
            run_id=run_id,
            dataset=benchmark_slug,
            split_name=split_name,
            sample=sample,
            method_name=setup.name,
            method_type="mad",
            backbone=backbone,
            provider=provider,
            cache=cache,
            throttle=throttle,
            **kwargs,
        ),
        build_debate_row=lambda sender, recipient_id, round_index: asdict(
            DebateMessageRecord(
                run_id=run_id,
                dataset=benchmark_slug,
                split=split_name,
                sample_id=sample.sample_id,
                method_name=setup.name,
                round_index=round_index,
                sender_agent_id=sender["agent_id"],
                recipient_agent_id=recipient_id,
                sender_answer=str(sender["validated_output"].get("final_answer", "")).strip(),
                sender_reasoning=(
                    str(sender.get("assistant_text", "")).strip()
                    if prompt_version == PAPER_PROMPT_VERSION
                    else str(sender["validated_output"].get("reasoning", "")).strip()
                ),
            )
        ),
    )
    prediction_row = build_shared_vanilla_mad_prediction(
        run_id=run_id,
        dataset=benchmark_slug,
        split_name=split_name,
        sample=sample,
        method_name=setup.name,
        method_type="mad",
        model_name=backbone.name,
        result=shared_result,
    )
    return shared_result["turn_rows"], shared_result["debate_rows"], prediction_row


def _build_control_prediction_row(
    *,
    control_name: str,
    method,
    sample: DatasetSample,
    final_vote: str,
    final_score: float,
    vote_counts: dict[str, int],
    final_consensus: bool,
    turn_rows: list[dict[str, Any]],
    backbone,
    benchmark_slug: str,
    split_name: str,
    run_id: str,
) -> dict[str, Any]:
    prompt_tokens = sum(float(row["prompt_tokens"]) for row in turn_rows)
    completion_tokens = sum(float(row["completion_tokens"]) for row in turn_rows)
    total_tokens = sum(float(row["total_tokens"]) for row in turn_rows)
    latency_ms = sum(float(row["latency_ms"]) for row in turn_rows)
    prediction_row = asdict(
        FinalPredictionRecord(
            run_id=run_id,
            dataset=benchmark_slug,
            split=split_name,
            sample_id=sample.sample_id,
            method_name=control_name,
            method_type="control",
            model_name=backbone.name,
            prediction=final_vote,
            gold=sample.reference_answer,
            score=final_score,
            initial_vote_prediction=final_vote,
            initial_vote_score=final_score,
            initial_vote_counts=vote_counts,
            initial_consensus=final_consensus,
            final_vote_prediction=final_vote,
            final_vote_score=final_score,
            final_vote_counts=vote_counts,
            prompt_tokens_per_question=prompt_tokens,
            completion_tokens_per_question=completion_tokens,
            total_tokens_per_question=total_tokens,
            latency_ms_per_question=latency_ms,
            initial_prompt_tokens_per_question=prompt_tokens,
            initial_completion_tokens_per_question=completion_tokens,
            initial_total_tokens_per_question=total_tokens,
            initial_latency_ms_per_question=latency_ms,
            debate_prompt_tokens_per_question=0.0,
            debate_completion_tokens_per_question=0.0,
            debate_total_tokens_per_question=0.0,
            debate_latency_ms_per_question=0.0,
            calls_per_question=method.budget_calls,
            debate_rounds=0,
            agent_count=1 if method.family == "cot" else method.budget_calls,
            final_consensus=final_consensus,
            initial_disagreement=False,
            vote_flipped=False,
            corrected_by_debate=False,
            harmed_by_debate=False,
            unchanged_correct=final_score == 1.0,
            unchanged_wrong=final_score < 1.0,
        )
    )
    prediction_row["vote_counts"] = vote_counts
    return prediction_row


def _execute_turn(
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    method_name: str,
    method_type: str,
    round_index: int,
    agent_id: int,
    role: str,
    visible_peer_count: int,
    messages: list[dict[str, str]],
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    throttle: RequestThrottle,
    temperature: float,
    top_p: float,
    seed: int,
    prompt_version: str = CONTROLLED_PROMPT_VERSION,
) -> dict[str, Any]:
    result = execute_cached_turn(
        backbone=backbone,
        provider=provider,
        cache=cache,
        throttle=throttle,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        schema_id=SCHEMA_ANSWER_CORE,
        dataset=dataset,
        use_response_format=prompt_version_uses_json_response_format(prompt_version),
    )
    final_answer = str(result.validated_output.get("final_answer") or "")
    normalized = normalize_prediction(dataset, final_answer) if final_answer else ""
    return asdict(
        AgentTurnRecord(
            run_id=run_id,
            dataset=dataset,
            split=split_name,
            sample_id=sample.sample_id,
            method_name=method_name,
            method_type=method_type,
            round_index=round_index,
            agent_id=agent_id,
            role=role,
            prompt_hash=result.prompt_hash,
            prediction=normalized,
            score=None,
            output_status=result.output_status,
            prompt_tokens=float(result.usage.get("prompt_tokens") or 0.0),
            completion_tokens=float(result.usage.get("completion_tokens") or 0.0),
            total_tokens=float(result.usage.get("total_tokens") or 0.0),
            latency_ms=float(result.response_payload.get("latency_ms") or 0.0),
            cache_hit=result.cache_hit,
            request_error=result.request_error,
            visible_peer_count=visible_peer_count,
            payload=result.payload,
            assistant_text=result.response_payload.get("assistant_text", ""),
            provider_reasoning_text=result.response_payload.get("provider_reasoning_text", ""),
            validated_output=result.validated_output,
        )
    ) | {"normalized_answer": normalized}


def _build_metrics(
    prediction_rows: list[dict[str, Any]],
    *,
    dataset_order: list[str],
    method_order: list[str],
    control_names: list[str],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in prediction_rows:
        grouped.setdefault((row["dataset"], row["model_name"], row["method_name"]), []).append(row)

    dataset_rows: list[dict[str, Any]] = []
    for (dataset, model_name, method_name), rows in grouped.items():
        dataset_rows.append(
            _summarize_prediction_rows(
                rows,
                dataset=dataset,
                model_name=model_name,
                method_name=method_name,
                aggregate_kind="dataset",
            )
        )

    macro_rows = _build_macro_rows(dataset_rows)
    micro_rows = _build_micro_rows(prediction_rows)
    summary = [*dataset_rows, *macro_rows, *micro_rows]
    _attach_comparison_fields(summary, control_names=control_names)
    summary.sort(key=lambda row: _summary_sort_key(row, dataset_order, method_order))
    return {"summary": summary}


def _build_cost_breakdown(
    turn_rows: list[dict[str, Any]],
    *,
    dataset_order: list[str],
    method_order: list[str],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in turn_rows:
        key = (str(row["dataset"]), str(row["method_name"]), str(row["method_type"]))
        bucket = grouped.setdefault(
            key,
            {
                "prompt_tokens": 0.0,
                "completion_tokens": 0.0,
                "total_tokens": 0.0,
                "latency_ms": 0.0,
                "turn_count": 0.0,
                "initial_tokens": 0.0,
                "debate_tokens": 0.0,
                "control_tokens": 0.0,
            },
        )
        total_tokens = float(row["total_tokens"])
        bucket["prompt_tokens"] += float(row["prompt_tokens"])
        bucket["completion_tokens"] += float(row["completion_tokens"])
        bucket["total_tokens"] += total_tokens
        bucket["latency_ms"] += float(row["latency_ms"])
        bucket["turn_count"] += 1.0
        role = str(row.get("role") or "")
        if role == "initial":
            bucket["initial_tokens"] += total_tokens
        elif role == "debate":
            bucket["debate_tokens"] += total_tokens
        else:
            bucket["control_tokens"] += total_tokens

    rows: list[dict[str, Any]] = []
    for (dataset, method_name, method_type), bucket in grouped.items():
        rows.append(
            {
                "dataset": dataset,
                "model_name": "",
                "method_name": method_name,
                "method_type": method_type,
                **{key: round(value, 6) for key, value in bucket.items()},
            }
        )
    rows.sort(key=lambda row: _summary_sort_key(row, dataset_order, method_order))
    return {"rows": rows}


def _build_debate_diagnostics(
    prediction_rows: list[dict[str, Any]],
    *,
    dataset_order: list[str],
    method_order: list[str],
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in prediction_rows:
        if row["method_type"] != "mad":
            continue
        grouped.setdefault((row["dataset"], row["model_name"], row["method_name"]), []).append(row)

    rows: list[dict[str, Any]] = []
    for (dataset, model_name, method_name), items in grouped.items():
        summary = _summarize_prediction_rows(
            items,
            dataset=dataset,
            model_name=model_name,
            method_name=method_name,
            aggregate_kind="dataset",
        )
        rows.append(
            {
                "dataset": dataset,
                "aggregate_kind": "dataset",
                "model_name": model_name,
                "method_name": method_name,
                "question_count": summary["question_count"],
                "initial_vote_accuracy_mean": summary["initial_vote_accuracy_mean"],
                "accuracy_mean": summary["accuracy_mean"],
                "debate_gain_over_initial_vote": summary["debate_gain_over_initial_vote"],
                "corrected_count": summary["corrected_count"],
                "harmed_count": summary["harmed_count"],
                "corrected_rate": summary["corrected_rate"],
                "harmed_rate": summary["harmed_rate"],
                "flip_rate": summary["flip_rate"],
                "initial_consensus_rate": summary["initial_consensus_rate"],
                "final_consensus_rate": summary["final_consensus_rate"],
                "communication_tokens_mean": summary["communication_tokens_mean"],
                "latency_ms_mean": summary["latency_ms_mean"],
            }
        )

    overall_grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in prediction_rows:
        if row["method_type"] != "mad":
            continue
        overall_grouped.setdefault((row["model_name"], row["method_name"]), []).append(row)
    for (model_name, method_name), items in overall_grouped.items():
        summary = _summarize_prediction_rows(
            items,
            dataset="overall",
            model_name=model_name,
            method_name=method_name,
            aggregate_kind="micro",
        )
        rows.append(
            {
                "dataset": "overall",
                "aggregate_kind": "micro",
                "model_name": model_name,
                "method_name": method_name,
                "question_count": summary["question_count"],
                "initial_vote_accuracy_mean": summary["initial_vote_accuracy_mean"],
                "accuracy_mean": summary["accuracy_mean"],
                "debate_gain_over_initial_vote": summary["debate_gain_over_initial_vote"],
                "corrected_count": summary["corrected_count"],
                "harmed_count": summary["harmed_count"],
                "corrected_rate": summary["corrected_rate"],
                "harmed_rate": summary["harmed_rate"],
                "flip_rate": summary["flip_rate"],
                "initial_consensus_rate": summary["initial_consensus_rate"],
                "final_consensus_rate": summary["final_consensus_rate"],
                "communication_tokens_mean": summary["communication_tokens_mean"],
                "latency_ms_mean": summary["latency_ms_mean"],
            }
        )

    rows.sort(key=lambda row: _summary_sort_key(row, dataset_order, method_order))
    return {"rows": rows}


def _estimate_work(
    experiment: BaselineCompareExperimentConfig,
    phase_name: str,
    benchmarks,
    setups: list[ExperimentSetup],
    controls,
) -> tuple[int, int]:
    phase_metadata(experiment, phase_name)
    total_calls = 0
    total_predictions = 0
    for benchmark in benchmarks:
        split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
        sample_count = len(load_split_ids(benchmark.cache_namespace or benchmark.slug, split_name))
        for setup in setups:
            protocol = load_protocol_config(setup.protocol)
            roster = load_roster_config(setup.roster)
            total_calls += sample_count * roster.agent_count * (1 + protocol.debate_rounds)
            total_predictions += sample_count
        for method_name in experiment.control_methods:
            total_calls += sample_count * controls[method_name].budget_calls
            total_predictions += sample_count
    return total_calls, total_predictions


def _active_setups(
    experiment: BaselineCompareExperimentConfig,
    phase_name: str,
) -> list[ExperimentSetup]:
    phase = phase_metadata(experiment, phase_name)
    requested = set(phase["setups"])
    available = {item.name: item for item in experiment.setups}
    missing = sorted(requested - set(available))
    if missing:
        raise RuntimeError(f"Unknown baseline_compare setups for phase {phase_name}: {', '.join(missing)}")
    return [available[name] for name in phase["setups"]]


def _resolve_split_name(
    experiment: BaselineCompareExperimentConfig,
    phase_name: str,
    benchmark_slug: str,
) -> str:
    return resolve_phase_split_name(experiment, phase_name, benchmark_slug)


def _load_selected_samples(benchmark, split_name: str) -> list[DatasetSample]:
    return select_samples(benchmark, split_name)


def _summarize_prediction_rows(
    rows: list[dict[str, Any]],
    *,
    dataset: str,
    model_name: str,
    method_name: str,
    aggregate_kind: str,
) -> dict[str, Any]:
    question_count = len(rows)
    accuracy = safe_mean(float(row["score"]) for row in rows)
    total_tokens = safe_mean(float(row["total_tokens_per_question"]) for row in rows)
    initial_accuracy = safe_mean(
        float(row["initial_vote_score"]) if row.get("initial_vote_score") is not None else float(row["score"])
        for row in rows
    )
    corrected_count = sum(1 for row in rows if row.get("corrected_by_debate"))
    harmed_count = sum(1 for row in rows if row.get("harmed_by_debate"))
    return {
        "dataset": dataset,
        "aggregate_kind": aggregate_kind,
        "model_name": model_name,
        "method_name": method_name,
        "method_type": rows[0]["method_type"],
        "question_count": question_count,
        "prediction_rows": question_count,
        "accuracy_mean": accuracy,
        "initial_vote_accuracy_mean": initial_accuracy,
        "debate_gain_over_initial_vote": round(accuracy - initial_accuracy, 6),
        "prompt_tokens_mean": safe_mean(float(row["prompt_tokens_per_question"]) for row in rows),
        "completion_tokens_mean": safe_mean(float(row["completion_tokens_per_question"]) for row in rows),
        "total_tokens_mean": total_tokens,
        "communication_tokens_mean": safe_mean(float(row["debate_total_tokens_per_question"]) for row in rows),
        "latency_ms_mean": safe_mean(float(row["latency_ms_per_question"]) for row in rows),
        "calls_per_question_mean": safe_mean(float(row["calls_per_question"]) for row in rows),
        "accuracy_per_1k_tokens": round((accuracy / total_tokens * 1000) if total_tokens else 0.0, 6),
        "debate_rounds": safe_mean(float(row["debate_rounds"]) for row in rows),
        "agent_count": safe_mean(float(row["agent_count"]) for row in rows),
        "corrected_count": corrected_count,
        "harmed_count": harmed_count,
        "corrected_rate": safe_ratio(corrected_count, question_count),
        "harmed_rate": safe_ratio(harmed_count, question_count),
        "flip_rate": safe_mean(1.0 if row.get("vote_flipped") else 0.0 for row in rows),
        "initial_consensus_rate": safe_mean(1.0 if row.get("initial_consensus") else 0.0 for row in rows),
        "final_consensus_rate": safe_mean(1.0 if row.get("final_consensus") else 0.0 for row in rows),
    }


def _build_macro_rows(dataset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in dataset_rows:
        grouped.setdefault((row["model_name"], row["method_name"]), []).append(row)

    macro_rows: list[dict[str, Any]] = []
    for (model_name, method_name), rows in grouped.items():
        accuracy = safe_mean(float(row["accuracy_mean"]) for row in rows)
        total_tokens = safe_mean(float(row["total_tokens_mean"]) for row in rows)
        initial_accuracy = safe_mean(float(row["initial_vote_accuracy_mean"]) for row in rows)
        macro_rows.append(
            {
                "dataset": "overall",
                "aggregate_kind": "macro",
                "model_name": model_name,
                "method_name": method_name,
                "method_type": rows[0]["method_type"],
                "question_count": sum(int(row["question_count"]) for row in rows),
                "prediction_rows": sum(int(row["prediction_rows"]) for row in rows),
                "accuracy_mean": accuracy,
                "initial_vote_accuracy_mean": initial_accuracy,
                "debate_gain_over_initial_vote": round(accuracy - initial_accuracy, 6),
                "prompt_tokens_mean": safe_mean(float(row["prompt_tokens_mean"]) for row in rows),
                "completion_tokens_mean": safe_mean(float(row["completion_tokens_mean"]) for row in rows),
                "total_tokens_mean": total_tokens,
                "communication_tokens_mean": safe_mean(float(row["communication_tokens_mean"]) for row in rows),
                "latency_ms_mean": safe_mean(float(row["latency_ms_mean"]) for row in rows),
                "calls_per_question_mean": safe_mean(float(row["calls_per_question_mean"]) for row in rows),
                "accuracy_per_1k_tokens": round((accuracy / total_tokens * 1000) if total_tokens else 0.0, 6),
                "debate_rounds": safe_mean(float(row["debate_rounds"]) for row in rows),
                "agent_count": safe_mean(float(row["agent_count"]) for row in rows),
                "corrected_count": sum(int(row["corrected_count"]) for row in rows),
                "harmed_count": sum(int(row["harmed_count"]) for row in rows),
                "corrected_rate": safe_mean(float(row["corrected_rate"]) for row in rows),
                "harmed_rate": safe_mean(float(row["harmed_rate"]) for row in rows),
                "flip_rate": safe_mean(float(row["flip_rate"]) for row in rows),
                "initial_consensus_rate": safe_mean(float(row["initial_consensus_rate"]) for row in rows),
                "final_consensus_rate": safe_mean(float(row["final_consensus_rate"]) for row in rows),
            }
        )
    return macro_rows


def _build_micro_rows(prediction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in prediction_rows:
        grouped.setdefault((row["model_name"], row["method_name"]), []).append(row)

    micro_rows: list[dict[str, Any]] = []
    for (model_name, method_name), rows in grouped.items():
        micro_rows.append(
            _summarize_prediction_rows(
                rows,
                dataset="overall_micro",
                model_name=model_name,
                method_name=method_name,
                aggregate_kind="micro",
            )
        )
    return micro_rows


def _attach_comparison_fields(summary_rows: list[dict[str, Any]], *, control_names: list[str]) -> None:
    control_set = set(control_names)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in summary_rows:
        grouped.setdefault((str(row["dataset"]), str(row["model_name"])), []).append(row)

    for (_, _), rows in grouped.items():
        cot_row = next((row for row in rows if row["method_name"] == "cot_1"), None)
        no_comm_rows = [row for row in rows if row["method_name"] in control_set]
        best_no_comm = max(no_comm_rows, key=lambda item: float(item["accuracy_mean"])) if no_comm_rows else None
        for row in rows:
            row["accuracy_delta_vs_cot_1"] = _delta_against(row, cot_row, "accuracy_mean")
            row["token_ratio_vs_cot_1"] = _ratio_against(row, cot_row, "total_tokens_mean")
            row["calls_ratio_vs_cot_1"] = _ratio_against(row, cot_row, "calls_per_question_mean")
            row["best_no_comm_method"] = None if best_no_comm is None else best_no_comm["method_name"]
            row["best_no_comm_accuracy"] = None if best_no_comm is None else best_no_comm["accuracy_mean"]
            row["accuracy_delta_vs_best_no_comm"] = _delta_against(row, best_no_comm, "accuracy_mean")
            row["token_ratio_vs_best_no_comm"] = _ratio_against(row, best_no_comm, "total_tokens_mean")
            row["calls_ratio_vs_best_no_comm"] = _ratio_against(row, best_no_comm, "calls_per_question_mean")


def _delta_against(row: dict[str, Any], reference: dict[str, Any] | None, key: str) -> float | None:
    if reference is None:
        return None
    return round(float(row[key]) - float(reference[key]), 6)


def _ratio_against(row: dict[str, Any], reference: dict[str, Any] | None, key: str) -> float | None:
    if reference is None or not float(reference[key]):
        return None
    return round(float(row[key]) / float(reference[key]), 6)


def _summary_sort_key(
    row: dict[str, Any],
    dataset_order: list[str],
    method_order: list[str],
) -> tuple[int, int, str]:
    dataset_rank = {name: index for index, name in enumerate(dataset_order)}
    method_rank = {name: index for index, name in enumerate(method_order)}
    special_rank = {
        "overall": len(dataset_order),
        "overall_micro": len(dataset_order) + 1,
    }
    return (
        special_rank.get(row["dataset"], dataset_rank.get(row["dataset"], len(dataset_order) + 2)),
        method_rank.get(row["method_name"], 999),
        str(row["model_name"]),
    )
