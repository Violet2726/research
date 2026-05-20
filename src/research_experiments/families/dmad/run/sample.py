"""Sample-level execution helpers for DMAD."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import partial
import json
from typing import Any

from research_experiments.core.data.datasets import DatasetSample, load_split_ids, select_samples
from research_experiments.core.data.evaluation import aggregate_majority, normalize_prediction, score_prediction
from research_experiments.core.execution.runner_common import execute_cached_turn, run_indexed_batch
from research_experiments.core.structured_outputs import ARTIFACT_VERSION, SCHEMA_ANSWER_CORE
from research_experiments.families.dmad.config import DmadExperimentConfig, DmadMethodSpec, ProtocolConfig, RosterConfig
from research_experiments.families.dmad.prompts import (
    build_answer_stage_messages,
    build_initial_messages,
    build_mrp_method_selection_messages,
    build_mrp_solution_messages,
    build_reasoning_stage_messages,
    build_reflection_feedback_messages,
    build_reflection_revision_messages,
    build_self_contrast_checklist_messages,
    build_self_contrast_revision_messages,
)
from research_experiments.families.shared.common import resolve_phase_split_name, safe_mean, safe_ratio
from research_experiments.families.shared.pot_execution import (
    build_pot_answer_artifact,
    build_pot_process_artifact,
    is_pot_reasoning,
)
from research_experiments.families.shared.reasoning_methods import normalize_reasoning_method_name


@dataclass(frozen=True)
class AgentTurnRecord:
    """One model call inside a DMAD run."""

    run_id: str
    dataset: str
    split: str
    sample_id: str
    method_name: str
    method_type: str
    diversity_mode: str
    round_index: int
    agent_id: int
    role: str
    persona_name: str
    strategy_name: str
    prompt_hash: str
    prediction: str
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
    program_text: str | None = None
    execution_result: str | None = None
    execution_status: str | None = None
    execution_resolution: str | None = None
    execution_error: str | None = None


@dataclass(frozen=True)
class DebateMessageRecord:
    """One visible peer message inside a debate round."""

    run_id: str
    dataset: str
    split: str
    sample_id: str
    method_name: str
    diversity_mode: str
    round_index: int
    sender_agent_id: int
    recipient_agent_id: int
    sender_persona_name: str
    sender_strategy_name: str
    sender_answer: str
    sender_reasoning: str


@dataclass(frozen=True)
class FinalPredictionRecord:
    """The final question-level output for one method."""

    run_id: str
    dataset: str
    split: str
    sample_id: str
    method_name: str
    method_type: str
    diversity_mode: str
    strategy_names: list[str]
    configured_strategy_names: list[str]
    effective_strategy_names: list[str]
    persona_names: list[str]
    model_name: str
    prediction: str
    gold: str
    score: float
    initial_vote_prediction: str
    initial_vote_score: float
    final_vote_prediction: str
    final_vote_score: float
    initial_vote_counts: dict[str, int]
    final_vote_counts: dict[str, int]
    initial_consensus: bool
    final_consensus: bool
    initial_disagreement: bool
    vote_flipped: bool
    changed_after_debate: bool
    corrected_by_debate: bool
    harmed_by_debate: bool
    unchanged_correct: bool
    unchanged_wrong: bool
    prompt_tokens_per_question: float
    completion_tokens_per_question: float
    total_tokens_per_question: float
    latency_ms_per_question: float
    communication_tokens_per_question: float
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


@dataclass(frozen=True)
class RoundAgentState:
    round_index: int
    agent_id: int
    persona_name: str
    strategy_name: str
    process_row: dict[str, Any]
    answer_row: dict[str, Any]
    solving_process: str
    final_answer: str
    execution_result: str | None = None


def _resolved_strategy_name(dataset: str, strategy_name: str) -> str:
    return normalize_reasoning_method_name(dataset, strategy_name)


def _configured_strategy_name(strategy_name: str) -> str:
    normalized = str(strategy_name or "").strip().lower()
    if normalized == "pot_l2m":
        return "pot"
    if normalized.endswith("_sc"):
        return _configured_strategy_name(normalized[: -len("_sc")])
    return normalized


def _configured_strategy_names(strategy_names: list[str]) -> list[str]:
    return [_configured_strategy_name(item) for item in strategy_names]


def _turn_artifact_fields(validated_output: dict[str, Any]) -> dict[str, Any]:
    fields = {
        "program_text": _nullable_string(validated_output.get("program_text")),
        "execution_result": _nullable_string(validated_output.get("execution_result")),
        "execution_status": _nullable_string(validated_output.get("execution_status")),
        "execution_resolution": _nullable_string(validated_output.get("execution_resolution")),
        "execution_error": _nullable_string(validated_output.get("execution_error")),
    }
    return {key: value for key, value in fields.items() if value is not None}


def _serialize_turn_record(record: AgentTurnRecord, *, normalized_answer: str) -> dict[str, Any]:
    payload = asdict(record)
    payload["validated_output"] = _compact_validated_output(payload.get("validated_output") or {})
    for key in ["program_text", "execution_result", "execution_status", "execution_resolution", "execution_error"]:
        if payload.get(key) is None:
            payload.pop(key, None)
    return payload | {"normalized_answer": normalized_answer, "artifact_version": ARTIFACT_VERSION}


def _compact_validated_output(validated_output: dict[str, Any]) -> dict[str, Any]:
    keys_to_drop = ["program_text", "execution_result", "execution_status", "execution_resolution", "execution_error"]
    payload = dict(validated_output)
    for key in keys_to_drop:
        if payload.get(key) is None:
            payload.pop(key, None)
    return payload


def _active_methods(experiment: DmadExperimentConfig) -> list[DmadMethodSpec]:
    return list(experiment.methods)


def _estimate_work(
    experiment: DmadExperimentConfig,
    phase_name: str,
    benchmarks,
    protocol: ProtocolConfig,
    methods: list[DmadMethodSpec],
    rosters: dict[str, RosterConfig],
    controls: dict[str, Any],
    splits_root,
) -> tuple[int, int]:
    total_calls = 0
    total_predictions = 0
    for benchmark in benchmarks:
        split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
        sample_count = len(load_split_ids(benchmark.cache_namespace or benchmark.slug, split_name, splits_root=splits_root))
        total_predictions += sample_count * len(methods)
        for method in methods:
            total_calls += sample_count * _calls_per_question(method, protocol, rosters.get(method.name), controls)
    return total_calls, total_predictions


def _resolve_split_name(experiment: DmadExperimentConfig, phase_name: str, benchmark_slug: str) -> str:
    return resolve_phase_split_name(experiment, phase_name, benchmark_slug)


def _load_selected_samples(benchmark, split_name: str, splits_root) -> list[DatasetSample]:
    return select_samples(benchmark, split_name, splits_root=splits_root)


def _run_sample_batch(
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    samples: list[DatasetSample],
    protocol: ProtocolConfig,
    methods: list[DmadMethodSpec],
    rosters: dict[str, RosterConfig],
    controls: dict[str, Any],
    experiment: DmadExperimentConfig,
    backbone,
    provider,
    cache,
    limiter,
) -> list[tuple[int, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]]:
    worker = partial(
        _run_sample,
        run_id=run_id,
        benchmark_slug=benchmark_slug,
        split_name=split_name,
        protocol=protocol,
        methods=methods,
        rosters=rosters,
        controls=controls,
        experiment=experiment,
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
    )
    return [
        (sample_index, *result)
        for sample_index, result in run_indexed_batch(
            samples,
            worker=worker,
            max_concurrent_requests=experiment.max_concurrent_requests,
        )
    ]


def _write_sample_outputs(
    *,
    sample_results: list[tuple[int, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]],
    dataset_slug: str,
    progress,
    turn_handle,
    debate_handle,
    prediction_handle,
    all_turns: list[dict[str, Any]],
    debate_messages: list[dict[str, Any]],
    final_predictions: list[dict[str, Any]],
) -> None:
    for _, turn_rows, debate_rows, prediction_rows in sample_results:
        for row in turn_rows:
            turn_handle.write_row(row)
            progress.record_call(row, method_key="method_name")
        for row in debate_rows:
            debate_handle.write_row(row)
        for row in prediction_rows:
            prediction_handle.write_row(row)
            progress.record_predictions(1, dataset_slug, row["method_name"])
        all_turns.extend(turn_rows)
        debate_messages.extend(debate_rows)
        final_predictions.extend(prediction_rows)


def _run_sample(
    sample: DatasetSample,
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    protocol: ProtocolConfig,
    methods: list[DmadMethodSpec],
    rosters: dict[str, RosterConfig],
    controls: dict[str, Any],
    experiment: DmadExperimentConfig,
    backbone,
    provider,
    cache,
    limiter,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    turn_rows: list[dict[str, Any]] = []
    debate_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for method_index, method in enumerate(methods):
        roster = rosters.get(method.name)
        if method.mode in {"single_cot", "single_sbp", "single_pot", "single_l2m"}:
            method_turns, method_prediction = _run_single_reasoning(
                sample=sample,
                run_id=run_id,
                benchmark_slug=benchmark_slug,
                split_name=split_name,
                method=method,
                backbone=backbone,
                provider=provider,
                cache=cache,
                limiter=limiter,
                experiment=experiment,
                protocol=protocol,
                seed_offset=method_index * 10_000,
            )
            turn_rows.extend(method_turns)
            prediction_rows.append(method_prediction)
            continue
        if method.mode in {"single_cot_sc", "single_sbp_sc", "single_pot_sc", "single_l2m_sc"}:
            method_turns, method_prediction = _run_self_consistency(
                sample=sample,
                run_id=run_id,
                benchmark_slug=benchmark_slug,
                split_name=split_name,
                method=method,
                backbone=backbone,
                provider=provider,
                cache=cache,
                limiter=limiter,
                experiment=experiment,
                protocol=protocol,
                seed_offset=method_index * 10_000,
            )
            turn_rows.extend(method_turns)
            prediction_rows.append(method_prediction)
            continue
        if method.mode == "self_refine_cot":
            method_turns, method_prediction = _run_single_agent_reflection(
                sample=sample,
                run_id=run_id,
                benchmark_slug=benchmark_slug,
                split_name=split_name,
                method=method,
                backbone=backbone,
                provider=provider,
                cache=cache,
                limiter=limiter,
                experiment=experiment,
                protocol=protocol,
                seed_offset=method_index * 10_000,
            )
            turn_rows.extend(method_turns)
            prediction_rows.append(method_prediction)
            continue
        if method.mode in {"self_contrast_cot_sbp_pot", "self_contrast_cot_sbp_l2m"}:
            method_turns, method_prediction = _run_self_contrast(
                sample=sample,
                run_id=run_id,
                benchmark_slug=benchmark_slug,
                split_name=split_name,
                method=method,
                backbone=backbone,
                provider=provider,
                cache=cache,
                limiter=limiter,
                experiment=experiment,
                protocol=protocol,
                seed_offset=method_index * 10_000,
            )
            turn_rows.extend(method_turns)
            prediction_rows.append(method_prediction)
            continue
        if method.mode in {"mrp_cot_sbp_pot", "mrp_cot_sbp_l2m"}:
            method_turns, method_prediction = _run_mrp(
                sample=sample,
                run_id=run_id,
                benchmark_slug=benchmark_slug,
                split_name=split_name,
                method=method,
                backbone=backbone,
                provider=provider,
                cache=cache,
                limiter=limiter,
                experiment=experiment,
                protocol=protocol,
                seed_offset=method_index * 10_000,
            )
            turn_rows.extend(method_turns)
            prediction_rows.append(method_prediction)
            continue
        method_turns, method_debate_rows, method_prediction = _run_mad_method(
            sample=sample,
            run_id=run_id,
            benchmark_slug=benchmark_slug,
            split_name=split_name,
            method=method,
            roster=roster,
            backbone=backbone,
            provider=provider,
            cache=cache,
            limiter=limiter,
            experiment=experiment,
            protocol=protocol,
            seed_offset=method_index * 10_000,
        )
        turn_rows.extend(method_turns)
        debate_rows.extend(method_debate_rows)
        prediction_rows.append(method_prediction)
    return turn_rows, debate_rows, prediction_rows


def _run_single_reasoning(
    *,
    sample: DatasetSample,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    method: DmadMethodSpec,
    backbone,
    provider,
    cache,
    limiter,
    experiment: DmadExperimentConfig,
    protocol: ProtocolConfig,
    seed_offset: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    del protocol
    profile = _profile_for_method(agent_id=1, strategy_name=_mode_to_reasoning_method(method.mode))
    turn = _execute_reasoning_answer_turn(
        run_id=run_id,
        dataset=benchmark_slug,
        split_name=split_name,
        sample=sample,
        method_name=method.name,
        method_type="single_agent",
        diversity_mode="single_agent",
        round_index=0,
        role="initial",
        visible_peer_count=0,
        persona_name=profile.persona_name,
        strategy_name=_resolved_strategy_name(sample.dataset, profile.strategy_name),
        messages=build_initial_messages(sample, profile, prompt_version=experiment.prompt_version),
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        temperature=0.0,
        top_p=1.0,
        max_output_tokens=256,
        seed=experiment.global_seed + seed_offset,
        agent_id=profile.agent_id,
    )
    prediction = _build_final_prediction(
        run_id=run_id,
        dataset=benchmark_slug,
        split_name=split_name,
        sample=sample,
        method_name=method.name,
        method_type="single_agent",
        diversity_mode="single_agent",
        model_name=backbone.name,
        all_turns=[turn],
        initial_rows=[turn],
        final_rows=[turn],
        debate_rows=[],
        configured_strategy_names=[_configured_strategy_name(profile.strategy_name)],
        effective_strategy_names=[_resolved_strategy_name(sample.dataset, profile.strategy_name)],
        persona_names=[profile.persona_name],
        calls_per_question=1,
        debate_rounds=0,
        agent_count=1,
    )
    return [turn], prediction


def _run_single_agent_reflection(
    *,
    sample: DatasetSample,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    method: DmadMethodSpec,
    backbone,
    provider,
    cache,
    limiter,
    experiment: DmadExperimentConfig,
    protocol: ProtocolConfig,
    seed_offset: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    del protocol
    profile = _profile_for_method(agent_id=1, strategy_name="cot")
    initial_turn = _execute_turn(
        run_id=run_id,
        dataset=benchmark_slug,
        split_name=split_name,
        sample=sample,
        method_name=method.name,
        method_type="reflection",
        diversity_mode="single_agent",
        round_index=0,
        role="initial",
        visible_peer_count=0,
        persona_name=profile.persona_name,
        strategy_name=_resolved_strategy_name(sample.dataset, profile.strategy_name),
        messages=build_initial_messages(sample, profile, prompt_version=experiment.prompt_version),
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        temperature=0.0,
        top_p=1.0,
        max_output_tokens=256,
        seed=experiment.global_seed + seed_offset,
        agent_id=profile.agent_id,
    )
    feedback_turn = _execute_feedback_turn(
        run_id=run_id,
        dataset=benchmark_slug,
        split_name=split_name,
        sample=sample,
        method_name=method.name,
        method_type="reflection",
        diversity_mode="single_agent",
        round_index=1,
        role="feedback",
        visible_peer_count=0,
        persona_name=profile.persona_name,
        strategy_name=_resolved_strategy_name(sample.dataset, profile.strategy_name),
        messages=build_reflection_feedback_messages(
            sample,
            previous_reasoning=str(initial_turn["validated_output"].get("reasoning", "")),
            previous_answer=str(initial_turn["validated_output"].get("final_answer", "")),
            prompt_version=experiment.prompt_version,
        ),
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        temperature=0.0,
        top_p=1.0,
        max_output_tokens=160,
        seed=experiment.global_seed + seed_offset + 1,
        agent_id=profile.agent_id,
    )
    reflection_turn = _execute_turn(
        run_id=run_id,
        dataset=benchmark_slug,
        split_name=split_name,
        sample=sample,
        method_name=method.name,
        method_type="reflection",
        diversity_mode="single_agent",
        round_index=2,
        role="reflection",
        visible_peer_count=0,
        persona_name=profile.persona_name,
        strategy_name=_resolved_strategy_name(sample.dataset, profile.strategy_name),
        messages=build_reflection_revision_messages(
            sample,
            previous_reasoning=str(initial_turn["validated_output"].get("reasoning", "")),
            previous_answer=str(initial_turn["validated_output"].get("final_answer", "")),
            feedback=str(feedback_turn["validated_output"].get("feedback", "")),
            prompt_version=experiment.prompt_version,
        ),
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        temperature=0.0,
        top_p=1.0,
        max_output_tokens=256,
        seed=experiment.global_seed + seed_offset + 2,
        agent_id=profile.agent_id,
    )
    prediction = _build_final_prediction(
        run_id=run_id,
        dataset=benchmark_slug,
        split_name=split_name,
        sample=sample,
        method_name=method.name,
        method_type="reflection",
        diversity_mode="single_agent",
        model_name=backbone.name,
        all_turns=[initial_turn, feedback_turn, reflection_turn],
        initial_rows=[initial_turn],
        final_rows=[reflection_turn],
        debate_rows=[],
        configured_strategy_names=[_configured_strategy_name(profile.strategy_name)],
        effective_strategy_names=[_resolved_strategy_name(sample.dataset, profile.strategy_name)],
        persona_names=[profile.persona_name],
        calls_per_question=3,
        debate_rounds=0,
        agent_count=1,
    )
    return [initial_turn, feedback_turn, reflection_turn], prediction


def _run_self_consistency(
    *,
    sample: DatasetSample,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    method: DmadMethodSpec,
    backbone,
    provider,
    cache,
    limiter,
    experiment: DmadExperimentConfig,
    protocol: ProtocolConfig,
    seed_offset: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    del protocol
    strategy_name = _mode_to_reasoning_method(method.mode)
    turns: list[dict[str, Any]] = []
    for replicate_index in range(3):
        profile = _profile_for_method(agent_id=replicate_index + 1, strategy_name=strategy_name)
        turns.append(
            _execute_reasoning_answer_turn(
                run_id=run_id,
                dataset=benchmark_slug,
                split_name=split_name,
                sample=sample,
                method_name=method.name,
                method_type="single_agent",
                diversity_mode="self_consistency",
                round_index=0,
                role="sample",
                visible_peer_count=0,
                persona_name=profile.persona_name,
                strategy_name=_resolved_strategy_name(sample.dataset, profile.strategy_name),
                messages=build_initial_messages(sample, profile, prompt_version=experiment.prompt_version),
                backbone=backbone,
                provider=provider,
                cache=cache,
                limiter=limiter,
                temperature=0.7,
                top_p=1.0,
                max_output_tokens=256,
                seed=experiment.global_seed + seed_offset + replicate_index,
                agent_id=profile.agent_id,
            )
        )
    prediction = _build_final_prediction(
        run_id=run_id,
        dataset=benchmark_slug,
        split_name=split_name,
        sample=sample,
        method_name=method.name,
        method_type="single_agent",
        diversity_mode="self_consistency",
        model_name=backbone.name,
        all_turns=turns,
        initial_rows=turns,
        final_rows=turns,
        debate_rows=[],
        configured_strategy_names=[_configured_strategy_name(strategy_name)],
        effective_strategy_names=[_resolved_strategy_name(sample.dataset, strategy_name)],
        persona_names=["general_reasoner"],
        calls_per_question=3,
        debate_rounds=0,
        agent_count=1,
    )
    return turns, prediction


def _run_self_contrast(
    *,
    sample: DatasetSample,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    method: DmadMethodSpec,
    backbone,
    provider,
    cache,
    limiter,
    experiment: DmadExperimentConfig,
    protocol: ProtocolConfig,
    seed_offset: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    del protocol
    strategy_names = _contrast_reasoning_methods(method.mode)
    candidate_turns: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, str]] = []
    for index, strategy_name in enumerate(strategy_names):
        profile = _profile_for_method(agent_id=index + 1, strategy_name=strategy_name)
        turn = _execute_reasoning_answer_turn(
            run_id=run_id,
            dataset=benchmark_slug,
            split_name=split_name,
            sample=sample,
            method_name=method.name,
            method_type="self_contrast",
            diversity_mode="contrastive",
            round_index=0,
            role="candidate",
            visible_peer_count=0,
            persona_name=profile.persona_name,
            strategy_name=_resolved_strategy_name(sample.dataset, profile.strategy_name),
            messages=build_initial_messages(sample, profile, prompt_version=experiment.prompt_version),
            backbone=backbone,
            provider=provider,
            cache=cache,
            limiter=limiter,
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=256,
            seed=experiment.global_seed + seed_offset + index,
            agent_id=profile.agent_id,
        )
        candidate_turns.append(turn)
        candidate_rows.append(
            {
                "strategy_name": _resolved_strategy_name(sample.dataset, strategy_name),
                "reasoning": str(turn.get("validated_output", {}).get("reasoning") or ""),
                "answer": str(turn.get("validated_output", {}).get("final_answer") or ""),
            }
        )
    checklist_turn = _execute_feedback_turn(
        run_id=run_id,
        dataset=benchmark_slug,
        split_name=split_name,
        sample=sample,
        method_name=method.name,
        method_type="self_contrast",
        diversity_mode="contrastive",
        round_index=1,
        role="contrast_checklist",
        visible_peer_count=len(candidate_rows),
        persona_name="contrast_reviewer",
        strategy_name="contrastive_review",
        messages=build_self_contrast_checklist_messages(sample, candidate_rows, prompt_version=experiment.prompt_version),
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        temperature=0.0,
        top_p=1.0,
        max_output_tokens=256,
        seed=experiment.global_seed + seed_offset + 100,
        agent_id=0,
    )
    final_turn = _execute_turn(
        run_id=run_id,
        dataset=benchmark_slug,
        split_name=split_name,
        sample=sample,
        method_name=method.name,
        method_type="self_contrast",
        diversity_mode="contrastive",
        round_index=2,
        role="contrast_revision",
        visible_peer_count=len(candidate_rows),
        persona_name="contrast_solver",
        strategy_name="contrastive_revision",
        messages=build_self_contrast_revision_messages(
            sample,
            candidate_rows,
            checklist=str(checklist_turn.get("validated_output", {}).get("feedback") or ""),
            prompt_version=experiment.prompt_version,
        ),
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        temperature=0.0,
        top_p=1.0,
        max_output_tokens=256,
        seed=experiment.global_seed + seed_offset + 101,
        agent_id=0,
    )
    all_turns = candidate_turns + [checklist_turn, final_turn]
    prediction = _build_final_prediction(
        run_id=run_id,
        dataset=benchmark_slug,
        split_name=split_name,
        sample=sample,
        method_name=method.name,
        method_type="self_contrast",
        diversity_mode="contrastive",
        model_name=backbone.name,
        all_turns=all_turns,
        initial_rows=candidate_turns,
        final_rows=[final_turn],
        debate_rows=[],
        configured_strategy_names=_configured_strategy_names(strategy_names),
        effective_strategy_names=[_resolved_strategy_name(sample.dataset, item) for item in strategy_names],
        persona_names=["general_reasoner"],
        calls_per_question=len(all_turns),
        debate_rounds=0,
        agent_count=1,
    )
    return all_turns, prediction


def _run_mrp(
    *,
    sample: DatasetSample,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    method: DmadMethodSpec,
    backbone,
    provider,
    cache,
    limiter,
    experiment: DmadExperimentConfig,
    protocol: ProtocolConfig,
    seed_offset: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    del protocol
    strategy_names = _contrast_reasoning_methods(method.mode)
    selection_turn = _execute_method_selection_turn(
        run_id=run_id,
        dataset=benchmark_slug,
        split_name=split_name,
        sample=sample,
        method_name=method.name,
        candidate_methods=strategy_names,
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        seed=experiment.global_seed + seed_offset,
        prompt_version=experiment.prompt_version,
    )
    selected_method = str(selection_turn.get("validated_output", {}).get("selected_method") or strategy_names[0])
    solve_profile = _profile_for_method(agent_id=1, strategy_name=selected_method)
    final_turn = _execute_reasoning_answer_turn(
        run_id=run_id,
        dataset=benchmark_slug,
        split_name=split_name,
        sample=sample,
        method_name=method.name,
        method_type="mrp",
        diversity_mode="dynamic_reasoning",
        round_index=1,
        role="mrp_solution",
        visible_peer_count=0,
        persona_name=solve_profile.persona_name,
        strategy_name=solve_profile.strategy_name,
        messages=build_mrp_solution_messages(sample, selected_method, prompt_version=experiment.prompt_version),
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        temperature=0.0,
        top_p=1.0,
        max_output_tokens=256,
        seed=experiment.global_seed + seed_offset + 1,
        agent_id=solve_profile.agent_id,
    )
    all_turns = [selection_turn, final_turn]
    prediction = _build_final_prediction(
        run_id=run_id,
        dataset=benchmark_slug,
        split_name=split_name,
        sample=sample,
        method_name=method.name,
        method_type="mrp",
        diversity_mode="dynamic_reasoning",
        model_name=backbone.name,
        all_turns=all_turns,
        initial_rows=[final_turn],
        final_rows=[final_turn],
        debate_rows=[],
        configured_strategy_names=[_configured_strategy_name(selected_method)],
        effective_strategy_names=[_resolved_strategy_name(sample.dataset, selected_method)],
        persona_names=[solve_profile.persona_name],
        calls_per_question=len(all_turns),
        debate_rounds=0,
        agent_count=1,
    )
    return all_turns, prediction


def _run_mad_method(
    *,
    sample: DatasetSample,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    method: DmadMethodSpec,
    roster: RosterConfig | None,
    backbone,
    provider,
    cache,
    limiter,
    experiment: DmadExperimentConfig,
    protocol: ProtocolConfig,
    seed_offset: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if roster is None:
        raise RuntimeError(f"DMAD method {method.name} requires a roster.")
    debate_rows: list[dict[str, Any]] = []
    all_turns: list[dict[str, Any]] = []
    prior_rounds: list[list[dict[str, str]]] = []
    initial_turns: list[dict[str, Any]] = []
    final_turns: list[dict[str, Any]] = []
    final_states: list[RoundAgentState] = []

    for round_index in range(1, protocol.debate_rounds + 1):
        current_round: list[RoundAgentState] = []
        for index, profile in enumerate(roster.agents):
            visible_peer_count = len(roster.agents) - 1 if prior_rounds else 0
            if prior_rounds:
                for sender in prior_rounds[-1]:
                    if int(sender["agent_id"]) == profile.agent_id:
                        continue
                    debate_rows.append(
                        asdict(
                            DebateMessageRecord(
                                run_id=run_id,
                                dataset=benchmark_slug,
                                split=split_name,
                                sample_id=sample.sample_id,
                                method_name=method.name,
                                diversity_mode=roster.diversity_mode,
                                round_index=round_index,
                                sender_agent_id=int(sender["agent_id"]),
                                recipient_agent_id=profile.agent_id,
                                sender_persona_name=str(sender["persona_name"]),
                                sender_strategy_name=str(sender["strategy_name"]),
                                sender_answer=str(sender["answer"]),
                                sender_reasoning=str(sender["reasoning"]),
                            )
                        )
                    )
            process_role = "initial_process" if round_index == 1 else "debate_process"
            answer_role = "initial_answer" if round_index == 1 else "debate_answer"
            process_row = _execute_process_turn(
                run_id=run_id,
                dataset=benchmark_slug,
                split_name=split_name,
                sample=sample,
                method_name=method.name,
                method_type="mad",
                diversity_mode=roster.diversity_mode,
                round_index=round_index,
                role=process_role,
                visible_peer_count=visible_peer_count,
                persona_name=profile.persona_name,
                strategy_name=_resolved_strategy_name(sample.dataset, profile.strategy_name),
                messages=build_reasoning_stage_messages(
                    sample,
                    profile,
                    round_index=round_index,
                    prior_rounds=prior_rounds,
                    prompt_version=experiment.prompt_version,
                ),
                backbone=backbone,
                provider=provider,
                cache=cache,
                limiter=limiter,
                temperature=protocol.initial_temperature if round_index == 1 else protocol.debate_temperature,
                top_p=protocol.top_p,
                max_output_tokens=protocol.max_output_tokens,
                seed=experiment.global_seed + seed_offset + round_index * 100 + index * 2,
                agent_id=profile.agent_id,
            )
            solving_process = str(process_row["validated_output"].get("reasoning", "")).strip()
            execution_result = _nullable_string(process_row["validated_output"].get("execution_result"))
            execution_status = _nullable_string(process_row["validated_output"].get("execution_status"))
            execution_error = _nullable_string(process_row["validated_output"].get("execution_error"))
            answer_row = _execute_turn(
                run_id=run_id,
                dataset=benchmark_slug,
                split_name=split_name,
                sample=sample,
                method_name=method.name,
                method_type="mad",
                diversity_mode=roster.diversity_mode,
                round_index=round_index,
                role=answer_role,
                visible_peer_count=visible_peer_count,
                persona_name=profile.persona_name,
                strategy_name=_resolved_strategy_name(sample.dataset, profile.strategy_name),
                messages=build_answer_stage_messages(
                    sample,
                    profile,
                    round_index=round_index,
                    solving_process=solving_process,
                    execution_result=execution_result,
                    execution_status=execution_status,
                    execution_error=execution_error,
                    prior_rounds=prior_rounds,
                    prompt_version=experiment.prompt_version,
                ),
                backbone=backbone,
                provider=provider,
                cache=cache,
                limiter=limiter,
                temperature=0.0,
                top_p=protocol.top_p,
                max_output_tokens=min(128, protocol.max_output_tokens),
                seed=experiment.global_seed + seed_offset + round_index * 100 + index * 2 + 1,
                agent_id=profile.agent_id,
            )
            current_round.append(
                RoundAgentState(
                    round_index=round_index,
                    agent_id=profile.agent_id,
                    persona_name=profile.persona_name,
                    strategy_name=_resolved_strategy_name(sample.dataset, profile.strategy_name),
                    process_row=process_row,
                    answer_row=answer_row,
                    solving_process=solving_process,
                    final_answer=str(answer_row["validated_output"].get("final_answer", "")).strip(),
                    execution_result=execution_result,
                )
            )
            if _should_backfill_execution_from_answer(process_row, answer_row):
                _backfill_execution_from_answer(process_row, answer_row)
            all_turns.extend([process_row, answer_row])

        round_records = [
            {
                "round_index": str(state.round_index),
                "agent_id": str(state.agent_id),
                "persona_name": state.persona_name,
                "strategy_name": state.strategy_name,
                "reasoning": state.solving_process,
                "answer": state.final_answer,
                "execution_result": state.execution_result,
            }
            for state in current_round
        ]
        prior_rounds.append(round_records)
        answer_rows = [state.answer_row for state in current_round]
        if round_index == 1:
            initial_turns = answer_rows
        final_turns = answer_rows
        final_states = current_round
    prediction = _build_final_prediction(
        run_id=run_id,
        dataset=benchmark_slug,
        split_name=split_name,
        sample=sample,
        method_name=method.name,
        method_type="mad",
        diversity_mode=roster.diversity_mode,
        model_name=backbone.name,
        all_turns=all_turns,
        initial_rows=initial_turns,
        final_rows=final_turns,
        debate_rows=[row for row in all_turns if str(row["role"]).startswith("debate_")],
        configured_strategy_names=_configured_strategy_names([profile.strategy_name for profile in roster.agents]),
        effective_strategy_names=[_resolved_strategy_name(sample.dataset, profile.strategy_name) for profile in roster.agents],
        persona_names=[profile.persona_name for profile in roster.agents],
        calls_per_question=len(roster.agents) * protocol.debate_rounds * 2,
        debate_rounds=protocol.debate_rounds,
        agent_count=roster.agent_count,
    )
    return all_turns, debate_rows, prediction


def _build_final_prediction(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    method_name: str,
    method_type: str,
    diversity_mode: str,
    model_name: str,
    all_turns: list[dict[str, Any]],
    initial_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    debate_rows: list[dict[str, Any]],
    configured_strategy_names: list[str],
    effective_strategy_names: list[str],
    persona_names: list[str],
    calls_per_question: int,
    debate_rounds: int,
    agent_count: int,
) -> dict[str, Any]:
    initial_answers = [str(row["normalized_answer"]) for row in initial_rows]
    final_answers = [str(row["normalized_answer"]) for row in final_rows]
    initial_vote, initial_vote_counts = aggregate_majority(initial_answers)
    final_vote, final_vote_counts = aggregate_majority(final_answers)
    initial_vote_score = float(score_prediction(dataset, initial_vote, sample.reference_answer))
    final_vote_score = float(score_prediction(dataset, final_vote, sample.reference_answer))
    resolved_prediction = final_vote
    resolved_score = float(score_prediction(dataset, resolved_prediction, sample.reference_answer))
    initial_consensus = len(set(initial_answers)) == 1
    final_consensus = len(set(final_answers)) == 1
    initial_disagreement = len(set(initial_answers)) > 1
    initial_prompt_tokens = sum(float(row["prompt_tokens"]) for row in initial_rows)
    initial_completion_tokens = sum(float(row["completion_tokens"]) for row in initial_rows)
    initial_total_tokens = sum(float(row["total_tokens"]) for row in initial_rows)
    initial_latency = sum(float(row["latency_ms"]) for row in initial_rows)
    total_prompt_tokens = sum(float(row["prompt_tokens"]) for row in all_turns)
    total_completion_tokens = sum(float(row["completion_tokens"]) for row in all_turns)
    total_tokens = sum(float(row["total_tokens"]) for row in all_turns)
    total_latency = sum(float(row["latency_ms"]) for row in all_turns)
    debate_prompt_tokens = sum(float(row["prompt_tokens"]) for row in debate_rows)
    debate_completion_tokens = sum(float(row["completion_tokens"]) for row in debate_rows)
    debate_total_tokens = sum(float(row["total_tokens"]) for row in debate_rows)
    debate_latency = sum(float(row["latency_ms"]) for row in debate_rows)
    corrected = initial_vote_score < 1.0 and resolved_score == 1.0
    harmed = initial_vote_score == 1.0 and resolved_score < 1.0
    unchanged_correct = initial_vote_score == 1.0 and resolved_score == 1.0
    unchanged_wrong = initial_vote_score < 1.0 and resolved_score < 1.0
    payload = asdict(
        FinalPredictionRecord(
            run_id=run_id,
            dataset=dataset,
            split=split_name,
            sample_id=sample.sample_id,
            method_name=method_name,
            method_type=method_type,
            diversity_mode=diversity_mode,
            strategy_names=list(effective_strategy_names),
            configured_strategy_names=list(configured_strategy_names),
            effective_strategy_names=list(effective_strategy_names),
            persona_names=list(persona_names),
            model_name=model_name,
            prediction=resolved_prediction,
            gold=sample.reference_answer,
            score=resolved_score,
            initial_vote_prediction=initial_vote,
            initial_vote_score=initial_vote_score,
            final_vote_prediction=final_vote,
            final_vote_score=final_vote_score,
            initial_vote_counts=initial_vote_counts,
            final_vote_counts=final_vote_counts,
            initial_consensus=initial_consensus,
            final_consensus=final_consensus,
            initial_disagreement=initial_disagreement,
            vote_flipped=initial_vote != final_vote,
            changed_after_debate=initial_vote != resolved_prediction,
            corrected_by_debate=corrected,
            harmed_by_debate=harmed,
            unchanged_correct=unchanged_correct,
            unchanged_wrong=unchanged_wrong,
            prompt_tokens_per_question=total_prompt_tokens,
            completion_tokens_per_question=total_completion_tokens,
            total_tokens_per_question=total_tokens,
            latency_ms_per_question=total_latency,
            communication_tokens_per_question=debate_total_tokens,
            initial_prompt_tokens_per_question=initial_prompt_tokens,
            initial_completion_tokens_per_question=initial_completion_tokens,
            initial_total_tokens_per_question=initial_total_tokens,
            initial_latency_ms_per_question=initial_latency,
            debate_prompt_tokens_per_question=debate_prompt_tokens,
            debate_completion_tokens_per_question=debate_completion_tokens,
            debate_total_tokens_per_question=debate_total_tokens,
            debate_latency_ms_per_question=debate_latency,
            calls_per_question=calls_per_question,
            debate_rounds=debate_rounds,
            agent_count=agent_count,
        )
    )
    payload["selection_mode"] = "self_consistency"
    payload["selected_answer_matches_final_vote"] = True
    payload["paper_subject"] = str(sample.metadata.get("subject") or "")
    payload["paper_domain"] = str(sample.metadata.get("high_level_domain") or "")
    return payload


def _build_metrics(prediction_rows: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    grouped = _group_prediction_rows(prediction_rows)
    summary: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        summary.append(_build_summary_row(dataset=key[0], method_name=key[1], rows=rows, model_name=model_name))
    summary.extend(
        _build_summary_row(dataset="overall", method_name=method_name, rows=rows, model_name=model_name)
        for method_name, rows in sorted(_group_overall_rows(prediction_rows).items())
    )
    _annotate_gain_rows(summary)
    return {"summary": summary, "prediction_count": len(prediction_rows)}


def _build_paper_tables(
    prediction_rows: list[dict[str, Any]],
    *,
    evaluation_scope: str,
) -> dict[str, Any]:
    math_rows = [row for row in prediction_rows if str(row.get("dataset")) == "competition_math"]
    gpqa_rows = [row for row in prediction_rows if str(row.get("dataset")) == "gpqa_diamond"]
    appendix_rows = [row for row in prediction_rows if str(row.get("dataset")) == "mmlu_abstract_algebra"]
    extended_rows = [
        row
        for row in prediction_rows
        if str(row.get("dataset")) in {"strategyqa", "hotpotqa"}
    ]

    if evaluation_scope == "paper_main":
        overall_source = [row for row in prediction_rows if str(row.get("dataset")) in {"competition_math", "gpqa_diamond"}]
    elif evaluation_scope == "paper_appendix":
        overall_source = appendix_rows
    else:
        overall_source = prediction_rows

    return {
        "evaluation_scope": evaluation_scope,
        "overall_rows": _group_accuracy_rows(overall_source, field_name=None),
        "math_subject_rows": _group_accuracy_rows(math_rows, field_name="paper_subject"),
        "gpqa_domain_rows": _group_accuracy_rows(gpqa_rows, field_name="paper_domain"),
        "appendix_rows": _group_accuracy_rows(appendix_rows, field_name=None),
        "extended_dataset_rows": _group_accuracy_rows(extended_rows, field_name="dataset"),
    }


def _build_strategy_diagnostics(
    prediction_rows: list[dict[str, Any]],
    *,
    evaluation_scope: str,
) -> dict[str, Any]:
    grouped = _group_prediction_rows(prediction_rows)
    rows: list[dict[str, Any]] = []
    for key, grouped_rows in sorted(grouped.items()):
        rows.append(_build_strategy_row(dataset=key[0], method_name=key[1], rows=grouped_rows))
    rows.extend(
        _build_strategy_row(dataset="overall", method_name=method_name, rows=grouped_rows)
        for method_name, grouped_rows in sorted(_group_overall_rows(prediction_rows).items())
    )
    _annotate_gain_rows(rows)
    overall_lookup = {(str(row["dataset"]), str(row["method_name"])): row for row in rows}
    dmad_row = overall_lookup.get(("overall", "dmad_cot_sbp_pot")) or overall_lookup.get(("overall", "dmad_cot_sbp_l2m"))
    fixed_rows = [
        row
        for key, row in overall_lookup.items()
        if key[0] == "overall" and key[1] in {"mad_all_cot", "mad_all_sbp", "mad_all_pot", "mad_all_l2m"}
    ]
    best_fixed_row = max(fixed_rows, key=lambda item: float(item.get("accuracy_mean") or 0.0)) if fixed_rows else None
    gate_passed = bool(
        dmad_row
        and best_fixed_row
        and float(dmad_row.get("accuracy_mean") or 0.0) >= float(best_fixed_row.get("accuracy_mean") or 0.0)
    )
    return {
        "rows": rows,
        "gate_summary": {
            "paper_main_gate_passed": gate_passed,
            "dmad_accuracy_mean": None if dmad_row is None else float(dmad_row.get("accuracy_mean") or 0.0),
            "best_fixed_mad_accuracy_mean": None if best_fixed_row is None else float(best_fixed_row.get("accuracy_mean") or 0.0),
            "dmad_calls_per_question_mean": None if dmad_row is None else float(dmad_row.get("calls_per_question_mean") or 0.0),
            "best_fixed_mad_method_name": None if best_fixed_row is None else str(best_fixed_row.get("method_name") or ""),
        },
        "paper_main_gap_rows": _build_paper_main_gap_rows(prediction_rows, evaluation_scope=evaluation_scope),
    }


def _build_cost_breakdown(turn_rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], dict[str, float | str]] = {}
    for row in turn_rows:
        key = (str(row["dataset"]), str(row["method_name"]))
        bucket = grouped.setdefault(
            key,
            {
                "dataset": str(row["dataset"]),
                "method_name": str(row["method_name"]),
                "prompt_tokens": 0.0,
                "completion_tokens": 0.0,
                "total_tokens": 0.0,
                "latency_ms": 0.0,
                "turn_count": 0.0,
                "initial_tokens": 0.0,
                "debate_tokens": 0.0,
                "reflection_tokens": 0.0,
                "contrast_tokens": 0.0,
                "mrp_tokens": 0.0,
                "control_tokens": 0.0,
            },
        )
        total_tokens = float(row["total_tokens"])
        bucket["prompt_tokens"] = float(bucket["prompt_tokens"]) + float(row["prompt_tokens"])
        bucket["completion_tokens"] = float(bucket["completion_tokens"]) + float(row["completion_tokens"])
        bucket["total_tokens"] = float(bucket["total_tokens"]) + total_tokens
        bucket["latency_ms"] = float(bucket["latency_ms"]) + float(row["latency_ms"])
        bucket["turn_count"] = float(bucket["turn_count"]) + 1.0
        role = str(row["role"])
        if role.startswith("initial_") or role == "initial":
            bucket["initial_tokens"] = float(bucket["initial_tokens"]) + total_tokens
        elif role.startswith("debate_") or role == "debate":
            bucket["debate_tokens"] = float(bucket["debate_tokens"]) + total_tokens
        elif role == "reflection":
            bucket["reflection_tokens"] = float(bucket["reflection_tokens"]) + total_tokens
        elif role.startswith("contrast_") or role == "candidate":
            bucket["contrast_tokens"] = float(bucket["contrast_tokens"]) + total_tokens
        elif role.startswith("mrp_"):
            bucket["mrp_tokens"] = float(bucket["mrp_tokens"]) + total_tokens
        else:
            bucket["control_tokens"] = float(bucket["control_tokens"]) + total_tokens
    return {"rows": list(grouped.values())}


def _group_prediction_rows(prediction_rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in prediction_rows:
        key = (str(row["dataset"]), str(row["method_name"]))
        grouped.setdefault(key, []).append(row)
    return grouped


def _group_overall_rows(prediction_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in prediction_rows:
        grouped.setdefault(str(row["method_name"]), []).append(row)
    return grouped


def _group_accuracy_rows(rows: list[dict[str, Any]], *, field_name: str | None) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        group_name = "overall" if field_name is None else str(row.get(field_name) or "unknown")
        grouped.setdefault((str(row["method_name"]), group_name), []).append(row)
    return [
        {
            "method_name": method_name,
            "group_name": group_name,
            "question_count": len(group_rows),
            "accuracy_mean": safe_mean(float(item["score"]) for item in group_rows),
        }
        for (method_name, group_name), group_rows in sorted(grouped.items())
    ]


def _build_summary_row(*, dataset: str, method_name: str, rows: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    accuracy_mean = safe_mean(float(row["score"]) for row in rows)
    total_tokens_mean = safe_mean(float(row["total_tokens_per_question"]) for row in rows)
    configured_strategy_name = _joined_unique_labels(rows, "configured_strategy_names")
    effective_strategy_name = _effective_strategy_label(dataset=dataset, rows=rows)
    return {
        "dataset": dataset,
        "method_name": method_name,
        "method_type": rows[0]["method_type"],
        "model_name": model_name,
        "question_count": len(rows),
        "accuracy_mean": accuracy_mean,
        "prompt_tokens_mean": safe_mean(float(row["prompt_tokens_per_question"]) for row in rows),
        "completion_tokens_mean": safe_mean(float(row["completion_tokens_per_question"]) for row in rows),
        "total_tokens_mean": total_tokens_mean,
        "communication_tokens_mean": safe_mean(float(row["communication_tokens_per_question"]) for row in rows),
        "calls_per_question_mean": safe_mean(float(row["calls_per_question"]) for row in rows),
        "latency_ms_mean": safe_mean(float(row["latency_ms_per_question"]) for row in rows),
        "accuracy_per_1k_tokens": round((accuracy_mean / total_tokens_mean) * 1000, 6) if total_tokens_mean else 0.0,
        "diversity_mode": rows[0]["diversity_mode"],
        "strategy_name": configured_strategy_name,
        "configured_strategy_name": configured_strategy_name,
        "effective_strategy_name": effective_strategy_name,
        "initial_disagreement_rate": safe_ratio(sum(1 for row in rows if row["initial_disagreement"]), len(rows)),
        "final_consensus_rate": safe_ratio(sum(1 for row in rows if row["final_consensus"]), len(rows)),
        "changed_after_debate_rate": safe_ratio(sum(1 for row in rows if row["changed_after_debate"]), len(rows)),
        "correction_rate": safe_ratio(sum(1 for row in rows if row["corrected_by_debate"]), len(rows)),
        "degradation_rate": safe_ratio(sum(1 for row in rows if row["harmed_by_debate"]), len(rows)),
        "gain_over_best_fixed_mad": None,
        "gain_over_mad_persona_d": None,
        "gain_over_mad_persona_e": None,
    }


def _build_strategy_row(*, dataset: str, method_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    configured_strategy_name = _joined_unique_labels(rows, "configured_strategy_names")
    effective_strategy_name = _effective_strategy_label(dataset=dataset, rows=rows)
    return {
        "dataset": dataset,
        "method_name": method_name,
        "diversity_mode": rows[0]["diversity_mode"],
        "strategy_name": configured_strategy_name,
        "configured_strategy_name": configured_strategy_name,
        "effective_strategy_name": effective_strategy_name,
        "question_count": len(rows),
        "initial_disagreement_rate": safe_ratio(sum(1 for row in rows if row["initial_disagreement"]), len(rows)),
        "final_consensus_rate": safe_ratio(sum(1 for row in rows if row["final_consensus"]), len(rows)),
        "changed_after_debate_rate": safe_ratio(sum(1 for row in rows if row["changed_after_debate"]), len(rows)),
        "correction_rate": safe_ratio(sum(1 for row in rows if row["corrected_by_debate"]), len(rows)),
        "degradation_rate": safe_ratio(sum(1 for row in rows if row["harmed_by_debate"]), len(rows)),
        "accuracy_mean": safe_mean(float(row["score"]) for row in rows),
        "calls_per_question_mean": safe_mean(float(row["calls_per_question"]) for row in rows),
        "gain_over_best_fixed_mad": None,
        "gain_over_mad_persona_d": None,
        "gain_over_mad_persona_e": None,
    }


def _annotate_gain_rows(rows: list[dict[str, Any]]) -> None:
    lookup = {(str(row["dataset"]), str(row["method_name"])): row for row in rows}
    for row in rows:
        dataset = str(row["dataset"])
        fixed_candidates = [
            lookup.get((dataset, name))
            for name in ("mad_all_cot", "mad_all_sbp", "mad_all_pot", "mad_all_l2m")
        ]
        fixed_rows = [item for item in fixed_candidates if item is not None]
        best_fixed = max(fixed_rows, key=lambda item: float(item.get("accuracy_mean") or 0.0)) if fixed_rows else None
        persona_d = lookup.get((dataset, "mad_persona_d"))
        persona_e = lookup.get((dataset, "mad_persona_e"))
        row["gain_over_best_fixed_mad"] = (
            round(float(row["accuracy_mean"]) - float(best_fixed["accuracy_mean"]), 6)
            if best_fixed is not None and str(row["method_name"]) != str(best_fixed["method_name"])
            else None
        )
        row["gain_over_mad_persona_d"] = (
            round(float(row["accuracy_mean"]) - float(persona_d["accuracy_mean"]), 6)
            if persona_d is not None and str(row["method_name"]) != "mad_persona_d"
            else None
        )
        row["gain_over_mad_persona_e"] = (
            round(float(row["accuracy_mean"]) - float(persona_e["accuracy_mean"]), 6)
            if persona_e is not None and str(row["method_name"]) != "mad_persona_e"
            else None
        )


def _joined_unique_labels(rows: list[dict[str, Any]], key: str) -> str:
    labels = sorted({str(item) for row in rows for item in row.get(key, [])})
    return ", ".join(labels)


def _nullable_string(value: object) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _should_backfill_execution_from_answer(process_row: dict[str, Any], answer_row: dict[str, Any]) -> bool:
    process_status = str(process_row.get("execution_status") or "")
    if not process_status or process_status == "ok":
        return False
    final_answer = str(answer_row.get("validated_output", {}).get("final_answer") or "").strip()
    return bool(final_answer)


def _backfill_execution_from_answer(process_row: dict[str, Any], answer_row: dict[str, Any]) -> None:
    final_answer = str(answer_row.get("validated_output", {}).get("final_answer") or "").strip()
    if not final_answer:
        return
    process_row["execution_result"] = final_answer
    process_row["execution_status"] = "ok"
    process_row["execution_resolution"] = "paired_answer"
    process_row.pop("execution_error", None)
    validated_output = dict(process_row.get("validated_output") or {})
    validated_output["execution_result"] = final_answer
    validated_output["execution_status"] = "ok"
    validated_output["execution_resolution"] = "paired_answer"
    validated_output.pop("execution_error", None)
    process_row["validated_output"] = _compact_validated_output(validated_output)


def _effective_strategy_label(*, dataset: str, rows: list[dict[str, Any]]) -> str:
    if dataset != "overall":
        return _joined_unique_labels(rows, "effective_strategy_names")
    grouped_by_dataset: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped_by_dataset.setdefault(str(row.get("dataset") or "unknown"), []).append(row)
    return "; ".join(
        f"{dataset_name}={_joined_unique_labels(group_rows, 'effective_strategy_names')}"
        for dataset_name, group_rows in sorted(grouped_by_dataset.items())
    )


def _build_paper_main_gap_rows(
    prediction_rows: list[dict[str, Any]],
    *,
    evaluation_scope: str,
) -> list[dict[str, Any]]:
    if evaluation_scope != "paper_main":
        return []
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in prediction_rows:
        dataset = str(row.get("dataset") or "")
        if dataset not in {"competition_math", "gpqa_diamond"}:
            continue
        sample_id = str(row.get("sample_id") or "")
        grouped.setdefault((dataset, sample_id), {})[str(row.get("method_name") or "")] = row
    gap_rows: list[dict[str, Any]] = []
    for (dataset, sample_id), method_rows in sorted(grouped.items()):
        dmad_row = method_rows.get("dmad_cot_sbp_pot")
        persona_row = method_rows.get("mad_persona_d")
        fixed_candidates = [
            method_rows.get("mad_all_cot"),
            method_rows.get("mad_all_sbp"),
            method_rows.get("mad_all_pot"),
            method_rows.get("mad_all_l2m"),
        ]
        fixed_rows = [row for row in fixed_candidates if row is not None]
        best_fixed_row = max(fixed_rows, key=lambda item: float(item.get("score") or 0.0)) if fixed_rows else None
        if dmad_row is None:
            continue
        lags_fixed = bool(best_fixed_row and float(dmad_row.get("score") or 0.0) < float(best_fixed_row.get("score") or 0.0))
        lags_persona = bool(persona_row and float(dmad_row.get("score") or 0.0) < float(persona_row.get("score") or 0.0))
        if not lags_fixed and not lags_persona:
            continue
        gap_rows.append(
            {
                "dataset": dataset,
                "sample_id": sample_id,
                "gold": str(dmad_row.get("gold") or ""),
                "dmad_prediction": str(dmad_row.get("prediction") or ""),
                "dmad_score": float(dmad_row.get("score") or 0.0),
                "best_fixed_method_name": None if best_fixed_row is None else str(best_fixed_row.get("method_name") or ""),
                "best_fixed_prediction": None if best_fixed_row is None else str(best_fixed_row.get("prediction") or ""),
                "best_fixed_score": None if best_fixed_row is None else float(best_fixed_row.get("score") or 0.0),
                "persona_d_prediction": None if persona_row is None else str(persona_row.get("prediction") or ""),
                "persona_d_score": None if persona_row is None else float(persona_row.get("score") or 0.0),
                "lags_best_fixed_mad": lags_fixed,
                "lags_mad_persona_d": lags_persona,
            }
        )
    return gap_rows


def _profile_for_method(agent_id: int, strategy_name: str):
    from research_experiments.families.dmad.config import AgentProfile

    return AgentProfile(
        agent_id=agent_id,
        persona_name="general_reasoner",
        persona_instruction="Act as a careful general-purpose reasoning assistant.",
        strategy_name=strategy_name,
        strategy_instruction="Think step by step and make every inference explicit before committing to the final answer.",
    )


def _mode_to_reasoning_method(mode: str) -> str:
    mapping = {
        "single_cot": "cot",
        "single_sbp": "sbp",
        "single_pot": "pot",
        "single_l2m": "l2m",
        "single_cot_sc": "cot",
        "single_sbp_sc": "sbp",
        "single_pot_sc": "pot",
        "single_l2m_sc": "l2m",
    }
    try:
        return mapping[mode]
    except KeyError as exc:
        raise ValueError(f"Unsupported single reasoning mode: {mode}") from exc


def _contrast_reasoning_methods(mode: str) -> list[str]:
    mapping = {
        "self_contrast_cot_sbp_pot": ["cot", "sbp", "pot"],
        "self_contrast_cot_sbp_l2m": ["cot", "sbp", "l2m"],
        "mrp_cot_sbp_pot": ["cot", "sbp", "pot"],
        "mrp_cot_sbp_l2m": ["cot", "sbp", "l2m"],
    }
    try:
        return list(mapping[mode])
    except KeyError as exc:
        raise ValueError(f"Unsupported contrastive reasoning mode: {mode}") from exc


def _calls_per_question(
    method: DmadMethodSpec,
    protocol: ProtocolConfig,
    roster: RosterConfig | None,
    controls: dict[str, Any],
) -> int:
    del controls
    if method.mode in {"single_cot", "single_sbp", "single_pot", "single_l2m"}:
        return 1
    if method.mode in {"single_cot_sc", "single_sbp_sc", "single_pot_sc", "single_l2m_sc"}:
        return 3
    if method.mode == "self_refine_cot":
        return 3
    if method.mode in {"self_contrast_cot_sbp_pot", "self_contrast_cot_sbp_l2m"}:
        return 5
    if method.mode in {"mrp_cot_sbp_pot", "mrp_cot_sbp_l2m"}:
        return 2
    if roster is None:
        raise RuntimeError(f"DMAD method {method.name} is missing roster metadata.")
    return roster.agent_count * protocol.debate_rounds * 2


def _execute_process_turn(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    method_name: str,
    method_type: str,
    diversity_mode: str,
    round_index: int,
    agent_id: int,
    role: str,
    visible_peer_count: int,
    persona_name: str,
    strategy_name: str,
    messages: list[dict[str, str]],
    backbone,
    provider,
    cache,
    limiter,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    seed: int,
) -> dict[str, Any]:
    validator = (
        lambda assistant_text, provider_reasoning_text: asdict(
            build_pot_process_artifact(assistant_text, provider_reasoning_text)
        )
        if is_pot_reasoning(dataset, strategy_name)
        else _validate_reasoning_process_output(assistant_text, provider_reasoning_text)
    )
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
        validator=validator,
        dataset=dataset,
        use_response_format=False,
    )
    validated_output = dict(result.validated_output)
    reasoning_text = str(validated_output.get("reasoning") or "")
    validated_output.setdefault("reasoning", reasoning_text)
    return _serialize_turn_record(
        AgentTurnRecord(
            run_id=run_id,
            dataset=dataset,
            split=split_name,
            sample_id=sample.sample_id,
            method_name=method_name,
            method_type=method_type,
            diversity_mode=diversity_mode,
            round_index=round_index,
            agent_id=agent_id,
            role=role,
            persona_name=persona_name,
            strategy_name=strategy_name,
            prompt_hash=result.prompt_hash,
            prediction="",
            output_status=result.output_status,
            prompt_tokens=float(result.usage.get("prompt_tokens") or 0.0),
            completion_tokens=float(result.usage.get("completion_tokens") or 0.0),
            total_tokens=float(result.usage.get("total_tokens") or 0.0),
            latency_ms=float(result.response_payload.get("latency_ms") or 0.0),
            cache_hit=result.cache_hit,
            request_error=result.request_error,
            visible_peer_count=visible_peer_count,
            payload=result.payload,
            assistant_text=str(result.response_payload.get("assistant_text") or ""),
            provider_reasoning_text=str(result.response_payload.get("provider_reasoning_text") or ""),
            validated_output=validated_output,
            **_turn_artifact_fields(validated_output),
        ),
        normalized_answer="",
    )


def _execute_feedback_turn(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    method_name: str,
    method_type: str,
    diversity_mode: str,
    round_index: int,
    agent_id: int,
    role: str,
    visible_peer_count: int,
    persona_name: str,
    strategy_name: str,
    messages: list[dict[str, str]],
    backbone,
    provider,
    cache,
    limiter,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    seed: int,
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
        validator=_validate_feedback_output,
        dataset=dataset,
        use_response_format=False,
    )
    feedback_text = str(result.validated_output.get("feedback") or "")
    return _serialize_turn_record(
        AgentTurnRecord(
            run_id=run_id,
            dataset=dataset,
            split=split_name,
            sample_id=sample.sample_id,
            method_name=method_name,
            method_type=method_type,
            diversity_mode=diversity_mode,
            round_index=round_index,
            agent_id=agent_id,
            role=role,
            persona_name=persona_name,
            strategy_name=strategy_name,
            prompt_hash=result.prompt_hash,
            prediction="",
            output_status=result.output_status,
            prompt_tokens=float(result.usage.get("prompt_tokens") or 0.0),
            completion_tokens=float(result.usage.get("completion_tokens") or 0.0),
            total_tokens=float(result.usage.get("total_tokens") or 0.0),
            latency_ms=float(result.response_payload.get("latency_ms") or 0.0),
            cache_hit=result.cache_hit,
            request_error=result.request_error,
            visible_peer_count=visible_peer_count,
            payload=result.payload,
            assistant_text=str(result.response_payload.get("assistant_text") or ""),
            provider_reasoning_text=str(result.response_payload.get("provider_reasoning_text") or ""),
            validated_output={"feedback": feedback_text},
        ),
        normalized_answer="",
    )


def _execute_reasoning_answer_turn(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    method_name: str,
    method_type: str,
    diversity_mode: str,
    round_index: int,
    agent_id: int,
    role: str,
    visible_peer_count: int,
    persona_name: str,
    strategy_name: str,
    messages: list[dict[str, str]],
    backbone,
    provider,
    cache,
    limiter,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    seed: int,
) -> dict[str, Any]:
    if not is_pot_reasoning(dataset, strategy_name):
        return _execute_turn(
            run_id=run_id,
            dataset=dataset,
            split_name=split_name,
            sample=sample,
            method_name=method_name,
            method_type=method_type,
            diversity_mode=diversity_mode,
            round_index=round_index,
            agent_id=agent_id,
            role=role,
            visible_peer_count=visible_peer_count,
            persona_name=persona_name,
            strategy_name=strategy_name,
            messages=messages,
            backbone=backbone,
            provider=provider,
            cache=cache,
            limiter=limiter,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            seed=seed,
        )
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
        validator=lambda assistant_text, provider_reasoning_text: asdict(
            build_pot_answer_artifact(
                assistant_text,
                provider_reasoning_text,
                dataset=dataset,
            )
        ),
        dataset=dataset,
    )
    artifact = result.validated_output
    final_answer = str(artifact.get("final_answer") or "")
    normalized = normalize_prediction(dataset, final_answer) if final_answer else ""
    return _serialize_turn_record(
        AgentTurnRecord(
            run_id=run_id,
            dataset=dataset,
            split=split_name,
            sample_id=sample.sample_id,
            method_name=method_name,
            method_type=method_type,
            diversity_mode=diversity_mode,
            round_index=round_index,
            agent_id=agent_id,
            role=role,
            persona_name=persona_name,
            strategy_name=strategy_name,
            prompt_hash=result.prompt_hash,
            prediction=normalized,
            output_status=result.output_status,
            prompt_tokens=float(result.usage.get("prompt_tokens") or 0.0),
            completion_tokens=float(result.usage.get("completion_tokens") or 0.0),
            total_tokens=float(result.usage.get("total_tokens") or 0.0),
            latency_ms=float(result.response_payload.get("latency_ms") or 0.0),
            cache_hit=result.cache_hit,
            request_error=result.request_error,
            visible_peer_count=visible_peer_count,
            payload=result.payload,
            assistant_text=str(result.response_payload.get("assistant_text") or ""),
            provider_reasoning_text=str(result.response_payload.get("provider_reasoning_text") or ""),
            validated_output=dict(artifact),
            **_turn_artifact_fields(artifact),
        ),
        normalized_answer=normalized,
    )


def _execute_method_selection_turn(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    method_name: str,
    candidate_methods: list[str],
    backbone,
    provider,
    cache,
    limiter,
    seed: int,
    prompt_version: str,
) -> dict[str, Any]:
    result = execute_cached_turn(
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        messages=build_mrp_method_selection_messages(
            sample,
            candidate_methods,
            prompt_version=prompt_version,
        ),
        temperature=0.0,
        top_p=1.0,
        max_output_tokens=192,
        seed=seed,
        validator=lambda assistant_text, provider_reasoning_text: _validate_method_selection_output(
            assistant_text,
            provider_reasoning_text,
            candidate_methods=candidate_methods,
        ),
        dataset=dataset,
    )
    return _serialize_turn_record(
        AgentTurnRecord(
            run_id=run_id,
            dataset=dataset,
            split=split_name,
            sample_id=sample.sample_id,
            method_name=method_name,
            method_type="mrp",
            diversity_mode="dynamic_reasoning",
            round_index=0,
            agent_id=0,
            role="mrp_method_selection",
            persona_name="method_router",
            strategy_name="mrp_router",
            prompt_hash=result.prompt_hash,
            prediction="",
            output_status=result.output_status,
            prompt_tokens=float(result.usage.get("prompt_tokens") or 0.0),
            completion_tokens=float(result.usage.get("completion_tokens") or 0.0),
            total_tokens=float(result.usage.get("total_tokens") or 0.0),
            latency_ms=float(result.response_payload.get("latency_ms") or 0.0),
            cache_hit=result.cache_hit,
            request_error=result.request_error,
            visible_peer_count=0,
            payload=result.payload,
            assistant_text=str(result.response_payload.get("assistant_text") or ""),
            provider_reasoning_text=str(result.response_payload.get("provider_reasoning_text") or ""),
            validated_output=result.validated_output,
        ),
        normalized_answer="",
    )


def _execute_turn(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    method_name: str,
    method_type: str,
    diversity_mode: str,
    round_index: int,
    agent_id: int,
    role: str,
    visible_peer_count: int,
    persona_name: str,
    strategy_name: str,
    messages: list[dict[str, str]],
    backbone,
    provider,
    cache,
    limiter,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    seed: int,
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
        schema_id=SCHEMA_ANSWER_CORE,
        dataset=dataset,
    )
    final_answer = str(result.validated_output.get("final_answer") or "")
    normalized = normalize_prediction(dataset, final_answer) if final_answer else ""
    validated_output = dict(result.validated_output)
    return _serialize_turn_record(
        AgentTurnRecord(
            run_id=run_id,
            dataset=dataset,
            split=split_name,
            sample_id=sample.sample_id,
            method_name=method_name,
            method_type=method_type,
            diversity_mode=diversity_mode,
            round_index=round_index,
            agent_id=agent_id,
            role=role,
            persona_name=persona_name,
            strategy_name=strategy_name,
            prompt_hash=result.prompt_hash,
            prediction=normalized,
            output_status=result.output_status,
            prompt_tokens=float(result.usage.get("prompt_tokens") or 0.0),
            completion_tokens=float(result.usage.get("completion_tokens") or 0.0),
            total_tokens=float(result.usage.get("total_tokens") or 0.0),
            latency_ms=float(result.response_payload.get("latency_ms") or 0.0),
            cache_hit=result.cache_hit,
            request_error=result.request_error,
            visible_peer_count=visible_peer_count,
            payload=result.payload,
            assistant_text=str(result.response_payload.get("assistant_text") or ""),
            provider_reasoning_text=str(result.response_payload.get("provider_reasoning_text") or ""),
            validated_output=validated_output,
            **_turn_artifact_fields(validated_output),
        ),
        normalized_answer=normalized,
    )


def _validate_reasoning_process_output(assistant_text: str, provider_reasoning_text: str) -> dict[str, Any]:
    text = str(assistant_text or "").strip() or str(provider_reasoning_text or "").strip()
    if not text:
        raise ValueError("Reasoning process output is empty.")
    return {"reasoning": text}


def _validate_feedback_output(assistant_text: str, provider_reasoning_text: str) -> dict[str, Any]:
    text = str(assistant_text or "").strip() or str(provider_reasoning_text or "").strip()
    if not text:
        raise ValueError("Reflection feedback output is empty.")
    return {"feedback": text}


def _validate_method_selection_output(
    assistant_text: str,
    provider_reasoning_text: str,
    *,
    candidate_methods: list[str],
) -> dict[str, Any]:
    text = str(assistant_text or "").strip() or str(provider_reasoning_text or "").strip()
    if not text:
        raise ValueError("Method selection output is empty.")
    payload: dict[str, Any] = {}
    try:
        maybe_json = json.loads(text)
        if isinstance(maybe_json, dict):
            payload = maybe_json
    except Exception:
        payload = {}
    selected_method = str(payload.get("selected_method") or payload.get("method") or "").strip().lower()
    if selected_method not in candidate_methods:
        for method_name in candidate_methods:
            if method_name in text.lower():
                selected_method = method_name
                break
    if selected_method not in candidate_methods:
        raise ValueError("Method selection output did not identify a supported reasoning method.")
    return {
        "selected_method": selected_method,
        "reasoning": str(payload.get("reasoning") or payload.get("rationale") or "").strip(),
    }
