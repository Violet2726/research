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
    build_reasoning_stage_messages,
    build_reflection_feedback_messages,
    build_reflection_revision_messages,
    build_solution_selector_messages,
)
from research_experiments.families.shared.common import resolve_phase_split_name, safe_mean, safe_ratio


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


@dataclass(frozen=True)
class SelectionDecision:
    final_answer: str
    selected_agent_id: int | None
    rationale: str
    fallback_used: bool
    fallback_reason: str | None


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
) -> tuple[int, int]:
    total_calls = 0
    total_predictions = 0
    for benchmark in benchmarks:
        split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
        sample_count = len(load_split_ids(benchmark.cache_namespace or benchmark.slug, split_name))
        total_predictions += sample_count * len(methods)
        for method in methods:
            total_calls += sample_count * _calls_per_question(method, protocol, rosters.get(method.name), controls)
    return total_calls, total_predictions


def _resolve_split_name(experiment: DmadExperimentConfig, phase_name: str, benchmark_slug: str) -> str:
    return resolve_phase_split_name(experiment, phase_name, benchmark_slug)


def _load_selected_samples(benchmark, split_name: str) -> list[DatasetSample]:
    return select_samples(benchmark, split_name)


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
        if method.mode == "single_agent_cot":
            method_turns, method_prediction = _run_single_agent_cot(
                sample=sample,
                run_id=run_id,
                benchmark_slug=benchmark_slug,
                split_name=split_name,
                method=method,
                backbone=backbone,
                provider=provider,
                cache=cache,
                limiter=limiter,
                cot_method=_require_control_method(controls, "cot_1"),
                experiment=experiment,
                protocol=protocol,
                seed_offset=method_index * 10_000,
            )
            turn_rows.extend(method_turns)
            prediction_rows.append(method_prediction)
            continue
        if method.mode == "single_agent_reflection_r1":
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
                cot_method=_require_control_method(controls, "cot_1"),
                experiment=experiment,
                protocol=protocol,
                seed_offset=method_index * 10_000,
            )
            turn_rows.extend(method_turns)
            prediction_rows.append(method_prediction)
            continue
        if method.mode == "mv_6":
            method_turns, method_prediction = _run_vote_control(
                sample=sample,
                run_id=run_id,
                benchmark_slug=benchmark_slug,
                split_name=split_name,
                method=method,
                backbone=backbone,
                provider=provider,
                cache=cache,
                limiter=limiter,
                vote_method=_require_control_method(controls, "mv_6"),
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


def _run_single_agent_cot(
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
    cot_method,
    experiment: DmadExperimentConfig,
    protocol: ProtocolConfig,
    seed_offset: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    profile = _default_profile(agent_id=1)
    turn = _execute_turn(
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
        strategy_name=profile.strategy_name,
        messages=build_initial_messages(sample, profile, prompt_version=experiment.prompt_version),
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        temperature=float(cot_method.temperature),
        top_p=float(cot_method.top_p),
        max_output_tokens=int(cot_method.max_output_tokens),
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
        strategy_names=[profile.strategy_name],
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
    cot_method,
    experiment: DmadExperimentConfig,
    protocol: ProtocolConfig,
    seed_offset: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    profile = _default_profile(agent_id=1)
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
        strategy_name=profile.strategy_name,
        messages=build_initial_messages(sample, profile, prompt_version=experiment.prompt_version),
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        temperature=float(cot_method.temperature),
        top_p=float(cot_method.top_p),
        max_output_tokens=int(cot_method.max_output_tokens),
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
        strategy_name=profile.strategy_name,
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
        temperature=float(cot_method.temperature),
        top_p=float(cot_method.top_p),
        max_output_tokens=min(160, int(cot_method.max_output_tokens)),
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
        strategy_name=profile.strategy_name,
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
        temperature=float(cot_method.temperature),
        top_p=float(cot_method.top_p),
        max_output_tokens=int(cot_method.max_output_tokens),
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
        strategy_names=[profile.strategy_name],
        persona_names=[profile.persona_name],
        calls_per_question=3,
        debate_rounds=0,
        agent_count=1,
    )
    return [initial_turn, feedback_turn, reflection_turn], prediction


def _run_vote_control(
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
    vote_method,
    experiment: DmadExperimentConfig,
    protocol: ProtocolConfig,
    seed_offset: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    profiles = [_default_profile(agent_id=index + 1) for index in range(6)]
    turns: list[dict[str, Any]] = []
    for index, profile in enumerate(profiles):
        turns.append(
            _execute_turn(
                run_id=run_id,
                dataset=benchmark_slug,
                split_name=split_name,
                sample=sample,
                method_name=method.name,
                method_type="vote",
                diversity_mode="vote",
                round_index=0,
                role="control",
                visible_peer_count=0,
                persona_name=profile.persona_name,
                strategy_name=profile.strategy_name,
                messages=build_initial_messages(sample, profile, prompt_version=experiment.prompt_version),
                backbone=backbone,
                provider=provider,
                cache=cache,
                limiter=limiter,
                temperature=float(vote_method.temperature),
                top_p=float(vote_method.top_p),
                max_output_tokens=int(vote_method.max_output_tokens),
                seed=experiment.global_seed + seed_offset + index,
                agent_id=profile.agent_id,
            )
        )
    prediction = _build_final_prediction(
        run_id=run_id,
        dataset=benchmark_slug,
        split_name=split_name,
        sample=sample,
        method_name=method.name,
        method_type="vote",
        diversity_mode="vote",
        model_name=backbone.name,
        all_turns=turns,
        initial_rows=turns,
        final_rows=turns,
        debate_rows=[],
        strategy_names=[profile.strategy_name for profile in profiles],
        persona_names=[profile.persona_name for profile in profiles],
        calls_per_question=6,
        debate_rounds=0,
        agent_count=6,
    )
    return turns, prediction


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
                strategy_name=profile.strategy_name,
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
                strategy_name=profile.strategy_name,
                messages=build_answer_stage_messages(
                    sample,
                    profile,
                    round_index=round_index,
                    solving_process=solving_process,
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
                    strategy_name=profile.strategy_name,
                    process_row=process_row,
                    answer_row=answer_row,
                    solving_process=solving_process,
                    final_answer=str(answer_row["validated_output"].get("final_answer", "")).strip(),
                )
            )
            all_turns.extend([process_row, answer_row])

        round_records = [
            {
                "round_index": str(state.round_index),
                "agent_id": str(state.agent_id),
                "persona_name": state.persona_name,
                "strategy_name": state.strategy_name,
                "reasoning": state.solving_process,
                "answer": state.final_answer,
            }
            for state in current_round
        ]
        prior_rounds.append(round_records)
        answer_rows = [state.answer_row for state in current_round]
        if round_index == 1:
            initial_turns = answer_rows
        final_turns = answer_rows
        final_states = current_round

    selector_turn = _execute_selector_turn(
        run_id=run_id,
        dataset=benchmark_slug,
        split_name=split_name,
        sample=sample,
        method_name=method.name,
        method_type="mad",
        diversity_mode=roster.diversity_mode,
        round_index=protocol.debate_rounds + 1,
        role="selector",
        visible_peer_count=roster.agent_count,
        persona_name="solution_selector",
        strategy_name="best_solution_selection",
        messages=build_solution_selector_messages(
            sample,
            [
                {
                    "agent_id": state.agent_id,
                    "persona_name": state.persona_name,
                    "strategy_name": state.strategy_name,
                    "reasoning": state.solving_process,
                    "answer": state.final_answer,
                }
                for state in final_states
            ],
            prompt_version=experiment.prompt_version,
        ),
        candidate_answers={state.agent_id: state.final_answer for state in final_states},
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        temperature=0.0,
        top_p=protocol.top_p,
        max_output_tokens=min(192, protocol.max_output_tokens),
        seed=experiment.global_seed + seed_offset + 9_999,
        agent_id=0,
    )
    all_turns.append(selector_turn)
    selection = _build_selection_decision(selector_turn, final_turns)
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
        strategy_names=[profile.strategy_name for profile in roster.agents],
        persona_names=[profile.persona_name for profile in roster.agents],
        calls_per_question=len(roster.agents) * protocol.debate_rounds * 2 + 1,
        debate_rounds=protocol.debate_rounds,
        agent_count=roster.agent_count,
        selected_prediction=selection.final_answer,
        selected_agent_id=selection.selected_agent_id,
        selection_rationale=selection.rationale,
        selection_fallback_used=selection.fallback_used,
        selection_fallback_reason=selection.fallback_reason,
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
    strategy_names: list[str],
    persona_names: list[str],
    calls_per_question: int,
    debate_rounds: int,
    agent_count: int,
    selected_prediction: str | None = None,
    selected_agent_id: int | None = None,
    selection_rationale: str = "",
    selection_fallback_used: bool = False,
    selection_fallback_reason: str | None = None,
) -> dict[str, Any]:
    initial_answers = [str(row["normalized_answer"]) for row in initial_rows]
    final_answers = [str(row["normalized_answer"]) for row in final_rows]
    initial_vote, initial_vote_counts = aggregate_majority(initial_answers)
    final_vote, final_vote_counts = aggregate_majority(final_answers)
    initial_vote_score = float(score_prediction(dataset, initial_vote, sample.reference_answer))
    final_vote_score = float(score_prediction(dataset, final_vote, sample.reference_answer))
    resolved_prediction = selected_prediction or final_vote
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
            strategy_names=list(strategy_names),
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
    payload["selection_mode"] = "best_solution_selector" if selected_prediction is not None else "final_majority_vote"
    payload["selector_selected_agent_id"] = selected_agent_id
    payload["selector_rationale"] = selection_rationale
    payload["selector_fallback_used"] = selection_fallback_used
    payload["selector_fallback_reason"] = selection_fallback_reason
    payload["selected_answer_matches_final_vote"] = resolved_prediction == final_vote
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


def _build_strategy_diagnostics(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
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
    dmad_row = overall_lookup.get(("overall", "dmad_strategy_diverse_r1"))
    vanilla_row = overall_lookup.get(("overall", "vanilla_mad_r1"))
    gate_passed = bool(
        dmad_row
        and vanilla_row
        and float(dmad_row.get("accuracy_mean") or 0.0) >= float(vanilla_row.get("accuracy_mean") or 0.0)
        and float(dmad_row.get("calls_per_question_mean") or 0.0) <= float(vanilla_row.get("calls_per_question_mean") or 0.0)
    )
    return {
        "rows": rows,
        "gate_summary": {
            "count100_upgrade_gate_passed": gate_passed,
            "dmad_accuracy_mean": None if dmad_row is None else float(dmad_row.get("accuracy_mean") or 0.0),
            "vanilla_accuracy_mean": None if vanilla_row is None else float(vanilla_row.get("accuracy_mean") or 0.0),
            "dmad_calls_per_question_mean": None if dmad_row is None else float(dmad_row.get("calls_per_question_mean") or 0.0),
            "vanilla_calls_per_question_mean": None if vanilla_row is None else float(vanilla_row.get("calls_per_question_mean") or 0.0),
        },
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
                "selector_tokens": 0.0,
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
        elif role == "selector":
            bucket["selector_tokens"] = float(bucket["selector_tokens"]) + total_tokens
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


def _build_summary_row(*, dataset: str, method_name: str, rows: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    accuracy_mean = safe_mean(float(row["score"]) for row in rows)
    total_tokens_mean = safe_mean(float(row["total_tokens_per_question"]) for row in rows)
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
        "strategy_name": _joined_unique_labels(rows, "strategy_names"),
        "initial_disagreement_rate": safe_ratio(sum(1 for row in rows if row["initial_disagreement"]), len(rows)),
        "final_consensus_rate": safe_ratio(sum(1 for row in rows if row["final_consensus"]), len(rows)),
        "changed_after_debate_rate": safe_ratio(sum(1 for row in rows if row["changed_after_debate"]), len(rows)),
        "correction_rate": safe_ratio(sum(1 for row in rows if row["corrected_by_debate"]), len(rows)),
        "degradation_rate": safe_ratio(sum(1 for row in rows if row["harmed_by_debate"]), len(rows)),
        "gain_over_vanilla_mad": None,
        "gain_over_persona_diverse": None,
    }


def _build_strategy_row(*, dataset: str, method_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "method_name": method_name,
        "diversity_mode": rows[0]["diversity_mode"],
        "strategy_name": _joined_unique_labels(rows, "strategy_names"),
        "question_count": len(rows),
        "initial_disagreement_rate": safe_ratio(sum(1 for row in rows if row["initial_disagreement"]), len(rows)),
        "final_consensus_rate": safe_ratio(sum(1 for row in rows if row["final_consensus"]), len(rows)),
        "changed_after_debate_rate": safe_ratio(sum(1 for row in rows if row["changed_after_debate"]), len(rows)),
        "correction_rate": safe_ratio(sum(1 for row in rows if row["corrected_by_debate"]), len(rows)),
        "degradation_rate": safe_ratio(sum(1 for row in rows if row["harmed_by_debate"]), len(rows)),
        "accuracy_mean": safe_mean(float(row["score"]) for row in rows),
        "calls_per_question_mean": safe_mean(float(row["calls_per_question"]) for row in rows),
        "gain_over_vanilla_mad": None,
        "gain_over_persona_diverse": None,
    }


def _annotate_gain_rows(rows: list[dict[str, Any]]) -> None:
    lookup = {(str(row["dataset"]), str(row["method_name"])): row for row in rows}
    for row in rows:
        dataset = str(row["dataset"])
        vanilla = lookup.get((dataset, "vanilla_mad_r1"))
        persona = lookup.get((dataset, "persona_diverse_mad_r1"))
        row["gain_over_vanilla_mad"] = (
            round(float(row["accuracy_mean"]) - float(vanilla["accuracy_mean"]), 6)
            if vanilla is not None and str(row["method_name"]) != "vanilla_mad_r1"
            else None
        )
        row["gain_over_persona_diverse"] = (
            round(float(row["accuracy_mean"]) - float(persona["accuracy_mean"]), 6)
            if persona is not None and str(row["method_name"]) != "persona_diverse_mad_r1"
            else None
        )


def _joined_unique_labels(rows: list[dict[str, Any]], key: str) -> str:
    labels = sorted({str(item) for row in rows for item in row.get(key, [])})
    return ", ".join(labels)


def _default_profile(agent_id: int):
    from research_experiments.families.dmad.config import AgentProfile

    return AgentProfile(
        agent_id=agent_id,
        persona_name="general_reasoner",
        persona_instruction="Act as a careful general-purpose reasoning assistant.",
        strategy_name="cot",
        strategy_instruction="Think step by step and make every inference explicit before committing to the final answer.",
    )


def _calls_per_question(
    method: DmadMethodSpec,
    protocol: ProtocolConfig,
    roster: RosterConfig | None,
    controls: dict[str, Any],
) -> int:
    if method.mode == "single_agent_cot":
        return int(_require_control_method(controls, "cot_1").budget_calls)
    if method.mode == "single_agent_reflection_r1":
        return 3
    if method.mode == "mv_6":
        return int(_require_control_method(controls, "mv_6").budget_calls)
    if roster is None:
        raise RuntimeError(f"DMAD method {method.name} is missing roster metadata.")
    return roster.agent_count * protocol.debate_rounds * 2 + 1


def _require_control_method(controls: dict[str, Any], name: str):
    method = controls.get(name)
    if method is None:
        raise RuntimeError(f"DMAD baseline requires control method `{name}` to be defined in control_catalog.")
    return method


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
        validator=_validate_reasoning_process_output,
        dataset=dataset,
        use_response_format=False,
    )
    reasoning_text = str(result.validated_output.get("reasoning") or "")
    return asdict(
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
            validated_output={"reasoning": reasoning_text},
        )
    ) | {"normalized_answer": "", "artifact_version": ARTIFACT_VERSION}


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
    return asdict(
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
        )
    ) | {"normalized_answer": "", "artifact_version": ARTIFACT_VERSION}


def _execute_selector_turn(
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
    candidate_answers: dict[int, str],
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
        validator=lambda assistant_text, provider_reasoning_text: _validate_selection_output(
            assistant_text,
            provider_reasoning_text,
            candidate_answers=candidate_answers,
        ),
        dataset=dataset,
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
            validated_output=result.validated_output,
        )
    ) | {"normalized_answer": normalized, "artifact_version": ARTIFACT_VERSION}


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
    return asdict(
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
            validated_output=result.validated_output,
        )
    ) | {"normalized_answer": normalized, "artifact_version": ARTIFACT_VERSION}


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


def _validate_selection_output(
    assistant_text: str,
    provider_reasoning_text: str,
    *,
    candidate_answers: dict[int, str],
) -> dict[str, Any]:
    text = str(assistant_text or "").strip() or str(provider_reasoning_text or "").strip()
    if not text:
        raise ValueError("Selection output is empty.")
    payload: dict[str, Any] = {}
    try:
        maybe_json = json.loads(text)
        if isinstance(maybe_json, dict):
            payload = maybe_json
    except Exception:
        payload = {}
    selected_agent_id_raw = (
        payload.get("selected_agent_id")
        or payload.get("best_agent_id")
        or payload.get("Index")
        or payload.get("index")
    )
    selected_agent_id: int | None
    try:
        selected_agent_id = int(str(selected_agent_id_raw).strip()) if selected_agent_id_raw is not None else None
    except Exception:
        selected_agent_id = None
    final_answer = str(
        payload.get("final_answer")
        or payload.get("answer")
        or ""
    ).strip()
    if not final_answer and selected_agent_id in candidate_answers:
        final_answer = str(candidate_answers[selected_agent_id]).strip()
    if not final_answer and selected_agent_id == 4:
        return {
            "selected_agent_id": 4,
            "final_answer": "",
            "rationale": str(payload.get("rationale") or payload.get("Reason") or "selector_abstained"),
        }
    if not final_answer:
        for agent_id, answer in candidate_answers.items():
            if answer and answer in text:
                selected_agent_id = agent_id
                final_answer = answer
                break
    if not final_answer:
        raise ValueError("Selection output did not identify a supported final answer.")
    return {
        "selected_agent_id": selected_agent_id,
        "final_answer": final_answer,
        "rationale": str(payload.get("rationale") or payload.get("Reason") or "").strip(),
    }


def _build_selection_decision(selector_turn: dict[str, Any], final_rows: list[dict[str, Any]]) -> SelectionDecision:
    final_answers = [str(row["normalized_answer"]) for row in final_rows]
    final_vote, _ = aggregate_majority(final_answers)
    selected_agent_id = selector_turn.get("validated_output", {}).get("selected_agent_id")
    final_answer = str(selector_turn.get("validated_output", {}).get("final_answer") or "").strip()
    if selector_turn.get("output_status") != "ok" or not final_answer or int(selected_agent_id or 0) == 4:
        return SelectionDecision(
            final_answer=final_vote,
            selected_agent_id=_first_agent_with_answer(final_rows, final_vote),
            rationale="fallback_to_final_majority",
            fallback_used=True,
            fallback_reason="selector_unavailable_or_abstained",
        )
    return SelectionDecision(
        final_answer=final_answer,
        selected_agent_id=int(selected_agent_id) if selected_agent_id is not None else None,
        rationale=str(selector_turn.get("validated_output", {}).get("rationale") or "").strip(),
        fallback_used=False,
        fallback_reason=None,
    )


def _first_agent_with_answer(rows: list[dict[str, Any]], answer: str) -> int | None:
    for row in rows:
        if str(row.get("normalized_answer") or "") == answer:
            return int(row["agent_id"])
    return None
