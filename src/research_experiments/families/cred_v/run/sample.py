"""CRED-V 样本级执行辅助。"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any

from research_experiments.core.data.datasets import DatasetSample, load_split_ids, select_samples
from research_experiments.core.data.evaluation import normalize_prediction, score_prediction
from research_experiments.core.execution.runner_common import iter_indexed_batch
from research_experiments.families.cred_v.algorithms import (
    aggregate_adaptive_candidate_search,
    aggregate_evidence_repair_v5,
    aggregate_pairwise_selection,
    aggregate_repair_only_v6,
    aggregate_reasoning_first_selection,
    aggregate_safe_verification,
    aggregate_safe_select_v3,
    aggregate_shadow_evidence_select_v7,
    aggregate_shadow_select_v4,
    aggregate_stage_a_vote,
    aggregate_task_verification,
    answer_family_key,
    build_router_decision,
    choice_permutation,
    evidence_quality,
    expansion_mode_for_dataset,
    map_shuffled_choice_answer,
    select_verification_targets,
)
from research_experiments.families.cred_v.config import (
    CRED_ACS_METHODS,
    CRED_COMM_METHODS,
    CRED_RFS_ADAPTIVE_METHODS,
    CRED_RFS_EVIDENCE_REPAIR_METHODS,
    CRED_RFS_PAIRWISE_METHODS,
    CRED_RFS_PAIRWISE_SELECT_V2,
    CRED_RFS_REPAIR_ONLY_METHODS,
    CRED_RFS_SAFE_SELECT_METHODS,
    CRED_RFS_SHADOW_EVIDENCE_METHODS,
    CRED_RFS_SHADOW_SELECT_METHODS,
    CRED_SAFE_VERIFY_METHODS,
    CRED_VERIFY_METHODS,
    CredVExperimentConfig,
    CredVProtocolConfig,
)
from research_experiments.families.cred_v.prompts import (
    AGENT_ROLES,
    build_mc_blind_pairwise_duel_messages,
    build_mc_shadow_evidence_select_messages,
    build_choice_shuffle_solver_messages,
    build_hotpot_span_extractor_messages,
    build_math_symbolic_repair_messages,
    build_rfs_extra_solver_messages,
    build_safe_hetero_verifier_messages,
    build_stage_a_messages,
    build_strategyqa_minority_resample_messages,
    build_strategyqa_dual_polarity_messages,
    build_task_verifier_messages,
)
from research_experiments.family_runtime.common import resolve_phase_split_name, safe_mean, safe_ratio
from research_experiments.family_runtime.output_protocols import (
    FREE_TEXT_ANSWER_PROTOCOL_V1,
    execute_output_protocol_turn,
)

_PAIRWISE_DUEL_MODES = {
    "mc_blind_pairwise_duel",
    "gpqa_unanimous_pairwise_duel",
    "gpqa_2of3_retry_shadow",
    "mmlu_unanimous_pairwise_shadow",
    "direct_option_contrast_shadow",
    "constraint_elimination_shadow",
    "minimal_evidence_certificate_shadow",
}

_V7_SHADOW_EVIDENCE_MODES = (
    "direct_option_contrast_shadow",
    "constraint_elimination_shadow",
    "minimal_evidence_certificate_shadow",
)


@dataclass(frozen=True)
class CredTurnRecord:
    run_id: str
    dataset: str
    split: str
    sample_id: str
    method_name: str
    method_type: str
    round_index: int
    agent_id: int
    role: str
    agent_role: str
    prompt_hash: str
    prediction: str
    normalized_answer: str
    score: float | None
    output_status: str
    prompt_tokens: float
    completion_tokens: float
    total_tokens: float
    latency_ms: float
    cache_hit: bool
    request_error: str | None
    request_status: str
    raw_finish_reason: str | None
    output_protocol: str
    protocol_parse_status: str
    protocol_parse_error: str | None
    protocol_recovery: str
    reason_present: bool
    request_count: int
    cache_request_count: int
    network_request_count: int
    raw_prompt_tokens: float
    raw_completion_tokens: float
    raw_total_tokens: float
    raw_latency_ms: float
    visible_peer_count: int
    confidence_value: float | None
    key_evidence: str
    risk_level: str
    failure_risk: str
    evidence_quality: float
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
    message_kind: str


@dataclass(frozen=True)
class CredVerifierRuntime:
    model_ref: str
    backbone: Any
    provider: Any
    cache: Any
    throttle: Any


@dataclass(frozen=True)
class CredPredictionRecord:
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
    corrected_by_verification: bool
    harmed_by_verification: bool
    unchanged_correct: bool
    unchanged_wrong: bool
    triggered: bool
    router_reasons: list[str]
    resolver: str
    survival_support: dict[str, float]
    verification_support: dict[str, float]
    oracle_candidate_correct: bool
    stage_candidate_oracle_correct: bool
    wrong_majority_some_correct: bool
    target_correct: bool | None
    safe_repair_applied: bool
    hetero_agreement_applied: bool
    expansion_call_count: int
    method_expansion_call_count: int
    expansion_mode: str
    false_consensus_triggered: bool
    candidate_pool_oracle_correct: bool
    expansion_oracle_correct: bool
    expansion_validation_pass_count: int
    math_repair_applied: bool
    hotpot_span_repair_applied: bool
    choice_shuffle_agreement_count: int
    single_pro_promotion_blocked: bool
    strong_majority_locked: bool
    pairwise_duel_count: int
    pairwise_duel_win_count: int
    safe_selector_corrected: bool
    safe_selector_harmed: bool
    repair_only_corrected: bool
    repair_only_harmed: bool
    semantic_selector_corrected: bool
    semantic_selector_harmed: bool
    gpqa_unanimous_duel_count: int
    blocked_2of3_pairwise_count: int
    blocked_mmlu_pairwise_count: int
    blocked_strategyqa_probe_count: int
    shadow_counterfactual_answer: str
    shadow_counterfactual_resolver: str
    shadow_counterfactual_corrected: bool
    shadow_counterfactual_harmed: bool
    shadow_gate_passed: bool
    shadow_net_gain: int
    shadow_cross_view_agreement_count: int
    duel_invalid_count: int
    duel_retry_recoverable_count: int
    minority_probe_count: int
    non_answer_candidate_blocked: bool
    false_consensus_recovered: bool
    free_text_recovered_count: int
    pairwise_json_recovered_count: int
    json_truncated_count: int
    protocol_failures_per_question: int
    reason_missing_turns_per_question: int


def run_cred_batch(
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    samples: list[DatasetSample],
    experiment: CredVExperimentConfig,
    protocol: CredVProtocolConfig,
    backbone,
    provider,
    cache,
    throttle,
    verifier_runtimes: list[CredVerifierRuntime] | None = None,
) -> Iterator[tuple[int, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]]:
    worker = partial(
        _run_cred_sample,
        run_id=run_id,
        benchmark_slug=benchmark_slug,
        split_name=split_name,
        experiment=experiment,
        protocol=protocol,
        backbone=backbone,
        provider=provider,
        cache=cache,
        throttle=throttle,
        verifier_runtimes=verifier_runtimes or [],
    )
    for sample_index, result in iter_indexed_batch(
        samples,
        worker=worker,
        max_concurrent_requests=experiment.max_concurrent_requests,
    ):
        yield (sample_index, *result)


def append_outputs(
    *,
    sample_results: Iterable[tuple[int, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]],
    dataset_slug: str,
    progress,
    turn_writer,
    debate_writer,
    router_writer,
    prediction_writer,
    all_turns: list[dict[str, Any]],
    all_debate_rows: list[dict[str, Any]],
    all_router_rows: list[dict[str, Any]],
    all_predictions: list[dict[str, Any]],
) -> None:
    for _, turn_rows, debate_rows, router_rows, prediction_rows in sample_results:
        for row in turn_rows:
            turn_writer.write_row(row)
            progress.record_call(row, method_key="method_name")
        for row in debate_rows:
            debate_writer.write_row(row)
        for row in router_rows:
            router_writer.write_row(row)
        for row in prediction_rows:
            prediction_writer.write_row(row)
            progress.record_predictions(1, dataset_slug, row["method_name"])
        all_turns.extend(turn_rows)
        all_debate_rows.extend(debate_rows)
        all_router_rows.extend(router_rows)
        all_predictions.extend(prediction_rows)


def _run_cred_sample(
    sample: DatasetSample,
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    experiment: CredVExperimentConfig,
    protocol: CredVProtocolConfig,
    backbone,
    provider,
    cache,
    throttle,
    verifier_runtimes: list[CredVerifierRuntime],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    stage_rows: list[dict[str, Any]] = []
    for index, agent_role in enumerate(AGENT_ROLES[: protocol.stage_a_agent_count], start=1):
        stage_rows.append(
            _execute_turn(
                run_id=run_id,
                dataset=benchmark_slug,
                split_name=split_name,
                sample=sample,
                method_name="cred_stage_a",
                method_type="mad",
                round_index=0,
                agent_id=index,
                role="initial",
                agent_role=agent_role,
                visible_peer_count=0,
                messages=build_stage_a_messages(
                    sample,
                    agent_id=index,
                    agent_role=agent_role,
                    prompt_mode=protocol.stage_a_prompt_mode,
                    output_protocol=experiment.cred_stage_a_output_protocol,
                ),
                backbone=backbone,
                provider=provider,
                cache=cache,
                throttle=throttle,
                temperature=protocol.stage_a_temperature,
                top_p=protocol.top_p,
                seed=experiment.global_seed + index - 1,
                output_protocol=experiment.cred_stage_a_output_protocol,
                max_tokens=_positive_token_cap(protocol.stage_a_max_tokens),
            )
        )

    vote_decision = aggregate_stage_a_vote(stage_rows)
    router = build_router_decision(stage_rows, protocol=protocol)
    method_set = set(experiment.cred_methods)
    verification_rows: list[dict[str, Any]] = []
    safe_verifier_rows: list[dict[str, Any]] = []
    expansion_rows: list[dict[str, Any]] = []
    debate_rows: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []

    if router.triggered and CRED_COMM_METHODS & method_set:
        targets = select_verification_targets(
            dataset=benchmark_slug,
            rows=stage_rows,
            leading_answer=vote_decision.final_answer,
            max_verifications=protocol.max_verifications,
        )
    if router.triggered and CRED_VERIFY_METHODS & method_set:
        for verification_index, target in enumerate(targets, start=1):
            verifier_row = _execute_turn(
                run_id=run_id,
                dataset=benchmark_slug,
                split_name=split_name,
                sample=sample,
                method_name="cred_task_verifier",
                method_type="verification",
                round_index=1,
                agent_id=100 + verification_index,
                role="verification",
                agent_role="task_verifier",
                visible_peer_count=len(stage_rows),
                messages=build_task_verifier_messages(
                    sample,
                    leading_answer=vote_decision.final_answer,
                    target_row=target,
                    stage_rows=stage_rows,
                ),
                backbone=backbone,
                provider=provider,
                cache=cache,
                throttle=throttle,
                temperature=protocol.verifier_temperature,
                top_p=protocol.top_p,
                seed=experiment.global_seed + 100 + verification_index,
                output_protocol=experiment.cred_verification_output_protocol,
                max_tokens=_positive_token_cap(protocol.verifier_max_tokens),
            )
            verification_rows.append(verifier_row)
            debate_rows.append(_debate_message_row(run_id, benchmark_slug, split_name, sample, verifier_row, "task_verification"))

    if (
        router.triggered
        and CRED_SAFE_VERIFY_METHODS & method_set
        and "hetero_verified" in set(protocol.verification_modes)
        and verifier_runtimes
    ):
        safe_runtimes = [
            runtime
            for runtime in verifier_runtimes
            if protocol.allow_same_model_promotion or str(runtime.backbone.name) != str(backbone.name)
        ]
        call_count = min(int(protocol.max_verification_calls), len(targets))
        call_count = min(call_count, len(safe_runtimes))
        for verification_index in range(1, call_count + 1):
            target = targets[verification_index - 1]
            runtime = safe_runtimes[(verification_index - 1) % len(safe_runtimes)]
            verifier_row = _execute_turn(
                run_id=run_id,
                dataset=benchmark_slug,
                split_name=split_name,
                sample=sample,
                method_name="cred_safe_hetero_verifier",
                method_type="verification",
                round_index=1,
                agent_id=400 + verification_index,
                role="verification",
                agent_role=f"safe_hetero_verifier:{runtime.model_ref}",
                visible_peer_count=len(stage_rows),
                messages=build_safe_hetero_verifier_messages(
                    sample,
                    leading_answer=vote_decision.final_answer,
                    target_row=target,
                    stage_rows=stage_rows,
                ),
                backbone=runtime.backbone,
                provider=runtime.provider,
                cache=runtime.cache,
                throttle=runtime.throttle,
                temperature=protocol.verifier_temperature,
                top_p=protocol.top_p,
                seed=experiment.global_seed + 400 + verification_index,
                output_protocol=experiment.cred_verification_output_protocol,
                max_tokens=_positive_token_cap(protocol.verifier_max_tokens),
            )
            safe_verifier_rows.append(verifier_row)
            debate_rows.append(_debate_message_row(run_id, benchmark_slug, split_name, sample, verifier_row, "safe_hetero_verification"))

    if router.triggered and CRED_ACS_METHODS & method_set and verifier_runtimes and int(protocol.max_expansion_calls) > 0:
        expansion_rows.extend(
            _run_acs_expansions(
                run_id=run_id,
                benchmark_slug=benchmark_slug,
                split_name=split_name,
                sample=sample,
                experiment=experiment,
                protocol=protocol,
                stage_rows=stage_rows,
                vote_decision=vote_decision,
                verifier_runtimes=verifier_runtimes,
            )
        )
        for row in expansion_rows:
            debate_rows.append(_debate_message_row(run_id, benchmark_slug, split_name, sample, row, "acs_expansion"))

    if router.triggered and CRED_RFS_ADAPTIVE_METHODS & method_set:
        rfs_rows = _run_rfs_expansions(
            run_id=run_id,
            benchmark_slug=benchmark_slug,
            split_name=split_name,
            sample=sample,
            experiment=experiment,
            protocol=protocol,
            stage_rows=stage_rows,
            vote_decision=vote_decision,
            backbone=backbone,
            provider=provider,
            cache=cache,
            throttle=throttle,
            verifier_runtimes=verifier_runtimes,
            router_bucket=router.trigger_bucket,
        )
        expansion_rows.extend(rfs_rows)
        for row in rfs_rows:
            debate_rows.append(_debate_message_row(run_id, benchmark_slug, split_name, sample, row, "rfs_expansion"))

    if router.triggered and CRED_RFS_PAIRWISE_METHODS & method_set:
        pairwise_rows = _run_rfs_pairwise_selection(
            run_id=run_id,
            benchmark_slug=benchmark_slug,
            split_name=split_name,
            sample=sample,
            experiment=experiment,
            protocol=protocol,
            stage_rows=stage_rows,
            vote_decision=vote_decision,
            backbone=backbone,
            provider=provider,
            cache=cache,
            throttle=throttle,
            verifier_runtimes=verifier_runtimes,
            router_bucket=router.trigger_bucket,
            targets=targets,
        )
        expansion_rows.extend(pairwise_rows)
        for row in pairwise_rows:
            debate_rows.append(_debate_message_row(run_id, benchmark_slug, split_name, sample, row, "rfs_pairwise_selection"))

    all_turns = [*stage_rows, *verification_rows, *safe_verifier_rows, *expansion_rows]
    router_row = _router_row(
        run_id=run_id,
        dataset=benchmark_slug,
        split_name=split_name,
        sample=sample,
        router=router,
        self_verification_count=len(verification_rows),
        hetero_verification_count=len(safe_verifier_rows),
        expansion_count=len(expansion_rows),
    )
    prediction_rows: list[dict[str, Any]] = []
    for method_name in experiment.cred_methods:
        decision = vote_decision
        method_turns = list(stage_rows)
        debate_turns: list[dict[str, Any]] = []
        method_expansion_rows: list[dict[str, Any]] = []
        method_targets: list[dict[str, Any]] = []
        if method_name in CRED_VERIFY_METHODS:
            decision = aggregate_task_verification(
                dataset=benchmark_slug,
                stage_rows=stage_rows,
                verifier_rows=verification_rows,
                stage_winner=vote_decision.final_answer,
                promotion_confidence_min=protocol.promotion_confidence_min,
                promotion_score_margin=protocol.promotion_score_margin,
                concrete_evidence_min_chars=protocol.concrete_evidence_min_chars,
            )
            debate_turns = list(verification_rows)
            method_turns = [*stage_rows, *debate_turns]
            method_targets = list(targets)
        elif method_name in CRED_SAFE_VERIFY_METHODS:
            decision = aggregate_safe_verification(
                dataset=benchmark_slug,
                question=sample.question,
                context=sample.prompt_context,
                stage_rows=stage_rows,
                verifier_rows=verification_rows,
                hetero_verifier_rows=safe_verifier_rows,
                stage_winner=vote_decision.final_answer,
                verification_modes=protocol.verification_modes,
                allow_same_model_promotion=protocol.allow_same_model_promotion,
                concrete_evidence_min_chars=protocol.concrete_evidence_min_chars,
                strong_majority_count=protocol.strong_majority_count,
            )
            debate_turns = list(safe_verifier_rows)
            method_turns = [*stage_rows, *debate_turns]
            method_targets = list(targets)
        elif method_name in CRED_ACS_METHODS:
            decision = aggregate_adaptive_candidate_search(
                dataset=benchmark_slug,
                question=sample.question,
                context=sample.prompt_context,
                stage_rows=stage_rows,
                expansion_rows=expansion_rows,
                stage_winner=vote_decision.final_answer,
                expansion_modes=protocol.expansion_modes,
                promotion_min_independent_support=protocol.promotion_min_independent_support,
                promotion_margin_min=protocol.promotion_margin_min,
                strong_majority_count=protocol.strong_majority_count,
            )
            debate_turns = list(expansion_rows)
            method_turns = [*stage_rows, *debate_turns]
            method_expansion_rows = list(expansion_rows)
            method_targets = list(targets)
        elif method_name in CRED_RFS_ADAPTIVE_METHODS:
            decision = aggregate_reasoning_first_selection(
                dataset=benchmark_slug,
                question=sample.question,
                context=sample.prompt_context,
                stage_rows=stage_rows,
                expansion_rows=expansion_rows,
                stage_winner=vote_decision.final_answer,
                expansion_modes=protocol.expansion_modes,
                promotion_min_independent_support=protocol.promotion_min_independent_support,
                promotion_margin_min=protocol.promotion_margin_min,
                leader_lock_count=protocol.leader_lock_count,
                mc_shuffle_min_agreement=protocol.mc_shuffle_min_agreement,
                require_stage_a_challenger_support=protocol.require_stage_a_challenger_support,
            )
            debate_turns = list(expansion_rows)
            method_turns = [*stage_rows, *debate_turns]
            method_expansion_rows = list(expansion_rows)
            method_targets = list(targets)
        elif method_name in CRED_RFS_REPAIR_ONLY_METHODS:
            decision = aggregate_repair_only_v6(
                dataset=benchmark_slug,
                question=sample.question,
                context=sample.prompt_context,
                stage_rows=stage_rows,
                stage_winner=vote_decision.final_answer,
                selection_modes=protocol.selection_modes,
            )
            method_targets = list(targets)
        elif method_name in CRED_RFS_EVIDENCE_REPAIR_METHODS:
            actual_selection_rows = _safe_select_actual_rows(expansion_rows)
            decision = aggregate_evidence_repair_v5(
                dataset=benchmark_slug,
                question=sample.question,
                context=sample.prompt_context,
                stage_rows=stage_rows,
                selection_rows=actual_selection_rows,
                stage_winner=vote_decision.final_answer,
                selection_modes=protocol.selection_modes,
                leader_lock_count=protocol.leader_lock_count,
                pairwise_duel_replicates=protocol.pairwise_duel_replicates,
                pairwise_promotion_min_wins=protocol.pairwise_promotion_min_wins,
                pairwise_allowed_datasets=protocol.pairwise_allowed_datasets,
                pairwise_option_count_max=protocol.pairwise_option_count_max,
                option_count=len(_multiple_choice_options(sample)),
                require_stage_a_challenger_support=protocol.require_stage_a_challenger_support,
                allow_strong_majority_pairwise_promotion=protocol.allow_strong_majority_pairwise_promotion,
            )
            debate_turns = list(actual_selection_rows)
            method_turns = [*stage_rows, *debate_turns]
            method_expansion_rows = list(actual_selection_rows)
            method_targets = list(targets)
        elif method_name in CRED_RFS_SAFE_SELECT_METHODS:
            actual_selection_rows = _safe_select_actual_rows(expansion_rows)
            decision = aggregate_safe_select_v3(
                dataset=benchmark_slug,
                question=sample.question,
                context=sample.prompt_context,
                stage_rows=stage_rows,
                selection_rows=actual_selection_rows,
                stage_winner=vote_decision.final_answer,
                selection_modes=protocol.selection_modes,
                leader_lock_count=protocol.leader_lock_count,
                pairwise_duel_replicates=protocol.pairwise_duel_replicates,
                pairwise_promotion_min_wins=protocol.pairwise_promotion_min_wins,
                pairwise_allowed_datasets=protocol.pairwise_allowed_datasets,
                pairwise_option_count_max=protocol.pairwise_option_count_max,
                option_count=len(_multiple_choice_options(sample)),
                require_stage_a_challenger_support=protocol.require_stage_a_challenger_support,
                allow_strong_majority_pairwise_promotion=protocol.allow_strong_majority_pairwise_promotion,
            )
            debate_turns = list(actual_selection_rows)
            method_turns = [*stage_rows, *debate_turns]
            method_expansion_rows = list(actual_selection_rows)
            method_targets = list(targets)
        elif method_name in CRED_RFS_SHADOW_SELECT_METHODS:
            decision = aggregate_shadow_select_v4(
                dataset=benchmark_slug,
                question=sample.question,
                context=sample.prompt_context,
                stage_rows=stage_rows,
                selection_rows=expansion_rows,
                stage_winner=vote_decision.final_answer,
                selection_modes=protocol.selection_modes,
                leader_lock_count=protocol.leader_lock_count,
                pairwise_duel_replicates=protocol.pairwise_duel_replicates,
                pairwise_promotion_min_wins=protocol.pairwise_promotion_min_wins,
                pairwise_allowed_datasets=protocol.pairwise_allowed_datasets,
                pairwise_option_count_max=protocol.pairwise_option_count_max,
                option_count=len(_multiple_choice_options(sample)),
                require_stage_a_challenger_support=protocol.require_stage_a_challenger_support,
                allow_strong_majority_pairwise_promotion=protocol.allow_strong_majority_pairwise_promotion,
            )
            debate_turns = list(expansion_rows)
            method_turns = [*stage_rows, *debate_turns]
            method_expansion_rows = list(expansion_rows)
            method_targets = list(targets)
        elif method_name in CRED_RFS_SHADOW_EVIDENCE_METHODS:
            decision = aggregate_shadow_evidence_select_v7(
                dataset=benchmark_slug,
                question=sample.question,
                context=sample.prompt_context,
                stage_rows=stage_rows,
                selection_rows=expansion_rows,
                stage_winner=vote_decision.final_answer,
                selection_modes=protocol.selection_modes,
            )
            debate_turns = list(expansion_rows)
            method_turns = [*stage_rows, *debate_turns]
            method_expansion_rows = list(expansion_rows)
            method_targets = list(targets)
        elif method_name == CRED_RFS_PAIRWISE_SELECT_V2:
            decision = aggregate_pairwise_selection(
                dataset=benchmark_slug,
                question=sample.question,
                context=sample.prompt_context,
                stage_rows=stage_rows,
                selection_rows=expansion_rows,
                stage_winner=vote_decision.final_answer,
                selection_modes=protocol.selection_modes,
                leader_lock_count=protocol.leader_lock_count,
                pairwise_duel_replicates=protocol.pairwise_duel_replicates,
                pairwise_promotion_min_wins=protocol.pairwise_promotion_min_wins,
                require_stage_a_challenger_support=protocol.require_stage_a_challenger_support,
            )
            debate_turns = list(expansion_rows)
            method_turns = [*stage_rows, *debate_turns]
            method_expansion_rows = list(expansion_rows)
            method_targets = list(targets)
        prediction_rows.append(
            _prediction_row(
                run_id=run_id,
                dataset=benchmark_slug,
                split_name=split_name,
                sample=sample,
                method_name=method_name,
                backbone_name=backbone.name,
                decision=decision,
                stage_decision=vote_decision,
                router=router,
                stage_rows=stage_rows,
                debate_rows=debate_turns,
                method_turns=method_turns,
                targets=method_targets,
                expansion_rows=method_expansion_rows,
                protocol=protocol,
            )
        )
    return all_turns, debate_rows, [router_row], prediction_rows


def _run_acs_expansions(
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    sample: DatasetSample,
    experiment: CredVExperimentConfig,
    protocol: CredVProtocolConfig,
    stage_rows: list[dict[str, Any]],
    vote_decision,
    verifier_runtimes: list[CredVerifierRuntime],
) -> list[dict[str, Any]]:
    mode = expansion_mode_for_dataset(benchmark_slug, protocol.expansion_modes)
    if not mode:
        return []
    allowed_refs = set(protocol.expansion_model_refs) or set(experiment.verifier_model_refs)
    runtimes = [runtime for runtime in verifier_runtimes if not allowed_refs or runtime.model_ref in allowed_refs]
    if not runtimes:
        return []
    max_calls = int(protocol.max_expansion_calls)
    if mode in {"math_symbolic_repair", "hotpot_span_extract"}:
        max_calls = min(max_calls, 1)
    elif mode == "strategyqa_dual_polarity":
        max_calls = min(max_calls, 3)
    elif mode == "mc_choice_shuffle":
        max_calls = min(max_calls, 3)
        if not _multiple_choice_options(sample):
            return []
    rows: list[dict[str, Any]] = []
    for index in range(1, max_calls + 1):
        runtime = runtimes[(index - 1) % len(runtimes)]
        permutation: list[int] = []
        shuffled_options: list[str] = []
        if mode == "math_symbolic_repair":
            messages = build_math_symbolic_repair_messages(
                sample,
                leading_answer=vote_decision.final_answer,
                stage_rows=stage_rows,
            )
        elif mode == "hotpot_span_extract":
            messages = build_hotpot_span_extractor_messages(
                sample,
                leading_answer=vote_decision.final_answer,
                stage_rows=stage_rows,
            )
        elif mode == "strategyqa_dual_polarity":
            messages = build_strategyqa_dual_polarity_messages(
                sample,
                variant_index=index,
                leading_answer=vote_decision.final_answer,
                stage_rows=stage_rows,
            )
        else:
            options = _multiple_choice_options(sample)
            permutation = choice_permutation(len(options), index)
            shuffled_options = [options[position] for position in permutation]
            messages = build_choice_shuffle_solver_messages(
                sample,
                shuffled_options=shuffled_options,
                shuffle_label=f"shuffle_{index}",
            )
        row = _execute_turn(
            run_id=run_id,
            dataset=benchmark_slug,
            split_name=split_name,
            sample=sample,
            method_name="cred_acs_expansion",
            method_type="expansion",
            round_index=1,
            agent_id=600 + index,
            role="expansion",
            agent_role=f"acs:{mode}:{runtime.model_ref}",
            visible_peer_count=len(stage_rows),
            messages=messages,
            backbone=runtime.backbone,
            provider=runtime.provider,
            cache=runtime.cache,
            throttle=runtime.throttle,
            temperature=protocol.verifier_temperature,
            top_p=protocol.top_p,
            seed=experiment.global_seed + 600 + index,
            output_protocol=experiment.cred_verification_output_protocol,
            max_tokens=_positive_token_cap(protocol.verifier_max_tokens),
        )
        _annotate_expansion_row(
            row,
            sample=sample,
            mode=mode,
            expansion_index=index,
            permutation=permutation,
            shuffled_options=shuffled_options,
        )
        rows.append(row)
    return rows


def _run_rfs_expansions(
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    sample: DatasetSample,
    experiment: CredVExperimentConfig,
    protocol: CredVProtocolConfig,
    stage_rows: list[dict[str, Any]],
    vote_decision,
    backbone,
    provider,
    cache,
    throttle,
    verifier_runtimes: list[CredVerifierRuntime],
    router_bucket: str,
) -> list[dict[str, Any]]:
    if router_bucket != "weak_split" or benchmark_slug == "strategyqa":
        return []

    rows: list[dict[str, Any]] = []
    max_total_solver_calls = int(protocol.max_total_solver_calls)
    extra_solver_calls = int(protocol.adaptive_extra_solver_calls)
    if max_total_solver_calls > 0:
        extra_solver_calls = min(extra_solver_calls, max(0, max_total_solver_calls - len(stage_rows)))
    for index in range(1, extra_solver_calls + 1):
        row = _execute_turn(
            run_id=run_id,
            dataset=benchmark_slug,
            split_name=split_name,
            sample=sample,
            method_name="cred_rfs_expansion",
            method_type="expansion",
            round_index=1,
            agent_id=700 + index,
            role="expansion",
            agent_role=f"rfs_extra_solver:{index}",
            visible_peer_count=len(stage_rows),
            messages=build_rfs_extra_solver_messages(
                sample,
                variant_index=index,
                leading_answer=vote_decision.final_answer,
                stage_rows=stage_rows,
            ),
            backbone=backbone,
            provider=provider,
            cache=cache,
            throttle=throttle,
            temperature=protocol.stage_a_temperature,
            top_p=protocol.top_p,
            seed=experiment.global_seed + 700 + index,
            output_protocol=experiment.cred_output_protocol,
            max_tokens=_positive_token_cap(protocol.stage_a_max_tokens),
        )
        _annotate_expansion_row(
            row,
            sample=sample,
            mode="rfs_extra_solver",
            expansion_index=index,
            permutation=[],
            shuffled_options=[],
        )
        rows.append(row)

    mode = expansion_mode_for_dataset(benchmark_slug, protocol.expansion_modes, protocol.disabled_expansion_modes)
    if mode != "mc_choice_shuffle" or not verifier_runtimes or int(protocol.max_expansion_calls) <= 0:
        return rows
    options = _multiple_choice_options(sample)
    if not options:
        return rows
    allowed_refs = set(protocol.expansion_model_refs) or set(experiment.verifier_model_refs)
    runtimes = [runtime for runtime in verifier_runtimes if not allowed_refs or runtime.model_ref in allowed_refs]
    if not runtimes:
        return rows
    max_calls = min(int(protocol.max_expansion_calls), 3)
    for index in range(1, max_calls + 1):
        runtime = runtimes[(index - 1) % len(runtimes)]
        permutation = choice_permutation(len(options), index)
        shuffled_options = [options[position] for position in permutation]
        row = _execute_turn(
            run_id=run_id,
            dataset=benchmark_slug,
            split_name=split_name,
            sample=sample,
            method_name="cred_rfs_expansion",
            method_type="expansion",
            round_index=1,
            agent_id=800 + index,
            role="expansion",
            agent_role=f"rfs:mc_choice_shuffle:{runtime.model_ref}",
            visible_peer_count=len(stage_rows),
            messages=build_choice_shuffle_solver_messages(
                sample,
                shuffled_options=shuffled_options,
                shuffle_label=f"shuffle_{index}",
            ),
            backbone=runtime.backbone,
            provider=runtime.provider,
            cache=runtime.cache,
            throttle=runtime.throttle,
            temperature=protocol.verifier_temperature,
            top_p=protocol.top_p,
            seed=experiment.global_seed + 800 + index,
            output_protocol=experiment.cred_verification_output_protocol,
            max_tokens=_positive_token_cap(protocol.verifier_max_tokens),
        )
        _annotate_expansion_row(
            row,
            sample=sample,
            mode=mode,
            expansion_index=extra_solver_calls + index,
            permutation=permutation,
            shuffled_options=shuffled_options,
        )
        rows.append(row)
    return rows


def _run_rfs_pairwise_selection(
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    sample: DatasetSample,
    experiment: CredVExperimentConfig,
    protocol: CredVProtocolConfig,
    stage_rows: list[dict[str, Any]],
    vote_decision,
    backbone,
    provider,
    cache,
    throttle,
    verifier_runtimes: list[CredVerifierRuntime],
    router_bucket: str,
    targets: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    modes = set(protocol.selection_modes)
    shadow_modes = set(protocol.shadow_selection_modes)
    if router_bucket == "deterministic_repair_only":
        return []

    rows: list[dict[str, Any]] = []
    target = targets[0] if targets else None
    if not target:
        return rows
    challenger_answer = str(target.get("normalized_answer") or target.get("prediction") or "")
    if not challenger_answer:
        return rows

    v7_shadow_modes = [mode for mode in _V7_SHADOW_EVIDENCE_MODES if mode in shadow_modes]
    if (
        v7_shadow_modes
        and benchmark_slug in {"gpqa_diamond", "mmlu", "mmlu_abstract_algebra", "mmlu_pro"}
        and (not protocol.shadow_pairwise_allowed_datasets or benchmark_slug in {str(item) for item in protocol.shadow_pairwise_allowed_datasets})
        and router_bucket == "weak_split_select"
    ):
        options = _multiple_choice_options(sample)
        if not options or not verifier_runtimes:
            return rows
        runtimes = _selection_runtimes(experiment, protocol, verifier_runtimes)
        if not runtimes:
            return rows
        rows.extend(
            _run_shadow_evidence_rows(
                run_id=run_id,
                benchmark_slug=benchmark_slug,
                split_name=split_name,
                sample=sample,
                experiment=experiment,
                protocol=protocol,
                vote_decision=vote_decision,
                challenger_answer=challenger_answer,
                runtimes=runtimes,
                evidence_views=v7_shadow_modes,
                start_index=1,
                agent_id_base=1300,
                seed_base=1300,
            )
        )
        return rows

    if (
        "gpqa_unanimous_pairwise_duel" in modes
        and benchmark_slug in {str(item) for item in protocol.pairwise_allowed_datasets}
        and router_bucket == "weak_split_select"
    ):
        options = _multiple_choice_options(sample)
        if (
            not options
            or (
                int(protocol.pairwise_option_count_max) > 0
                and len(options) > int(protocol.pairwise_option_count_max)
            )
            or not verifier_runtimes
        ):
            return rows
        runtimes = _selection_runtimes(experiment, protocol, verifier_runtimes)
        if not runtimes:
            return rows
        rows.extend(
            _run_pairwise_duel_rows(
                run_id=run_id,
                benchmark_slug=benchmark_slug,
                split_name=split_name,
                sample=sample,
                experiment=experiment,
                protocol=protocol,
                vote_decision=vote_decision,
                challenger_answer=challenger_answer,
                runtimes=runtimes,
                count=max(0, int(protocol.pairwise_duel_replicates)),
                start_index=1,
                agent_id_base=900,
                seed_base=900,
                mode="gpqa_unanimous_pairwise_duel",
            )
        )
        if (
            "gpqa_2of3_retry_shadow" in shadow_modes
            and _valid_pairwise_duel_count(rows, mode="gpqa_unanimous_pairwise_duel") == int(protocol.pairwise_duel_replicates)
            and _challenger_pairwise_win_count(rows, mode="gpqa_unanimous_pairwise_duel") == 2
        ):
            rows.extend(
                _run_pairwise_duel_rows(
                    run_id=run_id,
                    benchmark_slug=benchmark_slug,
                    split_name=split_name,
                    sample=sample,
                    experiment=experiment,
                    protocol=protocol,
                    vote_decision=vote_decision,
                    challenger_answer=challenger_answer,
                    runtimes=runtimes,
                    count=max(0, int(protocol.shadow_pairwise_retry_replicates)),
                    start_index=int(protocol.pairwise_duel_replicates) + 1,
                    agent_id_base=1000,
                    seed_base=1000,
                    mode="gpqa_2of3_retry_shadow",
                )
            )
        return rows

    if (
        benchmark_slug in {"mmlu", "mmlu_abstract_algebra", "mmlu_pro"}
        and benchmark_slug in {str(item) for item in protocol.shadow_pairwise_allowed_datasets}
        and "mmlu_unanimous_pairwise_shadow" in shadow_modes
        and router_bucket == "weak_split_select"
    ):
        options = _multiple_choice_options(sample)
        if not options or not verifier_runtimes:
            return rows
        runtimes = _selection_runtimes(experiment, protocol, verifier_runtimes)
        if not runtimes:
            return rows
        rows.extend(
            _run_pairwise_duel_rows(
                run_id=run_id,
                benchmark_slug=benchmark_slug,
                split_name=split_name,
                sample=sample,
                experiment=experiment,
                protocol=protocol,
                vote_decision=vote_decision,
                challenger_answer=challenger_answer,
                runtimes=runtimes,
                count=max(0, int(protocol.pairwise_duel_replicates)),
                start_index=1,
                agent_id_base=1100,
                seed_base=1100,
                mode="mmlu_unanimous_pairwise_shadow",
            )
        )
        return rows

    if (
        benchmark_slug in {"gpqa_diamond", "mmlu", "mmlu_abstract_algebra", "mmlu_pro"}
        and "mc_blind_pairwise_duel" in modes
        and router_bucket in {"weak_split_select", "minority_probe"}
    ):
        options = _multiple_choice_options(sample)
        if not options or not verifier_runtimes:
            return rows
        allowed_refs = set(protocol.expansion_model_refs) or set(experiment.verifier_model_refs)
        runtimes = [runtime for runtime in verifier_runtimes if not allowed_refs or runtime.model_ref in allowed_refs]
        if not runtimes:
            return rows
        for index in range(1, max(0, int(protocol.pairwise_duel_replicates)) + 1):
            runtime = runtimes[(index - 1) % len(runtimes)]
            row = _execute_turn(
                run_id=run_id,
                dataset=benchmark_slug,
                split_name=split_name,
                sample=sample,
                method_name="cred_rfs_expansion",
                method_type="selection",
                round_index=1,
                agent_id=900 + index,
                role="selection",
                agent_role=f"rfs:mc_blind_pairwise_duel:{runtime.model_ref}",
                visible_peer_count=0,
                messages=build_mc_blind_pairwise_duel_messages(
                    sample,
                    leader_answer=vote_decision.final_answer,
                    challenger_answer=challenger_answer,
                    variant_index=index,
                ),
                backbone=runtime.backbone,
                provider=runtime.provider,
                cache=runtime.cache,
                throttle=runtime.throttle,
                temperature=protocol.verifier_temperature,
                top_p=protocol.top_p,
                seed=experiment.global_seed + 900 + index,
                output_protocol=experiment.cred_verification_output_protocol,
                max_tokens=_positive_token_cap(protocol.verifier_max_tokens),
            )
            _annotate_pairwise_duel_row(
                row,
                leader_answer=vote_decision.final_answer,
                challenger_answer=challenger_answer,
                variant_index=index,
                mode="mc_blind_pairwise_duel",
            )
            rows.append(row)
        return rows

    if benchmark_slug == "strategyqa" and "strategyqa_minority_resample" in modes and router_bucket == "minority_probe":
        for index in range(1, max(0, int(protocol.adaptive_extra_solver_calls)) + 1):
            row = _execute_turn(
                run_id=run_id,
                dataset=benchmark_slug,
                split_name=split_name,
                sample=sample,
                method_name="cred_rfs_expansion",
                method_type="selection",
                round_index=1,
                agent_id=950 + index,
                role="selection",
                agent_role=f"rfs:strategyqa_minority_resample:{index}",
                visible_peer_count=0,
                messages=build_strategyqa_minority_resample_messages(
                    sample,
                    variant_index=index,
                    leading_answer=vote_decision.final_answer,
                    challenger_answer=challenger_answer,
                ),
                backbone=backbone,
                provider=provider,
                cache=cache,
                throttle=throttle,
                temperature=protocol.stage_a_temperature,
                top_p=protocol.top_p,
                seed=experiment.global_seed + 950 + index,
                output_protocol=experiment.cred_output_protocol,
                max_tokens=_positive_token_cap(protocol.stage_a_max_tokens),
            )
            _annotate_expansion_row(
                row,
                sample=sample,
                mode="strategyqa_minority_resample",
                expansion_index=index,
                permutation=[],
                shuffled_options=[],
            )
            rows.append(row)
        return rows

    if benchmark_slug == "strategyqa" and "strategyqa_resample_shadow" in shadow_modes and router_bucket == "weak_split_select":
        for index in range(1, max(0, int(protocol.adaptive_extra_solver_calls)) + 1):
            row = _execute_turn(
                run_id=run_id,
                dataset=benchmark_slug,
                split_name=split_name,
                sample=sample,
                method_name="cred_rfs_expansion",
                method_type="selection",
                round_index=1,
                agent_id=1200 + index,
                role="selection",
                agent_role=f"rfs:strategyqa_resample_shadow:{index}",
                visible_peer_count=0,
                messages=build_strategyqa_minority_resample_messages(
                    sample,
                    variant_index=index,
                    leading_answer=vote_decision.final_answer,
                    challenger_answer=challenger_answer,
                ),
                backbone=backbone,
                provider=provider,
                cache=cache,
                throttle=throttle,
                temperature=protocol.stage_a_temperature,
                top_p=protocol.top_p,
                seed=experiment.global_seed + 1200 + index,
                output_protocol=experiment.cred_output_protocol,
                max_tokens=_positive_token_cap(protocol.stage_a_max_tokens),
            )
            _annotate_expansion_row(
                row,
                sample=sample,
                mode="strategyqa_resample_shadow",
                expansion_index=index,
                permutation=[],
                shuffled_options=[],
            )
            rows.append(row)
    return rows


def _selection_runtimes(
    experiment: CredVExperimentConfig,
    protocol: CredVProtocolConfig,
    verifier_runtimes: list[CredVerifierRuntime],
) -> list[CredVerifierRuntime]:
    allowed_refs = set(protocol.expansion_model_refs) or set(experiment.verifier_model_refs)
    return [runtime for runtime in verifier_runtimes if not allowed_refs or runtime.model_ref in allowed_refs]


def _run_pairwise_duel_rows(
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    sample: DatasetSample,
    experiment: CredVExperimentConfig,
    protocol: CredVProtocolConfig,
    vote_decision,
    challenger_answer: str,
    runtimes: list[CredVerifierRuntime],
    count: int,
    start_index: int,
    agent_id_base: int,
    seed_base: int,
    mode: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset in range(0, int(count)):
        index = int(start_index) + offset
        runtime = runtimes[offset % len(runtimes)]
        row = _execute_turn(
            run_id=run_id,
            dataset=benchmark_slug,
            split_name=split_name,
            sample=sample,
            method_name="cred_rfs_expansion",
            method_type="selection",
            round_index=1,
            agent_id=int(agent_id_base) + index,
            role="selection",
            agent_role=f"rfs:{mode}:{runtime.model_ref}",
            visible_peer_count=0,
            messages=build_mc_blind_pairwise_duel_messages(
                sample,
                leader_answer=vote_decision.final_answer,
                challenger_answer=challenger_answer,
                variant_index=index,
            ),
            backbone=runtime.backbone,
            provider=runtime.provider,
            cache=runtime.cache,
            throttle=runtime.throttle,
            temperature=protocol.verifier_temperature,
            top_p=protocol.top_p,
            seed=experiment.global_seed + int(seed_base) + index,
            output_protocol=experiment.cred_verification_output_protocol,
            max_tokens=_positive_token_cap(protocol.verifier_max_tokens),
        )
        _annotate_pairwise_duel_row(
            row,
            leader_answer=vote_decision.final_answer,
            challenger_answer=challenger_answer,
            variant_index=index,
            mode=mode,
        )
        rows.append(row)
    return rows


def _run_shadow_evidence_rows(
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    sample: DatasetSample,
    experiment: CredVExperimentConfig,
    protocol: CredVProtocolConfig,
    vote_decision,
    challenger_answer: str,
    runtimes: list[CredVerifierRuntime],
    evidence_views: list[str],
    start_index: int,
    agent_id_base: int,
    seed_base: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset, evidence_view in enumerate(evidence_views):
        index = int(start_index) + offset
        runtime = runtimes[offset % len(runtimes)]
        row = _execute_turn(
            run_id=run_id,
            dataset=benchmark_slug,
            split_name=split_name,
            sample=sample,
            method_name="cred_rfs_expansion",
            method_type="selection",
            round_index=1,
            agent_id=int(agent_id_base) + index,
            role="selection",
            agent_role=f"rfs:{evidence_view}:{runtime.model_ref}",
            visible_peer_count=0,
            messages=build_mc_shadow_evidence_select_messages(
                sample,
                leader_answer=vote_decision.final_answer,
                challenger_answer=challenger_answer,
                variant_index=index,
                evidence_view=evidence_view,
            ),
            backbone=runtime.backbone,
            provider=runtime.provider,
            cache=runtime.cache,
            throttle=runtime.throttle,
            temperature=protocol.verifier_temperature,
            top_p=protocol.top_p,
            seed=experiment.global_seed + int(seed_base) + index,
            output_protocol=experiment.cred_verification_output_protocol,
            max_tokens=_positive_token_cap(protocol.verifier_max_tokens),
        )
        _annotate_pairwise_duel_row(
            row,
            leader_answer=vote_decision.final_answer,
            challenger_answer=challenger_answer,
            variant_index=index,
            mode=evidence_view,
        )
        row["evidence_view"] = evidence_view
        if isinstance(row.get("validated_output"), dict):
            row["validated_output"]["evidence_view"] = evidence_view
        rows.append(row)
    return rows


def _execute_turn(
    *,
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
    provider,
    cache,
    throttle,
    temperature: float,
    top_p: float,
    seed: int,
    agent_role: str = "",
    output_protocol: str = FREE_TEXT_ANSWER_PROTOCOL_V1,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    result = execute_output_protocol_turn(
        backbone=backbone,
        provider=provider,
        cache=cache,
        throttle=throttle,
        sample=sample,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        dataset=dataset,
        role=role,
        output_protocol=output_protocol,
        max_tokens=max_tokens,
    )
    final_answer = str(result.validated_output.get("final_answer") or result.validated_output.get("answer") or "")
    normalized = normalize_prediction(dataset, final_answer) if final_answer else ""
    if _unusable_format_warning(result.validated_output):
        normalized = ""
    confidence = result.validated_output.get("confidence")
    try:
        confidence_value = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence_value = None
    key_evidence = str(result.validated_output.get("key_evidence") or result.validated_output.get("evidence") or "")
    risk_level = str(result.validated_output.get("risk_level") or "none").strip().lower()
    failure_risk = str(result.validated_output.get("risk_summary") or "")
    row = asdict(
        CredTurnRecord(
            run_id=run_id,
            dataset=dataset,
            split=split_name,
            sample_id=sample.sample_id,
            method_name=method_name,
            method_type=method_type,
            round_index=round_index,
            agent_id=agent_id,
            role=role,
            agent_role=agent_role,
            prompt_hash=result.prompt_hash,
            prediction=normalized,
            normalized_answer=normalized,
            score=None,
            output_status=result.output_status,
            prompt_tokens=float(result.usage.get("prompt_tokens") or 0.0),
            completion_tokens=float(result.usage.get("completion_tokens") or 0.0),
            total_tokens=float(result.usage.get("total_tokens") or 0.0),
            latency_ms=float(result.response_payload.get("latency_ms") or 0.0),
            cache_hit=result.cache_hit,
            request_error=result.request_error,
            request_status=result.request_status,
            raw_finish_reason=result.raw_finish_reason,
            output_protocol=result.output_protocol,
            protocol_parse_status=result.protocol_parse_status,
            protocol_parse_error=result.protocol_parse_error,
            protocol_recovery=str(result.validated_output.get("protocol_recovery") or ""),
            reason_present=result.reason_present,
            request_count=result.request_count,
            cache_request_count=result.cache_request_count,
            network_request_count=result.network_request_count,
            raw_prompt_tokens=float(result.usage.get("prompt_tokens") or 0.0),
            raw_completion_tokens=float(result.usage.get("completion_tokens") or 0.0),
            raw_total_tokens=float(result.usage.get("total_tokens") or 0.0),
            raw_latency_ms=float(result.response_payload.get("latency_ms") or 0.0),
            visible_peer_count=visible_peer_count,
            confidence_value=confidence_value,
            key_evidence=key_evidence,
            risk_level=risk_level if risk_level in {"none", "low", "medium", "high"} else "none",
            failure_risk=failure_risk,
            evidence_quality=0.0,
            payload=result.payload,
            assistant_text=result.response_payload.get("assistant_text", ""),
            provider_reasoning_text=result.response_payload.get("provider_reasoning_text", ""),
            validated_output=result.validated_output,
        )
    )
    row["evidence_quality"] = evidence_quality(row)
    return row


def _unusable_format_warning(payload: dict[str, Any]) -> bool:
    warning = str(payload.get("format_warning") or "").strip()
    return warning in {"answer_contains_reasoning_leak", "answer_too_long_for_final_slot"}


def _safe_select_actual_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("expansion_mode") == "gpqa_unanimous_pairwise_duel"]


def _valid_pairwise_duel_count(rows: list[dict[str, Any]], *, mode: str) -> int:
    return sum(1 for row in rows if row.get("expansion_mode") == mode and row.get("pairwise_validation_pass") is True)


def _challenger_pairwise_win_count(rows: list[dict[str, Any]], *, mode: str) -> int:
    return sum(
        1
        for row in rows
        if row.get("expansion_mode") == mode
        and row.get("pairwise_validation_pass") is True
        and str(row.get("pairwise_winner_family") or "") == str(row.get("pairwise_challenger_family") or "")
    )


def _annotate_expansion_row(
    row: dict[str, Any],
    *,
    sample: DatasetSample,
    mode: str,
    expansion_index: int,
    permutation: list[int],
    shuffled_options: list[str],
) -> None:
    row["expansion_mode"] = mode
    row["expansion_index"] = expansion_index
    row["expansion_shuffle_permutation"] = permutation
    row["expansion_shuffled_options"] = shuffled_options
    row["raw_shuffled_answer"] = ""
    if mode == "mc_choice_shuffle":
        raw_answer = str(row.get("normalized_answer") or row.get("prediction") or "")
        mapped = map_shuffled_choice_answer(raw_answer, permutation)
        row["raw_shuffled_answer"] = raw_answer
        if mapped:
            row["prediction"] = mapped
            row["normalized_answer"] = mapped
            if isinstance(row.get("validated_output"), dict):
                row["validated_output"]["answer"] = mapped
                row["validated_output"]["final_answer"] = mapped
                row["validated_output"]["raw_shuffled_answer"] = raw_answer
                row["validated_output"]["shuffle_permutation"] = permutation
    row["expansion_validation_pass"] = _expansion_validation_pass(row, sample=sample, mode=mode)
    row["evidence_quality"] = evidence_quality(row)


def _annotate_pairwise_duel_row(
    row: dict[str, Any],
    *,
    leader_answer: str,
    challenger_answer: str,
    variant_index: int,
    mode: str = "mc_blind_pairwise_duel",
) -> None:
    row["expansion_mode"] = mode
    row["expansion_index"] = variant_index
    row["expansion_shuffle_permutation"] = []
    row["expansion_shuffled_options"] = []
    row["raw_shuffled_answer"] = ""
    leader = str(leader_answer or "").strip()
    challenger = str(challenger_answer or "").strip()
    dataset = str(row.get("dataset") or "")
    side_map = {"X": "leader", "Y": "challenger"} if int(variant_index) % 2 == 0 else {"X": "challenger", "Y": "leader"}
    payload = row.get("validated_output") if isinstance(row.get("validated_output"), dict) else {}
    selected_side = str(payload.get("selected_side") or payload.get("answer") or row.get("prediction") or "").strip().upper()
    selected_side = selected_side[:1] if selected_side[:1] in {"X", "Y"} else ""
    winner_role = side_map.get(selected_side, "")
    winner_answer = challenger if winner_role == "challenger" else leader if winner_role == "leader" else ""
    row["pairwise_selected_side"] = selected_side
    row["pairwise_leader_answer"] = leader
    row["pairwise_challenger_answer"] = challenger
    row["pairwise_leader_family"] = answer_family_key(dataset, leader)
    row["pairwise_challenger_family"] = answer_family_key(dataset, challenger)
    row["pairwise_winner_role"] = winner_role
    row["pairwise_winner_answer"] = winner_answer
    row["pairwise_winner_family"] = answer_family_key(dataset, winner_answer) if winner_answer else ""
    row["pairwise_validation_pass"] = (
        row.get("request_status") == "ok"
        and row.get("output_status") == "ok"
        and row.get("protocol_parse_status") != "failed"
        and selected_side in {"X", "Y"}
        and bool(winner_answer)
    )
    row["expansion_validation_pass"] = row["pairwise_validation_pass"]
    if isinstance(payload, dict):
        payload["selected_side"] = selected_side
        payload["pairwise_winner_answer"] = winner_answer
        payload["pairwise_winner_role"] = winner_role
    row["evidence_quality"] = evidence_quality(row)


def _expansion_validation_pass(row: dict[str, Any], *, sample: DatasetSample, mode: str) -> bool:
    if row.get("request_status") != "ok" or row.get("output_status") != "ok" or row.get("protocol_parse_status") == "failed":
        return False
    answer = str(row.get("normalized_answer") or row.get("prediction") or "").strip()
    if not answer or answer.lower() in {"unknown", "n/a", "none"}:
        return False
    if mode == "mc_choice_shuffle":
        options = _multiple_choice_options(sample)
        return bool(options) and answer in {chr(ord("A") + index) for index in range(len(options))}
    if mode in {"strategyqa_dual_polarity", "strategyqa_minority_resample", "strategyqa_resample_shadow"}:
        return answer.lower() in {"yes", "no"}
    if mode == "hotpot_span_extract":
        return _context_supports_answer_text(sample.prompt_context, answer)
    return True


def _multiple_choice_options(sample: DatasetSample) -> list[str]:
    raw_options = sample.metadata.get("options") or sample.metadata.get("choices") or []
    if not isinstance(raw_options, list):
        return []
    return [str(option).strip() for option in raw_options if str(option).strip()]


def _context_supports_answer_text(context: str, answer: str) -> bool:
    context_norm = _normalize_span_text(context)
    answer_norm = _normalize_span_text(answer)
    return bool(answer_norm and answer_norm in context_norm)


def _normalize_span_text(value: str) -> str:
    import re
    import unicodedata

    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text.lower()).strip()
    return re.sub(r"\s+", " ", text)


def _prediction_row(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    method_name: str,
    backbone_name: str,
    decision,
    stage_decision,
    router,
    stage_rows: list[dict[str, Any]],
    debate_rows: list[dict[str, Any]],
    method_turns: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    expansion_rows: list[dict[str, Any]],
    protocol: CredVProtocolConfig,
) -> dict[str, Any]:
    prediction = decision.final_answer
    score = score_prediction(dataset, prediction, sample.reference_answer)
    initial_score = score_prediction(dataset, stage_decision.final_answer, sample.reference_answer)
    initial_consensus = len(stage_decision.support) == 1 and bool(stage_decision.support)
    debate_tokens = _sum_field(debate_rows, "total_tokens")
    prompt_tokens = _sum_field(method_turns, "prompt_tokens")
    completion_tokens = _sum_field(method_turns, "completion_tokens")
    total_tokens = _sum_field(method_turns, "total_tokens")
    latency = _sum_field(method_turns, "latency_ms")
    corrected = initial_score < 1.0 and score == 1.0
    harmed = initial_score == 1.0 and score < 1.0
    stage_candidate_oracle_correct = any(
        score_prediction(dataset, _candidate_answer_for_scoring(row), sample.reference_answer) == 1.0
        for row in stage_rows
    )
    oracle_candidate_correct = stage_candidate_oracle_correct
    expansion_oracle_correct = any(
        score_prediction(dataset, _candidate_answer_for_scoring(row), sample.reference_answer) == 1.0
        for row in expansion_rows
        if row.get("request_status") == "ok" and row.get("output_status") == "ok"
    )
    candidate_pool_oracle_correct = oracle_candidate_correct or expansion_oracle_correct
    target_correct = (
        None
        if not targets
        else any(
            score_prediction(dataset, str(row.get("normalized_answer") or row.get("prediction") or ""), sample.reference_answer) == 1.0
            for row in targets
        )
    )
    safe_repair_applied = str(decision.resolver).startswith("cred_verify_safe_deterministic") or str(decision.resolver).startswith(
        "cred_verify_safe_tool"
    )
    hetero_agreement_applied = str(decision.resolver) == "cred_verify_safe_hetero_promoted"
    math_repair_applied = str(decision.resolver) in {
        "cred_acs_math_repair",
        "cred_rfs_math_repair",
        "cred_rfs_v2_math_repair",
        "cred_rfs_v3_math_repair",
        "cred_rfs_v5_math_equivalence_repair_v2",
        "cred_rfs_v6_math_repair",
    }
    hotpot_span_repair_applied = str(decision.resolver) in {
        "cred_acs_hotpot_span_repair",
        "cred_rfs_hotpot_span_repair",
        "cred_rfs_v2_hotpot_span_repair",
        "cred_rfs_v3_hotpot_span_repair",
        "cred_rfs_v5_hotpot_context_span_repair_v2",
        "cred_rfs_v6_hotpot_span_repair",
    }
    single_pro_blocked = str(decision.resolver) in {"cred_acs_single_pro_blocked", "cred_rfs_single_pro_blocked"}
    strong_majority_locked = str(decision.resolver) in {
        "cred_rfs_strong_majority_locked",
        "cred_rfs_v2_strong_majority_locked",
        "cred_rfs_v3_strong_majority_locked",
        "cred_rfs_v5_strong_majority_locked",
    }
    expansion_mode = str(next((row.get("expansion_mode") for row in expansion_rows if row.get("expansion_mode")), ""))
    expansion_validation_pass_count = sum(1 for row in expansion_rows if row.get("expansion_validation_pass") is True)
    choice_shuffle_agreement_count = _choice_shuffle_agreement_count(dataset, expansion_rows)
    pairwise_rows = [row for row in expansion_rows if row.get("expansion_mode") in _PAIRWISE_DUEL_MODES]
    gpqa_unanimous_rows = [row for row in expansion_rows if row.get("expansion_mode") == "gpqa_unanimous_pairwise_duel"]
    leader_family = answer_family_key(dataset, stage_decision.final_answer)
    pairwise_duel_win_count = sum(
        1
        for row in pairwise_rows
        if row.get("pairwise_validation_pass") is True and str(row.get("pairwise_winner_family") or "") != leader_family
    )
    gpqa_unanimous_valid_rows = [row for row in gpqa_unanimous_rows if row.get("pairwise_validation_pass") is True]
    gpqa_unanimous_win_count = sum(1 for row in gpqa_unanimous_valid_rows if str(row.get("pairwise_winner_family") or "") != leader_family)
    resolver = str(decision.resolver)
    safe_selector_corrected = resolver.startswith(("cred_rfs_v3_", "cred_rfs_v5_")) and corrected
    safe_selector_harmed = resolver.startswith(("cred_rfs_v3_", "cred_rfs_v5_")) and harmed
    repair_only_corrected = resolver.startswith("cred_rfs_v6_") and corrected
    repair_only_harmed = resolver.startswith("cred_rfs_v6_") and harmed
    semantic_selector_corrected = _is_semantic_selector_resolver(resolver) and corrected
    semantic_selector_harmed = _is_semantic_selector_resolver(resolver) and harmed
    blocked_2of3_pairwise_count = int(
        resolver in {"cred_rfs_v3_pairwise_rejected", "cred_rfs_v5_pairwise_rejected"}
        and len(gpqa_unanimous_valid_rows) == 3
        and gpqa_unanimous_win_count == 2
    )
    blocked_mmlu_pairwise_count = int(
        resolver in {"cred_rfs_v3_pairwise_dataset_blocked", "cred_rfs_v5_pairwise_dataset_blocked"}
        and dataset in {"mmlu", "mmlu_abstract_algebra", "mmlu_pro"}
    )
    blocked_strategyqa_probe_count = int(
        dataset == "strategyqa"
        and router.trigger_bucket == "weak_split_select"
        and resolver.startswith(("cred_rfs_v3_", "cred_rfs_v4_", "cred_rfs_v5_"))
        and not any(row.get("expansion_mode") in {"strategyqa_minority_resample", "strategyqa_resample_shadow"} for row in expansion_rows)
    )
    shadow = _shadow_counterfactual(
        dataset=dataset,
        stage_winner=stage_decision.final_answer,
        expansion_rows=expansion_rows,
        protocol=protocol,
    )
    shadow_answer = str(shadow.get("answer") or "")
    shadow_score = score_prediction(dataset, shadow_answer, sample.reference_answer) if shadow_answer else 0.0
    shadow_corrected = bool(shadow_answer and initial_score < 1.0 and shadow_score == 1.0)
    shadow_harmed = bool(shadow_answer and initial_score == 1.0 and shadow_score < 1.0)
    shadow_cross_view_agreement_count = _shadow_cross_view_agreement_count(dataset, expansion_rows, leader_family)
    duel_invalid_count = sum(1 for row in pairwise_rows if row.get("pairwise_validation_pass") is not True)
    duel_retry_recoverable_count = int(duel_invalid_count > 0 and dataset in {"gpqa_diamond", "mmlu", "mmlu_abstract_algebra", "mmlu_pro"})
    minority_probe_count = sum(1 for row in expansion_rows if row.get("expansion_mode") == "strategyqa_minority_resample")
    false_consensus_recovered = corrected and (
        "false_consensus_probe" in set(router.reasons)
        or "minority_probe" in set(router.reasons)
        or router.trigger_bucket in {"false_consensus_probe", "minority_probe"}
    )
    free_text_recovered_count = sum(
        1
        for row in method_turns
        if row.get("output_protocol") == FREE_TEXT_ANSWER_PROTOCOL_V1 and row.get("protocol_recovery")
    )
    pairwise_json_recovered_count = sum(
        1 for row in method_turns if row.get("protocol_recovery") == "pairwise_selected_side_fallback"
    )
    json_truncated_count = sum(
        1
        for row in method_turns
        if row.get("output_protocol") == "json_object_answer_v3" and str(row.get("raw_finish_reason") or "") == "length"
    )
    row = asdict(
        CredPredictionRecord(
            run_id=run_id,
            dataset=dataset,
            split=split_name,
            sample_id=sample.sample_id,
            method_name=method_name,
            method_type="mad",
            model_name=backbone_name,
            prediction=prediction,
            gold=sample.reference_answer,
            score=score,
            initial_vote_prediction=stage_decision.final_answer,
            initial_vote_score=initial_score,
            initial_vote_counts={key: int(value) for key, value in stage_decision.support.items()},
            initial_consensus=initial_consensus,
            final_vote_prediction=prediction,
            final_vote_score=score,
            final_vote_counts={prediction: 1} if prediction else {},
            prompt_tokens_per_question=prompt_tokens,
            completion_tokens_per_question=completion_tokens,
            total_tokens_per_question=total_tokens,
            latency_ms_per_question=latency,
            initial_prompt_tokens_per_question=_sum_field(stage_rows, "prompt_tokens"),
            initial_completion_tokens_per_question=_sum_field(stage_rows, "completion_tokens"),
            initial_total_tokens_per_question=_sum_field(stage_rows, "total_tokens"),
            initial_latency_ms_per_question=_sum_field(stage_rows, "latency_ms"),
            debate_prompt_tokens_per_question=_sum_field(debate_rows, "prompt_tokens"),
            debate_completion_tokens_per_question=_sum_field(debate_rows, "completion_tokens"),
            debate_total_tokens_per_question=debate_tokens,
            debate_latency_ms_per_question=_sum_field(debate_rows, "latency_ms"),
            calls_per_question=len(method_turns),
            debate_rounds=1 if debate_rows else 0,
            agent_count=len(stage_rows),
            final_consensus=not decision.changed,
            initial_disagreement=not initial_consensus,
            vote_flipped=decision.changed,
            corrected_by_debate=corrected,
            harmed_by_debate=harmed,
            corrected_by_verification=corrected,
            harmed_by_verification=harmed,
            unchanged_correct=not decision.changed and score == 1.0,
            unchanged_wrong=not decision.changed and score < 1.0,
            triggered=router.triggered,
            router_reasons=list(router.reasons),
            resolver=decision.resolver,
            survival_support={key: round(float(value), 6) for key, value in decision.support.items()},
            verification_support={key: round(float(value), 6) for key, value in decision.support.items()},
            oracle_candidate_correct=oracle_candidate_correct,
            stage_candidate_oracle_correct=stage_candidate_oracle_correct,
            wrong_majority_some_correct=initial_score < 1.0 and oracle_candidate_correct,
            target_correct=target_correct,
            safe_repair_applied=safe_repair_applied,
            hetero_agreement_applied=hetero_agreement_applied,
            expansion_call_count=len(expansion_rows),
            method_expansion_call_count=len(expansion_rows),
            expansion_mode=expansion_mode,
            false_consensus_triggered=(
                "false_consensus_probe" in set(router.reasons)
                or "minority_probe" in set(router.reasons)
                or router.trigger_bucket in {"false_consensus_probe", "minority_probe"}
            ),
            candidate_pool_oracle_correct=candidate_pool_oracle_correct,
            expansion_oracle_correct=expansion_oracle_correct,
            expansion_validation_pass_count=expansion_validation_pass_count,
            math_repair_applied=math_repair_applied,
            hotpot_span_repair_applied=hotpot_span_repair_applied,
            choice_shuffle_agreement_count=choice_shuffle_agreement_count,
            single_pro_promotion_blocked=single_pro_blocked,
            strong_majority_locked=strong_majority_locked,
            pairwise_duel_count=len(pairwise_rows),
            pairwise_duel_win_count=pairwise_duel_win_count,
            safe_selector_corrected=safe_selector_corrected,
            safe_selector_harmed=safe_selector_harmed,
            repair_only_corrected=repair_only_corrected,
            repair_only_harmed=repair_only_harmed,
            semantic_selector_corrected=semantic_selector_corrected,
            semantic_selector_harmed=semantic_selector_harmed,
            gpqa_unanimous_duel_count=len(gpqa_unanimous_rows),
            blocked_2of3_pairwise_count=blocked_2of3_pairwise_count,
            blocked_mmlu_pairwise_count=blocked_mmlu_pairwise_count,
            blocked_strategyqa_probe_count=blocked_strategyqa_probe_count,
            shadow_counterfactual_answer=shadow_answer,
            shadow_counterfactual_resolver=str(shadow.get("resolver") or ""),
            shadow_counterfactual_corrected=shadow_corrected,
            shadow_counterfactual_harmed=shadow_harmed,
            shadow_gate_passed=bool(shadow.get("gate_passed")),
            shadow_net_gain=(1 if shadow_corrected else 0) - (1 if shadow_harmed else 0),
            shadow_cross_view_agreement_count=shadow_cross_view_agreement_count,
            duel_invalid_count=duel_invalid_count,
            duel_retry_recoverable_count=duel_retry_recoverable_count,
            minority_probe_count=minority_probe_count,
            non_answer_candidate_blocked=resolver
            in {"cred_rfs_v2_non_answer_blocked", "cred_rfs_v3_non_answer_blocked", "cred_rfs_v5_non_answer_blocked", "cred_rfs_v6_non_answer_blocked"},
            false_consensus_recovered=false_consensus_recovered,
            free_text_recovered_count=free_text_recovered_count,
            pairwise_json_recovered_count=pairwise_json_recovered_count,
            json_truncated_count=json_truncated_count,
            protocol_failures_per_question=sum(1 for row in method_turns if row.get("protocol_parse_status") == "failed"),
            reason_missing_turns_per_question=sum(1 for row in method_turns if not row.get("reason_present")),
        )
    )
    row["vote_counts"] = row["initial_vote_counts"]
    return row


def _choice_shuffle_agreement_count(dataset: str, expansion_rows: list[dict[str, Any]]) -> int:
    if dataset not in {"gpqa_diamond", "mmlu", "mmlu_abstract_algebra", "mmlu_pro"}:
        return 0
    counts: dict[str, int] = {}
    for row in expansion_rows:
        if row.get("expansion_mode") != "mc_choice_shuffle" or row.get("expansion_validation_pass") is not True:
            continue
        answer = str(row.get("normalized_answer") or row.get("prediction") or "").strip()
        if answer:
            counts[answer] = counts.get(answer, 0) + 1
    return max(counts.values(), default=0)


def _shadow_cross_view_agreement_count(dataset: str, expansion_rows: list[dict[str, Any]], leader_family: str) -> int:
    if dataset not in {"gpqa_diamond", "mmlu", "mmlu_abstract_algebra", "mmlu_pro"}:
        return 0
    counts: dict[str, int] = {}
    for row in expansion_rows:
        if row.get("expansion_mode") not in _V7_SHADOW_EVIDENCE_MODES or row.get("pairwise_validation_pass") is not True:
            continue
        winner_family = str(row.get("pairwise_winner_family") or "")
        if not winner_family or winner_family == leader_family:
            continue
        counts[winner_family] = counts.get(winner_family, 0) + 1
    return max(counts.values(), default=0)


def _is_semantic_selector_resolver(resolver: str) -> bool:
    return str(resolver or "") in {
        "cred_rfs_v2_pairwise_unanimous_promoted",
        "cred_rfs_v2_pairwise_promoted",
        "cred_rfs_v2_strategyqa_minority_promoted",
        "cred_rfs_v3_gpqa_unanimous_pairwise_promoted",
        "cred_rfs_v5_gpqa_unanimous_pairwise_promoted",
    }


def _shadow_counterfactual(
    *,
    dataset: str,
    stage_winner: str,
    expansion_rows: list[dict[str, Any]],
    protocol: CredVProtocolConfig,
) -> dict[str, Any]:
    leader_family = answer_family_key(dataset, stage_winner)
    v7_modes = tuple(mode for mode in _V7_SHADOW_EVIDENCE_MODES if mode in set(protocol.shadow_selection_modes))
    if v7_modes and dataset in {"gpqa_diamond", "mmlu", "mmlu_abstract_algebra", "mmlu_pro"}:
        candidate = _pairwise_shadow_candidate(
            dataset=dataset,
            expansion_rows=expansion_rows,
            leader_family=leader_family,
            mode_names=v7_modes,
            min_valid=len(v7_modes),
            min_wins=len(v7_modes),
            resolver="cred_rfs_v7_cross_view_unanimous_shadow",
        )
        if candidate:
            return candidate
    if dataset == "gpqa_diamond":
        candidate = _pairwise_shadow_candidate(
            dataset=dataset,
            expansion_rows=expansion_rows,
            leader_family=leader_family,
            mode_names=("gpqa_unanimous_pairwise_duel", "gpqa_2of3_retry_shadow"),
            min_valid=int(protocol.shadow_gate_min_valid_duels),
            min_wins=int(protocol.shadow_gate_min_wins),
            resolver="cred_rfs_v4_gpqa_4of5_shadow_gate",
        )
        if candidate:
            return candidate
    if dataset in {"mmlu", "mmlu_abstract_algebra", "mmlu_pro"}:
        candidate = _pairwise_shadow_candidate(
            dataset=dataset,
            expansion_rows=expansion_rows,
            leader_family=leader_family,
            mode_names=("mmlu_unanimous_pairwise_shadow",),
            min_valid=int(protocol.pairwise_duel_replicates),
            min_wins=int(protocol.pairwise_duel_replicates),
            resolver="cred_rfs_v4_mmlu_unanimous_shadow",
        )
        if candidate:
            return candidate
    if dataset == "strategyqa":
        counts: dict[str, tuple[int, str]] = {}
        for row in expansion_rows:
            if row.get("expansion_mode") != "strategyqa_resample_shadow" or row.get("expansion_validation_pass") is not True:
                continue
            answer = str(row.get("normalized_answer") or row.get("prediction") or "").strip()
            family = answer_family_key(dataset, answer)
            if not answer or family == leader_family:
                continue
            count, representative = counts.get(family, (0, answer))
            counts[family] = (count + 1, representative)
        eligible = [(count, answer) for count, answer in counts.values() if count >= 2]
        if eligible:
            _, answer = max(eligible, key=lambda item: (item[0], item[1]))
            return {
                "answer": answer,
                "resolver": "cred_rfs_v4_strategyqa_resample_shadow",
                "gate_passed": True,
            }
    return {"answer": "", "resolver": "", "gate_passed": False}


def _pairwise_shadow_candidate(
    *,
    dataset: str,
    expansion_rows: list[dict[str, Any]],
    leader_family: str,
    mode_names: tuple[str, ...],
    min_valid: int,
    min_wins: int,
    resolver: str,
) -> dict[str, Any]:
    modes = set(mode_names)
    totals: dict[str, int] = {}
    wins: dict[str, int] = {}
    representatives: dict[str, str] = {}
    for row in expansion_rows:
        if row.get("expansion_mode") not in modes or row.get("pairwise_validation_pass") is not True:
            continue
        family = str(row.get("pairwise_challenger_family") or "")
        answer = str(row.get("pairwise_challenger_answer") or "")
        if not family or family == leader_family or not answer:
            continue
        totals[family] = totals.get(family, 0) + 1
        representatives.setdefault(family, answer)
        if str(row.get("pairwise_winner_family") or "") == family:
            wins[family] = wins.get(family, 0) + 1
    eligible = [
        (wins.get(family, 0), total, representatives.get(family, ""))
        for family, total in totals.items()
        if total >= int(min_valid) and wins.get(family, 0) >= int(min_wins)
    ]
    if not eligible:
        return {}
    _, _, answer = max(eligible, key=lambda item: (item[0], item[1], item[2]))
    return {"answer": answer, "resolver": resolver, "gate_passed": True}


def _candidate_answer_for_scoring(row: dict[str, Any]) -> str:
    if row.get("expansion_mode") in _PAIRWISE_DUEL_MODES and row.get("pairwise_validation_pass") is True:
        return str(row.get("pairwise_winner_answer") or "")
    return str(row.get("normalized_answer") or row.get("prediction") or "")


def build_control_prediction_row(
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
    del method
    prompt_tokens = _sum_field(turn_rows, "prompt_tokens")
    completion_tokens = _sum_field(turn_rows, "completion_tokens")
    total_tokens = _sum_field(turn_rows, "total_tokens")
    latency_ms = _sum_field(turn_rows, "latency_ms")
    return {
        "run_id": run_id,
        "dataset": benchmark_slug,
        "split": split_name,
        "sample_id": sample.sample_id,
        "method_name": control_name,
        "method_type": "control",
        "model_name": backbone.name,
        "prediction": final_vote,
        "gold": sample.reference_answer,
        "score": final_score,
        "initial_vote_prediction": final_vote,
        "initial_vote_score": final_score,
        "initial_vote_counts": vote_counts,
        "initial_consensus": final_consensus,
        "final_vote_prediction": final_vote,
        "final_vote_score": final_score,
        "final_vote_counts": vote_counts,
        "prompt_tokens_per_question": prompt_tokens,
        "completion_tokens_per_question": completion_tokens,
        "total_tokens_per_question": total_tokens,
        "latency_ms_per_question": latency_ms,
        "initial_prompt_tokens_per_question": prompt_tokens,
        "initial_completion_tokens_per_question": completion_tokens,
        "initial_total_tokens_per_question": total_tokens,
        "initial_latency_ms_per_question": latency_ms,
        "debate_prompt_tokens_per_question": 0.0,
        "debate_completion_tokens_per_question": 0.0,
        "debate_total_tokens_per_question": 0.0,
        "debate_latency_ms_per_question": 0.0,
        "calls_per_question": len(turn_rows),
        "debate_rounds": 0,
        "agent_count": len(turn_rows),
        "final_consensus": final_consensus,
        "initial_disagreement": False,
        "vote_flipped": False,
        "corrected_by_debate": False,
        "harmed_by_debate": False,
        "corrected_by_verification": False,
        "harmed_by_verification": False,
        "unchanged_correct": final_score == 1.0,
        "unchanged_wrong": final_score < 1.0,
        "triggered": False,
        "router_reasons": [],
        "resolver": "no_comm_control",
        "survival_support": {},
        "verification_support": {},
        "oracle_candidate_correct": final_score == 1.0,
        "stage_candidate_oracle_correct": final_score == 1.0,
        "wrong_majority_some_correct": False,
        "target_correct": None,
        "safe_repair_applied": False,
        "hetero_agreement_applied": False,
        "expansion_call_count": 0,
        "method_expansion_call_count": 0,
        "expansion_mode": "",
        "false_consensus_triggered": False,
        "candidate_pool_oracle_correct": final_score == 1.0,
        "expansion_oracle_correct": False,
        "expansion_validation_pass_count": 0,
        "math_repair_applied": False,
        "hotpot_span_repair_applied": False,
        "choice_shuffle_agreement_count": 0,
        "single_pro_promotion_blocked": False,
        "strong_majority_locked": False,
        "pairwise_duel_count": 0,
        "pairwise_duel_win_count": 0,
        "safe_selector_corrected": False,
        "safe_selector_harmed": False,
        "repair_only_corrected": False,
        "repair_only_harmed": False,
        "semantic_selector_corrected": False,
        "semantic_selector_harmed": False,
        "gpqa_unanimous_duel_count": 0,
        "blocked_2of3_pairwise_count": 0,
        "blocked_mmlu_pairwise_count": 0,
        "blocked_strategyqa_probe_count": 0,
        "shadow_counterfactual_answer": "",
        "shadow_counterfactual_resolver": "",
        "shadow_counterfactual_corrected": False,
        "shadow_counterfactual_harmed": False,
        "shadow_gate_passed": False,
        "shadow_net_gain": 0,
        "shadow_cross_view_agreement_count": 0,
        "duel_invalid_count": 0,
        "duel_retry_recoverable_count": 0,
        "minority_probe_count": 0,
        "non_answer_candidate_blocked": False,
        "false_consensus_recovered": False,
        "free_text_recovered_count": 0,
        "pairwise_json_recovered_count": 0,
        "json_truncated_count": 0,
        "protocol_failures_per_question": sum(1 for row in turn_rows if row.get("protocol_parse_status") == "failed"),
        "reason_missing_turns_per_question": sum(1 for row in turn_rows if not row.get("reason_present")),
        "vote_counts": vote_counts,
    }


def build_metrics(
    prediction_rows: list[dict[str, Any]],
    *,
    dataset_order: list[str],
    method_order: list[str],
    control_names: list[str],
) -> dict[str, Any]:
    dataset_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in prediction_rows:
        grouped.setdefault((row["dataset"], row["model_name"], row["method_name"]), []).append(row)
    for (dataset, model_name, method_name), rows in grouped.items():
        dataset_rows.append(_summarize_prediction_rows(rows, dataset=dataset, model_name=model_name, method_name=method_name, aggregate_kind="dataset"))
    summary = [*dataset_rows, *_build_macro_rows(dataset_rows), *_build_micro_rows(prediction_rows)]
    _attach_comparison_fields(summary, control_names=control_names)
    _attach_v5_incremental_fields(summary, prediction_rows)
    summary.sort(key=lambda row: _summary_sort_key(row, dataset_order, method_order))
    paired = build_paired_comparisons(
        prediction_rows,
        dataset_order=dataset_order,
        method_order=method_order,
        reference_method="sc_5",
    )
    return {"summary": summary, "paired_comparisons": paired}


def _attach_v5_incremental_fields(summary_rows: list[dict[str, Any]], prediction_rows: list[dict[str, Any]]) -> None:
    v5_method = "cred_rfs_evidence_repair_v5"
    v3_method = "cred_rfs_safe_select_v3"
    for row in summary_rows:
        row["math_equivalence_repair_v2_count"] = int(row.get("math_repair_count") or 0) if row.get("method_name") == v5_method else 0
        row["hotpot_span_repair_v2_count"] = int(row.get("hotpot_span_repair_count") or 0) if row.get("method_name") == v5_method else 0
        row["v5_incremental_corrected_vs_v3"] = 0
        row["v5_incremental_harmed_vs_v3"] = 0
        row["v5_actual_gain_vs_v3"] = 0.0

    by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in prediction_rows:
        key = (
            str(row.get("dataset") or ""),
            str(row.get("model_name") or ""),
            str(row.get("sample_id") or ""),
            str(row.get("method_name") or ""),
        )
        by_key[key] = row

    for summary_row in summary_rows:
        if summary_row.get("method_name") != v5_method:
            continue
        dataset = str(summary_row.get("dataset") or "")
        model_name = str(summary_row.get("model_name") or "")
        scoped = [
            row
            for row in prediction_rows
            if row.get("method_name") == v5_method
            and str(row.get("model_name") or "") == model_name
            and (dataset in {"overall", "overall_micro"} or str(row.get("dataset") or "") == dataset)
        ]
        deltas: list[float] = []
        corrected = 0
        harmed = 0
        for row in scoped:
            ref = by_key.get(
                (
                    str(row.get("dataset") or ""),
                    str(row.get("model_name") or ""),
                    str(row.get("sample_id") or ""),
                    v3_method,
                )
            )
            if ref is None:
                continue
            v5_score = float(row.get("score") or 0.0)
            v3_score = float(ref.get("score") or 0.0)
            deltas.append(v5_score - v3_score)
            if v3_score < 1.0 and v5_score == 1.0:
                corrected += 1
            elif v3_score == 1.0 and v5_score < 1.0:
                harmed += 1
        summary_row["v5_incremental_corrected_vs_v3"] = corrected
        summary_row["v5_incremental_harmed_vs_v3"] = harmed
        summary_row["v5_actual_gain_vs_v3"] = round(sum(deltas) / len(deltas), 6) if deltas else 0.0


def build_paired_comparisons(
    prediction_rows: list[dict[str, Any]],
    *,
    dataset_order: list[str],
    method_order: list[str],
    reference_method: str = "sc_5",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    datasets = [*dataset_order, "overall"]
    methods = [method for method in method_order if method != reference_method]
    for dataset in datasets:
        scoped = [
            row
            for row in prediction_rows
            if dataset == "overall" or str(row.get("dataset") or "") == dataset
        ]
        by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
        for row in scoped:
            key = (str(row.get("dataset") or ""), str(row.get("sample_id") or ""), str(row.get("method_name") or ""))
            by_key[key] = row
        for method in methods:
            pairs: list[tuple[float, float]] = []
            sample_keys = {
                (str(row.get("dataset") or ""), str(row.get("sample_id") or ""))
                for row in scoped
                if row.get("method_name") in {method, reference_method}
            }
            for sample_key in sorted(sample_keys):
                method_row = by_key.get((*sample_key, method))
                reference_row = by_key.get((*sample_key, reference_method))
                if method_row is None or reference_row is None:
                    continue
                pairs.append((float(method_row.get("score") or 0.0), float(reference_row.get("score") or 0.0)))
            if not pairs:
                continue
            deltas = [method_score - reference_score for method_score, reference_score in pairs]
            wins = sum(1 for method_score, reference_score in pairs if method_score > reference_score)
            losses = sum(1 for method_score, reference_score in pairs if method_score < reference_score)
            ties = len(pairs) - wins - losses
            ci_low, ci_high = _bootstrap_mean_ci(deltas)
            rows.append(
                {
                    "dataset": dataset,
                    "method_name": method,
                    "reference_method": reference_method,
                    "paired_count": len(pairs),
                    "accuracy_delta": round(sum(deltas) / len(deltas), 6),
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "mcnemar_p": _mcnemar_exact_p(wins, losses),
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "bootstrap_significant_positive": ci_low > 0.0,
                }
            )
    order = {name: index for index, name in enumerate(method_order)}
    dataset_rank = {name: index for index, name in enumerate(dataset_order)}
    rows.sort(key=lambda row: (dataset_rank.get(str(row["dataset"]), len(dataset_order)), order.get(str(row["method_name"]), 999)))
    return rows


def build_debate_diagnostics(
    prediction_rows: list[dict[str, Any]],
    *,
    dataset_order: list[str],
    method_order: list[str],
) -> dict[str, Any]:
    rows = [
        {
            "dataset": row["dataset"],
            "sample_id": row["sample_id"],
            "method_name": row["method_name"],
            "triggered": row.get("triggered"),
            "router_reasons": row.get("router_reasons"),
            "resolver": row.get("resolver"),
            "corrected_by_debate": row.get("corrected_by_debate"),
            "harmed_by_debate": row.get("harmed_by_debate"),
            "corrected_by_verification": row.get("corrected_by_verification"),
            "harmed_by_verification": row.get("harmed_by_verification"),
            "oracle_candidate_correct": row.get("oracle_candidate_correct"),
            "stage_candidate_oracle_correct": row.get("stage_candidate_oracle_correct"),
            "wrong_majority_some_correct": row.get("wrong_majority_some_correct"),
            "target_correct": row.get("target_correct"),
            "safe_repair_applied": row.get("safe_repair_applied"),
            "hetero_agreement_applied": row.get("hetero_agreement_applied"),
            "expansion_call_count": row.get("expansion_call_count"),
            "expansion_mode": row.get("expansion_mode"),
            "candidate_pool_oracle_correct": row.get("candidate_pool_oracle_correct"),
            "expansion_oracle_correct": row.get("expansion_oracle_correct"),
            "math_repair_applied": row.get("math_repair_applied"),
            "hotpot_span_repair_applied": row.get("hotpot_span_repair_applied"),
            "single_pro_promotion_blocked": row.get("single_pro_promotion_blocked"),
            "strong_majority_locked": row.get("strong_majority_locked"),
            "safe_selector_corrected": row.get("safe_selector_corrected"),
            "safe_selector_harmed": row.get("safe_selector_harmed"),
            "repair_only_corrected": row.get("repair_only_corrected"),
            "repair_only_harmed": row.get("repair_only_harmed"),
            "semantic_selector_corrected": row.get("semantic_selector_corrected"),
            "semantic_selector_harmed": row.get("semantic_selector_harmed"),
            "gpqa_unanimous_duel_count": row.get("gpqa_unanimous_duel_count"),
            "blocked_2of3_pairwise_count": row.get("blocked_2of3_pairwise_count"),
            "blocked_mmlu_pairwise_count": row.get("blocked_mmlu_pairwise_count"),
            "blocked_strategyqa_probe_count": row.get("blocked_strategyqa_probe_count"),
            "method_expansion_call_count": row.get("method_expansion_call_count"),
            "shadow_counterfactual_answer": row.get("shadow_counterfactual_answer"),
            "shadow_counterfactual_resolver": row.get("shadow_counterfactual_resolver"),
            "shadow_counterfactual_corrected": row.get("shadow_counterfactual_corrected"),
            "shadow_counterfactual_harmed": row.get("shadow_counterfactual_harmed"),
            "shadow_gate_passed": row.get("shadow_gate_passed"),
            "shadow_net_gain": row.get("shadow_net_gain"),
            "shadow_cross_view_agreement_count": row.get("shadow_cross_view_agreement_count"),
            "duel_invalid_count": row.get("duel_invalid_count"),
            "duel_retry_recoverable_count": row.get("duel_retry_recoverable_count"),
            "free_text_recovered_count": row.get("free_text_recovered_count"),
            "pairwise_json_recovered_count": row.get("pairwise_json_recovered_count"),
            "json_truncated_count": row.get("json_truncated_count"),
            "initial_vote_prediction": row.get("initial_vote_prediction"),
            "prediction": row.get("prediction"),
            "debate_tokens": row.get("debate_total_tokens_per_question"),
        }
        for row in prediction_rows
        if row.get("method_type") == "mad"
    ]
    summary_rows = [
        row for row in build_metrics(prediction_rows, dataset_order=dataset_order, method_order=method_order, control_names=[]).get("summary", [])
        if row.get("method_type") == "mad"
    ]
    return {"sample_rows": rows, "summary_rows": summary_rows}


def build_router_eval(router_rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in router_rows:
        grouped.setdefault(str(row.get("dataset") or ""), []).append(row)
    summary_rows = []
    for dataset, rows in grouped.items():
        count = len(rows)
        summary_rows.append(
            {
                "dataset": dataset,
                "question_count": count,
                "trigger_rate": safe_mean(1.0 if row.get("triggered") else 0.0 for row in rows),
                "avg_risk_count": safe_mean(float(row.get("risk_count") or 0.0) for row in rows),
                "avg_evidence_quality": safe_mean(float(row.get("evidence_quality_mean") or 0.0) for row in rows),
                "refutation_calls_mean": safe_mean(float(row.get("refutation_count") or 0.0) for row in rows),
                "self_verification_calls_mean": safe_mean(float(row.get("self_verification_count") or 0.0) for row in rows),
                "hetero_verification_calls_mean": safe_mean(float(row.get("hetero_verification_count") or 0.0) for row in rows),
                "expansion_calls_mean": safe_mean(float(row.get("expansion_count") or 0.0) for row in rows),
                "false_consensus_trigger_count": sum(1 for row in rows if row.get("trigger_bucket") in {"false_consensus_probe", "minority_probe"}),
                "weak_split_trigger_count": sum(1 for row in rows if row.get("trigger_bucket") in {"weak_split", "weak_split_select"}),
                "deterministic_repair_trigger_count": sum(1 for row in rows if row.get("trigger_bucket") == "deterministic_repair_only"),
                "minority_probe_trigger_count": sum(1 for row in rows if row.get("trigger_bucket") == "minority_probe"),
                "clean_skip_count": sum(1 for row in rows if row.get("trigger_bucket") in {"clean_skip", "clean_anchor_skip"}),
            }
        )
    if router_rows:
        summary_rows.append(
            {
                "dataset": "overall",
                "question_count": len(router_rows),
                "trigger_rate": safe_mean(1.0 if row.get("triggered") else 0.0 for row in router_rows),
                "avg_risk_count": safe_mean(float(row.get("risk_count") or 0.0) for row in router_rows),
                "avg_evidence_quality": safe_mean(float(row.get("evidence_quality_mean") or 0.0) for row in router_rows),
                "refutation_calls_mean": safe_mean(float(row.get("refutation_count") or 0.0) for row in router_rows),
                "self_verification_calls_mean": safe_mean(float(row.get("self_verification_count") or 0.0) for row in router_rows),
                "hetero_verification_calls_mean": safe_mean(float(row.get("hetero_verification_count") or 0.0) for row in router_rows),
                "expansion_calls_mean": safe_mean(float(row.get("expansion_count") or 0.0) for row in router_rows),
                "false_consensus_trigger_count": sum(1 for row in router_rows if row.get("trigger_bucket") in {"false_consensus_probe", "minority_probe"}),
                "weak_split_trigger_count": sum(1 for row in router_rows if row.get("trigger_bucket") in {"weak_split", "weak_split_select"}),
                "deterministic_repair_trigger_count": sum(1 for row in router_rows if row.get("trigger_bucket") == "deterministic_repair_only"),
                "minority_probe_trigger_count": sum(1 for row in router_rows if row.get("trigger_bucket") == "minority_probe"),
                "clean_skip_count": sum(1 for row in router_rows if row.get("trigger_bucket") in {"clean_skip", "clean_anchor_skip"}),
            }
        )
    return {"sample_rows": router_rows, "summary_rows": summary_rows}


def estimate_work(experiment: CredVExperimentConfig, phase_name: str, benchmarks, controls, protocol: CredVProtocolConfig) -> tuple[int, int]:
    total_calls = 0
    total_predictions = 0
    for benchmark in benchmarks:
        split_name = resolve_phase_split_name(experiment, phase_name, benchmark.slug)
        sample_count = len(load_split_ids(benchmark.cache_namespace or benchmark.slug, split_name))
        total_calls += sample_count * protocol.stage_a_agent_count
        if CRED_VERIFY_METHODS & set(experiment.cred_methods):
            total_calls += sample_count * protocol.max_verifications
        if CRED_SAFE_VERIFY_METHODS & set(experiment.cred_methods) and experiment.verifier_model_refs:
            total_calls += sample_count * protocol.max_verification_calls
        if CRED_ACS_METHODS & set(experiment.cred_methods) and experiment.verifier_model_refs:
            total_calls += sample_count * protocol.max_expansion_calls
        if CRED_RFS_ADAPTIVE_METHODS & set(experiment.cred_methods):
            extra_solver_calls = int(protocol.adaptive_extra_solver_calls)
            if int(protocol.max_total_solver_calls) > 0:
                extra_solver_calls = min(extra_solver_calls, max(0, int(protocol.max_total_solver_calls) - int(protocol.stage_a_agent_count)))
            total_calls += sample_count * extra_solver_calls
            if experiment.verifier_model_refs and "mc_choice_shuffle" in set(protocol.expansion_modes) - set(protocol.disabled_expansion_modes):
                total_calls += sample_count * min(int(protocol.max_expansion_calls), 3)
        if CRED_RFS_PAIRWISE_METHODS & set(experiment.cred_methods):
            pairwise_calls = 0
            selection_modes = set(protocol.selection_modes) - set(protocol.disabled_selection_modes)
            shadow_modes = set(protocol.shadow_selection_modes)
            if experiment.verifier_model_refs:
                if "gpqa_unanimous_pairwise_duel" in selection_modes:
                    allowed_datasets = set(protocol.pairwise_allowed_datasets)
                    pairwise_calls = int(protocol.pairwise_duel_replicates) if benchmark.slug in allowed_datasets else 0
                    if "gpqa_2of3_retry_shadow" in shadow_modes and benchmark.slug in allowed_datasets:
                        pairwise_calls += int(protocol.shadow_pairwise_retry_replicates)
                if "mmlu_unanimous_pairwise_shadow" in shadow_modes and benchmark.slug in set(protocol.shadow_pairwise_allowed_datasets):
                    pairwise_calls = max(pairwise_calls, int(protocol.pairwise_duel_replicates))
                elif "mc_blind_pairwise_duel" in selection_modes:
                    pairwise_calls = int(protocol.pairwise_duel_replicates)
            strategy_calls = (
                int(protocol.adaptive_extra_solver_calls)
                if "strategyqa_minority_resample" in selection_modes or ("strategyqa_resample_shadow" in shadow_modes and benchmark.slug == "strategyqa")
                else 0
            )
            total_calls += sample_count * max(pairwise_calls, strategy_calls)
        total_predictions += sample_count * len(experiment.cred_methods)
        for method_name in experiment.control_methods:
            total_calls += sample_count * controls[method_name].budget_calls
            total_predictions += sample_count
    return total_calls, total_predictions


def load_selected_samples(benchmark, split_name: str) -> list[DatasetSample]:
    return select_samples(benchmark, split_name)


def resolve_split_name(experiment: CredVExperimentConfig, phase_name: str, benchmark_slug: str) -> str:
    return resolve_phase_split_name(experiment, phase_name, benchmark_slug)


def _router_row(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    router,
    self_verification_count: int,
    hetero_verification_count: int,
    expansion_count: int,
) -> dict[str, Any]:
    verification_count = int(self_verification_count) + int(hetero_verification_count)
    return {
        "run_id": run_id,
        "dataset": dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "policy_name": "cred_acs_router_v1",
        "triggered": router.triggered,
        "trigger_bucket": router.trigger_bucket,
        "trigger_reasons": list(router.reasons),
        "leading_answer": router.leading_answer,
        "vote_counts": router.vote_counts,
        "risk_count": router.risk_count,
        "evidence_quality_mean": router.evidence_quality_mean,
        "leading_count": router.leading_count,
        "verification_count": verification_count,
        "self_verification_count": self_verification_count,
        "hetero_verification_count": hetero_verification_count,
        "expansion_count": expansion_count,
        "refutation_count": verification_count,
        "defense_count": 0,
        "judge_used": False,
    }


def _debate_message_row(
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    row: dict[str, Any],
    message_kind: str,
) -> dict[str, Any]:
    return asdict(
        DebateMessageRecord(
            run_id=run_id,
            dataset=dataset,
            split=split_name,
            sample_id=sample.sample_id,
            method_name=str(row.get("method_name") or ""),
            round_index=int(row.get("round_index") or 0),
            sender_agent_id=int(row.get("agent_id") or 0),
            recipient_agent_id=0,
            sender_answer=str(row.get("normalized_answer") or ""),
            sender_reasoning=str((row.get("validated_output") or {}).get("reasoning") or ""),
            message_kind=message_kind,
        )
    )


def _sum_field(rows: list[dict[str, Any]], field: str) -> float:
    return round(sum(float(row.get(field) or 0.0) for row in rows), 6)


def _positive_token_cap(value: int | None) -> int | None:
    if value is None or int(value) <= 0:
        return None
    return int(value)


def _summarize_prediction_rows(rows: list[dict[str, Any]], *, dataset: str, model_name: str, method_name: str, aggregate_kind: str) -> dict[str, Any]:
    question_count = len(rows)
    accuracy = safe_mean(float(row["score"]) for row in rows)
    total_tokens = safe_mean(float(row["total_tokens_per_question"]) for row in rows)
    initial_accuracy = safe_mean(_initial_vote_score(row) for row in rows)
    corrected_count = sum(1 for row in rows if row.get("corrected_by_debate"))
    harmed_count = sum(1 for row in rows if row.get("harmed_by_debate"))
    triggered_count = sum(1 for row in rows if row.get("triggered"))
    oracle_accuracy = safe_mean(1.0 if row.get("oracle_candidate_correct") else 0.0 for row in rows)
    stage_candidate_oracle_accuracy = safe_mean(1.0 if row.get("stage_candidate_oracle_correct", row.get("oracle_candidate_correct")) else 0.0 for row in rows)
    candidate_pool_oracle_accuracy = safe_mean(1.0 if row.get("candidate_pool_oracle_correct") else 0.0 for row in rows)
    expansion_oracle_accuracy = safe_mean(1.0 if row.get("expansion_oracle_correct") else 0.0 for row in rows)
    target_rows = [row for row in rows if row.get("wrong_majority_some_correct") and row.get("target_correct") is not None]
    candidate_pool_target_rows = [row for row in rows if _initial_vote_score(row) < 1.0 and row.get("candidate_pool_oracle_correct")]
    promotion_events = corrected_count + harmed_count
    safe_repair_count = sum(1 for row in rows if row.get("safe_repair_applied"))
    hetero_agreement_count = sum(1 for row in rows if row.get("hetero_agreement_applied"))
    math_repair_count = sum(1 for row in rows if row.get("math_repair_applied"))
    hotpot_span_repair_count = sum(1 for row in rows if row.get("hotpot_span_repair_applied"))
    single_pro_blocked_count = sum(1 for row in rows if row.get("single_pro_promotion_blocked"))
    strong_majority_locked_count = sum(1 for row in rows if row.get("strong_majority_locked"))
    validator_pass_count = sum(int(row.get("expansion_validation_pass_count") or 0) for row in rows)
    choice_shuffle_agreement_count = sum(int(row.get("choice_shuffle_agreement_count") or 0) for row in rows)
    pairwise_duel_count = sum(int(row.get("pairwise_duel_count") or 0) for row in rows)
    pairwise_duel_win_count = sum(int(row.get("pairwise_duel_win_count") or 0) for row in rows)
    safe_selector_corrected_count = sum(1 for row in rows if row.get("safe_selector_corrected"))
    safe_selector_harmed_count = sum(1 for row in rows if row.get("safe_selector_harmed"))
    repair_only_corrected_count = sum(1 for row in rows if row.get("repair_only_corrected"))
    repair_only_harmed_count = sum(1 for row in rows if row.get("repair_only_harmed"))
    semantic_selector_corrected_count = sum(1 for row in rows if row.get("semantic_selector_corrected"))
    semantic_selector_harmed_count = sum(1 for row in rows if row.get("semantic_selector_harmed"))
    gpqa_unanimous_duel_count = sum(int(row.get("gpqa_unanimous_duel_count") or 0) for row in rows)
    blocked_2of3_pairwise_count = sum(int(row.get("blocked_2of3_pairwise_count") or 0) for row in rows)
    blocked_mmlu_pairwise_count = sum(int(row.get("blocked_mmlu_pairwise_count") or 0) for row in rows)
    blocked_strategyqa_probe_count = sum(int(row.get("blocked_strategyqa_probe_count") or 0) for row in rows)
    method_expansion_call_count = sum(int(row.get("method_expansion_call_count", row.get("expansion_call_count")) or 0) for row in rows)
    shadow_counterfactual_corrected_count = sum(1 for row in rows if row.get("shadow_counterfactual_corrected"))
    shadow_counterfactual_harmed_count = sum(1 for row in rows if row.get("shadow_counterfactual_harmed"))
    shadow_gate_passed_count = sum(1 for row in rows if row.get("shadow_gate_passed"))
    shadow_net_gain = sum(int(row.get("shadow_net_gain") or 0) for row in rows)
    shadow_cross_view_agreement_count = sum(int(row.get("shadow_cross_view_agreement_count") or 0) for row in rows)
    duel_invalid_count = sum(int(row.get("duel_invalid_count") or 0) for row in rows)
    duel_retry_recoverable_count = sum(int(row.get("duel_retry_recoverable_count") or 0) for row in rows)
    minority_probe_count = sum(int(row.get("minority_probe_count") or 0) for row in rows)
    free_text_recovered_count = sum(int(row.get("free_text_recovered_count") or 0) for row in rows)
    pairwise_json_recovered_count = sum(int(row.get("pairwise_json_recovered_count") or 0) for row in rows)
    json_truncated_count = sum(int(row.get("json_truncated_count") or 0) for row in rows)
    pairwise_corrected = sum(1 for row in rows if _is_semantic_selector_resolver(str(row.get("resolver") or "")) and row.get("corrected_by_debate"))
    pairwise_harmed = sum(1 for row in rows if _is_semantic_selector_resolver(str(row.get("resolver") or "")) and row.get("harmed_by_debate"))
    minority_corrected = sum(1 for row in rows if row.get("resolver") == "cred_rfs_v2_strategyqa_minority_promoted" and row.get("corrected_by_debate"))
    minority_harmed = sum(1 for row in rows if row.get("resolver") == "cred_rfs_v2_strategyqa_minority_promoted" and row.get("harmed_by_debate"))
    adaptive_trigger_rate = safe_mean(1.0 if int(row.get("expansion_call_count") or 0) > 0 else 0.0 for row in rows)
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
        "oracle_accuracy_mean": oracle_accuracy,
        "oracle_gap": round(oracle_accuracy - initial_accuracy, 6),
        "stage_candidate_oracle_accuracy": stage_candidate_oracle_accuracy,
        "candidate_pool_oracle_accuracy": candidate_pool_oracle_accuracy,
        "selection_loss": round(candidate_pool_oracle_accuracy - accuracy, 6),
        "expansion_oracle_accuracy": expansion_oracle_accuracy,
        "expansion_oracle_gain": round(candidate_pool_oracle_accuracy - oracle_accuracy, 6),
        "target_precision_on_wrong_majority": safe_mean(1.0 if row.get("target_correct") else 0.0 for row in target_rows),
        "promotion_recall_on_wrong_majority": safe_ratio(corrected_count, len(candidate_pool_target_rows)),
        "selection_recall_on_pool_correct": safe_ratio(corrected_count, len(candidate_pool_target_rows)),
        "actual_gain": round(accuracy - initial_accuracy, 6),
        "debate_gain_over_initial_vote": round(accuracy - initial_accuracy, 6),
        "verification_gain_over_initial_vote": round(accuracy - initial_accuracy, 6),
        "prompt_tokens_mean": safe_mean(float(row["prompt_tokens_per_question"]) for row in rows),
        "completion_tokens_mean": safe_mean(float(row["completion_tokens_per_question"]) for row in rows),
        "total_tokens_mean": total_tokens,
        "communication_tokens_mean": safe_mean(float(row["debate_total_tokens_per_question"]) for row in rows),
        "latency_ms_mean": safe_mean(float(row["latency_ms_per_question"]) for row in rows),
        "calls_per_question_mean": safe_mean(float(row["calls_per_question"]) for row in rows),
        "protocol_failures_per_question_mean": safe_mean(float(row.get("protocol_failures_per_question") or 0.0) for row in rows),
        "reason_missing_turns_per_question_mean": safe_mean(float(row.get("reason_missing_turns_per_question") or 0.0) for row in rows),
        "accuracy_per_1k_tokens": round((accuracy / total_tokens * 1000) if total_tokens else 0.0, 6),
        "debate_rounds": safe_mean(float(row["debate_rounds"]) for row in rows),
        "agent_count": safe_mean(float(row["agent_count"]) for row in rows),
        "trigger_rate": safe_ratio(triggered_count, question_count),
        "expansion_trigger_rate": adaptive_trigger_rate,
        "adaptive_trigger_rate": adaptive_trigger_rate,
        "false_consensus_trigger_count": sum(1 for row in rows if row.get("false_consensus_triggered")),
        "corrected_count": corrected_count,
        "harmed_count": harmed_count,
        "verified_corrected_count": corrected_count,
        "verified_harmed_count": harmed_count,
        "promotion_precision": safe_ratio(corrected_count, promotion_events),
        "harm_per_correction": round((harmed_count / corrected_count) if corrected_count else float(harmed_count), 6),
        "safe_repair_count": safe_repair_count,
        "hetero_agreement_count": hetero_agreement_count,
        "math_repair_count": math_repair_count,
        "hotpot_span_repair_count": hotpot_span_repair_count,
        "validator_pass_count": validator_pass_count,
        "choice_shuffle_agreement_count": choice_shuffle_agreement_count,
        "pairwise_duel_count": pairwise_duel_count,
        "pairwise_duel_win_count": pairwise_duel_win_count,
        "pairwise_duel_precision": safe_ratio(pairwise_corrected, pairwise_corrected + pairwise_harmed),
        "safe_selector_corrected_count": safe_selector_corrected_count,
        "safe_selector_harmed_count": safe_selector_harmed_count,
        "repair_only_corrected_count": repair_only_corrected_count,
        "repair_only_harmed_count": repair_only_harmed_count,
        "semantic_selector_corrected_count": semantic_selector_corrected_count,
        "semantic_selector_harmed_count": semantic_selector_harmed_count,
        "semantic_selector_precision": safe_ratio(
            semantic_selector_corrected_count,
            semantic_selector_corrected_count + semantic_selector_harmed_count,
        ),
        "gpqa_unanimous_duel_count": gpqa_unanimous_duel_count,
        "blocked_2of3_pairwise_count": blocked_2of3_pairwise_count,
        "blocked_mmlu_pairwise_count": blocked_mmlu_pairwise_count,
        "blocked_strategyqa_probe_count": blocked_strategyqa_probe_count,
        "method_expansion_call_count": method_expansion_call_count,
        "shadow_counterfactual_corrected_count": shadow_counterfactual_corrected_count,
        "shadow_counterfactual_harmed_count": shadow_counterfactual_harmed_count,
        "shadow_precision": safe_ratio(
            shadow_counterfactual_corrected_count,
            shadow_counterfactual_corrected_count + shadow_counterfactual_harmed_count,
        ),
        "shadow_counterfactual_precision": safe_ratio(
            shadow_counterfactual_corrected_count,
            shadow_counterfactual_corrected_count + shadow_counterfactual_harmed_count,
        ),
        "shadow_net_gain": shadow_net_gain,
        "shadow_counterfactual_net_gain": shadow_net_gain,
        "shadow_possible_gain": safe_ratio(shadow_net_gain, question_count),
        "shadow_gate_passed_count": shadow_gate_passed_count,
        "shadow_gate_passed": shadow_gate_passed_count,
        "shadow_cross_view_agreement_count": shadow_cross_view_agreement_count,
        "duel_invalid_count": duel_invalid_count,
        "duel_retry_recoverable_count": duel_retry_recoverable_count,
        "minority_probe_count": minority_probe_count,
        "free_text_recovered_count": free_text_recovered_count,
        "pairwise_json_recovered_count": pairwise_json_recovered_count,
        "json_truncated_count": json_truncated_count,
        "minority_probe_precision": safe_ratio(minority_corrected, minority_corrected + minority_harmed),
        "false_consensus_recovered_count": sum(1 for row in rows if row.get("false_consensus_recovered")),
        "non_answer_candidate_blocked_count": sum(1 for row in rows if row.get("non_answer_candidate_blocked")),
        "single_pro_promotion_blocked_count": single_pro_blocked_count,
        "strong_majority_locked_count": strong_majority_locked_count,
        "corrected_rate": safe_ratio(corrected_count, question_count),
        "harmed_rate": safe_ratio(harmed_count, question_count),
        "flip_rate": safe_mean(1.0 if row.get("vote_flipped") else 0.0 for row in rows),
        "initial_consensus_rate": safe_mean(1.0 if row.get("initial_consensus") else 0.0 for row in rows),
        "final_consensus_rate": safe_mean(1.0 if row.get("final_consensus") else 0.0 for row in rows),
    }


def _initial_vote_score(row: dict[str, Any]) -> float:
    value = row.get("initial_vote_score")
    if value is None:
        value = row["score"]
    return float(value)


def _mcnemar_exact_p(wins: int, losses: int) -> float:
    discordant = int(wins) + int(losses)
    if discordant == 0:
        return 1.0
    tail = min(int(wins), int(losses))
    probability = sum(math.comb(discordant, index) for index in range(tail + 1)) / (2**discordant)
    return round(min(1.0, 2.0 * probability), 6)


def _bootstrap_mean_ci(deltas: list[float], *, iterations: int = 1000) -> tuple[float, float]:
    if not deltas:
        return 0.0, 0.0
    rng = random.Random(20260627 + len(deltas))
    count = len(deltas)
    estimates = []
    for _ in range(iterations):
        estimates.append(sum(deltas[rng.randrange(count)] for _ in range(count)) / count)
    estimates.sort()
    low_index = int(0.025 * (iterations - 1))
    high_index = int(0.975 * (iterations - 1))
    return round(estimates[low_index], 6), round(estimates[high_index], 6)


def _build_macro_rows(dataset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in dataset_rows:
        grouped.setdefault((row["model_name"], row["method_name"]), []).append(row)
    rows = []
    for (model_name, method_name), items in grouped.items():
        rows.append(_macro_from_rows(items, dataset="overall", model_name=model_name, method_name=method_name, aggregate_kind="macro"))
    return rows


def _build_micro_rows(prediction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in prediction_rows:
        grouped.setdefault((row["model_name"], row["method_name"]), []).append(row)
    return [
        _summarize_prediction_rows(items, dataset="overall_micro", model_name=model_name, method_name=method_name, aggregate_kind="micro")
        for (model_name, method_name), items in grouped.items()
    ]


def _macro_from_rows(rows: list[dict[str, Any]], *, dataset: str, model_name: str, method_name: str, aggregate_kind: str) -> dict[str, Any]:
    accuracy = safe_mean(float(row["accuracy_mean"]) for row in rows)
    total_tokens = safe_mean(float(row["total_tokens_mean"]) for row in rows)
    initial_accuracy = safe_mean(float(row["initial_vote_accuracy_mean"]) for row in rows)
    oracle_accuracy = safe_mean(float(row.get("oracle_accuracy_mean") or 0.0) for row in rows)
    stage_candidate_oracle_accuracy = safe_mean(float(row.get("stage_candidate_oracle_accuracy", row.get("oracle_accuracy_mean")) or 0.0) for row in rows)
    candidate_pool_oracle_accuracy = safe_mean(float(row.get("candidate_pool_oracle_accuracy") or 0.0) for row in rows)
    expansion_oracle_accuracy = safe_mean(float(row.get("expansion_oracle_accuracy") or 0.0) for row in rows)
    corrected_count = sum(int(row["corrected_count"]) for row in rows)
    harmed_count = sum(int(row["harmed_count"]) for row in rows)
    adaptive_trigger_rate = safe_mean(float(row.get("adaptive_trigger_rate", row.get("expansion_trigger_rate")) or 0.0) for row in rows)
    return {
        "dataset": dataset,
        "aggregate_kind": aggregate_kind,
        "model_name": model_name,
        "method_name": method_name,
        "method_type": rows[0]["method_type"],
        "question_count": sum(int(row["question_count"]) for row in rows),
        "prediction_rows": sum(int(row["prediction_rows"]) for row in rows),
        "accuracy_mean": accuracy,
        "initial_vote_accuracy_mean": initial_accuracy,
        "oracle_accuracy_mean": oracle_accuracy,
        "oracle_gap": round(oracle_accuracy - initial_accuracy, 6),
        "stage_candidate_oracle_accuracy": stage_candidate_oracle_accuracy,
        "candidate_pool_oracle_accuracy": candidate_pool_oracle_accuracy,
        "selection_loss": round(candidate_pool_oracle_accuracy - accuracy, 6),
        "expansion_oracle_accuracy": expansion_oracle_accuracy,
        "expansion_oracle_gain": round(candidate_pool_oracle_accuracy - oracle_accuracy, 6),
        "target_precision_on_wrong_majority": safe_mean(float(row.get("target_precision_on_wrong_majority") or 0.0) for row in rows),
        "promotion_recall_on_wrong_majority": safe_mean(float(row.get("promotion_recall_on_wrong_majority") or 0.0) for row in rows),
        "selection_recall_on_pool_correct": safe_mean(float(row.get("selection_recall_on_pool_correct") or 0.0) for row in rows),
        "actual_gain": round(accuracy - initial_accuracy, 6),
        "debate_gain_over_initial_vote": round(accuracy - initial_accuracy, 6),
        "verification_gain_over_initial_vote": round(accuracy - initial_accuracy, 6),
        "prompt_tokens_mean": safe_mean(float(row["prompt_tokens_mean"]) for row in rows),
        "completion_tokens_mean": safe_mean(float(row["completion_tokens_mean"]) for row in rows),
        "total_tokens_mean": total_tokens,
        "communication_tokens_mean": safe_mean(float(row["communication_tokens_mean"]) for row in rows),
        "latency_ms_mean": safe_mean(float(row["latency_ms_mean"]) for row in rows),
        "calls_per_question_mean": safe_mean(float(row["calls_per_question_mean"]) for row in rows),
        "protocol_failures_per_question_mean": safe_mean(float(row["protocol_failures_per_question_mean"]) for row in rows),
        "reason_missing_turns_per_question_mean": safe_mean(float(row["reason_missing_turns_per_question_mean"]) for row in rows),
        "accuracy_per_1k_tokens": round((accuracy / total_tokens * 1000) if total_tokens else 0.0, 6),
        "debate_rounds": safe_mean(float(row["debate_rounds"]) for row in rows),
        "agent_count": safe_mean(float(row["agent_count"]) for row in rows),
        "trigger_rate": safe_mean(float(row.get("trigger_rate") or 0.0) for row in rows),
        "expansion_trigger_rate": adaptive_trigger_rate,
        "adaptive_trigger_rate": adaptive_trigger_rate,
        "false_consensus_trigger_count": sum(int(row.get("false_consensus_trigger_count") or 0) for row in rows),
        "corrected_count": corrected_count,
        "harmed_count": harmed_count,
        "verified_corrected_count": corrected_count,
        "verified_harmed_count": harmed_count,
        "promotion_precision": safe_ratio(corrected_count, corrected_count + harmed_count),
        "harm_per_correction": round((harmed_count / corrected_count) if corrected_count else float(harmed_count), 6),
        "safe_repair_count": sum(int(row.get("safe_repair_count") or 0) for row in rows),
        "hetero_agreement_count": sum(int(row.get("hetero_agreement_count") or 0) for row in rows),
        "math_repair_count": sum(int(row.get("math_repair_count") or 0) for row in rows),
        "hotpot_span_repair_count": sum(int(row.get("hotpot_span_repair_count") or 0) for row in rows),
        "validator_pass_count": sum(int(row.get("validator_pass_count") or 0) for row in rows),
        "choice_shuffle_agreement_count": sum(int(row.get("choice_shuffle_agreement_count") or 0) for row in rows),
        "pairwise_duel_count": sum(int(row.get("pairwise_duel_count") or 0) for row in rows),
        "pairwise_duel_win_count": sum(int(row.get("pairwise_duel_win_count") or 0) for row in rows),
        "pairwise_duel_precision": safe_mean(float(row.get("pairwise_duel_precision") or 0.0) for row in rows),
        "safe_selector_corrected_count": sum(int(row.get("safe_selector_corrected_count") or 0) for row in rows),
        "safe_selector_harmed_count": sum(int(row.get("safe_selector_harmed_count") or 0) for row in rows),
        "repair_only_corrected_count": sum(int(row.get("repair_only_corrected_count") or 0) for row in rows),
        "repair_only_harmed_count": sum(int(row.get("repair_only_harmed_count") or 0) for row in rows),
        "semantic_selector_corrected_count": sum(int(row.get("semantic_selector_corrected_count") or 0) for row in rows),
        "semantic_selector_harmed_count": sum(int(row.get("semantic_selector_harmed_count") or 0) for row in rows),
        "semantic_selector_precision": safe_mean(float(row.get("semantic_selector_precision") or 0.0) for row in rows),
        "gpqa_unanimous_duel_count": sum(int(row.get("gpqa_unanimous_duel_count") or 0) for row in rows),
        "blocked_2of3_pairwise_count": sum(int(row.get("blocked_2of3_pairwise_count") or 0) for row in rows),
        "blocked_mmlu_pairwise_count": sum(int(row.get("blocked_mmlu_pairwise_count") or 0) for row in rows),
        "blocked_strategyqa_probe_count": sum(int(row.get("blocked_strategyqa_probe_count") or 0) for row in rows),
        "method_expansion_call_count": sum(int(row.get("method_expansion_call_count") or 0) for row in rows),
        "shadow_counterfactual_corrected_count": sum(int(row.get("shadow_counterfactual_corrected_count") or 0) for row in rows),
        "shadow_counterfactual_harmed_count": sum(int(row.get("shadow_counterfactual_harmed_count") or 0) for row in rows),
        "shadow_precision": safe_mean(float(row.get("shadow_precision") or 0.0) for row in rows),
        "shadow_counterfactual_precision": safe_mean(float(row.get("shadow_counterfactual_precision", row.get("shadow_precision")) or 0.0) for row in rows),
        "shadow_net_gain": sum(int(row.get("shadow_net_gain") or 0) for row in rows),
        "shadow_counterfactual_net_gain": sum(int(row.get("shadow_counterfactual_net_gain", row.get("shadow_net_gain")) or 0) for row in rows),
        "shadow_possible_gain": safe_mean(float(row.get("shadow_possible_gain") or 0.0) for row in rows),
        "shadow_gate_passed_count": sum(int(row.get("shadow_gate_passed_count", row.get("shadow_gate_passed")) or 0) for row in rows),
        "shadow_gate_passed": sum(int(row.get("shadow_gate_passed", row.get("shadow_gate_passed_count")) or 0) for row in rows),
        "shadow_cross_view_agreement_count": sum(int(row.get("shadow_cross_view_agreement_count") or 0) for row in rows),
        "duel_invalid_count": sum(int(row.get("duel_invalid_count") or 0) for row in rows),
        "duel_retry_recoverable_count": sum(int(row.get("duel_retry_recoverable_count") or 0) for row in rows),
        "minority_probe_count": sum(int(row.get("minority_probe_count") or 0) for row in rows),
        "free_text_recovered_count": sum(int(row.get("free_text_recovered_count") or 0) for row in rows),
        "pairwise_json_recovered_count": sum(int(row.get("pairwise_json_recovered_count") or 0) for row in rows),
        "json_truncated_count": sum(int(row.get("json_truncated_count") or 0) for row in rows),
        "minority_probe_precision": safe_mean(float(row.get("minority_probe_precision") or 0.0) for row in rows),
        "false_consensus_recovered_count": sum(int(row.get("false_consensus_recovered_count") or 0) for row in rows),
        "non_answer_candidate_blocked_count": sum(int(row.get("non_answer_candidate_blocked_count") or 0) for row in rows),
        "single_pro_promotion_blocked_count": sum(int(row.get("single_pro_promotion_blocked_count") or 0) for row in rows),
        "strong_majority_locked_count": sum(int(row.get("strong_majority_locked_count") or 0) for row in rows),
        "corrected_rate": safe_mean(float(row["corrected_rate"]) for row in rows),
        "harmed_rate": safe_mean(float(row["harmed_rate"]) for row in rows),
        "flip_rate": safe_mean(float(row["flip_rate"]) for row in rows),
        "initial_consensus_rate": safe_mean(float(row["initial_consensus_rate"]) for row in rows),
        "final_consensus_rate": safe_mean(float(row["final_consensus_rate"]) for row in rows),
    }


def _attach_comparison_fields(summary_rows: list[dict[str, Any]], *, control_names: list[str]) -> None:
    control_set = set(control_names)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in summary_rows:
        grouped.setdefault((str(row["dataset"]), str(row["model_name"])), []).append(row)
    for rows in grouped.values():
        cot_row = next((row for row in rows if row["method_name"] == "cot_1"), None)
        sc5_row = next((row for row in rows if row["method_name"] == "sc_5"), None)
        controls = [row for row in rows if row["method_name"] in control_set]
        best_no_comm = max(controls, key=lambda row: float(row["accuracy_mean"])) if controls else None
        for row in rows:
            row["accuracy_delta_vs_cot_1"] = _delta_against(row, cot_row, "accuracy_mean")
            row["token_ratio_vs_cot_1"] = _ratio_against(row, cot_row, "total_tokens_mean")
            row["calls_ratio_vs_cot_1"] = _ratio_against(row, cot_row, "calls_per_question_mean")
            row["accuracy_delta_vs_sc5"] = _delta_against(row, sc5_row, "accuracy_mean")
            row["repair_only_gain_vs_sc5"] = row["accuracy_delta_vs_sc5"] if row["method_name"] == "cred_rfs_repair_only_v6" else 0.0
            row["token_ratio_vs_sc5"] = _ratio_against(row, sc5_row, "total_tokens_mean")
            row["calls_ratio_vs_sc5"] = _ratio_against(row, sc5_row, "calls_per_question_mean")
            row["base_vote_delta_vs_sc5"] = _delta_against(row, sc5_row, "initial_vote_accuracy_mean")
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


def _summary_sort_key(row: dict[str, Any], dataset_order: list[str], method_order: list[str]) -> tuple[int, int, str]:
    dataset_rank = {name: index for index, name in enumerate(dataset_order)}
    method_rank = {name: index for index, name in enumerate(method_order)}
    special_rank = {"overall": len(dataset_order), "overall_micro": len(dataset_order) + 1}
    return (
        special_rank.get(row["dataset"], dataset_rank.get(row["dataset"], len(dataset_order) + 2)),
        method_rank.get(row["method_name"], 999),
        str(row["model_name"]),
    )
