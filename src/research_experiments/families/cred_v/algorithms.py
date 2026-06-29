"""CRED-V 验证器路由与聚合逻辑。"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

_NON_ANSWERS = {"", "unknown", "n/a", "none"}
_MATH_DATASETS = {"gsm8k", "math500", "competition_math"}
_SPAN_DATASETS = {"hotpotqa"}
_HETERO_DATASETS = {"gpqa_diamond", "mmlu", "mmlu_abstract_algebra", "mmlu_pro", "strategyqa"}
_MC_DATASETS = {"gpqa_diamond", "mmlu", "mmlu_abstract_algebra", "mmlu_pro"}
_ACS_EXPANSION_METHOD = "cred_acs_expansion"
_RFS_EXPANSION_METHOD = "cred_rfs_expansion"
_EXPANSION_METHODS = {_ACS_EXPANSION_METHOD, _RFS_EXPANSION_METHOD}
_HOTPOT_BANNED_EXTRA_TOKENS = {
    "and",
    "or",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
}
_HOTPOT_ALLOWED_EXTRA_TOKENS = {
    "captain",
    "center",
    "centre",
    "city",
    "club",
    "college",
    "company",
    "county",
    "country",
    "episode",
    "episodes",
    "excellence",
    "fc",
    "film",
    "football",
    "inc",
    "language",
    "ltd",
    "of",
    "organization",
    "organisation",
    "province",
    "river",
    "school",
    "state",
    "student",
    "students",
    "team",
    "the",
    "title",
    "university",
}


@dataclass(frozen=True)
class CredRouterDecision:
    triggered: bool
    reasons: tuple[str, ...]
    trigger_bucket: str
    leading_answer: str
    vote_counts: dict[str, int]
    risk_count: int
    evidence_quality_mean: float
    leading_count: int


@dataclass(frozen=True)
class CredAggregateDecision:
    final_answer: str
    support: dict[str, float]
    resolver: str
    changed: bool
    source: str


def answer_family_key(dataset: str, answer: str) -> str:
    normalized = str(answer or "").strip()
    if not normalized:
        return "unknown"
    if dataset in {"gpqa_diamond", "mmlu", "mmlu_abstract_algebra", "mmlu_pro"}:
        match = re.search(r"\b([A-J])\b", normalized.upper())
        return f"mc:{match.group(1)}" if match else "mc_text:" + _coarse_text(normalized)
    if dataset == "strategyqa":
        lowered = normalized.lower()
        if lowered.startswith("yes"):
            return "bool:yes"
        if lowered.startswith("no"):
            return "bool:no"
    return _coarse_text(normalized)


def build_router_decision(rows: list[dict[str, Any]], *, protocol) -> CredRouterDecision:
    dataset = _dataset_from_rows(rows)
    selection_modes = {str(mode) for mode in getattr(protocol, "selection_modes", ())}
    shadow_modes = {str(mode) for mode in getattr(protocol, "shadow_selection_modes", ())}
    pairwise_mode = bool(
        {
            "mc_blind_pairwise_duel",
            "gpqa_unanimous_pairwise_duel",
            "direct_option_contrast_shadow",
            "constraint_elimination_shadow",
            "minimal_evidence_certificate_shadow",
            "strategyqa_resample_shadow",
        }
        & (selection_modes | shadow_modes)
    )
    grouped, _ = _stage_candidate_groups(dataset, rows)
    stage_decision = aggregate_stage_a_vote(rows)
    leading_answer = stage_decision.final_answer
    leading_family = answer_family_key(dataset, leading_answer)
    vote_counts = {key: int(value) for key, value in stage_decision.support.items()}
    leading_count = len(grouped.get(leading_family, [])) if leading_answer else 0
    risk_count = sum(1 for row in rows if _row_risk_level(row) in {"medium", "high"})
    evidence_mean = _mean(evidence_quality(row) for row in rows)
    strong_majority_count = int(getattr(protocol, "leader_lock_count", getattr(protocol, "strong_majority_count", 4)))
    reasons: list[str] = []
    if not leading_answer:
        reasons.append("no_valid_stage_a_answer")
        bucket = "format_risk"
    elif leading_count < strong_majority_count:
        reasons.append("weak_split_no_strong_majority")
        bucket = "weak_split_select" if pairwise_mode else "weak_split"
    elif _format_repair_risk(dataset, rows, leading_family, leading_answer):
        reasons.append("deterministic_repair_candidate")
        bucket = "deterministic_repair_only"
    elif _format_risk(dataset, rows):
        reasons.append("answer_format_risk")
        bucket = "format_risk"
    elif getattr(protocol, "false_consensus_probe", False) and _false_consensus_probe_needed(
        dataset=dataset,
        rows=rows,
        grouped=grouped,
        leading_family=leading_family,
        leading_count=leading_count,
        risk_count=risk_count,
        evidence_mean=evidence_mean,
        strong_majority_count=strong_majority_count,
        pairwise_mode=pairwise_mode,
    ):
        bucket = "minority_probe" if pairwise_mode else "false_consensus_probe"
        reasons.append(bucket)
    else:
        bucket = "clean_anchor_skip" if pairwise_mode else "clean_skip"
    enabled_buckets = _enabled_trigger_buckets(protocol)
    clean_bucket = "clean_anchor_skip" if pairwise_mode else "clean_skip"
    triggered = bucket != clean_bucket and bucket in enabled_buckets
    return CredRouterDecision(
        triggered=triggered,
        reasons=tuple(reasons if triggered else ("strong_majority_skip",)),
        trigger_bucket=bucket if triggered else clean_bucket,
        leading_answer=leading_answer,
        vote_counts=vote_counts,
        risk_count=risk_count,
        evidence_quality_mean=round(evidence_mean, 6),
        leading_count=leading_count,
    )


def aggregate_stage_a_vote(rows: list[dict[str, Any]]) -> CredAggregateDecision:
    dataset = _dataset_from_rows(rows)
    grouped, ordered_families = _stage_candidate_groups(dataset, rows)
    if not grouped:
        return CredAggregateDecision(
            final_answer="",
            support={},
            resolver="cred_v_vote_5_empty",
            changed=False,
            source="stage_a_vote",
        )
    count_family = max(ordered_families, key=lambda family: (len(grouped[family]), -ordered_families.index(family)))
    winner = _representative_answer(dataset, grouped[count_family])
    counts = _stage_support_counts(dataset, grouped, ordered_families)
    return CredAggregateDecision(
        final_answer=winner,
        support={answer: float(count) for answer, count in counts.items()},
        resolver="cred_v_vote_5_family_majority",
        changed=False,
        source="stage_a_vote",
    )


def aggregate_task_verification(
    *,
    dataset: str,
    stage_rows: list[dict[str, Any]],
    verifier_rows: list[dict[str, Any]],
    stage_winner: str,
    promotion_confidence_min: float,
    promotion_score_margin: float,
    concrete_evidence_min_chars: int,
) -> CredAggregateDecision:
    stage_support = dict(aggregate_stage_a_vote(stage_rows).support)
    if not verifier_rows or not stage_winner:
        return CredAggregateDecision(
            stage_winner,
            stage_support,
            "cred_v_task_verify_empty_fallback",
            False,
            "stage_a",
        )

    leader_family = answer_family_key(dataset, stage_winner)
    eligible: list[tuple[float, float, float, str, dict[str, Any]]] = []
    for row in verifier_rows:
        candidate = _row_answer(row)
        if not _is_candidate(candidate):
            continue
        candidate_family = answer_family_key(dataset, candidate)
        if candidate_family == leader_family:
            continue
        if not _stage_has_family(dataset, stage_rows, candidate_family):
            continue
        if not _row_bool(row, "promote"):
            continue
        if not _has_concrete_evidence(row, concrete_evidence_min_chars):
            continue
        confidence = _row_confidence(row)
        if confidence < float(promotion_confidence_min):
            continue
        leader_score = _row_float(row, "leader_score", default=0.5)
        challenger_score = _row_float(row, "challenger_score", default=0.5)
        margin = challenger_score - leader_score
        if margin < float(promotion_score_margin):
            continue
        eligible.append((margin, confidence, challenger_score, candidate, row))

    if not eligible:
        return CredAggregateDecision(
            stage_winner,
            _verification_support(stage_support, verifier_rows, stage_winner),
            "cred_v_task_verify_rejected",
            False,
            "stage_a",
        )

    margin, confidence, challenger_score, winner, _ = max(eligible, key=lambda item: (item[0], item[1], item[2], item[3]))
    support = _verification_support(stage_support, verifier_rows, stage_winner)
    support[winner] = round(max(float(support.get(winner, 0.0)), 5.0 + margin + confidence + challenger_score), 6)
    return CredAggregateDecision(
        winner,
        support,
        "cred_v_task_verify_promoted",
        True,
        "task_verifier",
    )


def aggregate_safe_verification(
    *,
    dataset: str,
    question: str,
    context: str,
    stage_rows: list[dict[str, Any]],
    verifier_rows: list[dict[str, Any]],
    hetero_verifier_rows: list[dict[str, Any]],
    stage_winner: str,
    verification_modes: tuple[str, ...] | list[str],
    allow_same_model_promotion: bool,
    concrete_evidence_min_chars: int,
    strong_majority_count: int,
) -> CredAggregateDecision:
    del question
    stage_support = dict(aggregate_stage_a_vote(stage_rows).support)
    if not stage_rows or not stage_winner:
        return CredAggregateDecision(
            stage_winner,
            stage_support,
            "cred_verify_safe_empty_fallback",
            False,
            "stage_a",
        )

    modes = {str(mode) for mode in verification_modes}
    challengers = _safe_challenger_rows(dataset, stage_rows, stage_winner)
    if "deterministic_repair" in modes:
        for challenger in challengers:
            candidate = _row_answer(challenger)
            if _deterministic_repair_allows(dataset, stage_winner, candidate, context):
                return _safe_promoted_decision(
                    dataset=dataset,
                    stage_support=stage_support,
                    stage_rows=stage_rows,
                    stage_winner=stage_winner,
                    winner=candidate,
                    resolver="cred_verify_safe_deterministic_repair",
                    source="deterministic_repair",
                )

    if "tool_verified" in modes:
        for challenger in challengers:
            candidate = _row_answer(challenger)
            if _tool_verification_allows(dataset, stage_winner, candidate, context):
                return _safe_promoted_decision(
                    dataset=dataset,
                    stage_support=stage_support,
                    stage_rows=stage_rows,
                    stage_winner=stage_winner,
                    winner=candidate,
                    resolver="cred_verify_safe_tool_verified",
                    source="tool_verified",
                )

    hetero_rows = list(hetero_verifier_rows)
    if allow_same_model_promotion:
        hetero_rows.extend(verifier_rows)
    if "hetero_verified" in modes and dataset in _HETERO_DATASETS:
        leader_family = answer_family_key(dataset, stage_winner)
        leader_count = _stage_family_count(dataset, stage_rows, leader_family)
        if leader_count < int(strong_majority_count):
            eligible: list[tuple[float, str, dict[str, Any]]] = []
            for row in hetero_rows:
                candidate = _row_answer(row)
                candidate_family = answer_family_key(dataset, candidate)
                if not _is_candidate(candidate) or candidate_family == leader_family:
                    continue
                if not _stage_has_family(dataset, stage_rows, candidate_family):
                    continue
                if not _row_bool(row, "promote"):
                    continue
                if _row_bool(row, "leader_pass") or not _row_bool(row, "challenger_pass"):
                    continue
                if not _has_concrete_evidence(row, concrete_evidence_min_chars):
                    continue
                eligible.append((_row_confidence(row), candidate, row))
            if eligible:
                _, winner, _ = max(eligible, key=lambda item: (item[0], item[1]))
                return _safe_promoted_decision(
                    dataset=dataset,
                    stage_support=stage_support,
                    stage_rows=stage_rows,
                    stage_winner=stage_winner,
                    winner=winner,
                    resolver="cred_verify_safe_hetero_promoted",
                    source="hetero_verifier",
                )

    return CredAggregateDecision(
        stage_winner,
        _verification_support(stage_support, [*verifier_rows, *hetero_verifier_rows], stage_winner),
        "cred_verify_safe_rejected",
        False,
        "stage_a",
    )


def aggregate_adaptive_candidate_search(
    *,
    dataset: str,
    question: str,
    context: str,
    stage_rows: list[dict[str, Any]],
    expansion_rows: list[dict[str, Any]],
    stage_winner: str,
    expansion_modes: tuple[str, ...] | list[str],
    promotion_min_independent_support: int,
    promotion_margin_min: float,
    strong_majority_count: int,
) -> CredAggregateDecision:
    del question
    stage_decision = aggregate_stage_a_vote(stage_rows)
    stage_support = dict(stage_decision.support)
    if not stage_rows or not stage_winner:
        return CredAggregateDecision(
            stage_winner,
            stage_support,
            "cred_acs_empty_fallback",
            False,
            "stage_a",
        )

    modes = {str(mode) for mode in expansion_modes}
    candidate_rows = [*stage_rows, *_valid_expansion_rows(expansion_rows)]
    if {"deterministic_repair", "tool_verified", "math_symbolic_repair", "hotpot_span_extract"} & modes:
        for candidate in _safe_challenger_rows(dataset, candidate_rows, stage_winner):
            challenger = _row_answer(candidate)
            if _deterministic_repair_allows(dataset, stage_winner, challenger, context):
                resolver = "cred_acs_math_repair" if dataset in _MATH_DATASETS else "cred_acs_hotpot_span_repair"
                return _safe_promoted_decision(
                    dataset=dataset,
                    stage_support=stage_support,
                    stage_rows=candidate_rows,
                    stage_winner=stage_winner,
                    winner=challenger,
                    resolver=resolver,
                    source="deterministic_repair",
                )

    leader_family = answer_family_key(dataset, stage_winner)
    leader_count = _stage_family_count(dataset, stage_rows, leader_family)
    scored = _acs_family_scores(dataset, stage_rows=stage_rows, expansion_rows=expansion_rows)
    leader_score = scored.get(leader_family, (0.0, stage_winner))[0]
    eligible: list[tuple[float, float, str, str]] = []
    for family, (score, representative) in scored.items():
        if family == leader_family or not _is_candidate(representative):
            continue
        expansion_support = _expansion_family_count(dataset, expansion_rows, family)
        if dataset in _HETERO_DATASETS and expansion_support < int(promotion_min_independent_support):
            continue
        if leader_count >= int(strong_majority_count) and expansion_support < int(promotion_min_independent_support):
            continue
        margin = score - leader_score
        if margin < float(promotion_margin_min):
            continue
        eligible.append((margin, score, family, representative))

    support = _acs_answer_support(dataset, scored)
    if not eligible:
        if _has_single_blocked_expansion(dataset, expansion_rows, leader_family):
            resolver = "cred_acs_single_pro_blocked"
        else:
            resolver = "cred_acs_rejected"
        return CredAggregateDecision(
            stage_winner,
            support,
            resolver,
            False,
            "stage_a",
        )

    _, _, _, winner = max(eligible, key=lambda item: (item[0], item[1], item[3]))
    return CredAggregateDecision(
        winner,
        support,
        "cred_acs_candidate_promoted",
        answer_family_key(dataset, winner) != leader_family,
        "adaptive_candidate_search",
    )


def aggregate_reasoning_first_selection(
    *,
    dataset: str,
    question: str,
    context: str,
    stage_rows: list[dict[str, Any]],
    expansion_rows: list[dict[str, Any]],
    stage_winner: str,
    expansion_modes: tuple[str, ...] | list[str],
    promotion_min_independent_support: int,
    promotion_margin_min: float,
    leader_lock_count: int,
    mc_shuffle_min_agreement: int,
    require_stage_a_challenger_support: bool,
) -> CredAggregateDecision:
    del question
    stage_decision = aggregate_stage_a_vote(stage_rows)
    stage_support = dict(stage_decision.support)
    if not stage_rows or not stage_winner:
        return CredAggregateDecision(
            stage_winner,
            stage_support,
            "cred_rfs_empty_fallback",
            False,
            "stage_a",
        )

    candidate_rows = [*stage_rows, *_valid_expansion_rows(expansion_rows)]
    for candidate in _safe_challenger_rows(dataset, candidate_rows, stage_winner):
        challenger = _row_answer(candidate)
        if _deterministic_repair_allows(dataset, stage_winner, challenger, context):
            resolver = "cred_rfs_math_repair" if dataset in _MATH_DATASETS else "cred_rfs_hotpot_span_repair"
            return _safe_promoted_decision(
                dataset=dataset,
                stage_support=stage_support,
                stage_rows=candidate_rows,
                stage_winner=stage_winner,
                winner=challenger,
                resolver=resolver,
                source="deterministic_repair",
            )

    leader_family = answer_family_key(dataset, stage_winner)
    leader_count = _stage_family_count(dataset, stage_rows, leader_family)
    modes = {str(mode) for mode in expansion_modes}
    scored = _rfs_family_scores(dataset, stage_rows=stage_rows, expansion_rows=expansion_rows)
    support = _acs_answer_support(dataset, scored)

    if leader_count >= int(leader_lock_count):
        return CredAggregateDecision(
            stage_winner,
            support,
            "cred_rfs_strong_majority_locked",
            False,
            "stage_a",
        )

    if dataset == "strategyqa":
        return CredAggregateDecision(
            stage_winner,
            support,
            "cred_rfs_strategyqa_promotion_disabled",
            False,
            "stage_a",
        )

    leader_score = scored.get(leader_family, (0.0, stage_winner))[0]
    eligible: list[tuple[float, float, str, str]] = []
    for family, (score, representative) in scored.items():
        if family == leader_family or not _is_candidate(representative):
            continue
        if require_stage_a_challenger_support and not _stage_has_family(dataset, stage_rows, family):
            continue
        expansion_support = _expansion_family_count(dataset, expansion_rows, family)
        if expansion_support < int(promotion_min_independent_support):
            continue
        if dataset in _MC_DATASETS and "mc_choice_shuffle" in modes and _mc_shuffle_family_count(
            dataset,
            expansion_rows,
            family,
        ) < int(mc_shuffle_min_agreement):
            continue
        margin = score - leader_score
        if margin < float(promotion_margin_min):
            continue
        eligible.append((margin, score, family, representative))

    if not eligible:
        resolver = "cred_rfs_single_pro_blocked" if _has_single_blocked_expansion(dataset, expansion_rows, leader_family) else "cred_rfs_rejected"
        return CredAggregateDecision(
            stage_winner,
            support,
            resolver,
            False,
            "stage_a",
        )

    _, _, _, winner = max(eligible, key=lambda item: (item[0], item[1], item[3]))
    return CredAggregateDecision(
        winner,
        support,
        "cred_rfs_candidate_promoted",
        answer_family_key(dataset, winner) != leader_family,
        "reasoning_first_selection",
    )


def aggregate_pairwise_selection(
    *,
    dataset: str,
    question: str,
    context: str,
    stage_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    stage_winner: str,
    selection_modes: tuple[str, ...] | list[str],
    leader_lock_count: int,
    pairwise_duel_replicates: int,
    pairwise_promotion_min_wins: int,
    require_stage_a_challenger_support: bool,
) -> CredAggregateDecision:
    del question
    stage_decision = aggregate_stage_a_vote(stage_rows)
    stage_support = dict(stage_decision.support)
    if not stage_rows or not stage_winner:
        return CredAggregateDecision(
            stage_winner,
            stage_support,
            "cred_rfs_v2_empty_fallback",
            False,
            "stage_a",
        )

    modes = {str(mode) for mode in selection_modes}
    leader_family = answer_family_key(dataset, stage_winner)
    leader_count = _stage_family_count(dataset, stage_rows, leader_family)

    if {"deterministic_repair", "hotpot_context_span_repair"} & modes:
        for candidate in _safe_challenger_rows(dataset, stage_rows, stage_winner):
            challenger = _row_answer(candidate)
            if _deterministic_repair_allows(dataset, stage_winner, challenger, context):
                resolver = "cred_rfs_v2_math_repair" if dataset in _MATH_DATASETS else "cred_rfs_v2_hotpot_span_repair"
                return _safe_promoted_decision(
                    dataset=dataset,
                    stage_support=stage_support,
                    stage_rows=stage_rows,
                    stage_winner=stage_winner,
                    winner=challenger,
                    resolver=resolver,
                    source="deterministic_repair",
                )

    if dataset in _SPAN_DATASETS and _has_non_answer_challenger(dataset, stage_rows, leader_family):
        return CredAggregateDecision(
            stage_winner,
            _pairwise_support(dataset, stage_support, selection_rows),
            "cred_rfs_v2_non_answer_blocked",
            False,
            "stage_a",
        )

    pairwise_eligible = _pairwise_duel_promotions(
        dataset=dataset,
        stage_rows=stage_rows,
        selection_rows=selection_rows,
        leader_family=leader_family,
        require_stage_a_challenger_support=require_stage_a_challenger_support,
        min_wins=int(pairwise_promotion_min_wins),
    )
    strategy_eligible = _strategyqa_minority_promotions(
        dataset=dataset,
        stage_rows=stage_rows,
        selection_rows=selection_rows,
        leader_family=leader_family,
        require_stage_a_challenger_support=require_stage_a_challenger_support,
    )
    support = _pairwise_support(dataset, stage_support, selection_rows)

    if dataset == "strategyqa" and "strategyqa_minority_resample" in modes and strategy_eligible:
        _, stage_count, probe_count, winner = max(strategy_eligible, key=lambda item: (item[0], item[1], item[2], item[3]))
        support[winner] = round(max(float(support.get(winner, 0.0)), float(stage_count + probe_count)), 6)
        return CredAggregateDecision(
            winner,
            support,
            "cred_rfs_v2_strategyqa_minority_promoted",
            answer_family_key(dataset, winner) != leader_family,
            "strategyqa_minority_resample",
        )

    unanimous_pairwise = [
        item for item in pairwise_eligible if item[0] >= int(pairwise_duel_replicates) and int(pairwise_duel_replicates) > 0
    ]
    if leader_count >= int(leader_lock_count):
        if dataset in _MC_DATASETS and "mc_blind_pairwise_duel" in modes and unanimous_pairwise:
            wins, total, _, winner = max(unanimous_pairwise, key=lambda item: (item[0], item[1], item[3]))
            support[winner] = round(max(float(support.get(winner, 0.0)), float(wins)), 6)
            return CredAggregateDecision(
                winner,
                support,
                "cred_rfs_v2_pairwise_unanimous_promoted",
                answer_family_key(dataset, winner) != leader_family,
                "mc_blind_pairwise_duel",
            )
        return CredAggregateDecision(
            stage_winner,
            support,
            "cred_rfs_v2_strong_majority_locked",
            False,
            "stage_a",
        )

    if dataset in _MC_DATASETS and "mc_blind_pairwise_duel" in modes and pairwise_eligible:
        wins, total, _, winner = max(pairwise_eligible, key=lambda item: (item[0], item[1], item[3]))
        support[winner] = round(max(float(support.get(winner, 0.0)), float(wins)), 6)
        return CredAggregateDecision(
            winner,
            support,
            "cred_rfs_v2_pairwise_promoted",
            answer_family_key(dataset, winner) != leader_family,
            "mc_blind_pairwise_duel",
        )

    if _has_pairwise_candidate_blocked(dataset, selection_rows, leader_family):
        resolver = "cred_rfs_v2_pairwise_rejected"
    elif _has_strategyqa_probe_blocked(dataset, selection_rows, leader_family):
        resolver = "cred_rfs_v2_strategyqa_minority_rejected"
    else:
        resolver = "cred_rfs_v2_rejected"
    return CredAggregateDecision(
        stage_winner,
        support,
        resolver,
        False,
        "stage_a",
    )


def aggregate_safe_select_v3(
    *,
    dataset: str,
    question: str,
    context: str,
    stage_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    stage_winner: str,
    selection_modes: tuple[str, ...] | list[str],
    leader_lock_count: int,
    pairwise_duel_replicates: int,
    pairwise_promotion_min_wins: int,
    pairwise_allowed_datasets: tuple[str, ...] | list[str],
    pairwise_option_count_max: int,
    option_count: int,
    require_stage_a_challenger_support: bool,
    allow_strong_majority_pairwise_promotion: bool,
) -> CredAggregateDecision:
    del question
    stage_decision = aggregate_stage_a_vote(stage_rows)
    stage_support = dict(stage_decision.support)
    if not stage_rows or not stage_winner:
        return CredAggregateDecision(
            stage_winner,
            stage_support,
            "cred_rfs_v3_empty_fallback",
            False,
            "stage_a",
        )

    modes = {str(mode) for mode in selection_modes}
    leader_family = answer_family_key(dataset, stage_winner)
    leader_count = _stage_family_count(dataset, stage_rows, leader_family)

    if {"deterministic_repair", "hotpot_context_span_repair"} & modes:
        for candidate in _safe_challenger_rows(dataset, stage_rows, stage_winner):
            challenger = _row_answer(candidate)
            if _deterministic_repair_allows(dataset, stage_winner, challenger, context):
                resolver = "cred_rfs_v3_math_repair" if dataset in _MATH_DATASETS else "cred_rfs_v3_hotpot_span_repair"
                return _safe_promoted_decision(
                    dataset=dataset,
                    stage_support=stage_support,
                    stage_rows=stage_rows,
                    stage_winner=stage_winner,
                    winner=challenger,
                    resolver=resolver,
                    source="deterministic_repair",
                )

    support = _pairwise_support(dataset, stage_support, selection_rows)
    if dataset in _SPAN_DATASETS and _has_non_answer_challenger(dataset, stage_rows, leader_family):
        return CredAggregateDecision(
            stage_winner,
            support,
            "cred_rfs_v3_non_answer_blocked",
            False,
            "stage_a",
        )

    if leader_count >= int(leader_lock_count) and not allow_strong_majority_pairwise_promotion:
        return CredAggregateDecision(
            stage_winner,
            support,
            "cred_rfs_v3_strong_majority_locked",
            False,
            "stage_a",
        )

    if "gpqa_unanimous_pairwise_duel" not in modes:
        return CredAggregateDecision(
            stage_winner,
            support,
            "cred_rfs_v3_pairwise_disabled",
            False,
            "stage_a",
        )

    allowed_datasets = {str(item) for item in pairwise_allowed_datasets}
    if dataset not in allowed_datasets:
        return CredAggregateDecision(
            stage_winner,
            support,
            "cred_rfs_v3_pairwise_dataset_blocked",
            False,
            "stage_a",
        )

    if int(pairwise_option_count_max) > 0 and int(option_count) > int(pairwise_option_count_max):
        return CredAggregateDecision(
            stage_winner,
            support,
            "cred_rfs_v3_pairwise_option_count_blocked",
            False,
            "stage_a",
        )

    pairwise_eligible = _pairwise_duel_promotions(
        dataset=dataset,
        stage_rows=stage_rows,
        selection_rows=selection_rows,
        leader_family=leader_family,
        require_stage_a_challenger_support=require_stage_a_challenger_support,
        min_wins=max(int(pairwise_promotion_min_wins), int(pairwise_duel_replicates)),
        mode_names=("gpqa_unanimous_pairwise_duel",),
    )
    unanimous_pairwise = [
        item
        for item in pairwise_eligible
        if item[0] >= int(pairwise_duel_replicates) and item[1] >= int(pairwise_duel_replicates) and int(pairwise_duel_replicates) > 0
    ]
    if unanimous_pairwise:
        wins, total, _, winner = max(unanimous_pairwise, key=lambda item: (item[0], item[1], item[3]))
        support[winner] = round(max(float(support.get(winner, 0.0)), float(wins)), 6)
        return CredAggregateDecision(
            winner,
            support,
            "cred_rfs_v3_gpqa_unanimous_pairwise_promoted",
            answer_family_key(dataset, winner) != leader_family,
            "gpqa_unanimous_pairwise_duel",
        )

    resolver = "cred_rfs_v3_pairwise_rejected" if _has_pairwise_candidate_blocked(dataset, selection_rows, leader_family) else "cred_rfs_v3_rejected"
    return CredAggregateDecision(
        stage_winner,
        support,
        resolver,
        False,
        "stage_a",
    )


def aggregate_shadow_select_v4(
    *,
    dataset: str,
    question: str,
    context: str,
    stage_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    stage_winner: str,
    selection_modes: tuple[str, ...] | list[str],
    leader_lock_count: int,
    pairwise_duel_replicates: int,
    pairwise_promotion_min_wins: int,
    pairwise_allowed_datasets: tuple[str, ...] | list[str],
    pairwise_option_count_max: int,
    option_count: int,
    require_stage_a_challenger_support: bool,
    allow_strong_majority_pairwise_promotion: bool,
) -> CredAggregateDecision:
    actual = aggregate_safe_select_v3(
        dataset=dataset,
        question=question,
        context=context,
        stage_rows=stage_rows,
        selection_rows=selection_rows,
        stage_winner=stage_winner,
        selection_modes=selection_modes,
        leader_lock_count=leader_lock_count,
        pairwise_duel_replicates=pairwise_duel_replicates,
        pairwise_promotion_min_wins=pairwise_promotion_min_wins,
        pairwise_allowed_datasets=pairwise_allowed_datasets,
        pairwise_option_count_max=pairwise_option_count_max,
        option_count=option_count,
        require_stage_a_challenger_support=require_stage_a_challenger_support,
        allow_strong_majority_pairwise_promotion=allow_strong_majority_pairwise_promotion,
    )
    return CredAggregateDecision(
        actual.final_answer,
        actual.support,
        "cred_rfs_v4_shadow_no_promotion",
        actual.changed,
        "shadow_calibration",
    )


def aggregate_evidence_repair_v5(
    *,
    dataset: str,
    question: str,
    context: str,
    stage_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    stage_winner: str,
    selection_modes: tuple[str, ...] | list[str],
    leader_lock_count: int,
    pairwise_duel_replicates: int,
    pairwise_promotion_min_wins: int,
    pairwise_allowed_datasets: tuple[str, ...] | list[str],
    pairwise_option_count_max: int,
    option_count: int,
    require_stage_a_challenger_support: bool,
    allow_strong_majority_pairwise_promotion: bool,
) -> CredAggregateDecision:
    del question
    stage_decision = aggregate_stage_a_vote(stage_rows)
    stage_support = dict(stage_decision.support)
    if not stage_rows or not stage_winner:
        return CredAggregateDecision(
            stage_winner,
            stage_support,
            "cred_rfs_v5_empty_fallback",
            False,
            "stage_a",
        )

    modes = {str(mode) for mode in selection_modes}
    leader_family = answer_family_key(dataset, stage_winner)
    leader_count = _stage_family_count(dataset, stage_rows, leader_family)

    for candidate in _safe_challenger_rows(dataset, stage_rows, stage_winner):
        challenger = _row_answer(candidate)
        if _math_equivalence_repair_v2_allows(dataset, stage_winner, challenger, modes):
            return _safe_promoted_decision(
                dataset=dataset,
                stage_support=stage_support,
                stage_rows=stage_rows,
                stage_winner=stage_winner,
                winner=challenger,
                resolver="cred_rfs_v5_math_equivalence_repair_v2",
                source="math_equivalence_repair_v2",
            )
        if _hotpot_context_span_repair_v2_allows(dataset, stage_winner, challenger, context, modes):
            return _safe_promoted_decision(
                dataset=dataset,
                stage_support=stage_support,
                stage_rows=stage_rows,
                stage_winner=stage_winner,
                winner=challenger,
                resolver="cred_rfs_v5_hotpot_context_span_repair_v2",
                source="hotpot_context_span_repair_v2",
            )

    support = _pairwise_support(dataset, stage_support, selection_rows)
    if dataset in _SPAN_DATASETS and _has_non_answer_challenger(dataset, stage_rows, leader_family):
        return CredAggregateDecision(
            stage_winner,
            support,
            "cred_rfs_v5_non_answer_blocked",
            False,
            "stage_a",
        )

    if leader_count >= int(leader_lock_count) and not allow_strong_majority_pairwise_promotion:
        return CredAggregateDecision(
            stage_winner,
            support,
            "cred_rfs_v5_strong_majority_locked",
            False,
            "stage_a",
        )

    if "gpqa_unanimous_pairwise_duel" not in modes:
        return CredAggregateDecision(
            stage_winner,
            support,
            "cred_rfs_v5_pairwise_disabled",
            False,
            "stage_a",
        )

    allowed_datasets = {str(item) for item in pairwise_allowed_datasets}
    if dataset not in allowed_datasets:
        return CredAggregateDecision(
            stage_winner,
            support,
            "cred_rfs_v5_pairwise_dataset_blocked",
            False,
            "stage_a",
        )

    if int(pairwise_option_count_max) > 0 and int(option_count) > int(pairwise_option_count_max):
        return CredAggregateDecision(
            stage_winner,
            support,
            "cred_rfs_v5_pairwise_option_count_blocked",
            False,
            "stage_a",
        )

    pairwise_eligible = _pairwise_duel_promotions(
        dataset=dataset,
        stage_rows=stage_rows,
        selection_rows=selection_rows,
        leader_family=leader_family,
        require_stage_a_challenger_support=require_stage_a_challenger_support,
        min_wins=max(int(pairwise_promotion_min_wins), int(pairwise_duel_replicates)),
        mode_names=("gpqa_unanimous_pairwise_duel",),
    )
    unanimous_pairwise = [
        item
        for item in pairwise_eligible
        if item[0] >= int(pairwise_duel_replicates) and item[1] >= int(pairwise_duel_replicates) and int(pairwise_duel_replicates) > 0
    ]
    if unanimous_pairwise:
        wins, total, _, winner = max(unanimous_pairwise, key=lambda item: (item[0], item[1], item[3]))
        support[winner] = round(max(float(support.get(winner, 0.0)), float(wins)), 6)
        return CredAggregateDecision(
            winner,
            support,
            "cred_rfs_v5_gpqa_unanimous_pairwise_promoted",
            answer_family_key(dataset, winner) != leader_family,
            "gpqa_unanimous_pairwise_duel",
        )

    resolver = "cred_rfs_v5_pairwise_rejected" if _has_pairwise_candidate_blocked(dataset, selection_rows, leader_family) else "cred_rfs_v5_rejected"
    return CredAggregateDecision(
        stage_winner,
        support,
        resolver,
        False,
        "stage_a",
    )


def aggregate_repair_only_v6(
    *,
    dataset: str,
    question: str,
    context: str,
    stage_rows: list[dict[str, Any]],
    stage_winner: str,
    selection_modes: tuple[str, ...] | list[str],
) -> CredAggregateDecision:
    del question
    stage_decision = aggregate_stage_a_vote(stage_rows)
    stage_support = dict(stage_decision.support)
    if not stage_rows or not stage_winner:
        return CredAggregateDecision(
            stage_winner,
            stage_support,
            "cred_rfs_v6_empty_fallback",
            False,
            "stage_a",
        )

    modes = {str(mode) for mode in selection_modes}
    leader_family = answer_family_key(dataset, stage_winner)
    repair_enabled = (
        dataset in _MATH_DATASETS
        and {"deterministic_repair", "math_deterministic_repair"} & modes
        or dataset in _SPAN_DATASETS
        and {"deterministic_repair", "hotpot_context_span_repair"} & modes
    )
    if repair_enabled:
        for candidate in _safe_challenger_rows(dataset, stage_rows, stage_winner):
            challenger = _row_answer(candidate)
            if _repair_only_v6_allows(dataset, stage_winner, challenger, context):
                resolver = "cred_rfs_v6_math_repair" if dataset in _MATH_DATASETS else "cred_rfs_v6_hotpot_span_repair"
                return _safe_promoted_decision(
                    dataset=dataset,
                    stage_support=stage_support,
                    stage_rows=stage_rows,
                    stage_winner=stage_winner,
                    winner=challenger,
                    resolver=resolver,
                    source="deterministic_repair",
                )

    if dataset in _SPAN_DATASETS and _has_non_answer_challenger(dataset, stage_rows, leader_family):
        return CredAggregateDecision(
            stage_winner,
            stage_support,
            "cred_rfs_v6_non_answer_blocked",
            False,
            "stage_a",
        )

    return CredAggregateDecision(
        stage_winner,
        stage_support,
        "cred_rfs_v6_repair_only_rejected",
        False,
        "stage_a",
    )


def aggregate_shadow_evidence_select_v7(
    *,
    dataset: str,
    question: str,
    context: str,
    stage_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    stage_winner: str,
    selection_modes: tuple[str, ...] | list[str],
) -> CredAggregateDecision:
    del selection_rows
    return aggregate_repair_only_v6(
        dataset=dataset,
        question=question,
        context=context,
        stage_rows=stage_rows,
        stage_winner=stage_winner,
        selection_modes=selection_modes,
    )


def aggregate_repair_bank_v8(
    *,
    dataset: str,
    question: str,
    context: str,
    stage_rows: list[dict[str, Any]],
    stage_winner: str,
    selection_modes: tuple[str, ...] | list[str],
    option_texts: tuple[str, ...] | list[str] = (),
) -> CredAggregateDecision:
    del question
    stage_decision = aggregate_stage_a_vote(stage_rows)
    stage_support = dict(stage_decision.support)
    if not stage_rows or not stage_winner:
        return CredAggregateDecision(
            stage_winner,
            stage_support,
            "cred_rfs_v8_empty_fallback",
            False,
            "stage_a",
        )

    modes = {str(mode) for mode in selection_modes}
    leader_family = answer_family_key(dataset, stage_winner)
    if dataset in _MC_DATASETS and {"deterministic_repair", "mc_option_text_repair"} & modes:
        mapped = _mc_option_text_letter(stage_winner, option_texts)
        if mapped and answer_family_key(dataset, mapped) != leader_family:
            support = dict(stage_support)
            support[mapped] = max(float(support.get(mapped, 0.0)), float(_stage_family_count(dataset, stage_rows, leader_family)) + 0.5)
            return CredAggregateDecision(
                mapped,
                {answer: round(float(value), 6) for answer, value in support.items()},
                "cred_rfs_v8_mc_option_text_repair",
                True,
                "deterministic_repair",
            )

    repair_enabled = (
        dataset in _MATH_DATASETS
        and {"deterministic_repair", "math_deterministic_repair", "math_repair_bank_v8"} & modes
        or dataset in _SPAN_DATASETS
        and {"deterministic_repair", "hotpot_context_span_repair", "hotpot_context_span_repair_v2"} & modes
    )
    if repair_enabled:
        for candidate in _safe_challenger_rows(dataset, stage_rows, stage_winner):
            challenger = _row_answer(candidate)
            if _repair_bank_v8_allows(dataset, stage_winner, challenger, context):
                resolver = "cred_rfs_v8_math_repair" if dataset in _MATH_DATASETS else "cred_rfs_v8_hotpot_span_repair"
                return _safe_promoted_decision(
                    dataset=dataset,
                    stage_support=stage_support,
                    stage_rows=stage_rows,
                    stage_winner=stage_winner,
                    winner=challenger,
                    resolver=resolver,
                    source="deterministic_repair",
                )

    if dataset in _SPAN_DATASETS and _has_non_answer_challenger(dataset, stage_rows, leader_family):
        return CredAggregateDecision(
            stage_winner,
            stage_support,
            "cred_rfs_v8_non_answer_blocked",
            False,
            "stage_a",
        )

    return CredAggregateDecision(
        stage_winner,
        stage_support,
        "cred_rfs_v8_repair_bank_rejected",
        False,
        "stage_a",
    )


def aggregate_certificate_shadow_v9(
    *,
    dataset: str,
    question: str,
    context: str,
    stage_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    stage_winner: str,
    selection_modes: tuple[str, ...] | list[str],
    option_texts: tuple[str, ...] | list[str] = (),
) -> CredAggregateDecision:
    del selection_rows
    actual = aggregate_repair_bank_v8(
        dataset=dataset,
        question=question,
        context=context,
        stage_rows=stage_rows,
        stage_winner=stage_winner,
        selection_modes=selection_modes,
        option_texts=option_texts,
    )
    return CredAggregateDecision(
        actual.final_answer,
        actual.support,
        actual.resolver,
        actual.changed,
        "certificate_shadow",
    )


def select_verification_targets(
    *,
    dataset: str,
    rows: list[dict[str, Any]],
    leading_answer: str,
    max_verifications: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        answer = _row_answer(row)
        if _is_candidate(answer):
            grouped[answer_family_key(dataset, answer)].append(row)
    leading_family = answer_family_key(dataset, leading_answer)
    candidates = [
        max(items, key=lambda row: (_row_confidence(row), evidence_quality(row)))
        for family, items in grouped.items()
        if family != leading_family
    ]
    candidates.sort(
        key=lambda row: (_row_confidence(row), evidence_quality(row), str(row.get("agent_role") or "")),
        reverse=True,
    )
    return candidates[: int(max_verifications)]


def expansion_mode_for_dataset(
    dataset: str,
    expansion_modes: tuple[str, ...] | list[str],
    disabled_expansion_modes: tuple[str, ...] | list[str] = (),
) -> str:
    modes = {str(mode) for mode in expansion_modes} - {str(mode) for mode in disabled_expansion_modes}
    if dataset in _MATH_DATASETS and "math_symbolic_repair" in modes:
        return "math_symbolic_repair"
    if dataset in _SPAN_DATASETS and "hotpot_span_extract" in modes:
        return "hotpot_span_extract"
    if dataset in _MC_DATASETS and "mc_choice_shuffle" in modes:
        return "mc_choice_shuffle"
    if dataset == "strategyqa" and "strategyqa_dual_polarity" in modes:
        return "strategyqa_dual_polarity"
    return ""


def map_shuffled_choice_answer(answer: str, permutation: list[int]) -> str:
    letter = _choice_letter_from_text(answer)
    if not letter:
        return ""
    shuffled_index = ord(letter) - ord("A")
    if shuffled_index < 0 or shuffled_index >= len(permutation):
        return ""
    original_index = int(permutation[shuffled_index])
    if original_index < 0 or original_index >= 10:
        return ""
    return chr(ord("A") + original_index)


def choice_permutation(option_count: int, variant_index: int) -> list[int]:
    if option_count <= 0:
        return []
    base = list(range(option_count))
    variant = int(variant_index) % 3
    if variant == 1:
        return list(reversed(base))
    if variant == 2:
        shift = max(1, option_count // 2)
        return base[shift:] + base[:shift]
    return base[1::2] + base[0::2]


def evidence_quality(row: dict[str, Any]) -> float:
    text = " ".join(
        str(row.get(key) or row.get("validated_output", {}).get(key) or "")
        for key in ("key_evidence", "evidence", "contract", "risk_summary", "reasoning")
    )
    cleaned = " ".join(text.split())
    score = 0.0
    if len(cleaned) >= 12:
        score += 0.35
    if any(char.isdigit() for char in cleaned):
        score += 0.15
    if any(token in cleaned.lower() for token in ("because", "therefore", "context", "option", "constraint", "calculate", "span")):
        score += 0.2
    if _is_candidate(_row_answer(row)) and _row_answer(row).lower() in cleaned.lower():
        score += 0.2
    return min(1.0, round(score, 6))


def _format_repair_risk(dataset: str, rows: list[dict[str, Any]], leading_family: str, leading_answer: str) -> bool:
    if dataset not in (_MATH_DATASETS | _SPAN_DATASETS):
        return False
    for row in rows:
        answer = _row_answer(row)
        if not _is_candidate(answer) or answer_family_key(dataset, answer) == leading_family:
            continue
        if dataset in _MATH_DATASETS and _canonical_math_text(answer) == _canonical_math_text(leading_answer):
            return True
        if dataset in _SPAN_DATASETS and _span_tokens(answer):
            return True
    return False


def _format_risk(dataset: str, rows: list[dict[str, Any]]) -> bool:
    valid = [_row_answer(row) for row in rows if _is_candidate(_row_answer(row))]
    if len(valid) < max(1, len(rows) - 1):
        return True
    if dataset in _MC_DATASETS:
        return any(not re.fullmatch(r"[A-J]", answer.strip().upper()) for answer in valid)
    if dataset == "strategyqa":
        return any(answer.strip().lower() not in {"yes", "no"} for answer in valid)
    return False


def _false_consensus_probe_needed(
    *,
    dataset: str,
    rows: list[dict[str, Any]],
    grouped: dict[str, list[dict[str, Any]]],
    leading_family: str,
    leading_count: int,
    risk_count: int,
    evidence_mean: float,
    strong_majority_count: int,
    pairwise_mode: bool = False,
) -> bool:
    if leading_count < strong_majority_count:
        return False
    if dataset not in (_HETERO_DATASETS | _MATH_DATASETS | _SPAN_DATASETS):
        return False
    leader_rows = grouped.get(leading_family, [])
    minority_rows = [row for family, items in grouped.items() if family != leading_family for row in items]
    leader_conf = _mean(_row_confidence(row) for row in leader_rows)
    minority_conf = max((_row_confidence(row) for row in minority_rows), default=0.0)
    minority_evidence = max((evidence_quality(row) for row in minority_rows), default=0.0)
    leader_evidence = _mean(evidence_quality(row) for row in leader_rows)
    if pairwise_mode and minority_rows and dataset in (_MC_DATASETS | _MATH_DATASETS | {"strategyqa"}):
        if leading_count != strong_majority_count:
            return False
        return minority_conf >= leader_conf + 0.10 or minority_evidence >= leader_evidence + 0.15 or risk_count >= 2
    if minority_rows and minority_conf >= leader_conf + 0.10 and minority_evidence >= leader_evidence:
        return True
    if minority_rows and risk_count >= 2:
        return True
    return leading_count >= len(rows) and risk_count >= 3 and evidence_mean < 0.45


def _valid_expansion_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row.get("method_name") in _EXPANSION_METHODS
        and row.get("request_status") == "ok"
        and row.get("output_status") == "ok"
        and row.get("protocol_parse_status") != "failed"
        and _is_candidate(_row_answer(row))
    ]


def _enabled_trigger_buckets(protocol) -> set[str]:
    raw = getattr(protocol, "trigger_buckets", None)
    if raw is None:
        return {
            "weak_split",
            "weak_split_select",
            "format_risk",
            "deterministic_repair_only",
            "false_consensus_probe",
            "minority_probe",
        }
    return {str(item) for item in raw}


def _acs_family_scores(
    dataset: str,
    *,
    stage_rows: list[dict[str, Any]],
    expansion_rows: list[dict[str, Any]],
) -> dict[str, tuple[float, str]]:
    support: dict[str, float] = defaultdict(float)
    representatives: dict[str, str] = {}
    rep_score: dict[str, tuple[float, float]] = {}
    for row in stage_rows:
        answer = _row_answer(row)
        if not _is_candidate(answer):
            continue
        family = answer_family_key(dataset, answer)
        support[family] += 1.0
        score = (_row_confidence(row), evidence_quality(row))
        if family not in rep_score or score > rep_score[family]:
            representatives[family] = answer
            rep_score[family] = score
    for row in _valid_expansion_rows(expansion_rows):
        answer = _row_answer(row)
        family = answer_family_key(dataset, answer)
        support[family] += 1.0 + _expansion_validation_bonus(row)
        score = (_row_confidence(row) + _expansion_validation_bonus(row), evidence_quality(row))
        if family not in rep_score or score > rep_score[family]:
            representatives[family] = answer
            rep_score[family] = score
    return {family: (round(score, 6), representatives.get(family, "")) for family, score in support.items()}


def _rfs_family_scores(
    dataset: str,
    *,
    stage_rows: list[dict[str, Any]],
    expansion_rows: list[dict[str, Any]],
) -> dict[str, tuple[float, str]]:
    support: dict[str, float] = defaultdict(float)
    representatives: dict[str, str] = {}
    rep_score: dict[str, tuple[float, float]] = {}
    for row in stage_rows:
        answer = _row_answer(row)
        if not _is_candidate(answer):
            continue
        family = answer_family_key(dataset, answer)
        support[family] += 1.0
        score = (_row_confidence(row), evidence_quality(row))
        if family not in rep_score or score > rep_score[family]:
            representatives[family] = answer
            rep_score[family] = score
    for row in _valid_expansion_rows(expansion_rows):
        answer = _row_answer(row)
        family = answer_family_key(dataset, answer)
        support[family] += 1.0
        score = (_row_confidence(row), evidence_quality(row))
        if family not in rep_score or score > rep_score[family]:
            representatives[family] = answer
            rep_score[family] = score
    return {family: (round(score, 6), representatives.get(family, "")) for family, score in support.items()}


def _acs_answer_support(dataset: str, scored: dict[str, tuple[float, str]]) -> dict[str, float]:
    del dataset
    support: dict[str, float] = {}
    for _, (score, answer) in scored.items():
        if answer:
            support[answer] = round(max(float(support.get(answer, 0.0)), float(score)), 6)
    return support


def _expansion_validation_bonus(row: dict[str, Any]) -> float:
    mode = str(row.get("expansion_mode") or "")
    if row.get("expansion_validation_pass") is True:
        return 0.5
    if mode in {"mc_choice_shuffle", "strategyqa_dual_polarity"}:
        return 0.25
    return 0.0


def _expansion_family_count(dataset: str, rows: list[dict[str, Any]], family: str) -> int:
    return sum(1 for row in _valid_expansion_rows(rows) if answer_family_key(dataset, _row_answer(row)) == family)


def _mc_shuffle_family_count(dataset: str, rows: list[dict[str, Any]], family: str) -> int:
    return sum(
        1
        for row in _valid_expansion_rows(rows)
        if row.get("expansion_mode") == "mc_choice_shuffle"
        and row.get("expansion_validation_pass") is True
        and answer_family_key(dataset, _row_answer(row)) == family
    )


def _has_single_blocked_expansion(dataset: str, rows: list[dict[str, Any]], leader_family: str) -> bool:
    families = {
        answer_family_key(dataset, _row_answer(row))
        for row in _valid_expansion_rows(rows)
        if answer_family_key(dataset, _row_answer(row)) != leader_family
    }
    return any(_expansion_family_count(dataset, rows, family) == 1 for family in families)


def _pairwise_duel_promotions(
    *,
    dataset: str,
    stage_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    leader_family: str,
    require_stage_a_challenger_support: bool,
    min_wins: int,
    mode_names: tuple[str, ...] | list[str] = ("mc_blind_pairwise_duel",),
) -> list[tuple[int, int, int, str]]:
    if dataset not in _MC_DATASETS:
        return []
    allowed_modes = {str(mode) for mode in mode_names}
    totals: dict[str, int] = defaultdict(int)
    wins: dict[str, int] = defaultdict(int)
    representatives: dict[str, str] = {}
    stage_counts: dict[str, int] = {
        answer_family_key(dataset, _row_answer(row)): _stage_family_count(dataset, stage_rows, answer_family_key(dataset, _row_answer(row)))
        for row in stage_rows
        if _is_candidate(_row_answer(row))
    }
    for row in selection_rows:
        if row.get("expansion_mode") not in allowed_modes or row.get("pairwise_validation_pass") is not True:
            continue
        challenger_family = str(row.get("pairwise_challenger_family") or "")
        if not challenger_family or challenger_family == leader_family:
            continue
        challenger_answer = str(row.get("pairwise_challenger_answer") or "")
        if not _is_candidate(challenger_answer):
            continue
        if require_stage_a_challenger_support and not _stage_has_family(dataset, stage_rows, challenger_family):
            continue
        totals[challenger_family] += 1
        representatives.setdefault(challenger_family, challenger_answer)
        if str(row.get("pairwise_winner_family") or "") == challenger_family:
            wins[challenger_family] += 1
    eligible: list[tuple[int, int, int, str]] = []
    for family, total in totals.items():
        win_count = wins.get(family, 0)
        if win_count < int(min_wins):
            continue
        eligible.append((win_count, total, stage_counts.get(family, 0), representatives.get(family, "")))
    return eligible


def _strategyqa_minority_promotions(
    *,
    dataset: str,
    stage_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    leader_family: str,
    require_stage_a_challenger_support: bool,
) -> list[tuple[int, int, int, str]]:
    if dataset != "strategyqa":
        return []
    probe_counts: dict[str, int] = defaultdict(int)
    representatives: dict[str, str] = {}
    for row in selection_rows:
        if row.get("expansion_mode") != "strategyqa_minority_resample" or row.get("expansion_validation_pass") is not True:
            continue
        answer = _row_answer(row)
        family = answer_family_key(dataset, answer)
        if family == leader_family or not _is_candidate(answer):
            continue
        if require_stage_a_challenger_support and not _stage_has_family(dataset, stage_rows, family):
            continue
        probe_counts[family] += 1
        representatives.setdefault(family, answer)
    eligible: list[tuple[int, int, int, str]] = []
    for family, probe_count in probe_counts.items():
        stage_count = _stage_family_count(dataset, stage_rows, family)
        if probe_count >= 2 and stage_count + probe_count >= 3:
            eligible.append((stage_count + probe_count, stage_count, probe_count, representatives.get(family, "")))
    return eligible


def _pairwise_support(dataset: str, stage_support: dict[str, float], selection_rows: list[dict[str, Any]]) -> dict[str, float]:
    support = {answer: float(value) for answer, value in stage_support.items()}
    for row in selection_rows:
        mode = str(row.get("expansion_mode") or "")
        if mode in {"mc_blind_pairwise_duel", "gpqa_unanimous_pairwise_duel"}:
            if row.get("pairwise_validation_pass") is not True:
                continue
            winner = str(row.get("pairwise_winner_answer") or "")
            if _is_candidate(winner):
                support[winner] = support.get(winner, 0.0) + 0.25
        elif mode == "strategyqa_minority_resample" and row.get("expansion_validation_pass") is True:
            answer = _row_answer(row)
            if _is_candidate(answer):
                support[answer] = support.get(answer, 0.0) + 1.0
    return {answer: round(value, 6) for answer, value in support.items()}


def _has_pairwise_candidate_blocked(dataset: str, selection_rows: list[dict[str, Any]], leader_family: str) -> bool:
    if dataset not in _MC_DATASETS:
        return False
    return any(
        row.get("expansion_mode") in {"mc_blind_pairwise_duel", "gpqa_unanimous_pairwise_duel"}
        and row.get("pairwise_validation_pass") is True
        and str(row.get("pairwise_challenger_family") or "") != leader_family
        for row in selection_rows
    )


def _has_strategyqa_probe_blocked(dataset: str, selection_rows: list[dict[str, Any]], leader_family: str) -> bool:
    if dataset != "strategyqa":
        return False
    return any(
        row.get("expansion_mode") == "strategyqa_minority_resample"
        and row.get("expansion_validation_pass") is True
        and answer_family_key(dataset, _row_answer(row)) != leader_family
        for row in selection_rows
    )


def _has_non_answer_challenger(dataset: str, rows: list[dict[str, Any]], leader_family: str) -> bool:
    return any(
        _non_answer_family(dataset, _row_answer(row)) and answer_family_key(dataset, _row_answer(row)) != leader_family
        for row in rows
    )


def _non_answer_family(dataset: str, answer: str) -> bool:
    if dataset not in _SPAN_DATASETS:
        return False
    normalized = _normalized_span(answer)
    phrases = (
        "not specified",
        "not stated",
        "not provided",
        "not mentioned",
        "not in context",
        "cannot determine",
        "unknown",
    )
    return any(phrase in normalized for phrase in phrases)


def _safe_promoted_decision(
    *,
    dataset: str,
    stage_support: dict[str, float],
    stage_rows: list[dict[str, Any]],
    stage_winner: str,
    winner: str,
    resolver: str,
    source: str,
) -> CredAggregateDecision:
    support = dict(stage_support)
    support.setdefault(stage_winner, 0.0)
    support[winner] = max(
        float(support.get(winner, 0.0)),
        float(_stage_family_count(dataset, stage_rows, answer_family_key(dataset, winner))) + 0.5,
    )
    return CredAggregateDecision(
        winner,
        {answer: round(float(value), 6) for answer, value in support.items()},
        resolver,
        answer_family_key(dataset, winner) != answer_family_key(dataset, stage_winner),
        source,
    )


def _safe_challenger_rows(dataset: str, stage_rows: list[dict[str, Any]], stage_winner: str) -> list[dict[str, Any]]:
    leading_family = answer_family_key(dataset, stage_winner)
    grouped, ordered_families = _stage_candidate_groups(dataset, stage_rows)
    challengers = [
        max(grouped[family], key=lambda row: (_row_confidence(row), evidence_quality(row), -len(_row_answer(row))))
        for family in ordered_families
        if family != leading_family
    ]
    challengers.sort(key=lambda row: (_row_confidence(row), evidence_quality(row)), reverse=True)
    return challengers


def _deterministic_repair_allows(dataset: str, leader: str, challenger: str, context: str) -> bool:
    if dataset in _MATH_DATASETS:
        return _canonical_math_text(leader) == _canonical_math_text(challenger) and str(leader).strip() != str(challenger).strip()
    if dataset in _SPAN_DATASETS:
        return _hotpot_span_repair_allows(leader, challenger, context)
    return False


def _repair_only_v6_allows(dataset: str, leader: str, challenger: str, context: str) -> bool:
    if dataset in _SPAN_DATASETS:
        return _hotpot_span_repair_allows(leader, challenger, context)
    if dataset not in _MATH_DATASETS or str(leader).strip() == str(challenger).strip():
        return False
    return _math_scorer_canonical_repair_allows(leader, challenger)


def _repair_bank_v8_allows(dataset: str, leader: str, challenger: str, context: str) -> bool:
    if _repair_only_v6_allows(dataset, leader, challenger, context):
        return True
    if dataset not in _MATH_DATASETS or str(leader).strip() == str(challenger).strip():
        return False
    return _math_repair_bank_v8_allows(leader, challenger)


def _math_repair_bank_v8_allows(leader: str, challenger: str) -> bool:
    leader_text = str(leader or "").strip()
    challenger_text = str(challenger or "").strip()
    if not leader_text or not challenger_text:
        return False
    if _canonical_math_text(_math_pi_ascii_text(leader_text)) != _canonical_math_text(_math_pi_ascii_text(challenger_text)):
        return False
    leader_ascii = _math_pi_ascii_text(leader_text)
    challenger_ascii = _math_pi_ascii_text(challenger_text)
    if re.search(r"\\(?:boxed|text|mathrm|textrm|mbox)\s*\{", leader_ascii) and not re.search(
        r"\\(?:boxed|text|mathrm|textrm|mbox)\s*\{",
        challenger_ascii,
    ):
        return True
    return _canonical_unit_text(leader_ascii) == _canonical_unit_text(challenger_ascii) and _math_has_unit_spacing_repair(
        leader_ascii,
        challenger_ascii,
    )


def _math_has_unit_spacing_repair(leader: str, challenger: str) -> bool:
    return bool(re.search(r"\d\s+[a-zA-Z%]", str(leader or ""))) and not bool(
        re.search(r"\d\s+[a-zA-Z%]", str(challenger or ""))
    )


def _mc_option_text_letter(answer: str, option_texts: tuple[str, ...] | list[str]) -> str:
    raw = str(answer or "").strip()
    if not raw or re.fullmatch(r"[A-J]", raw.upper()):
        return ""
    normalized_answer = _normalized_option_text(raw)
    if not normalized_answer:
        return ""
    matches = [
        index
        for index, option in enumerate(option_texts)
        if _normalized_option_text(str(option or "")) == normalized_answer
    ]
    if len(matches) != 1:
        return ""
    return chr(ord("A") + int(matches[0]))


def _normalized_option_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"^\s*(?:option|choice|answer)?\s*[A-J]\s*[\).:\-]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text.lower()).strip()
    return re.sub(r"\s+", " ", text)


def _math_scorer_canonical_repair_allows(leader: str, challenger: str) -> bool:
    leader_text = str(leader or "").strip()
    challenger_text = str(challenger or "").strip()
    if _canonical_math_text(_math_pi_ascii_text(leader_text)) != _canonical_math_text(_math_pi_ascii_text(challenger_text)):
        return False
    leader_lower = leader_text.lower()
    challenger_lower = challenger_text.lower()
    leader_has_ascii_pi = bool(re.search(r"(?<![a-z])pi(?![a-z])", leader_lower))
    challenger_has_ascii_pi = bool(re.search(r"(?<![a-z])pi(?![a-z])", challenger_lower))
    leader_has_pi_symbol = any(symbol in leader_text for symbol in ("π", "蟺"))
    challenger_has_pi_symbol = any(symbol in challenger_text for symbol in ("π", "蟺"))
    if leader_has_pi_symbol and challenger_has_ascii_pi and not challenger_has_pi_symbol:
        return True
    if leader_has_ascii_pi and challenger_has_pi_symbol:
        return False
    leader_has_textual_inf = bool(re.search(r"(?<![a-z])(?:infinity|inf|oo)(?![a-z])", leader_lower))
    challenger_has_latex_inf = "\\infty" in challenger_lower
    leader_has_latex_inf = "\\infty" in leader_lower
    if leader_has_textual_inf and challenger_has_latex_inf:
        return True
    if leader_has_latex_inf and not challenger_has_latex_inf:
        return False
    return False


def _math_pi_ascii_text(value: str) -> str:
    return str(value or "").replace("π", "pi").replace("蟺", "pi")


def _math_equivalence_repair_v2_allows(dataset: str, leader: str, challenger: str, modes: set[str]) -> bool:
    if dataset not in _MATH_DATASETS or "math_equivalence_repair_v2" not in modes:
        return False
    if str(leader).strip() == str(challenger).strip():
        return False
    leader_interval = _canonical_interval_text(leader)
    challenger_interval = _canonical_interval_text(challenger)
    if leader_interval and challenger_interval:
        return leader_interval == challenger_interval
    if _canonical_unit_text(leader) == _canonical_unit_text(challenger):
        return True
    return _math_equivalent(leader, challenger)


def _hotpot_context_span_repair_v2_allows(dataset: str, leader: str, challenger: str, context: str, modes: set[str]) -> bool:
    if dataset not in _SPAN_DATASETS or "hotpot_context_span_repair_v2" not in modes:
        return False
    return _hotpot_span_repair_allows(leader, challenger, context)


def _tool_verification_allows(dataset: str, leader: str, challenger: str, context: str) -> bool:
    if dataset in _MATH_DATASETS:
        return _math_equivalent(leader, challenger) and str(leader).strip() != str(challenger).strip()
    if dataset in _SPAN_DATASETS:
        return _hotpot_span_repair_allows(leader, challenger, context)
    return False


def _canonical_math_text(value: str) -> str:
    text = str(value or "").strip().lower()
    replacements = {
        "\\left": "",
        "\\right": "",
        "\\,": "",
        "\\ ": " ",
        "$": "",
        "`": "",
        "∞": "inf",
        "\\infty": "inf",
        "infinity": "inf",
        "+inf": "inf",
        "+oo": "inf",
        "oo": "inf",
        "\\cot": "cot",
        "\\tan": "tan",
        "\\sin": "sin",
        "\\cos": "cos",
        "\\sec": "sec",
        "\\csc": "csc",
        "\\log": "log",
        "\\ln": "ln",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"\\boxed\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s*([,;:()\[\]{}=+\-*/^])\s*", r"\1", text)
    text = text.replace("{", "").replace("}", "")
    return text


def _math_equivalent(left: str, right: str) -> bool:
    if _canonical_math_text(left) == _canonical_math_text(right):
        return True
    left_interval = _canonical_interval_text(left)
    right_interval = _canonical_interval_text(right)
    if left_interval and right_interval:
        return left_interval == right_interval
    try:
        import sympy as sp
        from sympy.parsing.sympy_parser import (
            convert_xor,
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )
    except Exception:
        return False
    transformations = standard_transformations + (implicit_multiplication_application, convert_xor)
    try:
        left_expr = parse_expr(_sympy_math_text(left), transformations=transformations, evaluate=True)
        right_expr = parse_expr(_sympy_math_text(right), transformations=transformations, evaluate=True)
        return bool(sp.simplify(left_expr - right_expr) == 0)
    except Exception:
        return False


def _sympy_math_text(value: str) -> str:
    text = _canonical_math_text(value)
    text = text.replace("inf", "oo")
    text = text.replace("\\pi", "pi")
    text = text.replace("π", "pi")
    text = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", text)
    text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", text)
    return text


def _canonical_interval_text(value: str) -> str:
    text = _canonical_math_text(value)
    compact = re.sub(r"\s+", "", text)
    match = re.fullmatch(r"([\(\[])([^,]+),([^,\)\]]+)([\)\]])", compact)
    if match:
        left, lower, upper, right = match.groups()
        return f"{left}{_canonical_interval_endpoint(lower)},{_canonical_interval_endpoint(upper)}{right}"
    chain = re.fullmatch(r"([^<>=]+)(<=|<)[a-z]+(<=|<)([^<>=]+)", compact)
    if chain:
        lower, left_op, right_op, upper = chain.groups()
        left = "[" if left_op == "<=" else "("
        right = "]" if right_op == "<=" else ")"
        return f"{left}{_canonical_interval_endpoint(lower)},{_canonical_interval_endpoint(upper)}{right}"
    return ""


def _canonical_interval_endpoint(value: str) -> str:
    endpoint = _canonical_math_text(value)
    endpoint = endpoint.replace("+inf", "inf").replace("+oo", "inf").replace("oo", "inf")
    endpoint = endpoint.replace("-infinity", "-inf").replace("-oo", "-inf")
    return endpoint


def _canonical_unit_text(value: str) -> str:
    text = _canonical_math_text(value)
    text = re.sub(r"(?<=\d)\s+(?=[a-zA-Z%]+(?:\b|[-/]))", "", text)
    text = re.sub(r"\s*([*/^])\s*", r"\1", text)
    return text


def _choice_letter_from_text(text: str) -> str:
    cleaned = str(text or "").strip().upper()
    if not cleaned:
        return ""
    if re.fullmatch(r"[A-J]", cleaned):
        return cleaned
    match = re.match(r"^\(?([A-J])\)?(?:[.)]|:|,|-)?(?:\s|$)", cleaned)
    if match:
        return match.group(1)
    option_match = re.search(r"\b(?:OPTION|CHOICE|ANSWER)\s*(?:IS|:)?\s*([A-J])\b", cleaned)
    return option_match.group(1) if option_match else ""


def _hotpot_span_repair_allows(leader: str, challenger: str, context: str) -> bool:
    leader_tokens = _span_tokens(leader)
    challenger_tokens = _span_tokens(challenger)
    if not leader_tokens or not challenger_tokens:
        return False
    if _unsafe_span_answer(challenger):
        return False
    if not _context_supports_answer(context, challenger):
        return False
    span_index = _subsequence_index(challenger_tokens, leader_tokens)
    if span_index < 0:
        return False
    extra = challenger_tokens[:span_index] + challenger_tokens[span_index + len(leader_tokens) :]
    if not extra or len(extra) > 3:
        return False
    if any(token in _HOTPOT_BANNED_EXTRA_TOKENS for token in extra):
        return False
    return all(token in _HOTPOT_ALLOWED_EXTRA_TOKENS for token in extra)


def _context_supports_answer(context: str, answer: str) -> bool:
    context_norm = _normalized_span(context)
    answer_norm = _normalized_span(answer)
    return bool(answer_norm and answer_norm in context_norm)


def _unsafe_span_answer(answer: str) -> bool:
    normalized = _normalized_span(answer)
    return any(phrase in normalized for phrase in ("not mentioned", "unknown", "not in context", "cannot determine"))


def _span_tokens(value: str) -> list[str]:
    return _normalized_span(value).split()


def _normalized_span(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text.lower()).strip()
    return re.sub(r"\s+", " ", text)


def _subsequence_index(items: list[str], needle: list[str]) -> int:
    if not needle or len(needle) > len(items):
        return -1
    for index in range(0, len(items) - len(needle) + 1):
        if items[index : index + len(needle)] == needle:
            return index
    return -1


def _verification_support(stage_support: dict[str, float], verifier_rows: list[dict[str, Any]], stage_winner: str) -> dict[str, float]:
    support = {answer: float(value) for answer, value in stage_support.items()}
    if stage_winner:
        support.setdefault(stage_winner, 0.0)
    for row in verifier_rows:
        answer = _row_answer(row)
        if not _is_candidate(answer):
            continue
        leader_score = _row_float(row, "leader_score", default=0.5)
        challenger_score = _row_float(row, "challenger_score", default=0.5)
        if _row_bool(row, "promote"):
            support[answer] = max(float(support.get(answer, 0.0)), 4.0 + challenger_score + _row_confidence(row))
            if stage_winner:
                support[stage_winner] = max(float(support.get(stage_winner, 0.0)), 4.0 + leader_score)
        elif stage_winner:
            support[stage_winner] = max(float(support.get(stage_winner, 0.0)), 4.0 + leader_score + _row_confidence(row))
    return {answer: round(value, 6) for answer, value in support.items()}


def _row_answer(row: dict[str, Any]) -> str:
    payload = row.get("validated_output") if isinstance(row.get("validated_output"), dict) else {}
    if str(payload.get("format_warning") or row.get("format_warning") or "") in {"answer_contains_reasoning_leak", "answer_too_long_for_final_slot"}:
        return ""
    return str(row.get("normalized_answer") or row.get("prediction") or "").strip()


def _dataset_from_rows(rows: list[dict[str, Any]]) -> str:
    return str(next((row.get("dataset") for row in rows if row.get("dataset")), ""))


def _stage_candidate_groups(dataset: str, rows: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ordered_families: list[str] = []
    for row in rows:
        answer = _row_answer(row)
        if not _is_candidate(answer):
            continue
        family = answer_family_key(dataset, answer)
        if family not in grouped:
            ordered_families.append(family)
        grouped[family].append(row)
    return grouped, ordered_families


def _stage_support_counts(
    dataset: str,
    grouped: dict[str, list[dict[str, Any]]],
    ordered_families: list[str],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for family in ordered_families:
        representative = _representative_answer(dataset, grouped[family])
        counts[representative] = counts.get(representative, 0) + len(grouped[family])
    return counts


def _representative_answer(dataset: str, rows: list[dict[str, Any]]) -> str:
    del dataset
    if not rows:
        return ""
    row = max(
        rows,
        key=lambda item: (
            _row_confidence(item),
            evidence_quality(item),
            -len(_row_answer(item)),
        ),
    )
    return _row_answer(row)


def _stage_has_family(dataset: str, rows: list[dict[str, Any]], family: str) -> bool:
    return any(answer_family_key(dataset, _row_answer(row)) == family for row in rows)


def _stage_family_count(dataset: str, rows: list[dict[str, Any]], family: str) -> int:
    return sum(1 for row in rows if answer_family_key(dataset, _row_answer(row)) == family)


def _row_confidence(row: dict[str, Any]) -> float:
    value = row.get("confidence_value")
    if value is None and isinstance(row.get("validated_output"), dict):
        value = row["validated_output"].get("confidence")
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def _row_risk_level(row: dict[str, Any]) -> str:
    payload = row.get("validated_output") if isinstance(row.get("validated_output"), dict) else {}
    normalized = str(row.get("risk_level") or payload.get("risk_level") or "").strip().lower()
    return normalized if normalized in {"none", "low", "medium", "high"} else "none"


def _is_candidate(answer: str) -> bool:
    return str(answer or "").strip().lower() not in _NON_ANSWERS


def _has_concrete_evidence(row: dict[str, Any], min_chars: int) -> bool:
    payload = row.get("validated_output") if isinstance(row.get("validated_output"), dict) else {}
    text = str(row.get("key_evidence") or payload.get("key_evidence") or payload.get("evidence") or "").strip()
    return len(text) >= int(min_chars)


def _row_bool(row: dict[str, Any], key: str) -> bool:
    payload = row.get("validated_output") if isinstance(row.get("validated_output"), dict) else {}
    value = payload.get(key, row.get(key))
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"true", "yes", "1"}


def _row_float(row: dict[str, Any], key: str, *, default: float) -> float:
    payload = row.get("validated_output") if isinstance(row.get("validated_output"), dict) else {}
    value = payload.get(key, row.get(key))
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _mean(values) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return sum(items) / len(items)


def _coarse_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()
