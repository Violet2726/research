"""BRD-MAD 的样本级执行与诊断。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

from research_experiments.core.data.datasets import DatasetSample, load_split_ids, select_samples
from research_experiments.core.data.evaluation import normalize_prediction, score_prediction
from research_experiments.core.execution.runner_common import iter_indexed_batch
from research_experiments.families.blind_reconstructive_mad.algorithms import (
    ReviewerBoard,
    build_reviewer_board,
    build_stage_a_decision,
    decide_existing_candidate_quorum,
    reviewer_error_correlation,
)
from research_experiments.families.blind_reconstructive_mad.config import BrdMadExperimentConfig, BrdProtocolConfig
from research_experiments.families.blind_reconstructive_mad.prompts import (
    build_reviewer_messages,
    build_stage_a_messages,
)
from research_experiments.family_runtime.common import resolve_phase_split_name, safe_mean, safe_ratio
from research_experiments.family_runtime.output_protocols import (
    FREE_TEXT_ANSWER_PROTOCOL_V1,
    execute_output_protocol_turn,
)


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
    """Execute one tagged free-text turn without relying on another family."""

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
    answer = str(result.validated_output.get("final_answer") or result.validated_output.get("answer") or "")
    normalized = normalize_prediction(dataset, answer) if answer else ""
    return {
        "run_id": run_id,
        "dataset": dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "method_name": method_name,
        "method_type": method_type,
        "round_index": round_index,
        "agent_id": agent_id,
        "role": role,
        "agent_role": agent_role,
        "prompt_hash": result.prompt_hash,
        "prediction": normalized,
        "normalized_answer": normalized,
        "score": None,
        "output_status": result.output_status,
        "prompt_tokens": float(result.usage.get("prompt_tokens") or 0.0),
        "completion_tokens": float(result.usage.get("completion_tokens") or 0.0),
        "total_tokens": float(result.usage.get("total_tokens") or 0.0),
        "latency_ms": float(result.response_payload.get("latency_ms") or 0.0),
        "cache_hit": result.cache_hit,
        "request_error": result.request_error,
        "request_status": result.request_status,
        "raw_finish_reason": result.raw_finish_reason,
        "output_protocol": result.output_protocol,
        "protocol_parse_status": result.protocol_parse_status,
        "protocol_parse_error": result.protocol_parse_error,
        "reason_present": result.reason_present,
        "request_count": result.request_count,
        "cache_request_count": result.cache_request_count,
        "network_request_count": result.network_request_count,
        "raw_prompt_tokens": float(result.usage.get("prompt_tokens") or 0.0),
        "raw_completion_tokens": float(result.usage.get("completion_tokens") or 0.0),
        "raw_total_tokens": float(result.usage.get("total_tokens") or 0.0),
        "raw_latency_ms": float(result.response_payload.get("latency_ms") or 0.0),
        "visible_peer_count": visible_peer_count,
        "payload": result.payload,
        "assistant_text": str(result.response_payload.get("assistant_text") or ""),
        "provider_reasoning_text": str(result.response_payload.get("provider_reasoning_text") or ""),
        "validated_output": result.validated_output,
    }


def append_outputs(
    *,
    sample_results,
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
    """Write completed sample outputs in the shared artifact shape."""

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
    """Create a matched no-communication prediction record."""

    del method
    prompt_tokens = _sum(turn_rows, "prompt_tokens")
    completion_tokens = _sum(turn_rows, "completion_tokens")
    total_tokens = _sum(turn_rows, "total_tokens")
    latency_ms = _sum(turn_rows, "latency_ms")
    return {
        "run_id": run_id,
        "dataset": benchmark_slug,
        "split": split_name,
        "sample_id": sample.sample_id,
        "task": sample.metadata.get("task"),
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
        "unchanged_correct": final_score == 1.0,
        "unchanged_wrong": final_score < 1.0,
        "triggered": False,
        "router_reasons": [],
        "resolver": "no_comm_control",
        "survival_support": {},
        "protocol_failures_per_question": sum(row.get("protocol_parse_status") == "failed" for row in turn_rows),
        "reason_missing_turns_per_question": sum(not row.get("reason_present") for row in turn_rows),
        "vote_counts": vote_counts,
    }


def run_brd_batch(
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    samples: list[DatasetSample],
    experiment: BrdMadExperimentConfig,
    protocol: BrdProtocolConfig,
    active_methods: list[str],
    backbone,
    provider,
    cache,
    throttle,
) -> Iterator[tuple[int, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]]:
    worker = partial(
        _run_brd_sample,
        run_id=run_id,
        benchmark_slug=benchmark_slug,
        split_name=split_name,
        experiment=experiment,
        protocol=protocol,
        active_methods=active_methods,
        backbone=backbone,
        provider=provider,
        cache=cache,
        throttle=throttle,
    )
    for sample_index, result in iter_indexed_batch(
        samples,
        worker=worker,
        max_concurrent_requests=experiment.max_concurrent_requests,
    ):
        yield (sample_index, *result)


def _run_brd_sample(
    sample: DatasetSample,
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    experiment: BrdMadExperimentConfig,
    protocol: BrdProtocolConfig,
    active_methods: list[str],
    backbone,
    provider,
    cache,
    throttle,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    stage_rows = _run_stage_a(
        run_id=run_id,
        sample=sample,
        dataset=benchmark_slug,
        split_name=split_name,
        experiment=experiment,
        protocol=protocol,
        backbone=backbone,
        provider=provider,
        cache=cache,
        throttle=throttle,
    )
    stage = build_stage_a_decision(stage_rows)
    stage_score = score_prediction(benchmark_slug, stage.anchor_answer, sample.reference_answer)
    candidate_oracle = any(
        score_prediction(benchmark_slug, family.answer, sample.reference_answer) == 1.0 for family in stage.families
    )
    router_row = {
        "run_id": run_id,
        "dataset": benchmark_slug,
        "split": split_name,
        "sample_id": sample.sample_id,
        "policy_name": "brd_answer_disagreement_v1",
        "triggered": stage.triggered,
        "trigger_mode": protocol.trigger_mode,
        "anchor_answer": stage.anchor_answer,
        "anchor_score": stage_score,
        "candidate_oracle_correct": candidate_oracle,
        "candidate_count": len(stage.families),
        "vote_counts": stage.vote_counts,
        "disagreement_pattern": stage.disagreement_pattern,
        "stage_protocol_failures": sum(row.get("protocol_parse_status") == "failed" for row in stage_rows),
    }
    all_turns = list(stage_rows)
    debate_rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    shared_panels: dict[
        str,
        tuple[list[dict[str, Any]], list[ReviewerBoard], list[str | None]],
    ] = {}
    for method_index, method_name in enumerate(active_methods):
        method_turns = list(stage_rows)
        reviewer_turns: list[dict[str, Any]] = []
        boards: list[ReviewerBoard] = []
        rendered_boards: list[str | None] = []
        if stage.triggered:
            panel_key, request_method, show_support = _review_panel_spec(method_name)
            if panel_key in shared_panels:
                reviewer_turns, boards, rendered_boards = shared_panels[panel_key]
            else:
                reviewer_requests: list[tuple[ReviewerBoard, dict[str, Any], str | None]] = []
                for reviewer_id in range(1, protocol.reviewer_count + 1):
                    board = build_reviewer_board(
                        stage,
                        global_seed=experiment.global_seed,
                        sample_id=sample.sample_id,
                        method_name=panel_key,
                        reviewer_id=reviewer_id,
                        show_support=show_support,
                    )
                    if request_method == "conditional_resample_3":
                        rendered_board = None
                        # This comparator is conditional extra sampling, not a
                        # reviewer.  Reuse the exact SC/Stage-A output contract
                        # and avoid a family-specific completion cap.
                        messages = build_stage_a_messages(sample, reviewer_id)
                        request_max_tokens = None
                        agent_role = "conditional_sc_resample"
                    else:
                        rendered_board = _bounded_board(board, protocol.representative_max_chars)
                        messages = build_reviewer_messages(
                            sample,
                            candidate_board=rendered_board,
                            method_name=request_method,
                        )
                        request_max_tokens = protocol.reviewer_max_tokens
                        agent_role = "blind_reconstructive_reviewer"
                    reviewer_requests.append(
                        (
                            board,
                            {
                            "run_id": run_id,
                            "dataset": benchmark_slug,
                            "split_name": split_name,
                            "sample": sample,
                            "method_name": panel_key,
                            "method_type": "brd_review",
                            "round_index": 1,
                            "agent_id": reviewer_id,
                            "role": "reviewer",
                            "visible_peer_count": 0,
                            "messages": messages,
                            "backbone": backbone,
                            "provider": provider,
                            "cache": cache,
                            "throttle": throttle,
                            "temperature": protocol.reviewer_temperature,
                            "top_p": protocol.top_p,
                            "seed": experiment.global_seed + 10_000 + method_index * 10 + reviewer_id,
                            "agent_role": agent_role,
                            "max_tokens": request_max_tokens,
                            },
                            rendered_board,
                        )
                    )
                boards = [board for board, _, _ in reviewer_requests]
                rendered_boards = [rendered for _, _, rendered in reviewer_requests]
                with ThreadPoolExecutor(max_workers=protocol.reviewer_count) as executor:
                    reviewer_turns = list(executor.map(lambda item: _execute_turn(**item[1]), reviewer_requests))
                shared_panels[panel_key] = (reviewer_turns, boards, rendered_boards)
                all_turns.extend(reviewer_turns)
                debate_rows.extend(
                    _reviewer_message_rows(
                        run_id,
                        benchmark_slug,
                        split_name,
                        sample,
                        panel_key,
                        reviewer_turns,
                    )
                )
            method_turns.extend(reviewer_turns)
            unanimous = method_name in {
                "sgsa_unanimous_3",
                "sgsa_visible_support_3",
                "concise_brd_unanimous_3",
            }
            quorum = decide_existing_candidate_quorum(
                stage,
                [str(row.get("normalized_answer") or "") for row in reviewer_turns],
                strong_majority_quorum=protocol.strong_majority_quorum,
                default_quorum=protocol.reviewer_count if unanimous else protocol.default_quorum,
            )
        else:
            quorum = decide_existing_candidate_quorum(
                stage,
                [],
                strong_majority_quorum=protocol.strong_majority_quorum,
                default_quorum=protocol.default_quorum,
            )
        predictions.append(
            _prediction_row(
                run_id=run_id,
                dataset=benchmark_slug,
                split_name=split_name,
                sample=sample,
                backbone_name=backbone.name,
                method_name=method_name,
                stage=stage,
                quorum=quorum,
                stage_rows=stage_rows,
                reviewer_turns=reviewer_turns,
                method_turns=method_turns,
                candidate_oracle=candidate_oracle,
                boards=boards,
                rendered_boards=rendered_boards,
            )
        )
    return all_turns, debate_rows, [router_row], predictions


def _review_panel_spec(method_name: str) -> tuple[str, str, bool]:
    """Map derived selectors to one physical three-call reviewer panel."""

    if method_name in {"gsa_quorum_3", "sgsa_unanimous_3"}:
        return "gsa_shared_panel", "gsa_quorum_3", False
    if method_name == "sgsa_visible_support_3":
        return "gsa_visible_panel", "sgsa_visible_support_3", True
    if method_name in {"concise_brd_quorum_3", "concise_brd_unanimous_3"}:
        return "concise_brd_shared_panel", "concise_brd_quorum_3", False
    return method_name, method_name, method_name == "brd_visible_support_3"


def _run_stage_a(
    *,
    run_id: str,
    sample: DatasetSample,
    dataset: str,
    split_name: str,
    experiment: BrdMadExperimentConfig,
    protocol: BrdProtocolConfig,
    backbone,
    provider,
    cache,
    throttle,
) -> list[dict[str, Any]]:
    requests = []
    for agent_id in range(1, protocol.stage_a_candidates + 1):
        # These calls intentionally match the shared sc_5 implementation:
        # same free-CoT messages, temperature/top-p, and seed global_seed+i.
        requests.append(
            {
                "run_id": run_id,
                "dataset": dataset,
                "split_name": split_name,
                "sample": sample,
                "method_name": "brd_stage_a_shared",
                "method_type": "brd_stage_a",
                "round_index": 0,
                "agent_id": agent_id,
                "role": "initial",
                "visible_peer_count": 0,
                "messages": build_stage_a_messages(sample, agent_id),
                "backbone": backbone,
                "provider": provider,
                "cache": cache,
                "throttle": throttle,
                "temperature": protocol.stage_a_temperature,
                "top_p": protocol.top_p,
                "seed": experiment.global_seed + agent_id - 1,
                "agent_role": "sc_5_aligned_stage_a",
                # `sc_5` has no family-specific cap.  Keeping this None makes
                # cache fingerprints identical to the standard control.
                "max_tokens": None,
            }
        )
    with ThreadPoolExecutor(max_workers=protocol.stage_a_candidates) as executor:
        return list(executor.map(lambda kwargs: _execute_turn(**kwargs), requests))


def _bounded_board(board: ReviewerBoard, max_chars: int) -> str:
    return board.rendered(max_chars=max_chars)


def _reviewer_message_rows(
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    method_name: str,
    turns: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            "run_id": run_id,
            "dataset": dataset,
            "split": split_name,
            "sample_id": sample.sample_id,
            "method_name": method_name,
            "round_index": 1,
            "sender_agent_id": int(turn["agent_id"]),
            "recipient_agent_id": 0,
            "sender_answer": str(turn.get("normalized_answer") or ""),
            "sender_reasoning": str(turn.get("assistant_text") or ""),
            "message_kind": "independent_blind_review",
        }
        for turn in turns
    ]


def _prediction_row(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    backbone_name: str,
    method_name: str,
    stage,
    quorum,
    stage_rows: list[dict[str, Any]],
    reviewer_turns: list[dict[str, Any]],
    method_turns: list[dict[str, Any]],
    candidate_oracle: bool,
    boards: list[ReviewerBoard],
    rendered_boards: list[str | None],
) -> dict[str, Any]:
    prediction = quorum.final_answer
    score = score_prediction(dataset, prediction, sample.reference_answer)
    initial_score = score_prediction(dataset, stage.anchor_answer, sample.reference_answer)
    reviewer_correctness = [
        score_prediction(dataset, str(turn.get("normalized_answer") or ""), sample.reference_answer) == 1.0
        if str(turn.get("normalized_answer") or "")
        else None
        for turn in reviewer_turns
    ]
    board_visibility = [
        all(f"Candidate {label}\n" in rendered for label in board.labels)
        for board, rendered in zip(boards, rendered_boards, strict=True)
        if rendered is not None
    ]
    return {
        "run_id": run_id,
        "dataset": dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "task": sample.metadata.get("task"),
        "method_name": method_name,
        "method_type": "brd",
        "model_name": backbone_name,
        "prediction": prediction,
        "gold": sample.reference_answer,
        "score": score,
        "initial_vote_prediction": stage.anchor_answer,
        "initial_vote_score": initial_score,
        "initial_vote_counts": stage.vote_counts,
        "initial_consensus": not stage.triggered,
        "final_vote_prediction": prediction,
        "final_vote_score": score,
        "final_vote_counts": quorum.reviewer_votes,
        "prompt_tokens_per_question": _sum(method_turns, "prompt_tokens"),
        "completion_tokens_per_question": _sum(method_turns, "completion_tokens"),
        "total_tokens_per_question": _sum(method_turns, "total_tokens"),
        "latency_ms_per_question": _sum(method_turns, "latency_ms"),
        "initial_prompt_tokens_per_question": _sum(stage_rows, "prompt_tokens"),
        "initial_completion_tokens_per_question": _sum(stage_rows, "completion_tokens"),
        "initial_total_tokens_per_question": _sum(stage_rows, "total_tokens"),
        "initial_latency_ms_per_question": _sum(stage_rows, "latency_ms"),
        "debate_prompt_tokens_per_question": _sum(reviewer_turns, "prompt_tokens"),
        "debate_completion_tokens_per_question": _sum(reviewer_turns, "completion_tokens"),
        "debate_total_tokens_per_question": _sum(reviewer_turns, "total_tokens"),
        "debate_latency_ms_per_question": _sum(reviewer_turns, "latency_ms"),
        "calls_per_question": len(method_turns),
        "debate_rounds": 1 if reviewer_turns else 0,
        "agent_count": len(stage_rows),
        "final_consensus": quorum.quorum_met,
        "initial_disagreement": stage.triggered,
        "vote_flipped": quorum.override_accepted,
        "corrected_by_debate": initial_score < 1.0 and score == 1.0,
        "harmed_by_debate": initial_score == 1.0 and score < 1.0,
        "unchanged_correct": not quorum.override_accepted and score == 1.0,
        "unchanged_wrong": not quorum.override_accepted and score < 1.0,
        "triggered": stage.triggered,
        "router_reasons": ["answer_disagreement"] if stage.triggered else [],
        "resolver": quorum.resolver,
        "survival_support": quorum.reviewer_votes,
        "protocol_failures_per_question": sum(row.get("protocol_parse_status") == "failed" for row in method_turns),
        "reason_missing_turns_per_question": sum(not row.get("reason_present") for row in method_turns),
        "vote_counts": stage.vote_counts,
        "candidate_oracle_correct": candidate_oracle,
        "candidate_answers": sorted(stage.candidate_answers),
        "disagreement_pattern": stage.disagreement_pattern,
        "anchor_support": stage.anchor_support,
        "quorum_required": quorum.quorum_required,
        "quorum_met": quorum.quorum_met,
        "override_accepted": quorum.override_accepted,
        "promoted_answer": quorum.promoted_answer,
        "shadow_answers": list(quorum.shadow_answers),
        "novel_answer_mode": "shadow",
        "reviewer_answers": [str(turn.get("normalized_answer") or "") for turn in reviewer_turns],
        "reviewer_correctness": reviewer_correctness,
        "label_permutations": [board.label_to_answer() for board in boards],
        "candidate_board_all_candidates_visible": all(board_visibility) if board_visibility else None,
        "candidate_board_char_counts": [len(rendered) for rendered in rendered_boards if rendered is not None],
        "candidate_board_truncated_rationale_counts": [
            rendered.count("[representative truncated]")
            for rendered in rendered_boards
            if rendered is not None
        ],
    }


def _sum(rows: list[dict[str, Any]], field: str) -> float:
    return sum(float(row.get(field) or 0.0) for row in rows)


def build_metrics(
    prediction_rows: list[dict[str, Any]],
    *,
    dataset_order: list[str],
    method_order: list[str],
    control_names: list[str],
) -> dict[str, Any]:
    """Build compact dataset, macro, and micro summaries for BRD artifacts."""

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        grouped[(str(row["dataset"]), str(row["model_name"]), str(row["method_name"]))].append(row)
    dataset_rows = [_summarize_rows(rows, dataset, model, method, "dataset") for (dataset, model, method), rows in grouped.items()]
    macro_rows = _macro_rows(dataset_rows)
    micro_grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        micro_grouped[(str(row["model_name"]), str(row["method_name"]))].append(row)
    micro_rows = [_summarize_rows(rows, "overall_micro", model, method, "micro") for (model, method), rows in micro_grouped.items()]
    summary = [*dataset_rows, *macro_rows, *micro_rows]
    _refresh_accuracy_comparisons(summary, control_names)
    dataset_rank = {name: index for index, name in enumerate(dataset_order)}
    method_rank = {name: index for index, name in enumerate(method_order)}
    summary.sort(
        key=lambda row: (
            {"overall": len(dataset_order), "overall_micro": len(dataset_order) + 1}.get(str(row["dataset"]), dataset_rank.get(str(row["dataset"]), 999)),
            method_rank.get(str(row["method_name"]), 999),
            str(row["model_name"]),
        )
    )
    return {"summary": summary}


def _summarize_rows(rows: list[dict[str, Any]], dataset: str, model_name: str, method_name: str, aggregate_kind: str) -> dict[str, Any]:
    count = len(rows)
    accuracy = safe_mean(float(row.get("score") or 0.0) for row in rows)
    initial = safe_mean(float(row.get("initial_vote_score") if row.get("initial_vote_score") is not None else row.get("score") or 0.0) for row in rows)
    total_tokens = safe_mean(float(row.get("total_tokens_per_question") or 0.0) for row in rows)
    corrected = sum(bool(row.get("corrected_by_debate")) for row in rows)
    harmed = sum(bool(row.get("harmed_by_debate")) for row in rows)
    triggered = sum(bool(row.get("triggered")) for row in rows)
    return {
        "dataset": dataset,
        "aggregate_kind": aggregate_kind,
        "model_name": model_name,
        "method_name": method_name,
        "method_type": rows[0]["method_type"],
        "question_count": count,
        "prediction_rows": count,
        "accuracy_mean": accuracy,
        "initial_vote_accuracy_mean": initial,
        "debate_gain_over_initial_vote": round(accuracy - initial, 6),
        "prompt_tokens_mean": safe_mean(float(row.get("prompt_tokens_per_question") or 0.0) for row in rows),
        "completion_tokens_mean": safe_mean(float(row.get("completion_tokens_per_question") or 0.0) for row in rows),
        "total_tokens_mean": total_tokens,
        "communication_tokens_mean": safe_mean(float(row.get("debate_total_tokens_per_question") or 0.0) for row in rows),
        "latency_ms_mean": safe_mean(float(row.get("latency_ms_per_question") or 0.0) for row in rows),
        "calls_per_question_mean": safe_mean(float(row.get("calls_per_question") or 0.0) for row in rows),
        "protocol_failures_per_question_mean": safe_mean(float(row.get("protocol_failures_per_question") or 0.0) for row in rows),
        "reason_missing_turns_per_question_mean": safe_mean(float(row.get("reason_missing_turns_per_question") or 0.0) for row in rows),
        "accuracy_per_1k_tokens": round(accuracy / total_tokens * 1000, 6) if total_tokens else 0.0,
        "debate_rounds": safe_mean(float(row.get("debate_rounds") or 0.0) for row in rows),
        "agent_count": safe_mean(float(row.get("agent_count") or 0.0) for row in rows),
        "trigger_rate": safe_ratio(triggered, count),
        "corrected_count": corrected,
        "harmed_count": harmed,
        "corrected_rate": safe_ratio(corrected, count),
        "harmed_rate": safe_ratio(harmed, count),
        "flip_rate": safe_mean(1.0 if row.get("vote_flipped") else 0.0 for row in rows),
        "initial_consensus_rate": safe_mean(1.0 if row.get("initial_consensus") else 0.0 for row in rows),
        "final_consensus_rate": safe_mean(1.0 if row.get("final_consensus") else 0.0 for row in rows),
    }


def _macro_rows(dataset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in dataset_rows:
        grouped[(str(row["model_name"]), str(row["method_name"]))].append(row)
    return [_macro_summary(rows, model, method) for (model, method), rows in grouped.items()]


def _macro_summary(rows: list[dict[str, Any]], model_name: str, method_name: str) -> dict[str, Any]:
    accuracy = safe_mean(float(row["accuracy_mean"]) for row in rows)
    initial = safe_mean(float(row["initial_vote_accuracy_mean"]) for row in rows)
    total_tokens = safe_mean(float(row["total_tokens_mean"]) for row in rows)
    return {
        "dataset": "overall",
        "aggregate_kind": "macro",
        "model_name": model_name,
        "method_name": method_name,
        "method_type": rows[0]["method_type"],
        "question_count": sum(int(row["question_count"]) for row in rows),
        "prediction_rows": sum(int(row["prediction_rows"]) for row in rows),
        "accuracy_mean": accuracy,
        "initial_vote_accuracy_mean": initial,
        "debate_gain_over_initial_vote": round(accuracy - initial, 6),
        "prompt_tokens_mean": safe_mean(float(row["prompt_tokens_mean"]) for row in rows),
        "completion_tokens_mean": safe_mean(float(row["completion_tokens_mean"]) for row in rows),
        "total_tokens_mean": total_tokens,
        "communication_tokens_mean": safe_mean(float(row["communication_tokens_mean"]) for row in rows),
        "latency_ms_mean": safe_mean(float(row["latency_ms_mean"]) for row in rows),
        "calls_per_question_mean": safe_mean(float(row["calls_per_question_mean"]) for row in rows),
        "protocol_failures_per_question_mean": safe_mean(float(row["protocol_failures_per_question_mean"]) for row in rows),
        "reason_missing_turns_per_question_mean": safe_mean(float(row["reason_missing_turns_per_question_mean"]) for row in rows),
        "accuracy_per_1k_tokens": round(accuracy / total_tokens * 1000, 6) if total_tokens else 0.0,
        "debate_rounds": safe_mean(float(row["debate_rounds"]) for row in rows),
        "agent_count": safe_mean(float(row["agent_count"]) for row in rows),
        "trigger_rate": safe_mean(float(row["trigger_rate"]) for row in rows),
        "corrected_count": sum(int(row["corrected_count"]) for row in rows),
        "harmed_count": sum(int(row["harmed_count"]) for row in rows),
        "corrected_rate": safe_mean(float(row["corrected_rate"]) for row in rows),
        "harmed_rate": safe_mean(float(row["harmed_rate"]) for row in rows),
        "flip_rate": safe_mean(float(row["flip_rate"]) for row in rows),
        "initial_consensus_rate": safe_mean(float(row["initial_consensus_rate"]) for row in rows),
        "final_consensus_rate": safe_mean(float(row["final_consensus_rate"]) for row in rows),
    }


def build_brd_diagnostics(prediction_rows: list[dict[str, Any]], *, dataset_order: list[str], method_order: list[str]) -> dict[str, Any]:
    brd_rows = [row for row in prediction_rows if row.get("method_type") == "brd"]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in brd_rows:
        grouped[(str(row["dataset"]), str(row["method_name"]))].append(row)
    summary = [_diagnostic_summary(dataset, method, rows) for (dataset, method), rows in grouped.items()]
    for method in {str(row["method_name"]) for row in brd_rows}:
        rows = [row for row in brd_rows if row["method_name"] == method]
        summary.append(_diagnostic_summary("overall", method, rows))
    dataset_rank = {name: index for index, name in enumerate(dataset_order)}
    method_rank = {name: index for index, name in enumerate(method_order)}
    summary.sort(key=lambda row: (dataset_rank.get(row["dataset"], len(dataset_rank) + (row["dataset"] == "overall")), method_rank.get(row["method_name"], 999)))
    return {
        "summary_rows": summary,
        "sample_rows": brd_rows,
        "theory_note": "IID 2/3 quorum probabilities are not empirical guarantees; inspect reviewer_error_correlation and effective_reviewer_count.",
    }


def apply_bbeh_harmonic_primary(
    metrics: dict[str, Any],
    prediction_rows: list[dict[str, Any]],
    *,
    control_names: list[str],
    use_harmonic: bool = True,
) -> dict[str, Any]:
    """Use task harmonic accuracy as BBEH's primary metric, retaining micro."""

    if not use_harmonic:
        for row in metrics.get("summary") or []:
            if row.get("dataset") == "bbeh":
                row["micro_accuracy_mean"] = float(row.get("accuracy_mean") or 0.0)
                row["bbeh_task_harmonic_accuracy"] = None
                row["primary_accuracy_metric"] = "micro_accuracy"
        metrics["bbeh_metric"] = {"primary": "micro_accuracy", "secondary": None}
        return metrics

    harmonics = _bbeh_harmonics_by_method(prediction_rows)
    summary = list(metrics.get("summary") or [])
    for row in summary:
        key = (str(row.get("model_name") or ""), str(row.get("method_name") or ""))
        if row.get("dataset") == "bbeh":
            micro = float(row.get("accuracy_mean") or 0.0)
            harmonic = harmonics.get(key, {}).get("final", 0.0)
            initial_harmonic = harmonics.get(key, {}).get("initial", 0.0)
            row["micro_accuracy_mean"] = micro
            row["bbeh_task_harmonic_accuracy"] = harmonic
            row["primary_accuracy_metric"] = "task_harmonic"
            row["accuracy_mean"] = harmonic
            row["initial_vote_accuracy_mean"] = initial_harmonic
            row["debate_gain_over_initial_vote"] = round(harmonic - initial_harmonic, 6)
            tokens = float(row.get("total_tokens_mean") or 0.0)
            row["accuracy_per_1k_tokens"] = round(harmonic / tokens * 1000, 6) if tokens else 0.0
        elif row.get("dataset") == "overall_micro":
            row["primary_accuracy_metric"] = "micro_accuracy"
        elif row.get("aggregate_kind") == "dataset":
            row["primary_accuracy_metric"] = "exact_accuracy"

    dataset_rows = [row for row in summary if row.get("aggregate_kind") == "dataset"]
    for row in [row for row in summary if row.get("dataset") == "overall" and row.get("aggregate_kind") == "macro"]:
        items = [item for item in dataset_rows if item.get("model_name") == row.get("model_name") and item.get("method_name") == row.get("method_name")]
        if not items:
            continue
        accuracy = safe_mean(float(item.get("accuracy_mean") or 0.0) for item in items)
        initial = safe_mean(float(item.get("initial_vote_accuracy_mean") or 0.0) for item in items)
        row["accuracy_mean"] = accuracy
        row["initial_vote_accuracy_mean"] = initial
        row["debate_gain_over_initial_vote"] = round(accuracy - initial, 6)
        tokens = float(row.get("total_tokens_mean") or 0.0)
        row["accuracy_per_1k_tokens"] = round(accuracy / tokens * 1000, 6) if tokens else 0.0
        row["primary_accuracy_metric"] = "macro_with_bbeh_task_harmonic"
    _refresh_accuracy_comparisons(summary, control_names)
    metrics["summary"] = summary
    metrics["bbeh_metric"] = {"primary": "task_harmonic_accuracy", "secondary": "micro_accuracy"}
    return metrics


def _bbeh_harmonics_by_method(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, float]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("dataset") == "bbeh":
            grouped[(str(row.get("model_name") or ""), str(row.get("method_name") or ""))].append(row)
    return {
        key: {"final": _task_harmonic(items), "initial": _task_harmonic(items, score_field="initial_vote_score")}
        for key, items in grouped.items()
    }


def _refresh_accuracy_comparisons(rows: list[dict[str, Any]], control_names: list[str]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("dataset") or ""), str(row.get("model_name") or ""))].append(row)
    for items in grouped.values():
        cot = next((item for item in items if item.get("method_name") == "cot_1"), None)
        controls = [item for item in items if item.get("method_name") in control_names]
        best = max(controls, key=lambda item: float(item.get("accuracy_mean") or 0.0)) if controls else None
        for item in items:
            accuracy = float(item.get("accuracy_mean") or 0.0)
            item["accuracy_delta_vs_cot_1"] = None if cot is None else round(accuracy - float(cot.get("accuracy_mean") or 0.0), 6)
            item["best_no_comm_method"] = None if best is None else best.get("method_name")
            item["best_no_comm_accuracy"] = None if best is None else best.get("accuracy_mean")
            item["accuracy_delta_vs_best_no_comm"] = None if best is None else round(accuracy - float(best.get("accuracy_mean") or 0.0), 6)


def _diagnostic_summary(dataset: str, method_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    triggered = [row for row in rows if row.get("triggered")]
    overrides = [row for row in rows if row.get("override_accepted")]
    corrections = [row for row in rows if row.get("corrected_by_debate")]
    harms = [row for row in rows if row.get("harmed_by_debate")]
    correct_overrides = [row for row in overrides if row.get("score") == 1.0]
    needed_overrides = [row for row in rows if row.get("initial_vote_score") != 1.0 and row.get("candidate_oracle_correct")]
    recovered = [row for row in needed_overrides if row.get("override_accepted") and row.get("score") == 1.0]
    task_harmonic = _task_harmonic(rows) if dataset == "bbeh" else None
    review_errors = [
        1.0 - float(item)
        for row in triggered
        for item in row.get("reviewer_correctness", [])
        if item is not None
    ]
    return {
        "dataset": dataset,
        "method_name": method_name,
        "question_count": len(rows),
        "trigger_count": len(triggered),
        "trigger_rate": safe_ratio(len(triggered), len(rows)),
        "candidate_oracle_accuracy": safe_mean(1.0 if row.get("candidate_oracle_correct") else 0.0 for row in rows),
        "candidate_oracle_gap_over_anchor": safe_mean(float(bool(row.get("candidate_oracle_correct"))) - float(row.get("initial_vote_score") or 0.0) for row in rows),
        "override_count": len(overrides),
        "override_precision": safe_ratio(len(correct_overrides), len(overrides)),
        "override_recall_on_oracle_opportunities": safe_ratio(len(recovered), len(needed_overrides)),
        "corrected_count": len(corrections),
        "harmed_count": len(harms),
        "net_corrected": len(corrections) - len(harms),
        "shadow_novel_answer_count": sum(len(row.get("shadow_answers") or []) for row in rows),
        "quorum_error_rate": safe_ratio(sum(1 for row in overrides if row.get("score") != 1.0), len(overrides)),
        "reviewer_error_rate": safe_mean(review_errors),
        "reviewer_error_correlation": reviewer_error_correlation(triggered),
        "bbeh_task_harmonic_accuracy": task_harmonic,
        "calls_mean": safe_mean(float(row.get("calls_per_question") or 0.0) for row in rows),
        "tokens_mean": safe_mean(float(row.get("total_tokens_per_question") or 0.0) for row in rows),
        "latency_ms_mean": safe_mean(float(row.get("latency_ms_per_question") or 0.0) for row in rows),
    }


def _task_harmonic(rows: list[dict[str, Any]], *, score_field: str = "score") -> float | None:
    by_task: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        task = str(row.get("task") or "")
        if task:
            value = row.get(score_field)
            if value is None and score_field != "score":
                value = row.get("score")
            by_task[task].append(float(value or 0.0))
    accuracies = [sum(values) / len(values) for values in by_task.values() if values]
    if not accuracies or any(value <= 0.0 for value in accuracies):
        return 0.0 if accuracies else None
    return len(accuracies) / sum(1.0 / value for value in accuracies)


def estimate_work(experiment: BrdMadExperimentConfig, phase_name: str, benchmarks, controls, protocol: BrdProtocolConfig, active_methods: list[str]) -> tuple[int, int]:
    calls = predictions = 0
    for benchmark in benchmarks:
        split_name = resolve_phase_split_name(experiment, phase_name, benchmark.slug)
        count = len(load_split_ids(benchmark.cache_namespace or benchmark.slug, split_name, random_seed=benchmark.random_seed))
        # Reviewer calls are an upper bound because unanimous Stage A exits.
        calls += count * (protocol.stage_a_candidates + protocol.reviewer_count * len(active_methods))
        predictions += count * len(active_methods)
        for control_name in experiment.control_methods:
            calls += count * controls[control_name].budget_calls
            predictions += count
    return calls, predictions


def load_selected_samples(benchmark, split_name: str) -> list[DatasetSample]:
    return select_samples(benchmark, split_name)


def resolve_split_name(experiment: BrdMadExperimentConfig, phase_name: str, benchmark_slug: str) -> str:
    return resolve_phase_split_name(experiment, phase_name, benchmark_slug)


def control_results_as_brd_shape(control_results):
    for sample_index, turn_rows, debate_rows, prediction_row in control_results:
        yield sample_index, turn_rows, debate_rows, [], [prediction_row]


__all__ = [
    "_execute_turn",
    "append_outputs",
    "apply_bbeh_harmonic_primary",
    "build_brd_diagnostics",
    "build_control_prediction_row",
    "build_metrics",
    "control_results_as_brd_shape",
    "estimate_work",
    "load_selected_samples",
    "resolve_split_name",
    "run_brd_batch",
]
