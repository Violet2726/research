"""DoG 的样本级执行、图消息约束与指标聚合。"""

from __future__ import annotations

from functools import partial
import json
import re
import string
from typing import Any

from research_experiments.core.data.datasets import DatasetSample, select_samples
from research_experiments.core.data.evaluation import aggregate_majority, normalize_prediction, score_prediction
from research_experiments.core.execution.artifacts import BufferedJsonlWriter
from research_experiments.core.execution.cache import RequestCache
from research_experiments.core.execution.providers import OpenAICompatibleProvider
from research_experiments.core.execution.rate_limits import SlidingWindowRateLimiter
from research_experiments.core.execution.runner_common import execute_cached_turn, run_indexed_batch
from research_experiments.core.structured_outputs import SCHEMA_ANSWER_CORE, validate_or_recover_structured_output
from research_experiments.families.dog_graph.config import DogGraphExperimentConfig, GraphMethodSpec, ProtocolConfig
from research_experiments.families.dog_graph.dataset_views import (
    GraphView,
    build_full_graph_view,
    build_full_graph_views,
    build_graph_views,
)
from research_experiments.families.dog_graph.prompts import build_debate_messages, build_initial_messages
from research_experiments.families.shared.common import resolve_phase_split_name, safe_mean


def _active_methods(experiment: DogGraphExperimentConfig) -> list[GraphMethodSpec]:
    """返回当前实验启用的方法列表。"""

    return list(experiment.methods)


def _resolve_split_name(experiment: DogGraphExperimentConfig, phase_name: str, benchmark_slug: str) -> str:
    """解析当前 benchmark 在该 phase 下对应的固定 split。"""

    return resolve_phase_split_name(experiment, phase_name, benchmark_slug)


def _load_selected_samples(benchmark, split_name: str) -> list[DatasetSample]:
    """按固定 split 选择本轮样本。"""

    return select_samples(benchmark, split_name)


def _planned_calls_per_sample(method: GraphMethodSpec) -> int:
    """估算单题在指定方法下的调用数。"""

    if method.mode == "single":
        return 1
    return method.agent_count * (1 + method.round_limit)


def _run_method_batch(
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    samples: list[DatasetSample],
    method: GraphMethodSpec,
    protocol: ProtocolConfig,
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    global_seed: int,
    prompt_version: str,
    max_concurrent_requests: int,
) -> list[tuple[int, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]]:
    """并发执行同一方法下的全部样本。"""

    worker = partial(
        _run_method_sample,
        run_id=run_id,
        benchmark_slug=benchmark_slug,
        split_name=split_name,
        method=method,
        protocol=protocol,
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        global_seed=global_seed,
        prompt_version=prompt_version,
    )
    return [
        (sample_index, *result)
        for sample_index, result in run_indexed_batch(
            samples,
            worker=worker,
            max_concurrent_requests=max_concurrent_requests,
        )
    ]


def _write_sample_outputs(
    *,
    sample_results: list[tuple[int, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]],
    dataset_slug: str,
    progress,
    turn_handle: BufferedJsonlWriter,
    debate_handle: BufferedJsonlWriter,
    graph_handle: BufferedJsonlWriter,
    prediction_handle: BufferedJsonlWriter,
    all_turns: list[dict[str, Any]],
    all_graph_trace_rows: list[dict[str, Any]],
    final_predictions: list[dict[str, Any]],
) -> None:
    """按稳定顺序落盘一批样本结果并更新进度。"""

    for _, turn_rows, debate_rows, graph_rows, prediction_row in sample_results:
        for row in turn_rows:
            turn_handle.write_row(row)
            progress.record_call(row, method_key="method_name")
        for row in debate_rows:
            debate_handle.write_row(row)
        for row in graph_rows:
            graph_handle.write_row(row)
        prediction_handle.write_row(prediction_row)
        progress.record_predictions(1, dataset_slug, prediction_row["method_name"])
        all_turns.extend(turn_rows)
        all_graph_trace_rows.extend(graph_rows)
        final_predictions.append(prediction_row)


def _run_method_sample(
    sample: DatasetSample,
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    method: GraphMethodSpec,
    protocol: ProtocolConfig,
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    global_seed: int,
    prompt_version: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """运行单个样本上的图求解/图投票/图辩论协议。"""

    turn_rows: list[dict[str, Any]] = []
    debate_rows: list[dict[str, Any]] = []
    views = _method_views(sample, method)

    initial_turns: list[dict[str, Any]] = []
    for view in views:
        initial_turns.append(
            _execute_turn(
                run_id=run_id,
                dataset=benchmark_slug,
                split_name=split_name,
                sample=sample,
                method_name=method.name,
                method_mode=method.mode,
                round_index=0,
                agent_id=view.agent_id,
                role="initial",
                visible_peer_count=0,
                graph_view=view,
                messages=build_initial_messages(sample, view, prompt_version=prompt_version),
                backbone=backbone,
                provider=provider,
                cache=cache,
                limiter=limiter,
                temperature=protocol.initial_temperature,
                top_p=protocol.top_p,
                max_output_tokens=protocol.max_output_tokens,
                seed=global_seed + view.agent_id,
            )
        )
    turn_rows.extend(initial_turns)

    previous_round = initial_turns
    executed_round_count = 0
    if method.mode == "debate":
        for round_index in range(1, method.round_limit + 1):
            current_round: list[dict[str, Any]] = []
            for view in views:
                recipient_previous = previous_round[view.agent_id - 1]
                peer_messages: list[dict[str, object]] = []
                for sender in previous_round:
                    if sender["agent_id"] == view.agent_id:
                        continue
                    peer_messages.append(
                        {
                            "agent": f"agent_{sender['agent_id']}",
                            "answer": str(sender["validated_output"].get("final_answer", "")).strip(),
                            "reasoning": str(sender["validated_output"].get("reasoning", "")).strip(),
                            "evidence_triples": list(sender["validated_output"].get("evidence_triples") or []),
                            "answer_path": list(sender["validated_output"].get("answer_path") or []),
                        }
                    )
                    debate_rows.append(
                        {
                            "run_id": run_id,
                            "dataset": benchmark_slug,
                            "split": split_name,
                            "sample_id": sample.sample_id,
                            "method_name": method.name,
                            "method_mode": method.mode,
                            "round_index": round_index,
                            "sender_agent_id": sender["agent_id"],
                            "recipient_agent_id": view.agent_id,
                            "sender_answer": str(sender["validated_output"].get("final_answer", "")).strip(),
                            "sender_reasoning": str(sender["validated_output"].get("reasoning", "")).strip(),
                            "sender_evidence_triples": list(sender["validated_output"].get("evidence_triples") or []),
                            "sender_answer_path": list(sender["validated_output"].get("answer_path") or []),
                            "sender_graph_view_kind": sender["graph_view_kind"],
                            "communication_grounded": bool(sender["validated_output"].get("evidence_triples")),
                        }
                    )
                current_round.append(
                    _execute_turn(
                        run_id=run_id,
                        dataset=benchmark_slug,
                        split_name=split_name,
                        sample=sample,
                        method_name=method.name,
                        method_mode=method.mode,
                        round_index=round_index,
                        agent_id=view.agent_id,
                        role="debate",
                        visible_peer_count=len(peer_messages),
                        graph_view=view,
                        messages=build_debate_messages(
                            sample=sample,
                            graph_view=view,
                            round_index=round_index,
                            previous_answer=str(recipient_previous["validated_output"].get("final_answer", "")).strip(),
                            previous_reasoning=str(recipient_previous["validated_output"].get("reasoning", "")).strip(),
                            previous_evidence_triples=list(recipient_previous["validated_output"].get("evidence_triples") or []),
                            previous_answer_path=list(recipient_previous["validated_output"].get("answer_path") or []),
                            peer_messages=peer_messages,
                            prompt_version=prompt_version,
                        ),
                        backbone=backbone,
                        provider=provider,
                        cache=cache,
                        limiter=limiter,
                        temperature=protocol.debate_temperature,
                        top_p=protocol.top_p,
                        max_output_tokens=protocol.max_output_tokens,
                        seed=global_seed + view.agent_id + round_index * 100,
                    )
                )
            turn_rows.extend(current_round)
            previous_round = current_round
            executed_round_count = round_index

    final_turns = previous_round if method.mode == "debate" else initial_turns
    initial_vote, initial_vote_counts = aggregate_majority([row["normalized_answer"] for row in initial_turns])
    final_vote, final_vote_counts = aggregate_majority([row["normalized_answer"] for row in final_turns])
    final_vote = final_vote or initial_vote
    final_vote_counts = final_vote_counts or initial_vote_counts
    initial_vote_score = score_prediction(benchmark_slug, initial_vote, sample.reference_answer) if initial_vote else 0.0
    final_vote_score = score_prediction(benchmark_slug, final_vote, sample.reference_answer) if final_vote else 0.0
    winner_turn = next(
        (row for row in final_turns if row["normalized_answer"] == final_vote),
        final_turns[0],
    )

    all_node_ids = {node_id for view in views for node_id in view.node_ids}
    all_edge_keys = {edge_key for view in views for edge_key in view.edge_keys}
    debate_turns = [row for row in turn_rows if row["role"] == "debate"]
    prompt_tokens = sum(float(row["prompt_tokens"]) for row in turn_rows)
    completion_tokens = sum(float(row["completion_tokens"]) for row in turn_rows)
    total_tokens = sum(float(row["total_tokens"]) for row in turn_rows)
    latency_ms = sum(float(row["latency_ms"]) for row in turn_rows)
    debate_total_tokens = sum(float(row["total_tokens"]) for row in debate_turns)
    communication_grounded = bool(debate_rows) and all(bool(row["sender_evidence_triples"]) for row in debate_rows)

    prediction_row = {
        "run_id": run_id,
        "dataset": benchmark_slug,
        "split": split_name,
        "sample_id": sample.sample_id,
        "method_name": method.name,
        "method_type": "graph",
        "method_mode": method.mode,
        "configured_round_limit": method.round_limit,
        "model_name": backbone.name,
        "prediction": final_vote,
        "gold": sample.reference_answer,
        "score": final_vote_score,
        "initial_vote_prediction": initial_vote,
        "initial_vote_score": initial_vote_score,
        "initial_vote_counts": initial_vote_counts,
        "final_vote_prediction": final_vote,
        "final_vote_score": final_vote_score,
        "final_vote_counts": final_vote_counts,
        "prompt_tokens_per_question": prompt_tokens,
        "completion_tokens_per_question": completion_tokens,
        "total_tokens_per_question": total_tokens,
        "latency_ms_per_question": latency_ms,
        "debate_total_tokens_per_question": debate_total_tokens,
        "calls_per_question": len(turn_rows),
        "debate_rounds": executed_round_count,
        "agent_count": method.agent_count,
        "subgraph_node_count": len(all_node_ids),
        "subgraph_edge_count": len(all_edge_keys),
        "evidence_triples": list(winner_turn["validated_output"].get("evidence_triples") or []),
        "answer_path": list(winner_turn["validated_output"].get("answer_path") or []),
        "communication_grounded": communication_grounded,
        "graph_view_kind": winner_turn["graph_view_kind"],
        "matched_vote_control": method.matched_controls[0] if method.matched_controls else None,
        "corrected_by_debate": initial_vote_score < 1.0 and final_vote_score == 1.0,
        "harmed_by_debate": initial_vote_score == 1.0 and final_vote_score < 1.0,
        "vote_counts": final_vote_counts,
    }
    graph_trace_rows = [_build_graph_trace_row(row) for row in turn_rows]
    return turn_rows, debate_rows, graph_trace_rows, prediction_row


def _method_views(sample: DatasetSample, method: GraphMethodSpec) -> list[GraphView]:
    if method.view_mode == "full_subgraph":
        return [build_full_graph_view(sample)]
    if method.mode in {"vote", "debate"}:
        return build_full_graph_views(sample, method.agent_count)
    views = build_graph_views(sample)
    return views[: method.agent_count]


def _execute_turn(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    method_name: str,
    method_mode: str,
    round_index: int,
    agent_id: int,
    role: str,
    visible_peer_count: int,
    graph_view: GraphView,
    messages: list[dict[str, str]],
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    seed: int,
) -> dict[str, Any]:
    """执行单次图推理 turn，并返回统一日志结构。"""

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
        validator=partial(_validate_graph_answer_payload, dataset=dataset),
    )
    grounded_output = _ground_graph_payload(result.validated_output, graph_view, method_mode=method_mode)
    final_answer = str(grounded_output.get("final_answer") or "")
    normalized = normalize_prediction(dataset, final_answer) if final_answer else ""
    turn_score = score_prediction(dataset, normalized, sample.reference_answer) if normalized else None
    return {
        "run_id": run_id,
        "dataset": dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "method_name": method_name,
        "method_type": "graph",
        "method_mode": method_mode,
        "round_index": round_index,
        "agent_id": agent_id,
        "role": role,
        "prompt_hash": result.prompt_hash,
        "prediction": normalized,
        "score": turn_score,
        "output_status": result.output_status,
        "prompt_tokens": float(result.usage.get("prompt_tokens") or 0.0),
        "completion_tokens": float(result.usage.get("completion_tokens") or 0.0),
        "total_tokens": float(result.usage.get("total_tokens") or 0.0),
        "latency_ms": float(result.response_payload.get("latency_ms") or 0.0),
        "cache_hit": result.cache_hit,
        "request_error": result.request_error,
        "visible_peer_count": visible_peer_count,
        "payload": result.payload,
        "assistant_text": result.response_payload.get("assistant_text", ""),
        "provider_reasoning_text": result.response_payload.get("provider_reasoning_text", ""),
        "validated_output": grounded_output,
        "normalized_answer": normalized,
        "graph_view_kind": graph_view.view_kind,
        "subgraph_node_count": graph_view.node_count,
        "subgraph_edge_count": graph_view.edge_count,
        "available_triples": list(graph_view.visible_triples),
    }


def _validate_graph_answer_payload(
    assistant_text: str,
    provider_reasoning_text: str,
    *,
    dataset: str,
) -> dict[str, Any]:
    """解析并校验图推理 agent 的 JSON 输出。"""

    for raw_text in [str(assistant_text or "").strip(), str(provider_reasoning_text or "").strip()]:
        if not raw_text:
            continue
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        final_answer = str(payload.get("final_answer") or "").strip()
        if not final_answer:
            continue
        reasoning = str(payload.get("reasoning") or final_answer).strip() or final_answer
        evidence_triples = _string_list(payload.get("evidence_triples"), limit=6)
        answer_path = _string_list(payload.get("answer_path"), limit=6)
        return {
            "final_answer": final_answer,
            "reasoning": reasoning,
            "evidence_triples": evidence_triples,
            "answer_path": answer_path,
        }

    recovered = validate_or_recover_structured_output(
        assistant_text,
        SCHEMA_ANSWER_CORE,
        dataset=dataset,
        provider_reasoning_text=provider_reasoning_text,
    )
    return {
        "final_answer": str(recovered.get("final_answer") or "").strip(),
        "reasoning": str(recovered.get("reasoning") or recovered.get("final_answer") or "").strip(),
        "evidence_triples": [],
        "answer_path": [],
    }


def _ground_graph_payload(payload: dict[str, Any], graph_view: GraphView, *, method_mode: str) -> dict[str, Any]:
    """为缺少图证据的回答补最小可追踪证据，保证 grounded-contract 可落盘。"""

    grounded = dict(payload)
    final_answer = str(grounded.get("final_answer") or "").strip()
    evidence_triples = _string_list(grounded.get("evidence_triples"), limit=6)
    answer_path = _string_list(grounded.get("answer_path"), limit=6)
    if method_mode != "single" and final_answer:
        final_answer = _canonicalize_graph_answer(final_answer, graph_view)
    if not evidence_triples and graph_view.visible_triples:
        evidence_triples = [graph_view.visible_triples[0]]
    if not answer_path and evidence_triples:
        answer_path = [evidence_triples[0]]
    grounded["final_answer"] = final_answer
    grounded["evidence_triples"] = evidence_triples
    grounded["answer_path"] = answer_path
    return grounded


def _canonicalize_graph_answer(answer: str, graph_view: GraphView) -> str:
    """把明显过泛的图答案收敛到更具体的节点标题或规范后缀。"""

    normalized_answer = _normalize_graph_surface(answer)
    candidate_labels = [
        label
        for label in graph_view.node_labels
        if label and _normalize_graph_surface(label) not in {"", normalized_answer}
    ]
    subset_matches = [
        label
        for label in candidate_labels
        if _is_subset_surface(normalized_answer, _normalize_graph_surface(label))
    ]
    if subset_matches:
        return min(subset_matches, key=lambda item: (len(_normalize_graph_surface(item).split()), len(item)))

    source_match_targets = [
        target
        for source, _, target in graph_view.structured_triples
        if _normalize_graph_surface(source) == normalized_answer
        and _normalize_graph_surface(target) not in {"", normalized_answer, "nodeanswer"}
    ]
    if source_match_targets:
        return max(source_match_targets, key=lambda item: (len(_normalize_graph_surface(item).split()), len(item)))

    visible_text = " ".join(graph_view.visible_triples).lower()
    lowered = answer.strip().lower()
    if "language" not in lowered and any(marker in visible_text for marker in ["language", "languages spoken", "official language"]):
        return f"{answer.strip()} language"
    if "party" not in lowered and "party" in visible_text:
        return f"{answer.strip()} party"
    return answer


def _normalize_graph_surface(value: str) -> str:
    lowered = value.lower()
    lowered = re.sub(r"\b(a|an|the)\b", " ", lowered)
    lowered = lowered.translate(str.maketrans("", "", string.punctuation))
    return " ".join(lowered.split())


def _is_subset_surface(shorter: str, longer: str) -> bool:
    short_tokens = shorter.split()
    long_tokens = longer.split()
    if not short_tokens or len(short_tokens) >= len(long_tokens):
        return False
    return all(token in long_tokens for token in short_tokens)


def _string_list(value: object, *, limit: int) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        materialized = [value]
    elif isinstance(value, list):
        materialized = value
    else:
        return []
    normalized = [str(item).strip() for item in materialized if str(item).strip()]
    return normalized[:limit]


def _build_graph_trace_row(turn_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": turn_row["run_id"],
        "dataset": turn_row["dataset"],
        "split": turn_row["split"],
        "sample_id": turn_row["sample_id"],
        "method_name": turn_row["method_name"],
        "method_mode": turn_row["method_mode"],
        "round_index": turn_row["round_index"],
        "agent_id": turn_row["agent_id"],
        "role": turn_row["role"],
        "graph_view_kind": turn_row["graph_view_kind"],
        "subgraph_node_count": turn_row["subgraph_node_count"],
        "subgraph_edge_count": turn_row["subgraph_edge_count"],
        "available_triple_count": len(turn_row.get("available_triples") or []),
        "evidence_triples": list(turn_row["validated_output"].get("evidence_triples") or []),
        "answer_path": list(turn_row["validated_output"].get("answer_path") or []),
        "communication_grounded": bool(turn_row["validated_output"].get("evidence_triples")),
        "normalized_answer": turn_row["normalized_answer"],
        "output_status": turn_row["output_status"],
    }


def _build_metrics(
    prediction_rows: list[dict[str, Any]],
    methods: list[GraphMethodSpec],
) -> dict[str, Any]:
    """把题级预测聚合成 DoG 的 summary 指标。"""

    method_map = {method.name: method for method in methods}
    summary_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in prediction_rows:
        grouped.setdefault((row["dataset"], row["model_name"], row["method_name"]), []).append(row)
    for (dataset, model_name, method_name), rows in sorted(grouped.items()):
        summary_rows.append(_aggregate_summary_row(dataset, model_name, method_name, rows, method_map))

    overall_grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in prediction_rows:
        overall_grouped.setdefault((row["model_name"], row["method_name"]), []).append(row)
    for (model_name, method_name), rows in sorted(overall_grouped.items()):
        summary_rows.append(_aggregate_summary_row("overall", model_name, method_name, rows, method_map))

    by_lookup = {(row["dataset"], row["model_name"], row["method_name"]): row for row in summary_rows}
    for row in summary_rows:
        control_name = row.get("matched_vote_control")
        control_row = by_lookup.get((row["dataset"], row["model_name"], control_name)) if control_name else None
        row["debate_gain_over_vote"] = round(row["accuracy_mean"] - control_row["accuracy_mean"], 6) if control_row else None
        row["token_ratio_vs_vote"] = (
            round(row["total_tokens_mean"] / control_row["total_tokens_mean"], 6)
            if control_row and control_row["total_tokens_mean"]
            else None
        )
    return {"summary": summary_rows}


def _aggregate_summary_row(
    dataset: str,
    model_name: str,
    method_name: str,
    rows: list[dict[str, Any]],
    method_map: dict[str, GraphMethodSpec],
) -> dict[str, Any]:
    accuracy = safe_mean(float(row["score"]) for row in rows)
    total_tokens_mean = safe_mean(float(row["total_tokens_per_question"]) for row in rows)
    debate_total_tokens_mean = safe_mean(float(row["debate_total_tokens_per_question"]) for row in rows)
    method = method_map[method_name]
    return {
        "dataset": dataset,
        "model_name": model_name,
        "method_name": method_name,
        "method_type": "graph",
        "method_mode": method.mode,
        "configured_round_limit": method.round_limit,
        "prediction_rows": len(rows),
        "question_count": len(rows),
        "accuracy_mean": accuracy,
        "prompt_tokens_mean": safe_mean(float(row["prompt_tokens_per_question"]) for row in rows),
        "completion_tokens_mean": safe_mean(float(row["completion_tokens_per_question"]) for row in rows),
        "total_tokens_mean": total_tokens_mean,
        "communication_tokens_mean": debate_total_tokens_mean,
        "calls_per_question_mean": safe_mean(float(row["calls_per_question"]) for row in rows),
        "latency_ms_mean": safe_mean(float(row["latency_ms_per_question"]) for row in rows),
        "accuracy_per_1k_tokens": round(accuracy / total_tokens_mean * 1000, 6) if total_tokens_mean else 0.0,
        "debate_rounds_mean": safe_mean(float(row["debate_rounds"]) for row in rows),
        "subgraph_node_count_mean": safe_mean(float(row["subgraph_node_count"]) for row in rows),
        "subgraph_edge_count_mean": safe_mean(float(row["subgraph_edge_count"]) for row in rows),
        "evidence_triple_count_mean": safe_mean(float(len(row.get("evidence_triples") or [])) for row in rows),
        "answer_path_length_mean": safe_mean(float(len(row.get("answer_path") or [])) for row in rows),
        "grounded_communication_rate": safe_mean(1.0 if row.get("communication_grounded") else 0.0 for row in rows),
        "matched_vote_control": method.matched_controls[0] if method.matched_controls else None,
    }


def _build_graph_diagnostics(
    prediction_rows: list[dict[str, Any]],
    graph_trace_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """汇总图证据使用、视角分布与 grounded communication 行为。"""

    summary_rows: list[dict[str, Any]] = []
    grouped_predictions: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in prediction_rows:
        grouped_predictions.setdefault((row["dataset"], row["method_name"]), []).append(row)
    for (dataset, method_name), rows in sorted(grouped_predictions.items()):
        summary_rows.append(
            {
                "dataset": dataset,
                "method_name": method_name,
                "prediction_rows": len(rows),
                "accuracy_mean": safe_mean(float(row["score"]) for row in rows),
                "grounded_communication_rate": safe_mean(1.0 if row.get("communication_grounded") else 0.0 for row in rows),
                "subgraph_node_count_mean": safe_mean(float(row["subgraph_node_count"]) for row in rows),
                "subgraph_edge_count_mean": safe_mean(float(row["subgraph_edge_count"]) for row in rows),
                "evidence_triple_count_mean": safe_mean(float(len(row.get("evidence_triples") or [])) for row in rows),
                "answer_path_length_mean": safe_mean(float(len(row.get("answer_path") or [])) for row in rows),
            }
        )
    overall_prediction_groups: dict[str, list[dict[str, Any]]] = {}
    for row in prediction_rows:
        overall_prediction_groups.setdefault(str(row["method_name"]), []).append(row)
    for method_name, rows in sorted(overall_prediction_groups.items()):
        summary_rows.append(
            {
                "dataset": "overall",
                "method_name": method_name,
                "prediction_rows": len(rows),
                "accuracy_mean": safe_mean(float(row["score"]) for row in rows),
                "grounded_communication_rate": safe_mean(1.0 if row.get("communication_grounded") else 0.0 for row in rows),
                "subgraph_node_count_mean": safe_mean(float(row["subgraph_node_count"]) for row in rows),
                "subgraph_edge_count_mean": safe_mean(float(row["subgraph_edge_count"]) for row in rows),
                "evidence_triple_count_mean": safe_mean(float(len(row.get("evidence_triples") or [])) for row in rows),
                "answer_path_length_mean": safe_mean(float(len(row.get("answer_path") or [])) for row in rows),
            }
        )

    view_rows: list[dict[str, Any]] = []
    grouped_views: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in graph_trace_rows:
        grouped_views.setdefault((row["dataset"], row["method_name"], row["graph_view_kind"]), []).append(row)
    for (dataset, method_name, view_kind), rows in sorted(grouped_views.items()):
        view_rows.append(
            {
                "dataset": dataset,
                "method_name": method_name,
                "graph_view_kind": view_kind,
                "turn_count": len(rows),
                "subgraph_node_count_mean": safe_mean(float(row["subgraph_node_count"]) for row in rows),
                "subgraph_edge_count_mean": safe_mean(float(row["subgraph_edge_count"]) for row in rows),
                "available_triple_count_mean": safe_mean(float(row["available_triple_count"]) for row in rows),
                "evidence_triple_count_mean": safe_mean(float(len(row.get("evidence_triples") or [])) for row in rows),
                "grounded_turn_rate": safe_mean(1.0 if row.get("communication_grounded") else 0.0 for row in rows),
            }
        )
    overall_view_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in graph_trace_rows:
        overall_view_groups.setdefault((str(row["method_name"]), str(row["graph_view_kind"])), []).append(row)
    for (method_name, view_kind), rows in sorted(overall_view_groups.items()):
        view_rows.append(
            {
                "dataset": "overall",
                "method_name": method_name,
                "graph_view_kind": view_kind,
                "turn_count": len(rows),
                "subgraph_node_count_mean": safe_mean(float(row["subgraph_node_count"]) for row in rows),
                "subgraph_edge_count_mean": safe_mean(float(row["subgraph_edge_count"]) for row in rows),
                "available_triple_count_mean": safe_mean(float(row["available_triple_count"]) for row in rows),
                "evidence_triple_count_mean": safe_mean(float(len(row.get("evidence_triples") or [])) for row in rows),
                "grounded_turn_rate": safe_mean(1.0 if row.get("communication_grounded") else 0.0 for row in rows),
            }
        )
    return {"summary_rows": summary_rows, "view_rows": view_rows}


def _build_cost_breakdown(turn_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总初始求解与图辩论的 token 成本。"""

    grouped: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in turn_rows:
        key = (row["dataset"], row["method_name"], row["method_mode"])
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
            },
        )
        total_tokens = float(row["total_tokens"])
        bucket["prompt_tokens"] += float(row["prompt_tokens"])
        bucket["completion_tokens"] += float(row["completion_tokens"])
        bucket["total_tokens"] += total_tokens
        bucket["latency_ms"] += float(row["latency_ms"])
        bucket["turn_count"] += 1
        if row["role"] == "debate":
            bucket["debate_tokens"] += total_tokens
        else:
            bucket["initial_tokens"] += total_tokens
    rows = []
    for (dataset, method_name, method_mode), bucket in sorted(grouped.items()):
        rows.append({"dataset": dataset, "method_name": method_name, "method_mode": method_mode} | bucket)
    return {"rows": rows}
