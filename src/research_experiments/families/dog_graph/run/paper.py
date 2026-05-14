"""DoG 原论文高保真主线的执行、追踪与指标聚合。"""

from __future__ import annotations

from functools import partial
from pathlib import Path
import json
from typing import Any

from research_experiments.core.data.datasets import DatasetSample, select_samples
from research_experiments.core.data.evaluation import normalize_prediction, score_prediction
from research_experiments.core.execution.artifacts import BufferedJsonlWriter
from research_experiments.core.execution.cache import RequestCache
from research_experiments.core.execution.providers import OpenAICompatibleProvider
from research_experiments.core.execution.rate_limits import SlidingWindowRateLimiter
from research_experiments.core.execution.runner_common import execute_cached_turn, run_indexed_batch
from research_experiments.families.dog_graph.config import DogGraphExperimentConfig, GraphMethodSpec, PaperProtocolConfig
from research_experiments.families.dog_graph.paper_backend import EntityRef
from research_experiments.families.dog_graph.paper_prompts import (
    build_direct_answer_messages,
    build_enough_answer_messages,
    build_relation_selection_messages,
    build_simplifier_messages,
    parse_enough_answer_output,
    parse_selected_relations,
    parse_simplified_question,
    validate_plain_text_output,
)
from research_experiments.families.shared.common import resolve_phase_split_name, safe_mean


PaperBackend = Any


def _active_methods(experiment: DogGraphExperimentConfig) -> list[GraphMethodSpec]:
    return list(experiment.methods)


def _resolve_split_name(experiment: DogGraphExperimentConfig, phase_name: str, benchmark_slug: str) -> str:
    return resolve_phase_split_name(experiment, phase_name, benchmark_slug)


def _load_selected_samples(benchmark, split_name: str) -> list[DatasetSample]:
    return select_samples(benchmark, split_name)


def _planned_calls_per_sample(method: GraphMethodSpec, sample: DatasetSample, protocol: PaperProtocolConfig) -> int:
    task_family = str(sample.metadata.get("dog_task_family") or "freebase")
    if task_family == "metaqa":
        hop_limit = int(sample.metadata.get("hop_count") or protocol.max_hops)
        simplifier_calls = max(0, hop_limit - 1) * 3 if method.mode == "paper_dog" else 0
        return hop_limit + simplifier_calls
    simplifier_calls = max(0, protocol.max_hops - 1) * 3 if method.mode == "paper_dog" else 0
    fallback_calls = 1 if protocol.direct_fallback_enabled else 0
    return protocol.max_hops * 2 + simplifier_calls + fallback_calls


def _run_method_batch(
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    samples: list[DatasetSample],
    method: GraphMethodSpec,
    protocol: PaperProtocolConfig,
    backend: PaperBackend,
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    global_seed: int,
    max_concurrent_requests: int,
) -> list[tuple[int, dict[str, Any]]]:
    worker = partial(
        _run_method_sample,
        run_id=run_id,
        benchmark_slug=benchmark_slug,
        split_name=split_name,
        method=method,
        protocol=protocol,
        backend=backend,
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        global_seed=global_seed,
    )
    return run_indexed_batch(
        samples,
        worker=worker,
        max_concurrent_requests=max_concurrent_requests,
    )


def _write_method_outputs(
    *,
    batch_results: list[tuple[int, dict[str, Any]]],
    dataset_slug: str,
    progress,
    turn_handle: BufferedJsonlWriter,
    debate_handle: BufferedJsonlWriter,
    graph_handle: BufferedJsonlWriter,
    prediction_handle: BufferedJsonlWriter,
    retrieval_handle: BufferedJsonlWriter,
    relation_handle: BufferedJsonlWriter,
    simplification_handle: BufferedJsonlWriter,
    answer_attempt_handle: BufferedJsonlWriter,
    all_turn_rows: list[dict[str, Any]],
    all_graph_rows: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
    relation_rows: list[dict[str, Any]],
    simplification_rows: list[dict[str, Any]],
    answer_attempt_rows: list[dict[str, Any]],
    final_predictions: list[dict[str, Any]],
) -> None:
    for _, payload in batch_results:
        for row in payload["turn_rows"]:
            turn_handle.write_row(row)
            progress.record_call(row, method_key="method_name")
        for row in payload["debate_rows"]:
            debate_handle.write_row(row)
        for row in payload["graph_rows"]:
            graph_handle.write_row(row)
        for row in payload["retrieval_rows"]:
            retrieval_handle.write_row(row)
        for row in payload["relation_rows"]:
            relation_handle.write_row(row)
        for row in payload["simplification_rows"]:
            simplification_handle.write_row(row)
        for row in payload["answer_attempt_rows"]:
            answer_attempt_handle.write_row(row)
        prediction_handle.write_row(payload["prediction_row"])
        progress.record_predictions(1, dataset_slug, payload["prediction_row"]["method_name"])

        all_turn_rows.extend(payload["turn_rows"])
        all_graph_rows.extend(payload["graph_rows"])
        retrieval_rows.extend(payload["retrieval_rows"])
        relation_rows.extend(payload["relation_rows"])
        simplification_rows.extend(payload["simplification_rows"])
        answer_attempt_rows.extend(payload["answer_attempt_rows"])
        final_predictions.append(payload["prediction_row"])


def _run_method_sample(
    sample: DatasetSample,
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    method: GraphMethodSpec,
    protocol: PaperProtocolConfig,
    backend: PaperBackend,
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    global_seed: int,
) -> dict[str, Any]:
    turn_rows: list[dict[str, Any]] = []
    debate_rows: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    relation_rows: list[dict[str, Any]] = []
    simplification_rows: list[dict[str, Any]] = []
    answer_attempt_rows: list[dict[str, Any]] = []

    question = sample.question
    original_question = sample.question
    topic_entity_id = str(sample.metadata.get("topic_entity_id") or sample.metadata.get("topic_entity_name") or "").strip()
    topic_entity_name = str(sample.metadata.get("topic_entity_name") or topic_entity_id).strip()
    head_entities = [EntityRef(topic_entity_id or topic_entity_name, topic_entity_name or topic_entity_id)] if (topic_entity_id or topic_entity_name) else []
    hop_limit = int(sample.metadata.get("hop_count") or protocol.max_hops)
    sample_backend = backend.for_sample(sample) if hasattr(backend, "for_sample") else backend
    retrieval_backend_name = str(getattr(sample_backend, "backend_name", getattr(backend, "backend_name", "paper_backend")))

    final_prediction = ""
    final_score = 0.0
    selected_relations: list[str] = []
    reasoning_triples: list[str] = []
    last_tail_entities: list[EntityRef] = []
    enough_answer_decision = "not_triggered"
    used_direct_fallback = False
    last_simplified_question = ""
    simplification_success = False
    final_hop_index = 0

    for hop_index in range(1, hop_limit + 1):
        final_hop_index = hop_index
        candidate_relations = sample_backend.list_relations(head_entities)
        relation_mapping = {f"relation_{index}": relation for index, relation in enumerate(candidate_relations, start=1)}
        selector_result = _execute_plain_turn(
            run_id=run_id,
            dataset=benchmark_slug,
            split_name=split_name,
            sample=sample,
            method_name=method.name,
            role="relation_selector",
            hop_index=hop_index,
            messages=build_relation_selection_messages(question, relation_mapping),
            backbone=backbone,
            provider=provider,
            cache=cache,
            limiter=limiter,
            temperature=protocol.selector_temperature,
            top_p=protocol.top_p,
            max_output_tokens=protocol.max_output_tokens,
            seed=global_seed + hop_index,
            extra_fields={"question_snapshot": question, "head_entity_count": len(head_entities)},
        )
        turn_rows.append(selector_result)
        selected_relations = parse_selected_relations(
            selector_result["validated_output"]["text"],
            relation_mapping,
            limit=protocol.max_selected_relations,
        )
        tail_entities: list[EntityRef] = []
        reasoning_triples = []
        for relation in selected_relations:
            expanded = sample_backend.expand_relation(head_entities, relation)
            tail_entities.extend(expanded)
            reasoning_triples.extend(sample_backend.build_reasoning_triples(head_entities, relation))
        tail_entities = _unique_entities(tail_entities)
        last_tail_entities = tail_entities

        retrieval_rows.append(
            {
                "run_id": run_id,
                "dataset": benchmark_slug,
                "split": split_name,
                "sample_id": sample.sample_id,
                "method_name": method.name,
                "hop_index": hop_index,
                "question_snapshot": question,
                "topic_entity_id": topic_entity_id,
                "head_entities": [entity.label for entity in head_entities],
                "candidate_relation_count": len(candidate_relations),
                "selected_relations": list(selected_relations),
                "tail_entities": [entity.label for entity in tail_entities],
                "reasoning_triples": list(reasoning_triples),
                "retrieval_backend": retrieval_backend_name,
            }
        )
        relation_rows.append(
            {
                "run_id": run_id,
                "dataset": benchmark_slug,
                "split": split_name,
                "sample_id": sample.sample_id,
                "method_name": method.name,
                "hop_index": hop_index,
                "candidate_relations": candidate_relations,
                "selected_relations": list(selected_relations),
                "selector_raw_text": selector_result["validated_output"]["text"],
                "empty_tail_expansion": not bool(tail_entities),
            }
        )
        graph_rows.append(
            {
                "run_id": run_id,
                "dataset": benchmark_slug,
                "split": split_name,
                "sample_id": sample.sample_id,
                "method_name": method.name,
                "method_mode": method.mode,
                "round_index": hop_index,
                "agent_id": 0,
                "role": "retrieval_hop",
                "graph_view_kind": "dynamic_retrieval",
                "subgraph_node_count": len(head_entities) + len(tail_entities),
                "subgraph_edge_count": len(reasoning_triples),
                "available_triple_count": len(reasoning_triples),
                "evidence_triples": list(reasoning_triples),
                "answer_path": list(reasoning_triples[:1]),
                "communication_grounded": bool(reasoning_triples),
                "normalized_answer": "",
                "output_status": "ok",
            }
        )

        if str(sample.metadata.get("dog_task_family") or "freebase") == "metaqa":
            final_prediction = " | ".join(entity.label for entity in tail_entities)
            if hop_index < hop_limit and method.mode == "paper_dog":
                simplification_payload = _run_simplification_team(
                    run_id=run_id,
                    dataset=benchmark_slug,
                    split_name=split_name,
                    sample=sample,
                    method_name=method.name,
                    hop_index=hop_index,
                    question=question,
                    reasoning_triples=reasoning_triples,
                    backbone=backbone,
                    provider=provider,
                    cache=cache,
                    limiter=limiter,
                    temperature=protocol.simplifier_temperature,
                    top_p=protocol.top_p,
                    max_output_tokens=protocol.max_output_tokens,
                    seed_base=global_seed + hop_index * 100,
                )
                turn_rows.extend(simplification_payload["turn_rows"])
                debate_rows.extend(simplification_payload["debate_rows"])
                simplification_rows.append(simplification_payload["trace_row"])
                if simplification_payload["trace_row"]["simplified_question"]:
                    question = simplification_payload["trace_row"]["simplified_question"]
                    last_simplified_question = question
                    simplification_success = True
            head_entities = tail_entities or head_entities
            continue

        enough_result = _execute_plain_turn(
            run_id=run_id,
            dataset=benchmark_slug,
            split_name=split_name,
            sample=sample,
            method_name=method.name,
            role="enough_answer",
            hop_index=hop_index,
            messages=build_enough_answer_messages(question, reasoning_triples),
            backbone=backbone,
            provider=provider,
            cache=cache,
            limiter=limiter,
            temperature=protocol.enough_answer_temperature,
            top_p=protocol.top_p,
            max_output_tokens=protocol.max_output_tokens,
            seed=global_seed + 10 + hop_index,
            extra_fields={"question_snapshot": question},
        )
        turn_rows.append(enough_result)
        enough_payload = parse_enough_answer_output(enough_result["validated_output"]["text"])
        enough_answer_decision = enough_payload["decision"]
        answer_attempt_rows.append(
            {
                "run_id": run_id,
                "dataset": benchmark_slug,
                "split": split_name,
                "sample_id": sample.sample_id,
                "method_name": method.name,
                "hop_index": hop_index,
                "attempt_kind": "enough_answer",
                "decision": enough_payload["decision"],
                "prediction": enough_payload["answer_text"],
                "raw_text": enough_payload["raw_text"],
                "reasoning_triples": list(reasoning_triples),
            }
        )
        if enough_payload["decision"] == "yes":
            final_prediction = enough_payload["answer_text"] or enough_payload["raw_text"]
            break

        if hop_index == hop_limit:
            if protocol.direct_fallback_enabled:
                fallback_result = _execute_plain_turn(
                    run_id=run_id,
                    dataset=benchmark_slug,
                    split_name=split_name,
                    sample=sample,
                    method_name=method.name,
                    role="direct_fallback",
                    hop_index=hop_index,
                    messages=build_direct_answer_messages(original_question),
                    backbone=backbone,
                    provider=provider,
                    cache=cache,
                    limiter=limiter,
                    temperature=protocol.fallback_temperature,
                    top_p=protocol.top_p,
                    max_output_tokens=protocol.max_output_tokens,
                    seed=global_seed + 1000 + hop_index,
                    extra_fields={"question_snapshot": original_question},
                )
                turn_rows.append(fallback_result)
                final_prediction = fallback_result["validated_output"]["text"]
                used_direct_fallback = True
                answer_attempt_rows.append(
                    {
                        "run_id": run_id,
                        "dataset": benchmark_slug,
                        "split": split_name,
                        "sample_id": sample.sample_id,
                        "method_name": method.name,
                        "hop_index": hop_index,
                        "attempt_kind": "direct_fallback",
                        "decision": "used",
                        "prediction": final_prediction,
                        "raw_text": final_prediction,
                        "reasoning_triples": list(reasoning_triples),
                    }
                )
            break

        if method.mode == "paper_dog":
            simplification_payload = _run_simplification_team(
                run_id=run_id,
                dataset=benchmark_slug,
                split_name=split_name,
                sample=sample,
                method_name=method.name,
                hop_index=hop_index,
                question=question,
                reasoning_triples=reasoning_triples,
                backbone=backbone,
                provider=provider,
                cache=cache,
                limiter=limiter,
                temperature=protocol.simplifier_temperature,
                top_p=protocol.top_p,
                max_output_tokens=protocol.max_output_tokens,
                seed_base=global_seed + hop_index * 100,
            )
            turn_rows.extend(simplification_payload["turn_rows"])
            debate_rows.extend(simplification_payload["debate_rows"])
            simplification_rows.append(simplification_payload["trace_row"])
            if simplification_payload["trace_row"]["simplified_question"]:
                question = simplification_payload["trace_row"]["simplified_question"]
                last_simplified_question = question
                simplification_success = True
        head_entities = tail_entities or head_entities

    final_prediction = final_prediction.strip()
    normalized_prediction = normalize_prediction(benchmark_slug, final_prediction) if final_prediction else ""
    final_score = score_prediction(benchmark_slug, final_prediction or normalized_prediction, sample.reference_answer) if final_prediction else 0.0
    prediction_row = {
        "run_id": run_id,
        "dataset": benchmark_slug,
        "split": split_name,
        "sample_id": sample.sample_id,
        "method_name": method.name,
        "method_type": "dog_paper",
        "method_mode": method.mode,
        "model_name": backbone.name,
        "prediction": final_prediction,
        "gold": sample.reference_answer,
        "score": final_score,
        "topic_entity_id": topic_entity_id,
        "hop_index": final_hop_index,
        "selected_relations": list(selected_relations),
        "reasoning_triples": list(reasoning_triples),
        "enough_answer_decision": enough_answer_decision,
        "simplified_question": last_simplified_question,
        "used_direct_fallback": used_direct_fallback,
        "retrieval_backend": retrieval_backend_name,
        "prompt_tokens_per_question": sum(float(row.get("prompt_tokens") or 0.0) for row in turn_rows),
        "completion_tokens_per_question": sum(float(row.get("completion_tokens") or 0.0) for row in turn_rows),
        "total_tokens_per_question": sum(float(row.get("total_tokens") or 0.0) for row in turn_rows),
        "latency_ms_per_question": sum(float(row.get("latency_ms") or 0.0) for row in turn_rows),
        "calls_per_question": len(turn_rows),
        "retrieval_hops": final_hop_index,
        "reasoning_triple_count": len(reasoning_triples),
        "tail_entity_count": len(last_tail_entities),
        "simplification_success": simplification_success,
        "question_changed": bool(last_simplified_question),
        "matched_control": method.matched_controls[0] if method.matched_controls else None,
    }
    return {
        "turn_rows": turn_rows,
        "debate_rows": debate_rows,
        "graph_rows": graph_rows,
        "retrieval_rows": retrieval_rows,
        "relation_rows": relation_rows,
        "simplification_rows": simplification_rows,
        "answer_attempt_rows": answer_attempt_rows,
        "prediction_row": prediction_row,
    }


def _execute_plain_turn(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    method_name: str,
    role: str,
    hop_index: int,
    messages: list[dict[str, str]],
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    seed: int,
    extra_fields: dict[str, Any] | None = None,
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
        validator=validate_plain_text_output,
        use_response_format=False,
    )
    row = {
        "run_id": run_id,
        "dataset": dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "method_name": method_name,
        "method_type": "dog_paper",
        "method_mode": role,
        "hop_index": hop_index,
        "role": role,
        "prompt_hash": result.prompt_hash,
        "output_status": result.output_status,
        "prompt_tokens": float(result.usage.get("prompt_tokens") or 0.0),
        "completion_tokens": float(result.usage.get("completion_tokens") or 0.0),
        "total_tokens": float(result.usage.get("total_tokens") or 0.0),
        "latency_ms": float(result.response_payload.get("latency_ms") or 0.0),
        "cache_hit": result.cache_hit,
        "request_error": result.request_error,
        "assistant_text": result.response_payload.get("assistant_text", ""),
        "provider_reasoning_text": result.response_payload.get("provider_reasoning_text", ""),
        "validated_output": result.validated_output,
    }
    if extra_fields:
        row.update(extra_fields)
    return row


def _run_simplification_team(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    method_name: str,
    hop_index: int,
    question: str,
    reasoning_triples: list[str],
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    limiter: SlidingWindowRateLimiter,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    seed_base: int,
) -> dict[str, Any]:
    role_names = ["Question Simplifying Expert", "Critic", "Linguist"]
    history: list[dict[str, str]] = []
    turn_rows: list[dict[str, Any]] = []
    debate_rows: list[dict[str, Any]] = []
    final_text = ""

    for role_index, role_name in enumerate(role_names, start=1):
        turn_row = _execute_plain_turn(
            run_id=run_id,
            dataset=dataset,
            split_name=split_name,
            sample=sample,
            method_name=method_name,
            role=f"simplifier:{role_name}",
            hop_index=hop_index,
            messages=build_simplifier_messages(
                question=question,
                reasoning_triples=reasoning_triples,
                role_name=role_name,
                chat_history=history,
            ),
            backbone=backbone,
            provider=provider,
            cache=cache,
            limiter=limiter,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            seed=seed_base + role_index,
            extra_fields={"agent_id": role_index},
        )
        turn_rows.append(turn_row)
        current_text = str(turn_row["validated_output"]["text"]).strip()
        history.append({"role": role_name, "content": current_text})
        final_text = current_text
        if role_index > 1:
            previous = turn_rows[role_index - 2]
            debate_rows.append(
                {
                    "run_id": run_id,
                    "dataset": dataset,
                    "split": split_name,
                    "sample_id": sample.sample_id,
                    "method_name": method_name,
                    "method_mode": "paper_simplification",
                    "round_index": hop_index,
                    "sender_agent_id": role_index - 1,
                    "recipient_agent_id": role_index,
                    "sender_answer": str(previous["validated_output"]["text"]).strip(),
                    "sender_reasoning": "",
                    "sender_evidence_triples": list(reasoning_triples),
                    "sender_answer_path": [],
                    "sender_graph_view_kind": "dynamic_retrieval",
                    "communication_grounded": bool(reasoning_triples),
                }
            )

    simplified_question = parse_simplified_question(final_text, question)
    return {
        "turn_rows": turn_rows,
        "debate_rows": debate_rows,
        "trace_row": {
            "run_id": run_id,
            "dataset": dataset,
            "split": split_name,
            "sample_id": sample.sample_id,
            "method_name": method_name,
            "hop_index": hop_index,
            "original_question": question,
            "simplified_question": simplified_question,
            "question_changed": bool(simplified_question),
            "reasoning_triples": list(reasoning_triples),
            "role_outputs": {row["role"]: row["validated_output"]["text"] for row in turn_rows},
        },
    }


def _unique_entities(entities: list[EntityRef]) -> list[EntityRef]:
    seen: set[tuple[str, str]] = set()
    unique: list[EntityRef] = []
    for entity in entities:
        key = (entity.entity_id, entity.label)
        if key in seen:
            continue
        seen.add(key)
        unique.append(entity)
    return unique


def _build_metrics(prediction_rows: list[dict[str, Any]], methods: list[GraphMethodSpec]) -> dict[str, Any]:
    method_map = {method.name: method for method in methods}
    summary_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in prediction_rows:
        grouped.setdefault((row["dataset"], row["model_name"], row["method_name"]), []).append(row)
    for (dataset, model_name, method_name), rows in sorted(grouped.items()):
        summary_rows.append(_aggregate_summary_row(dataset, model_name, method_name, rows, method_map))
    overall_groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in prediction_rows:
        overall_groups.setdefault((row["model_name"], row["method_name"]), []).append(row)
    for (model_name, method_name), rows in sorted(overall_groups.items()):
        summary_rows.append(_aggregate_summary_row("overall", model_name, method_name, rows, method_map))

    lookup = {(row["dataset"], row["model_name"], row["method_name"]): row for row in summary_rows}
    for row in summary_rows:
        control_name = row.get("matched_control")
        control_row = lookup.get((row["dataset"], row["model_name"], control_name)) if control_name else None
        row["gain_over_baseline"] = round(row["accuracy_mean"] - control_row["accuracy_mean"], 6) if control_row else None
        row["token_ratio_vs_baseline"] = (
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
    method = method_map[method_name]
    total_tokens_mean = safe_mean(float(row["total_tokens_per_question"]) for row in rows)
    return {
        "dataset": dataset,
        "model_name": model_name,
        "method_name": method_name,
        "method_type": "dog_paper",
        "method_mode": method.mode,
        "prediction_rows": len(rows),
        "question_count": len(rows),
        "accuracy_mean": safe_mean(float(row["score"]) for row in rows),
        "prompt_tokens_mean": safe_mean(float(row["prompt_tokens_per_question"]) for row in rows),
        "completion_tokens_mean": safe_mean(float(row["completion_tokens_per_question"]) for row in rows),
        "total_tokens_mean": total_tokens_mean,
        "calls_per_question_mean": safe_mean(float(row["calls_per_question"]) for row in rows),
        "latency_ms_mean": safe_mean(float(row["latency_ms_per_question"]) for row in rows),
        "accuracy_per_1k_tokens": round(safe_mean(float(row["score"]) for row in rows) / total_tokens_mean * 1000, 6) if total_tokens_mean else 0.0,
        "retrieval_hops_mean": safe_mean(float(row["retrieval_hops"]) for row in rows),
        "reasoning_triple_count_mean": safe_mean(float(row["reasoning_triple_count"]) for row in rows),
        "tail_entity_count_mean": safe_mean(float(row["tail_entity_count"]) for row in rows),
        "direct_fallback_rate": safe_mean(1.0 if row.get("used_direct_fallback") else 0.0 for row in rows),
        "simplification_success_rate": safe_mean(1.0 if row.get("simplification_success") else 0.0 for row in rows),
        "question_change_rate": safe_mean(1.0 if row.get("question_changed") else 0.0 for row in rows),
        "enough_answer_yes_rate": safe_mean(1.0 if row.get("enough_answer_decision") == "yes" else 0.0 for row in rows),
        "matched_control": method.matched_controls[0] if method.matched_controls else None,
    }


def _build_graph_diagnostics(
    prediction_rows: list[dict[str, Any]],
    retrieval_rows: list[dict[str, Any]],
    relation_rows: list[dict[str, Any]],
    simplification_rows: list[dict[str, Any]],
    answer_attempt_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summary_rows: list[dict[str, Any]] = []
    grouped_predictions: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in prediction_rows:
        grouped_predictions.setdefault((row["dataset"], row["method_name"]), []).append(row)
    for (dataset, method_name), rows in sorted(grouped_predictions.items()):
        relevant_relations = [row for row in relation_rows if row["dataset"] == dataset and row["method_name"] == method_name]
        relevant_attempts = [row for row in answer_attempt_rows if row["dataset"] == dataset and row["method_name"] == method_name]
        relevant_simplifications = [row for row in simplification_rows if row["dataset"] == dataset and row["method_name"] == method_name]
        summary_rows.append(
            {
                "dataset": dataset,
                "method_name": method_name,
                "accuracy_mean": safe_mean(float(row["score"]) for row in rows),
                "retrieval_hops_mean": safe_mean(float(row["retrieval_hops"]) for row in rows),
                "reasoning_triple_count_mean": safe_mean(float(row["reasoning_triple_count"]) for row in rows),
                "simplification_success_rate": safe_mean(1.0 if row.get("question_changed") else 0.0 for row in relevant_simplifications),
                "direct_fallback_rate": safe_mean(1.0 if row.get("used_direct_fallback") else 0.0 for row in rows),
                "false_positive_relation_rate": safe_mean(1.0 if row.get("empty_tail_expansion") else 0.0 for row in relevant_relations),
                "enough_answer_yes_rate": safe_mean(1.0 if row.get("decision") == "yes" else 0.0 for row in relevant_attempts if row.get("attempt_kind") == "enough_answer"),
            }
        )
    overall_method_names = sorted({str(row["method_name"]) for row in prediction_rows})
    for method_name in overall_method_names:
        rows = [row for row in prediction_rows if row["method_name"] == method_name]
        relevant_relations = [row for row in relation_rows if row["method_name"] == method_name]
        relevant_attempts = [row for row in answer_attempt_rows if row["method_name"] == method_name]
        relevant_simplifications = [row for row in simplification_rows if row["method_name"] == method_name]
        summary_rows.append(
            {
                "dataset": "overall",
                "method_name": method_name,
                "accuracy_mean": safe_mean(float(row["score"]) for row in rows),
                "retrieval_hops_mean": safe_mean(float(row["retrieval_hops"]) for row in rows),
                "reasoning_triple_count_mean": safe_mean(float(row["reasoning_triple_count"]) for row in rows),
                "simplification_success_rate": safe_mean(1.0 if row.get("question_changed") else 0.0 for row in relevant_simplifications),
                "direct_fallback_rate": safe_mean(1.0 if row.get("used_direct_fallback") else 0.0 for row in rows),
                "false_positive_relation_rate": safe_mean(1.0 if row.get("empty_tail_expansion") else 0.0 for row in relevant_relations),
                "enough_answer_yes_rate": safe_mean(1.0 if row.get("decision") == "yes" else 0.0 for row in relevant_attempts if row.get("attempt_kind") == "enough_answer"),
            }
        )

    view_rows: list[dict[str, Any]] = []
    grouped_retrievals: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in retrieval_rows:
        grouped_retrievals.setdefault((row["dataset"], row["method_name"]), []).append(row)
    for (dataset, method_name), rows in sorted(grouped_retrievals.items()):
        view_rows.append(
            {
                "dataset": dataset,
                "method_name": method_name,
                "graph_view_kind": "dynamic_retrieval",
                "turn_count": len(rows),
                "subgraph_node_count_mean": safe_mean(float(len(row.get("head_entities", [])) + len(row.get("tail_entities", []))) for row in rows),
                "subgraph_edge_count_mean": safe_mean(float(len(row.get("reasoning_triples", []))) for row in rows),
                "available_triple_count_mean": safe_mean(float(len(row.get("reasoning_triples", []))) for row in rows),
                "grounded_turn_rate": safe_mean(1.0 if row.get("reasoning_triples") else 0.0 for row in rows),
            }
        )
    return {
        "summary_rows": summary_rows,
        "view_rows": view_rows,
        "retrieval_trace_rows": len(retrieval_rows),
        "relation_selection_trace_rows": len(relation_rows),
        "simplification_trace_rows": len(simplification_rows),
        "answer_attempt_trace_rows": len(answer_attempt_rows),
    }


def _build_cost_breakdown(turn_rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in turn_rows:
        key = (row["dataset"], row["method_name"], row["role"])
        bucket = grouped.setdefault(
            key,
            {
                "prompt_tokens": 0.0,
                "completion_tokens": 0.0,
                "total_tokens": 0.0,
                "latency_ms": 0.0,
                "turn_count": 0.0,
            },
        )
        bucket["prompt_tokens"] += float(row.get("prompt_tokens") or 0.0)
        bucket["completion_tokens"] += float(row.get("completion_tokens") or 0.0)
        bucket["total_tokens"] += float(row.get("total_tokens") or 0.0)
        bucket["latency_ms"] += float(row.get("latency_ms") or 0.0)
        bucket["turn_count"] += 1
    rows = []
    for (dataset, method_name, role), bucket in sorted(grouped.items()):
        rows.append({"dataset": dataset, "method_name": method_name, "role": role} | bucket)
    return {"rows": rows}
