"""MacNet 的样本级拓扑执行、控制基线与指标汇总。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import json
from typing import Any

from research_experiments.core.data.datasets import DatasetSample, load_split_ids, select_samples
from research_experiments.core.data.evaluation import normalize_prediction, score_prediction
from research_experiments.core.execution.runner_common import execute_cached_turn, run_indexed_batch
from research_experiments.families.macnet.config import MacnetExperimentConfig, MacnetMethodSpec, ProtocolConfig
from research_experiments.families.macnet.profiles import pick_profile_text
from research_experiments.families.macnet.prompts import (
    build_actor_messages,
    build_instruction_messages,
    build_single_agent_messages,
    validate_actor_output,
    validate_instruction_output,
)
from research_experiments.families.macnet.topologies import TopologySpec, build_topology
from research_experiments.families.shared.common import build_question_preview, resolve_phase_split_name, safe_mean


METHOD_ORDER = (
    "single_agent_cot",
    "macnet_chain",
    "macnet_star",
    "macnet_tree",
    "macnet_mesh",
    "macnet_layer",
    "macnet_random",
    "vote_mvN_closed",
    "best_of_n_open",
)


@dataclass(frozen=True)
class SampleResult:
    artifact_rows: list[dict[str, Any]]
    instruction_rows: list[dict[str, Any]]
    prediction_rows: list[dict[str, Any]]
    topology_specs: list[dict[str, Any]]


def _active_methods(experiment: MacnetExperimentConfig) -> list[MacnetMethodSpec]:
    methods = {method.name: method for method in experiment.methods}
    return [methods[name] for name in METHOD_ORDER if name in methods]


def _resolve_split_name(experiment: MacnetExperimentConfig, phase_name: str, benchmark_slug: str) -> str:
    return resolve_phase_split_name(experiment, phase_name, benchmark_slug)


def _load_selected_samples(benchmark, split_name: str) -> list[DatasetSample]:
    return select_samples(benchmark, split_name)


def _phase_node_scale(experiment: MacnetExperimentConfig, phase_name: str) -> int:
    return int(experiment.raw["phases"][phase_name].get("node_scale") or 4)


def _phase_direction_mode(experiment: MacnetExperimentConfig, phase_name: str, protocol: ProtocolConfig) -> str:
    return str(experiment.raw["phases"][phase_name].get("direction_mode") or protocol.default_direction_mode)


def _phase_scaling_node_scales(experiment: MacnetExperimentConfig, phase_name: str) -> list[int]:
    payload = experiment.raw["phases"][phase_name]
    return [int(item) for item in payload.get("scaling_node_scales", [])]


def _phase_direction_modes(experiment: MacnetExperimentConfig, phase_name: str, protocol: ProtocolConfig) -> list[str]:
    payload = experiment.raw["phases"][phase_name]
    modes = payload.get("direction_modes")
    if not modes:
        return [protocol.default_direction_mode]
    return [str(item) for item in modes]


def _phase_dense_cap(experiment: MacnetExperimentConfig, phase_name: str) -> int:
    return int(experiment.raw["phases"][phase_name].get("dense_topology_scale_cap") or 16)


def _estimate_work(
    experiment: MacnetExperimentConfig,
    phase_name: str,
    benchmarks,
    protocol: ProtocolConfig,
    methods: list[MacnetMethodSpec],
) -> tuple[int, int]:
    total_calls = 0
    total_predictions = 0
    if experiment.experiment_kind == "scaling_study":
        node_scales = _phase_scaling_node_scales(experiment, phase_name)
        direction_modes = _phase_direction_modes(experiment, phase_name, protocol)
        dense_cap = _phase_dense_cap(experiment, phase_name)
    else:
        node_scales = [_phase_node_scale(experiment, phase_name)]
        direction_modes = [_phase_direction_mode(experiment, phase_name, protocol)]
        dense_cap = _phase_dense_cap(experiment, phase_name)

    for benchmark in benchmarks:
        split_name = _resolve_split_name(experiment, phase_name, benchmark.slug)
        sample_count = len(load_split_ids(benchmark.cache_namespace or benchmark.slug, split_name))
        for method in methods:
            budgets = _planned_method_budgets(
                method,
                node_scales=node_scales,
                direction_modes=direction_modes,
                dense_cap=dense_cap,
                seed=experiment.global_seed,
            )
            total_calls += sample_count * sum(budget["call_budget"] for budget in budgets)
            total_predictions += sample_count * len(budgets)
    return total_calls, total_predictions


def _run_sample_batch(
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    phase_name: str,
    samples: list[DatasetSample],
    protocol: ProtocolConfig,
    experiment: MacnetExperimentConfig,
    backbone,
    provider,
    cache,
    limiter,
    profile_bank: dict[str, list[str]],
) -> list[SampleResult]:
    worker = partial(
        _run_sample,
        run_id=run_id,
        benchmark_slug=benchmark_slug,
        split_name=split_name,
        phase_name=phase_name,
        protocol=protocol,
        experiment=experiment,
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        profile_bank=profile_bank,
    )
    return [result for _, result in run_indexed_batch(samples, worker=worker, max_concurrent_requests=experiment.max_concurrent_requests)]


def _run_sample(
    sample: DatasetSample,
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    phase_name: str,
    protocol: ProtocolConfig,
    experiment: MacnetExperimentConfig,
    backbone,
    provider,
    cache,
    limiter,
    profile_bank: dict[str, list[str]],
) -> SampleResult:
    artifact_rows: list[dict[str, Any]] = []
    instruction_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    topology_specs: list[dict[str, Any]] = []
    methods = _active_methods(experiment)

    if experiment.experiment_kind == "scaling_study":
        method_budgets = [
            (method, budget)
            for method in methods
            for budget in _planned_method_budgets(
                method,
                node_scales=_phase_scaling_node_scales(experiment, phase_name),
                direction_modes=_phase_direction_modes(experiment, phase_name, protocol),
                dense_cap=_phase_dense_cap(experiment, phase_name),
                seed=experiment.global_seed + _stable_sample_seed(sample.sample_id),
            )
        ]
    else:
        method_budgets = [
            (method, budget)
            for method in methods
            for budget in _planned_method_budgets(
                method,
                node_scales=[_phase_node_scale(experiment, phase_name)],
                direction_modes=[_phase_direction_mode(experiment, phase_name, protocol)],
                dense_cap=_phase_dense_cap(experiment, phase_name),
                seed=experiment.global_seed + _stable_sample_seed(sample.sample_id),
            )
        ]

    for method, budget in method_budgets:
        if method.mode == "single_agent_cot":
            artifact_row, prediction_row = _run_single_agent_method(
                sample=sample,
                run_id=run_id,
                benchmark_slug=benchmark_slug,
                split_name=split_name,
                method=method,
                backbone=backbone,
                provider=provider,
                cache=cache,
                limiter=limiter,
                protocol=protocol,
                experiment=experiment,
                sample_seed=experiment.global_seed + _stable_sample_seed(sample.sample_id),
            )
            artifact_rows.append(artifact_row)
            prediction_rows.append(prediction_row)
            continue
        if method.mode == "macnet_topology":
            artifact_chunk, instruction_chunk, prediction_row, topology_row = _run_topology_method(
                sample=sample,
                run_id=run_id,
                benchmark_slug=benchmark_slug,
                split_name=split_name,
                method=method,
                topology_type=str(budget["topology_type"]),
                direction_mode=str(budget["direction_mode"]),
                node_scale=int(budget["node_scale"]),
                backbone=backbone,
                provider=provider,
                cache=cache,
                limiter=limiter,
                protocol=protocol,
                experiment=experiment,
                profile_bank=profile_bank,
                sample_seed=experiment.global_seed + _stable_sample_seed(sample.sample_id),
            )
            artifact_rows.extend(artifact_chunk)
            instruction_rows.extend(instruction_chunk)
            prediction_rows.append(prediction_row)
            topology_specs.append(topology_row)
            continue
        artifact_chunk, prediction_row = _run_control_method(
            sample=sample,
            run_id=run_id,
            benchmark_slug=benchmark_slug,
            split_name=split_name,
            method=method,
            call_budget=int(budget["call_budget"]),
            node_scale=int(budget["node_scale"]),
            direction_mode=str(budget["direction_mode"]),
            backbone=backbone,
            provider=provider,
            cache=cache,
            limiter=limiter,
            protocol=protocol,
            experiment=experiment,
            sample_seed=experiment.global_seed + _stable_sample_seed(sample.sample_id),
        )
        artifact_rows.extend(artifact_chunk)
        prediction_rows.append(prediction_row)

    return SampleResult(
        artifact_rows=artifact_rows,
        instruction_rows=instruction_rows,
        prediction_rows=prediction_rows,
        topology_specs=topology_specs,
    )


def _planned_method_budgets(
    method: MacnetMethodSpec,
    *,
    node_scales: list[int],
    direction_modes: list[str],
    dense_cap: int,
    seed: int,
) -> list[dict[str, Any]]:
    if method.mode == "single_agent_cot":
        return [{"node_scale": 1, "direction_mode": "none", "call_budget": 1}]
    budgets: list[dict[str, Any]] = []
    for node_scale in node_scales:
        effective_scale = node_scale
        if method.topology_type in {"mesh", "layer"} and node_scale > dense_cap:
            effective_scale = dense_cap
        for direction_mode in direction_modes:
            if method.mode == "macnet_topology":
                topology = build_topology(
                    str(method.topology_type),
                    node_count=effective_scale,
                    direction_mode=direction_mode,
                    seed=seed + effective_scale,
                )
                call_budget = topology.node_count + len(topology.edges) + (1 if len(topology.sink_nodes) > 1 else 0)
                budgets.append(
                    {
                        "node_scale": effective_scale,
                        "direction_mode": direction_mode,
                        "call_budget": call_budget,
                        "topology_type": method.topology_type,
                    }
                )
            elif method.mode in {"vote_control", "best_of_n_control"}:
                budgets.append(
                    {
                        "node_scale": effective_scale,
                        "direction_mode": direction_mode,
                        "call_budget": max(1, 2 * effective_scale - 1),
                        "topology_type": None,
                    }
                )
    return budgets


def _run_single_agent_method(
    *,
    sample: DatasetSample,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    method: MacnetMethodSpec,
    backbone,
    provider,
    cache,
    limiter,
    protocol: ProtocolConfig,
    experiment: MacnetExperimentConfig,
    sample_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    turn = _execute_actor_turn(
        run_id=run_id,
        benchmark_slug=benchmark_slug,
        split_name=split_name,
        sample=sample,
        method_name=method.name,
        node_id=0,
        topology_type="single",
        direction_mode="none",
        profile_text="You are a strong single-agent baseline. Solve the task directly.",
        parent_artifacts=[],
        inbound_instructions=[],
        prompt_version=experiment.prompt_version,
        terminal_fuse=False,
        backbone=backbone,
        provider=provider,
        cache=cache,
        limiter=limiter,
        temperature=protocol.actor_temperature,
        top_p=protocol.top_p,
        max_output_tokens=protocol.max_output_tokens,
        seed=sample_seed,
    )
    score = score_prediction(benchmark_slug, str(turn["final_answer"]), sample.reference_answer)
    prediction = {
        "run_id": run_id,
        "dataset": benchmark_slug,
        "split": split_name,
        "sample_id": sample.sample_id,
        "question_preview": build_question_preview(sample.question),
        "model_name": backbone.name,
        "method_name": method.name,
        "topology_type": "single",
        "node_scale": 1,
        "topology_direction_mode": "none",
        "initial_artifact": turn["artifact"],
        "final_artifact": turn["artifact"],
        "initial_answer": str(turn["final_answer"]),
        "final_answer": str(turn["final_answer"]),
        "prediction": normalize_prediction(benchmark_slug, str(turn["final_answer"])),
        "score": score,
        "artifact_revision_count": 0,
        "inbound_instruction_count": 0,
        "max_context_tokens_observed": float(turn["prompt_tokens"]),
        "prompt_tokens_per_question": float(turn["prompt_tokens"]),
        "completion_tokens_per_question": float(turn["completion_tokens"]),
        "total_tokens_per_question": float(turn["total_tokens"]),
        "communication_tokens_per_question": 0.0,
        "latency_ms_per_question": float(turn["latency_ms"]),
        "calls_per_question": 1,
    }
    return turn, prediction


def _run_topology_method(
    *,
    sample: DatasetSample,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    method: MacnetMethodSpec,
    topology_type: str,
    direction_mode: str,
    node_scale: int,
    backbone,
    provider,
    cache,
    limiter,
    protocol: ProtocolConfig,
    experiment: MacnetExperimentConfig,
    profile_bank: dict[str, list[str]],
    sample_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    topology = build_topology(topology_type, node_count=node_scale, direction_mode=direction_mode, seed=sample_seed)
    state_by_node: dict[int, dict[str, Any]] = {}
    artifact_rows: list[dict[str, Any]] = []
    instruction_rows: list[dict[str, Any]] = []
    topo_order = _topological_order(topology)
    for node_id in topo_order:
        parent_nodes = [source for source, target in topology.edges if target == node_id]
        inbound_packets: list[dict[str, Any]] = []
        for parent_node in parent_nodes:
            parent_state = state_by_node[parent_node]
            critic_profile = pick_profile_text(
                profile_bank=profile_bank,
                dataset_slug=benchmark_slug,
                role_kind="critic",
                seed=sample_seed + parent_node * 17 + node_id,
            )
            instruction_row = _execute_instruction_turn(
                run_id=run_id,
                benchmark_slug=benchmark_slug,
                split_name=split_name,
                sample=sample,
                method_name=method.name,
                source_node_id=parent_node,
                target_node_id=node_id,
                source_artifact=str(parent_state["artifact"]),
                source_answer=str(parent_state["final_answer"]),
                profile_text=critic_profile,
                prompt_version=experiment.prompt_version,
                backbone=backbone,
                provider=provider,
                cache=cache,
                limiter=limiter,
                temperature=protocol.critic_temperature,
                top_p=protocol.top_p,
                max_output_tokens=min(512, protocol.max_output_tokens),
                seed=sample_seed + parent_node * 101 + node_id,
            )
            instruction_rows.append(instruction_row)
            inbound_packets.append(instruction_row)
        actor_profile = pick_profile_text(
            profile_bank=profile_bank,
            dataset_slug=benchmark_slug,
            role_kind="actor",
            seed=sample_seed + node_id,
        )
        actor_row = _execute_actor_turn(
            run_id=run_id,
            benchmark_slug=benchmark_slug,
            split_name=split_name,
            sample=sample,
            method_name=method.name,
            node_id=node_id,
            topology_type=topology_type,
            direction_mode=direction_mode,
            profile_text=actor_profile,
            parent_artifacts=[state_by_node[parent_node] for parent_node in parent_nodes[: protocol.memory_control_max_parents]],
            inbound_instructions=inbound_packets[: protocol.memory_control_max_parents],
            prompt_version=experiment.prompt_version,
            terminal_fuse=False,
            backbone=backbone,
            provider=provider,
            cache=cache,
            limiter=limiter,
            temperature=protocol.actor_temperature,
            top_p=protocol.top_p,
            max_output_tokens=protocol.max_output_tokens,
            seed=sample_seed + node_id * 3,
        )
        state_by_node[node_id] = actor_row
        artifact_rows.append(actor_row)

    final_state = state_by_node[topology.sink_nodes[0]]
    if len(topology.sink_nodes) > 1:
        fuse_profile = pick_profile_text(
            profile_bank=profile_bank,
            dataset_slug=benchmark_slug,
            role_kind="actor",
            seed=sample_seed + 9999,
        )
        final_state = _execute_actor_turn(
            run_id=run_id,
            benchmark_slug=benchmark_slug,
            split_name=split_name,
            sample=sample,
            method_name=method.name,
            node_id=-1,
            topology_type=topology_type,
            direction_mode=direction_mode,
            profile_text=fuse_profile,
            parent_artifacts=[state_by_node[node_id] for node_id in topology.sink_nodes[: protocol.memory_control_max_parents]],
            inbound_instructions=[],
            prompt_version=experiment.prompt_version,
            terminal_fuse=True,
            backbone=backbone,
            provider=provider,
            cache=cache,
            limiter=limiter,
            temperature=protocol.terminal_fuse_temperature,
            top_p=protocol.top_p,
            max_output_tokens=protocol.max_output_tokens,
            seed=sample_seed + 7777,
        )
        artifact_rows.append(final_state)

    score = score_prediction(benchmark_slug, str(final_state["final_answer"]), sample.reference_answer)
    first_source = topology.source_nodes[0]
    prediction = {
        "run_id": run_id,
        "dataset": benchmark_slug,
        "split": split_name,
        "sample_id": sample.sample_id,
        "question_preview": build_question_preview(sample.question),
        "model_name": backbone.name,
        "method_name": method.name,
        "topology_type": topology_type,
        "node_scale": node_scale,
        "topology_direction_mode": direction_mode,
        "initial_artifact": state_by_node[first_source]["artifact"],
        "final_artifact": final_state["artifact"],
        "initial_answer": str(state_by_node[first_source]["final_answer"]),
        "final_answer": str(final_state["final_answer"]),
        "prediction": normalize_prediction(benchmark_slug, str(final_state["final_answer"])),
        "score": score,
        "artifact_revision_count": max(0, len(artifact_rows) - 1),
        "inbound_instruction_count": len(instruction_rows),
        "max_context_tokens_observed": max(float(row["prompt_tokens"]) for row in artifact_rows) if artifact_rows else 0.0,
        "prompt_tokens_per_question": sum(float(row["prompt_tokens"]) for row in artifact_rows + instruction_rows),
        "completion_tokens_per_question": sum(float(row["completion_tokens"]) for row in artifact_rows + instruction_rows),
        "total_tokens_per_question": sum(float(row["total_tokens"]) for row in artifact_rows + instruction_rows),
        "communication_tokens_per_question": sum(float(row["total_tokens"]) for row in instruction_rows),
        "latency_ms_per_question": sum(float(row["latency_ms"]) for row in artifact_rows + instruction_rows),
        "calls_per_question": len(artifact_rows) + len(instruction_rows),
    }
    topology_row = {
        "method_name": method.name,
        "topology_type": topology.topology_type,
        "direction_mode": topology.direction_mode,
        "node_scale": topology.node_count,
        "node_count": topology.node_count,
        "edge_count": len(topology.edges),
        "edge_density": topology.edge_density,
        "dag_depth": topology.dag_depth,
        "average_out_degree": topology.average_out_degree,
        "average_in_degree": topology.average_in_degree,
        "source_nodes": topology.source_nodes,
        "sink_nodes": topology.sink_nodes,
    }
    return artifact_rows, instruction_rows, prediction, topology_row


def _run_control_method(
    *,
    sample: DatasetSample,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    method: MacnetMethodSpec,
    call_budget: int,
    node_scale: int,
    direction_mode: str,
    backbone,
    provider,
    cache,
    limiter,
    protocol: ProtocolConfig,
    experiment: MacnetExperimentConfig,
    sample_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for offset in range(call_budget):
        row = _execute_actor_turn(
            run_id=run_id,
            benchmark_slug=benchmark_slug,
            split_name=split_name,
            sample=sample,
            method_name=method.name,
            node_id=offset,
            topology_type="control",
            direction_mode=direction_mode,
            profile_text="You are an independent control solver. Produce a clean standalone answer.",
            parent_artifacts=[],
            inbound_instructions=[],
            prompt_version=experiment.prompt_version,
            terminal_fuse=False,
            backbone=backbone,
            provider=provider,
            cache=cache,
            limiter=limiter,
            temperature=protocol.actor_temperature,
            top_p=protocol.top_p,
            max_output_tokens=protocol.max_output_tokens,
            seed=sample_seed + offset,
        )
        turns.append(row)
    if method.mode == "vote_control":
        final_answer = _majority_answer(turns)
    else:
        final_answer = _best_confidence_answer(turns)
    score = score_prediction(benchmark_slug, final_answer, sample.reference_answer)
    prediction = {
        "run_id": run_id,
        "dataset": benchmark_slug,
        "split": split_name,
        "sample_id": sample.sample_id,
        "question_preview": build_question_preview(sample.question),
        "model_name": backbone.name,
        "method_name": method.name,
        "topology_type": "control",
        "node_scale": node_scale,
        "topology_direction_mode": direction_mode,
        "initial_artifact": turns[0]["artifact"],
        "final_artifact": turns[-1]["artifact"],
        "initial_answer": str(turns[0]["final_answer"]),
        "final_answer": final_answer,
        "prediction": normalize_prediction(benchmark_slug, final_answer),
        "score": score,
        "artifact_revision_count": max(0, len(turns) - 1),
        "inbound_instruction_count": 0,
        "max_context_tokens_observed": max(float(row["prompt_tokens"]) for row in turns),
        "prompt_tokens_per_question": sum(float(row["prompt_tokens"]) for row in turns),
        "completion_tokens_per_question": sum(float(row["completion_tokens"]) for row in turns),
        "total_tokens_per_question": sum(float(row["total_tokens"]) for row in turns),
        "communication_tokens_per_question": 0.0,
        "latency_ms_per_question": sum(float(row["latency_ms"]) for row in turns),
        "calls_per_question": len(turns),
    }
    return turns, prediction


def _majority_answer(turns: list[dict[str, Any]]) -> str:
    counts: dict[str, int] = {}
    ordered: list[str] = []
    for row in turns:
        answer = str(row["final_answer"])
        ordered.append(answer)
        counts[answer] = counts.get(answer, 0) + 1
    return max(counts, key=lambda key: (counts[key], -ordered.index(key)))


def _best_confidence_answer(turns: list[dict[str, Any]]) -> str:
    best_row = max(
        turns,
        key=lambda row: (
            float(_coerce_confidence(row.get("confidence_raw")) or 0.0),
            -int(row.get("node_id") or 0),
        ),
    )
    return str(best_row["final_answer"])


def _write_sample_outputs(
    *,
    sample_results: list[SampleResult],
    dataset_slug: str,
    progress,
    artifact_handle,
    instruction_handle,
    prediction_handle,
    all_artifact_rows: list[dict[str, Any]],
    all_instruction_rows: list[dict[str, Any]],
    final_predictions: list[dict[str, Any]],
    topology_specs: list[dict[str, Any]],
) -> None:
    for result in sample_results:
        for row in result.artifact_rows:
            artifact_handle.write_row(row)
            progress.record_call(row)
        for row in result.instruction_rows:
            instruction_handle.write_row(row)
            progress.record_call(row)
        for row in result.prediction_rows:
            prediction_handle.write_row(row)
            progress.record_predictions(1, dataset_slug, str(row["method_name"]))
        all_artifact_rows.extend(result.artifact_rows)
        all_instruction_rows.extend(result.instruction_rows)
        final_predictions.extend(result.prediction_rows)
        topology_specs.extend(result.topology_specs)


def _build_metrics(
    prediction_rows: list[dict[str, Any]],
    *,
    model_name: str,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, int, str], list[dict[str, Any]]] = {}
    for row in prediction_rows:
        key = (
            str(row["dataset"]),
            str(row["method_name"]),
            int(row["node_scale"]),
            str(row["topology_direction_mode"]),
        )
        grouped.setdefault(key, []).append(row)
    summary: list[dict[str, Any]] = []
    for (dataset, method_name, node_scale, direction_mode), rows in sorted(grouped.items()):
        summary.append(
            {
                "dataset": dataset,
                "model_name": model_name,
                "method_name": method_name,
                "node_scale": node_scale,
                "topology_direction_mode": direction_mode,
                "quality_mean": round(safe_mean(float(row.get("score") or 0.0) for row in rows), 6),
                "total_tokens_mean": round(safe_mean(float(row.get("total_tokens_per_question") or 0.0) for row in rows), 6),
                "communication_tokens_mean": round(safe_mean(float(row.get("communication_tokens_per_question") or 0.0) for row in rows), 6),
                "calls_per_question_mean": round(safe_mean(float(row.get("calls_per_question") or 0.0) for row in rows), 6),
                "artifact_revision_count_mean": round(safe_mean(float(row.get("artifact_revision_count") or 0.0) for row in rows), 6),
                "inbound_instruction_count_mean": round(safe_mean(float(row.get("inbound_instruction_count") or 0.0) for row in rows), 6),
                "max_context_tokens_observed_mean": round(safe_mean(float(row.get("max_context_tokens_observed") or 0.0) for row in rows), 6),
                "quality_per_1k_tokens": _quality_per_1k_tokens(rows),
                "question_count": len(rows),
            }
        )
    return {"summary": summary, "prediction_count": len(prediction_rows)}


def _build_scaling_summary(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for row in prediction_rows:
        key = f"{row['method_name']}::{row['topology_direction_mode']}"
        rows.setdefault(key, []).append(row)
    payload: list[dict[str, Any]] = []
    for key, items in sorted(rows.items()):
        method_name, direction_mode = key.split("::", 1)
        by_scale: dict[int, list[dict[str, Any]]] = {}
        for item in items:
            by_scale.setdefault(int(item["node_scale"]), []).append(item)
        payload.append(
            {
                "method_name": method_name,
                "topology_direction_mode": direction_mode,
                "scales": [
                    {
                        "node_scale": scale,
                        "quality_mean": round(safe_mean(float(row.get("score") or 0.0) for row in rows_at_scale), 6),
                        "total_tokens_mean": round(safe_mean(float(row.get("total_tokens_per_question") or 0.0) for row in rows_at_scale), 6),
                    }
                    for scale, rows_at_scale in sorted(by_scale.items())
                ],
            }
        )
    return {"series": payload}


def _execute_actor_turn(
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    sample: DatasetSample,
    method_name: str,
    node_id: int,
    topology_type: str,
    direction_mode: str,
    profile_text: str,
    parent_artifacts: list[dict[str, Any]],
    inbound_instructions: list[dict[str, Any]],
    prompt_version: str,
    terminal_fuse: bool,
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
        messages=build_actor_messages(
            sample,
            node_id=node_id,
            topology_type=topology_type,
            direction_mode=direction_mode,
            profile_text=profile_text,
            parent_artifacts=parent_artifacts,
            inbound_instructions=inbound_instructions,
            prompt_version=prompt_version,
            terminal_fuse=terminal_fuse,
        ),
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        seed=seed,
        validator=lambda raw_text, _: validate_actor_output(raw_text, benchmark_slug),
        dataset=benchmark_slug,
    )
    validated = result.validated_output
    return {
        "run_id": run_id,
        "dataset": benchmark_slug,
        "split": split_name,
        "sample_id": sample.sample_id,
        "method_name": method_name,
        "node_id": node_id,
        "role": "actor",
        "topology_type": topology_type,
        "direction_mode": direction_mode,
        "terminal_fuse": terminal_fuse,
        "artifact": str(validated.get("artifact") or ""),
        "final_answer": str(validated.get("final_answer") or ""),
        "reasoning_trace": str(validated.get("reasoning_trace") or ""),
        "confidence_raw": validated.get("confidence_raw"),
        "output_status": result.output_status,
        "prompt_tokens": float(result.usage.get("prompt_tokens") or 0.0),
        "completion_tokens": float(result.usage.get("completion_tokens") or 0.0),
        "total_tokens": float(result.usage.get("total_tokens") or 0.0),
        "latency_ms": float(result.response_payload.get("latency_ms") or 0.0),
        "cache_hit": result.cache_hit,
    }


def _execute_instruction_turn(
    *,
    run_id: str,
    benchmark_slug: str,
    split_name: str,
    sample: DatasetSample,
    method_name: str,
    source_node_id: int,
    target_node_id: int,
    source_artifact: str,
    source_answer: str,
    profile_text: str,
    prompt_version: str,
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
        messages=build_instruction_messages(
            sample,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            source_artifact=source_artifact,
            source_answer=source_answer,
            profile_text=profile_text,
            prompt_version=prompt_version,
        ),
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        seed=seed,
        validator=lambda raw_text, _: validate_instruction_output(raw_text),
        dataset=benchmark_slug,
    )
    validated = result.validated_output
    return {
        "run_id": run_id,
        "dataset": benchmark_slug,
        "split": split_name,
        "sample_id": sample.sample_id,
        "method_name": method_name,
        "source_node_id": source_node_id,
        "target_node_id": target_node_id,
        "instruction": str(validated.get("instruction") or ""),
        "focus_risk": str(validated.get("focus_risk") or ""),
        "preserve_strength": str(validated.get("preserve_strength") or ""),
        "output_status": result.output_status,
        "prompt_tokens": float(result.usage.get("prompt_tokens") or 0.0),
        "completion_tokens": float(result.usage.get("completion_tokens") or 0.0),
        "total_tokens": float(result.usage.get("total_tokens") or 0.0),
        "latency_ms": float(result.response_payload.get("latency_ms") or 0.0),
        "cache_hit": result.cache_hit,
    }


def _topological_order(topology: TopologySpec) -> list[int]:
    indegree = {node_id: 0 for node_id in range(topology.node_count)}
    adjacency: dict[int, list[int]] = {node_id: [] for node_id in range(topology.node_count)}
    for source, target in topology.edges:
        adjacency[source].append(target)
        indegree[target] += 1
    queue = sorted(node_id for node_id, value in indegree.items() if value == 0)
    order: list[int] = []
    while queue:
        node_id = queue.pop(0)
        order.append(node_id)
        for target in adjacency[node_id]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
                queue.sort()
    if len(order) != topology.node_count:
        raise ValueError("Topology is not acyclic.")
    return order


def _quality_per_1k_tokens(rows: list[dict[str, Any]]) -> float:
    quality = safe_mean(float(row.get("score") or 0.0) for row in rows)
    total_tokens = safe_mean(float(row.get("total_tokens_per_question") or 0.0) for row in rows)
    if total_tokens <= 0:
        return 0.0
    return round(quality / (total_tokens / 1000.0), 6)


def _coerce_confidence(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
    else:
        text = str(value).strip().rstrip("%")
        if not text:
            return None
        try:
            numeric = float(text)
        except ValueError:
            return None
    if numeric > 1.0:
        numeric = numeric / 100.0
    return max(0.0, min(1.0, numeric))


def _stable_sample_seed(sample_id: str) -> int:
    return sum(ord(char) for char in sample_id)
