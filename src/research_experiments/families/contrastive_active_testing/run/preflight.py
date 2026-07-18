"""CATCH-v2 不使用金标的结构可行性预检。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research_experiments.core.execution.runner_common import iter_indexed_batch
from research_experiments.families.contrastive_active_testing.algorithms import (
    StageDecision,
    TestBankValidation,
    build_hypothesis_labels,
    build_stage_decision,
    build_witness_packet,
    effective_pair_coordinates,
    parse_witness_answers_detailed,
    select_tests,
    test_to_dict,
    validate_test_bank,
)
from research_experiments.families.contrastive_active_testing.prompts import (
    build_designer_messages,
    build_witness_messages,
)
from research_experiments.families.contrastive_active_testing.run.sample import (
    _answer_turn,
    _json_turn,
)


class PreflightGateFailed(RuntimeError):
    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__("CATCH-v2 structural preflight failed")
        self.payload = payload


@dataclass(frozen=True)
class PreflightJob:
    sequence_index: int
    sample: Any
    split_name: str
    endpoint: Any


@dataclass(frozen=True)
class _StageState:
    job: PreflightJob
    rows: tuple[dict[str, Any], ...]
    stage: StageDecision


@dataclass(frozen=True)
class _DesignState:
    stage_state: _StageState
    row: dict[str, Any]
    validation: TestBankValidation
    hypothesis_to_key: dict[str, str]
    selection: Any


def run_structural_preflight(
    jobs: list[PreflightJob],
    *,
    run_id: str,
    experiment,
    protocol,
    network_budget,
    progress,
    turns_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Run Stage-A, designer, and then witness feasibility checks without gold labels."""

    stage_states: list[_StageState] = []
    with turns_path.open("w", encoding="utf-8") as handle:
        for _, state in iter_indexed_batch(
            jobs,
            worker=lambda job: _run_stage(
                job,
                run_id=run_id,
                protocol=protocol,
                network_budget=network_budget,
                seed=experiment.global_seed,
            ),
            max_concurrent_requests=experiment.max_concurrent_requests,
        ):
            stage_states.append(state)
            _write_rows(handle, state.rows, progress, state.job.sequence_index)

        triggered = [state for state in stage_states if state.stage.triggered]
        selected = _stratified_preflight_states(
            triggered,
            count=protocol.preflight_sample_count,
            seed=experiment.global_seed,
        )
        design_states: list[_DesignState] = []
        for _, state in iter_indexed_batch(
            selected,
            worker=lambda state: _run_designer(
                state,
                run_id=run_id,
                experiment=experiment,
                protocol=protocol,
                network_budget=network_budget,
            ),
            max_concurrent_requests=experiment.max_concurrent_requests,
        ):
            design_states.append(state)
            _write_rows(handle, (state.row,), progress, state.stage_state.job.sequence_index)

        designer_payload = _designer_gate(
            design_states,
            expected_count=protocol.preflight_sample_count,
            quote_threshold=protocol.preflight_quote_alignment_threshold,
            coverage_threshold=protocol.preflight_code_coverage_threshold,
        )
        if not designer_payload["passed"]:
            payload = _preflight_payload("designer_failed", selected, designer_payload, None)
            _write_json(output_path, payload)
            raise PreflightGateFailed(payload)

        eligible = [state for state in design_states if _selection_is_eligible(state.selection, 2)]
        witness_results = []
        for _, result in iter_indexed_batch(
            eligible,
            worker=lambda state: _run_witness_pair(
                state,
                run_id=run_id,
                experiment=experiment,
                protocol=protocol,
                network_budget=network_budget,
            ),
            max_concurrent_requests=experiment.max_concurrent_requests,
        ):
            witness_results.append(result)
            _write_rows(
                handle,
                tuple(result["rows"]),
                progress,
                int(result["sequence_index"]),
            )

    witness_payload = _witness_gate(
        witness_results,
        coordinate_threshold=protocol.preflight_coordinate_validity_threshold,
        usable_pair_threshold=protocol.preflight_usable_pair_threshold,
    )
    payload = _preflight_payload("passed" if witness_payload["passed"] else "witness_failed", selected, designer_payload, witness_payload)
    _write_json(output_path, payload)
    if not payload["passed"]:
        raise PreflightGateFailed(payload)
    return payload


def _run_stage(job: PreflightJob, *, run_id: str, protocol, network_budget, seed: int) -> _StageState:
    rows = tuple(
        _answer_turn(
            job.sample,
            run_id=run_id,
            split_name=job.split_name,
            endpoint=job.endpoint,
            network_budget=network_budget,
            method_name="catch_preflight_stage_a",
            role="stage_a_solver",
            agent_id=index,
            seed=42_000 + index,
            max_tokens=protocol.solver_max_tokens,
        )
        for index in range(1, protocol.stage_candidates + 1)
    )
    stage = build_stage_decision(list(rows), seed=seed, sample_id=job.sample.sample_id)
    return _StageState(job, rows, stage)


def _run_designer(state: _StageState, *, run_id: str, experiment, protocol, network_budget) -> _DesignState:
    sample = state.job.sample
    mapping = build_hypothesis_labels(
        state.stage,
        seed=experiment.global_seed,
        sample_id=sample.sample_id,
    )
    row, payload = _json_turn(
        sample,
        run_id=run_id,
        split_name=state.job.split_name,
        endpoint=state.job.endpoint,
        network_budget=network_budget,
        method_name="catch_preflight_designer",
        role="test_designer",
        agent_id=1,
        seed=43_000,
        max_tokens=protocol.role_max_tokens,
        messages=build_designer_messages(sample, stage=state.stage, hypothesis_to_key=mapping),
    )
    validation = validate_test_bank(
        payload,
        stage=state.stage,
        hypothesis_to_key=mapping,
        max_tests=protocol.max_proposed_tests,
    )
    row.update(
        {
            "validated_test_bank": [test_to_dict(test) for test in validation.tests],
            "dropped_tests": list(validation.dropped),
            "test_bank_protocol_error": validation.protocol_error,
            "evidence_quote_count": validation.evidence_quote_count,
            "aligned_evidence_quote_count": validation.aligned_evidence_quote_count,
            "leakage_count": validation.leakage_count,
        }
    )
    selection = select_tests(validation.tests, stage=state.stage, d_min=2, max_selected=protocol.max_selected_tests)
    return _DesignState(state, row, validation, mapping, selection)


def _run_witness_pair(state: _DesignState, *, run_id: str, experiment, protocol, network_budget) -> dict[str, Any]:
    rows = []
    parsed = []
    for panel_index in range(1, protocol.witness_count + 1):
        packet = build_witness_packet(
            state.selection.tests,
            seed=experiment.global_seed,
            sample_id=f"{state.stage_state.job.sample.sample_id}:d2",
            panel_index=panel_index,
        )
        row, payload = _json_turn(
            state.stage_state.job.sample,
            run_id=run_id,
            split_name=state.stage_state.job.split_name,
            endpoint=state.stage_state.job.endpoint,
            network_budget=network_budget,
            method_name="catch_preflight_witness_d2",
            role="blinded_witness",
            agent_id=panel_index,
            seed=44_000 + 20 + panel_index,
            max_tokens=protocol.role_max_tokens,
            messages=build_witness_messages(state.stage_state.job.sample, packet=packet),
        )
        result = parse_witness_answers_detailed(payload, packet=packet)
        row["witness_parse_diagnostics"] = {
            "top_level_valid": result.top_level_valid,
            "expected_coordinate_count": result.expected_coordinate_count,
            "valid_coordinate_count": result.valid_coordinate_count,
            "erased_rows": list(result.erased_rows),
        }
        row["witness_vector"] = result.vector
        rows.append(row)
        parsed.append(result)
    usable = _pair_is_usable(state, parsed, d_min=2)
    return {
        "sequence_index": state.stage_state.job.sequence_index,
        "sample_id": state.stage_state.job.sample.sample_id,
        "rows": rows,
        "parsed": parsed,
        "usable": usable,
    }


def _designer_gate(states: list[_DesignState], *, expected_count: int, quote_threshold: float, coverage_threshold: float) -> dict[str, Any]:
    quote_count = sum(state.validation.evidence_quote_count for state in states)
    aligned = sum(state.validation.aligned_evidence_quote_count for state in states)
    leakage = sum(state.validation.leakage_count for state in states)
    parsed = sum(state.validation.protocol_error is None and not state.row.get("request_error") for state in states)
    eligible = sum(_selection_is_eligible(state.selection, 2) for state in states)
    sample_count = len(states)
    conditions = {
        "selected_twenty_triggered_samples": sample_count == expected_count,
        "designer_schema_parse_rate_is_one": _ratio(parsed, sample_count) == 1.0,
        "evidence_quote_alignment_at_least_95_percent": _ratio(aligned, quote_count) >= quote_threshold,
        "zero_answer_or_candidate_leakage": leakage == 0,
        "d2_code_packet_coverage_at_least_60_percent": _ratio(eligible, sample_count) >= coverage_threshold,
    }
    return {
        "passed": all(conditions.values()),
        "conditions": conditions,
        "sample_count": sample_count,
        "schema_parsed": parsed,
        "evidence_quote_count": quote_count,
        "aligned_evidence_quote_count": aligned,
        "evidence_quote_alignment_rate": _ratio(aligned, quote_count),
        "leakage_count": leakage,
        "d2_code_eligible_count": eligible,
        "d2_code_packet_coverage": _ratio(eligible, sample_count),
    }


def _witness_gate(results: list[dict[str, Any]], *, coordinate_threshold: float, usable_pair_threshold: float) -> dict[str, Any]:
    parsed = [item for result in results for item in result["parsed"]]
    top_level_valid = sum(item.top_level_valid for item in parsed)
    expected_coordinates = sum(item.expected_coordinate_count for item in parsed)
    valid_coordinates = sum(item.valid_coordinate_count for item in parsed)
    usable = sum(bool(result["usable"]) for result in results)
    conditions = {
        "witness_top_level_parse_rate_is_one": _ratio(top_level_valid, len(parsed)) == 1.0,
        "valid_coordinate_rate_at_least_95_percent": _ratio(valid_coordinates, expected_coordinates) >= coordinate_threshold,
        "usable_double_panel_rate_at_least_90_percent": _ratio(usable, len(results)) >= usable_pair_threshold,
    }
    return {
        "passed": bool(results) and all(conditions.values()),
        "conditions": conditions,
        "packet_count": len(results),
        "panel_count": len(parsed),
        "top_level_valid_panel_count": top_level_valid,
        "valid_coordinate_count": valid_coordinates,
        "expected_coordinate_count": expected_coordinates,
        "valid_coordinate_rate": _ratio(valid_coordinates, expected_coordinates),
        "usable_pair_count": usable,
        "usable_pair_rate": _ratio(usable, len(results)),
    }


def _pair_is_usable(state: _DesignState, parsed: list[Any], *, d_min: int) -> bool:
    if len(parsed) != 2 or any(item.vector is None for item in parsed):
        return False
    for candidate in state.stage_state.stage.candidates:
        if candidate.key == state.stage_state.stage.anchor_key:
            continue
        if all(
            len(
                effective_pair_coordinates(
                    state.selection.tests,
                    state.stage_state.stage.anchor_key,
                    candidate.key,
                    available_test_ids=set(item.vector or {}),
                )
            )
            >= d_min
            for item in parsed
        ):
            return True
    return False


def _stratified_preflight_states(states: list[_StageState], *, count: int, seed: int) -> list[_StageState]:
    grouped: dict[str, list[_StageState]] = {}
    for state in states:
        task = str(state.job.sample.metadata.get("task") or "unknown")
        grouped.setdefault(task, []).append(state)
    for task, items in grouped.items():
        items.sort(key=lambda state: _sha(f"{seed}:{task}:{state.job.sample.sample_id}"))
    selected: list[_StageState] = []
    tasks = sorted(grouped, key=lambda task: _sha(f"{seed}:task:{task}"))
    while tasks and len(selected) < count:
        next_tasks = []
        for task in tasks:
            if grouped[task] and len(selected) < count:
                selected.append(grouped[task].pop(0))
            if grouped[task]:
                next_tasks.append(task)
        tasks = next_tasks
    return selected


def _selection_is_eligible(selection, d_min: int) -> bool:
    return any(int(distance) >= d_min for distance in selection.pair_distances.values())


def _preflight_payload(status: str, selected: list[_StageState], designer, witness) -> dict[str, Any]:
    return {
        "preflight_version": "catch_v2_structural_preflight_v1",
        "status": status,
        "passed": status == "passed",
        "uses_gold": False,
        "selected_sample_ids": [state.job.sample.sample_id for state in selected],
        "designer": designer,
        "witness": witness,
    }


def _write_rows(handle, rows, progress, sequence_index: int) -> None:
    for raw in rows:
        row = dict(raw)
        row["sample_sequence_index"] = sequence_index
        row["run_stage"] = "structural_preflight"
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        progress.record_call(row)
    handle.flush()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _ratio(numerator: float, denominator: int) -> float:
    return float(numerator) / denominator if denominator else 0.0


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
