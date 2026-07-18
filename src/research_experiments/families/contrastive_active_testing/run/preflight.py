"""CATCH-v2/v3 不使用金标的一次性结构可行性预检。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research_experiments.core.data.datasets import question_without_answer_contract
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
from research_experiments.families.contrastive_active_testing.icv import (
    ContrastValidation,
    IcvWitnessParseResult,
    TargetPair,
    build_icv_witness_packet,
    build_target_pairs,
    coordinate_to_dict,
    evidence_unit_to_dict,
    parse_icv_witness,
    segment_stage_evidence,
    validate_contrast_selector,
)
from research_experiments.families.contrastive_active_testing.prompts import (
    build_designer_messages,
    build_icv_selector_messages,
    build_icv_witness_messages,
    build_witness_messages,
)
from research_experiments.families.contrastive_active_testing.run.sample import (
    _answer_turn,
    _json_turn,
)


class PreflightGateFailed(RuntimeError):
    def __init__(self, payload: dict[str, Any]) -> None:
        super().__init__(f"CATCH structural preflight failed: {payload.get('status')}")
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


@dataclass(frozen=True)
class _IcvDesignState:
    stage_state: _StageState
    row: dict[str, Any]
    pairs: tuple[TargetPair, ...]
    validation: ContrastValidation


_HUMAN_AUDIT_CRITERIA = (
    "decidable",
    "mutually_exclusive",
    "atomic",
    "answer_leakage",
)


def evaluate_icv_human_audit(
    payload: dict[str, Any],
    *,
    expected_coordinate_hashes: set[str],
) -> dict[str, Any]:
    """Recompute the blind coordinate audit from its item-level labels."""

    items = payload.get("items")
    if not isinstance(items, list):
        items = []
    hashes = [str(item.get("coordinate_sha256") or "") for item in items if isinstance(item, dict)]
    unique_hashes = set(hashes)
    complete = True
    disagreement_count = 0
    adjudicated_disagreement_count = 0
    final_values = {criterion: [] for criterion in _HUMAN_AUDIT_CRITERIA}
    pooled_first: list[bool] = []
    pooled_second: list[bool] = []

    for item in items:
        if not isinstance(item, dict):
            complete = False
            continue
        first = item.get("annotator_1")
        second = item.get("annotator_2")
        adjudication = item.get("adjudication")
        if not isinstance(first, dict) or not isinstance(second, dict):
            complete = False
            continue
        for criterion in _HUMAN_AUDIT_CRITERIA:
            first_value = first.get(criterion)
            second_value = second.get(criterion)
            if not isinstance(first_value, bool) or not isinstance(second_value, bool):
                complete = False
                continue
            if criterion != "answer_leakage":
                pooled_first.append(first_value)
                pooled_second.append(second_value)
            if first_value == second_value:
                final_values[criterion].append(first_value)
                continue
            disagreement_count += 1
            if not isinstance(adjudication, dict) or not isinstance(adjudication.get(criterion), bool):
                complete = False
                continue
            adjudicated_disagreement_count += 1
            final_values[criterion].append(bool(adjudication[criterion]))

    rates = {
        criterion: _ratio(sum(values), len(values))
        for criterion, values in final_values.items()
    }
    kappa = _cohen_kappa(pooled_first, pooled_second)
    conditions = {
        "audit_version": payload.get("audit_version") == "catch_v3_icv_blind_coordinate_audit_v1",
        "blind_contract": payload.get("blind_to_gold_votes_and_candidate_answers") is True,
        "exactly_40_unique_items": len(items) == 40 and len(unique_hashes) == 40 and all(hashes),
        "coordinate_hashes_match_preflight_sample": unique_hashes == expected_coordinate_hashes,
        "two_complete_annotators": complete and all(
            len(values) == 40 for values in final_values.values()
        ),
        "all_disagreements_adjudicated": disagreement_count == adjudicated_disagreement_count,
        "decidable_at_least_90_percent": rates["decidable"] >= 0.90,
        "exclusive_at_least_90_percent": rates["mutually_exclusive"] >= 0.90,
        "atomic_at_least_90_percent": rates["atomic"] >= 0.90,
        "answer_leakage_is_zero": rates["answer_leakage"] == 0.0,
        "pooled_non_leakage_kappa_at_least_0_6": kappa >= 0.60,
    }
    return {
        "passed": all(conditions.values()),
        "conditions": conditions,
        "item_count": len(items),
        "coordinate_hashes": sorted(unique_hashes),
        "rates": {
            "decidable": rates["decidable"],
            "mutually_exclusive": rates["mutually_exclusive"],
            "atomic": rates["atomic"],
            "answer_leakage": rates["answer_leakage"],
        },
        "cohen_kappa_pooled_non_leakage": kappa,
        "disagreement_count": disagreement_count,
        "adjudicated_disagreement_count": adjudicated_disagreement_count,
    }


def run_icv_structural_preflight(
    jobs: list[PreflightJob],
    *,
    run_id: str,
    experiment,
    protocol,
    network_budget,
    progress,
    turns_path: Path,
    output_path: Path,
    config_sha: str,
) -> dict[str, Any]:
    """Run the one-shot CATCH-v3 selector/witness feasibility experiment."""

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
            _record_preflight_sample(progress, "stage_a_ready_samples")

        selected = _stratified_preflight_states(
            [state for state in stage_states if state.stage.triggered],
            count=protocol.preflight_sample_count,
            seed=experiment.global_seed,
        )
        design_states: list[_IcvDesignState] = []
        for _, state in iter_indexed_batch(
            selected,
            worker=lambda state: _run_icv_selector(
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
            _record_preflight_sample(progress, "selector_completed_samples")

        selector_gate = _icv_selector_gate(
            design_states,
            expected_count=protocol.preflight_sample_count,
            coverage_threshold=protocol.preflight_code_coverage_threshold,
        )
        if not selector_gate["passed"]:
            payload = _icv_preflight_payload(
                "selector_failed", selected, selector_gate, None, run_id=run_id, config_sha=config_sha
            )
            _write_json(output_path, payload)
            _write_human_audit_sample(
                output_path,
                design_states,
                seed=experiment.global_seed,
                run_id=run_id,
                config_sha=config_sha,
            )
            raise PreflightGateFailed(payload)

        eligible = [state for state in design_states if state.validation.eligible_challengers]
        witness_results: list[dict[str, Any]] = []
        for _, result in iter_indexed_batch(
            eligible,
            worker=lambda state: _run_icv_witness_pair(
                state,
                run_id=run_id,
                experiment=experiment,
                protocol=protocol,
                network_budget=network_budget,
            ),
            max_concurrent_requests=experiment.max_concurrent_requests,
        ):
            witness_results.append(result)
            _write_rows(handle, tuple(result["rows"]), progress, int(result["sequence_index"]))
            _record_preflight_sample(progress, "witness_completed_samples")

    witness_gate = _icv_witness_gate(
        witness_results,
        coordinate_threshold=protocol.preflight_coordinate_validity_threshold,
        decisive_threshold=protocol.preflight_decisive_threshold,
        usable_pair_threshold=protocol.preflight_usable_pair_threshold,
        agreement_threshold=protocol.preflight_panel_agreement_threshold,
    )
    status = "passed_awaiting_human_audit" if witness_gate["passed"] else "witness_failed"
    payload = _icv_preflight_payload(
        status, selected, selector_gate, witness_gate, run_id=run_id, config_sha=config_sha
    )
    _write_json(output_path, payload)
    _write_human_audit_sample(
        output_path,
        design_states,
        seed=experiment.global_seed,
        run_id=run_id,
        config_sha=config_sha,
    )
    if not witness_gate["passed"]:
        raise PreflightGateFailed(payload)
    return payload


def _run_icv_selector(
    state: _StageState,
    *,
    run_id: str,
    experiment,
    protocol,
    network_budget,
) -> _IcvDesignState:
    sample = state.job.sample
    pairs = build_target_pairs(
        state.stage,
        seed=experiment.global_seed,
        sample_id=sample.sample_id,
    )
    evidence = segment_stage_evidence(sample, state.stage)
    row, payload = _json_turn(
        sample,
        run_id=run_id,
        split_name=state.job.split_name,
        endpoint=state.job.endpoint,
        network_budget=network_budget,
        method_name="catch_preflight_icv_selector",
        role="icv_selector",
        agent_id=1,
        seed=43_000,
        max_tokens=protocol.role_max_tokens,
        messages=build_icv_selector_messages(sample, pairs=pairs, evidence=evidence),
    )
    validation = validate_contrast_selector(
        payload,
        pairs=pairs,
        evidence=evidence,
        max_per_pair=protocol.coordinates_per_pair,
        max_total=protocol.max_selected_contrasts,
    )
    row.update(
        {
            "target_pairs": [
                {
                    "pair_id": pair.pair_id,
                    "anchor_key": pair.anchor_key,
                    "challenger_key": pair.challenger_key,
                    "left_candidate_key": pair.left_candidate_key,
                    "right_candidate_key": pair.right_candidate_key,
                }
                for pair in pairs
            ],
            "evidence_units": {
                key: [evidence_unit_to_dict(unit) for unit in units]
                for key, units in evidence.items()
            },
            "validated_contrasts": [coordinate_to_dict(item) for item in validation.coordinates],
            "dropped_contrasts": list(validation.dropped),
            "selector_protocol_error": validation.protocol_error,
            "leakage_count": validation.leakage_count,
            "eligible_challengers": list(validation.eligible_challengers),
        }
    )
    if validation.protocol_error is not None:
        row["protocol_parse_status"] = "failed"
        row["protocol_parse_error"] = validation.protocol_error
    return _IcvDesignState(state, row, pairs, validation)


def _run_icv_witness_pair(
    state: _IcvDesignState,
    *,
    run_id: str,
    experiment,
    protocol,
    network_budget,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    parsed: list[IcvWitnessParseResult] = []
    eligible_coordinates = tuple(
        item
        for item in state.validation.coordinates
        if item.challenger_key in state.validation.eligible_challengers
    )
    for panel_index in range(1, protocol.witness_count + 1):
        packet = build_icv_witness_packet(
            eligible_coordinates,
            seed=experiment.global_seed,
            sample_id=state.stage_state.job.sample.sample_id,
            panel_index=panel_index,
        )
        row, payload = _json_turn(
            state.stage_state.job.sample,
            run_id=run_id,
            split_name=state.stage_state.job.split_name,
            endpoint=state.stage_state.job.endpoint,
            network_budget=network_budget,
            method_name="catch_preflight_icv_witness",
            role="icv_witness",
            agent_id=panel_index,
            seed=44_000 + panel_index,
            max_tokens=protocol.role_max_tokens,
            messages=build_icv_witness_messages(state.stage_state.job.sample, packet=packet),
        )
        result = parse_icv_witness(payload, packet=packet)
        if not result.top_level_valid:
            row["protocol_parse_status"] = "failed"
            row["protocol_parse_error"] = "witness_top_level_schema_failure"
        row["witness_packet"] = {
            "panel_index": packet.panel_index,
            "contrasts": list(packet.contrasts),
            "public_to_internal": packet.public_to_internal,
            "public_left_to_candidate": packet.public_left_to_candidate,
            "public_right_to_candidate": packet.public_right_to_candidate,
        }
        row["witness_observations"] = result.observations
        row["witness_parse_diagnostics"] = {
            "top_level_valid": result.top_level_valid,
            "expected_coordinate_count": result.expected_coordinate_count,
            "valid_coordinate_count": result.valid_coordinate_count,
            "decisive_coordinate_count": result.decisive_coordinate_count,
            "erased_rows": list(result.erased_rows),
        }
        rows.append(row)
        parsed.append(result)
    usable_pairs = _icv_usable_pair_count(state, parsed)
    return {
        "sequence_index": state.stage_state.job.sequence_index,
        "sample_id": state.stage_state.job.sample.sample_id,
        "rows": rows,
        "parsed": parsed,
        "eligible_pair_count": len(state.validation.eligible_challengers),
        "usable_pair_count": usable_pairs,
    }


def _icv_selector_gate(
    states: list[_IcvDesignState],
    *,
    expected_count: int,
    coverage_threshold: float,
) -> dict[str, Any]:
    sample_count = len(states)
    parsed = sum(
        state.validation.protocol_error is None and not state.row.get("request_error")
        for state in states
    )
    accepted = sum(len(state.validation.coordinates) for state in states)
    dropped = sum(len(state.validation.dropped) for state in states)
    leakage = sum(state.validation.leakage_count for state in states)
    eligible_samples = sum(bool(state.validation.eligible_challengers) for state in states)
    referenced = accepted + dropped
    conditions = {
        "selected_twenty_triggered_samples": sample_count == expected_count,
        "selector_top_level_parse_rate_is_one": _ratio(parsed, sample_count) == 1.0,
        "selector_id_and_group_validity_rate_is_one": _ratio(accepted, referenced) == 1.0,
        "zero_automated_answer_leakage": leakage == 0,
        "three_coordinate_sample_coverage_at_least_60_percent": (
            _ratio(eligible_samples, sample_count) >= coverage_threshold
        ),
        "at_least_40_accepted_coordinates_for_blind_audit": accepted >= 40,
    }
    return {
        "passed": all(conditions.values()),
        "conditions": conditions,
        "sample_count": sample_count,
        "schema_parsed": parsed,
        "accepted_coordinate_count": accepted,
        "dropped_coordinate_count": dropped,
        "coordinate_reference_validity_rate": _ratio(accepted, referenced),
        "leakage_count": leakage,
        "eligible_sample_count": eligible_samples,
        "eligible_sample_rate": _ratio(eligible_samples, sample_count),
    }


def _icv_witness_gate(
    results: list[dict[str, Any]],
    *,
    coordinate_threshold: float,
    decisive_threshold: float,
    usable_pair_threshold: float,
    agreement_threshold: float,
) -> dict[str, Any]:
    panels = [panel for result in results for panel in result["parsed"]]
    expected = sum(panel.expected_coordinate_count for panel in panels)
    valid = sum(panel.valid_coordinate_count for panel in panels)
    decisive = sum(panel.decisive_coordinate_count for panel in panels)
    top_level = sum(panel.top_level_valid for panel in panels)
    eligible_pairs = sum(int(result["eligible_pair_count"]) for result in results)
    usable_pairs = sum(int(result["usable_pair_count"]) for result in results)
    agreements = 0
    comparable = 0
    for result in results:
        if len(result["parsed"]) != 2:
            continue
        first, second = result["parsed"]
        for coordinate_id in set(first.observations) & set(second.observations):
            left = first.observations[coordinate_id]
            right = second.observations[coordinate_id]
            if "ERASURE" in {left, right}:
                continue
            comparable += 1
            agreements += int(left == right)
    conditions = {
        "witness_top_level_parse_rate_is_one": _ratio(top_level, len(panels)) == 1.0,
        "valid_coordinate_rate_at_least_95_percent": _ratio(valid, expected) >= coordinate_threshold,
        "decisive_coordinate_rate_at_least_80_percent": _ratio(decisive, expected) >= decisive_threshold,
        "usable_double_panel_pair_rate_at_least_90_percent": (
            _ratio(usable_pairs, eligible_pairs) >= usable_pair_threshold
        ),
        "inverse_mapped_panel_agreement_at_least_70_percent": (
            _ratio(agreements, comparable) >= agreement_threshold
        ),
    }
    return {
        "passed": bool(results) and all(conditions.values()),
        "conditions": conditions,
        "sample_packet_count": len(results),
        "panel_count": len(panels),
        "top_level_valid_panel_count": top_level,
        "expected_coordinate_count": expected,
        "valid_coordinate_count": valid,
        "valid_coordinate_rate": _ratio(valid, expected),
        "decisive_coordinate_count": decisive,
        "decisive_coordinate_rate": _ratio(decisive, expected),
        "eligible_pair_count": eligible_pairs,
        "usable_pair_count": usable_pairs,
        "usable_pair_rate": _ratio(usable_pairs, eligible_pairs),
        "agreement_count": agreements,
        "comparable_panel_coordinate_count": comparable,
        "panel_agreement_rate": _ratio(agreements, comparable),
    }


def _icv_usable_pair_count(state: _IcvDesignState, panels: list[IcvWitnessParseResult]) -> int:
    if len(panels) != 2 or any(not panel.top_level_valid for panel in panels):
        return 0
    count = 0
    anchor = state.stage_state.stage.anchor_key
    for challenger in state.validation.eligible_challengers:
        ids = {
            item.contrast_id
            for item in state.validation.coordinates
            if item.challenger_key == challenger
        }
        if all(
            sum(
                panel.observations.get(coordinate_id) in {anchor, challenger}
                for coordinate_id in ids
            )
            >= 2
            for panel in panels
        ):
            count += 1
    return count


def _icv_preflight_payload(
    status: str,
    selected: list[_StageState],
    selector,
    witness,
    *,
    run_id: str,
    config_sha: str,
) -> dict[str, Any]:
    return {
        "preflight_version": "catch_v3_icv_structural_preflight_v1",
        "status": status,
        "passed": status == "passed_awaiting_human_audit",
        "uses_gold": False,
        "source_preflight_run_id": run_id,
        "source_config_sha256": config_sha,
        "selected_sample_ids": [state.job.sample.sample_id for state in selected],
        "selector": selector,
        "witness": witness,
        "next_action": "complete_blinded_human_audit" if status == "passed_awaiting_human_audit" else "terminate_catch_line",
    }


def _write_human_audit_sample(
    output_path: Path,
    states: list[_IcvDesignState],
    *,
    seed: int,
    run_id: str,
    config_sha: str,
) -> None:
    items: list[dict[str, Any]] = []
    for state in states:
        source = question_without_answer_contract(state.stage_state.job.sample)
        for coordinate in state.validation.coordinates:
            items.append(
                {
                    "sample_id": state.stage_state.job.sample.sample_id,
                    "task": state.stage_state.job.sample.metadata.get("task"),
                    "contrast_id": coordinate.contrast_id,
                    "source_material": source,
                    "statement_left": coordinate.left_text,
                    "statement_right": coordinate.right_text,
                    "coordinate_sha256": coordinate.sha256,
                    "annotator_1": {criterion: None for criterion in _HUMAN_AUDIT_CRITERIA},
                    "annotator_2": {criterion: None for criterion in _HUMAN_AUDIT_CRITERIA},
                    "adjudication": {criterion: None for criterion in _HUMAN_AUDIT_CRITERIA},
                }
            )
    items.sort(key=lambda item: _sha(f"{seed}:{item['sample_id']}:{item['coordinate_sha256']}"))
    payload = {
        "audit_version": "catch_v3_icv_blind_coordinate_audit_v1",
        "source_preflight_run_id": run_id,
        "source_config_sha256": config_sha,
        "seed": seed,
        "blind_to_gold_votes_and_candidate_answers": True,
        "adjudication_complete": False,
        "requested_item_count": 40,
        "item_count": min(40, len(items)),
        "criteria": ["decidable", "mutually_exclusive", "atomic", "answer_leakage"],
        "items": items[:40],
    }
    target = output_path.with_name("preflight_human_audit_sample.json")
    _write_json(target, payload)


def _record_preflight_sample(progress, field: str) -> None:
    recorder = getattr(progress, "record_phase_sample", None)
    if callable(recorder):
        recorder(field.removesuffix("_samples"))


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


def _cohen_kappa(first: list[bool], second: list[bool]) -> float:
    if not first or len(first) != len(second):
        return 0.0
    count = len(first)
    observed = _ratio(sum(left == right for left, right in zip(first, second, strict=True)), count)
    first_positive = _ratio(sum(first), count)
    second_positive = _ratio(sum(second), count)
    expected = (
        first_positive * second_positive
        + (1.0 - first_positive) * (1.0 - second_positive)
    )
    if expected >= 1.0:
        # Kappa is undefined when both pooled marginals are constant.  Treat
        # that as non-passing rather than converting raw agreement into an
        # artificially perfect chance-corrected score.
        return 0.0
    return (observed - expected) / (1.0 - expected)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
