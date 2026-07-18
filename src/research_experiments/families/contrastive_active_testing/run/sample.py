"""共享 Stage-A、CATCH、adaptive-SC8 与 Judge-3 的逐样本执行。"""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Any

from research_experiments.core.controls.control_prompts import build_cot_messages
from research_experiments.core.data.datasets import DatasetSample, question_without_answer_contract
from research_experiments.core.data.evaluation import canonicalize_answer, score_prediction
from research_experiments.core.execution.runner_common import execute_cached_request
from research_experiments.families.contrastive_active_testing.algorithms import (
    DecodeDecision,
    build_hypothesis_labels,
    build_stage_decision,
    build_witness_packet,
    decide_direct_judges,
    decode_witnesses,
    parse_witness_answers_detailed,
    select_tests,
    test_to_dict,
    validate_test_bank,
)
from research_experiments.families.contrastive_active_testing.icv import (
    ContrastValidation,
    IcvWitnessParseResult,
    build_icv_witness_packet,
    build_target_pairs,
    coordinate_to_dict,
    decode_icv,
    evidence_unit_to_dict,
    parse_icv_witness,
    segment_stage_evidence,
    validate_contrast_selector,
)
from research_experiments.families.contrastive_active_testing.prompts import (
    build_designer_messages,
    build_direct_judge_messages,
    build_icv_selector_messages,
    build_icv_witness_messages,
    build_pair_judge_messages,
    build_witness_messages,
)
from research_experiments.family_runtime.free_text_protocol import FREE_TEXT_ANSWER_PROTOCOL_V1
from research_experiments.family_runtime.output_protocols import execute_output_protocol_turn


class CatchRunCancelled(RuntimeError):
    pass


class NetworkAttemptBudget:
    """Thread-safe soft network-attempt counter.

    ``limit`` is an operational warning threshold, not an admission gate.  The
    counter deliberately never raises when the configured threshold is crossed:
    long-running CATCH experiments must finish collecting the remaining samples
    so an isolated retry burst cannot invalidate an otherwise usable run.
    """

    def __init__(self, limit: int) -> None:
        self.limit = int(limit)
        self.actual = 0
        self._reserved = 0
        self.limit_exceeded = False
        self.first_exceeded_at: int | None = None
        self._lock = threading.Lock()

    def reserve(self, maximum: int = 5) -> int:
        with self._lock:
            self._reserved += maximum
        return maximum

    def settle(self, reservation: int, actual: int) -> None:
        with self._lock:
            self._reserved -= reservation
            self.actual += int(actual)
            if self.actual > self.limit and not self.limit_exceeded:
                self.limit_exceeded = True
                self.first_exceeded_at = self.actual

    @property
    def overage(self) -> int:
        with self._lock:
            return max(0, self.actual - self.limit)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "configured_limit": self.limit,
                "actual": self.actual,
                "overage": max(0, self.actual - self.limit),
                "limit_exceeded": self.limit_exceeded,
                "first_exceeded_at": self.first_exceeded_at,
                "reserved_retry_capacity": self._reserved,
            }


def run_catch_sample(
    sample: DatasetSample,
    *,
    run_id: str,
    split_name: str,
    experiment,
    protocol,
    endpoint,
    network_budget: NetworkAttemptBudget,
    phase_name: str,
    frozen_decoding: dict[str, int] | None = None,
    run_direct_judge: bool = True,
    precomputed_stage_rows: tuple[dict[str, Any], ...] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    if protocol.protocol_version == "catch_v3":
        return run_catch_icv_sample(
            sample,
            run_id=run_id,
            split_name=split_name,
            experiment=experiment,
            protocol=protocol,
            endpoint=endpoint,
            network_budget=network_budget,
            phase_name=phase_name,
            run_direct_judge=run_direct_judge,
            precomputed_stage_rows=precomputed_stage_rows,
        )
    if precomputed_stage_rows is not None:
        raise ValueError("Precomputed Stage-A rows are supported only by the frozen CATCH-v3 protocol.")
    stage_rows = [
        _answer_turn(
            sample,
            run_id=run_id,
            split_name=split_name,
            endpoint=endpoint,
            network_budget=network_budget,
            method_name="catch_stage_a_shared",
            role="stage_a_solver",
            agent_id=index,
            seed=42_000 + index,
            max_tokens=protocol.solver_max_tokens,
        )
        for index in range(1, protocol.stage_candidates + 1)
    ]
    stage = build_stage_decision(stage_rows, seed=experiment.global_seed, sample_id=sample.sample_id)
    physical_rows = list(stage_rows)
    resample_rows: list[dict[str, Any]] = []
    designer_row: dict[str, Any] | None = None
    witness_rows: list[dict[str, Any]] = []
    judge_rows: list[dict[str, Any]] = []
    catch_variants: list[tuple[int, int, Any, list[dict[str, Any]], dict[str, Any]]] = []
    bank_validation = None
    hypothesis_to_key: dict[str, str] = {}
    judge_selections: list[str | None] = []

    if stage.triggered:
        resample_rows = [
            _answer_turn(
                sample,
                run_id=run_id,
                split_name=split_name,
                endpoint=endpoint,
                network_budget=network_budget,
                method_name="catch_adaptive_resample_shared",
                role="independent_resample",
                agent_id=index,
                seed=45_000 + index,
                max_tokens=protocol.solver_max_tokens,
            )
            for index in range(1, protocol.resample_candidates + 1)
        ]
        hypothesis_to_key = build_hypothesis_labels(
            stage,
            seed=experiment.global_seed,
            sample_id=sample.sample_id,
        )
        designer_row, designer_payload = _json_turn(
            sample,
            run_id=run_id,
            split_name=split_name,
            endpoint=endpoint,
            network_budget=network_budget,
            method_name="catch_test_designer_shared",
            role="test_designer",
            agent_id=1,
            seed=43_000,
            max_tokens=protocol.role_max_tokens,
            messages=build_designer_messages(
                sample,
                stage=stage,
                hypothesis_to_key=hypothesis_to_key,
            ),
        )
        designer_row["hypothesis_to_answer_class_key"] = hypothesis_to_key
        bank_validation = validate_test_bank(
            designer_payload,
            stage=stage,
            hypothesis_to_key=hypothesis_to_key,
            max_tests=protocol.max_proposed_tests,
        )
        designer_row["test_bank_protocol_error"] = bank_validation.protocol_error
        designer_row["dropped_tests"] = list(bank_validation.dropped)
        designer_row["validated_test_bank"] = [test_to_dict(test) for test in bank_validation.tests]
        designer_row["evidence_quote_count"] = bank_validation.evidence_quote_count
        designer_row["aligned_evidence_quote_count"] = bank_validation.aligned_evidence_quote_count
        designer_row["leakage_count"] = bank_validation.leakage_count
        if bank_validation.protocol_error is not None:
            designer_row["protocol_parse_status"] = "failed"
            designer_row["protocol_parse_error"] = bank_validation.protocol_error

        d_values = (
            [int(frozen_decoding["d_min"])]
            if frozen_decoding is not None
            else list(protocol.d_min_grid)
        )
        margin_values = (
            [int(frozen_decoding["margin"])]
            if frozen_decoding is not None
            else list(protocol.margin_grid)
        )
        for d_min in d_values:
            selection = select_tests(
                bank_validation.tests,
                stage=stage,
                d_min=d_min,
                max_selected=protocol.max_selected_tests,
            )
            panel_vectors: list[dict[str, str] | None] = []
            current_witness_rows: list[dict[str, Any]] = []
            code_eligible = any(distance >= d_min for distance in selection.pair_distances.values())
            if code_eligible:
                for panel_index in range(1, protocol.witness_count + 1):
                    packet = build_witness_packet(
                        selection.tests,
                        seed=experiment.global_seed,
                        sample_id=f"{sample.sample_id}:d{d_min}",
                        panel_index=panel_index,
                    )
                    witness_row, witness_payload = _json_turn(
                        sample,
                        run_id=run_id,
                        split_name=split_name,
                        endpoint=endpoint,
                        network_budget=network_budget,
                        method_name=f"catch_witness_d{d_min}",
                        role="blinded_witness",
                        agent_id=panel_index,
                        seed=44_000 + d_min * 10 + panel_index,
                        max_tokens=protocol.role_max_tokens,
                        messages=build_witness_messages(sample, packet=packet),
                    )
                    parsed_witness = parse_witness_answers_detailed(witness_payload, packet=packet)
                    vector = parsed_witness.vector
                    if not parsed_witness.top_level_valid:
                        witness_row["protocol_parse_status"] = "failed"
                        witness_row["protocol_parse_error"] = "witness_top_level_schema_failure"
                    witness_row["selection_d_min"] = d_min
                    witness_row["witness_packet"] = {
                        "panel_index": packet.panel_index,
                        "tests": list(packet.tests),
                        "public_test_to_internal": packet.public_test_to_internal,
                        "public_outcome_to_internal": packet.public_outcome_to_internal,
                    }
                    witness_row["witness_parse_diagnostics"] = {
                        "top_level_valid": parsed_witness.top_level_valid,
                        "expected_coordinate_count": parsed_witness.expected_coordinate_count,
                        "valid_coordinate_count": parsed_witness.valid_coordinate_count,
                        "erased_rows": list(parsed_witness.erased_rows),
                    }
                    witness_row["witness_vector"] = vector
                    current_witness_rows.append(witness_row)
                    witness_rows.append(witness_row)
                    panel_vectors.append(vector)
            for margin in margin_values:
                decision = (
                    decode_witnesses(
                        stage,
                        selection.tests,
                        panel_vectors,
                        d_min=d_min,
                        margin=margin,
                    )
                    if code_eligible
                    else DecodeDecision(
                        stage.anchor_answer,
                        stage.anchor_key,
                        False,
                        "insufficient_code_distance",
                        (),
                        (),
                    )
                )
                diagnostic = {
                    "d_min": d_min,
                    "margin": margin,
                    "selected_test_ids": [test.test_id for test in selection.tests],
                    "pair_distances": selection.pair_distances,
                    "selection_objective": list(selection.objective),
                    "selection_tie_break_sha256": selection.tie_break_sha256,
                    "code_eligible": code_eligible,
                    "witness_vectors": panel_vectors,
                    "decision": {
                        "answer": decision.answer,
                        "answer_key": decision.answer_key,
                        "override_accepted": decision.override_accepted,
                        "resolver": decision.resolver,
                        "passing_challengers": list(decision.passing_challengers),
                        "panel_diagnostics": list(decision.panel_diagnostics),
                    },
                }
                catch_variants.append((d_min, margin, decision, current_witness_rows, diagnostic))

        if run_direct_judge:
            for judge_index in range(1, protocol.direct_judge_count + 1):
                judge_labels = build_hypothesis_labels(
                    stage,
                    seed=experiment.global_seed,
                    sample_id=f"{sample.sample_id}:judge:{judge_index}",
                )
                judge_row, judge_payload = _json_turn(
                    sample,
                    run_id=run_id,
                    split_name=split_name,
                    endpoint=endpoint,
                    network_budget=network_budget,
                    method_name="direct_judge_3",
                    role="direct_judge",
                    agent_id=judge_index,
                    seed=46_000 + judge_index,
                    max_tokens=protocol.role_max_tokens,
                    messages=build_direct_judge_messages(
                        sample,
                        stage=stage,
                        hypothesis_to_key=judge_labels,
                    ),
                )
                selected_id = str(judge_payload.get("selected_id") or "") if isinstance(judge_payload, dict) else ""
                selected_key = judge_labels.get(selected_id)
                if selected_key is None:
                    judge_row["protocol_parse_status"] = "failed"
                    judge_row["protocol_parse_error"] = "unknown_or_missing_candidate_id"
                judge_row["hypothesis_to_answer_class_key"] = judge_labels
                judge_row["selected_answer_class_key"] = selected_key
                judge_rows.append(judge_row)
                judge_selections.append(selected_key)
        physical_rows.extend([*resample_rows, designer_row, *witness_rows, *judge_rows])

    adaptive = build_stage_decision(
        [*stage_rows, *resample_rows],
        seed=experiment.global_seed,
        sample_id=sample.sample_id,
    )
    adaptive_answer = adaptive.anchor_answer or stage.anchor_answer
    candidate_oracle = any(_score(sample, candidate.answer) == 1.0 for candidate in stage.candidates)
    predictions = [
        _prediction(
            sample,
            run_id,
            split_name,
            "sc_5",
            stage.anchor_answer,
            stage.anchor_answer,
            stage,
            stage_rows,
            [],
            False,
            "stage_a_plurality",
            candidate_oracle,
        ),
        _prediction(
            sample,
            run_id,
            split_name,
            "adaptive_sc_8",
            adaptive_answer,
            stage.anchor_answer,
            stage,
            [*stage_rows, *resample_rows],
            resample_rows,
            adaptive_answer != stage.anchor_answer,
            "adaptive_answer_class_plurality" if stage.triggered else "no_answer_class_disagreement",
            candidate_oracle,
        ),
    ]
    catch_diagnostics: list[dict[str, Any]] = []
    for d_min, margin, decision, variant_witness_rows, diagnostic in catch_variants:
        method_name = "catch" if frozen_decoding is not None else f"catch_d{d_min}_m{margin}"
        predictions.append(
            _prediction(
                sample,
                run_id,
                split_name,
                method_name,
                decision.answer,
                stage.anchor_answer,
                stage,
                [*stage_rows, *([designer_row] if designer_row is not None else []), *variant_witness_rows],
                [*([designer_row] if designer_row is not None else []), *variant_witness_rows],
                decision.override_accepted,
                decision.resolver,
                candidate_oracle,
                extra={"d_min": d_min, "margin": margin},
            )
        )
        catch_diagnostics.append(diagnostic)
    if not stage.triggered:
        predictions.append(
            _prediction(
                sample,
                run_id,
                split_name,
                "catch" if frozen_decoding is not None else "catch_d2_m1",
                stage.anchor_answer,
                stage.anchor_answer,
                stage,
                stage_rows,
                [],
                False,
                "no_answer_class_disagreement",
                candidate_oracle,
                extra={
                    "d_min": int((frozen_decoding or {}).get("d_min", 2)),
                    "margin": int((frozen_decoding or {}).get("margin", 1)),
                },
            )
        )
        if frozen_decoding is None:
            for d_min in protocol.d_min_grid:
                for margin in protocol.margin_grid:
                    if (d_min, margin) == (2, 1):
                        continue
                    clone = dict(predictions[-1])
                    clone.update({"method_name": f"catch_d{d_min}_m{margin}", "d_min": d_min, "margin": margin})
                    predictions.append(clone)
    if run_direct_judge:
        judge_answer, judge_override, judge_resolver = (
            decide_direct_judges(stage, judge_selections)
            if stage.triggered
            else (stage.anchor_answer, False, "no_answer_class_disagreement")
        )
        predictions.append(
            _prediction(
                sample,
                run_id,
                split_name,
                "direct_judge_3",
                judge_answer,
                stage.anchor_answer,
                stage,
                [*stage_rows, *judge_rows],
                judge_rows,
                judge_override,
                judge_resolver,
                candidate_oracle,
            )
        )
    router = {
        "run_id": run_id,
        "dataset": sample.dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "task": sample.metadata.get("task"),
        "phase_name": phase_name,
        "triggered": stage.triggered,
        "valid_stage_answers": stage.valid_count,
        "anchor_answer": stage.anchor_answer,
        "anchor_key": stage.anchor_key,
        "answer_class_vote_counts": stage.vote_counts,
        "candidate_count": len(stage.candidates),
        "candidate_oracle_correct": candidate_oracle,
        "hypothesis_to_answer_class_key": hypothesis_to_key,
        "test_bank_protocol_error": bank_validation.protocol_error if bank_validation is not None else None,
        "valid_test_count": len(bank_validation.tests) if bank_validation is not None else 0,
        "dropped_tests": list(bank_validation.dropped) if bank_validation is not None else [],
        "catch_variants": catch_diagnostics,
        "judge_selections": judge_selections,
    }
    return [row for row in physical_rows if row is not None], router, predictions


def run_catch_icv_sample(
    sample: DatasetSample,
    *,
    run_id: str,
    split_name: str,
    experiment,
    protocol,
    endpoint,
    network_budget: NetworkAttemptBudget,
    phase_name: str,
    run_direct_judge: bool = True,
    precomputed_stage_rows: tuple[dict[str, Any], ...] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Execute the frozen CATCH-v3 indexed contrast protocol for one sample."""

    if precomputed_stage_rows is None:
        stage_rows = [
            _answer_turn(
                sample,
                run_id=run_id,
                split_name=split_name,
                endpoint=endpoint,
                network_budget=network_budget,
                method_name="catch_stage_a_shared",
                role="stage_a_solver",
                agent_id=index,
                seed=42_000 + index,
                max_tokens=protocol.solver_max_tokens,
            )
            for index in range(1, protocol.stage_candidates + 1)
        ]
    else:
        stage_rows = list(precomputed_stage_rows)
        if (
            len(stage_rows) != protocol.stage_candidates
            or any(row.get("role") != "stage_a_solver" for row in stage_rows)
            or any(str(row.get("sample_id") or "") != sample.sample_id for row in stage_rows)
        ):
            raise ValueError("Precomputed Stage-A rows do not match the current CATCH-v3 sample contract.")
    stage = build_stage_decision(stage_rows, seed=experiment.global_seed, sample_id=sample.sample_id)
    physical_rows: list[dict[str, Any]] = list(stage_rows)
    resample_rows: list[dict[str, Any]] = []
    selector_row: dict[str, Any] | None = None
    witness_rows: list[dict[str, Any]] = []
    direct_rows: list[dict[str, Any]] = []
    pair_judge_rows: list[dict[str, Any]] = []
    direct_selections: list[str | None] = []
    pair_selections: list[str | None] = []
    validation = ContrastValidation((), (), None, 0, ())
    panels: list[IcvWitnessParseResult] = []
    pairs = build_target_pairs(
        stage,
        seed=experiment.global_seed,
        sample_id=sample.sample_id,
    )
    evidence = segment_stage_evidence(sample, stage)

    decision = DecodeDecision(
        stage.anchor_answer,
        stage.anchor_key,
        False,
        "no_answer_class_disagreement",
        (),
        (),
    )
    if stage.triggered:
        resample_rows = [
            _answer_turn(
                sample,
                run_id=run_id,
                split_name=split_name,
                endpoint=endpoint,
                network_budget=network_budget,
                method_name="catch_adaptive_resample_shared",
                role="independent_resample",
                agent_id=index,
                seed=45_000 + index,
                max_tokens=protocol.solver_max_tokens,
            )
            for index in range(1, protocol.resample_candidates + 1)
        ]
        selector_row, selector_payload = _json_turn(
            sample,
            run_id=run_id,
            split_name=split_name,
            endpoint=endpoint,
            network_budget=network_budget,
            method_name="catch_icv_selector",
            role="icv_selector",
            agent_id=1,
            seed=43_000,
            max_tokens=protocol.role_max_tokens,
            messages=build_icv_selector_messages(sample, pairs=pairs, evidence=evidence),
        )
        validation = validate_contrast_selector(
            selector_payload,
            pairs=pairs,
            evidence=evidence,
            max_per_pair=protocol.coordinates_per_pair,
            max_total=protocol.max_selected_contrasts,
        )
        selector_row.update(
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
            selector_row["protocol_parse_status"] = "failed"
            selector_row["protocol_parse_error"] = validation.protocol_error

        if validation.eligible_challengers:
            eligible_coordinates = tuple(
                item
                for item in validation.coordinates
                if item.challenger_key in validation.eligible_challengers
            )
            for panel_index in range(1, protocol.witness_count + 1):
                packet = build_icv_witness_packet(
                    eligible_coordinates,
                    seed=experiment.global_seed,
                    sample_id=sample.sample_id,
                    panel_index=panel_index,
                )
                witness_row, witness_payload = _json_turn(
                    sample,
                    run_id=run_id,
                    split_name=split_name,
                    endpoint=endpoint,
                    network_budget=network_budget,
                    method_name="catch_icv_witness",
                    role="icv_witness",
                    agent_id=panel_index,
                    seed=44_000 + panel_index,
                    max_tokens=protocol.role_max_tokens,
                    messages=build_icv_witness_messages(sample, packet=packet),
                )
                parsed = parse_icv_witness(witness_payload, packet=packet)
                if not parsed.top_level_valid:
                    witness_row["protocol_parse_status"] = "failed"
                    witness_row["protocol_parse_error"] = "witness_top_level_schema_failure"
                witness_row.update(
                    {
                        "witness_packet": {
                            "panel_index": packet.panel_index,
                            "contrasts": list(packet.contrasts),
                            "public_to_internal": packet.public_to_internal,
                            "public_left_to_candidate": packet.public_left_to_candidate,
                            "public_right_to_candidate": packet.public_right_to_candidate,
                        },
                        "witness_observations": parsed.observations,
                        "witness_parse_diagnostics": {
                            "top_level_valid": parsed.top_level_valid,
                            "expected_coordinate_count": parsed.expected_coordinate_count,
                            "valid_coordinate_count": parsed.valid_coordinate_count,
                            "decisive_coordinate_count": parsed.decisive_coordinate_count,
                            "erased_rows": list(parsed.erased_rows),
                        },
                    }
                )
                witness_rows.append(witness_row)
                panels.append(parsed)
            decision = decode_icv(stage, eligible_coordinates, tuple(panels))
        else:
            decision = DecodeDecision(
                stage.anchor_answer,
                stage.anchor_key,
                False,
                "insufficient_indexed_contrast",
                (),
                (),
            )

        if run_direct_judge:
            for judge_index in range(1, protocol.direct_judge_count + 1):
                labels = build_hypothesis_labels(
                    stage,
                    seed=experiment.global_seed,
                    sample_id=f"{sample.sample_id}:direct:{judge_index}",
                )
                row, payload = _json_turn(
                    sample,
                    run_id=run_id,
                    split_name=split_name,
                    endpoint=endpoint,
                    network_budget=network_budget,
                    method_name="direct_judge_3",
                    role="direct_judge",
                    agent_id=judge_index,
                    seed=46_000 + judge_index,
                    max_tokens=protocol.role_max_tokens,
                    messages=build_direct_judge_messages(sample, stage=stage, hypothesis_to_key=labels),
                )
                selected_id = str(payload.get("selected_id") or "") if isinstance(payload, dict) else ""
                selected_key = labels.get(selected_id)
                if selected_key is None:
                    row["protocol_parse_status"] = "failed"
                    row["protocol_parse_error"] = "unknown_or_missing_candidate_id"
                row["hypothesis_to_answer_class_key"] = labels
                row["selected_answer_class_key"] = selected_key
                direct_rows.append(row)
                direct_selections.append(selected_key)

            target_keys = [stage.anchor_key, *(pair.challenger_key for pair in pairs)]
            for judge_index in range(1, protocol.pair_judge_count + 1):
                labels = _target_judge_labels(
                    target_keys,
                    seed=experiment.global_seed,
                    sample_id=f"{sample.sample_id}:pair-judge:{judge_index}",
                )
                row, payload = _json_turn(
                    sample,
                    run_id=run_id,
                    split_name=split_name,
                    endpoint=endpoint,
                    network_budget=network_budget,
                    method_name="pair_judge_3",
                    role="pair_judge",
                    agent_id=judge_index,
                    seed=47_000 + judge_index,
                    max_tokens=protocol.role_max_tokens,
                    messages=build_pair_judge_messages(sample, stage=stage, public_to_key=labels),
                )
                selected_id = str(payload.get("selected_id") or "") if isinstance(payload, dict) else ""
                selected_key = labels.get(selected_id)
                if selected_key is None:
                    row["protocol_parse_status"] = "failed"
                    row["protocol_parse_error"] = "unknown_or_missing_target_candidate_id"
                row["hypothesis_to_answer_class_key"] = labels
                row["selected_answer_class_key"] = selected_key
                pair_judge_rows.append(row)
                pair_selections.append(selected_key)

        physical_rows.extend(
            [
                *resample_rows,
                *([selector_row] if selector_row is not None else []),
                *witness_rows,
                *direct_rows,
                *pair_judge_rows,
            ]
        )

    adaptive = build_stage_decision(
        [*stage_rows, *resample_rows],
        seed=experiment.global_seed,
        sample_id=sample.sample_id,
    )
    adaptive_answer = adaptive.anchor_answer or stage.anchor_answer
    candidate_oracle = any(_score(sample, candidate.answer) == 1.0 for candidate in stage.candidates)
    target_keys = {stage.anchor_key, *(pair.challenger_key for pair in pairs)}
    target_oracle = any(
        candidate.key in target_keys and _score(sample, candidate.answer) == 1.0
        for candidate in stage.candidates
    )
    gold_candidate_key = next(
        (candidate.key for candidate in stage.candidates if _score(sample, candidate.answer) == 1.0),
        None,
    )
    common_extra = {
        "target_oracle_correct": target_oracle,
        "gold_candidate_key": gold_candidate_key,
        "target_candidate_count": len(target_keys),
        "protocol_version": "catch_v3",
    }
    predictions = [
        _prediction(
            sample,
            run_id,
            split_name,
            "sc_5",
            stage.anchor_answer,
            stage.anchor_answer,
            stage,
            stage_rows,
            [],
            False,
            "stage_a_plurality",
            candidate_oracle,
            extra=common_extra,
        ),
        _prediction(
            sample,
            run_id,
            split_name,
            "adaptive_sc_8",
            adaptive_answer,
            stage.anchor_answer,
            stage,
            [*stage_rows, *resample_rows],
            resample_rows,
            adaptive_answer != stage.anchor_answer,
            "adaptive_answer_class_plurality" if stage.triggered else "no_answer_class_disagreement",
            candidate_oracle,
            extra=common_extra,
        ),
        _prediction(
            sample,
            run_id,
            split_name,
            "catch",
            decision.answer,
            stage.anchor_answer,
            stage,
            [
                *stage_rows,
                *([selector_row] if selector_row is not None else []),
                *witness_rows,
            ],
            [*([selector_row] if selector_row is not None else []), *witness_rows],
            decision.override_accepted,
            decision.resolver,
            candidate_oracle,
            extra={
                **common_extra,
                "eligible_challenger_count": len(validation.eligible_challengers),
                "validated_contrast_count": len(validation.coordinates),
            },
        ),
    ]
    if run_direct_judge:
        direct_answer, direct_override, direct_resolver = (
            decide_direct_judges(stage, direct_selections)
            if stage.triggered
            else (stage.anchor_answer, False, "no_answer_class_disagreement")
        )
        pair_answer, pair_override, pair_resolver = (
            decide_direct_judges(stage, pair_selections)
            if stage.triggered
            else (stage.anchor_answer, False, "no_answer_class_disagreement")
        )
        predictions.extend(
            [
                _prediction(
                    sample,
                    run_id,
                    split_name,
                    "direct_judge_3",
                    direct_answer,
                    stage.anchor_answer,
                    stage,
                    [*stage_rows, *direct_rows],
                    direct_rows,
                    direct_override,
                    direct_resolver,
                    candidate_oracle,
                    extra=common_extra,
                ),
                _prediction(
                    sample,
                    run_id,
                    split_name,
                    "pair_judge_3",
                    pair_answer,
                    stage.anchor_answer,
                    stage,
                    [*stage_rows, *pair_judge_rows],
                    pair_judge_rows,
                    pair_override,
                    pair_resolver,
                    candidate_oracle,
                    extra=common_extra,
                ),
            ]
        )

    router = {
        "run_id": run_id,
        "dataset": sample.dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "task": sample.metadata.get("task"),
        "audit_source_question": question_without_answer_contract(sample),
        "phase_name": phase_name,
        "protocol_version": "catch_v3",
        "triggered": stage.triggered,
        "valid_stage_answers": stage.valid_count,
        "anchor_answer": stage.anchor_answer,
        "anchor_key": stage.anchor_key,
        "answer_class_vote_counts": stage.vote_counts,
        "candidate_count": len(stage.candidates),
        "candidate_oracle_correct": candidate_oracle,
        "target_oracle_correct": target_oracle,
        "gold_candidate_key": gold_candidate_key,
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
        "selector_protocol_error": validation.protocol_error,
        "dropped_contrasts": list(validation.dropped),
        "validated_contrasts": [coordinate_to_dict(item) for item in validation.coordinates],
        "eligible_challengers": list(validation.eligible_challengers),
        "witness_panels": [
            {
                "top_level_valid": panel.top_level_valid,
                "observations": panel.observations,
                "valid_coordinate_count": panel.valid_coordinate_count,
                "decisive_coordinate_count": panel.decisive_coordinate_count,
            }
            for panel in panels
        ],
        "decision": {
            "answer_key": decision.answer_key,
            "override_accepted": decision.override_accepted,
            "resolver": decision.resolver,
            "passing_challengers": list(decision.passing_challengers),
            "panel_diagnostics": list(decision.panel_diagnostics),
        },
        "direct_judge_selections": direct_selections,
        "pair_judge_selections": pair_selections,
    }
    return physical_rows, router, predictions


def _target_judge_labels(keys: list[str], *, seed: int, sample_id: str) -> dict[str, str]:
    unique = list(dict.fromkeys(key for key in keys if key))
    random_seed = int(_sha256(f"{seed}\0{sample_id}\0pair-judge")[:16], 16)
    import random

    random.Random(random_seed).shuffle(unique)
    return {f"J{index}": key for index, key in enumerate(unique)}


def _answer_turn(
    sample: DatasetSample,
    *,
    run_id: str,
    split_name: str,
    endpoint,
    network_budget: NetworkAttemptBudget,
    method_name: str,
    role: str,
    agent_id: int,
    seed: int,
    max_tokens: int,
) -> dict[str, Any]:
    _raise_if_cancelled(endpoint)
    reservation = network_budget.reserve()
    cache = _cache_for_role(endpoint, role)
    try:
        result = execute_output_protocol_turn(
            backbone=endpoint.backbone,
            provider=endpoint.provider,
            cache=cache,
            throttle=endpoint.throttle,
            sample=sample,
            messages=build_cot_messages(sample, agent_id, "single_agent_free_text_v1"),
            temperature=0.7,
            top_p=1.0,
            seed=seed,
            dataset=sample.dataset,
            role=role,
            output_protocol=FREE_TEXT_ANSWER_PROTOCOL_V1,
            max_tokens=max_tokens,
        )
        network_budget.settle(reservation, result.network_request_count)
    except BaseException:
        network_budget.settle(reservation, 0)
        raise
    raw_answer = str(result.validated_output.get("final_answer") or "")
    canonical = canonicalize_answer(sample, raw_answer) if result.output_status == "ok" else None
    parsed_valid = result.validated_output.get("canonical_valid")
    if parsed_valid is False:
        canonical = None
    canonical_key = (
        str(result.validated_output.get("canonical_key") or "")
        if parsed_valid is True
        else canonical.key if canonical is not None and canonical.valid else ""
    )
    invalid_reason = (
        str(result.validated_output.get("canonical_invalid_reason") or "invalid_sample_answer_output")
        if parsed_valid is False
        else canonical.invalid_reason if canonical is not None else "request_or_protocol_failure"
    )
    row = _turn_base(
        run_id,
        sample,
        split_name,
        method_name,
        role,
        agent_id,
        seed,
        result.payload,
        result.response_payload,
        result.cache_key,
        result.cache_hit,
        result.request_error,
        result.raw_finish_reason,
        result.usage,
        "ok" if canonical_key else "failed",
        dict(result.validated_output),
        canonical_key,
        invalid_reason,
        request_count=result.request_count,
        cache_request_count=result.cache_request_count,
        network_request_count=result.network_request_count,
    )
    _annotate_cache_audit(row, endpoint=endpoint, cache=cache, cache_key=result.cache_key)
    return row


def _json_turn(
    sample: DatasetSample,
    *,
    run_id: str,
    split_name: str,
    endpoint,
    network_budget: NetworkAttemptBudget,
    method_name: str,
    role: str,
    agent_id: int,
    seed: int,
    max_tokens: int,
    messages: list[dict[str, str]],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    _raise_if_cancelled(endpoint)
    reservation = network_budget.reserve()
    cache = _cache_for_role(endpoint, role)
    try:
        request = execute_cached_request(
            backbone=endpoint.backbone,
            provider=endpoint.provider,
            cache=cache,
            throttle=endpoint.throttle,
            messages=messages,
            temperature=0.7,
            top_p=1.0,
            seed=seed,
            use_response_format=True,
            max_tokens=max_tokens,
        )
        attempts = 0 if request.cache_hit else max(1, int(request.response_payload.get("network_attempt_count") or 1))
        network_budget.settle(reservation, attempts)
    except BaseException:
        network_budget.settle(reservation, 0)
        raise
    parsed: dict[str, Any] | None = None
    parse_error = request.request_error
    if not parse_error:
        try:
            candidate = json.loads(str(request.response_payload.get("assistant_text") or ""))
            if not isinstance(candidate, dict):
                raise ValueError("JSON output must be an object")
            parsed = candidate
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            parse_error = str(exc)
    row = _turn_base(
        run_id,
        sample,
        split_name,
        method_name,
        role,
        agent_id,
        seed,
        request.payload,
        request.response_payload,
        request.cache_key,
        request.cache_hit,
        request.request_error,
        request.response_payload.get("finish_reason"),
        request.usage,
        "ok" if parsed is not None else "failed",
        parsed or {},
        "",
        parse_error,
        request_count=max(1, attempts),
        cache_request_count=int(request.cache_hit),
        network_request_count=attempts,
    )
    _annotate_cache_audit(row, endpoint=endpoint, cache=cache, cache_key=request.cache_key)
    return row, parsed


def _turn_base(
    run_id,
    sample,
    split_name,
    method_name,
    role,
    agent_id,
    seed,
    payload,
    response,
    cache_key,
    cache_hit,
    request_error,
    finish_reason,
    usage,
    parse_status,
    validated,
    answer_key,
    invalid_reason,
    *,
    request_count,
    cache_request_count,
    network_request_count,
):
    reported_usage = dict(response.get("usage_reported") or {})
    details = reported_usage.get("completion_tokens_details") or {}
    return {
        "run_id": run_id,
        "dataset": sample.dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "task": sample.metadata.get("task"),
        "method_name": method_name,
        "role": role,
        "agent_id": agent_id,
        "request_seed": seed,
        "payload": payload,
        "cache_key": cache_key,
        "cache_namespace": None,
        "request_source": "role_cache_pending",
        "prompt_hash": _sha256(json.dumps(payload.get("messages") or [], ensure_ascii=False, sort_keys=True)),
        "cache_hit": cache_hit,
        "request_error": request_error,
        "request_status": "request_fail" if request_error else "ok",
        "raw_finish_reason": finish_reason,
        "provider_request_id": response.get("provider_request_id"),
        "response_id": response.get("response_id"),
        "attempt_timeline": [] if cache_hit else list(response.get("attempt_timeline") or []),
        "cached_response_origin_attempt_timeline": (
            list(response.get("attempt_timeline") or []) if cache_hit else []
        ),
        "cache_lookup_timeline": dict(response.get("cache_lookup_timeline") or {}),
        "network_attempt_count": network_request_count,
        "network_request_count": network_request_count,
        "request_count": request_count,
        "cache_request_count": cache_request_count,
        "prompt_tokens": float(usage.get("prompt_tokens") or 0),
        "completion_tokens": float(usage.get("completion_tokens") or 0),
        "total_tokens": float(usage.get("total_tokens") or 0),
        "reasoning_tokens": reported_usage.get("reasoning_tokens", details.get("reasoning_tokens")),
        "usage_source": response.get("usage_source"),
        "usage_reported": reported_usage,
        "actual_prompt_tokens": reported_usage.get("prompt_tokens"),
        "actual_completion_tokens": reported_usage.get("completion_tokens"),
        "actual_total_tokens": reported_usage.get("total_tokens"),
        "latency_ms": float(response.get("latency_ms") or 0),
        "assistant_text": str(response.get("assistant_text") or ""),
        "provider_reasoning_text": str(response.get("provider_reasoning_text") or ""),
        "validated_output": validated,
        "protocol_parse_status": parse_status,
        "protocol_parse_error": invalid_reason,
        "prediction": answer_key,
        "normalized_answer": answer_key,
        "answer_class_key": answer_key,
        "canonicalization_invalid_reason": invalid_reason,
    }


def _prediction(
    sample,
    run_id,
    split_name,
    method_name,
    prediction,
    initial,
    stage,
    logical_rows,
    intervention_rows,
    override,
    resolver,
    candidate_oracle,
    *,
    extra: dict[str, Any] | None = None,
):
    score = _score(sample, prediction)
    initial_score = _score(sample, initial)
    intervention_call_budget = 3 if stage.triggered and method_name != "sc_5" else 0
    actual_intervention_calls = len(intervention_rows)
    logical_calls = len(logical_rows)
    payload = {
        "run_id": run_id,
        "dataset": sample.dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "task": sample.metadata.get("task"),
        "method_name": method_name,
        "prediction": prediction,
        "gold": sample.reference_answer,
        "score": score,
        "initial_vote_prediction": initial,
        "initial_vote_score": initial_score,
        "initial_answer_class_key": stage.anchor_key,
        "initial_vote_counts": stage.vote_counts,
        "candidate_oracle_correct": candidate_oracle,
        "triggered": stage.triggered,
        "override_accepted": override,
        "vote_flipped": override,
        "corrected_by_debate": override and initial_score < 1 and score == 1,
        "harmed_by_debate": override and initial_score == 1 and score < 1,
        "resolver": resolver,
        "calls_per_question": logical_calls,
        "logical_calls_per_question": logical_calls,
        "actual_executed_calls_per_question": len(logical_rows),
        "total_tokens_per_question": sum(float(row.get("actual_total_tokens") or row.get("total_tokens") or 0) for row in logical_rows),
        "completion_tokens_per_question": sum(float(row.get("actual_completion_tokens") or row.get("completion_tokens") or 0) for row in logical_rows),
        "network_attempts_per_question": sum(int(row.get("network_attempt_count") or 0) for row in logical_rows),
        "intervention_call_budget_per_question": intervention_call_budget,
        "intervention_calls_per_question": actual_intervention_calls,
        "actual_intervention_calls_per_question": actual_intervention_calls,
    }
    payload.update(extra or {})
    return payload


def _score(sample: DatasetSample, answer: str) -> float:
    return score_prediction(sample.dataset, answer, sample.reference_answer, sample=sample) if answer else 0.0


def _cache_for_role(endpoint, role: str):
    if hasattr(endpoint, "cache_for_role"):
        return endpoint.cache_for_role(role)
    return endpoint.cache


def _raise_if_cancelled(endpoint) -> None:
    event = getattr(endpoint, "stop_event", None)
    if event is not None and event.is_set():
        raise CatchRunCancelled("CATCH run cancelled after a sibling sample failed")


def _annotate_cache_audit(row: dict[str, Any], *, endpoint, cache, cache_key: str) -> None:
    source = cache.source_for(cache_key) if hasattr(cache, "source_for") else endpoint.cache_namespace
    active_namespace = str(getattr(endpoint, "cache_namespace", source))
    row["cache_namespace"] = active_namespace if source == "network" else source
    row["cache_write_namespace"] = active_namespace
    row["cache_lookup_namespaces"] = list(
        getattr(endpoint, "cache_lookup_namespaces_for_role", lambda _role: (active_namespace,))(row["role"])
    )
    if row.get("cache_hit"):
        row["request_source"] = "predecessor_cache" if source != active_namespace else "active_cache"
    else:
        row["request_source"] = "network"


def _sha256(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()
