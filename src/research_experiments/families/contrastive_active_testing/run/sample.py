"""共享 Stage-A、CATCH、adaptive-SC8 与 Judge-3 的逐样本执行。"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import replace
from typing import Any

from research_experiments.core.controls.control_prompts import (
    FREE_TEXT_V1_PROMPT_VERSION,
    build_cot_messages,
)
from research_experiments.core.data.datasets import DatasetSample, question_without_answer_contract
from research_experiments.core.data.evaluation import (
    canonicalize_answer,
    evaluate_seqbench_prediction,
    score_prediction,
    validate_seqbench_plan,
)
from research_experiments.core.execution.runner_common import execute_cached_request
from research_experiments.families.contrastive_active_testing.algorithms import (
    DecodeDecision,
    StageDecision,
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
from research_experiments.families.contrastive_active_testing.cert_prompts import (
    build_certificate_designer_messages,
    build_certificate_verifier_messages,
)
from research_experiments.families.contrastive_active_testing.cert_prompts_v2 import (
    build_certificate_designer_messages_v2,
    build_certificate_verifier_messages_v2,
)
from research_experiments.families.contrastive_active_testing.certificates import (
    CertificateBankValidation,
    CertificateVerifierParseResult,
    build_certificate_verifier_packet,
    build_claim_graphs,
    build_task_contract,
    certificate_test_to_dict,
    certificate_to_dict,
    claim_graph_to_dict,
    decode_certificates,
    parse_certificate_verifier,
    task_contract_to_dict,
    validate_certificate_bank,
    verifier_result_to_dict,
)
from research_experiments.families.contrastive_active_testing.certificates_v2 import (
    CertificateBankValidationV2,
    CertificateVerifierParseResultV2,
    adapter_result_to_dict,
    build_all_candidate_pairs_v2,
    build_candidate_answer_nodes,
    build_certificate_verifier_packet_v2,
    build_source_span_graph,
    build_task_contract_v2,
    candidate_answer_node_to_dict,
    certificate_test_v2_to_dict,
    certificate_v2_to_dict,
    decode_certificates_v2,
    pair_v2_to_dict,
    parse_certificate_verifier_v2,
    run_deterministic_adapters_v2,
    source_span_graph_to_dict,
    task_contract_v2_to_dict,
    validate_certificate_bank_v2,
    verifier_result_v2_to_dict,
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
from research_experiments.families.contrastive_active_testing.kernel import (
    KernelDecision,
    answer_obligation_graph_to_dict,
    bind_verifier_capabilities,
    build_proof_results,
    build_task_semantics,
    compile_answer_obligation_graph,
    compile_local_certificate_bank,
    compile_typed_obligations,
    decide_with_proof_kernel,
    decide_with_unary_proof_kernel,
    kernel_decision_to_decode,
    kernel_decision_to_dict,
    proof_result_to_dict,
    semantics_requires_designer,
    task_contract_from_semantics,
    task_semantics_to_dict,
    typed_obligation_to_dict,
    validate_kernel_certificate_bank,
    verifier_binding_to_dict,
)
from research_experiments.families.contrastive_active_testing.kernel_adapters import (
    candidate_adapter_result_to_dict,
    run_kernel_adapters,
    run_kernel_unary_adapters,
)
from research_experiments.families.contrastive_active_testing.kernel_d3 import (
    D3_CAPABILITY_REGISTRY_VERSION,
    D3_IR_SCHEMA,
    RouteDecision,
    SolverCertificate,
    SourceIR,
    answer_schema_for_sample,
    candidate_evaluation_to_dict,
    canonical_ir,
    evaluate_candidate,
    parse_source_ir,
    route_for_sample,
    solve_exact,
    solve_numeric_ir,
    solver_certificate_to_dict,
    source_ir_from_exact_sample,
    source_ir_to_dict,
)
from research_experiments.families.contrastive_active_testing.kernel_d3 import (
    KernelDecision as D3KernelDecision,
)
from research_experiments.families.contrastive_active_testing.kernel_d4 import (
    D4_CAPABILITY_REGISTRY_VERSION,
    D4_IR_SCHEMA,
    D4_PROMPT_VERSION,
    D4SolverResult,
    build_proof_package,
    compile_exact_source_ir,
    load_risk_evidence,
    metamorphic_checks_passed,
    parse_source_ir_v3,
    proof_package_to_dict,
    risk_gate_snapshot,
    risk_gate_to_dict,
    run_metamorphic_checks,
)
from research_experiments.families.contrastive_active_testing.kernel_d4 import (
    answer_contract_for_sample as d4_answer_contract_for_sample,
)
from research_experiments.families.contrastive_active_testing.kernel_d4 import (
    evaluate_candidates as d4_evaluate_candidates,
)
from research_experiments.families.contrastive_active_testing.kernel_d4 import (
    route_for_sample as d4_route_for_sample,
)
from research_experiments.families.contrastive_active_testing.kernel_d4 import (
    solve_source_ir as d4_solve_source_ir,
)
from research_experiments.families.contrastive_active_testing.kernel_d4 import (
    solver_result_to_dict as d4_solver_result_to_dict,
)
from research_experiments.families.contrastive_active_testing.kernel_d4 import (
    source_ir_to_dict as d4_source_ir_to_dict,
)
from research_experiments.families.contrastive_active_testing.kernel_prompts import (
    build_d3_source_compiler_messages,
    build_d4_source_compiler_messages,
    build_kernel_designer_messages,
    build_kernel_verifier_messages,
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


def run_stage_a_only_sample(
    sample: DatasetSample,
    *,
    run_id: str,
    split_name: str,
    experiment,
    protocol,
    endpoint,
    network_budget: NetworkAttemptBudget,
) -> tuple[list[dict[str, Any]], Any]:
    """Run only the shared five-solver screening pass for CATCH-Cert.

    The screening pass is deliberately independent of gold and of all
    certificate calls.  Its returned rows can be replayed as the fixed
    Stage-A candidate set for the disagreement subset.
    """

    rows = [
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
    return rows, build_stage_decision(rows, seed=experiment.global_seed, sample_id=sample.sample_id)


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
    if protocol.protocol_version in {"catch_cert_v2", "catch_kernel_v1"}:
        if (
            protocol.protocol_version == "catch_kernel_v1"
            and str(getattr(experiment, "raw", {}).get("kernel_revision") or "")
            == "d4_proof_carrying_v1"
        ):
            return run_catch_kernel_d4_sample(
                sample,
                run_id=run_id,
                split_name=split_name,
                experiment=experiment,
                protocol=protocol,
                endpoint=endpoint,
                network_budget=network_budget,
                phase_name=phase_name,
                precomputed_stage_rows=precomputed_stage_rows,
            )
        if (
            protocol.protocol_version == "catch_kernel_v1"
            and str(getattr(experiment, "raw", {}).get("kernel_revision") or "") == "d3_source_blind_v1"
        ):
            return run_catch_kernel_d3_sample(
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
        return run_catch_cert_v2_sample(
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
    if protocol.protocol_version == "catch_cert_v1":
        return run_catch_cert_sample(
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

        d_values = [int(frozen_decoding["d_min"])] if frozen_decoding is not None else list(protocol.d_min_grid)
        margin_values = [int(frozen_decoding["margin"])] if frozen_decoding is not None else list(protocol.margin_grid)
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
                    max_tokens=protocol.judge_max_tokens,
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
                    key: [evidence_unit_to_dict(unit) for unit in units] for key, units in evidence.items()
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
                item for item in validation.coordinates if item.challenger_key in validation.eligible_challengers
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
                    max_tokens=protocol.judge_max_tokens,
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
                    max_tokens=protocol.judge_max_tokens,
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
        candidate.key in target_keys and _score(sample, candidate.answer) == 1.0 for candidate in stage.candidates
    )
    gold_candidate_keys = tuple(
        candidate.key for candidate in stage.candidates if _score(sample, candidate.answer) == 1.0
    )
    gold_candidate_key = gold_candidate_keys[0] if gold_candidate_keys else None
    common_extra = {
        "target_oracle_correct": target_oracle,
        "gold_candidate_key": gold_candidate_key,
        "gold_candidate_keys": list(gold_candidate_keys),
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
        "gold_candidate_keys": list(gold_candidate_keys),
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


def run_catch_cert_sample(
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
    """Execute CATCH-Cert with one designer and two blinded verifier calls."""

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
            raise ValueError("Precomputed Stage-A rows do not match the CATCH-Cert sample contract.")

    stage = build_stage_decision(stage_rows, seed=experiment.global_seed, sample_id=sample.sample_id)
    physical_rows: list[dict[str, Any]] = list(stage_rows)
    resample_rows: list[dict[str, Any]] = []
    designer_row: dict[str, Any] | None = None
    verifier_rows: list[dict[str, Any]] = []
    direct_rows: list[dict[str, Any]] = []
    pair_judge_rows: list[dict[str, Any]] = []
    direct_selections: list[str | None] = []
    pair_selections: list[str | None] = []
    panels: list[CertificateVerifierParseResult] = []
    contract = build_task_contract(sample)
    public_to_key = build_hypothesis_labels(stage, seed=experiment.global_seed, sample_id=sample.sample_id)
    key_to_public = {key: public for public, key in public_to_key.items()}
    graphs = build_claim_graphs(stage, public_to_key=public_to_key)
    target_pairs = build_target_pairs(stage, seed=experiment.global_seed, sample_id=sample.sample_id)
    public_pairs = tuple(
        {
            "pair_id": pair.pair_id,
            "left_candidate": key_to_public[pair.left_candidate_key],
            "right_candidate": key_to_public[pair.right_candidate_key],
        }
        for pair in target_pairs
    )
    pair_candidates = {pair["pair_id"]: (pair["left_candidate"], pair["right_candidate"]) for pair in public_pairs}
    validation = CertificateBankValidation((), (), (), None, 0, (), ())
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
        designer_row, designer_payload = _json_turn(
            sample,
            run_id=run_id,
            split_name=split_name,
            endpoint=endpoint,
            network_budget=network_budget,
            method_name="catch_cert",
            role="certificate_designer",
            agent_id=1,
            seed=48_000,
            max_tokens=protocol.role_max_tokens,
            messages=build_certificate_designer_messages(
                sample,
                contract=contract,
                graphs=graphs,
                public_pairs=public_pairs,
            ),
        )
        validation = validate_certificate_bank(
            designer_payload,
            contract=contract,
            stage=stage,
            public_to_key=public_to_key,
            graphs=graphs,
            pair_candidates=pair_candidates,
            max_tests=max(2, int(protocol.max_selected_tests or 6)),
        )
        designer_row.update(
            {
                "task_contract": task_contract_to_dict(contract),
                "claim_graphs": {key: claim_graph_to_dict(graph) for key, graph in graphs.items()},
                "candidate_public_to_answer_class_key": public_to_key,
                "public_pairs": list(public_pairs),
                "validated_certificates": [certificate_to_dict(item) for item in validation.certificates],
                "validated_certificate_tests": [certificate_test_to_dict(item) for item in validation.tests],
                "dropped_certificate_items": list(validation.dropped),
                "certificate_protocol_error": validation.protocol_error,
                "certificate_leakage_count": validation.leakage_count,
                "adapter_conflicts": list(validation.adapter_conflicts),
                "eligible_challengers": list(validation.eligible_challengers),
            }
        )
        if validation.protocol_error is not None:
            designer_row["protocol_parse_status"] = "failed"
            designer_row["protocol_parse_error"] = validation.protocol_error

        if validation.eligible_challengers and not validation.adapter_conflicts:
            eligible_public = {key_to_public[key] for key in validation.eligible_challengers}
            anchor_public = key_to_public.get(stage.anchor_key)
            eligible_tests = tuple(
                test
                for test in validation.tests
                if set(test.expected_outcome_by_candidate) & eligible_public
                and anchor_public in test.expected_outcome_by_candidate
            )
            for panel_index in range(1, protocol.witness_count + 1):
                packet = build_certificate_verifier_packet(
                    eligible_tests,
                    seed=experiment.global_seed,
                    sample_id=sample.sample_id,
                    panel_index=panel_index,
                )
                row, payload = _json_turn(
                    sample,
                    run_id=run_id,
                    split_name=split_name,
                    endpoint=endpoint,
                    network_budget=network_budget,
                    method_name="catch_cert",
                    role="certificate_verifier",
                    agent_id=panel_index,
                    seed=49_000 + panel_index,
                    max_tokens=protocol.role_max_tokens,
                    messages=build_certificate_verifier_messages(sample, contract=contract, packet=packet),
                )
                parsed = parse_certificate_verifier(payload, packet=packet)
                if not parsed.top_level_valid:
                    row["protocol_parse_status"] = "failed"
                    row["protocol_parse_error"] = "certificate_verifier_top_level_schema_failure"
                row.update(
                    {
                        "certificate_verifier_packet": {
                            "panel_index": packet.panel_index,
                            "role": packet.role,
                            "tests": list(packet.tests),
                            "public_test_to_internal": packet.public_test_to_internal,
                            "public_outcome_to_internal": packet.public_outcome_to_internal,
                        },
                        "certificate_verifier_results": {
                            key: verifier_result_to_dict(value) for key, value in parsed.results.items()
                        },
                        "certificate_verifier_parse_diagnostics": {
                            "top_level_valid": parsed.top_level_valid,
                            "expected_test_count": parsed.expected_test_count,
                            "valid_test_count": parsed.valid_test_count,
                            "erased_rows": list(parsed.erased_rows),
                        },
                    }
                )
                verifier_rows.append(row)
                panels.append(parsed)
            decision = decode_certificates(
                stage,
                validation=validation,
                public_to_key=public_to_key,
                panels=tuple(panels),
            )
        elif validation.adapter_conflicts:
            decision = DecodeDecision(stage.anchor_answer, stage.anchor_key, False, "adapter_conflict", (), ())
        else:
            resolver = "no_certificate" if not validation.tests else "certificate_invalid"
            decision = DecodeDecision(stage.anchor_answer, stage.anchor_key, False, resolver, (), ())

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
                    max_tokens=protocol.judge_max_tokens,
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

            target_keys = [stage.anchor_key, *(pair.challenger_key for pair in target_pairs)]
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
                    max_tokens=protocol.judge_max_tokens,
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
                *([designer_row] if designer_row is not None else []),
                *verifier_rows,
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
    target_keys = {stage.anchor_key, *(pair.challenger_key for pair in target_pairs)}
    target_oracle = any(
        candidate.key in target_keys and _score(sample, candidate.answer) == 1.0 for candidate in stage.candidates
    )
    gold_candidate_keys = tuple(
        candidate.key for candidate in stage.candidates if _score(sample, candidate.answer) == 1.0
    )
    gold_candidate_key = gold_candidate_keys[0] if gold_candidate_keys else None
    common_extra = {
        "target_oracle_correct": target_oracle,
        "gold_candidate_key": gold_candidate_key,
        "gold_candidate_keys": list(gold_candidate_keys),
        "target_candidate_count": len(target_keys),
        "protocol_version": "catch_cert_v1",
        "task_family": contract.family,
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
            "catch_cert",
            decision.answer,
            stage.anchor_answer,
            stage,
            [*stage_rows, *([designer_row] if designer_row is not None else []), *verifier_rows],
            [*([designer_row] if designer_row is not None else []), *verifier_rows],
            decision.override_accepted,
            decision.resolver,
            candidate_oracle,
            extra={
                **common_extra,
                "certificate_count": len(validation.certificates),
                "certificate_test_count": len(validation.tests),
                "eligible_challenger_count": len(validation.eligible_challengers),
                "certificate_coverage": float(bool(validation.eligible_challengers)),
                "certificate_abstained": not decision.override_accepted,
                "adapter_conflict_count": len(validation.adapter_conflicts),
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
        "protocol_version": "catch_cert_v1",
        "triggered": stage.triggered,
        "valid_stage_answers": stage.valid_count,
        "anchor_answer": stage.anchor_answer,
        "anchor_key": stage.anchor_key,
        "answer_class_vote_counts": stage.vote_counts,
        "candidate_count": len(stage.candidates),
        "candidate_oracle_correct": candidate_oracle,
        "target_oracle_correct": target_oracle,
        "gold_candidate_key": gold_candidate_key,
        "task_contract": task_contract_to_dict(contract),
        "candidate_public_to_answer_class_key": public_to_key,
        "public_pairs": list(public_pairs),
        "claim_graphs": {key: claim_graph_to_dict(graph) for key, graph in graphs.items()},
        "certificate_protocol_error": validation.protocol_error,
        "dropped_certificate_items": list(validation.dropped),
        "certificates": [certificate_to_dict(item) for item in validation.certificates],
        "certificate_tests": [certificate_test_to_dict(item) for item in validation.tests],
        "certificate_leakage_count": validation.leakage_count,
        "adapter_conflicts": list(validation.adapter_conflicts),
        "eligible_challengers": list(validation.eligible_challengers),
        "verifier_panels": [
            {
                "top_level_valid": panel.top_level_valid,
                "results": {key: verifier_result_to_dict(value) for key, value in panel.results.items()},
                "valid_test_count": panel.valid_test_count,
                "expected_test_count": panel.expected_test_count,
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


def run_catch_kernel_d4_sample(
    sample: DatasetSample,
    *,
    run_id: str,
    split_name: str,
    experiment,
    protocol,
    endpoint,
    network_budget: NetworkAttemptBudget,
    phase_name: str,
    precomputed_stage_rows: tuple[dict[str, Any], ...] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Run the five D4 main comparators plus supplementary adaptive-SC8.

    Physical calls may be shared across comparators, but each prediction row
    records its own logical calls: 5 for exact/soft routes and 8 for semantic
    compiler routes.
    """

    d4_output_config = dict(getattr(experiment, "raw", {}).get("d4_output") or {})
    if d4_output_config.get("stage_a_protocol") != "tagged_text":
        raise ValueError("D4 has one Stage-A protocol: tagged_text.")
    output_mode = "tagged_text"
    stage_output_protocol = FREE_TEXT_ANSWER_PROTOCOL_V1
    stage_prompt_version = FREE_TEXT_V1_PROMPT_VERSION
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
                output_protocol=stage_output_protocol,
                prompt_version=stage_prompt_version,
            )
            for index in range(1, protocol.stage_candidates + 1)
        ]
    else:
        stage_rows = list(precomputed_stage_rows)
        if len(stage_rows) != protocol.stage_candidates:
            raise ValueError("D4 precomputed Stage-A rows do not match the configured candidate count.")
    stage = build_stage_decision(stage_rows, seed=experiment.global_seed, sample_id=sample.sample_id)
    phase_config = dict(((getattr(experiment, "raw", {}) or {}).get("phases") or {}).get(phase_name) or {})
    evaluation_role = str(phase_config.get("evaluation_role") or "")
    if evaluation_role.startswith("d4_output_protocol_independent_validation_"):
        candidate_oracle = any(_score(sample, candidate.answer) == 1.0 for candidate in stage.candidates)
        prediction = _prediction(
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
            extra={
                "kernel_revision": "d4_proof_carrying_v1",
                "d4_stage_a_output_protocol": output_mode,
                "output_protocol_validation_only": True,
                "main_table_eligible": False,
            },
        )
        router = {
            "run_id": run_id,
            "dataset": sample.dataset,
            "split": split_name,
            "sample_id": sample.sample_id,
            "task": sample.metadata.get("task"),
            "phase_name": phase_name,
            "protocol_version": protocol.protocol_version,
            "kernel_revision": "d4_proof_carrying_v1",
            "route": "OUTPUT_PROTOCOL_AB_ONLY",
            "output_protocol": output_mode,
            "anchor_answer": stage.anchor_answer,
            "first_failure_layer": "NONE",
        }
        return stage_rows, router, [prediction]
    resample_rows = [
        _answer_turn(
            sample,
            run_id=run_id,
            split_name=split_name,
            endpoint=endpoint,
            network_budget=network_budget,
            method_name="catch_fixed_resample_shared",
            role="independent_resample",
            agent_id=index,
            seed=45_000 + index,
            max_tokens=protocol.solver_max_tokens,
            output_protocol=stage_output_protocol,
            prompt_version=stage_prompt_version,
        )
        for index in range(1, protocol.resample_candidates + 1)
    ]
    fixed_stage = build_stage_decision(
        [*stage_rows, *resample_rows], seed=experiment.global_seed, sample_id=sample.sample_id
    )
    fixed_answer = fixed_stage.anchor_answer or stage.anchor_answer
    adaptive_stage = fixed_stage if stage.triggered else stage
    adaptive_answer = adaptive_stage.anchor_answer or stage.anchor_answer

    decision = d4_route_for_sample(sample)
    source_ir = None
    solver = D4SolverResult(
        status="UNSUPPORTED",
        canonical_answer=None,
        answer_text=None,
        solver_trace=(),
        reference_checker_status="NOT_RUN",
        concrete_witness_status={"status": "NOT_AVAILABLE"},
        reason="route_not_compiled",
    )
    compiler_rows: list[dict[str, Any]] = []
    parsed_irs: list[tuple[int, Any]] = []
    compiler_vote_hashes: tuple[str, ...] = ()
    compiler_verifications: tuple[dict[str, Any], ...] = ()
    compiler_agreement = False
    metamorphic_status: dict[str, str] = {}
    if decision.route == "EXACT_EXECUTABLE":
        source_ir, compile_reason = compile_exact_source_ir(sample, decision)
        if source_ir is None:
            solver = replace(solver, reason=compile_reason)
        else:
            solver = d4_solve_source_ir(sample, decision, source_ir)
    elif decision.route == "SEMANTIC_EXECUTABLE":
        graph = build_source_span_graph(sample)
        source_spans = [{"span_id": span.span_id, "text": span.text} for span in graph.spans]
        answer_contract = d4_answer_contract_for_sample(sample)
        for compiler_index in range(1, protocol.resample_candidates + 1):
            row, payload = _json_turn(
                sample,
                run_id=run_id,
                split_name=split_name,
                endpoint=endpoint,
                network_budget=network_budget,
                method_name="catch_kernel_d4_source_compiler",
                role="d4_source_compiler",
                agent_id=compiler_index,
                seed=49_000 + compiler_index,
                max_tokens=protocol.role_max_tokens,
                messages=build_d4_source_compiler_messages(
                    sample,
                    source_spans=source_spans,
                    answer_contract=answer_contract,
                    decision=decision,
                ),
            )
            parsed, reason = parse_source_ir_v3(payload, sample=sample, decision=decision)
            row["d4_ir_schema"] = D4_IR_SCHEMA
            row["d4_source_ir"] = d4_source_ir_to_dict(parsed) if parsed is not None else None
            row["d4_source_ir_parse_reason"] = reason
            if parsed is None:
                row["protocol_parse_status"] = "failed"
                row["protocol_parse_error"] = reason
                # JSON syntax was valid, but the D4 source contract was not.
                # Do not let this raw row poison a later resumed D4 run.
                _cache_for_role(endpoint, "d4_source_compiler").delete(str(row.get("cache_key") or ""))
                _record_d4_completion(endpoint, row)
            else:
                parsed_irs.append((compiler_index, parsed))
            compiler_rows.append(row)
        (
            source_ir,
            solver,
            compiler_vote_hashes,
            compiler_verifications,
            compiler_agreement,
            metamorphic_status,
        ) = verify_d4_semantic_compiler_consensus(
            sample=sample,
            decision=decision,
            parsed_irs=parsed_irs,
            required_compiler_count=protocol.resample_candidates,
            fallback_solver=solver,
        )
        verification_by_index = {
            int(item["compiler_index"]): item for item in compiler_verifications
        }
        for row in compiler_rows:
            compiler_index = int(row.get("agent_id") or 0)
            verification = verification_by_index.get(compiler_index)
            row["d4_compiler_verification"] = dict(verification) if verification is not None else None
            row["d4_compiler_verification_passed"] = bool(
                verification is not None and verification.get("passed")
            )
            if row.get("protocol_parse_status") == "ok" and not row["d4_compiler_verification_passed"]:
                # A syntactically valid but locally unverifiable SourceIR is a
                # D4 protocol failure, not a reusable shared response.
                _cache_for_role(endpoint, "d4_source_compiler").delete(str(row.get("cache_key") or ""))
            _record_d4_completion(endpoint, row)

    candidate_evaluation = d4_evaluate_candidates(
        sample,
        [candidate.answer for candidate in stage.candidates],
        solver,
    )
    candidate_present = any(item.get("status") == "VALID" for item in candidate_evaluation)
    if decision.route == "EXACT_EXECUTABLE" and source_ir is not None:
        metamorphic_status = run_metamorphic_checks(source_ir, solver, sample=sample, decision=decision)
    d4_risk = dict(getattr(experiment, "raw", {}).get("d4_risk") or {})
    evidence = load_risk_evidence(d4_risk.get("evidence_path"))
    risk_snapshot = risk_gate_snapshot(
        decision.capability_id,
        route=decision.route,
        evidence=evidence,
        phase="development",
    )
    meta_passed = bool(
        source_ir is not None
        and metamorphic_checks_passed(source_ir, solver, metamorphic_status)
    )
    gate_active = risk_snapshot.route_activation_state.startswith("ACTIVE")
    route_enabled = (
        decision.foundation
        or (
            decision.route == "EXACT_EXECUTABLE"
            and bool(d4_risk.get("new_exact_override_enabled", False))
            and gate_active
        )
        or (
            decision.route == "SEMANTIC_EXECUTABLE"
            and bool(d4_risk.get("semantic_override_enabled", False))
            and gate_active
        )
    )
    solver_unique = solver.status == "UNIQUE" and bool(solver.canonical_answer)
    d4_authorized = bool(solver_unique and meta_passed and route_enabled)
    d4_answer = solver.canonical_answer if d4_authorized else stage.anchor_answer
    stage_canonical = canonicalize_answer(sample, stage.anchor_answer)
    d4_override = bool(
        d4_authorized
        and (not stage_canonical.valid or stage_canonical.key != solver.canonical_answer)
    )
    shadow_override = bool(
        solver_unique
        and (not stage_canonical.valid or stage_canonical.key != solver.canonical_answer)
    )
    shadow_score = _score(sample, solver.canonical_answer) if solver_unique else None
    initial_score = _score(sample, stage.anchor_answer)
    d4_resolver = (
        "d4_candidate_completion"
        if d4_override and not candidate_present
        else "d4_solver_direct"
        if d4_authorized
        else "d4_risk_abstain"
        if solver_unique
        else "d4_jurisdiction_abstain"
    )

    # SSV-raw shares the three candidate-blind compiler calls but omits the
    # frozen risk gate and metamorphic/provenance authorization layer.
    ssv_authorized = bool(
        decision.route == "SEMANTIC_EXECUTABLE" and compiler_agreement and solver_unique
    )
    ssv_answer = solver.canonical_answer if ssv_authorized else stage.anchor_answer
    ssv_override = bool(
        ssv_authorized and (not stage_canonical.valid or stage_canonical.key != solver.canonical_answer)
    )

    d3_decision = route_for_sample(sample)
    d3_certificate = (
        solve_exact(sample, d3_decision)
        if d3_decision.route == "EXACT_EXECUTABLE"
        else None
    )
    d3_unique = bool(
        d3_certificate is not None
        and d3_certificate.status == "UNIQUE"
        and d3_certificate.canonical_answer
    )
    d3_answer = d3_certificate.canonical_answer if d3_unique else stage.anchor_answer
    d3_override = bool(
        d3_unique and (not stage_canonical.valid or stage_canonical.key != d3_certificate.canonical_answer)
    )

    first_failure_layer = (
        "JURISDICTION"
        if decision.route == "SOFT_UNSUPPORTED"
        else "PARSE"
        if source_ir is None and decision.route == "EXACT_EXECUTABLE"
        else "COMPILER_PARSE_OR_AGREEMENT"
        if decision.route == "SEMANTIC_EXECUTABLE" and not compiler_agreement
        else "SOLVER"
        if not solver_unique
        else "METAMORPHIC"
        if not meta_passed
        else "RISK_GATE"
        if not route_enabled
        else "NONE"
    )
    proof_package = build_proof_package(
        sample=sample,
        ir=source_ir,
        solver=solver,
        compiler_vote_hashes=compiler_vote_hashes,
        compiler_verifications=compiler_verifications,
        candidate_evaluation=candidate_evaluation,
        metamorphic_status=metamorphic_status,
        risk_snapshot=risk_snapshot,
        compiler_input_fields=(
            "source",
            "answer_contract",
            "source_spans",
            "capability_id",
            "query_operator",
        ),
        first_failure_layer=first_failure_layer,
    )
    candidate_oracle = any(_score(sample, candidate.answer) == 1.0 for candidate in stage.candidates)
    semantic_rows = [*stage_rows, *compiler_rows]
    phase_sample_limit = int(dict(phase_config.get("sample_limits") or {}).get(sample.dataset, 0))
    common_extra = {
        "protocol_version": protocol.protocol_version,
        "kernel_revision": "d4_proof_carrying_v1",
        "d4_capability_registry_version": D4_CAPABILITY_REGISTRY_VERSION,
        "d4_prompt_version": D4_PROMPT_VERSION,
        "d4_stage_a_output_protocol": output_mode,
        "d4_route": decision.route,
        "d4_kernel_id": decision.kernel_id,
        "d4_capability_id": decision.capability_id,
        "d4_query_operator": decision.query_operator,
        "d4_route_reason": decision.reason,
        "d4_source_ir": d4_source_ir_to_dict(source_ir) if source_ir is not None else None,
        "d4_solver_result": d4_solver_result_to_dict(solver),
        "d4_proof_package": proof_package_to_dict(proof_package),
        "d4_risk_gate_snapshot": risk_gate_to_dict(risk_snapshot),
        "d4_first_failure_layer": first_failure_layer,
        "d4_candidate_completion": d4_resolver == "d4_candidate_completion",
        "d4_solver_direct": d4_resolver == "d4_solver_direct",
        "d4_compiler_agreement": compiler_agreement,
        "d4_compiler_verifications": [dict(item) for item in compiler_verifications],
        "d4_shadow_answer": solver.canonical_answer if solver_unique else None,
        "d4_shadow_score": shadow_score,
        "d4_shadow_override": shadow_override,
        "d4_shadow_correction": bool(shadow_override and initial_score < 1 and shadow_score == 1),
        "d4_shadow_harm": bool(shadow_override and initial_score == 1 and shadow_score is not None and shadow_score < 1),
        "d4_metamorphic_checks_passed": meta_passed,
        "primary_metric": _d3_primary_metric(
            sample.dataset,
            split_name,
            phase_name=phase_name,
            sample_limit=phase_sample_limit,
        ),
    }
    predictions = [
        _prediction(
            sample, run_id, split_name, "sc_5", stage.anchor_answer, stage.anchor_answer,
            stage, stage_rows, [], False, "stage_a_plurality", candidate_oracle,
            extra={**common_extra, "main_table_eligible": True},
        ),
        _prediction(
            sample, run_id, split_name, "fixed_sc_8", fixed_answer, stage.anchor_answer,
            stage, [*stage_rows, *resample_rows], resample_rows,
            fixed_answer != stage.anchor_answer, "fixed_answer_class_plurality", candidate_oracle,
            extra={**common_extra, "main_table_eligible": True, "intervention_call_budget_per_question": 3},
        ),
        _prediction(
            sample, run_id, split_name, "catch_d3_exact_only", d3_answer, stage.anchor_answer,
            stage, stage_rows, [], d3_override,
            "d3_exact_solver" if d3_unique else "d3_exact_anchor", candidate_oracle,
            extra={**common_extra, "main_table_eligible": True},
        ),
        _prediction(
            sample, run_id, split_name, "ssv_raw", ssv_answer, stage.anchor_answer,
            stage, semantic_rows if decision.route == "SEMANTIC_EXECUTABLE" else stage_rows,
            compiler_rows if decision.route == "SEMANTIC_EXECUTABLE" else [],
            ssv_override, "ssv_raw_solver" if ssv_authorized else "ssv_raw_abstain", candidate_oracle,
            extra={
                **common_extra,
                "main_table_eligible": True,
                "ssv_raw": True,
                "intervention_call_budget_per_question": 3
                if decision.route == "SEMANTIC_EXECUTABLE"
                else 0,
            },
        ),
        _prediction(
            sample, run_id, split_name, "catch_kernel_d4", d4_answer, stage.anchor_answer,
            stage, semantic_rows if decision.route == "SEMANTIC_EXECUTABLE" else stage_rows,
            compiler_rows if decision.route == "SEMANTIC_EXECUTABLE" else [],
            d4_override, d4_resolver, candidate_oracle,
            extra={
                **common_extra,
                "main_table_eligible": True,
                "certificate_count": int(solver_unique),
                "certificate_coverage": float(solver_unique),
                "certificate_abstained": not d4_authorized,
                "solver_status": solver.status,
                "intervention_call_budget_per_question": 3
                if decision.route == "SEMANTIC_EXECUTABLE"
                else 0,
            },
        ),
        _prediction(
            sample, run_id, split_name, "adaptive_sc_8", adaptive_answer, stage.anchor_answer,
            stage, [*stage_rows, *(resample_rows if stage.triggered else [])],
            resample_rows if stage.triggered else [],
            bool(stage.triggered and adaptive_answer != stage.anchor_answer),
            "adaptive_answer_class_plurality" if stage.triggered else "no_answer_class_disagreement",
            candidate_oracle,
            extra={**common_extra, "main_table_eligible": False, "supplementary_only": True},
        ),
    ]
    router = {
        "run_id": run_id,
        "dataset": sample.dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "task": sample.metadata.get("task"),
        "phase_name": phase_name,
        "protocol_version": protocol.protocol_version,
        "kernel_revision": "d4_proof_carrying_v1",
        "route": decision.route,
        "kernel_id": decision.kernel_id,
        "capability_id": decision.capability_id,
        "query_operator": decision.query_operator,
        "route_reason": decision.reason,
        "source_ir": d4_source_ir_to_dict(source_ir) if source_ir is not None else None,
        "solver_result": d4_solver_result_to_dict(solver),
        "proof_package": proof_package_to_dict(proof_package),
        "risk_gate_snapshot": risk_gate_to_dict(risk_snapshot),
        "anchor_answer": stage.anchor_answer,
        "decision": {
            "answer": d4_answer,
            "override_accepted": d4_override,
            "resolver": d4_resolver,
            "candidate_completion": d4_resolver == "d4_candidate_completion",
        },
        "first_failure_layer": first_failure_layer,
        "compiler_diagnostics": [
            {
                "agent_id": row.get("agent_id"),
                "parse_status": row.get("protocol_parse_status"),
                "parse_reason": row.get("d4_source_ir_parse_reason"),
                "ir_hash": (row.get("d4_source_ir") or {}).get("canonical_ir_hash"),
            }
            for row in compiler_rows
        ],
    }
    return [*stage_rows, *resample_rows, *compiler_rows], router, predictions


def run_catch_kernel_d3_sample(
    sample: DatasetSample,
    *,
    run_id: str,
    split_name: str,
    experiment,
    protocol,
    endpoint,
    network_budget: NetworkAttemptBudget,
    phase_name: str,
    run_direct_judge: bool = False,
    precomputed_stage_rows: tuple[dict[str, Any], ...] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """D3 route-exclusive execution with source-blind certificates.

    Exact routes never spend additional model calls.  Semantic routes spend
    exactly ``resample_candidates`` compiler calls and require agreement.  The
    soft route resamples every item, including unanimous Stage-A items, so the
    confident-wrong failure mode is not silently excluded.
    """

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
        if len(stage_rows) != protocol.stage_candidates:
            raise ValueError("D3 precomputed Stage-A rows do not match the configured candidate count.")

    stage = build_stage_decision(stage_rows, seed=experiment.global_seed, sample_id=sample.sample_id)
    risk_config = dict(getattr(experiment, "raw", {}).get("d3_risk") or {})
    route = route_for_sample(sample)
    # Semantic compilation remains an explicit shadow experiment until the
    # source compiler passes its independent IR and metamorphic audit.  The
    # production D3 route therefore fails closed at the jurisdiction boundary
    # unless a config explicitly opts into shadow collection.
    if route.route == "SEMANTIC_COMPILABLE" and not bool(risk_config.get("semantic_shadow_enabled", False)):
        route = RouteDecision("SOFT_UNSUPPORTED", "none", "semantic_shadow_disabled_until_audit")
    source_graph = build_source_span_graph(sample)
    source_ir: SourceIR | None = None
    certificate = None
    compiler_rows: list[dict[str, Any]] = []
    # These three rows are shared by fixed-SC8, adaptive-SC8 when triggered,
    # and D3's soft route.  They are not charged to exact/semantic D3.
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

    if route.route == "EXACT_EXECUTABLE":
        source_ir = source_ir_from_exact_sample(sample, route)
        certificate = solve_exact(sample, route, source_ir)
    elif route.route == "SEMANTIC_COMPILABLE":
        source_spans = [{"span_id": span.span_id, "text": span.text} for span in source_graph.spans]
        answer_schema = answer_schema_for_sample(sample)
        parsed_irs: list[SourceIR] = []
        for compiler_index in range(1, protocol.resample_candidates + 1):
            row, payload = _json_turn(
                sample,
                run_id=run_id,
                split_name=split_name,
                endpoint=endpoint,
                network_budget=network_budget,
                method_name="catch_kernel_d3_source_compiler",
                role="d3_source_compiler",
                agent_id=compiler_index,
                seed=49_000 + compiler_index,
                max_tokens=protocol.role_max_tokens,
                messages=build_d3_source_compiler_messages(
                    sample,
                    source_spans=source_spans,
                    answer_schema=answer_schema,
                    operation_kind=route.operation_kind,
                ),
            )
            parsed, parse_reason = parse_source_ir(payload, sample=sample, decision=route)
            row["d3_ir_schema"] = D3_IR_SCHEMA
            row["d3_source_ir"] = source_ir_to_dict(parsed) if parsed is not None else None
            row["d3_source_ir_parse_reason"] = parse_reason
            if parsed is None:
                row["protocol_parse_status"] = "failed"
                row["protocol_parse_error"] = parse_reason
            else:
                parsed_irs.append(parsed)
            compiler_rows.append(row)
        if len(parsed_irs) == protocol.resample_candidates and len({canonical_ir(item) for item in parsed_irs}) == 1:
            source_ir = parsed_irs[0]
            certificate = solve_numeric_ir(sample, route, source_ir)
        else:
            certificate = SolverCertificate(
                status="UNSUPPORTED",
                canonical_answer=None,
                route=route.route,
                operation_kind=route.operation_kind,
                source_hash=_sha256(question_without_answer_contract(sample)),
                ir_hash="",
                solver_version="catch_d3_safe_numeric_v1",
                cross_check_status="NOT_RUN",
                metamorphic_test_status="NOT_RUN",
                reason="compiler_ir_non_agreement_or_parse_failure",
            )
    fixed_sc = build_stage_decision(
        [*stage_rows, *resample_rows], seed=experiment.global_seed, sample_id=sample.sample_id
    )
    adaptive = fixed_sc if stage.triggered else stage
    adaptive_answer = adaptive.anchor_answer or stage.anchor_answer
    fixed_answer = fixed_sc.anchor_answer or stage.anchor_answer
    stage_canonical = canonicalize_answer(sample, stage.anchor_answer)
    evaluations = []
    if certificate is not None and certificate.status == "UNIQUE":
        evaluations = [evaluate_candidate(sample, candidate.answer, certificate) for candidate in stage.candidates]
    solver_answer = certificate.canonical_answer if certificate is not None else None
    certificate_unique = bool(certificate is not None and certificate.status == "UNIQUE" and solver_answer)
    candidate_present = any(item.status == "VALID" for item in evaluations)
    semantic_override_enabled = bool(risk_config.get("semantic_override_enabled", False))
    semantic_gate_passed = all(
        bool(risk_config.get(field, False))
        for field in (
            "semantic_precision_gate_passed",
            "semantic_metamorphic_suite_passed",
            "semantic_human_audit_passed",
        )
    )
    solver_authorized = bool(
        solver_answer
        and certificate is not None
        and certificate.status == "UNIQUE"
        and (
            route.route == "EXACT_EXECUTABLE"
            or (semantic_override_enabled and semantic_gate_passed)
        )
    )
    if solver_authorized:
        final_answer = solver_answer
        resolver = "d3_solver_direct" if candidate_present else "d3_candidate_completion"
        override = not stage_canonical.valid or stage_canonical.key != solver_answer
    elif solver_answer and certificate is not None and certificate.status == "UNIQUE":
        final_answer = stage.anchor_answer
        resolver = "d3_semantic_shadow"
        override = False
    elif route.route == "SOFT_UNSUPPORTED":
        soft_fallback = str(risk_config.get("soft_fallback") or "stage_a_anchor")
        if soft_fallback == "stage_a_anchor":
            final_answer = stage.anchor_answer
            resolver = "d3_soft_anchor"
            override = False
        elif soft_fallback in {"adaptive_sc_8", "fixed_sc_8"}:
            final_answer = fixed_answer
            resolver = f"d3_soft_{soft_fallback}"
            override = bool(fixed_answer != stage.anchor_answer)
        else:
            raise ValueError(f"Unsupported D3 soft_fallback: {soft_fallback!r}")
    else:
        final_answer = stage.anchor_answer
        resolver = "d3_abstain"
        override = False

    candidate_oracle = any(_score(sample, candidate.answer) == 1.0 for candidate in stage.candidates)
    gold_keys = tuple(candidate.key for candidate in stage.candidates if _score(sample, candidate.answer) == 1.0)
    soft_fallback = str(risk_config.get("soft_fallback") or "stage_a_anchor")
    interventions = (
        compiler_rows
        if route.route == "SEMANTIC_COMPILABLE"
        else resample_rows
        if route.route == "SOFT_UNSUPPORTED" and soft_fallback in {"adaptive_sc_8", "fixed_sc_8"}
        else []
    )
    direct_rows: list[dict[str, Any]] = []
    pair_judge_rows: list[dict[str, Any]] = []
    direct_selections: list[str | None] = []
    pair_selections: list[str | None] = []
    if run_direct_judge:
        for judge_index in range(1, protocol.direct_judge_count + 1):
            labels = build_hypothesis_labels(
                stage,
                seed=experiment.global_seed,
                sample_id=f"{sample.sample_id}:d3-direct:{judge_index}",
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
                max_tokens=protocol.judge_max_tokens,
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

        all_candidate_keys = [candidate.key for candidate in stage.candidates]
        for judge_index in range(1, protocol.pair_judge_count + 1):
            labels = _target_judge_labels(
                all_candidate_keys,
                seed=experiment.global_seed,
                sample_id=f"{sample.sample_id}:d3-pair:{judge_index}",
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
                max_tokens=protocol.judge_max_tokens,
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

    physical_rows = [*stage_rows, *resample_rows, *compiler_rows, *direct_rows, *pair_judge_rows]
    risk_tier = (
        "TIER_1_EXACT"
        if route.route == "EXACT_EXECUTABLE"
        else "TIER_2_SEMANTIC"
        if route.route == "SEMANTIC_COMPILABLE"
        else "TIER_3_SOFT"
    )
    certificate_hash = _sha256(solver_certificate_to_dict(certificate)) if certificate is not None else None
    decision_action = (
        "candidate_completion"
        if resolver == "d3_candidate_completion"
        else "solver_direct"
        if resolver == "d3_solver_direct"
        else "keep_anchor"
    )
    decision_record = D3KernelDecision(
        route=route.route,
        anchor=stage.anchor_answer,
        final_answer=final_answer,
        action=decision_action,
        override_reason=resolver,
        certificate_hash=certificate_hash,
        risk_tier=risk_tier,
    )
    first_failure_layer, first_failure_reason = _d3_first_failure(
        route=route.route,
        method_rows=[*stage_rows, *interventions],
        source_ir=source_ir,
        certificate=certificate,
    )
    common_extra = {
        "protocol_version": protocol.protocol_version,
        "kernel_revision": "d3_source_blind_v1",
        "d3_capability_registry_version": D3_CAPABILITY_REGISTRY_VERSION,
        "d3_route": route.route,
        "d3_operation_kind": route.operation_kind,
        "d3_route_reason": route.reason,
        "d3_source_ir": source_ir_to_dict(source_ir) if source_ir is not None else None,
        "d3_solver_certificate": solver_certificate_to_dict(certificate) if certificate is not None else None,
        "d3_candidate_evaluations": [candidate_evaluation_to_dict(item) for item in evaluations],
        "d3_candidate_completion": resolver == "d3_candidate_completion",
        "d3_solver_direct": resolver == "d3_solver_direct",
        "d3_semantic_override_enabled": semantic_override_enabled,
        "d3_semantic_gate_passed": semantic_gate_passed,
        "d3_risk_tier": risk_tier,
        "d3_first_failure_layer": first_failure_layer,
        "d3_first_failure_reason": first_failure_reason,
        "d3_kernel_decision": {
            "route": decision_record.route,
            "anchor": decision_record.anchor,
            "final_answer": decision_record.final_answer,
            "action": decision_record.action,
            "override_reason": decision_record.override_reason,
            "certificate_hash": decision_record.certificate_hash,
            "risk_tier": decision_record.risk_tier,
        },
        "gold_candidate_keys": list(gold_keys),
        "gold_candidate_key": gold_keys[0] if gold_keys else None,
        "stage_candidate_oracle_correct": candidate_oracle,
        "target_oracle_correct": candidate_oracle,
        "target_candidate_count": len(stage.candidates),
        "task_family": str(sample.metadata.get("task") or ""),
        "query_operator": route.operation_kind,
        "adapter_kind": route.operation_kind,
        "primary_metric": _d3_primary_metric(
            sample.dataset,
            split_name,
            phase_name=phase_name,
            sample_limit=int(
                ((getattr(experiment, "raw", {}) or {}).get("phases", {}).get(phase_name, {})
                 .get("sample_limits", {}) or {}).get(sample.dataset, 0)
            ),
        ),
    }
    predictions = [
        _prediction(
            sample, run_id, split_name, "sc_5", stage.anchor_answer, stage.anchor_answer,
            stage, stage_rows, [], False, "stage_a_plurality", candidate_oracle, extra=common_extra,
        ),
        _prediction(
            sample, run_id, split_name, "adaptive_sc_8", adaptive_answer, stage.anchor_answer,
            stage, [*stage_rows, *(resample_rows if stage.triggered else [])],
            resample_rows if stage.triggered else [],
            bool(stage.triggered and adaptive_answer != stage.anchor_answer),
            "adaptive_answer_class_plurality" if stage.triggered else "no_answer_class_disagreement",
            candidate_oracle,
            extra={**common_extra, "intervention_call_budget_per_question": 3},
        ),
        _prediction(
            sample, run_id, split_name, "fixed_sc_8", fixed_answer, stage.anchor_answer,
            stage, [*stage_rows, *resample_rows], resample_rows,
            bool(fixed_answer != stage.anchor_answer), "fixed_answer_class_plurality",
            candidate_oracle,
            extra={**common_extra, "intervention_call_budget_per_question": 3},
        ),
        _prediction(
            sample, run_id, split_name, "catch_kernel", final_answer, stage.anchor_answer,
            stage, [*stage_rows, *interventions], interventions, override, resolver, candidate_oracle,
            extra={
                **common_extra,
                "certificate_count": int(certificate is not None and certificate.status == "UNIQUE"),
                "certificate_coverage": float(certificate_unique),
                "certificate_abstained": bool(
                    route.route != "SOFT_UNSUPPORTED"
                    and not (certificate is not None and certificate.status == "UNIQUE")
                ),
                "solver_status": (
                    certificate.status
                    if certificate is not None
                    else "NOT_APPLICABLE"
                    if route.route == "SOFT_UNSUPPORTED"
                    else "UNSUPPORTED"
                ),
                "solver_certificate_hash": certificate.ir_hash if certificate is not None else None,
                "source_ir_coverage": (
                    len(source_ir.covered_span_ids)
                    / max(1, len(source_ir.covered_span_ids) + len(source_ir.uncovered_span_ids))
                    if source_ir is not None else 0.0
                ),
                "typed_compilation_validity": float(source_ir is not None),
                # A local parser/solver certificate is conditional on its
                # source-to-IR interpretation; without the required human
                # semantic audit it is not a semantic-validity estimate.
                "semantic_validity": None,
                "verifier_jurisdiction_coverage": float(route.route != "SOFT_UNSUPPORTED"),
                "proof_completeness": float(certificate is not None and certificate.status == "UNIQUE"),
                "override_reason": resolver,
                "intervention_call_budget_per_question": (
                    0
                    if route.route == "EXACT_EXECUTABLE" or (route.route == "SOFT_UNSUPPORTED" and not interventions)
                    else 3
                ),
            },
        ),
    ]

    # Full-run, post-hoc ablation: allow only the executable exact route to
    # override and keep the Stage-A anchor everywhere else.  This is useful
    # for deciding the next frozen protocol without mixing semantic shadow or
    # soft resampling into the same estimate.
    exact_only_answer = solver_answer if route.route == "EXACT_EXECUTABLE" and certificate_unique else stage.anchor_answer
    exact_only_override = bool(
        route.route == "EXACT_EXECUTABLE" and certificate_unique and exact_only_answer != stage.anchor_answer
    )
    predictions.append(
        _prediction(
            sample,
            run_id,
            split_name,
            "catch_d3_exact_only_ablation",
            exact_only_answer,
            stage.anchor_answer,
            stage,
            stage_rows,
            [],
            exact_only_override,
            "d3_exact_only_solver" if exact_only_override else "d3_exact_only_anchor",
            candidate_oracle,
            extra={
                **common_extra,
                "d3_variant": "catch_d3_exact_only_ablation",
                "experimental_only": True,
                "certificate_count": int(route.route == "EXACT_EXECUTABLE" and certificate_unique),
                "certificate_coverage": float(route.route == "EXACT_EXECUTABLE" and certificate_unique),
                "certificate_abstained": route.route != "EXACT_EXECUTABLE" or not certificate_unique,
                "solver_status": certificate.status if certificate is not None else "UNSUPPORTED",
                "intervention_call_budget_per_question": 0,
            },
        )
    )
    variant_rows = [*stage_rows, *compiler_rows]
    variant_interventions = compiler_rows
    if route.route == "EXACT_EXECUTABLE":
        exact_without_completion = solver_answer if certificate_unique and candidate_present else stage.anchor_answer
        exact_without_completion_override = bool(
            certificate_unique and candidate_present and exact_without_completion != stage.anchor_answer
        )
        for method_name, answer, variant_override, resolver_name in (
            (
                "catch_d3_exact_no_completion",
                exact_without_completion,
                exact_without_completion_override,
                (
                    "d3_exact_no_candidate_completion"
                    if certificate_unique and not candidate_present
                    else "d3_solver_direct"
                    if certificate_unique
                    else "d3_exact_abstain"
                ),
            ),
            (
                "catch_d3_exact_completion",
                solver_answer if certificate_unique else stage.anchor_answer,
                bool(certificate_unique and solver_answer != stage.anchor_answer),
                (
                    "d3_candidate_completion"
                    if certificate_unique and not candidate_present
                    else "d3_solver_direct"
                    if certificate_unique
                    else "d3_exact_abstain"
                ),
            ),
        ):
            predictions.append(
                _prediction(
                    sample, run_id, split_name, method_name, answer, stage.anchor_answer,
                    stage, variant_rows, variant_interventions, variant_override, resolver_name,
                    candidate_oracle,
                    extra={
                        **common_extra,
                        "d3_variant": method_name,
                        "experimental_only": True,
                        "certificate_count": int(certificate_unique),
                        "certificate_coverage": float(certificate_unique),
                        "certificate_abstained": not certificate_unique,
                        "solver_status": certificate.status if certificate is not None else "UNSUPPORTED",
                        "intervention_call_budget_per_question": 0,
                    },
                )
            )
    elif route.route == "SEMANTIC_COMPILABLE":
        semantic_answer = solver_answer if certificate_unique else stage.anchor_answer
        predictions.append(
            _prediction(
                sample, run_id, split_name, "catch_d3_semantic_compiler", semantic_answer, stage.anchor_answer,
                stage, variant_rows, variant_interventions, bool(certificate_unique and semantic_answer != stage.anchor_answer),
                "d3_semantic_compiler_ablation" if certificate_unique else "d3_semantic_compiler_abstain", candidate_oracle,
                extra={
                    **common_extra,
                    "d3_variant": "catch_d3_semantic_compiler",
                    "experimental_only": True,
                    "certificate_count": int(certificate_unique),
                    "certificate_coverage": float(certificate_unique),
                    "certificate_abstained": not certificate_unique,
                    "solver_status": certificate.status if certificate is not None else "UNSUPPORTED",
                    "intervention_call_budget_per_question": 3,
                },
            )
        )
    if route.route != "SOFT_UNSUPPORTED":
        solver_direct_answer = solver_answer if certificate_unique else stage.anchor_answer
        solver_direct_override = bool(certificate_unique and (not stage_canonical.valid or stage_canonical.key != solver_answer))
        predictions.insert(
            2,
            _prediction(
                sample, run_id, split_name, "solver_direct", solver_direct_answer, stage.anchor_answer,
                stage, [*stage_rows, *compiler_rows], compiler_rows,
                solver_direct_override, "d3_solver_direct_shadow" if certificate_unique else "d3_solver_direct_abstain",
                candidate_oracle,
                extra={
                    **common_extra,
                    "certificate_count": int(certificate_unique),
                    "certificate_coverage": float(certificate_unique),
                    "certificate_abstained": not certificate_unique,
                    "solver_status": certificate.status if certificate is not None else "UNSUPPORTED",
                    "intervention_call_budget_per_question": 0 if route.route == "EXACT_EXECUTABLE" else 3,
                },
            ),
        )
    if run_direct_judge:
        direct_answer, direct_override, direct_resolver = decide_direct_judges(stage, direct_selections)
        pair_answer, pair_override, pair_resolver = decide_direct_judges(stage, pair_selections)
        predictions.extend(
            [
                _prediction(
                    sample, run_id, split_name, "direct_judge_3", direct_answer, stage.anchor_answer,
                    stage, [*stage_rows, *direct_rows], direct_rows,
                    direct_override, direct_resolver, candidate_oracle,
                    extra={**common_extra, "intervention_call_budget_per_question": 3},
                ),
                _prediction(
                    sample, run_id, split_name, "pair_judge_3", pair_answer, stage.anchor_answer,
                    stage, [*stage_rows, *pair_judge_rows], pair_judge_rows,
                    pair_override, pair_resolver, candidate_oracle,
                    extra={**common_extra, "intervention_call_budget_per_question": 3},
                ),
            ]
        )
    router = {
        "run_id": run_id,
        "dataset": sample.dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "task": sample.metadata.get("task"),
        "phase_name": phase_name,
        "protocol_version": protocol.protocol_version,
        "kernel_revision": "d3_source_blind_v1",
        "route": route.route,
        "operation_kind": route.operation_kind,
        "route_reason": route.reason,
        "audit_source_question": question_without_answer_contract(sample),
        "triggered": stage.triggered,
        "valid_stage_answers": stage.valid_count,
        "anchor_answer": stage.anchor_answer,
        "anchor_key": stage.anchor_key,
        "answer_class_vote_counts": stage.vote_counts,
        "candidate_count": len(stage.candidates),
        "candidate_oracle_correct": candidate_oracle,
        "source_ir": source_ir_to_dict(source_ir) if source_ir is not None else None,
        "solver_certificate": solver_certificate_to_dict(certificate) if certificate is not None else None,
        "candidate_evaluations": [candidate_evaluation_to_dict(item) for item in evaluations],
        "compiler_diagnostics": [
            {
                "agent_id": row.get("agent_id"),
                "parse_status": row.get("protocol_parse_status"),
                "parse_reason": row.get("d3_source_ir_parse_reason"),
                "ir_hash": _sha256(row.get("d3_source_ir")) if row.get("d3_source_ir") else None,
            }
            for row in compiler_rows
        ],
        "resample_call_count": len(resample_rows),
        "direct_judge_selections": direct_selections,
        "pair_judge_selections": pair_selections,
        "decision": {
            "answer": final_answer,
            "override_accepted": override,
            "resolver": resolver,
            "candidate_completion": resolver == "d3_candidate_completion",
            "certificate_hash": certificate_hash,
            "risk_tier": risk_tier,
        },
        "first_failure_layer": first_failure_layer,
        "first_failure_reason": first_failure_reason,
    }
    return physical_rows, router, predictions


def verify_d4_semantic_compiler_consensus(
    *,
    sample: DatasetSample,
    decision,
    parsed_irs: list[tuple[int, Any]],
    required_compiler_count: int,
    fallback_solver: D4SolverResult,
) -> tuple[Any | None, D4SolverResult, tuple[str, ...], tuple[dict[str, Any], ...], bool, dict[str, str]]:
    """Require independent complete proof chains to agree on one answer.

    This intentionally does not compare SourceIR byte hashes: distinct valid
    representations are acceptable only when every compiler independently
    reaches a unique, reference-checked, metamorphic-passing answer and all
    canonical answers are identical.
    """

    verifications: list[dict[str, Any]] = []
    solvers_by_index: dict[int, D4SolverResult] = {}
    for compiler_index, item in parsed_irs:
        item_solver = d4_solve_source_ir(sample, decision, item)
        solvers_by_index[compiler_index] = item_solver
        item_metamorphic = run_metamorphic_checks(item, item_solver, sample=sample, decision=decision)
        item_passed = bool(
            item_solver.status == "UNIQUE"
            and bool(item_solver.canonical_answer)
            and str(item_solver.reference_checker_status).startswith("PASSED_")
            and item_solver.concrete_witness_status.get("status") == "PASSED"
            and metamorphic_checks_passed(item, item_solver, item_metamorphic)
        )
        verifications.append(
            {
                "compiler_index": compiler_index,
                "ir_hash": item.canonical_ir_hash,
                "solver_status": item_solver.status,
                "canonical_answer": item_solver.canonical_answer,
                "solver_trace": list(item_solver.solver_trace),
                "reference_checker_status": item_solver.reference_checker_status,
                "concrete_witness_status": dict(item_solver.concrete_witness_status),
                "metamorphic_status": item_metamorphic,
                "passed": item_passed,
            }
        )
    ordered = tuple(sorted(verifications, key=lambda item: int(item["compiler_index"])))
    vote_hashes = tuple(str(item["ir_hash"]) for item in ordered)
    agreement = bool(
        len(ordered) == required_compiler_count
        and all(bool(item["passed"]) for item in ordered)
        and len({str(item["canonical_answer"]) for item in ordered}) == 1
    )
    if not agreement:
        return (
            None,
            replace(fallback_solver, reason="compiler_ir_non_agreement_or_parse_failure"),
            vote_hashes,
            ordered,
            False,
            {},
        )
    representative_index = int(ordered[0]["compiler_index"])
    representative = next(item for index, item in parsed_irs if index == representative_index)
    return (
        representative,
        solvers_by_index[representative_index],
        vote_hashes,
        ordered,
        True,
        dict(ordered[0]["metamorphic_status"]),
    )


def run_catch_cert_v2_sample(
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
    """Execute frozen Cert-v2 or jurisdiction-aware CATCH-Kernel."""

    kernel_mode = protocol.protocol_version == "catch_kernel_v1"
    kernel_d2_mode = kernel_mode and str(getattr(experiment, "raw", {}).get("kernel_revision") or "") == "d2_unary_exact_v1"
    primary_method = "catch_kernel" if kernel_mode else "catch_cert_v2"
    designer_role = "kernel_obligation_filler" if kernel_mode else "certificate_designer_v2"
    verifier_role = "kernel_atomic_verifier" if kernel_mode else "certificate_verifier_v2"

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
            raise ValueError("Precomputed Stage-A rows do not match the CATCH-Cert v2 sample contract.")

    stage = build_stage_decision(stage_rows, seed=experiment.global_seed, sample_id=sample.sample_id)
    physical_rows: list[dict[str, Any]] = list(stage_rows)
    resample_rows: list[dict[str, Any]] = []
    designer_row: dict[str, Any] | None = None
    verifier_rows: list[dict[str, Any]] = []
    direct_rows: list[dict[str, Any]] = []
    pair_judge_rows: list[dict[str, Any]] = []
    direct_selections: list[str | None] = []
    pair_selections: list[str | None] = []
    panels: list[CertificateVerifierParseResultV2] = []
    source_graph = build_source_span_graph(sample)
    semantics = build_task_semantics(sample, source_graph) if kernel_mode else None
    contract = (
        task_contract_from_semantics(semantics, source_graph)
        if semantics is not None
        else build_task_contract_v2(sample, source_graph)
    )
    public_to_key = build_hypothesis_labels(stage, seed=experiment.global_seed, sample_id=sample.sample_id)
    key_to_public = {key: public for public, key in public_to_key.items()}
    answer_nodes = build_candidate_answer_nodes(sample, stage, public_to_key=public_to_key)
    pairs = build_all_candidate_pairs_v2(
        stage,
        public_to_key=public_to_key,
        seed=experiment.global_seed,
        sample_id=sample.sample_id,
    )
    reasoning_claims = _answer_connected_reasoning_claims(stage, public_to_key=public_to_key)
    validation = CertificateBankValidationV2((), (), (), None, (), (), 0.0, 0.0)
    adapter_results = {}
    typed_obligations = ()
    obligation_graph = None
    verifier_bindings = {}
    proof_results = ()
    unary_results = {}
    kernel_decision: KernelDecision | None = None
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
        skeleton = None
        if kernel_mode and semantics is not None:
            skeleton = compile_local_certificate_bank(
                sample=sample,
                semantics=semantics,
                stage=stage,
                public_to_key=public_to_key,
                answer_nodes=answer_nodes,
                source_graph=source_graph,
                pairs=pairs,
            )
            if not semantics_requires_designer(semantics) or skeleton.protocol_error is not None:
                validation = skeleton
        if not kernel_mode or (
            semantics is not None
            and semantics_requires_designer(semantics)
            and skeleton is not None
            and skeleton.protocol_error is None
        ):
            designer_messages = (
                build_kernel_designer_messages(
                    sample,
                    semantics=semantics,
                    answer_nodes=answer_nodes,
                    source_graph=source_graph,
                    skeleton=skeleton,
                    reasoning_claims=reasoning_claims,
                )
                if kernel_mode and semantics is not None and skeleton is not None
                else build_certificate_designer_messages_v2(
                    sample,
                    contract=contract,
                    answer_nodes=answer_nodes,
                    source_graph=source_graph,
                    pairs=pairs,
                    reasoning_claims=reasoning_claims,
                )
            )
            designer_row, designer_payload = _json_turn(
                sample,
                run_id=run_id,
                split_name=split_name,
                endpoint=endpoint,
                network_budget=network_budget,
                method_name=primary_method,
                role=designer_role,
                agent_id=1,
                seed=48_000,
                max_tokens=protocol.role_max_tokens,
                messages=designer_messages,
            )
            validation = (
                validate_kernel_certificate_bank(
                    designer_payload,
                    semantics=semantics,
                    skeleton=skeleton,
                    max_tests=max(1, int(protocol.max_selected_tests or 6)),
                )
                if kernel_mode and semantics is not None and skeleton is not None
                else validate_certificate_bank_v2(
                    designer_payload,
                    contract=contract,
                    stage=stage,
                    public_to_key=public_to_key,
                    answer_nodes=answer_nodes,
                    source_graph=source_graph,
                    pairs=pairs,
                    max_tests=max(1, int(protocol.max_selected_tests or 6)),
                )
            )
        if kernel_mode and semantics is not None:
            typed_obligations = compile_typed_obligations(semantics, validation)
            obligation_graph = compile_answer_obligation_graph(semantics, typed_obligations)
            verifier_bindings = bind_verifier_capabilities(semantics, validation)
            executable_tests = tuple(
                test
                for test in validation.tests
                if verifier_bindings.get(test.test_id) is not None
                and verifier_bindings[test.test_id].guarantee_level == "executable"
            )
            adapter_results = run_kernel_adapters(
                sample,
                contract=contract,
                tests=executable_tests,
                answer_nodes=answer_nodes,
                pairs=pairs,
            )
            if kernel_d2_mode:
                unary_results = run_kernel_unary_adapters(
                    sample,
                    tests=executable_tests,
                    answer_nodes=answer_nodes,
                )
        else:
            adapter_results = run_deterministic_adapters_v2(
                sample,
                contract=contract,
                tests=validation.tests,
                answer_nodes=answer_nodes,
                pairs=pairs,
            )
        if designer_row is not None:
            designer_row.update(
                {
                    "task_contract_v2": task_contract_v2_to_dict(contract),
                    "task_semantics": task_semantics_to_dict(semantics) if semantics is not None else None,
                    "source_span_graph": source_span_graph_to_dict(source_graph),
                    "candidate_answer_nodes": {
                        key: candidate_answer_node_to_dict(node) for key, node in answer_nodes.items()
                    },
                    "candidate_public_to_answer_class_key": public_to_key,
                    "public_pairs": [pair_v2_to_dict(pair) for pair in pairs],
                    "validated_certificates_v2": [certificate_v2_to_dict(item) for item in validation.certificates],
                    "validated_certificate_tests_v2": [certificate_test_v2_to_dict(item) for item in validation.tests],
                    "dropped_certificate_items": list(validation.dropped),
                    "certificate_protocol_error": validation.protocol_error,
                    "adapter_results": {key: adapter_result_to_dict(value) for key, value in adapter_results.items()},
                    "eligible_challengers": list(validation.eligible_challengers),
                    "obligation_coverage": validation.obligation_coverage,
                    "answer_link_coverage": validation.answer_link_coverage,
                    "typed_obligations": [typed_obligation_to_dict(item) for item in typed_obligations],
                    "answer_obligation_graph": (
                        answer_obligation_graph_to_dict(obligation_graph) if obligation_graph is not None else None
                    ),
                    "verifier_bindings": {
                        key: verifier_binding_to_dict(value) for key, value in verifier_bindings.items()
                    },
                }
            )
        if designer_row is not None and validation.protocol_error is not None:
            designer_row["protocol_parse_status"] = "failed"
            designer_row["protocol_parse_error"] = validation.protocol_error

        if validation.eligible_challengers:
            cert_by_public = {item.candidate_key_anon: item for item in validation.certificates}
            required_ids = {
                test_id
                for challenger in validation.eligible_challengers
                for test_id in cert_by_public[key_to_public[challenger]].required_test_ids
            }
            eligible_tests = tuple(test for test in validation.tests if test.test_id in required_ids)
            all_executable = bool(eligible_tests) and all(
                verifier_bindings.get(test.test_id) is not None
                and verifier_bindings[test.test_id].guarantee_level == "executable"
                and adapter_results.get(test.test_id) is not None
                and adapter_results[test.test_id].execution_status == "EXECUTED"
                for test in eligible_tests
            ) if kernel_mode else bool(eligible_tests) and all(
                adapter_results[test.test_id].execution_status == "EXECUTED" for test in eligible_tests
            )
            needs_model_panels = (
                any(
                    verifier_bindings.get(test.test_id) is not None
                    and verifier_bindings[test.test_id].guarantee_level == "bounded_semantic"
                    for test in eligible_tests
                )
                if kernel_mode
                else not all_executable
            )
            if needs_model_panels:
                for panel_index in range(1, protocol.witness_count + 1):
                    packet = build_certificate_verifier_packet_v2(
                        eligible_tests,
                        source_graph=source_graph,
                        seed=experiment.global_seed,
                        sample_id=sample.sample_id,
                        panel_index=panel_index,
                    )
                    row, payload = _json_turn(
                        sample,
                        run_id=run_id,
                        split_name=split_name,
                        endpoint=endpoint,
                        network_budget=network_budget,
                        method_name=primary_method,
                        role=verifier_role,
                        agent_id=panel_index,
                        seed=49_000 + panel_index,
                        max_tokens=protocol.role_max_tokens,
                        messages=(
                            build_kernel_verifier_messages(
                                sample,
                                semantics=semantics,
                                packet=packet,
                            )
                            if kernel_mode and semantics is not None
                            else build_certificate_verifier_messages_v2(sample, contract=contract, packet=packet)
                        ),
                    )
                    parsed = parse_certificate_verifier_v2(payload, packet=packet)
                    if not parsed.top_level_valid:
                        row["protocol_parse_status"] = "failed"
                        row["protocol_parse_error"] = (
                            "kernel_atomic_verifier_top_level_schema_failure"
                            if kernel_mode
                            else "certificate_verifier_v2_top_level_schema_failure"
                        )
                    row.update(
                        {
                            "certificate_verifier_packet_v2": {
                                "panel_index": packet.panel_index,
                                "role": packet.role,
                                "tests": list(packet.tests),
                                "source_spans": list(packet.source_spans),
                                "public_test_to_internal": packet.public_test_to_internal,
                                "public_outcome_to_internal": packet.public_outcome_to_internal,
                            },
                            "certificate_verifier_results_v2": {
                                key: verifier_result_v2_to_dict(value) for key, value in parsed.results.items()
                            },
                            "certificate_verifier_parse_diagnostics": {
                                "top_level_valid": parsed.top_level_valid,
                                "expected_test_count": parsed.expected_test_count,
                                "valid_test_count": parsed.valid_test_count,
                                "erased_rows": list(parsed.erased_rows),
                                "format_repair_count": parsed.format_repair_count,
                            },
                        }
                    )
                    verifier_rows.append(row)
                    panels.append(parsed)
            if kernel_mode and semantics is not None:
                proof_results = build_proof_results(
                    stage=stage,
                    semantics=semantics,
                    validation=validation,
                    public_to_key=public_to_key,
                    bindings=verifier_bindings,
                    adapter_results=adapter_results,
                    panels=tuple(panels),
                )
                kernel_decision = decide_with_proof_kernel(
                    stage,
                    semantics=semantics,
                    validation=validation,
                    public_to_key=public_to_key,
                    obligations=typed_obligations,
                    proofs=proof_results,
                ) if not kernel_d2_mode else decide_with_unary_proof_kernel(
                    stage,
                    semantics=semantics,
                    validation=validation,
                    public_to_key=public_to_key,
                    candidate_results=unary_results,
                )
                decision = kernel_decision_to_decode(stage, kernel_decision, public_to_key=public_to_key)
            else:
                decision = decode_certificates_v2(
                    stage,
                    validation=validation,
                    public_to_key=public_to_key,
                    panels=tuple(panels),
                    adapter_results=adapter_results,
                )
        else:
            resolver = "no_certificate" if not validation.tests else "certificate_invalid"
            decision = DecodeDecision(stage.anchor_answer, stage.anchor_key, False, resolver, (), ())

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
                    max_tokens=protocol.judge_max_tokens,
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
                    max_tokens=protocol.judge_max_tokens,
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
                *([designer_row] if designer_row is not None else []),
                *verifier_rows,
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
        candidate.key in target_keys and _score(sample, candidate.answer) == 1.0 for candidate in stage.candidates
    )
    gold_candidate_keys = tuple(
        candidate.key for candidate in stage.candidates if _score(sample, candidate.answer) == 1.0
    )
    gold_candidate_key = gold_candidate_keys[0] if gold_candidate_keys else None
    common_extra = {
        "target_oracle_correct": target_oracle,
        "gold_candidate_key": gold_candidate_key,
        "gold_candidate_keys": list(gold_candidate_keys),
        "target_candidate_count": len(target_keys),
        "protocol_version": protocol.protocol_version,
        "kernel_revision": str(getattr(experiment, "raw", {}).get("kernel_revision") or "d1_pairwise_v1")
        if kernel_mode
        else None,
        "task_family": contract.family,
        "query_operator": contract.query_operator,
        "adapter_kind": contract.adapter_kind,
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
            primary_method,
            decision.answer,
            stage.anchor_answer,
            stage,
            [*stage_rows, *([designer_row] if designer_row is not None else []), *verifier_rows],
            [*([designer_row] if designer_row is not None else []), *verifier_rows],
            decision.override_accepted,
            decision.resolver,
            candidate_oracle,
            extra={
                **common_extra,
                "certificate_count": len(validation.certificates),
                "certificate_test_count": len(validation.tests),
                "eligible_challenger_count": len(validation.eligible_challengers),
                "certificate_coverage": float(bool(validation.eligible_challengers)),
                "certificate_abstained": not decision.override_accepted,
                "adapter_conflict_count": int(decision.resolver == "adapter_conflict"),
                "answer_link_coverage": validation.answer_link_coverage,
                "obligation_coverage": validation.obligation_coverage,
                "adapter_executed_test_count": sum(
                    item.execution_status == "EXECUTED" for item in adapter_results.values()
                ),
                "verifier_format_repair_count": sum(panel.format_repair_count for panel in panels),
                "syntax_validity": float(
                    (designer_row is None or designer_row.get("protocol_parse_status") == "ok")
                    and all(panel.top_level_valid for panel in panels)
                ),
                "schema_validity": float(
                    validation.protocol_error is None and all(panel.top_level_valid for panel in panels)
                ),
                "typed_compilation_validity": float(
                    (bool(typed_obligations) or not stage.triggered)
                    if kernel_mode
                    else validation.protocol_error is None
                ),
                "semantic_validity": None if kernel_mode else float(validation.protocol_error is None),
                "contract_accuracy": None if kernel_mode else float(validation.protocol_error is None),
                "verifier_jurisdiction_coverage": (
                    sum(item.binding_status == "BOUND" for item in verifier_bindings.values()) / len(verifier_bindings)
                    if verifier_bindings
                    else float(not stage.triggered)
                ),
                "proof_completeness": (
                    sum(
                        item.status == "PASS"
                        and item.provenance_valid
                        and item.entailment_valid
                        and item.obligation_valid
                        and item.sufficiency_valid
                        for item in proof_results
                    )
                    / len(proof_results)
                    if proof_results
                    else float(not stage.triggered)
                ),
                "structural_obligation_completeness": (
                    sum(item.obligation_valid and item.sufficiency_valid for item in proof_results) / len(proof_results)
                    if proof_results
                    else float(not stage.triggered)
                ),
                "provenance_validity": (
                    sum(item.provenance_valid for item in proof_results) / len(proof_results)
                    if proof_results
                    else float(not stage.triggered)
                ),
                "entailment_validity": (
                    sum(item.entailment_valid for item in proof_results) / len(proof_results)
                    if proof_results
                    else float(not stage.triggered)
                ),
                "proof_pass_count": sum(item.status == "PASS" for item in proof_results),
                "proof_conflict_count": sum(item.status == "CONFLICT" for item in proof_results),
                "proof_unsupported_count": sum(item.status == "UNSUPPORTED" for item in proof_results),
                "proof_unknown_count": sum(item.status == "UNKNOWN" for item in proof_results),
                "panel_disagreement_count": sum(item.detail == "panel_disagreement" for item in proof_results),
                "adapter_conflict_test_count": sum(
                    item.execution_status == "CONFLICT" for item in adapter_results.values()
                ),
                "adapter_unsupported_test_count": sum(
                    item.execution_status == "UNSUPPORTED" for item in adapter_results.values()
                ),
                "adapter_invalid_test_count": sum(
                    item.execution_status == "INVALID" for item in adapter_results.values()
                ),
                "kernel_failure_layer": kernel_decision.failure_layer if kernel_decision else None,
                "unary_valid_candidate_count": sum(item.status == "VALID" for item in unary_results.values()),
                "unary_invalid_candidate_count": sum(item.status == "INVALID" for item in unary_results.values()),
                "unary_unsupported_candidate_count": sum(item.status == "UNSUPPORTED" for item in unary_results.values()),
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
        "protocol_version": protocol.protocol_version,
        "kernel_revision": str(getattr(experiment, "raw", {}).get("kernel_revision") or "d1_pairwise_v1")
        if kernel_mode
        else None,
        "triggered": stage.triggered,
        "valid_stage_answers": stage.valid_count,
        "anchor_answer": stage.anchor_answer,
        "anchor_key": stage.anchor_key,
        "answer_class_vote_counts": stage.vote_counts,
        "candidate_count": len(stage.candidates),
        "candidate_oracle_correct": candidate_oracle,
        "target_oracle_correct": target_oracle,
        "gold_candidate_key": gold_candidate_key,
        "gold_candidate_keys": list(gold_candidate_keys),
        "task_contract_v2": task_contract_v2_to_dict(contract),
        "task_semantics": task_semantics_to_dict(semantics) if semantics is not None else None,
        "source_span_graph": source_span_graph_to_dict(source_graph),
        "candidate_answer_nodes": {key: candidate_answer_node_to_dict(node) for key, node in answer_nodes.items()},
        "candidate_public_to_answer_class_key": public_to_key,
        "public_pairs": [pair_v2_to_dict(pair) for pair in pairs],
        "certificate_protocol_error": validation.protocol_error,
        "dropped_certificate_items": list(validation.dropped),
        "certificates": [certificate_v2_to_dict(item) for item in validation.certificates],
        "certificate_tests": [certificate_test_v2_to_dict(item) for item in validation.tests],
        "adapter_results": {key: adapter_result_to_dict(value) for key, value in adapter_results.items()},
        "unary_adapter_results": {key: candidate_adapter_result_to_dict(value) for key, value in unary_results.items()},
        "eligible_challengers": list(validation.eligible_challengers),
        "answer_link_coverage": validation.answer_link_coverage,
        "obligation_coverage": validation.obligation_coverage,
        "typed_obligations": [typed_obligation_to_dict(item) for item in typed_obligations],
        "answer_obligation_graph": (
            answer_obligation_graph_to_dict(obligation_graph) if obligation_graph is not None else None
        ),
        "verifier_bindings": {key: verifier_binding_to_dict(value) for key, value in verifier_bindings.items()},
        "proof_results": [proof_result_to_dict(item) for item in proof_results],
        "kernel_decision": kernel_decision_to_dict(kernel_decision) if kernel_decision else None,
        "verifier_panels": [
            {
                "top_level_valid": panel.top_level_valid,
                "results": {key: verifier_result_v2_to_dict(value) for key, value in panel.results.items()},
                "valid_test_count": panel.valid_test_count,
                "expected_test_count": panel.expected_test_count,
                "format_repair_count": panel.format_repair_count,
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


def _answer_connected_reasoning_claims(
    stage: StageDecision,
    *,
    public_to_key: dict[str, str],
    max_claims: int = 12,
    max_characters: int = 4_096,
) -> dict[str, tuple[str, ...]]:
    """Keep a bounded suffix of reasoning while the answer node carries semantics."""

    candidate_by_key = {candidate.key: candidate for candidate in stage.candidates}
    claims: dict[str, tuple[str, ...]] = {}
    for public, key in public_to_key.items():
        reasoning = str(candidate_by_key[key].representative_reasoning or "")[-max_characters:]
        sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+|\r?\n+", reasoning) if item.strip()]
        claims[public] = tuple(sentences[-max_claims:])
    return claims


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
    output_protocol: str = FREE_TEXT_ANSWER_PROTOCOL_V1,
    prompt_version: str = "single_agent_free_text_v1",
) -> dict[str, Any]:
    completed = _lookup_d4_completion(
        endpoint,
        sample=sample,
        method_name=method_name,
        role=role,
        agent_id=agent_id,
        seed=seed,
    )
    if completed is not None:
        return completed
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
            messages=build_cot_messages(sample, agent_id, prompt_version),
            temperature=0.7,
            top_p=1.0,
            seed=seed,
            dataset=sample.dataset,
            role=role,
            output_protocol=output_protocol,
            max_tokens=max_tokens,
            delete_cache_on_protocol_failure=getattr(endpoint, "completion_ledger", None) is not None,
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
        else canonical.key
        if canonical is not None and canonical.valid
        else ""
    )
    invalid_reason = (
        str(result.validated_output.get("canonical_invalid_reason") or "invalid_sample_answer_output")
        if parsed_valid is False
        else canonical.invalid_reason
        if canonical is not None
        else "request_or_protocol_failure"
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
    _record_d4_completion(endpoint, row)
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
    completed = _lookup_d4_completion(
        endpoint,
        sample=sample,
        method_name=method_name,
        role=role,
        agent_id=agent_id,
        seed=seed,
    )
    if completed is not None:
        parsed = None
        if completed.get("request_error") is None and completed.get("protocol_parse_status") == "ok":
            try:
                candidate = json.loads(
                    str(completed.get("assistant_text") or ""),
                    object_pairs_hook=_reject_duplicate_json_keys,
                )
                if isinstance(candidate, dict):
                    parsed = candidate
            except (TypeError, ValueError, json.JSONDecodeError):
                parsed = None
        return completed, parsed
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
            response_validator=_admit_json_object_response,
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
            candidate = json.loads(
                str(request.response_payload.get("assistant_text") or ""),
                object_pairs_hook=_reject_duplicate_json_keys,
            )
            if not isinstance(candidate, dict):
                raise ValueError("JSON output must be an object")
            parsed = candidate
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            parse_error = str(exc)
    if parsed is None and not request.request_error:
        cache.delete(request.cache_key)
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
    if not (getattr(endpoint, "completion_ledger", None) is not None and role == "d4_source_compiler"):
        _record_d4_completion(endpoint, row)
    return row, parsed


def _admit_json_object_response(response: dict[str, Any]) -> dict[str, Any]:
    candidate = json.loads(
        str(response.get("assistant_text") or ""),
        object_pairs_hook=_reject_duplicate_json_keys,
    )
    if not isinstance(candidate, dict):
        raise ValueError("JSON output must be an object")
    return candidate


def _lookup_d4_completion(
    endpoint,
    *,
    sample: DatasetSample,
    method_name: str,
    role: str,
    agent_id: int,
    seed: int,
) -> dict[str, Any] | None:
    ledger = getattr(endpoint, "completion_ledger", None)
    if ledger is None:
        return None
    return ledger.lookup(
        sample_id=sample.sample_id,
        method_name=method_name,
        role=role,
        agent_id=agent_id,
        seed=seed,
    )


def _record_d4_completion(endpoint, row: dict[str, Any]) -> None:
    ledger = getattr(endpoint, "completion_ledger", None)
    if ledger is not None:
        ledger.record(row)


def _reject_duplicate_json_keys(items: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in items:
        if key in output:
            raise ValueError(f"duplicate_json_key:{key}")
        output[key] = value
    return output


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
        "high_level_domain": sample.metadata.get("high_level_domain"),
        "domain": sample.metadata.get("domain"),
        "field": sample.metadata.get("field"),
        "discipline": sample.metadata.get("discipline"),
        "subdomain": sample.metadata.get("subdomain"),
        "reasoning_type": (
            sample.metadata.get("reasoning_type")
            or sample.metadata.get("quantitative_conceptual")
        ),
        "method_name": method_name,
        "role": role,
        "agent_id": agent_id,
        "request_seed": seed,
        "payload": payload,
        "cache_key": cache_key,
        "request_completion_cap": next(
            (
                int(payload[field])
                for field in ("max_completion_tokens", "max_tokens")
                if isinstance(payload.get(field), int) and int(payload[field]) > 0
            ),
            None,
        ),
        "cache_origin_completion_cap": response.get("cache_origin_completion_cap"),
        "cache_key_policy": response.get("cache_origin_key_policy"),
        "cache_policy": "global_validated_response_v3",
        "request_source": "global_cache_pending",
        "prompt_hash": _sha256(json.dumps(payload.get("messages") or [], ensure_ascii=False, sort_keys=True)),
        "cache_hit": cache_hit,
        "request_error": request_error,
        "request_status": "request_fail" if request_error else "ok",
        "raw_finish_reason": finish_reason,
        "provider_request_id": response.get("provider_request_id"),
        "response_id": response.get("response_id"),
        "attempt_timeline": [] if cache_hit else list(response.get("attempt_timeline") or []),
        "cached_response_origin_attempt_timeline": (list(response.get("attempt_timeline") or []) if cache_hit else []),
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
        "high_level_domain": sample.metadata.get("high_level_domain"),
        "domain": sample.metadata.get("domain"),
        "field": sample.metadata.get("field"),
        "discipline": sample.metadata.get("discipline"),
        "subdomain": sample.metadata.get("subdomain"),
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
        "total_tokens_per_question": sum(
            float(row.get("actual_total_tokens") or row.get("total_tokens") or 0) for row in logical_rows
        ),
        "completion_tokens_per_question": sum(
            float(row.get("actual_completion_tokens") or row.get("completion_tokens") or 0) for row in logical_rows
        ),
        "network_attempts_per_question": sum(int(row.get("network_attempt_count") or 0) for row in logical_rows),
        "latency_ms_per_question": sum(float(row.get("latency_ms") or 0) for row in logical_rows),
        "cache_hits_per_question": sum(bool(row.get("cache_hit")) for row in logical_rows),
        "network_calls_per_question": sum(int(row.get("network_request_count") or 0) for row in logical_rows),
        "intervention_call_budget_per_question": intervention_call_budget,
        "intervention_calls_per_question": actual_intervention_calls,
        "actual_intervention_calls_per_question": actual_intervention_calls,
    }
    if sample.dataset == "seqbench":
        sequence_metrics = evaluate_seqbench_prediction(prediction, sample.reference_answer, sample=sample)
        plan_validation = validate_seqbench_plan(prediction, sample=sample)
        payload.update(
            {
                "seqbench_exact_match": sequence_metrics.exact_match,
                "seqbench_progress_ratio": sequence_metrics.progress_ratio,
                "seqbench_precision": sequence_metrics.precision,
                "seqbench_recall": sequence_metrics.recall,
                "seqbench_valid_action_rate": sequence_metrics.valid_action_rate,
                "seqbench_execution_prefix_ratio": sequence_metrics.execution_prefix_ratio,
                "seqbench_first_invalid_action_index": sequence_metrics.first_invalid_action_index,
                "seqbench_first_invalid_action_reason": sequence_metrics.first_invalid_action_reason,
                "seqbench_predicted_action_count": sequence_metrics.predicted_action_count,
                "seqbench_gold_action_count": sequence_metrics.gold_action_count,
                "seqbench_completion_validity": float(plan_validation.complete),
                "seqbench_completion_failure": plan_validation.first_failure,
            }
        )
    payload.update(extra or {})
    return payload


def _score(sample: DatasetSample, answer: str) -> float:
    return score_prediction(sample.dataset, answer, sample.reference_answer, sample=sample) if answer else 0.0


def _d3_primary_metric(
    dataset: str,
    split_name: str,
    *,
    phase_name: str | None = None,
    sample_limit: int | None = None,
) -> str:
    if dataset in {"bbeh", "bbeh_extension"}:
        if split_name == "bbeh_mini460_seed42":
            return "bbeh_mini_micro"
        if split_name == "full4520_seed42" and int(sample_limit or 0) >= 4520:
            return "bbeh_full_adjusted_harmonic"
        return "bbeh_task_stratified_micro"
    if dataset in {"musr", "musr_x"}:
        return "musr_task_macro"
    if dataset == "gpqa_diamond":
        return "gpqa_accuracy"
    return "accuracy"


def _d3_first_failure(
    *,
    route: str,
    method_rows: list[dict[str, Any]],
    source_ir: SourceIR | None,
    certificate: SolverCertificate | None,
) -> tuple[str, str]:
    for row in method_rows:
        if row.get("request_error"):
            return "REQUEST", str(row["request_error"])
    for row in method_rows:
        if row.get("protocol_parse_status") == "failed":
            return "PARSE", str(
                row.get("d3_source_ir_parse_reason")
                or row.get("protocol_parse_error")
                or "structured_output_parse_failed"
            )
    if route == "SOFT_UNSUPPORTED":
        return "JURISDICTION_BOUNDARY", "no_frozen_executable_capability"
    if source_ir is None:
        return "SOURCE_IR_AGREEMENT", (
            certificate.reason if certificate is not None else "source_ir_unavailable"
        )
    if certificate is None:
        return "SOLVER", "solver_certificate_missing"
    if certificate.status != "UNIQUE":
        return "SOLVER", f"{certificate.status}:{certificate.reason}"
    return "NONE", "none"


def _cache_for_role(endpoint, role: str):
    if hasattr(endpoint, "cache_for_role"):
        return endpoint.cache_for_role(role)
    return endpoint.cache


def _raise_if_cancelled(endpoint) -> None:
    event = getattr(endpoint, "stop_event", None)
    if event is not None and event.is_set():
        raise CatchRunCancelled("CATCH run cancelled after a sibling sample failed")


def _annotate_cache_audit(row: dict[str, Any], *, endpoint, cache, cache_key: str) -> None:
    del endpoint, cache, cache_key
    row["cache_policy"] = "global_validated_response_v3"
    row["request_source"] = "global_cache" if row.get("cache_hit") else "network"


def _sha256(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()
