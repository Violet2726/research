"""CATCH-Cert v2：答案连接且覆盖全局义务的证书协议。

This module intentionally lives beside :mod:`certificates` instead of
replacing it.  The v1 artifact protocol is frozen and must remain replayable.
All builders in this module are gold-free: they may expose the semantic
content of an existing candidate answer, but never the reference answer,
candidate vote counts, or a hidden correctness label.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any, Literal

from research_experiments.core.data.datasets import DatasetSample, question_without_answer_contract
from research_experiments.core.data.evaluation import canonicalize_answer, validate_seqbench_plan
from research_experiments.families.contrastive_active_testing.algorithms import DecodeDecision, StageDecision

TaskFamilyV2 = Literal["proof_state", "state_transition", "equation", "set_count", "semantic"]
QueryOperator = Literal[
    "point_value",
    "earliest",
    "latest",
    "argmax",
    "exact_set",
    "exact_sequence",
    "final_state",
    "three_way_entailment",
    "multiple_choice_truth",
]
ObligationKind = Literal[
    "candidate_validity",
    "prefix_validity",
    "final_state",
    "global_completeness",
    "comparative_dominance",
    "unit_consistency",
    "constraint_satisfaction",
]
AnswerType = Literal["choice", "scalar", "set", "sequence", "entity", "proposition"]
SupportStatusV2 = Literal["ENTAILED", "CONTRADICTED", "UNDERDETERMINED"]
ExecutionStatus = Literal["EXECUTED", "UNSUPPORTED", "INVALID", "CONFLICT"]


@dataclass(frozen=True)
class CandidateAnswerNode:
    candidate_key_anon: str
    answer_type: AnswerType
    canonical_value: str
    rendered_content: str
    answer_hash: str


@dataclass(frozen=True)
class SourceSpan:
    span_id: str
    start: int
    end: int
    text: str
    sha256: str


@dataclass(frozen=True)
class SourceSpanGraph:
    source_hash: str
    spans: tuple[SourceSpan, ...]


@dataclass(frozen=True)
class QuestionObligation:
    obligation_id: str
    kind: ObligationKind
    scope: str
    required: bool
    source_span_ids: tuple[str, ...]


@dataclass(frozen=True)
class TaskContractV2:
    family: TaskFamilyV2
    query_operator: QueryOperator
    adapter_kind: str
    answer_schema: str
    mandatory_obligations: tuple[QuestionObligation, ...]
    tolerance_policy: dict[str, float]
    adapter_version: str = "catch_cert_task_contract_v2"


@dataclass(frozen=True)
class CandidatePairV2:
    pair_id: str
    anchor_key: str
    challenger_key: str
    left_candidate: str
    right_candidate: str


@dataclass(frozen=True)
class CertificateOutcomeV2:
    outcome_id: str
    text: str


@dataclass(frozen=True)
class CertificateTestV2:
    test_id: str
    pair_id: str
    obligation_ids: tuple[str, ...]
    operation_kind: str
    question_or_operation: str
    finite_outcomes: tuple[CertificateOutcomeV2, ...]
    expected_outcome_by_candidate: dict[str, str]
    source_span_ids: tuple[str, ...]
    deterministic_payload: dict[str, Any]


@dataclass(frozen=True)
class AnswerCertificateV2:
    candidate_key_anon: str
    answer_hash: str
    required_test_ids: tuple[str, ...]
    derived_refutation_test_ids: tuple[str, ...]
    certificate_hash: str


@dataclass(frozen=True)
class CertificateBankValidationV2:
    certificates: tuple[AnswerCertificateV2, ...]
    tests: tuple[CertificateTestV2, ...]
    dropped: tuple[dict[str, str], ...]
    protocol_error: str | None
    adapter_conflicts: tuple[dict[str, str], ...]
    eligible_challengers: tuple[str, ...]
    obligation_coverage: float
    answer_link_coverage: float


@dataclass(frozen=True)
class CertificateVerifierPacketV2:
    panel_index: int
    role: str
    tests: tuple[dict[str, Any], ...]
    source_spans: tuple[dict[str, str], ...]
    public_test_to_internal: dict[str, str]
    public_outcome_to_internal: dict[str, dict[str, str]]


@dataclass(frozen=True)
class VerifierResultV2:
    test_id: str
    observed_outcome: str
    support_status: SupportStatusV2
    source_span_ids: tuple[str, ...]
    ruled_out_outcomes: tuple[str, ...]
    parse_status: str
    format_repaired: bool = False


@dataclass(frozen=True)
class CertificateVerifierParseResultV2:
    top_level_valid: bool
    results: dict[str, VerifierResultV2]
    expected_test_count: int
    valid_test_count: int
    erased_rows: tuple[dict[str, str], ...]
    format_repair_count: int


@dataclass(frozen=True)
class AdapterResult:
    test_id: str
    observed_outcome: str | None
    execution_status: ExecutionStatus
    residual_or_first_failure: str
    execution_trace_hash: str


_PROOF_TASKS = {
    "boardgame_qa",
    "boolean_expressions",
    "causal_understanding",
    "formal_fallacies_syllogisms_negation",
    "web_of_lies",
}
_ARGMAX_TASKS = {"movie_recommendation", "murder_mysteries"}
_CONSTRAINT_TASKS = {"zebra_puzzles", "team_allocation"}


def build_source_span_graph(sample: DatasetSample, *, max_span_length: int = 512) -> SourceSpanGraph:
    """Index exact source-task spans so verifier provenance is checkable."""

    source = question_without_answer_contract(sample)
    ranges = _source_ranges(source, max_span_length=max_span_length)
    spans = tuple(
        SourceSpan(f"S{index}", start, end, source[start:end], _sha256(source[start:end]))
        for index, (start, end) in enumerate(ranges)
        if source[start:end].strip()
    )
    return SourceSpanGraph(_sha256(source), spans)


def build_task_contract_v2(sample: DatasetSample, source_graph: SourceSpanGraph) -> TaskContractV2:
    """Build a gold-free task contract from dataset and task metadata."""

    task = str(sample.metadata.get("task") or sample.metadata.get("domain") or "").strip().casefold()
    domain = str(sample.metadata.get("high_level_domain") or "").strip().casefold()
    span_ids = tuple(span.span_id for span in source_graph.spans)

    family: TaskFamilyV2 = "semantic"
    operator: QueryOperator = "point_value"
    adapter_kind = "semantic"
    answer_schema = "free_text"
    kinds: tuple[ObligationKind, ...] = ("candidate_validity",)

    if sample.dataset == "seqbench":
        family = "state_transition"
        operator = "exact_sequence"
        adapter_kind = "seq_plan"
        answer_schema = "ordered_action_sequence"
        kinds = ("candidate_validity", "final_state", "global_completeness")
    elif task == "dyck_languages":
        family = "state_transition"
        operator = "earliest"
        adapter_kind = "stack_trace"
        answer_schema = "first_error_index"
        kinds = ("candidate_validity", "prefix_validity")
    elif task == "spatial_reasoning":
        family = "state_transition"
        operator = "final_state"
        adapter_kind = "grid_path"
        answer_schema = "entity_at_final_coordinate"
        kinds = ("candidate_validity", "final_state")
    elif task == "shuffled_objects":
        family = "state_transition"
        operator = "final_state"
        adapter_kind = "permutation"
        answer_schema = "final_permutation"
        kinds = ("candidate_validity", "final_state")
    elif task == "word_sorting":
        family = "set_count"
        operator = "exact_sequence"
        adapter_kind = "sort_order"
        answer_schema = "ordered_tokens"
        kinds = ("candidate_validity", "global_completeness")
    elif task in {"multistep_arithmetic", "time_arithmetic", "temporal_sequence"}:
        family = "equation"
        operator = "point_value"
        adapter_kind = "arithmetic_dsl"
        answer_schema = "scalar_or_tuple"
        kinds = ("candidate_validity", "unit_consistency")
    elif task == "object_placements":
        family = "state_transition"
        operator = "final_state"
        adapter_kind = "object_state"
        answer_schema = "entity_location"
        kinds = ("candidate_validity", "final_state")
    elif task in _CONSTRAINT_TASKS:
        family = "set_count"
        operator = "exact_set"
        adapter_kind = "constraint_witness"
        answer_schema = "constraint_assignment_or_choice"
        kinds = ("candidate_validity", "constraint_satisfaction", "global_completeness")
    elif task in _PROOF_TASKS:
        family = "proof_state"
        operator = "three_way_entailment"
        adapter_kind = "proof_state"
        answer_schema = "entailed_contradicted_unknown"
        kinds = ("candidate_validity", "constraint_satisfaction")
    elif task in _ARGMAX_TASKS:
        family = "semantic"
        operator = "argmax"
        adapter_kind = "semantic_comparative"
        answer_schema = "best_candidate_entity"
        kinds = ("candidate_validity", "comparative_dominance")
    elif sample.dataset == "gpqa_diamond":
        operator = "multiple_choice_truth"
        answer_schema = "scientific_option_proposition"
        kinds = ("candidate_validity",)
        if domain == "physics":
            family = "equation"
            adapter_kind = "gpqa_physics"
            kinds = ("candidate_validity", "unit_consistency")
        elif domain == "chemistry":
            adapter_kind = "gpqa_chemistry"
            kinds = ("candidate_validity", "constraint_satisfaction")
        elif domain == "biology":
            adapter_kind = "gpqa_biology"
    elif _has_choice_contract(sample):
        operator = "multiple_choice_truth"
        answer_schema = "option_proposition"
    elif task == "sarc_triples":
        operator = "exact_sequence"
        answer_schema = "ordered_labels"
        kinds = ("candidate_validity", "global_completeness")

    obligations = tuple(
        QuestionObligation(
            obligation_id=f"Q{index}",
            kind=kind,
            scope=_obligation_scope(kind, operator),
            required=True,
            source_span_ids=span_ids,
        )
        for index, kind in enumerate(kinds)
    )
    return TaskContractV2(
        family=family,
        query_operator=operator,
        adapter_kind=adapter_kind,
        answer_schema=answer_schema,
        mandatory_obligations=obligations,
        tolerance_policy={"absolute": 1e-9, "relative": 1e-6},
    )


def build_candidate_answer_nodes(
    sample: DatasetSample,
    stage: StageDecision,
    *,
    public_to_key: dict[str, str],
) -> dict[str, CandidateAnswerNode]:
    """Expose the meaning of existing answers without exposing correctness."""

    candidate_by_key = {candidate.key: candidate for candidate in stage.candidates}
    options = _option_text_by_label(sample)
    nodes: dict[str, CandidateAnswerNode] = {}
    for public_id, key in public_to_key.items():
        candidate = candidate_by_key[key]
        canonical = canonicalize_answer(sample, candidate.answer)
        canonical_value = canonical.key if canonical.valid else candidate.key
        rendered = _render_candidate_answer(candidate.answer, canonical_value, options)
        answer_type = _infer_answer_type(sample, canonical_value, rendered)
        canonical_payload = {
            "candidate_key_anon": public_id,
            "answer_type": answer_type,
            "canonical_value": canonical_value,
            "rendered_content": rendered,
        }
        nodes[public_id] = CandidateAnswerNode(
            **canonical_payload,
            answer_hash=_json_sha256(canonical_payload),
        )
    return nodes


def build_all_candidate_pairs_v2(
    stage: StageDecision,
    *,
    public_to_key: dict[str, str],
    seed: int,
    sample_id: str,
    max_challengers: int = 4,
) -> tuple[CandidatePairV2, ...]:
    """Pair the plurality answer with every existing challenger, up to Stage-A's maximum."""

    key_to_public = {key: public for public, key in public_to_key.items()}
    challengers = [candidate.key for candidate in stage.candidates if candidate.key != stage.anchor_key]
    pairs: list[CandidatePairV2] = []
    for index, challenger in enumerate(challengers[:max_challengers]):
        values = [key_to_public[stage.anchor_key], key_to_public[challenger]]
        random.Random(_stable_hash(seed, sample_id, f"cert-v2-pair:{index}")).shuffle(values)
        pairs.append(
            CandidatePairV2(
                pair_id=f"P{index}",
                anchor_key=stage.anchor_key,
                challenger_key=challenger,
                left_candidate=values[0],
                right_candidate=values[1],
            )
        )
    return tuple(pairs)


def validate_certificate_bank_v2(
    payload: dict[str, Any] | None,
    *,
    contract: TaskContractV2,
    stage: StageDecision,
    public_to_key: dict[str, str],
    answer_nodes: dict[str, CandidateAnswerNode],
    source_graph: SourceSpanGraph,
    pairs: tuple[CandidatePairV2, ...],
    max_tests: int = 6,
) -> CertificateBankValidationV2:
    """Compile designer output into answer-linked certificates without gold."""

    if not isinstance(payload, dict) or set(payload) != {"certificates", "tests"}:
        return _empty_validation("certificate_v2_top_level_schema_failure")
    raw_tests = payload.get("tests")
    raw_certificates = payload.get("certificates")
    if not isinstance(raw_tests, list) or not isinstance(raw_certificates, list):
        return _empty_validation("certificates_and_tests_must_be_lists")
    if not 1 <= len(raw_tests) <= max_tests:
        return _empty_validation("certificate_test_count_invalid")

    pair_by_id = {pair.pair_id: pair for pair in pairs}
    source_ids = {span.span_id for span in source_graph.spans}
    obligation_ids = {item.obligation_id for item in contract.mandatory_obligations}
    tests: list[CertificateTestV2] = []
    dropped: list[dict[str, str]] = []
    seen_tests: set[str] = set()
    for index, raw in enumerate(raw_tests):
        reason, test = _validate_test_v2(
            raw,
            index=index,
            pair_by_id=pair_by_id,
            source_ids=source_ids,
            obligation_ids=obligation_ids,
            seen_tests=seen_tests,
        )
        if reason is not None or test is None:
            dropped.append({"item": f"test:{_raw_id(raw, index)}", "reason": reason or "invalid_test"})
            continue
        seen_tests.add(test.test_id)
        tests.append(test)

    tests_by_id = {test.test_id: test for test in tests}
    public_by_key = {key: public for public, key in public_to_key.items()}
    challenger_public = {public_by_key[pair.challenger_key] for pair in pairs}
    certificates: list[AnswerCertificateV2] = []
    seen_candidates: set[str] = set()
    for index, raw in enumerate(raw_certificates):
        reason, certificate = _compile_certificate_v2(
            raw,
            index=index,
            answer_nodes=answer_nodes,
            tests_by_id=tests_by_id,
            pairs=pairs,
            required_obligations=obligation_ids,
        )
        if reason is not None or certificate is None:
            dropped.append({"item": f"certificate:{_raw_id(raw, index)}", "reason": reason or "invalid_certificate"})
            continue
        if certificate.candidate_key_anon in seen_candidates:
            dropped.append(
                {"item": f"certificate:{certificate.candidate_key_anon}", "reason": "duplicate_candidate_certificate"}
            )
            continue
        seen_candidates.add(certificate.candidate_key_anon)
        certificates.append(certificate)

    cert_by_public = {item.candidate_key_anon: item for item in certificates}
    eligible = tuple(pair.challenger_key for pair in pairs if public_by_key[pair.challenger_key] in cert_by_public)
    covered_obligations = {
        obligation
        for certificate in certificates
        if certificate.candidate_key_anon in challenger_public
        for test_id in certificate.required_test_ids
        for obligation in tests_by_id[test_id].obligation_ids
    }
    obligation_coverage = len(covered_obligations & obligation_ids) / len(obligation_ids) if obligation_ids else 1.0
    answer_link_coverage = (
        sum(
            cert_by_public.get(public) is not None
            and cert_by_public[public].answer_hash == answer_nodes[public].answer_hash
            for public in challenger_public
        )
        / len(challenger_public)
        if challenger_public
        else 1.0
    )
    protocol_error = None
    if not tests:
        protocol_error = "no_certificate_tests"
    elif not certificates:
        protocol_error = "no_valid_certificates"
    elif not eligible:
        protocol_error = "no_eligible_challenger_certificate"
    return CertificateBankValidationV2(
        certificates=tuple(certificates),
        tests=tuple(tests),
        dropped=tuple(dropped),
        protocol_error=protocol_error,
        adapter_conflicts=(),
        eligible_challengers=eligible,
        obligation_coverage=obligation_coverage,
        answer_link_coverage=answer_link_coverage,
    )


def build_certificate_verifier_packet_v2(
    tests: tuple[CertificateTestV2, ...],
    *,
    source_graph: SourceSpanGraph,
    seed: int,
    sample_id: str,
    panel_index: int,
) -> CertificateVerifierPacketV2:
    rng = random.Random(_stable_hash(seed, sample_id, f"certificate-v2-verifier:{panel_index}"))
    ordered = list(tests)
    rng.shuffle(ordered)
    rendered: list[dict[str, Any]] = []
    public_test_to_internal: dict[str, str] = {}
    public_outcome_to_internal: dict[str, dict[str, str]] = {}
    for index, test in enumerate(ordered):
        public_test = f"Q{index}"
        outcomes = list(test.finite_outcomes)
        rng.shuffle(outcomes)
        outcome_map: dict[str, str] = {}
        rendered_outcomes: list[dict[str, str]] = []
        for outcome_index, outcome in enumerate(outcomes):
            public_outcome = f"R{outcome_index}"
            outcome_map[public_outcome] = outcome.outcome_id
            rendered_outcomes.append({"outcome_id": public_outcome, "text": outcome.text})
        rendered.append(
            {
                "test_id": public_test,
                "question_or_operation": test.question_or_operation,
                "finite_outcomes": rendered_outcomes,
                "obligation_ids": list(test.obligation_ids),
                "operation_kind": test.operation_kind,
            }
        )
        public_test_to_internal[public_test] = test.test_id
        public_outcome_to_internal[public_test] = outcome_map
    return CertificateVerifierPacketV2(
        panel_index=panel_index,
        role="support_auditor" if panel_index == 1 else "incompatibility_auditor",
        tests=tuple(rendered),
        source_spans=tuple({"span_id": span.span_id, "text": span.text} for span in source_graph.spans),
        public_test_to_internal=public_test_to_internal,
        public_outcome_to_internal=public_outcome_to_internal,
    )


def parse_certificate_verifier_v2(
    payload: dict[str, Any] | None,
    *,
    packet: CertificateVerifierPacketV2,
) -> CertificateVerifierParseResultV2:
    expected = len(packet.public_test_to_internal)
    if not isinstance(payload, dict) or set(payload) != {"results"} or not isinstance(payload.get("results"), list):
        return CertificateVerifierParseResultV2(False, {}, expected, 0, (), 0)
    source_ids = {item["span_id"] for item in packet.source_spans}
    raw_by_test: dict[str, list[dict[str, Any]]] = {}
    erased: list[dict[str, str]] = []
    for index, raw in enumerate(payload["results"]):
        if not isinstance(raw, dict) or set(raw) != {
            "test_id",
            "observed_outcome",
            "support_status",
            "source_span_ids",
            "ruled_out_outcomes",
        }:
            erased.append({"row": str(index), "reason": "invalid_verifier_v2_row_schema"})
            continue
        public_test = str(raw.get("test_id") or "")
        if public_test not in packet.public_test_to_internal:
            erased.append({"row": str(index), "reason": "unknown_public_test_id"})
            continue
        raw_by_test.setdefault(public_test, []).append(raw)

    results: dict[str, VerifierResultV2] = {}
    repair_count = 0
    statuses = ("ENTAILED", "CONTRADICTED", "UNDERDETERMINED")
    for public_test, internal_test in packet.public_test_to_internal.items():
        values = raw_by_test.get(public_test, [])
        if len(values) != 1:
            erased.append({"row": public_test, "reason": "missing_or_duplicate_verifier_result"})
            continue
        raw = values[0]
        public_outcome, outcome_repaired = _repair_enum(
            str(raw.get("observed_outcome") or ""), tuple(packet.public_outcome_to_internal[public_test])
        )
        status, status_repaired = _repair_enum(str(raw.get("support_status") or ""), statuses)
        internal_outcome = packet.public_outcome_to_internal[public_test].get(public_outcome or "")
        raw_refs = raw.get("source_span_ids")
        raw_ruled_out = raw.get("ruled_out_outcomes")
        if not isinstance(raw_refs, list) or not isinstance(raw_ruled_out, list):
            erased.append({"row": public_test, "reason": "source_or_ruled_out_must_be_lists"})
            continue
        refs = tuple(str(item) for item in raw_refs)
        if any(ref not in source_ids for ref in refs):
            erased.append({"row": public_test, "reason": "unknown_source_span_id"})
            continue
        if status != "UNDERDETERMINED" and not refs:
            erased.append({"row": public_test, "reason": "decisive_result_requires_source_span"})
            continue
        ruled_out: list[str] = []
        invalid_ruled_out = False
        repaired = bool(outcome_repaired or status_repaired)
        for value in raw_ruled_out:
            public_value, value_repaired = _repair_enum(
                str(value),
                tuple(packet.public_outcome_to_internal[public_test]),
            )
            internal_value = packet.public_outcome_to_internal[public_test].get(public_value or "")
            if internal_value is None:
                invalid_ruled_out = True
                break
            repaired = repaired or value_repaired
            ruled_out.append(internal_value)
        if internal_outcome is None or status is None or invalid_ruled_out:
            erased.append({"row": public_test, "reason": "invalid_outcome_status_or_ruled_out"})
            continue
        repair_count += int(repaired)
        results[internal_test] = VerifierResultV2(
            test_id=internal_test,
            observed_outcome=internal_outcome,
            support_status=status,  # type: ignore[arg-type]
            source_span_ids=refs,
            ruled_out_outcomes=tuple(ruled_out),
            parse_status="format_repaired" if repaired else "ok",
            format_repaired=repaired,
        )
    return CertificateVerifierParseResultV2(
        True,
        results,
        expected,
        len(results),
        tuple(erased),
        repair_count,
    )


def run_deterministic_adapters_v2(
    sample: DatasetSample,
    *,
    contract: TaskContractV2,
    tests: tuple[CertificateTestV2, ...],
    answer_nodes: dict[str, CandidateAnswerNode],
    pairs: tuple[CandidatePairV2, ...],
) -> dict[str, AdapterResult]:
    """Execute explicit certificate operations; never enumerate hidden answers."""

    pair_by_id = {pair.pair_id: pair for pair in pairs}
    results: dict[str, AdapterResult] = {}
    for test in tests:
        pair = pair_by_id[test.pair_id]
        candidates = (pair.left_candidate, pair.right_candidate)
        checks: dict[str, tuple[bool | None, str]] = {}
        for candidate in candidates:
            node = answer_nodes[candidate]
            checks[candidate] = _execute_candidate_adapter(
                sample,
                contract=contract,
                operation_kind=test.operation_kind,
                payload=test.deterministic_payload,
                node=node,
            )
        valid_candidates = [candidate for candidate, (valid, _) in checks.items() if valid is True]
        status: ExecutionStatus = "EXECUTED"
        observed: str | None = None
        detail = "; ".join(f"{candidate}:{checks[candidate][1]}" for candidate in candidates)
        if any(valid is None for valid, _ in checks.values()):
            status = "UNSUPPORTED"
        elif len(valid_candidates) != 1:
            status = "CONFLICT"
        else:
            observed = test.expected_outcome_by_candidate[valid_candidates[0]]
        trace = {
            "test_id": test.test_id,
            "adapter_kind": contract.adapter_kind,
            "operation_kind": test.operation_kind,
            "checks": checks,
            "status": status,
            "observed": observed,
        }
        results[test.test_id] = AdapterResult(
            test_id=test.test_id,
            observed_outcome=observed,
            execution_status=status,
            residual_or_first_failure=detail,
            execution_trace_hash=_json_sha256(trace),
        )
    return results


def decode_certificates_v2(
    stage: StageDecision,
    *,
    validation: CertificateBankValidationV2,
    public_to_key: dict[str, str],
    panels: tuple[CertificateVerifierParseResultV2, ...],
    adapter_results: dict[str, AdapterResult],
) -> DecodeDecision:
    """Select one challenger only after answer-link and obligation checks pass."""

    if validation.protocol_error is not None:
        resolver = "no_certificate" if validation.protocol_error == "no_certificate_tests" else "certificate_invalid"
        return DecodeDecision(stage.anchor_answer, stage.anchor_key, False, resolver, (), ())
    if validation.adapter_conflicts:
        return DecodeDecision(stage.anchor_answer, stage.anchor_key, False, "adapter_conflict", (), ())
    if panels and (len(panels) != 2 or any(not panel.top_level_valid for panel in panels)):
        return DecodeDecision(stage.anchor_answer, stage.anchor_key, False, "verifier_ambiguous", (), ())

    test_by_id = {test.test_id: test for test in validation.tests}
    public_by_key = {key: public for public, key in public_to_key.items()}
    cert_by_public = {item.candidate_key_anon: item for item in validation.certificates}
    diagnostics: list[dict[str, Any]] = []
    passing: list[str] = []
    saw_adapter_conflict = False
    for challenger in validation.eligible_challengers:
        challenger_public = public_by_key[challenger]
        anchor_public = public_by_key[stage.anchor_key]
        certificate = cert_by_public[challenger_public]
        passed = True
        for test_id in certificate.required_test_ids:
            test = test_by_id[test_id]
            adapter = adapter_results.get(test_id)
            observed: str | None = None
            route = "verifier"
            if adapter is not None and adapter.execution_status == "EXECUTED":
                observed = adapter.observed_outcome
                route = "adapter"
                for panel in panels:
                    result = panel.results.get(test_id)
                    if (
                        result is not None
                        and result.support_status == "ENTAILED"
                        and result.observed_outcome != observed
                    ):
                        saw_adapter_conflict = True
                        passed = False
            else:
                if len(panels) != 2:
                    passed = False
                else:
                    panel_results = [panel.results.get(test_id) for panel in panels]
                    if any(result is None or result.support_status != "ENTAILED" for result in panel_results) or (
                        panel_results[0].observed_outcome != panel_results[1].observed_outcome  # type: ignore[union-attr]
                    ):
                        passed = False
                    else:
                        observed = panel_results[0].observed_outcome  # type: ignore[union-attr]
            expected_challenger = test.expected_outcome_by_candidate.get(challenger_public)
            expected_anchor = test.expected_outcome_by_candidate.get(anchor_public)
            answer_match = observed is not None and observed == expected_challenger
            refutes_anchor = test_id not in certificate.derived_refutation_test_ids or (
                expected_anchor is not None and expected_anchor != expected_challenger
            )
            passed = passed and answer_match and refutes_anchor
            diagnostics.append(
                {
                    "challenger_key": challenger,
                    "test_id": test_id,
                    "route": route,
                    "observed_outcome": observed,
                    "expected_challenger": expected_challenger,
                    "expected_anchor": expected_anchor,
                    "answer_match": answer_match,
                    "refutes_anchor": refutes_anchor,
                    "passed": passed,
                }
            )
        if passed and certificate.derived_refutation_test_ids:
            passing.append(challenger)
    if saw_adapter_conflict:
        return DecodeDecision(stage.anchor_answer, stage.anchor_key, False, "adapter_conflict", (), tuple(diagnostics))
    if len(passing) != 1:
        resolver = "multiple_certificates_passed" if len(passing) > 1 else "abstention"
        return DecodeDecision(
            stage.anchor_answer, stage.anchor_key, False, resolver, tuple(passing), tuple(diagnostics)
        )
    winner = next(candidate for candidate in stage.candidates if candidate.key == passing[0])
    return DecodeDecision(
        winner.answer, winner.key, True, "certificate_v2_verified_override", tuple(passing), tuple(diagnostics)
    )


def candidate_answer_node_to_dict(node: CandidateAnswerNode) -> dict[str, Any]:
    return asdict(node)


def source_span_graph_to_dict(graph: SourceSpanGraph) -> dict[str, Any]:
    return asdict(graph)


def task_contract_v2_to_dict(contract: TaskContractV2) -> dict[str, Any]:
    return asdict(contract)


def pair_v2_to_dict(pair: CandidatePairV2) -> dict[str, Any]:
    return {
        "pair_id": pair.pair_id,
        "left_candidate": pair.left_candidate,
        "right_candidate": pair.right_candidate,
    }


def certificate_test_v2_to_dict(test: CertificateTestV2) -> dict[str, Any]:
    return asdict(test)


def certificate_v2_to_dict(certificate: AnswerCertificateV2) -> dict[str, Any]:
    return asdict(certificate)


def verifier_result_v2_to_dict(result: VerifierResultV2) -> dict[str, Any]:
    return asdict(result)


def adapter_result_to_dict(result: AdapterResult) -> dict[str, Any]:
    return asdict(result)


def _validate_test_v2(
    raw: Any,
    *,
    index: int,
    pair_by_id: dict[str, CandidatePairV2],
    source_ids: set[str],
    obligation_ids: set[str],
    seen_tests: set[str],
) -> tuple[str | None, CertificateTestV2 | None]:
    fields = {
        "test_id",
        "pair_id",
        "obligation_ids",
        "operation_kind",
        "question_or_operation",
        "finite_outcomes",
        "expected_outcome_by_candidate",
        "source_span_ids",
        "deterministic_payload",
    }
    if not isinstance(raw, dict) or set(raw) != fields:
        return "invalid_certificate_v2_test_schema", None
    test_id = str(raw.get("test_id") or "")
    if not re.fullmatch(r"T[0-9]+", test_id) or test_id in seen_tests:
        return "invalid_or_duplicate_test_id", None
    pair_id = str(raw.get("pair_id") or "")
    pair = pair_by_id.get(pair_id)
    if pair is None:
        return "invalid_pair_id", None
    raw_obligations = raw.get("obligation_ids")
    if (
        not isinstance(raw_obligations, list)
        or not raw_obligations
        or any(str(item) not in obligation_ids for item in raw_obligations)
    ):
        return "invalid_obligation_ids", None
    operation_kind = _normalize(raw.get("operation_kind"))
    question = _normalize(raw.get("question_or_operation"))
    if not operation_kind or not 8 <= len(question) <= 512:
        return "operation_or_question_invalid", None
    raw_outcomes = raw.get("finite_outcomes")
    if not isinstance(raw_outcomes, list) or not 2 <= len(raw_outcomes) <= 5:
        return "finite_outcomes_count_invalid", None
    outcomes: list[CertificateOutcomeV2] = []
    seen_outcomes: set[str] = set()
    for outcome in raw_outcomes:
        if not isinstance(outcome, dict) or set(outcome) != {"outcome_id", "text"}:
            return "invalid_outcome_schema", None
        outcome_id = str(outcome.get("outcome_id") or "")
        text = _normalize(outcome.get("text"))
        if not re.fullmatch(r"O[0-9]+", outcome_id) or outcome_id in seen_outcomes or not text:
            return "invalid_or_duplicate_outcome", None
        seen_outcomes.add(outcome_id)
        outcomes.append(CertificateOutcomeV2(outcome_id, text))
    expected = raw.get("expected_outcome_by_candidate")
    expected_candidates = {pair.left_candidate, pair.right_candidate}
    if not isinstance(expected, dict) or set(expected) != expected_candidates:
        return "invalid_candidate_commitments", None
    expected_map = {str(key): str(value) for key, value in expected.items()}
    if any(value not in seen_outcomes for value in expected_map.values()) or len(set(expected_map.values())) != 2:
        return "non_discriminating_or_unknown_commitment", None
    raw_sources = raw.get("source_span_ids")
    if not isinstance(raw_sources, list) or any(str(item) not in source_ids for item in raw_sources):
        return "invalid_source_span_ids", None
    payload = raw.get("deterministic_payload")
    if not isinstance(payload, dict):
        return "deterministic_payload_must_be_object", None
    return None, CertificateTestV2(
        test_id=test_id,
        pair_id=pair_id,
        obligation_ids=tuple(str(item) for item in raw_obligations),
        operation_kind=operation_kind,
        question_or_operation=question,
        finite_outcomes=tuple(outcomes),
        expected_outcome_by_candidate=expected_map,
        source_span_ids=tuple(str(item) for item in raw_sources),
        deterministic_payload=dict(payload),
    )


def _compile_certificate_v2(
    raw: Any,
    *,
    index: int,
    answer_nodes: dict[str, CandidateAnswerNode],
    tests_by_id: dict[str, CertificateTestV2],
    pairs: tuple[CandidatePairV2, ...],
    required_obligations: set[str],
) -> tuple[str | None, AnswerCertificateV2 | None]:
    if not isinstance(raw, dict) or set(raw) != {"candidate_key_anon", "answer_hash", "required_test_ids"}:
        return "invalid_answer_certificate_v2_schema", None
    candidate = str(raw.get("candidate_key_anon") or "")
    node = answer_nodes.get(candidate)
    if node is None:
        return "unknown_certificate_candidate", None
    if str(raw.get("answer_hash") or "") != node.answer_hash:
        return "answer_hash_mismatch", None
    raw_required = raw.get("required_test_ids")
    if not isinstance(raw_required, list) or not raw_required:
        return "required_test_ids_invalid", None
    # Rebuild references after invalid tests are removed instead of cascading.
    required = tuple(dict.fromkeys(str(item) for item in raw_required if str(item) in tests_by_id))
    if not required:
        return "no_surviving_required_tests", None
    candidate_pairs = [pair for pair in pairs if candidate in {pair.left_candidate, pair.right_candidate}]
    if not candidate_pairs:
        return "candidate_not_in_any_pair", None
    relevant_pair_ids = {pair.pair_id for pair in candidate_pairs}
    if any(tests_by_id[test_id].pair_id not in relevant_pair_ids for test_id in required):
        return "required_test_for_unrelated_pair", None
    covered = {item for test_id in required for item in tests_by_id[test_id].obligation_ids}
    if not required_obligations.issubset(covered):
        return "mandatory_obligation_missing", None
    refutations: list[str] = []
    for test_id in required:
        test = tests_by_id[test_id]
        values = test.expected_outcome_by_candidate
        if candidate in values and len(set(values.values())) == 2:
            refutations.append(test_id)
    if not refutations:
        return "no_derived_refutation_test", None
    canonical = {
        "candidate_key_anon": candidate,
        "answer_hash": node.answer_hash,
        "required_test_ids": required,
        "derived_refutation_test_ids": tuple(refutations),
    }
    return None, AnswerCertificateV2(**canonical, certificate_hash=_json_sha256(canonical))


def _execute_candidate_adapter(
    sample: DatasetSample,
    *,
    contract: TaskContractV2,
    operation_kind: str,
    payload: dict[str, Any],
    node: CandidateAnswerNode,
) -> tuple[bool | None, str]:
    kind = operation_kind or contract.adapter_kind
    if kind == "seq_plan":
        result = validate_seqbench_plan(node.canonical_value, sample=sample)
        return result.complete, result.first_failure or ("complete" if result.complete else "incomplete")
    if kind == "sort_order":
        values = [item.strip() for item in node.rendered_content.split(",") if item.strip()]
        if not values:
            return False, "empty_sequence"
        return values == sorted(values, key=str.casefold), "sorted" if values == sorted(
            values, key=str.casefold
        ) else "not_sorted"
    if kind == "stack_trace":
        earliest, detail = _earliest_stack_error(question_without_answer_contract(sample))
        if earliest is None:
            return None, detail
        candidate = _first_integer(node.canonical_value)
        return candidate == earliest, f"earliest_error={earliest};candidate={candidate}"
    if kind == "grid_path":
        answer, detail = _grid_path_answer(question_without_answer_contract(sample))
        if answer is None:
            return None, detail
        return _normalize(answer).casefold() == _normalize(node.rendered_content).casefold(), f"final_entity={answer}"
    if kind == "permutation":
        initial = payload.get("initial_order")
        swaps = payload.get("swaps")
        if not isinstance(initial, list) or not initial or not isinstance(swaps, list):
            return None, "explicit_initial_order_or_swaps_missing"
        order = [str(item) for item in initial]
        for index, swap in enumerate(swaps):
            if not isinstance(swap, list) or len(swap) != 2:
                return None, f"swap_{index}_invalid"
            left, right = str(swap[0]), str(swap[1])
            if left not in order or right not in order:
                return None, f"swap_{index}_unknown_item"
            left_index, right_index = order.index(left), order.index(right)
            order[left_index], order[right_index] = order[right_index], order[left_index]
        query_item = payload.get("query_item")
        query_position = payload.get("query_position")
        if query_item is not None and str(query_item) in order:
            expected = str(order.index(str(query_item)) + 1)
        elif isinstance(query_position, int) and 1 <= query_position <= len(order):
            expected = order[query_position - 1]
        else:
            return None, "explicit_permutation_query_missing"
        return _answer_contains_value(node, expected), f"permutation_result={expected}"
    if kind == "object_state":
        initial = payload.get("initial_locations")
        events = payload.get("events")
        query_entity = str(payload.get("query_entity") or "")
        if not isinstance(initial, dict) or not isinstance(events, list) or not query_entity:
            return None, "explicit_object_state_payload_missing"
        locations = {str(entity): str(location) for entity, location in initial.items()}
        for index, event in enumerate(events):
            if not isinstance(event, dict) or set(event) != {"entity", "to"}:
                return None, f"event_{index}_invalid"
            locations[str(event["entity"])] = str(event["to"])
        expected = locations.get(query_entity)
        if expected is None:
            return None, "query_entity_unknown"
        return _answer_contains_value(node, expected), f"final_location={expected}"
    if kind == "arithmetic_dsl":
        per_candidate = payload.get("checks_by_candidate")
        check = per_candidate.get(node.candidate_key_anon) if isinstance(per_candidate, dict) else payload
        check = check if isinstance(check, dict) else {}
        left = check.get("left_expression")
        right = check.get("right_expression")
        if not isinstance(left, str) or not isinstance(right, str):
            return None, "explicit_numeric_expressions_missing"
        try:
            left_value = _safe_arithmetic(left)
            right_value = _safe_arithmetic(right)
        except (ArithmeticError, SyntaxError, ValueError):
            return None, "numeric_expression_invalid"
        valid = math.isclose(
            left_value,
            right_value,
            rel_tol=float(contract.tolerance_policy.get("relative", 1e-6)),
            abs_tol=float(contract.tolerance_policy.get("absolute", 1e-9)),
        )
        return valid, f"residual={left_value - right_value}"
    if kind == "constraint_witness":
        assignments = payload.get("assignments_by_candidate")
        assignment = (
            assignments.get(node.candidate_key_anon) if isinstance(assignments, dict) else payload.get("assignment")
        )
        constraints = payload.get("constraints")
        if not isinstance(assignment, dict) or not isinstance(constraints, list):
            return None, "explicit_assignment_or_constraints_missing"
        valid, detail = _check_constraint_witness(assignment, constraints)
        return valid, detail
    # The remaining registered kinds are explicit extension points.  Returning
    # UNSUPPORTED forces blinded verification rather than guessing.
    return None, f"adapter_not_executable:{kind}"


def _earliest_stack_error(source: str) -> tuple[int | None, str]:
    rows: list[tuple[int, str, tuple[str, ...]]] = []
    pattern = re.compile(r"Thought\s+(\d+)\s*:\s*([\[\]{}()<>])\s*;\s*stack\s*:\s*([^\r\n]*)", re.IGNORECASE)
    for match in pattern.finditer(source):
        rows.append((int(match.group(1)), match.group(2), tuple(re.findall(r"[\[\]{}()<>]", match.group(3)))))
    if not rows:
        return None, "thought_stack_rows_missing"
    stack: list[str] = []
    matching = {"]": "[", "}": "{", ")": "(", ">": "<"}
    openings = set(matching.values())
    for index, symbol, reported in rows:
        valid_transition = True
        if symbol in openings:
            stack.append(symbol)
        elif not stack or stack[-1] != matching[symbol]:
            valid_transition = False
        else:
            stack.pop()
        if not valid_transition or tuple(stack) != reported:
            return index, "invalid_transition" if not valid_transition else "reported_stack_mismatch"
    return None, "no_error_detected"


def _grid_path_answer(source: str) -> tuple[str | None, str]:
    direction = {
        "up-right": (1, 0),
        "down-right": (0, 1),
        "down-left": (-1, 0),
        "up-left": (0, -1),
        "up": (1, -1),
        "down": (-1, 1),
    }
    initial = re.search(r"Initially,.*?where you find (?:an?|the)\s+([^.,]+)", source, re.IGNORECASE | re.DOTALL)
    if initial is None:
        return None, "initial_entity_missing"
    coordinate = (0, 0)
    entities: dict[tuple[int, int], str] = {coordinate: initial.group(1).strip()}
    move_pattern = re.compile(
        r"You move\s+(up-right|down-right|down-left|up-left|up|down)\s+by one step(?:,\s*where you find (?:an?|the)\s+([^.,?]+))?",
        re.IGNORECASE,
    )
    moves = list(move_pattern.finditer(source))
    if not moves:
        return None, "moves_missing"
    for match in moves:
        dx, dy = direction[match.group(1).casefold()]
        coordinate = (coordinate[0] + dx, coordinate[1] + dy)
        if match.group(2):
            entities[coordinate] = match.group(2).strip()
    return entities.get(coordinate), f"final_coordinate={coordinate}"


def _check_constraint_witness(assignment: dict[str, Any], constraints: list[Any]) -> tuple[bool, str]:
    for index, constraint in enumerate(constraints):
        if not isinstance(constraint, dict) or set(constraint) != {"left", "operator", "right"}:
            return False, f"constraint_{index}_schema_invalid"
        left = assignment.get(str(constraint["left"]), constraint["left"])
        right = assignment.get(str(constraint["right"]), constraint["right"])
        operator = str(constraint["operator"])
        valid = (left == right) if operator == "==" else (left != right) if operator == "!=" else False
        if not valid:
            return False, f"constraint_{index}_violated"
    return True, "all_explicit_constraints_satisfied"


def _answer_contains_value(node: CandidateAnswerNode, expected: str) -> bool:
    normalized_expected = _normalize(expected).casefold()
    values = {
        _normalize(node.canonical_value).casefold(),
        _normalize(node.rendered_content).casefold(),
    }
    return normalized_expected in values or any(
        re.search(rf"(?<!\w){re.escape(normalized_expected)}(?!\w)", value) is not None for value in values
    )


def _empty_validation(error: str) -> CertificateBankValidationV2:
    return CertificateBankValidationV2((), (), (), error, (), (), 0.0, 0.0)


def _source_ranges(source: str, *, max_span_length: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for line_match in re.finditer(r"[^\r\n]+", source):
        line_start, line_end = line_match.span()
        cursor = line_start
        sentence_matches = list(re.finditer(r".+?(?:[.!?](?=\s|$)|$)", line_match.group(0)))
        for sentence in sentence_matches:
            start = line_start + sentence.start()
            end = line_start + sentence.end()
            while end - start > max_span_length:
                split = source.rfind(" ", start, start + max_span_length + 1)
                split = split if split > start else start + max_span_length
                ranges.append((start, split))
                start = split + int(split < end and source[split] == " ")
            if source[start:end].strip():
                ranges.append((start, end))
            cursor = end
        if cursor < line_end and source[cursor:line_end].strip():
            ranges.append((cursor, line_end))
    return ranges


def _option_text_by_label(sample: DatasetSample) -> dict[str, str]:
    contract = sample.metadata.get("answer_contract")
    raw_options = contract.get("options") if isinstance(contract, dict) else sample.metadata.get("options")
    mapping: dict[str, str] = {}
    if isinstance(raw_options, list):
        for index, raw in enumerate(raw_options):
            if isinstance(raw, dict):
                label = str(raw.get("label") or chr(ord("A") + index)).upper()
                text = str(raw.get("text") or "").strip()
            else:
                label = chr(ord("A") + index)
                text = str(raw).strip()
            if text:
                mapping[label] = text
    return mapping


def _render_candidate_answer(answer: str, canonical: str, options: dict[str, str]) -> str:
    labels = [character for character in canonical.upper() if character in options]
    if labels and len(labels) == len(canonical.replace(",", "").replace(" ", "")):
        return " | ".join(options[label] for label in labels)
    if canonical.upper() in options:
        return options[canonical.upper()]
    return str(answer).strip()


def _infer_answer_type(sample: DatasetSample, canonical: str, rendered: str) -> AnswerType:
    if _has_choice_contract(sample):
        return "choice"
    if sample.dataset == "seqbench" or canonical.startswith("["):
        return "sequence"
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", canonical):
        return "scalar"
    if "," in canonical:
        return "set"
    if len(rendered.split()) <= 8:
        return "entity"
    return "proposition"


def _has_choice_contract(sample: DatasetSample) -> bool:
    contract = sample.metadata.get("answer_contract")
    return isinstance(contract, dict) and str(contract.get("kind") or "") in {"single_choice", "multi_choice"}


def _obligation_scope(kind: ObligationKind, operator: QueryOperator) -> str:
    scopes = {
        "candidate_validity": "candidate answer satisfies the source query",
        "prefix_validity": "all ordered steps before the candidate index are valid",
        "final_state": "state after every ordered update at query time",
        "global_completeness": "no required member or action is omitted and no extra is introduced",
        "comparative_dominance": "candidate is preferred after comparing all source-supported alternatives",
        "unit_consistency": "equations, dimensions, assumptions, and units are mutually consistent",
        "constraint_satisfaction": "the explicit witness satisfies every cited source constraint",
    }
    return f"{operator}:{scopes[kind]}"


def _repair_enum(value: str, allowed: tuple[str, ...]) -> tuple[str | None, bool]:
    normalized = unicodedata.normalize("NFKC", value).strip().upper()
    by_normalized = {unicodedata.normalize("NFKC", item).strip().upper(): item for item in allowed}
    if normalized in by_normalized:
        canonical = by_normalized[normalized]
        return canonical, canonical != value
    candidates = [item for key, item in by_normalized.items() if _edit_distance_at_most_one(normalized, key)]
    if len(candidates) == 1:
        return candidates[0], True
    return None, False


def _edit_distance_at_most_one(left: str, right: str) -> bool:
    if left == right:
        return True
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right, strict=True)) <= 1
    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    index_short = index_long = differences = 0
    while index_short < len(shorter) and index_long < len(longer):
        if shorter[index_short] == longer[index_long]:
            index_short += 1
            index_long += 1
        else:
            differences += 1
            index_long += 1
            if differences > 1:
                return False
    return True


def _safe_arithmetic(expression: str) -> float:
    if not re.fullmatch(r"[-+*/(). 0-9eE]+", expression):
        raise ValueError("unsafe expression")
    value = eval(expression, {"__builtins__": {}}, {})  # noqa: S307 - numeric grammar above
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ArithmeticError("non-finite expression")
    return numeric


def _first_integer(value: str) -> int | None:
    match = re.search(r"\d+", value)
    return int(match.group(0)) if match else None


def _raw_id(raw: Any, fallback: int) -> str:
    if not isinstance(raw, dict):
        return str(fallback)
    return str(raw.get("test_id") or raw.get("candidate_key_anon") or fallback)


def _normalize(value: Any) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split())


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_sha256(value: Any) -> str:
    return _sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str))


def _stable_hash(seed: int, sample_id: str, role: str) -> int:
    return int(_sha256(f"{seed}\0{sample_id}\0{role}")[:16], 16)
