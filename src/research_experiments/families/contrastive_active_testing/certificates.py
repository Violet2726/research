"""面向同质模型测试时验证的问题条件证书。

The certificate protocol is deliberately candidate restricted.  A designer may
describe finite tests that distinguish anonymous Stage-A candidates, while the
two verifier panels only observe the source task and the tests.  Gold answers
are never passed to this module's public builders.
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

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.families.contrastive_active_testing.algorithms import (
    DecodeDecision,
    StageDecision,
)

TaskFamily = Literal["proof_state", "state_transition", "equation", "set_count", "semantic"]
SupportStatus = Literal["ENTAILED", "CONTRADICTED", "UNDERDETERMINED"]


@dataclass(frozen=True)
class TaskContract:
    family: TaskFamily
    query_type: str
    entity_schema: tuple[str, ...]
    temporal_scope: str
    outcome_schema: tuple[str, ...]
    tolerance_policy: dict[str, float]
    certificate_requirements: tuple[str, ...]
    adapter_version: str = "catch_cert_task_contract_v1"


@dataclass(frozen=True)
class SourceSpanRef:
    source: str
    start: int
    end: int
    sha256: str


@dataclass(frozen=True)
class ClaimNode:
    claim_id: str
    claim_type: str
    normalized_value: str
    source_span_refs: tuple[SourceSpanRef, ...]
    temporal_scope: str


@dataclass(frozen=True)
class ClaimEdge:
    source_claim_id: str
    target_claim_id: str
    relation: Literal["depends_on", "precedes", "contradicts", "computes"]


@dataclass(frozen=True)
class ClaimGraph:
    candidate_key_anon: str
    nodes: tuple[ClaimNode, ...]
    edges: tuple[ClaimEdge, ...]
    final_claim: str
    trace_hash: str


@dataclass(frozen=True)
class CertificateOutcome:
    outcome_id: str
    text: str


@dataclass(frozen=True)
class CertificateTest:
    test_id: str
    pair_id: str
    question_or_operation: str
    finite_outcomes: tuple[CertificateOutcome, ...]
    expected_outcome_by_candidate: dict[str, str]
    provenance_refs: tuple[str, ...]
    task_family: TaskFamily


@dataclass(frozen=True)
class AnswerCertificate:
    candidate_key_anon: str
    required_conditions: tuple[str, ...]
    predicted_outcome: str
    refutation_condition: str
    evidence_refs: tuple[str, ...]
    dependency_refs: tuple[str, ...]
    certificate_hash: str


@dataclass(frozen=True)
class CertificateBankValidation:
    certificates: tuple[AnswerCertificate, ...]
    tests: tuple[CertificateTest, ...]
    dropped: tuple[dict[str, str], ...]
    protocol_error: str | None
    leakage_count: int
    adapter_conflicts: tuple[dict[str, str], ...]
    eligible_challengers: tuple[str, ...]


@dataclass(frozen=True)
class CertificateVerifierPacket:
    panel_index: int
    role: str
    tests: tuple[dict[str, Any], ...]
    public_test_to_internal: dict[str, str]
    public_outcome_to_internal: dict[str, dict[str, str]]


@dataclass(frozen=True)
class VerifierResult:
    test_id: str
    observed_outcome: str
    support_status: SupportStatus
    counterexample: str
    source_refs: tuple[str, ...]
    parse_status: str


@dataclass(frozen=True)
class CertificateVerifierParseResult:
    top_level_valid: bool
    results: dict[str, VerifierResult]
    expected_test_count: int
    valid_test_count: int
    erased_rows: tuple[dict[str, str], ...]


_PROOF_TASKS = {
    "boardgame_qa",
    "boolean_expressions",
    "causal_understanding",
    "formal_fallacies_syllogisms_negation",
    "web_of_lies",
}
_EQUATION_TASKS = {
    "multistep_arithmetic",
    "time_arithmetic",
    "temporal_sequence",
}
_SET_TASKS = {
    "movie_recommendation",
    "object_counting",
    "object_properties",
    "sportqa",
    "team_allocation",
    "word_sorting",
}


def build_task_contract(sample: DatasetSample) -> TaskContract:
    """Infer a gold-free task contract from dataset and public task metadata."""

    task = str(sample.metadata.get("task") or "").strip().casefold()
    if sample.dataset == "seqbench" or task == "object_placements":
        family: TaskFamily = "state_transition"
        query_type = "final_state_after_ordered_actions"
        schema = ("entity", "location", "time", "action")
        temporal_scope = "ordered_events_and_query_time"
        outcomes = ("VALID_FINAL_STATE", "INVALID_FINAL_STATE", "UNDERDETERMINED")
        requirements = ("ordered_actions", "final_state", "first_invalid_transition")
    elif (sample.dataset == "gpqa_diamond" and task in {"physics", "chemistry"}) or task in _EQUATION_TASKS:
        family = "equation"
        query_type = "equation_chain_and_units"
        schema = ("quantity", "unit", "equation", "assumption")
        temporal_scope = "not_applicable"
        outcomes = ("CONSISTENT", "INCONSISTENT", "UNDERDETERMINED")
        requirements = ("equation_chain", "unit_check", "numerical_or_symbolic_outcome")
    elif task in _PROOF_TASKS:
        family = "proof_state"
        query_type = "three_valued_proof_status"
        schema = ("premise", "rule", "conclusion")
        temporal_scope = "proof_order"
        outcomes = ("ENTAILED", "CONTRADICTED", "UNKNOWN")
        requirements = ("premises", "rule_chain", "proof_status")
    elif task in _SET_TASKS:
        family = "set_count"
        query_type = "set_or_count_constraint"
        schema = ("item", "property", "membership", "count")
        temporal_scope = "latest_update_if_present"
        outcomes = ("SATISFIED", "VIOLATED", "UNDERDETERMINED")
        requirements = ("membership_or_count", "comparison", "final_outcome")
    else:
        family = "semantic"
        query_type = "evidence_chain_and_counterexample"
        schema = ("claim", "evidence", "necessary_condition", "counterexample")
        temporal_scope = "question_conditioned"
        outcomes = ("SUPPORTED", "REFUTED", "UNDERDETERMINED")
        requirements = ("necessary_conditions", "source_grounding", "refutation_condition")
    return TaskContract(
        family=family,
        query_type=query_type,
        entity_schema=schema,
        temporal_scope=temporal_scope,
        outcome_schema=outcomes,
        tolerance_policy={"absolute": 1e-9, "relative": 1e-6},
        certificate_requirements=requirements,
    )


def build_claim_graphs(
    stage: StageDecision,
    *,
    public_to_key: dict[str, str],
) -> dict[str, ClaimGraph]:
    """Build stable, non-filtered provenance graphs from representative traces."""

    candidate_by_key = {candidate.key: candidate for candidate in stage.candidates}
    graphs: dict[str, ClaimGraph] = {}
    for public_id, key in public_to_key.items():
        candidate = candidate_by_key[key]
        source = _normalize(candidate.representative_reasoning)
        spans = _sentence_spans(source)
        nodes: list[ClaimNode] = []
        for index, (start, end) in enumerate(spans):
            text = source[start:end].strip()
            if not text:
                continue
            actual_start = source.find(text, start, end)
            actual_end = actual_start + len(text)
            claim_id = f"{public_id}:N{index}"
            nodes.append(
                ClaimNode(
                    claim_id=claim_id,
                    claim_type=_infer_claim_type(text),
                    normalized_value=text,
                    source_span_refs=(
                        SourceSpanRef(
                            source="candidate_trace",
                            start=actual_start,
                            end=actual_end,
                            sha256=_sha256(text),
                        ),
                    ),
                    temporal_scope=_infer_temporal_scope(text),
                )
            )
        edges = tuple(
            ClaimEdge(nodes[index - 1].claim_id, nodes[index].claim_id, _infer_edge(nodes[index].normalized_value))
            for index in range(1, len(nodes))
        )
        graphs[public_id] = ClaimGraph(
            candidate_key_anon=public_id,
            nodes=tuple(nodes),
            edges=edges,
            final_claim=nodes[-1].claim_id if nodes else "",
            trace_hash=candidate.representative_trace_sha256,
        )
    return graphs


def validate_certificate_bank(
    payload: dict[str, Any] | None,
    *,
    contract: TaskContract,
    stage: StageDecision,
    public_to_key: dict[str, str],
    graphs: dict[str, ClaimGraph],
    pair_candidates: dict[str, tuple[str, str]],
    max_tests: int = 6,
) -> CertificateBankValidation:
    """Validate designer output without consulting a reference answer."""

    if not isinstance(payload, dict) or set(payload) != {"certificates", "tests"}:
        return CertificateBankValidation((), (), (), "certificate_top_level_schema_failure", 0, (), ())
    raw_certificates = payload.get("certificates")
    raw_tests = payload.get("tests")
    if not isinstance(raw_certificates, list) or not isinstance(raw_tests, list):
        return CertificateBankValidation((), (), (), "certificates_and_tests_must_be_lists", 0, (), ())
    if len(raw_tests) > max_tests:
        return CertificateBankValidation((), (), (), "too_many_certificate_tests", 0, (), ())

    graph_refs = {
        node.claim_id
        for graph in graphs.values()
        for node in graph.nodes
    }
    public_ids = set(public_to_key)
    tests: list[CertificateTest] = []
    dropped: list[dict[str, str]] = []
    leakage_count = 0
    seen_test_ids: set[str] = set()
    for index, raw in enumerate(raw_tests):
        reason, test, leaked = _validate_test(
            raw,
            index=index,
            contract=contract,
            public_ids=public_ids,
            graph_refs=graph_refs,
            seen_test_ids=seen_test_ids,
            pair_candidates=pair_candidates,
        )
        leakage_count += int(leaked)
        if reason is not None or test is None:
            dropped.append({"test_id": str(raw.get("test_id") if isinstance(raw, dict) else index), "reason": reason or "invalid_test"})
            continue
        seen_test_ids.add(test.test_id)
        tests.append(test)

    tests_by_id = {test.test_id: test for test in tests}
    certificates: list[AnswerCertificate] = []
    seen_candidates: set[str] = set()
    for index, raw in enumerate(raw_certificates):
        reason, certificate = _validate_certificate(
            raw,
            index=index,
            public_ids=public_ids,
            graph_refs=graph_refs,
            tests_by_id=tests_by_id,
        )
        if reason is not None or certificate is None:
            dropped.append({"certificate_id": str(index), "reason": reason or "invalid_certificate"})
            continue
        if certificate.candidate_key_anon in seen_candidates:
            dropped.append({"certificate_id": str(index), "reason": "duplicate_candidate_certificate"})
            continue
        seen_candidates.add(certificate.candidate_key_anon)
        certificates.append(certificate)

    adapter_conflicts = tuple(_adapter_conflicts(contract, tuple(tests)))
    cert_by_public = {certificate.candidate_key_anon: certificate for certificate in certificates}
    key_to_public = {key: public for public, key in public_to_key.items()}
    eligible: list[str] = []
    for candidate in stage.candidates:
        if candidate.key == stage.anchor_key:
            continue
        public_id = key_to_public.get(candidate.key)
        certificate = cert_by_public.get(public_id or "")
        if certificate is None or adapter_conflicts:
            continue
        required = [tests_by_id.get(test_id) for test_id in certificate.required_conditions]
        if required and all(test is not None for test in required):
            eligible.append(candidate.key)

    protocol_error = None
    if not tests:
        protocol_error = "no_certificate_tests"
    elif not certificates:
        protocol_error = "no_valid_certificates"
    return CertificateBankValidation(
        certificates=tuple(certificates),
        tests=tuple(tests),
        dropped=tuple(dropped),
        protocol_error=protocol_error,
        leakage_count=leakage_count,
        adapter_conflicts=adapter_conflicts,
        eligible_challengers=tuple(eligible),
    )


def build_certificate_verifier_packet(
    tests: tuple[CertificateTest, ...],
    *,
    seed: int,
    sample_id: str,
    panel_index: int,
) -> CertificateVerifierPacket:
    """Blind candidate commitments and permute finite outcomes per panel."""

    rng = random.Random(_stable_hash(seed, sample_id, f"certificate-verifier:{panel_index}"))
    ordered = list(tests)
    rng.shuffle(ordered)
    rendered: list[dict[str, Any]] = []
    public_test_to_internal: dict[str, str] = {}
    public_outcome_to_internal: dict[str, dict[str, str]] = {}
    for test_index, test in enumerate(ordered):
        public_test = f"Q{test_index}"
        outcomes = list(test.finite_outcomes)
        rng.shuffle(outcomes)
        rendered_outcomes = []
        outcome_map: dict[str, str] = {}
        for outcome_index, outcome in enumerate(outcomes):
            public_outcome = f"R{outcome_index}"
            rendered_outcomes.append({"outcome_id": public_outcome, "text": outcome.text})
            outcome_map[public_outcome] = outcome.outcome_id
        rendered.append(
            {
                "test_id": public_test,
                "question_or_operation": test.question_or_operation,
                "finite_outcomes": rendered_outcomes,
                "task_family": test.task_family,
            }
        )
        public_test_to_internal[public_test] = test.test_id
        public_outcome_to_internal[public_test] = outcome_map
    return CertificateVerifierPacket(
        panel_index=panel_index,
        role="support_auditor" if panel_index == 1 else "refutation_auditor",
        tests=tuple(rendered),
        public_test_to_internal=public_test_to_internal,
        public_outcome_to_internal=public_outcome_to_internal,
    )


def parse_certificate_verifier(
    payload: dict[str, Any] | None,
    *,
    packet: CertificateVerifierPacket,
) -> CertificateVerifierParseResult:
    expected = len(packet.public_test_to_internal)
    if not isinstance(payload, dict) or set(payload) != {"results"} or not isinstance(payload.get("results"), list):
        return CertificateVerifierParseResult(False, {}, expected, 0, ())
    raw_by_test: dict[str, list[dict[str, Any]]] = {}
    erased: list[dict[str, str]] = []
    for index, raw in enumerate(payload["results"]):
        if not isinstance(raw, dict) or set(raw) != {
            "test_id",
            "observed_outcome",
            "support_status",
            "counterexample",
            "source_refs",
        }:
            erased.append({"row": str(index), "reason": "invalid_verifier_row_schema"})
            continue
        public_test = str(raw.get("test_id") or "")
        if public_test not in packet.public_test_to_internal:
            erased.append({"row": str(index), "reason": "unknown_public_test_id"})
            continue
        raw_by_test.setdefault(public_test, []).append(raw)

    results: dict[str, VerifierResult] = {}
    statuses = {"ENTAILED", "CONTRADICTED", "UNDERDETERMINED"}
    for public_test, internal_test in packet.public_test_to_internal.items():
        values = raw_by_test.get(public_test, [])
        if len(values) != 1:
            erased.append({"row": public_test, "reason": "missing_or_duplicate_verifier_result"})
            continue
        raw = values[0]
        public_outcome = str(raw.get("observed_outcome") or "")
        internal_outcome = packet.public_outcome_to_internal[public_test].get(public_outcome)
        status = str(raw.get("support_status") or "")
        refs = raw.get("source_refs")
        if (
            internal_outcome is None
            or status not in statuses
            or not isinstance(refs, list)
            or not refs
            or any(_looks_like_public_identity(str(item)) for item in refs)
        ):
            erased.append({"row": public_test, "reason": "invalid_outcome_status_or_source_refs"})
            continue
        results[internal_test] = VerifierResult(
            test_id=internal_test,
            observed_outcome=internal_outcome,
            support_status=status,  # type: ignore[arg-type]
            counterexample=str(raw.get("counterexample") or "").strip(),
            source_refs=tuple(str(item).strip() for item in refs if str(item).strip()),
            parse_status="ok",
        )
    return CertificateVerifierParseResult(True, results, expected, len(results), tuple(erased))


def decode_certificates(
    stage: StageDecision,
    *,
    validation: CertificateBankValidation,
    public_to_key: dict[str, str],
    panels: tuple[CertificateVerifierParseResult, ...],
) -> DecodeDecision:
    """Accept one challenger only when its certificate passes both panels."""

    if validation.protocol_error is not None:
        resolver = "no_certificate" if validation.protocol_error == "no_certificate_tests" else "certificate_invalid"
        return DecodeDecision(stage.anchor_answer, stage.anchor_key, False, resolver, (), ())
    if validation.adapter_conflicts:
        return DecodeDecision(stage.anchor_answer, stage.anchor_key, False, "adapter_conflict", (), ())
    if len(panels) != 2 or any(not panel.top_level_valid for panel in panels):
        return DecodeDecision(stage.anchor_answer, stage.anchor_key, False, "verifier_ambiguous", (), ())

    test_by_id = {test.test_id: test for test in validation.tests}
    public_by_key = {key: public for public, key in public_to_key.items()}
    certificate_by_public = {certificate.candidate_key_anon: certificate for certificate in validation.certificates}
    diagnostics: list[dict[str, Any]] = []
    passing: list[str] = []
    for challenger in validation.eligible_challengers:
        challenger_public = public_by_key.get(challenger, "")
        anchor_public = public_by_key.get(stage.anchor_key, "")
        certificate = certificate_by_public.get(challenger_public)
        if certificate is None:
            continue
        panel_passes: list[bool] = []
        for panel_index, panel in enumerate(panels, start=1):
            required_passes = 0
            refutations = 0
            ambiguous = 0
            for test_id in certificate.required_conditions:
                test = test_by_id.get(test_id)
                result = panel.results.get(test_id)
                if test is None or result is None or result.support_status != "ENTAILED":
                    ambiguous += 1
                    continue
                expected_challenger = test.expected_outcome_by_candidate.get(challenger_public)
                expected_anchor = test.expected_outcome_by_candidate.get(anchor_public)
                if result.observed_outcome == expected_challenger:
                    required_passes += 1
                    if expected_anchor is not None and expected_anchor != expected_challenger:
                        refutations += 1
            refutation_test = test_by_id.get(certificate.refutation_condition)
            refutation_result = panel.results.get(certificate.refutation_condition)
            refutation_pass = False
            if refutation_test is not None and refutation_result is not None:
                expected_challenger = refutation_test.expected_outcome_by_candidate.get(challenger_public)
                expected_anchor = refutation_test.expected_outcome_by_candidate.get(anchor_public)
                refutation_pass = (
                    refutation_result.support_status == "ENTAILED"
                    and refutation_result.observed_outcome == expected_challenger
                    and expected_anchor is not None
                    and expected_anchor != expected_challenger
                    and (panel_index == 1 or bool(refutation_result.counterexample))
                )
            passed = (
                required_passes == len(certificate.required_conditions)
                and required_passes > 0
                and refutations > 0
                and refutation_pass
                and ambiguous == 0
            )
            panel_passes.append(passed)
            diagnostics.append(
                {
                    "challenger_key": challenger,
                    "panel_index": panel_index,
                    "required_passes": required_passes,
                    "required_total": len(certificate.required_conditions),
                    "anchor_refutations": refutations,
                    "refutation_condition": certificate.refutation_condition,
                    "refutation_pass": refutation_pass,
                    "ambiguous": ambiguous,
                    "passed": passed,
                }
            )
        if all(panel_passes):
            passing.append(challenger)
    if len(passing) != 1:
        resolver = "multiple_certificates_passed" if len(passing) > 1 else "abstention"
        return DecodeDecision(stage.anchor_answer, stage.anchor_key, False, resolver, tuple(passing), tuple(diagnostics))
    winner = next(candidate for candidate in stage.candidates if candidate.key == passing[0])
    return DecodeDecision(
        winner.answer,
        winner.key,
        True,
        "certificate_verified_override",
        tuple(passing),
        tuple(diagnostics),
    )


def task_contract_to_dict(contract: TaskContract) -> dict[str, Any]:
    return asdict(contract)


def claim_graph_to_dict(graph: ClaimGraph) -> dict[str, Any]:
    return asdict(graph)


def certificate_to_dict(certificate: AnswerCertificate) -> dict[str, Any]:
    return asdict(certificate)


def certificate_test_to_dict(test: CertificateTest) -> dict[str, Any]:
    return asdict(test)


def verifier_result_to_dict(result: VerifierResult) -> dict[str, Any]:
    return asdict(result)


def _validate_test(
    raw: Any,
    *,
    index: int,
    contract: TaskContract,
    public_ids: set[str],
    graph_refs: set[str],
    seen_test_ids: set[str],
    pair_candidates: dict[str, tuple[str, str]],
) -> tuple[str | None, CertificateTest | None, bool]:
    if not isinstance(raw, dict) or set(raw) != {
        "test_id",
        "pair_id",
        "question_or_operation",
        "finite_outcomes",
        "expected_outcome_by_candidate",
        "provenance_refs",
        "task_family",
    }:
        return "invalid_certificate_test_schema", None, False
    test_id = str(raw.get("test_id") or "")
    if not re.fullmatch(r"T[0-9]+", test_id) or test_id in seen_test_ids:
        return "invalid_or_duplicate_test_id", None, False
    pair_id = str(raw.get("pair_id") or "")
    if not re.fullmatch(r"P[0-9]+", pair_id) or pair_id not in pair_candidates:
        return "invalid_pair_id", None, False
    question = _normalize(raw.get("question_or_operation"))
    if not 8 <= len(question) <= 512:
        return "question_or_operation_length_invalid", None, False
    leaked = _looks_like_public_identity(question)
    if leaked:
        return "candidate_identity_leakage", None, True
    family = str(raw.get("task_family") or "")
    if family != contract.family:
        return "task_family_mismatch", None, False
    raw_outcomes = raw.get("finite_outcomes")
    if not isinstance(raw_outcomes, list) or not 2 <= len(raw_outcomes) <= 5:
        return "finite_outcomes_count_invalid", None, False
    outcomes: list[CertificateOutcome] = []
    seen_outcomes: set[str] = set()
    for outcome in raw_outcomes:
        if not isinstance(outcome, dict) or set(outcome) != {"outcome_id", "text"}:
            return "invalid_outcome_schema", None, False
        outcome_id = str(outcome.get("outcome_id") or "")
        text = _normalize(outcome.get("text"))
        if (
            not re.fullmatch(r"O[0-9]+", outcome_id)
            or outcome_id in seen_outcomes
            or not 1 <= len(text) <= 256
            or _looks_like_public_identity(text)
        ):
            return "invalid_or_duplicate_outcome", None, False
        seen_outcomes.add(outcome_id)
        outcomes.append(CertificateOutcome(outcome_id, text))
    expected = raw.get("expected_outcome_by_candidate")
    expected_candidates = set(pair_candidates[pair_id])
    if (
        not isinstance(expected, dict)
        or set(expected) != expected_candidates
        or not set(expected).issubset(public_ids)
    ):
        return "invalid_candidate_commitments", None, False
    expected_map = {str(key): str(value) for key, value in expected.items()}
    if any(value not in seen_outcomes for value in expected_map.values()) or len(set(expected_map.values())) < 2:
        return "non_discriminating_or_unknown_commitment", None, False
    refs = raw.get("provenance_refs")
    if not isinstance(refs, list) or not refs or any(str(ref) not in graph_refs for ref in refs):
        return "invalid_provenance_refs", None, False
    return (
        None,
        CertificateTest(
            test_id=test_id,
            pair_id=pair_id,
            question_or_operation=question,
            finite_outcomes=tuple(outcomes),
            expected_outcome_by_candidate=expected_map,
            provenance_refs=tuple(str(ref) for ref in refs),
            task_family=contract.family,
        ),
        False,
    )


def _validate_certificate(
    raw: Any,
    *,
    index: int,
    public_ids: set[str],
    graph_refs: set[str],
    tests_by_id: dict[str, CertificateTest],
) -> tuple[str | None, AnswerCertificate | None]:
    if not isinstance(raw, dict) or set(raw) != {
        "candidate_key_anon",
        "required_conditions",
        "predicted_outcome",
        "refutation_condition",
        "evidence_refs",
        "dependency_refs",
    }:
        return "invalid_answer_certificate_schema", None
    candidate = str(raw.get("candidate_key_anon") or "")
    required = raw.get("required_conditions")
    evidence = raw.get("evidence_refs")
    dependencies = raw.get("dependency_refs")
    if candidate not in public_ids:
        return "unknown_certificate_candidate", None
    if not isinstance(required, list) or not required or len(required) > 4:
        return "required_conditions_invalid", None
    required_ids = tuple(str(item) for item in required)
    if len(set(required_ids)) != len(required_ids) or any(item not in tests_by_id for item in required_ids):
        return "unknown_or_duplicate_required_condition", None
    if not isinstance(evidence, list) or not evidence or any(str(ref) not in graph_refs for ref in evidence):
        return "certificate_evidence_refs_invalid", None
    if not isinstance(dependencies, list) or any(str(ref) not in graph_refs for ref in dependencies):
        return "certificate_dependency_refs_invalid", None
    refutation = str(raw.get("refutation_condition") or "")
    if refutation not in tests_by_id:
        return "unknown_refutation_condition", None
    predicted = _normalize(raw.get("predicted_outcome"))
    if not predicted:
        return "empty_predicted_outcome", None
    canonical = {
        "candidate_key_anon": candidate,
        "required_conditions": required_ids,
        "predicted_outcome": predicted,
        "refutation_condition": refutation,
        "evidence_refs": tuple(str(ref) for ref in evidence),
        "dependency_refs": tuple(str(ref) for ref in dependencies),
    }
    return None, AnswerCertificate(**canonical, certificate_hash=_json_sha256(canonical))


def _adapter_conflicts(contract: TaskContract, tests: tuple[CertificateTest, ...]) -> list[dict[str, str]]:
    conflicts: list[dict[str, str]] = []
    for test in tests:
        if contract.family == "equation":
            conflict = _equation_conflict(test.question_or_operation, contract.tolerance_policy)
            if conflict:
                conflicts.append({"test_id": test.test_id, "reason": conflict})
        if contract.family == "proof_state":
            texts = {outcome.text.strip().upper() for outcome in test.finite_outcomes}
            allowed = {value.upper() for value in contract.outcome_schema}
            if not texts & allowed:
                conflicts.append({"test_id": test.test_id, "reason": "proof_outcome_schema_missing"})
    return conflicts


def _equation_conflict(operation: str, tolerance: dict[str, float]) -> str | None:
    """Check explicit ``CHECK: lhs == rhs`` arithmetic without solving the task."""

    match = re.fullmatch(r"\s*CHECK:\s*([-+*/(). 0-9eE]+)\s*(?:==|=)\s*([-+*/(). 0-9eE]+)\s*", operation)
    if match is None:
        return None
    try:
        left = _safe_arithmetic(match.group(1))
        right = _safe_arithmetic(match.group(2))
    except (ArithmeticError, ValueError, SyntaxError):
        return "equation_operation_invalid"
    if not math.isclose(
        left,
        right,
        rel_tol=float(tolerance.get("relative", 1e-6)),
        abs_tol=float(tolerance.get("absolute", 1e-9)),
    ):
        return "equation_residual_nonzero"
    return None


def _safe_arithmetic(expression: str) -> float:
    if not re.fullmatch(r"[-+*/(). 0-9eE]+", expression):
        raise ValueError("unsafe expression")
    value = eval(expression, {"__builtins__": {}}, {})  # noqa: S307 - grammar is numeric-only above
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ArithmeticError("non-finite expression")
    return numeric


def _sentence_spans(source: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"(?:[.!?\u3002\uff01\uff1f]+(?=\s|$)|\n+)", source):
        end = match.end()
        if source[start:end].strip():
            spans.append((start, end))
        start = end
    if source[start:].strip():
        spans.append((start, len(source)))
    return spans or ([(0, len(source))] if source else [])


def _infer_claim_type(text: str) -> str:
    lowered = text.casefold()
    if re.search(r"[-+]?\d+(?:\.\d+)?\s*(?:=|<|>|\u2248|\u2264|\u2265)", text):
        return "equation"
    if any(token in lowered for token in ("before", "after", "later", "then", "finally", "moved")):
        return "state_transition"
    if any(token in lowered for token in ("therefore", "thus", "hence", "implies", "rule")):
        return "inference"
    if any(token in lowered for token in ("count", "total", "set", "collection")):
        return "set_or_count"
    return "claim"


def _infer_temporal_scope(text: str) -> str:
    lowered = text.casefold()
    if "finally" in lowered or "final" in lowered:
        return "final"
    if "later" in lowered or "after" in lowered or "then" in lowered:
        return "ordered_update"
    if "before" in lowered or "initial" in lowered or "original" in lowered:
        return "initial_or_prior"
    return "unspecified"


def _infer_edge(text: str) -> Literal["depends_on", "precedes", "contradicts", "computes"]:
    lowered = text.casefold()
    if any(token in lowered for token in ("calculate", "compute", "=", "ratio")):
        return "computes"
    if any(token in lowered for token in ("however", "but", "contradict")):
        return "contradicts"
    if any(token in lowered for token in ("then", "later", "after", "next")):
        return "precedes"
    return "depends_on"


def _normalize(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).replace("\r\n", "\n").replace("\r", "\n").strip()


def _looks_like_public_identity(value: str) -> bool:
    """Reject candidate/hypothesis labels from blinded test material."""

    return bool(re.search(r"(?i)\b(?:candidate|hypothesis)\s*H?[0-9]+\b|\bH[0-9]+\b", value))


def _json_sha256(payload: Any) -> str:
    return _sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=list))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stable_hash(seed: int, sample_id: str, role: str) -> int:
    return int(_sha256(f"{seed}\0{sample_id}\0{role}")[:16], 16)
