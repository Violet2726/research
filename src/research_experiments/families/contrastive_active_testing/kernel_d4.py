"""CATCH-Kernel D4 的可执行 jurisdiction 与 proof-carrying 路由。

D4 keeps source compilation candidate-blind and separates three claims:

1. the source was compiled into a closed ``SourceIRv2``;
2. the local solver answered that IR;
3. a frozen empirical risk gate authorizes (or refuses) an override.

The proof package is therefore conditional evidence for ``source/IR ->
answer``.  It is never represented as a proof that the IR is semantically
equivalent to the benchmark source or that the answer equals gold.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from scipy.stats import beta

from research_experiments.core.data.datasets import DatasetSample, question_without_answer_contract
from research_experiments.core.data.evaluation import canonicalize_answer
from research_experiments.families.contrastive_active_testing.certificates_v2 import build_source_span_graph
from research_experiments.families.contrastive_active_testing.kernel_d3 import (
    answer_schema_for_sample,
)
from research_experiments.families.contrastive_active_testing.kernel_d3 import (
    route_for_sample as d3_route_for_sample,
)
from research_experiments.families.contrastive_active_testing.kernel_d3 import (
    solve_exact as d3_solve_exact,
)

D4_IR_SCHEMA = "catch_source_ir_v2"
D4_IR_VERSION = "2"
D4_PROOF_SCHEMA = "catch_proof_package_v2"
D4_RISK_GATE_VERSION = "catch_d4_risk_gate_v1"
D4_SOLVER_VERSION = "catch_d4_local_solver_v1"
D4_CAPABILITY_REGISTRY_VERSION = "catch_d4_capability_registry_v1"
D4_PROMPT_VERSION = "catch_d4_candidate_blind_compiler_v1"
D4_DECODER_VERSION = "catch_d4_proof_carrying_decoder_v1"
D4_DEVELOPMENT_FAMILYWISE_ALPHA = 0.05

Route = Literal["EXACT_EXECUTABLE", "SEMANTIC_EXECUTABLE", "SOFT_UNSUPPORTED"]
SolverStatus = Literal["UNIQUE", "MULTIPLE", "UNSAT", "UNSUPPORTED"]

_IR_KEYS = {
    "capability_id",
    "query_operator",
    "entities",
    "facts",
    "events",
    "constraints",
    "query",
    "answer_contract",
    "source_span_map",
    "mandatory_spans",
    "uncovered_spans",
    "canonical_ir_hash",
}
_FORBIDDEN_IR_KEYS = {
    "stage_a",
    "stage_a_candidates",
    "candidate",
    "candidates",
    "anchor",
    "anchor_answer",
    "vote_counts",
    "votes",
    "gold",
    "reference_answer",
    "candidate_oracle",
}
_FOUNDATION_CAPABILITIES = {
    "sequence.dyck_trace_v2",
    "sequence.custom_sort_v2",
    "sequence.spatial_path_v2",
}
_SEMANTIC_CAPABILITIES = {
    "sequence.word_sort_error_trace_v1",
    "sequence.temporal_interval_trace_v1",
    "event.structured_state_ledger_v1",
    "event.musr_object_belief_ledger_v1",
    "event.musr_team_constraint_ledger_v1",
    "constraint.truth_formula_v1",
    "constraint.explicit_calculator_v1",
}
_NEW_EXACT_CAPABILITIES = {
    "sequence.shuffled_swap_v1",
    "constraint.truth_graph_v1",
}
_CAPABILITY_SPECS = {
    "sequence.dyck_trace_v2": ("sequence_trace_kernel_v1", "EXACT_EXECUTABLE", True),
    "sequence.custom_sort_v2": ("sequence_trace_kernel_v1", "EXACT_EXECUTABLE", True),
    "sequence.spatial_path_v2": ("sequence_trace_kernel_v1", "EXACT_EXECUTABLE", True),
    "sequence.shuffled_swap_v1": ("sequence_trace_kernel_v1", "EXACT_EXECUTABLE", False),
    "constraint.truth_graph_v1": ("constraint_calculator_kernel_v1", "EXACT_EXECUTABLE", False),
    "sequence.word_sort_error_trace_v1": ("sequence_trace_kernel_v1", "SEMANTIC_EXECUTABLE", False),
    "sequence.temporal_interval_trace_v1": ("sequence_trace_kernel_v1", "SEMANTIC_EXECUTABLE", False),
    "event.structured_state_ledger_v1": ("event_state_kernel_v1", "SEMANTIC_EXECUTABLE", False),
    "event.musr_object_belief_ledger_v1": ("event_state_kernel_v1", "SEMANTIC_EXECUTABLE", False),
    "event.musr_team_constraint_ledger_v1": ("event_state_kernel_v1", "SEMANTIC_EXECUTABLE", False),
    "constraint.truth_formula_v1": ("constraint_calculator_kernel_v1", "SEMANTIC_EXECUTABLE", False),
    "constraint.explicit_calculator_v1": ("constraint_calculator_kernel_v1", "SEMANTIC_EXECUTABLE", False),
}
_DEVELOPMENT_GATE_CAPABILITIES = frozenset(_NEW_EXACT_CAPABILITIES | _SEMANTIC_CAPABILITIES)
D4_DEVELOPMENT_GATE_FAMILY_SIZE = len(_DEVELOPMENT_GATE_CAPABILITIES)
_SOFT_TASKS = {
    "nycc",
    "movie_recommendation",
    "causal_understanding",
    "hyperbaton",
    "sportqa",
    "linguini",
}
_METAMORPHIC_NAMES = (
    "option_permutation",
    "entity_renaming",
    "constraint_order_permutation",
    "irrelevant_text_insertion",
    "unit_scaling",
    "reversible_event",
    "independent_event_commutation",
    "earliest_error_shift",
    "algebraic_equivalence",
    "answer_label_permutation",
)


@dataclass(frozen=True)
class D4RouteDecision:
    route: Route
    kernel_id: str
    capability_id: str
    query_operator: str
    reason: str
    foundation: bool = False


@dataclass(frozen=True)
class SourceIRv2:
    capability_id: str
    query_operator: str
    entities: tuple[dict[str, Any], ...]
    facts: tuple[dict[str, Any], ...]
    events: tuple[dict[str, Any], ...]
    constraints: tuple[dict[str, Any], ...]
    query: dict[str, Any]
    answer_contract: dict[str, Any]
    source_span_map: tuple[dict[str, str], ...]
    mandatory_spans: tuple[str, ...]
    uncovered_spans: tuple[str, ...]
    canonical_ir_hash: str


@dataclass(frozen=True)
class D4SolverResult:
    status: SolverStatus
    canonical_answer: str | None
    answer_text: str | None
    solver_trace: tuple[dict[str, Any], ...]
    reference_checker_status: str
    concrete_witness_status: dict[str, Any]
    reason: str


@dataclass(frozen=True)
class RiskGateSnapshot:
    capability_id: str
    route_specific_override_count: int
    correction_count: int
    harm_count: int
    precision_one_sided_95_lower: float | None
    harm_one_sided_95_upper: float | None
    audit_sample_count: int
    inter_rater_agreement: float | None
    metamorphic_pass_rate: float | None
    coverage: float | None
    route_activation_state: str
    confidence_alpha: float | None = None
    multiplicity_correction: str | None = None
    gate_family_size: int | None = None
    gate_version: str = D4_RISK_GATE_VERSION


@dataclass(frozen=True)
class ProofPackageV2:
    schema: str
    compiler_vote_hashes: tuple[str, ...]
    solver_status: SolverStatus
    solver_trace: tuple[dict[str, Any], ...]
    candidate_evaluation: tuple[dict[str, Any], ...]
    concrete_witness_status: dict[str, Any]
    metamorphic_transformation_status: dict[str, str]
    reference_checker_status: str
    candidate_blindness_audit: dict[str, Any]
    first_failure_layer: str
    risk_gate_version: str
    risk_gate_snapshot: dict[str, Any]
    source_hash: str
    code_hash: str


class SequenceTraceKernel:
    kernel_id = "sequence_trace_kernel_v1"

    def compile_exact(self, sample: DatasetSample, decision: D4RouteDecision) -> tuple[SourceIRv2 | None, str]:
        if decision.capability_id in _FOUNDATION_CAPABILITIES:
            return _foundation_ir(sample, decision), "ok"
        if decision.capability_id == "sequence.shuffled_swap_v1":
            return _compile_shuffled_swap(sample, decision)
        return None, "sequence_exact_capability_unsupported"

    def solve(self, sample: DatasetSample, decision: D4RouteDecision, ir: SourceIRv2) -> D4SolverResult:
        if decision.capability_id in _FOUNDATION_CAPABILITIES:
            d3 = d3_route_for_sample(sample)
            certificate = d3_solve_exact(sample, d3)
            return D4SolverResult(
                status=certificate.status,
                canonical_answer=certificate.canonical_answer,
                answer_text=certificate.canonical_answer,
                solver_trace=(
                    {
                        "step": "d3_frozen_exact_solver",
                        "operation_kind": d3.operation_kind,
                        "certificate_reason": certificate.reason,
                        "certificate_hash": _sha256(asdict(certificate)),
                    },
                ),
                reference_checker_status=certificate.cross_check_status,
                concrete_witness_status={"status": "PASSED", "kind": d3.operation_kind},
                reason=certificate.reason,
            )
        if decision.capability_id == "sequence.shuffled_swap_v1":
            return _solve_shuffled_swap(sample, ir)
        if decision.capability_id == "sequence.word_sort_error_trace_v1":
            return _solve_word_sort_trace(sample, ir)
        if decision.capability_id == "sequence.temporal_interval_trace_v1":
            return _solve_temporal_interval_trace(sample, ir)
        return _unsupported_solver("sequence_solver_capability_unsupported")


class EventStateKernel:
    kernel_id = "event_state_kernel_v1"

    def solve(self, sample: DatasetSample, decision: D4RouteDecision, ir: SourceIRv2) -> D4SolverResult:
        if decision.capability_id not in {
            "event.structured_state_ledger_v1",
            "event.musr_object_belief_ledger_v1",
            "event.musr_team_constraint_ledger_v1",
        }:
            return _unsupported_solver("event_state_capability_unsupported")
        return _solve_event_state_ledger(sample, decision, ir)


class ConstraintCalculatorKernel:
    kernel_id = "constraint_calculator_kernel_v1"

    def compile_exact(self, sample: DatasetSample, decision: D4RouteDecision) -> tuple[SourceIRv2 | None, str]:
        if decision.capability_id == "constraint.truth_graph_v1":
            return _compile_truth_graph(sample, decision)
        return None, "constraint_exact_capability_unsupported"

    def solve(self, sample: DatasetSample, decision: D4RouteDecision, ir: SourceIRv2) -> D4SolverResult:
        if decision.capability_id == "constraint.truth_graph_v1":
            return _solve_truth_graph(sample, ir)
        if decision.query_operator == "evaluate_numeric_expression":
            return _solve_numeric_expression(sample, ir)
        return _unsupported_solver("constraint_solver_operator_unsupported")


def route_for_sample(sample: DatasetSample) -> D4RouteDecision:
    """Route using source/query signatures before any Stage-A value exists."""

    source = question_without_answer_contract(sample)
    lowered = source.casefold()
    task = str(sample.metadata.get("task") or "").strip().casefold()
    if task in _SOFT_TASKS:
        return D4RouteDecision("SOFT_UNSUPPORTED", "none", "soft.open_world_v1", "none", "frozen_soft_policy")

    d3 = d3_route_for_sample(sample)
    if d3.route == "EXACT_EXECUTABLE":
        capability = {
            "stack_trace": "sequence.dyck_trace_v2",
            "custom_sort_order": "sequence.custom_sort_v2",
            "grid_path": "sequence.spatial_path_v2",
        }.get(d3.operation_kind)
        if capability:
            return D4RouteDecision(
                "EXACT_EXECUTABLE",
                SequenceTraceKernel.kernel_id,
                capability,
                d3.operation_kind,
                "frozen_d3_exact_foundation",
                foundation=True,
            )
    if _looks_like_shuffled_swap(lowered):
        return D4RouteDecision(
            "EXACT_EXECUTABLE",
            SequenceTraceKernel.kernel_id,
            "sequence.shuffled_swap_v1",
            "final_state_after_ordered_swaps",
            "source_signature_shuffled_swap",
        )
    if _looks_like_truth_graph(lowered):
        return D4RouteDecision(
            "EXACT_EXECUTABLE",
            ConstraintCalculatorKernel.kernel_id,
            "constraint.truth_graph_v1",
            "truth_assignment",
            "source_signature_truth_graph",
        )
    if "assume each person either always tells the truth or always lies" in lowered:
        return D4RouteDecision(
            "SEMANTIC_EXECUTABLE",
            ConstraintCalculatorKernel.kernel_id,
            "constraint.truth_formula_v1",
            "truth_assignment",
            "source_signature_general_truth_formula",
        )
    if _looks_like_word_sort_trace(lowered):
        return D4RouteDecision(
            "SEMANTIC_EXECUTABLE",
            SequenceTraceKernel.kernel_id,
            "sequence.word_sort_error_trace_v1",
            "earliest_trace_divergence",
            "source_signature_word_sort_trace",
        )
    if _looks_like_temporal_schedule(lowered):
        return D4RouteDecision(
            "SEMANTIC_EXECUTABLE",
            SequenceTraceKernel.kernel_id,
            "sequence.temporal_interval_trace_v1",
            "longest_feasible_interval",
            "source_signature_temporal_schedule",
        )
    if _looks_like_event_state(sample, lowered):
        capability = (
            "event.musr_object_belief_ledger_v1"
            if sample.dataset in {"musr", "musr_x"} and task == "object_placements"
            else "event.musr_team_constraint_ledger_v1"
            if sample.dataset in {"musr", "musr_x"} and task == "team_allocation"
            else "event.structured_state_ledger_v1"
        )
        operator = "belief_state_at_query_time" if "object" in capability else "constraint_state_at_query_time"
        return D4RouteDecision(
            "SEMANTIC_EXECUTABLE",
            EventStateKernel.kernel_id,
            capability,
            operator,
            "source_signature_event_state",
        )
    if _looks_like_calculator(sample, lowered):
        return D4RouteDecision(
            "SEMANTIC_EXECUTABLE",
            ConstraintCalculatorKernel.kernel_id,
            "constraint.explicit_calculator_v1",
            "evaluate_numeric_expression",
            "source_signature_explicit_calculation",
        )
    return D4RouteDecision("SOFT_UNSUPPORTED", "none", "soft.unsupported_v1", "none", "no_closed_operator")


def capability_registry() -> dict[str, Any]:
    return {
        "version": D4_CAPABILITY_REGISTRY_VERSION,
        "routing_principle": "source_signature_plus_query_operator_not_stage_a_candidates",
        "kernels": {
            SequenceTraceKernel.kernel_id: {
                "foundation_exact": sorted(_FOUNDATION_CAPABILITIES),
                "new_exact_shadow_until_gate": ["sequence.shuffled_swap_v1"],
                "semantic_shadow_until_gate": [
                    "sequence.word_sort_error_trace_v1",
                    "sequence.temporal_interval_trace_v1",
                ],
            },
            EventStateKernel.kernel_id: {
                "semantic_shadow_until_gate": [
                    "event.structured_state_ledger_v1",
                    "event.musr_object_belief_ledger_v1",
                    "event.musr_team_constraint_ledger_v1",
                ],
                "murder_mysteries": "shadow_only_commonsense_boundary",
            },
            ConstraintCalculatorKernel.kernel_id: {
                "new_exact_shadow_until_gate": ["constraint.truth_graph_v1"],
                "semantic_shadow_until_gate": [
                    "constraint.explicit_calculator_v1",
                    "constraint.truth_formula_v1",
                ],
                "retrieval": "forbidden",
            },
        },
        "development_gate": {
            "capabilities": sorted(_DEVELOPMENT_GATE_CAPABILITIES),
            "family_size": D4_DEVELOPMENT_GATE_FAMILY_SIZE,
            "familywise_alpha": D4_DEVELOPMENT_FAMILYWISE_ALPHA,
            "correction": "bonferroni_fixed_preregistered_capability_family",
        },
        "soft_tasks": sorted(_SOFT_TASKS),
    }


def capability_spec(capability_id: str) -> dict[str, Any] | None:
    """Return the frozen route/kernel identity used by evidence validators."""

    raw = _CAPABILITY_SPECS.get(str(capability_id))
    if raw is None:
        return None
    kernel_id, route, foundation = raw
    return {"kernel_id": kernel_id, "route": route, "foundation": foundation}


def parse_source_ir_v2(
    payload: Any,
    *,
    sample: DatasetSample,
    decision: D4RouteDecision,
    require_complete_provenance: bool = True,
) -> tuple[SourceIRv2 | None, str]:
    if not isinstance(payload, dict):
        return None, "source_ir_v2_not_object"
    if set(payload) != _IR_KEYS:
        return None, "source_ir_v2_keys_invalid"
    leaked = sorted(_find_forbidden_keys(payload))
    if leaked:
        return None, "source_ir_v2_candidate_leakage:" + ",".join(leaked)
    if str(payload.get("capability_id") or "") != decision.capability_id:
        return None, "source_ir_v2_capability_mismatch"
    if str(payload.get("query_operator") or "") != decision.query_operator:
        return None, "source_ir_v2_query_operator_mismatch"
    if payload.get("canonical_ir_hash") not in {"", None}:
        return None, "source_ir_v2_untrusted_hash_must_be_empty"

    graph = build_source_span_graph(sample)
    expected_spans = {span.span_id: span.text for span in graph.spans}
    span_map = payload.get("source_span_map")
    if not isinstance(span_map, list) or any(
        not isinstance(item, dict) or set(item) != {"span_id", "text"} for item in span_map
    ):
        return None, "source_ir_v2_span_map_invalid"
    normalized_span_map = tuple(
        {"span_id": str(item["span_id"]), "text": str(item["text"])} for item in span_map
    )
    if (
        len(normalized_span_map) != len(expected_spans)
        or len({item["span_id"] for item in normalized_span_map}) != len(normalized_span_map)
        or {item["span_id"]: item["text"] for item in normalized_span_map} != expected_spans
    ):
        return None, "source_ir_v2_span_map_not_exact_source"
    mandatory = _string_tuple(payload.get("mandatory_spans"))
    uncovered = _string_tuple(payload.get("uncovered_spans"))
    known = set(expected_spans)
    if (
        not mandatory
        or len(mandatory) != len(set(mandatory))
        or len(uncovered) != len(set(uncovered))
        or set(mandatory) - known
        or set(uncovered) - known
        or set(mandatory) & set(uncovered)
        or set(mandatory) | set(uncovered) != known
    ):
        return None, "source_ir_v2_span_partition_invalid"
    referenced = _referenced_span_ids(payload)
    if referenced - known:
        return None, "source_ir_v2_unknown_provenance_span"
    if require_complete_provenance and (
        not _decisive_records_have_provenance(payload)
        or not set(mandatory).issubset(referenced)
        or bool(referenced & set(uncovered))
    ):
        return None, "source_ir_v2_mandatory_provenance_uncovered"

    expected_contract = _answer_contract_for_sample(sample)
    if payload.get("answer_contract") != expected_contract:
        return None, "source_ir_v2_answer_contract_mismatch"
    collections: dict[str, tuple[dict[str, Any], ...]] = {}
    for name in ("entities", "facts", "events", "constraints"):
        value = payload.get(name)
        if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
            return None, f"source_ir_v2_{name}_invalid"
        collections[name] = tuple(dict(item) for item in value)
    query = payload.get("query")
    if not isinstance(query, dict) or not query:
        return None, "source_ir_v2_query_invalid"

    ir = SourceIRv2(
        capability_id=decision.capability_id,
        query_operator=decision.query_operator,
        entities=collections["entities"],
        facts=collections["facts"],
        events=collections["events"],
        constraints=collections["constraints"],
        query=dict(query),
        answer_contract=expected_contract,
        source_span_map=normalized_span_map,
        mandatory_spans=tuple(sorted(mandatory)),
        uncovered_spans=tuple(sorted(uncovered)),
        canonical_ir_hash="",
    )
    return replace(ir, canonical_ir_hash=canonical_ir_hash(ir)), "ok"


def canonical_ir_hash(ir: SourceIRv2) -> str:
    payload = asdict(ir)
    payload["canonical_ir_hash"] = ""
    payload["entities"] = sorted(payload["entities"], key=_canonical_sort_key)
    payload["facts"] = sorted(payload["facts"], key=_canonical_sort_key)
    payload["constraints"] = sorted(payload["constraints"], key=_canonical_sort_key)
    payload["mandatory_spans"] = sorted(payload["mandatory_spans"])
    payload["uncovered_spans"] = sorted(payload["uncovered_spans"])
    return _sha256(payload)


def solve_source_ir(sample: DatasetSample, decision: D4RouteDecision, ir: SourceIRv2) -> D4SolverResult:
    if decision.kernel_id == SequenceTraceKernel.kernel_id:
        return SequenceTraceKernel().solve(sample, decision, ir)
    if decision.kernel_id == EventStateKernel.kernel_id:
        return EventStateKernel().solve(sample, decision, ir)
    if decision.kernel_id == ConstraintCalculatorKernel.kernel_id:
        return ConstraintCalculatorKernel().solve(sample, decision, ir)
    return _unsupported_solver("kernel_id_unregistered")


def compile_exact_source_ir(sample: DatasetSample, decision: D4RouteDecision) -> tuple[SourceIRv2 | None, str]:
    if decision.kernel_id == SequenceTraceKernel.kernel_id:
        return SequenceTraceKernel().compile_exact(sample, decision)
    if decision.kernel_id == ConstraintCalculatorKernel.kernel_id:
        return ConstraintCalculatorKernel().compile_exact(sample, decision)
    return None, "exact_kernel_id_unregistered"


def evaluate_candidates(
    sample: DatasetSample,
    candidates: list[str],
    solver: D4SolverResult,
) -> tuple[dict[str, Any], ...]:
    output = []
    for answer in candidates:
        canonical = canonicalize_answer(sample, answer)
        status = (
            "UNSUPPORTED"
            if not canonical.valid or not canonical.key or solver.status != "UNIQUE" or not solver.canonical_answer
            else "VALID"
            if canonical.key == solver.canonical_answer
            else "INVALID"
        )
        output.append(
            {
                "candidate_hash": _sha256(str(answer)),
                "canonical_answer": canonical.key if canonical.valid else None,
                "status": status,
                "trace_hash": _sha256(
                    {"candidate": canonical.key if canonical.valid else None, "solver": solver.canonical_answer}
                ),
            }
        )
    return tuple(output)


def risk_gate_snapshot(
    capability_id: str,
    *,
    route: Route,
    evidence: dict[str, Any] | None = None,
    phase: str = "development",
) -> RiskGateSnapshot:
    if capability_id in _FOUNDATION_CAPABILITIES:
        return RiskGateSnapshot(
            capability_id=capability_id,
            route_specific_override_count=0,
            correction_count=0,
            harm_count=0,
            precision_one_sided_95_lower=None,
            harm_one_sided_95_upper=None,
            audit_sample_count=0,
            inter_rater_agreement=None,
            metamorphic_pass_rate=1.0,
            coverage=None,
            route_activation_state="ACTIVE_FROZEN_D3_EXACT_FOUNDATION",
        )
    row = dict((evidence or {}).get("capabilities", {}).get(capability_id) or {})
    overrides = max(0, int(row.get("override_count") or 0))
    corrections = max(0, int(row.get("correction_count") or 0))
    harms = max(0, int(row.get("harm_count") or 0))
    correct_overrides = max(0, int(row.get("correct_override_count", corrections) or 0))
    audit_n = max(0, int(row.get("audit_sample_count") or 0))
    agreement = _optional_float(row.get("inter_rater_agreement"))
    meta_rate = _optional_float(row.get("metamorphic_pass_rate"))
    coverage = _optional_float(row.get("coverage"))
    alpha = (
        D4_DEVELOPMENT_FAMILYWISE_ALPHA / D4_DEVELOPMENT_GATE_FAMILY_SIZE
        if phase == "development"
        else 0.05
    )
    precision_lower = _clopper_lower(correct_overrides, overrides, alpha=alpha) if overrides else None
    harm_upper = _clopper_upper(harms, overrides, alpha=alpha) if overrides else None
    minimum_overrides = 59 if phase == "confirmation" else _minimum_zero_failure_trials_for_lower(
        0.90,
        alpha=alpha,
    )
    precision_threshold = 0.95 if phase == "confirmation" else 0.90
    counts_consistent = (
        corrections <= overrides
        and harms <= overrides
        and correct_overrides <= overrides
        and corrections + harms <= overrides
        and correct_overrides == corrections
    )
    conditions = {
        "evidence_frozen": bool(row.get("evidence_frozen")),
        "counts_consistent": counts_consistent,
        "override_count": overrides >= minimum_overrides,
        "precision": precision_lower is not None and precision_lower >= precision_threshold,
        "harm": phase != "confirmation" or (harm_upper is not None and harm_upper <= 0.05),
        "metamorphic": meta_rate is not None and meta_rate >= 1.0,
    }
    if capability_id in _SEMANTIC_CAPABILITIES or route == "SEMANTIC_EXECUTABLE":
        conditions.update(
            {
                "audit_count": audit_n >= 60,
                "agreement": agreement is not None
                and agreement >= 0.80
                and row.get("inter_rater_agreement_metric") == "gwet_ac1",
                "critical_error": float(row.get("critical_semantic_error_rate", 1.0)) <= 0.02,
                "validity": float(row.get("adjudicated_ir_validity") or 0.0) >= 0.95,
                "false_pass": int(row.get("unexplained_high_severity_false_pass") or 0) == 0,
            }
        )
    active = all(conditions.values()) and route != "SOFT_UNSUPPORTED"
    return RiskGateSnapshot(
        capability_id=capability_id,
        route_specific_override_count=overrides,
        correction_count=corrections,
        harm_count=harms,
        precision_one_sided_95_lower=precision_lower,
        harm_one_sided_95_upper=harm_upper,
        audit_sample_count=audit_n,
        inter_rater_agreement=agreement,
        metamorphic_pass_rate=meta_rate,
        coverage=coverage,
        route_activation_state="ACTIVE" if active else "SHADOW_GATE_NOT_MET",
        confidence_alpha=alpha,
        multiplicity_correction=(
            "bonferroni_fixed_preregistered_capability_family" if phase == "development" else "none"
        ),
        gate_family_size=D4_DEVELOPMENT_GATE_FAMILY_SIZE if phase == "development" else 1,
    )


def load_risk_evidence(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {"schema": "catch_d4_gate_evidence_v1", "capabilities": {}}
    target = Path(path)
    if not target.exists():
        return {"schema": "catch_d4_gate_evidence_v1", "capabilities": {}, "missing_path": target.as_posix()}
    stat = target.stat()
    return _load_risk_evidence_file(
        target.resolve().as_posix(),
        stat.st_mtime_ns,
        stat.st_size,
    )


@lru_cache(maxsize=16)
def _load_risk_evidence_file(path: str, _mtime_ns: int, _size: int) -> dict[str, Any]:
    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if payload.get("schema") != "catch_d4_gate_evidence_v1" or not isinstance(payload.get("capabilities"), dict):
        raise ValueError("Invalid D4 risk evidence schema.")
    if payload.get("status") == "template_not_passing" and not payload.get("capabilities"):
        return payload
    from research_experiments.families.contrastive_active_testing.d4_audit import (
        validate_d4_gate_evidence,
    )

    validation = validate_d4_gate_evidence(payload, verify_source_files=True)
    if not validation.get("passed"):
        raise ValueError(f"Invalid D4 frozen risk evidence: {validation.get('conditions', {})}")
    return payload


def run_metamorphic_checks(
    ir: SourceIRv2,
    solver: D4SolverResult,
    *,
    sample: DatasetSample | None = None,
    decision: D4RouteDecision | None = None,
) -> dict[str, str]:
    status = {name: "NOT_APPLICABLE" for name in _METAMORPHIC_NAMES}
    status["option_permutation"] = _option_contract_invariant(
        ir,
        solver,
        sample=sample,
        decision=decision,
    )
    status["answer_label_permutation"] = _answer_label_permutation_check(
        ir,
        solver,
        sample=sample,
        decision=decision,
    )
    if ir.capability_id == "sequence.shuffled_swap_v1":
        status["entity_renaming"] = _shuffled_entity_renaming_check(ir, solver)
        status["reversible_event"] = _shuffled_reversible_event_check(ir, solver)
        status["independent_event_commutation"] = _shuffled_commutation_check(ir, solver)
    elif ir.capability_id == "constraint.truth_graph_v1":
        status["entity_renaming"] = _truth_entity_renaming_check(ir, solver)
        status["constraint_order_permutation"] = _truth_constraint_order_check(ir, solver)
    elif ir.query_operator == "evaluate_numeric_expression":
        status["algebraic_equivalence"] = _numeric_algebraic_equivalence_check(ir, solver)
    return status


def metamorphic_checks_passed(
    ir: SourceIRv2,
    solver: D4SolverResult,
    status: dict[str, str],
) -> bool:
    """Require at least one executed, passing relation for every new route.

    Frozen D3 foundation routes keep their independently audited behavior.  A
    dictionary containing only ``NOT_APPLICABLE`` values is not evidence that
    a new per-item metamorphic check ran.
    """

    if ir.capability_id in _FOUNDATION_CAPABILITIES:
        return solver.status == "UNIQUE" and not any(value == "FAILED" for value in status.values())
    executed = [value for value in status.values() if value not in {"NOT_APPLICABLE", "NOT_RUN"}]
    return bool(executed) and all(value == "PASSED" for value in executed)


def build_proof_package(
    *,
    sample: DatasetSample,
    ir: SourceIRv2 | None,
    solver: D4SolverResult,
    compiler_vote_hashes: tuple[str, ...],
    candidate_evaluation: tuple[dict[str, Any], ...],
    metamorphic_status: dict[str, str],
    risk_snapshot: RiskGateSnapshot,
    compiler_input_fields: tuple[str, ...],
    first_failure_layer: str,
) -> ProofPackageV2:
    return ProofPackageV2(
        schema=D4_PROOF_SCHEMA,
        compiler_vote_hashes=compiler_vote_hashes,
        solver_status=solver.status,
        solver_trace=solver.solver_trace,
        candidate_evaluation=candidate_evaluation,
        concrete_witness_status=dict(solver.concrete_witness_status),
        metamorphic_transformation_status=dict(metamorphic_status),
        reference_checker_status=solver.reference_checker_status,
        candidate_blindness_audit={
            "input_fields": list(compiler_input_fields),
            "forbidden_fields": sorted(_FORBIDDEN_IR_KEYS),
            "passed": not bool(set(compiler_input_fields) & _FORBIDDEN_IR_KEYS),
        },
        first_failure_layer=first_failure_layer,
        risk_gate_version=risk_snapshot.gate_version,
        risk_gate_snapshot=asdict(risk_snapshot),
        source_hash=_sha256(question_without_answer_contract(sample)),
        code_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )


def source_ir_to_dict(ir: SourceIRv2) -> dict[str, Any]:
    return asdict(ir)


def answer_contract_for_sample(sample: DatasetSample) -> dict[str, Any]:
    return _answer_contract_for_sample(sample)


def solver_result_to_dict(result: D4SolverResult) -> dict[str, Any]:
    return asdict(result)


def proof_package_to_dict(package: ProofPackageV2) -> dict[str, Any]:
    return asdict(package)


def risk_gate_to_dict(snapshot: RiskGateSnapshot) -> dict[str, Any]:
    return asdict(snapshot)


def _foundation_ir(sample: DatasetSample, decision: D4RouteDecision) -> SourceIRv2:
    graph = build_source_span_graph(sample)
    spans = tuple({"span_id": span.span_id, "text": span.text} for span in graph.spans)
    mandatory = tuple(item["span_id"] for item in spans)
    ir = SourceIRv2(
        capability_id=decision.capability_id,
        query_operator=decision.query_operator,
        entities=(),
        facts=({"kind": "source_bound_exact_parser", "source_span_ids": list(mandatory)},),
        events=(),
        constraints=(),
        query={"kind": decision.query_operator, "source_span_ids": list(mandatory)},
        answer_contract=_answer_contract_for_sample(sample),
        source_span_map=spans,
        mandatory_spans=mandatory,
        uncovered_spans=(),
        canonical_ir_hash="",
    )
    return replace(ir, canonical_ir_hash=canonical_ir_hash(ir))


def _compile_shuffled_swap(
    sample: DatasetSample,
    decision: D4RouteDecision,
) -> tuple[SourceIRv2 | None, str]:
    source = question_without_answer_contract(sample)
    prefix_match = re.match(r"(?s)^(.+?) are .+?\.\s+At the start[^:]*:\s*(.+?)\.\s+", source)
    if prefix_match is None:
        return None, "shuffled_initial_clause_missing"
    participants = _name_list(prefix_match.group(1))
    if len(participants) < 3:
        return None, "shuffled_participants_invalid"
    initial_clause = prefix_match.group(2)
    initial: dict[str, str] = {}
    initial_ranges: dict[str, tuple[int, int]] = {}
    for index, name in enumerate(participants):
        following = participants[index + 1] if index + 1 < len(participants) else None
        boundary = rf",\s+(?:and\s+)?{re.escape(following)}\b" if following else r"$"
        match = re.search(
            rf"\b{re.escape(name)}\s+(?:gets|has|is dancing with|is playing)\s+(.+?)(?={boundary})",
            initial_clause,
        )
        if match is None:
            return None, f"shuffled_initial_assignment_missing:{name}"
        initial[name] = _strip_indefinite_article(match.group(1).strip())
        initial_ranges[name] = (
            prefix_match.start(2) + match.start(),
            prefix_match.start(2) + match.end(),
        )
    if len(set(initial.values())) != len(initial):
        return None, "shuffled_initial_values_not_unique"

    query_match = re.search(
        rf"At the end[^,]*,\s*({'|'.join(map(re.escape, participants))})\s+(?:has|is dancing with|is playing)\b",
        source,
    )
    if query_match is None:
        return None, "shuffled_query_entity_missing"
    query_entity = query_match.group(1)
    action_text = source[prefix_match.end() : query_match.start()]
    graph = build_source_span_graph(sample)
    definitions: dict[str, tuple[tuple[str, str] | None, tuple[str, ...]]] = {}
    events: list[dict[str, Any]] = []
    for sentence_match in re.finditer(r".+?(?:[.!?](?=\s|$)|$)", action_text):
        sentence = sentence_match.group(0).strip()
        if not sentence:
            continue
        repeat = re.search(r"\bAction\s+(\d+)\s+repeats\b", sentence, re.IGNORECASE)
        label = re.search(r"let's call it Action\s+(\d+)", sentence, re.IGNORECASE)
        pair = re.search(
            rf"\b({'|'.join(map(re.escape, participants))})\s+and\s+"
            rf"({'|'.join(map(re.escape, participants))})\s+"
            r"(?:swap|switch|trade)\b",
            sentence,
            re.IGNORECASE,
        )
        no_op = re.search(
            rf"\b(?:nothing happens for a while|"
            rf"(?:{'|'.join(map(re.escape, participants))})\s+and\s+"
            rf"(?:{'|'.join(map(re.escape, participants))})\s+discuss something)\b",
            sentence,
            re.IGNORECASE,
        )
        preamble = re.fullmatch(
            r"(?:As the semester proceeds, they start trading around the new books|"
            r"Throughout the song, the dancers often trade partners|"
            r"As the game progresses, pairs of players trade balls|"
            r"As the event progresses, pairs of people swap gifts|"
            r"As the game progresses, pairs of players occasionally swap positions)\.",
            sentence,
            re.IGNORECASE,
        )
        operation: tuple[str, str] | None
        definition_span_ids: tuple[str, ...] = ()
        if repeat:
            key = repeat.group(1)
            if key not in definitions:
                return None, f"shuffled_action_repeat_undefined:{key}"
            operation, definition_span_ids = definitions[key]
        elif pair:
            lookup = {name.casefold(): name for name in participants}
            left = lookup.get(pair.group(1).casefold())
            right = lookup.get(pair.group(2).casefold())
            if not left or not right or left == right:
                return None, "shuffled_swap_entities_invalid"
            operation = (left, right)
        elif no_op or preamble:
            operation = None
        else:
            return None, "shuffled_action_sentence_unparsed"
        sentence_span_ids = _span_ids_for_range(
            graph,
            prefix_match.end() + sentence_match.start(),
            prefix_match.end() + sentence_match.end(),
        )
        if label:
            key = label.group(1)
            if key in definitions:
                return None, f"shuffled_action_duplicate_definition:{key}"
            definitions[key] = (operation, sentence_span_ids)
        if operation is not None:
            events.append(
                {
                    "event_id": f"E{len(events)}",
                    "kind": "swap",
                    "left": operation[0],
                    "right": operation[1],
                    "source_span_ids": list(dict.fromkeys((*definition_span_ids, *sentence_span_ids))),
                }
            )
    if not events:
        return None, "shuffled_no_swap_events"
    spans = tuple({"span_id": span.span_id, "text": span.text} for span in graph.spans)
    facts = tuple(
        {
            "fact_id": f"F{index}",
            "kind": "initial_holder_value",
            "entity": name,
            "value": initial[name],
            "source_span_ids": list(_span_ids_for_range(graph, *initial_ranges[name])),
        }
        for index, name in enumerate(participants)
    )
    query_span_ids = _span_ids_for_range(graph, query_match.start(), query_match.end())
    mandatory_set = {
        *(span_id for row in facts for span_id in row["source_span_ids"]),
        *(span_id for row in events for span_id in row["source_span_ids"]),
        *query_span_ids,
    }
    mandatory = tuple(span.span_id for span in graph.spans if span.span_id in mandatory_set)
    uncovered = tuple(span.span_id for span in graph.spans if span.span_id not in mandatory_set)
    ir = SourceIRv2(
        capability_id=decision.capability_id,
        query_operator=decision.query_operator,
        entities=tuple({"entity_id": name, "kind": "holder"} for name in participants),
        facts=facts,
        events=tuple(events),
        constraints=(),
        query={
            "kind": "value_held_by_entity_after_all_events",
            "entity": query_entity,
            "source_span_ids": list(query_span_ids),
        },
        answer_contract=_answer_contract_for_sample(sample),
        source_span_map=spans,
        mandatory_spans=mandatory,
        uncovered_spans=uncovered,
        canonical_ir_hash="",
    )
    return replace(ir, canonical_ir_hash=canonical_ir_hash(ir)), "ok"


def _solve_shuffled_swap(sample: DatasetSample, ir: SourceIRv2) -> D4SolverResult:
    state = {
        str(fact.get("entity")): str(fact.get("value"))
        for fact in ir.facts
        if fact.get("kind") == "initial_holder_value"
    }
    trace: list[dict[str, Any]] = []
    for event in ir.events:
        if event.get("kind") != "swap":
            return _unsupported_solver("shuffled_event_kind_invalid")
        left, right = str(event.get("left") or ""), str(event.get("right") or "")
        if left not in state or right not in state or left == right:
            return _unsupported_solver("shuffled_event_entity_invalid")
        state[left], state[right] = state[right], state[left]
        trace.append(
            {
                "event_id": event.get("event_id"),
                "kind": "swap",
                "state_hash": _sha256(state),
            }
        )
    query_entity = str(ir.query.get("entity") or "")
    if query_entity not in state:
        return _unsupported_solver("shuffled_query_entity_invalid")
    answer_text = state[query_entity]
    canonical = canonicalize_answer(sample, answer_text)
    if not canonical.valid or not canonical.key:
        return _unsupported_solver("shuffled_answer_outside_contract")
    return D4SolverResult(
        status="UNIQUE",
        canonical_answer=canonical.key,
        answer_text=answer_text,
        solver_trace=tuple(trace),
        reference_checker_status="PASSED_BIJECTIVE_STATE_REFERENCE_CHECKER",
        concrete_witness_status={
            "status": "PASSED",
            "query_entity": query_entity,
            "final_value_hash": _sha256(answer_text),
            "event_count": len(ir.events),
        },
        reason="ordered_swap_trace_unique",
    )


def _solve_event_state_ledger(
    sample: DatasetSample,
    decision: D4RouteDecision,
    ir: SourceIRv2,
) -> D4SolverResult:
    """Solve a conservative typed event ledger emitted by the blind compiler.

    The operator deliberately accepts only a small closed vocabulary. It does
    not infer events from prose and therefore cannot turn an unverified parser
    into an exact route; semantic activation still requires the independent IR
    audit and risk gate.  The benefit is that a valid audited ledger now has a
    deterministic local execution path instead of an unconditional unsupported
    result.
    """

    capability = decision.capability_id
    if capability == "event.structured_state_ledger_v1":
        fact_kinds = {"state_initial"}
        event_kinds = {"state_set", "state_delete", "state_append"}
        query_kinds = {"state_value", "state_membership", "state_count"}
    elif capability == "event.musr_object_belief_ledger_v1":
        fact_kinds = {"object_location_initial"}
        event_kinds = {"object_move", "object_location_set"}
        query_kinds = {"object_location"}
    else:
        fact_kinds = {"team_assignment_initial"}
        event_kinds = {"team_assign"}
        query_kinds = {"team_assignment"}

    state: dict[str, Any] = {}
    trace: list[dict[str, Any]] = []
    for fact in ir.facts:
        kind = str(fact.get("kind") or "")
        if kind not in fact_kinds:
            return _unsupported_solver(f"event_state_fact_kind_unsupported:{kind or 'missing'}")
        key = str(fact.get("key") or fact.get("entity") or fact.get("object") or fact.get("person") or "")
        if not key or key in state:
            return _unsupported_solver("event_state_duplicate_or_empty_initial_key")
        if "value" not in fact and "location" not in fact and "task" not in fact:
            return _unsupported_solver("event_state_initial_value_missing")
        value = fact.get("value", fact.get("location", fact.get("task")))
        state[key] = value

    for index, event in enumerate(ir.events):
        kind = str(event.get("kind") or "")
        key = str(event.get("key") or event.get("entity") or event.get("object") or event.get("person") or "")
        if kind not in event_kinds or not key:
            return _unsupported_solver(f"event_state_event_invalid:{kind or 'missing'}")
        if kind in {"state_set", "state_append", "object_location_set", "team_assign"}:
            if "value" in event:
                value = event["value"]
            elif "location" in event:
                value = event["location"]
            elif "task" in event:
                value = event["task"]
            else:
                return _unsupported_solver("event_state_event_value_missing")
            if kind == "state_append":
                if key not in state:
                    return _unsupported_solver("event_state_append_unknown_key")
                current = state[key]
                if not isinstance(current, list):
                    return _unsupported_solver("event_state_append_target_not_list")
                state[key] = [*current, value]
            else:
                state[key] = value
        elif kind == "object_move":
            if key not in state:
                return _unsupported_solver("event_state_move_unknown_object")
            destination = str(event.get("to") or event.get("location") or "").strip()
            if not destination:
                return _unsupported_solver("event_state_move_destination_missing")
            state[key] = destination
        elif kind == "state_delete":
            if key not in state:
                return _unsupported_solver("event_state_delete_unknown_key")
            del state[key]
        trace.append(
            {
                "event_index": index,
                "event_kind": kind,
                "state_hash": _sha256(state),
            }
        )

    for constraint in ir.constraints:
        kind = str(constraint.get("kind") or "")
        if kind == "all_different":
            keys = [str(value) for value in constraint.get("keys") or []]
            values = [state.get(key) for key in keys]
            if not keys or len(keys) != len(set(keys)) or any(key not in state for key in keys):
                return _unsupported_solver("event_state_all_different_keys_invalid")
            if len(values) != len(set(json.dumps(value, sort_keys=True) for value in values)):
                return _unsupported_solver("event_state_all_different_constraint_failed")
        elif kind == "domain":
            key = str(constraint.get("key") or "")
            domain = constraint.get("values")
            if key not in state or not isinstance(domain, list) or state[key] not in domain:
                return _unsupported_solver("event_state_domain_constraint_failed")
        else:
            return _unsupported_solver(f"event_state_constraint_unsupported:{kind or 'missing'}")

    query_kind = str(ir.query.get("kind") or "")
    if query_kind not in query_kinds:
        return _unsupported_solver(f"event_state_query_kind_unsupported:{query_kind or 'missing'}")
    key = str(ir.query.get("key") or ir.query.get("entity") or ir.query.get("object") or ir.query.get("person") or "")
    if not key:
        return _unsupported_solver("event_state_query_key_missing")
    if query_kind == "state_membership":
        value = key in state
    elif query_kind == "state_count":
        current = state.get(key)
        if not isinstance(current, (list, dict, str)):
            return _unsupported_solver("event_state_count_target_invalid")
        value = len(current)
    else:
        if key not in state:
            if "default" not in ir.query:
                return _unsupported_solver("event_state_query_unknown_key")
            value = ir.query["default"]
        else:
            value = state[key]
    answer_text = _event_state_answer_text(sample, value)
    canonical = canonicalize_answer(sample, answer_text)
    if not canonical.valid or not canonical.key:
        return _unsupported_solver("event_state_answer_outside_contract")
    return D4SolverResult(
        status="UNIQUE",
        canonical_answer=canonical.key,
        answer_text=answer_text,
        solver_trace=tuple(trace),
        reference_checker_status="PASSED_TYPED_EVENT_STATE_CHECKER",
        concrete_witness_status={
            "status": "PASSED",
            "kind": capability,
            "event_count": len(ir.events),
            "state_key_count": len(state),
        },
        reason="typed_event_state_ledger_unique",
    )


def _event_state_answer_text(sample: DatasetSample, value: Any) -> str:
    if isinstance(value, bool):
        candidates = ("true", "yes") if value else ("false", "no")
        contract = sample.metadata.get("answer_contract")
        options = contract.get("options") if isinstance(contract, dict) else []
        for option in options if isinstance(options, list) else []:
            text = str(option.get("text") or "") if isinstance(option, dict) else str(option)
            if text.casefold() in candidates:
                return text
        return candidates[0]
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


def _solve_word_sort_trace(sample: DatasetSample, ir: SourceIRv2) -> D4SolverResult:
    """Check a compiler-emitted word-sort trace without sorting prose itself."""

    if len(ir.facts) != 1 or str(ir.facts[0].get("kind") or "") != "word_sort_target":
        return _unsupported_solver("word_sort_target_fact_missing")
    words = ir.facts[0].get("words")
    if not isinstance(words, list) or not words or any(not isinstance(word, str) or not word.strip() for word in words):
        return _unsupported_solver("word_sort_target_words_invalid")
    expected_multiset = sorted(str(word).casefold() for word in words)
    first_error: int | None = None
    trace: list[dict[str, Any]] = []
    for expected_index, event in enumerate(ir.events, start=1):
        if str(event.get("kind") or "") != "word_sort_step" or int(event.get("step_index") or 0) != expected_index:
            return _unsupported_solver("word_sort_step_sequence_invalid")
        observed = event.get("observed")
        expected = event.get("expected")
        if not isinstance(observed, list) or not isinstance(expected, list):
            return _unsupported_solver("word_sort_step_lists_invalid")
        observed_text = [str(word).strip() for word in observed]
        expected_text = [str(word).strip() for word in expected]
        if sorted(observed_text, key=str.casefold) != expected_multiset or sorted(expected_text, key=str.casefold) != expected_multiset:
            return _unsupported_solver("word_sort_step_word_set_invalid")
        mismatch = observed_text != expected_text
        if mismatch and first_error is None:
            first_error = expected_index
        trace.append(
            {
                "step_index": expected_index,
                "matches_expected": not mismatch,
                "observed_hash": _sha256(observed_text),
                "expected_hash": _sha256(expected_text),
            }
        )
    if first_error is None:
        no_error = ir.query.get("no_error_value")
        if no_error is None:
            return _unsupported_solver("word_sort_no_error_value_missing")
        answer_text = str(no_error)
    else:
        answer_text = str(first_error)
    canonical = canonicalize_answer(sample, answer_text)
    if not canonical.valid or not canonical.key:
        return _unsupported_solver("word_sort_answer_outside_contract")
    return D4SolverResult(
        status="UNIQUE",
        canonical_answer=canonical.key,
        answer_text=answer_text,
        solver_trace=tuple(trace),
        reference_checker_status="PASSED_WORD_SORT_TRACE_CHECKER",
        concrete_witness_status={"status": "PASSED", "step_count": len(trace)},
        reason="word_sort_trace_unique",
    )


def _solve_temporal_interval_trace(sample: DatasetSample, ir: SourceIRv2) -> D4SolverResult:
    """Solve compiler-emitted feasible windows on a fixed start-time grid."""

    if str(ir.query.get("kind") or "") != "longest_feasible_interval":
        return _unsupported_solver("temporal_query_kind_invalid")
    try:
        grid = int(ir.query.get("grid_minutes") or 30)
    except (TypeError, ValueError):
        return _unsupported_solver("temporal_grid_invalid")
    if grid <= 0 or not ir.facts:
        return _unsupported_solver("temporal_grid_or_windows_missing")
    best_duration = -1
    best_count = 0
    trace: list[dict[str, Any]] = []
    for index, fact in enumerate(ir.facts):
        if str(fact.get("kind") or "") != "feasible_window":
            return _unsupported_solver("temporal_window_kind_invalid")
        try:
            start = float(fact["start_min"])
            end = float(fact["end_min"])
        except (KeyError, TypeError, ValueError):
            return _unsupported_solver("temporal_window_bounds_invalid")
        if not end > start:
            return _unsupported_solver("temporal_window_order_invalid")
        first_start = int(math.ceil(start / grid) * grid)
        last_start = int(math.floor(end / grid) * grid)
        if first_start >= last_start:
            continue
        duration = int(last_start - first_start)
        if duration > best_duration:
            best_duration = duration
            best_count = 1
        elif duration == best_duration:
            best_count += 1
        trace.append(
            {
                "window_index": index,
                "start_grid": first_start,
                "end_grid": last_start,
                "duration": duration,
            }
        )
    answer_text = "0, 0" if best_duration < 0 else f"{best_duration:g}, {best_count}"
    canonical = canonicalize_answer(sample, answer_text)
    if not canonical.valid or not canonical.key:
        return _unsupported_solver("temporal_answer_outside_contract")
    return D4SolverResult(
        status="UNIQUE",
        canonical_answer=canonical.key,
        answer_text=answer_text,
        solver_trace=tuple(trace),
        reference_checker_status="PASSED_TEMPORAL_INTERVAL_CHECKER",
        concrete_witness_status={
            "status": "PASSED",
            "window_count": len(trace),
            "grid_minutes": grid,
        },
        reason="temporal_interval_trace_unique",
    )


def _compile_truth_graph(
    sample: DatasetSample,
    decision: D4RouteDecision,
) -> tuple[SourceIRv2 | None, str]:
    source = question_without_answer_contract(sample)
    graph = build_source_span_graph(sample)
    relations = []
    for match in re.finditer(
        r"The person at the ([^.]+?) says (?:that )?(?:the )?person at the ([^.]+?) (tells the truth|lies)\.",
        source,
        re.IGNORECASE,
    ):
        relations.append(
            {
                "constraint_id": f"C{len(relations)}",
                "kind": "truth_equivalence",
                "speaker": match.group(1).strip(),
                "target": match.group(2).strip(),
                "target_claim": "truth" if match.group(3).casefold().startswith("tells") else "lie",
                "source_span_ids": list(_span_ids_for_range(graph, match.start(), match.end())),
            }
        )
    directs = []
    for sentence_match in re.finditer(r".+?(?:[.!?](?=\s|$)|$)", source):
        match = re.fullmatch(
            r"The person at the ((?:(?! says ).)+?) (tells the truth|lies)\.",
            sentence_match.group(0).strip(),
            re.IGNORECASE,
        )
        if match is not None:
            directs.append(
                {
                    "constraint_id": f"D{len(directs)}",
                    "kind": "truth_constant",
                    "entity": match.group(1).strip(),
                    "value": match.group(2).casefold().startswith("tells"),
                    "source_span_ids": list(
                        _span_ids_for_range(graph, sentence_match.start(), sentence_match.end())
                    ),
                }
            )
    query_matches = list(
        re.finditer(
            r"Does the person at the (.+?) tell the truth\?",
            source,
            re.IGNORECASE,
        )
    )
    queries = [match.group(1).strip() for match in query_matches]
    if not relations or not queries:
        return None, "truth_graph_constraints_or_query_missing"
    entities = sorted(
        {
            *queries,
            *(str(row["speaker"]) for row in relations),
            *(str(row["target"]) for row in relations),
            *(str(row["entity"]) for row in directs),
        }
    )
    spans = tuple({"span_id": span.span_id, "text": span.text} for span in graph.spans)
    constraints = tuple([*relations, *directs])
    query_span_ids = tuple(
        dict.fromkeys(
            span_id
            for match in query_matches
            for span_id in _span_ids_for_range(graph, match.start(), match.end())
        )
    )
    mandatory_set = {
        *(span_id for row in constraints for span_id in row["source_span_ids"]),
        *query_span_ids,
    }
    mandatory = tuple(span.span_id for span in graph.spans if span.span_id in mandatory_set)
    uncovered = tuple(span.span_id for span in graph.spans if span.span_id not in mandatory_set)
    ir = SourceIRv2(
        capability_id=decision.capability_id,
        query_operator=decision.query_operator,
        entities=tuple({"entity_id": entity, "kind": "truth_agent"} for entity in entities),
        facts=(),
        events=(),
        constraints=constraints,
        query={"kind": "truth_values", "entities": queries, "source_span_ids": list(query_span_ids)},
        answer_contract=_answer_contract_for_sample(sample),
        source_span_map=spans,
        mandatory_spans=mandatory,
        uncovered_spans=uncovered,
        canonical_ir_hash="",
    )
    return replace(ir, canonical_ir_hash=canonical_ir_hash(ir)), "ok"


def _solve_truth_graph(sample: DatasetSample, ir: SourceIRv2) -> D4SolverResult:
    status, values, trace, reason = _truth_assignment(ir)
    if status != "UNIQUE":
        return _status_solver(status, reason)
    answer_text = _truth_answer_text(ir, values)
    canonical = canonicalize_answer(sample, answer_text)
    if not canonical.valid or not canonical.key:
        return _unsupported_solver("truth_graph_answer_outside_contract")
    return D4SolverResult(
        status="UNIQUE",
        canonical_answer=canonical.key,
        answer_text=answer_text,
        solver_trace=trace,
        reference_checker_status="PASSED_PARITY_GRAPH_REFERENCE_CHECKER",
        concrete_witness_status={
            "status": "PASSED",
            "anchored_query_count": len(list(ir.query.get("entities") or [])),
        },
        reason=reason,
    )


def _solve_numeric_expression(sample: DatasetSample, ir: SourceIRv2) -> D4SolverResult:
    if ir.facts or ir.events or len(ir.constraints) != 1:
        return _unsupported_solver("numeric_ir_contains_unexpected_records")
    expressions = [
        str(item.get("expression") or "")
        for item in ir.constraints
        if item.get("kind") == "numeric_expression"
    ]
    if len(expressions) != 1:
        return _unsupported_solver("numeric_expression_constraint_not_unique")
    constraint = ir.constraints[0]
    span_text = {row["span_id"]: row["text"] for row in ir.source_span_map}
    bound_text = "\n".join(
        span_text.get(str(span_id), "")
        for span_id in constraint.get("source_span_ids") or []
    )
    from research_experiments.families.contrastive_active_testing.kernel_d3 import (
        _numeric_answer_matches,
        _safe_numeric_value,
        _validate_numeric_source_fidelity,
    )

    fidelity = _validate_numeric_source_fidelity(expressions[0], bound_text)
    if fidelity != "ok":
        return _unsupported_solver(fidelity)

    value, reason = _safe_numeric_value(expressions[0])
    if value is None:
        return _unsupported_solver(reason)
    matches = _numeric_answer_matches(sample, value)
    if len(matches) > 1:
        return _status_solver("MULTIPLE", "numeric_answer_multiple_contract_matches")
    if not matches:
        return _status_solver("UNSAT", "numeric_answer_no_contract_match")
    canonical = canonicalize_answer(sample, matches[0])
    if not canonical.valid or not canonical.key:
        return _unsupported_solver("numeric_answer_canonicalization_failed")
    return D4SolverResult(
        status="UNIQUE",
        canonical_answer=canonical.key,
        answer_text=next(
            str(option.get("text") or "")
            for option in answer_schema_for_sample(sample)
            if str(option.get("label") or "") == matches[0]
        ),
        solver_trace=({"expression_hash": _sha256(expressions[0]), "value": value},),
        reference_checker_status="PASSED_TYPED_NUMERIC_REFERENCE_CHECKER",
        concrete_witness_status={"status": "PASSED", "value": value},
        reason="numeric_expression_unique",
    )


def _answer_contract_for_sample(sample: DatasetSample) -> dict[str, Any]:
    options = answer_schema_for_sample(sample)
    raw = sample.metadata.get("answer_contract")
    kind = str(raw.get("kind") or "single_choice") if isinstance(raw, dict) else "free_text"
    return {
        "kind": kind,
        "options": options,
        "selection_mode": str(raw.get("selection_mode") or "single") if isinstance(raw, dict) else "single",
    }


def _referenced_span_ids(payload: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in {"source_span_ids", "provenance_span_ids"}:
                found.update(_string_tuple(value))
            elif key not in {"source_span_map", "mandatory_spans", "uncovered_spans"}:
                found.update(_referenced_span_ids(value))
    elif isinstance(payload, list):
        for value in payload:
            found.update(_referenced_span_ids(value))
    return found


def _decisive_records_have_provenance(payload: dict[str, Any]) -> bool:
    for name in ("facts", "events", "constraints"):
        for row in payload.get(name) or []:
            span_ids = row.get("source_span_ids") if isinstance(row, dict) else None
            if not isinstance(span_ids, list) or not span_ids or any(not isinstance(item, str) for item in span_ids):
                return False
    query = payload.get("query")
    query_spans = query.get("source_span_ids") if isinstance(query, dict) else None
    return bool(
        isinstance(query_spans, list)
        and query_spans
        and all(isinstance(item, str) for item in query_spans)
    )


def _find_forbidden_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).strip().casefold().replace("-", "_")
            if normalized in _FORBIDDEN_IR_KEYS:
                found.add(normalized)
            found.update(_find_forbidden_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_find_forbidden_keys(item))
    return found


def _looks_like_shuffled_swap(lowered: str) -> bool:
    return "at the start" in lowered and "at the end" in lowered and bool(
        re.search(r"\b(?:swap|switch|trade)(?:s|ped)?\b", lowered)
    )


def _looks_like_truth_graph(lowered: str) -> bool:
    return "assume each person either always tells the truth or always lies" in lowered and "does the person at" in lowered


def _looks_like_word_sort_trace(lowered: str) -> bool:
    return "expert in word sorting" in lowered and "first step that was a mistake" in lowered


def _looks_like_temporal_schedule(lowered: str) -> bool:
    return "schedule for" in lowered and "longest meeting" in lowered and "booked at the following times" in lowered


def _looks_like_event_state(sample: DatasetSample, lowered: str) -> bool:
    task = str(sample.metadata.get("task") or "").casefold()
    return (
        ("collection" in lowered and "went through a few changes" in lowered)
        or ("table was converted to markdown" in lowered and "mistakenly replaced" in lowered)
        or (
            sample.dataset in {"musr", "musr_x"}
            and task == "object_placements"
            and "narrative:" in lowered
            and "question:" in lowered
            and "which location is the most likely place" in lowered
            and "would look to find" in lowered
        )
        or (
            sample.dataset in {"musr", "musr_x"}
            and task == "team_allocation"
            and "narrative:" in lowered
            and "question:" in lowered
            and "how would you uniquely allocate each person" in lowered
            and "both tasks are accomplished efficiently" in lowered
        )
    )


def _looks_like_calculator(sample: DatasetSample, lowered: str) -> bool:
    task = str(sample.metadata.get("task") or "").casefold()
    if task in {"multistep_arithmetic", "boolean_expressions", "time_arithmetic", "object_counting"}:
        return True
    if sample.dataset in {"gpqa_diamond", "supergpqa", "supergpqa_science"}:
        return bool(re.search(r"(?:\d\s*[+*/^=-]\s*\d|\b(?:calculate|compute|equation|ratio|probability)\b)", lowered))
    return False


def _name_list(value: str) -> list[str]:
    normalized = re.sub(r",?\s+and\s+", ", ", value.strip())
    return [item.strip() for item in normalized.split(",") if re.fullmatch(r"[A-Z][A-Za-z'-]*", item.strip())]


def _strip_indefinite_article(value: str) -> str:
    return re.sub(r"^(?:a|an)\s+", "", value.strip(), flags=re.IGNORECASE)


def _span_ids_for_range(graph: Any, start: int, end: int) -> tuple[str, ...]:
    return tuple(
        str(span.span_id)
        for span in graph.spans
        if int(span.start) < int(end) and int(span.end) > int(start)
    )


def _canonical_sort_key(value: dict[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        return ()
    return tuple(value)


def _option_contract_invariant(
    ir: SourceIRv2,
    solver: D4SolverResult,
    *,
    sample: DatasetSample | None,
    decision: D4RouteDecision | None,
) -> str:
    options = ir.answer_contract.get("options")
    if not isinstance(options, list) or not options:
        return "NOT_APPLICABLE"
    if any(not isinstance(item, dict) for item in options):
        return "FAILED"
    before = {str(item.get("label") or ""): str(item.get("text") or "") for item in options}
    if not before or len(before) != len(options):
        return "FAILED"
    if solver.status != "UNIQUE" or not solver.canonical_answer or sample is None or decision is None:
        return "NOT_RUN"
    transformed_contract = {**ir.answer_contract, "options": list(reversed(options))}
    transformed_sample = _sample_with_answer_contract(sample, transformed_contract)
    transformed_ir = replace(ir, answer_contract=transformed_contract)
    transformed = solve_source_ir(transformed_sample, decision, transformed_ir)
    return "PASSED" if (
        transformed.status == "UNIQUE"
        and transformed.canonical_answer == solver.canonical_answer
        and transformed.answer_text == solver.answer_text
    ) else "FAILED"


def _answer_label_permutation_check(
    ir: SourceIRv2,
    solver: D4SolverResult,
    *,
    sample: DatasetSample | None,
    decision: D4RouteDecision | None,
) -> str:
    options = ir.answer_contract.get("options")
    if not isinstance(options, list) or not options:
        return "NOT_APPLICABLE"
    if (
        solver.status != "UNIQUE"
        or not solver.canonical_answer
        or solver.answer_text is None
        or sample is None
        or decision is None
    ):
        return "NOT_RUN"
    parsed = [
        (str(item.get("label") or ""), str(item.get("text") or ""))
        for item in options
        if isinstance(item, dict)
    ]
    if len(parsed) != len(options) or len({label for label, _ in parsed}) != len(parsed):
        return "FAILED"
    if len({text for _, text in parsed}) != len(parsed):
        return "FAILED"
    original = [label for label, text in parsed if text == solver.answer_text]
    if len(original) != 1 or original[0] != solver.canonical_answer:
        return "NOT_APPLICABLE"
    labels = [label for label, _ in parsed]
    permuted_labels = labels[1:] + labels[:1]
    relabeled_options = [
        {**item, "label": new_label}
        for item, new_label in zip(options, permuted_labels, strict=True)
    ]
    expected_label = next(
        new_label
        for (_, text), new_label in zip(parsed, permuted_labels, strict=True)
        if text == solver.answer_text
    )
    transformed_contract = {**ir.answer_contract, "options": relabeled_options}
    transformed_sample = _sample_with_answer_contract(sample, transformed_contract)
    transformed_ir = replace(ir, answer_contract=transformed_contract)
    transformed = solve_source_ir(transformed_sample, decision, transformed_ir)
    return "PASSED" if (
        transformed.status == "UNIQUE"
        and transformed.canonical_answer == expected_label
        and transformed.answer_text == solver.answer_text
    ) else "FAILED"


def _sample_with_answer_contract(sample: DatasetSample, contract: dict[str, Any]) -> DatasetSample:
    return replace(
        sample,
        metadata={**sample.metadata, "answer_contract": contract},
    )


def _shuffled_state(ir: SourceIRv2, events: tuple[dict[str, Any], ...] | None = None) -> dict[str, str] | None:
    state = {str(row.get("entity")): str(row.get("value")) for row in ir.facts}
    for event in events if events is not None else ir.events:
        left, right = str(event.get("left") or ""), str(event.get("right") or "")
        if left not in state or right not in state:
            return None
        state[left], state[right] = state[right], state[left]
    return state


def _shuffled_entity_renaming_check(ir: SourceIRv2, solver: D4SolverResult) -> str:
    names = [str(row.get("entity_id") or "") for row in ir.entities]
    if not names or any(not name for name in names):
        return "FAILED"
    rename = {name: f"entity_{index}" for index, name in enumerate(names)}
    facts = tuple(dict(row, entity=rename.get(str(row.get("entity") or ""), "")) for row in ir.facts)
    events = tuple(
        dict(
            row,
            left=rename.get(str(row.get("left") or ""), ""),
            right=rename.get(str(row.get("right") or ""), ""),
        )
        for row in ir.events
    )
    query = dict(ir.query, entity=rename.get(str(ir.query.get("entity") or ""), ""))
    renamed = replace(ir, facts=facts, events=events, query=query)
    state = _shuffled_state(renamed)
    return "PASSED" if state is not None and state.get(str(query["entity"])) == solver.answer_text else "FAILED"


def _shuffled_reversible_event_check(ir: SourceIRv2, solver: D4SolverResult) -> str:
    if not ir.events:
        return "NOT_APPLICABLE"
    event = ir.events[-1]
    state = _shuffled_state(ir, (*ir.events, event, event))
    query = str(ir.query.get("entity") or "")
    return "PASSED" if state is not None and state.get(query) == solver.answer_text else "FAILED"


def _shuffled_commutation_check(ir: SourceIRv2, solver: D4SolverResult) -> str:
    for index in range(len(ir.events) - 1):
        left = {str(ir.events[index].get("left")), str(ir.events[index].get("right"))}
        right = {str(ir.events[index + 1].get("left")), str(ir.events[index + 1].get("right"))}
        if left.isdisjoint(right):
            events = list(ir.events)
            events[index], events[index + 1] = events[index + 1], events[index]
            state = _shuffled_state(ir, tuple(events))
            query = str(ir.query.get("entity") or "")
            return "PASSED" if state is not None and state.get(query) == solver.answer_text else "FAILED"
    return "NOT_APPLICABLE"


def _truth_assignment(ir: SourceIRv2) -> tuple[SolverStatus, dict[str, int], tuple[dict[str, Any], ...], str]:
    adjacency: dict[str, list[tuple[str, int]]] = {}
    fixed: dict[str, int] = {}
    for row in ir.constraints:
        if row.get("kind") == "truth_equivalence":
            left, right = str(row.get("speaker") or ""), str(row.get("target") or "")
            parity = 0 if row.get("target_claim") == "truth" else 1
            adjacency.setdefault(left, []).append((right, parity))
            adjacency.setdefault(right, []).append((left, parity))
        elif row.get("kind") == "truth_constant":
            entity, value = str(row.get("entity") or ""), int(bool(row.get("value")))
            if entity in fixed and fixed[entity] != value:
                return "UNSAT", {}, (), "truth_graph_conflicting_constants"
            fixed[entity] = value
        else:
            return "UNSUPPORTED", {}, (), "truth_graph_constraint_kind_invalid"
    values = dict(fixed)
    queue = list(fixed)
    trace = []
    while queue:
        current = queue.pop(0)
        for neighbor, parity in adjacency.get(current, []):
            proposed = values[current] ^ parity
            if neighbor in values and values[neighbor] != proposed:
                return "UNSAT", {}, (), "truth_graph_parity_conflict"
            if neighbor not in values:
                values[neighbor] = proposed
                queue.append(neighbor)
                trace.append({"from": current, "to": neighbor, "parity": parity, "value": bool(proposed)})
    queries = [str(item) for item in ir.query.get("entities") or []]
    if any(item not in values for item in queries):
        return "MULTIPLE", values, tuple(trace), "truth_graph_query_component_unanchored"
    return "UNIQUE", values, tuple(trace), "truth_graph_unique"


def _truth_answer_text(ir: SourceIRv2, values: dict[str, int]) -> str:
    return ", ".join("yes" if values[str(item)] else "no" for item in ir.query.get("entities") or [])


def _truth_entity_renaming_check(ir: SourceIRv2, solver: D4SolverResult) -> str:
    if solver.status != "UNIQUE" or solver.answer_text is None:
        return "NOT_RUN"
    names = [str(row.get("entity_id") or "") for row in ir.entities]
    if not names or any(not name for name in names):
        return "FAILED"
    rename = {name: f"truth_agent_{index}" for index, name in enumerate(names)}
    constraints = []
    for row in ir.constraints:
        transformed = dict(row)
        for field in ("speaker", "target", "entity"):
            if field in transformed:
                transformed[field] = rename.get(str(transformed[field]), "")
        constraints.append(transformed)
    renamed = replace(
        ir,
        entities=tuple({**row, "entity_id": rename[str(row["entity_id"])]} for row in ir.entities),
        constraints=tuple(constraints),
        query=dict(ir.query, entities=[rename[str(item)] for item in ir.query.get("entities") or []]),
    )
    status, values, _trace, _reason = _truth_assignment(renamed)
    return "PASSED" if status == "UNIQUE" and _truth_answer_text(renamed, values) == solver.answer_text else "FAILED"


def _truth_constraint_order_check(ir: SourceIRv2, solver: D4SolverResult) -> str:
    if solver.status != "UNIQUE" or solver.answer_text is None:
        return "NOT_RUN"
    reordered = replace(ir, constraints=tuple(reversed(ir.constraints)))
    status, values, _trace, _reason = _truth_assignment(reordered)
    return "PASSED" if status == "UNIQUE" and _truth_answer_text(reordered, values) == solver.answer_text else "FAILED"


def _numeric_algebraic_equivalence_check(ir: SourceIRv2, solver: D4SolverResult) -> str:
    if solver.status != "UNIQUE":
        return "NOT_RUN"
    expressions = [
        str(item.get("expression") or "")
        for item in ir.constraints
        if item.get("kind") == "numeric_expression"
    ]
    if len(expressions) != 1:
        return "FAILED"
    from research_experiments.families.contrastive_active_testing.kernel_d3 import _safe_numeric_value

    original, _ = _safe_numeric_value(expressions[0])
    transformed, _ = _safe_numeric_value(f"(({expressions[0]})) + 0")
    if original is None or transformed is None:
        return "FAILED"
    return "PASSED" if abs(original - transformed) <= max(1e-12, abs(original) * 1e-12) else "FAILED"


def _unsupported_solver(reason: str) -> D4SolverResult:
    return _status_solver("UNSUPPORTED", reason)


def _status_solver(status: SolverStatus, reason: str) -> D4SolverResult:
    return D4SolverResult(
        status=status,
        canonical_answer=None,
        answer_text=None,
        solver_trace=(),
        reference_checker_status="NOT_RUN",
        concrete_witness_status={"status": "NOT_AVAILABLE"},
        reason=reason,
    )


def _clopper_lower(successes: int, total: int, *, alpha: float = 0.05) -> float:
    if total <= 0 or successes <= 0:
        return 0.0
    return float(beta.ppf(alpha, successes, total - successes + 1))


def _clopper_upper(failures: int, total: int, *, alpha: float = 0.05) -> float:
    if total <= 0:
        return 1.0
    if failures >= total:
        return 1.0
    return float(beta.ppf(1.0 - alpha, failures + 1, total - failures))


def _minimum_zero_failure_trials_for_lower(target: float, *, alpha: float) -> int:
    if not 0.0 < target < 1.0 or not 0.0 < alpha < 1.0:
        raise ValueError("Risk-gate target and alpha must be strictly between zero and one.")
    total = 1
    while _clopper_lower(total, total, alpha=alpha) < target:
        total += 1
    return total


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _sha256(value: Any) -> str:
    raw = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
