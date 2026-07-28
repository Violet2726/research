"""CATCH-Kernel D3 source-blind routing and conditional certificates.

This module intentionally keeps the D3 surface separate from the frozen D2
pair/candidate protocol.  A source IR is created without looking at Stage-A
answers; candidate evaluation happens only after a deterministic solution has
been obtained.  The certificate therefore describes a conditional claim
(``source -> IR -> answer``), never a claim that the IR is the benchmark gold.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

from research_experiments.core.data.datasets import DatasetSample, question_without_answer_contract
from research_experiments.core.data.evaluation import canonicalize_answer
from research_experiments.families.contrastive_active_testing.certificates_v2 import (
    build_source_span_graph,
)
from research_experiments.families.contrastive_active_testing.kernel_adapters import (
    _compile_circular_path,
    _compile_custom_sort,
    _compile_ordered_grid_path,
    _normalize,
    _strict_dyck_earliest_error,
)

D3_IR_SCHEMA = "catch_d3_source_ir_v1"
D3_IR_VERSION = "1"
D3_SOLVER_VERSION = "catch_d3_safe_numeric_v1"
# Bump the registry identity after the D3 development audit: multistep
# arithmetic is explicitly withdrawn from semantic routing and soft routes
# now keep the Stage-A anchor by default.  This prevents confirmation
# artifacts from being confused with the earlier development policy.
D3_CAPABILITY_REGISTRY_VERSION = "catch_d3_capability_registry_v2"

Route = Literal["EXACT_EXECUTABLE", "SEMANTIC_COMPILABLE", "SOFT_UNSUPPORTED"]
CertificateStatus = Literal["UNIQUE", "MULTIPLE", "UNSAT", "UNSUPPORTED"]
CandidateStatus = Literal["VALID", "INVALID", "UNSUPPORTED"]
DecisionAction = Literal["solver_direct", "candidate_completion", "keep_anchor"]


@dataclass(frozen=True)
class RouteDecision:
    route: Route
    operation_kind: str
    reason: str


@dataclass(frozen=True)
class SourceIR:
    dataset: str
    task_family: str
    ir_version: str
    source_hash: str
    answer_schema: tuple[dict[str, Any], ...]
    query: dict[str, Any]
    constraints: tuple[dict[str, Any], ...]
    covered_span_ids: tuple[str, ...]
    uncovered_span_ids: tuple[str, ...]
    route: Route


@dataclass(frozen=True)
class SolverCertificate:
    status: CertificateStatus
    canonical_answer: str | None
    route: Route
    operation_kind: str
    source_hash: str
    ir_hash: str
    solver_version: str
    cross_check_status: str
    metamorphic_test_status: str
    reason: str


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_hash: str
    canonical_answer: str | None
    status: CandidateStatus
    trace_hash: str
    reason: str


@dataclass(frozen=True)
class KernelDecision:
    """Auditable D3 decision record, separate from the legacy D1/D2 decoder."""

    route: Route
    anchor: str
    final_answer: str
    action: DecisionAction
    override_reason: str
    certificate_hash: str | None
    risk_tier: str


def route_for_sample(sample: DatasetSample) -> RouteDecision:
    """Select a route from frozen dataset/task capability, never from candidates."""

    task = str(sample.metadata.get("task") or "").strip().casefold()
    if sample.dataset == "bbeh" and task == "dyck_languages":
        return RouteDecision("EXACT_EXECUTABLE", "stack_trace", "frozen_dyck_parser")
    if sample.dataset == "bbeh" and task == "word_sorting":
        source = question_without_answer_contract(sample)
        if "identify the first step" not in source.casefold() and "mistake in thought" not in source.casefold():
            return RouteDecision("EXACT_EXECUTABLE", "custom_sort_order", "frozen_custom_sort_parser")
    if sample.dataset == "bbeh" and task == "spatial_reasoning":
        source = question_without_answer_contract(sample)
        compiled, error = _compile_circular_path(source)
        if compiled is not None:
            return RouteDecision("EXACT_EXECUTABLE", "grid_path", "frozen_circular_path_parser")
        compiled, error = _compile_ordered_grid_path(source)
        if compiled is not None:
            return RouteDecision("EXACT_EXECUTABLE", "grid_path", "frozen_ordered_grid_parser")
        return RouteDecision("SOFT_UNSUPPORTED", "grid_path", error or "grid_parser_unsupported")
    # BBEH multistep arithmetic uses task-defined operators and recursive
    # bindings.  The current safe numeric IR cannot represent that language;
    # keep it outside the executable jurisdiction until a typed AST adapter
    # and its metamorphic suite exist.
    if sample.dataset == "bbeh" and task == "multistep_arithmetic":
        return RouteDecision("SOFT_UNSUPPORTED", "none", "semantic_ast_adapter_unregistered")
    # A small, explicitly scoped semantic route is enabled only where a
    # numeric answer can be checked without importing domain knowledge.  The
    # execution config may keep this route as a shadow audit, but it is never
    # an override path before the independent gates pass.
    if sample.dataset == "gpqa_diamond":
        domain = str(sample.metadata.get("high_level_domain") or "").casefold()
        if domain == "physics":
            return RouteDecision("SEMANTIC_COMPILABLE", "safe_numeric_expression", "gpqa_physics_numeric_scope")
    return RouteDecision("SOFT_UNSUPPORTED", "none", "no_frozen_executable_capability")


def capability_registry() -> dict[str, Any]:
    """Return the frozen, candidate-independent D3 jurisdiction registry."""

    return {
        "version": D3_CAPABILITY_REGISTRY_VERSION,
        "exact": {
            "bbeh.dyck_languages": "stack_trace",
            "bbeh.word_sorting": "custom_sort_order_except_error_trace",
            "bbeh.spatial_reasoning": "grid_path_when_local_parser_succeeds",
        },
        "semantic": {
            "gpqa_diamond.Physics": "safe_numeric_expression",
        },
        "soft_default": True,
    }


def source_ir_from_exact_sample(sample: DatasetSample, decision: RouteDecision) -> SourceIR | None:
    graph = build_source_span_graph(sample)
    source = question_without_answer_contract(sample)
    options = answer_schema_for_sample(sample)
    if decision.operation_kind == "stack_trace":
        query_kind = "earliest_invalid_stack_transition"
    elif decision.operation_kind == "custom_sort_order":
        query_kind = "source_defined_word_order"
    elif decision.operation_kind == "grid_path":
        query_kind = "final_path_entity"
    else:
        return None
    span_ids = tuple(span.span_id for span in graph.spans)
    return SourceIR(
        dataset=sample.dataset,
        task_family=str(sample.metadata.get("task") or ""),
        ir_version=D3_IR_VERSION,
        source_hash=_sha256(source),
        answer_schema=tuple(options),
        query={"kind": query_kind, "source_span_ids": list(span_ids), "constraint_ids": []},
        constraints=(),
        covered_span_ids=span_ids,
        uncovered_span_ids=(),
        route=decision.route,
    )


def parse_source_ir(
    payload: Any,
    *,
    sample: DatasetSample,
    decision: RouteDecision,
) -> tuple[SourceIR | None, str]:
    """Validate the closed semantic IR returned by a candidate-blind compiler."""

    if not isinstance(payload, dict):
        return None, "source_ir_not_object"
    allowed = {
        "schema",
        "ir_version",
        "query",
        "constraints",
        "covered_span_ids",
        "uncovered_span_ids",
    }
    if set(payload) != allowed:
        return None, "source_ir_schema_keys_invalid"
    if payload.get("schema") != D3_IR_SCHEMA or str(payload.get("ir_version")) != D3_IR_VERSION:
        return None, "source_ir_version_invalid"
    if decision.operation_kind != "safe_numeric_expression":
        return None, "source_ir_operation_unsupported"
    source = question_without_answer_contract(sample)
    graph = build_source_span_graph(sample)
    known = {span.span_id for span in graph.spans}
    covered = _string_tuple(payload.get("covered_span_ids"))
    uncovered = _string_tuple(payload.get("uncovered_span_ids"))
    if not covered or set(covered) - known or set(uncovered) - known or set(covered) & set(uncovered):
        return None, "source_ir_span_partition_invalid"
    if set(covered) | set(uncovered) != known:
        return None, "source_ir_span_coverage_incomplete"
    constraints = payload.get("constraints")
    if not isinstance(constraints, list) or len(constraints) != 1 or any(not isinstance(item, dict) for item in constraints):
        return None, "source_ir_constraints_invalid"
    constraint = dict(constraints[0])
    required_constraint_keys = {"constraint_id", "kind", "expression", "source_span_ids"}
    if set(constraint) != required_constraint_keys:
        return None, "source_ir_constraint_schema_invalid"
    if constraint.get("constraint_id") != "C0" or constraint.get("kind") != "numeric_expression":
        return None, "source_ir_constraint_type_invalid"
    expression = constraint.get("expression")
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 512:
        return None, "source_ir_expression_invalid"
    constraint_spans = _string_tuple(constraint.get("source_span_ids"))
    if not constraint_spans or set(constraint_spans) - set(covered):
        return None, "source_ir_constraint_binding_invalid"
    query = payload.get("query")
    required_query_keys = {"kind", "source_span_ids", "constraint_ids"}
    if not isinstance(query, dict) or set(query) != required_query_keys:
        return None, "source_ir_query_invalid"
    query_spans = _string_tuple(query.get("source_span_ids"))
    query_constraints = _string_tuple(query.get("constraint_ids"))
    if (
        query.get("kind") != "evaluate_numeric_expression"
        or not query_spans
        or set(query_spans) - set(covered)
        or query_constraints != ("C0",)
    ):
        return None, "source_ir_query_binding_invalid"
    bound_text = "\n".join(span.text for span in graph.spans if span.span_id in set(constraint_spans))
    fidelity_reason = _validate_numeric_source_fidelity(expression.strip(), bound_text)
    if fidelity_reason != "ok":
        return None, fidelity_reason
    normalized_constraint = {
        "constraint_id": "C0",
        "kind": "numeric_expression",
        "expression": expression.strip(),
        "source_span_ids": list(sorted(constraint_spans)),
    }
    normalized_query = {
        "kind": "evaluate_numeric_expression",
        "source_span_ids": list(sorted(query_spans)),
        "constraint_ids": ["C0"],
    }
    return (
        SourceIR(
            dataset=sample.dataset,
            task_family=str(sample.metadata.get("task") or ""),
            ir_version=D3_IR_VERSION,
            source_hash=_sha256(source),
            answer_schema=tuple(answer_schema_for_sample(sample)),
            query=normalized_query,
            constraints=(normalized_constraint,),
            covered_span_ids=tuple(sorted(covered)),
            uncovered_span_ids=tuple(sorted(uncovered)),
            route=decision.route,
        ),
        "ok",
    )


def solve_exact(sample: DatasetSample, decision: RouteDecision, ir: SourceIR | None = None) -> SolverCertificate:
    """Solve only the currently implemented exact jurisdictions."""

    source = question_without_answer_contract(sample)
    ir = ir or source_ir_from_exact_sample(sample, decision)
    if ir is None:
        return _unsupported(decision, source, "exact_ir_unavailable")
    answer: str | None = None
    reason = ""
    if decision.operation_kind == "stack_trace":
        answer, reason = _strict_dyck_earliest_error(source)
    elif decision.operation_kind == "custom_sort_order":
        payload, error = _compile_custom_sort(source, strict=True)
        if payload is None:
            return _unsupported(decision, source, error or "custom_sort_compile_failed", ir=ir)
        rank = {character: index for index, character in enumerate(payload["alphabet"])}
        words = list(payload["words"])
        words.sort(key=lambda value: tuple(rank.get(char.casefold(), len(rank) + ord(char)) for char in value))
        answer, reason = ",".join(words), "custom_sort_unique"
    elif decision.operation_kind == "grid_path":
        payload, error = _compile_circular_path(source)
        if payload is None:
            payload, error = _compile_ordered_grid_path(source)
        if payload is None:
            return _unsupported(decision, source, error or "grid_path_compile_failed", ir=ir)
        if payload.get("path_kind") == "ordered":
            answer, reason = str(payload.get("expected_entity") or ""), "ordered_grid_unique"
        else:
            entities = list(payload["entities_clockwise"])
            position = next(
                index for index, item in enumerate(entities) if _normalize(item) == _normalize(payload["start_entity"])
            )
            for move in payload["signed_moves"]:
                position = (position + int(move)) % len(entities)
            answer, reason = entities[position], "circular_grid_unique"
    else:
        return _unsupported(decision, source, "exact_operation_unregistered", ir=ir)
    canonical = canonicalize_answer(sample, answer or "")
    if not canonical.valid or not canonical.key:
        return _unsupported(decision, source, "solver_answer_outside_answer_contract", ir=ir)
    return SolverCertificate(
        status="UNIQUE",
        canonical_answer=canonical.key,
        route=decision.route,
        operation_kind=decision.operation_kind,
        source_hash=ir.source_hash,
        ir_hash=_sha256(asdict(ir)),
        solver_version=D3_SOLVER_VERSION,
        cross_check_status="PASSED_LOCAL_REFERENCE_CHECKER",
        metamorphic_test_status="FROZEN_EXACT_UNIT_SUITE",
        reason=reason,
    )


def solve_numeric_ir(sample: DatasetSample, decision: RouteDecision, ir: SourceIR) -> SolverCertificate:
    expression = _numeric_expression(ir)
    if decision.operation_kind != "safe_numeric_expression" or not expression:
        return _unsupported(decision, question_without_answer_contract(sample), "numeric_ir_unavailable", ir=ir)
    value, reason = _safe_numeric_value(expression)
    if value is None:
        return _unsupported(decision, question_without_answer_contract(sample), reason, ir=ir)
    matches = _numeric_answer_matches(sample, value)
    if len(matches) > 1:
        return _unsupported(
            decision,
            question_without_answer_contract(sample),
            "numeric_answer_contract_multiple_matches",
            ir=ir,
            status="MULTIPLE",
        )
    if not matches:
        return _unsupported(
            decision,
            question_without_answer_contract(sample),
            "numeric_value_not_in_answer_contract",
            ir=ir,
            status="UNSAT",
        )
    answer = matches[0]
    canonical = canonicalize_answer(sample, answer)
    if not canonical.valid or not canonical.key:
        return _unsupported(decision, question_without_answer_contract(sample), "numeric_answer_canonicalization_failed", ir=ir)
    return SolverCertificate(
        status="UNIQUE",
        canonical_answer=canonical.key,
        route=decision.route,
        operation_kind=decision.operation_kind,
        source_hash=ir.source_hash,
        ir_hash=_sha256(asdict(ir)),
        solver_version=D3_SOLVER_VERSION,
        cross_check_status="PASSED_SAFE_NUMERIC_REFERENCE_CHECKER",
        # Compiler metamorphics require transformed source calls and are frozen
        # as a development audit, not silently inferred from solver execution.
        metamorphic_test_status="NOT_RUN_RUNTIME_REQUIRES_FROZEN_AUDIT",
        reason="numeric_expression_unique",
    )


def evaluate_candidate(sample: DatasetSample, answer: str, certificate: SolverCertificate) -> CandidateEvaluation:
    canonical = canonicalize_answer(sample, answer)
    candidate_hash = _sha256(str(answer))
    if not canonical.valid or not canonical.key:
        return CandidateEvaluation(candidate_hash, None, "UNSUPPORTED", _sha256({"answer": answer}), "candidate_not_canonical")
    if certificate.status != "UNIQUE" or not certificate.canonical_answer:
        return CandidateEvaluation(candidate_hash, canonical.key, "UNSUPPORTED", _sha256({"answer": answer}), "certificate_not_unique")
    status: CandidateStatus = "VALID" if canonical.key == certificate.canonical_answer else "INVALID"
    trace = {"candidate_hash": candidate_hash, "candidate_key": canonical.key, "solver_key": certificate.canonical_answer}
    return CandidateEvaluation(candidate_hash, canonical.key, status, _sha256(trace), "canonical_match" if status == "VALID" else "canonical_mismatch")


def canonical_ir(ir: SourceIR) -> str:
    payload = asdict(ir)
    payload["answer_schema"] = sorted(payload["answer_schema"], key=lambda item: str(item.get("label") or ""))
    payload["covered_span_ids"] = sorted(payload["covered_span_ids"])
    payload["uncovered_span_ids"] = sorted(payload["uncovered_span_ids"])
    payload["constraints"] = sorted(payload["constraints"], key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False))
    for constraint in payload["constraints"]:
        if constraint.get("kind") == "numeric_expression" and isinstance(constraint.get("expression"), str):
            constraint["expression"] = canonical_numeric_expression(constraint["expression"])
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_numeric_expression(expression: str) -> str:
    """Normalize harmless numeric-expression formatting for IR agreement.

    This deliberately does not erase source-span or constraint provenance;
    those fields remain part of ``canonical_ir``.  It only prevents whitespace,
    redundant parentheses, and the common ``^`` power spelling from creating a
    false disagreement.
    """

    normalized = str(expression).strip().replace("^", "**")
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError:
        return re.sub(r"\s+", "", normalized)
    return ast.dump(tree, annotate_fields=True, include_attributes=False)


def source_ir_to_dict(ir: SourceIR) -> dict[str, Any]:
    return asdict(ir)


def solver_certificate_to_dict(certificate: SolverCertificate) -> dict[str, Any]:
    return asdict(certificate)


def candidate_evaluation_to_dict(evaluation: CandidateEvaluation) -> dict[str, Any]:
    return asdict(evaluation)


def answer_schema_for_sample(sample: DatasetSample) -> list[dict[str, Any]]:
    contract = sample.metadata.get("answer_contract")
    options = contract.get("options") if isinstance(contract, dict) else sample.metadata.get("options")
    if not isinstance(options, list):
        return []
    structured = [
        {"label": str(item.get("label") or ""), "text": str(item.get("text") or "")}
        for item in options
        if isinstance(item, dict)
    ]
    if structured:
        return structured
    return [
        {"label": chr(ord("A") + index), "text": str(item)}
        for index, item in enumerate(options)
        if str(item).strip()
    ]


def _safe_numeric_value(expression: str) -> tuple[float | None, str]:
    """Evaluate a closed numeric expression without executing arbitrary code."""

    allowed_names = {"pi": math.pi, "e": math.e, "tau": math.tau}
    allowed_calls = {"sqrt", "sin", "cos", "tan", "log", "log10", "exp", "fabs"}
    allowed_binops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
    allowed_unaryops = (ast.UAdd, ast.USub)
    try:
        tree = ast.parse(str(expression).replace("^", "**"), mode="eval")
    except SyntaxError:
        return None, "numeric_expression_syntax_invalid"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Expression, ast.Load, ast.Constant)):
            if isinstance(node, ast.Constant) and (
                isinstance(node.value, bool) or not isinstance(node.value, (int, float))
            ):
                return None, "numeric_expression_constant_unsupported"
            continue
        if isinstance(node, allowed_binops + allowed_unaryops):
            continue
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            return None, "numeric_expression_symbol_unsupported"
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in allowed_calls or len(node.keywords) != 0:
                return None, "numeric_expression_call_unsupported"
            continue
        if isinstance(node, ast.BinOp):
            if not isinstance(node.op, allowed_binops):
                return None, "numeric_expression_operator_unsupported"
            continue
        if isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, allowed_unaryops):
                return None, "numeric_expression_operator_unsupported"
            continue
        if isinstance(node, (ast.Attribute, ast.Lambda, ast.List, ast.Dict, ast.Set, ast.comprehension)):
            return None, "numeric_expression_ast_unsupported"
        return None, "numeric_expression_ast_unsupported"
    namespace = {**allowed_names, **{name: getattr(math, name) for name in allowed_calls}}
    try:
        value = float(eval(compile(tree, "<catch-d3-numeric>", "eval"), {"__builtins__": {}}, namespace))
    except (ArithmeticError, TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None, "numeric_expression_evaluation_failed"
    if not math.isfinite(value):
        return None, "numeric_expression_non_finite"
    return value, "ok"


def _numeric_expression(ir: SourceIR) -> str | None:
    for constraint in ir.constraints:
        if constraint.get("kind") == "numeric_expression":
            value = constraint.get("expression")
            return str(value).strip() if isinstance(value, str) and value.strip() else None
    return None


def _validate_numeric_source_fidelity(expression: str, bound_text: str) -> str:
    """Reject constants/functions that are absent from the bound source spans.

    This is deliberately conservative.  It prevents the compiler from adding
    remembered conversion factors or domain facts that the source never states.
    """

    source_numbers = {_normalize_number_token(item) for item in _numeric_tokens(bound_text)}
    expression_numbers = {_normalize_number_token(item) for item in _numeric_tokens(expression)}
    if not expression_numbers.issubset(source_numbers):
        return "source_ir_undeclared_numeric_constant"
    for name in re.findall(r"\b[A-Za-z_]\w*\b", expression):
        if not re.search(rf"\b{re.escape(name)}\b", bound_text, flags=re.IGNORECASE):
            return "source_ir_undeclared_symbol_or_function"
    return "ok"


def _numeric_tokens(value: str) -> tuple[str, ...]:
    return tuple(re.findall(r"(?<![A-Za-z_])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", value))


def _normalize_number_token(value: str) -> str:
    try:
        return format(float(value), ".15g")
    except ValueError:
        return value


def _match_numeric_answer(sample: DatasetSample, value: float) -> str | None:
    matches = _numeric_answer_matches(sample, value)
    return matches[0] if len(matches) == 1 else None


def _numeric_answer_matches(sample: DatasetSample, value: float) -> list[str]:
    options = answer_schema_for_sample(sample)
    if not options:
        if math.isclose(value, round(value), rel_tol=0.0, abs_tol=1e-9):
            return [str(int(round(value)))]
        return [format(value, ".15g")]
    matches: list[str] = []
    for item in options:
        text = str(item.get("text") or "")
        parsed = _extract_number(text)
        tolerance = max(1e-15, 1e-9 * max(abs(value), abs(parsed or 0.0)))
        if parsed is not None and math.isclose(value, parsed, rel_tol=1e-6, abs_tol=tolerance):
            matches.append(str(item.get("label") or ""))
    return matches


def _extract_number(value: str) -> float | None:
    cleaned = value.replace(",", "")
    power = re.search(r"10\s*\^\s*\{?\s*([+-]?\d+)\s*\}?", cleaned)
    if power:
        try:
            return 10.0 ** int(power.group(1))
        except (OverflowError, ValueError):
            return None
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", cleaned)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        return ()
    return tuple(str(item) for item in value)


def _unsupported(
    decision: RouteDecision,
    source: str,
    reason: str,
    *,
    ir: SourceIR | None = None,
    status: CertificateStatus = "UNSUPPORTED",
) -> SolverCertificate:
    return SolverCertificate(
        status=status,
        canonical_answer=None,
        route=decision.route,
        operation_kind=decision.operation_kind,
        source_hash=ir.source_hash if ir is not None else _sha256(source),
        ir_hash=_sha256(asdict(ir)) if ir is not None else "",
        solver_version=D3_SOLVER_VERSION,
        cross_check_status="not_run",
        metamorphic_test_status="not_run",
        reason=reason,
    )


def _sha256(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
