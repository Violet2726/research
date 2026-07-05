"""数学答案的受限符号证书校验。"""

from __future__ import annotations

import ast
import re
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

import sympy

from research_experiments.families.cred_v.certificates.types import CertificateValidation

_NUMBER_RE = re.compile(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?")
_SAFE_CALLS = {
    "abs": sympy.Abs,
    "Abs": sympy.Abs,
    "binomial": sympy.binomial,
    "factorial": sympy.factorial,
    "sqrt": sympy.sqrt,
}


class UnsafeExpression(ValueError):
    pass


@dataclass(frozen=True)
class MathCheckSpec:
    certificate_type: str
    problem_expression: str
    problem_constants: tuple[str, ...]
    problem_variables: tuple[str, ...] = ()
    unit: str = ""


def compile_math_check_spec(question: str) -> MathCheckSpec | None:
    """从题面编译可由本地 DSL 独立验证的窄型数学任务。"""
    text = str(question or "").strip()
    candidates = _plain_expression_candidates(text)
    if "interval" in text.lower():
        candidates.extend(re.findall(r"[\[(]\s*[^,\n]+\s*,\s*[^\])\n]+\s*[\])]", text))

    seen: set[str] = set()
    for raw_source in candidates:
        source = str(raw_source or "").strip()
        source = source[:-1].rstrip() if source.endswith(".") else source
        source = source.strip("$").strip()
        source = source.removeprefix(":").strip()
        source = source.strip("$").strip()
        canonical = _canonical_spec_source(source)
        if not source or canonical in seen:
            continue
        seen.add(canonical)
        try:
            if source.startswith(("(", "[")) and source.endswith((")", "]")) and "," in source:
                _parse_interval(source, [])
                certificate_type = "interval_equivalence"
            else:
                value = _parse_expression(source, [])
                if value.free_symbols:
                    continue
                certificate_type = "expression_evaluation"
        except (UnsafeExpression, TypeError, ValueError, SyntaxError, sympy.SympifyError):
            continue
        constants = tuple(token.lstrip("+") for token in _NUMBER_RE.findall(source))
        return MathCheckSpec(
            certificate_type=certificate_type,
            problem_expression=source,
            problem_constants=constants,
        )
    return None


def verify_math_certificate(
    *,
    question: str,
    leader_answer: str,
    payload: dict[str, Any],
) -> CertificateValidation:
    started = time.perf_counter()
    certificate_type = str(payload.get("certificate_type") or "").strip()
    answer = _normalize_answer(str(payload.get("answer") or payload.get("final_answer") or ""))
    problem_expression = str(payload.get("problem_expression") or "").strip()
    variables = _string_list(payload.get("problem_variables"))
    constants = _string_list(payload.get("problem_constants"))
    unit = str(payload.get("unit") or "").strip()

    def result(
        *,
        valid: bool,
        challenger_pass: bool = False,
        leader_pass: bool = False,
        failure_reason: str = "",
    ) -> CertificateValidation:
        return CertificateValidation(
            valid=valid,
            certificate_type=certificate_type,
            normalized_answer=answer,
            challenger_pass=challenger_pass,
            leader_pass=leader_pass,
            failure_reason=failure_reason,
            checker_runtime_ms=round((time.perf_counter() - started) * 1000.0, 6),
        )

    spec = compile_math_check_spec(question)
    if spec is None:
        return result(valid=False, failure_reason="uncompilable_question")
    if certificate_type not in {"expression_evaluation", "interval_equivalence"}:
        return result(valid=False, failure_reason="unsupported_certificate_type")
    if not answer or not problem_expression:
        return result(valid=False, failure_reason="incomplete_certificate")
    if (
        certificate_type != spec.certificate_type
        or _canonical_spec_source(problem_expression) != _canonical_spec_source(spec.problem_expression)
    ):
        return result(valid=False, failure_reason="certificate_spec_mismatch")
    if (
        tuple(token.lstrip("+") for token in constants) != spec.problem_constants
        or tuple(variables) != spec.problem_variables
        or unit != spec.unit
    ):
        return result(valid=False, failure_reason="problem_signature_mismatch")
    try:
        if certificate_type == "interval_equivalence":
            target_value = _parse_interval(problem_expression, variables)
            challenger_value = _parse_interval(_strip_unit(answer, unit), variables)
            challenger_pass = challenger_value == target_value
            leader_pass = _safe_interval_check(leader_answer, unit, variables, target_value)
        else:
            target_value = _parse_expression(problem_expression, variables)
            challenger_value = _parse_expression(_strip_unit(answer, unit), variables)
            challenger_pass = _equivalent(challenger_value, target_value)
            leader_pass = _safe_expression_check(leader_answer, unit, variables, target_value)
    except (UnsafeExpression, TypeError, ValueError, SyntaxError, sympy.SympifyError):
        return result(valid=False, failure_reason="unsafe_expression")
    if not challenger_pass:
        return result(valid=False, challenger_pass=False, leader_pass=leader_pass, failure_reason="challenger_check_failed")
    return result(valid=True, challenger_pass=True, leader_pass=leader_pass)


def _parse_expression(value: str, variables: list[str]) -> sympy.Expr:
    prepared = _prepare_expression(value)
    tree = ast.parse(prepared, mode="eval")
    symbols = {name: sympy.Symbol(name, real=True) for name in variables}
    return _from_ast(tree.body, symbols)


def _from_ast(node: ast.AST, symbols: dict[str, sympy.Symbol]) -> sympy.Expr:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return sympy.Integer(node.value) if isinstance(node.value, int) else sympy.Rational(str(node.value))
    if isinstance(node, ast.Name):
        if node.id == "pi":
            return sympy.pi
        if node.id in {"inf", "infinity", "oo"}:
            return sympy.oo
        if node.id in symbols:
            return symbols[node.id]
        raise UnsafeExpression(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _from_ast(node.operand, symbols)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
        left = _from_ast(node.left, symbols)
        right = _from_ast(node.right, symbols)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        return left**right
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _SAFE_CALLS:
        if node.keywords:
            raise UnsafeExpression("keywords")
        return _SAFE_CALLS[node.func.id](*[_from_ast(arg, symbols) for arg in node.args])
    raise UnsafeExpression(type(node).__name__)


def _parse_interval(value: str, variables: list[str]) -> tuple[str, sympy.Expr, sympy.Expr, str]:
    text = _prepare_expression(value).strip()
    match = re.fullmatch(r"([\[(])\s*(.+?)\s*,\s*(.+?)\s*([\])])", text)
    if not match:
        raise UnsafeExpression("interval")
    return (
        match.group(1),
        _parse_expression(match.group(2), variables),
        _parse_expression(match.group(3), variables),
        match.group(4),
    )


def _satisfies_equation(equation: str, candidate: str, variables: list[str]) -> bool:
    if len(variables) != 1 or equation.count("=") != 1:
        raise UnsafeExpression("equation")
    left, right = equation.split("=", 1)
    symbol = sympy.Symbol(variables[0], real=True)
    value = _parse_expression(candidate, variables)
    residual = _parse_expression(left, variables) - _parse_expression(right, variables)
    return _equivalent(residual.subs(symbol, value), sympy.Integer(0))


def _equivalent(left: sympy.Expr, right: sympy.Expr) -> bool:
    return bool(sympy.simplify(left - right) == 0)


def _prepare_expression(value: str) -> str:
    text = _normalize_answer(value)
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\cdot", "*").replace("\\times", "*")
    text = text.replace("^", "**")
    text = _replace_latex_fraction(text)
    text = _replace_latex_sqrt(text)
    text = re.sub(r"(?<=\d)(?=(?:pi|[A-Za-z]))", "*", text)
    return text


def _replace_latex_fraction(value: str) -> str:
    pattern = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
    previous = ""
    while value != previous:
        previous = value
        value = pattern.sub(r"((\1)/(\2))", value)
    return value


def _replace_latex_sqrt(value: str) -> str:
    return re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", value)


def _normalize_answer(value: str) -> str:
    return str(value or "").strip().replace("π", "pi").replace("\\pi", "pi").replace("\\infty", "infinity")


def _strip_unit(value: str, unit: str) -> str:
    if not unit:
        return value
    normalized_unit = re.sub(r"\s+", " ", unit.strip())
    match = re.fullmatch(rf"\s*(.*?)\s*{re.escape(normalized_unit)}\s*", value)
    if not match:
        raise UnsafeExpression("unit")
    return match.group(1)


def _source_is_bound_to_question(question: str, source: str) -> bool:
    normalized_question = re.sub(r"\s+", "", _prepare_expression(question)).lower()
    normalized_source = re.sub(r"\s+", "", _prepare_expression(source)).lower()
    return bool(normalized_source and normalized_source in normalized_question)


def _plain_expression_candidates(question: str) -> list[str]:
    candidates: list[str] = []
    for pattern in (
        r"(?is)\b(?:evaluate|compute|calculate|simplify)\s+(?:the\s+)?(?:value\s+of\s+)?([^\n]+)",
        r"(?is)\bwhat\s+is\s+([^?\n]+)",
    ):
        match = re.search(pattern, question)
        if match:
            candidates.append(match.group(1).strip())
    return candidates


def _canonical_spec_source(value: str) -> str:
    return re.sub(r"\s+", "", _prepare_expression(value)).lower()


def _problem_signature_matches(
    question: str,
    source: str,
    constants: list[str],
    variables: list[str],
    unit: str,
) -> bool:
    source_numbers = Counter(token.lstrip("+") for token in _NUMBER_RE.findall(source))
    declared_numbers = Counter(token.lstrip("+") for token in constants)
    if source_numbers != declared_numbers:
        return False
    if not all(
        re.search(rf"\b{re.escape(name)}\b", question) and re.search(rf"\b{re.escape(name)}\b", source)
        for name in variables
    ):
        return False
    return not unit or _normalize_text(unit) in _normalize_text(question)


def _safe_expression_check(answer: str, unit: str, variables: list[str], target: sympy.Expr) -> bool:
    try:
        return _equivalent(_parse_expression(_strip_unit(_normalize_answer(answer), unit), variables), target)
    except (UnsafeExpression, TypeError, ValueError, SyntaxError, sympy.SympifyError):
        return False


def _safe_interval_check(
    answer: str,
    unit: str,
    variables: list[str],
    target: tuple[str, sympy.Expr, sympy.Expr, str],
) -> bool:
    try:
        return _parse_interval(_strip_unit(_normalize_answer(answer), unit), variables) == target
    except (UnsafeExpression, TypeError, ValueError, SyntaxError, sympy.SympifyError):
        return False


def _safe_equation_check(equation: str, answer: str, unit: str, variables: list[str]) -> bool:
    try:
        return _satisfies_equation(equation, _strip_unit(_normalize_answer(answer), unit), variables)
    except (UnsafeExpression, TypeError, ValueError, SyntaxError, sympy.SympifyError):
        return False


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]
