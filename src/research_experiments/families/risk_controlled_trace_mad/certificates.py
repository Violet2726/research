"""数据集无关、失败即 unsupported 的 RCTA 可执行证书。"""

from __future__ import annotations

import ast
import json
import math
import operator
import re
from collections import Counter
from typing import Any

import sympy

CERTIFICATE_TYPES = frozenset({"arithmetic", "symbolic", "ordering", "boolean", "unsupported"})
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg, ast.Not: operator.not_}
_BOOL_OPS = {ast.And: all, ast.Or: any}
_COMPARE_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


def verify_certificate(*, question: str, final_answer: str, certificate_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    kind = str(certificate_type or "unsupported").strip().lower()
    if kind not in CERTIFICATE_TYPES or kind == "unsupported" or not isinstance(payload, dict):
        return _result("unsupported", kind if kind in CERTIFICATE_TYPES else "unsupported", "no_supported_certificate")
    try:
        if kind == "arithmetic":
            passed, detail = _verify_arithmetic(question, final_answer, payload)
        elif kind == "symbolic":
            passed, detail = _verify_symbolic(question, final_answer, payload)
        elif kind == "ordering":
            passed, detail = _verify_ordering(question, final_answer, payload)
        else:
            passed, detail = _verify_boolean(question, final_answer, payload)
    except (ArithmeticError, TypeError, ValueError, SyntaxError, sympy.SympifyError, OverflowError) as exc:
        return _result("unsupported", kind, f"unsafe_or_unbound:{type(exc).__name__}")
    return _result("pass" if passed else "fail", kind, detail)


def _verify_arithmetic(question: str, final_answer: str, payload: dict[str, Any]) -> tuple[bool, str]:
    expression = _text(payload, "expression")
    claimed = payload.get("claimed_value", final_answer)
    _require_operands_grounded(expression, question)
    value = _eval_ast(ast.parse(expression, mode="eval").body, {})
    claimed_value = _eval_ast(ast.parse(str(claimed), mode="eval").body, {})
    final_value = _eval_ast(ast.parse(str(final_answer), mode="eval").body, {})
    if not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in (value, claimed_value, final_value)):
        raise ValueError("arithmetic certificate is not numeric")
    passed = math.isclose(float(value), float(claimed_value), rel_tol=1e-9, abs_tol=1e-9)
    passed = passed and math.isclose(float(claimed_value), float(final_value), rel_tol=1e-9, abs_tol=1e-9)
    return passed, "arithmetic_recomputed_and_final_bound"


def _verify_symbolic(question: str, final_answer: str, payload: dict[str, Any]) -> tuple[bool, str]:
    left = _text(payload, "left")
    right = _text(payload, "right")
    _require_symbols_grounded(left + " " + right, question + " " + final_answer)
    substitutions = payload.get("substitutions") or {}
    if not isinstance(substitutions, dict):
        raise ValueError("substitutions must be an object")
    normalized_final = re.sub(r"\s+", "", final_answer).replace("^", "**")
    normalized_left = re.sub(r"\s+", "", left).replace("^", "**")
    normalized_right = re.sub(r"\s+", "", right).replace("^", "**")
    normalized_equations = {f"{normalized_left}={normalized_right}", f"{normalized_right}={normalized_left}"}
    if normalized_final not in {normalized_left, normalized_right, *normalized_equations}:
        raise ValueError("symbolic certificate is not bound to final answer")
    _require_operands_grounded(left, question + " " + final_answer)
    _require_operands_grounded(right, question + " " + final_answer)
    symbols: dict[str, sympy.Symbol] = {}
    left_expr = _sympy_ast(ast.parse(left.replace("^", "**"), mode="eval").body, symbols)
    right_expr = _sympy_ast(ast.parse(right.replace("^", "**"), mode="eval").body, symbols)
    bound = {}
    for key, raw_value in substitutions.items():
        if str(key) not in symbols or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", str(key)):
            raise ValueError("unbound substitution")
        _require_operands_grounded(str(raw_value), question + " " + final_answer)
        bound[symbols[str(key)]] = _sympy_ast(ast.parse(str(raw_value).replace("^", "**"), mode="eval").body, {})
    residual = sympy.simplify((left_expr - right_expr).subs(bound))
    return bool(residual == 0), "symbolic_equivalence_checked"


def _verify_ordering(question: str, final_answer: str, payload: dict[str, Any]) -> tuple[bool, str]:
    items = payload.get("items")
    ordered = payload.get("ordered_items")
    direction = str(payload.get("direction") or "ascending").lower()
    if not isinstance(items, list) or not isinstance(ordered, list) or not all(isinstance(item, str) for item in [*items, *ordered]):
        raise ValueError("ordering lists are invalid")
    if not items or Counter(items) != Counter(ordered):
        return False, "ordering_not_a_permutation"
    if any(item.casefold() not in question.casefold() for item in items):
        raise ValueError("ordering item is not grounded")
    if direction not in {"ascending", "descending"}:
        raise ValueError("unsupported ordering direction")
    expected = sorted(items, key=lambda value: value.casefold(), reverse=direction == "descending")
    final_tokens = re.findall(r"[A-Za-z0-9_]+", final_answer.casefold())
    ordered_tokens = [item.casefold() for item in ordered]
    return ordered == expected and final_tokens == ordered_tokens, "ordering_recomputed_and_final_bound"


def _verify_boolean(question: str, final_answer: str, payload: dict[str, Any]) -> tuple[bool, str]:
    expression = _text(payload, "expression")
    claimed = payload.get("claimed_value")
    variables = payload.get("variables") or {}
    if not isinstance(claimed, bool) or not isinstance(variables, dict) or not all(isinstance(value, bool) for value in variables.values()):
        raise ValueError("boolean payload is invalid")
    compact_expression = re.sub(r"\s+", "", expression).lower()
    compact_question = re.sub(r"\s+", "", question).lower()
    if compact_expression not in compact_question:
        raise ValueError("boolean expression is not grounded verbatim")
    _require_boolean_bindings_grounded(question, variables)
    value = _eval_ast(ast.parse(expression, mode="eval").body, {str(key): value for key, value in variables.items()})
    normalized_final = final_answer.strip().casefold()
    if normalized_final not in {"true", "false"}:
        raise ValueError("boolean final answer is not a literal")
    final_value = normalized_final == "true"
    return value is claimed and claimed is final_value, "boolean_expression_evaluated_and_final_bound"


def _eval_ast(node: ast.AST, variables: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
        return node.value
    if isinstance(node, ast.Name) and node.id in variables:
        return variables[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_ast(node.left, variables)
        right = _eval_ast(node.right, variables)
        if isinstance(node.op, ast.Pow) and abs(float(right)) > 12:
            raise ValueError("exponent too large")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_ast(node.operand, variables))
    if isinstance(node, ast.BoolOp) and type(node.op) in _BOOL_OPS:
        return _BOOL_OPS[type(node.op)]([bool(_eval_ast(item, variables)) for item in node.values])
    if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators):
        left = _eval_ast(node.left, variables)
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            if type(op) not in _COMPARE_OPS:
                raise ValueError("comparison is not allowed")
            right = _eval_ast(comparator, variables)
            if not _COMPARE_OPS[type(op)](left, right):
                return False
            left = right
        return True
    raise ValueError(f"unsafe AST node: {type(node).__name__}")


def _sympy_ast(node: ast.AST, symbols: dict[str, sympy.Symbol]) -> sympy.Expr:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return sympy.Rational(str(node.value))
    if isinstance(node, ast.Name):
        if node.id == "pi":
            return sympy.pi
        if node.id not in symbols:
            symbols[node.id] = sympy.Symbol(node.id, real=True)
        return symbols[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _sympy_ast(node.left, symbols)
        right = _sympy_ast(node.right, symbols)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
        raise ValueError("unsupported symbolic operator")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _sympy_ast(node.operand, symbols)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sqrt" and len(node.args) == 1:
        return sympy.sqrt(_sympy_ast(node.args[0], symbols))
    raise ValueError(f"unsafe symbolic node: {type(node).__name__}")


def _require_operands_grounded(expression: str, question: str) -> None:
    literals = re.findall(r"(?<![A-Za-z_])\d+(?:\.\d+)?", expression)
    question_literals = Counter(re.findall(r"(?<![A-Za-z_])\d+(?:\.\d+)?", question))
    for literal, count in Counter(literals).items():
        if question_literals[literal] < count:
            raise ValueError("arithmetic operand is not grounded")


def _require_symbols_grounded(expression: str, context: str) -> None:
    names = set(re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", expression)) - {"sqrt", "pi"}
    context_names = set(re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", context))
    if names - context_names:
        raise ValueError("symbol is not grounded")


def _require_boolean_bindings_grounded(question: str, variables: dict[str, bool]) -> None:
    for name, value in variables.items():
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", str(name)):
            raise ValueError("invalid boolean variable")
        literal = "true" if value else "false"
        pattern = rf"\b{re.escape(str(name))}\b\s*(?:=|:|\bis\b)\s*{literal}\b"
        if re.search(pattern, question, flags=re.IGNORECASE) is None:
            raise ValueError("boolean binding is not grounded")


def _text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip() or len(value) > 1000:
        raise ValueError(f"{key} must be bounded text")
    return value.strip()


def _result(status: str, kind: str, detail: str) -> dict[str, Any]:
    return {"status": status, "certificate_type": kind, "detail": detail}


def canonical_payload_hash(payload: dict[str, Any]) -> str:
    """Stable serialized payload used by diagnostics/tests without executing it."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
