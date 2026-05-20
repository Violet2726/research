"""共享的 PoT 程序抽取与安全执行工具。"""

from __future__ import annotations

from dataclasses import dataclass
import ast
import json
import re
import subprocess
import sys
import tempfile

from research_experiments.core.structured_outputs import SCHEMA_ANSWER_CORE, validate_or_recover_structured_output
from research_experiments.families.shared.reasoning_methods import normalize_reasoning_method_name


ALLOWED_IMPORT_ROOTS = {
    "collections",
    "datetime",
    "decimal",
    "fractions",
    "functools",
    "itertools",
    "math",
    "operator",
    "statistics",
    "sympy",
}

_FORBIDDEN_NAME_PARTS = {
    "__builtins__",
    "__class__",
    "__dict__",
    "__getattribute__",
    "__import__",
    "__loader__",
    "__mro__",
    "__subclasses__",
    "__traceback__",
    "breakpoint",
    "compile",
    "delattr",
    "eval",
    "exec",
    "exit",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "os",
    "pathlib",
    "setattr",
    "socket",
    "subprocess",
    "sys",
}

_FORBIDDEN_NODE_TYPES = (
    ast.AsyncFor,
    ast.AsyncFunctionDef,
    ast.AsyncWith,
    ast.Await,
    ast.ClassDef,
    ast.Global,
    ast.Nonlocal,
    ast.Raise,
    ast.Try,
    ast.With,
)

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)
@dataclass(frozen=True)
class PotExecutionArtifact:
    """PoT 程序解析与执行后的标准产物。"""

    reasoning: str
    final_answer: str
    program_text: str | None
    execution_result: str | None
    execution_status: str
    execution_resolution: str | None
    execution_error: str | None


def is_pot_reasoning(dataset: str, strategy_name: str) -> bool:
    """判断当前策略在该数据集上是否实际走 PoT 协议。"""

    return normalize_reasoning_method_name(dataset, strategy_name) == "pot"


def build_pot_process_artifact(assistant_text: str, provider_reasoning_text: str) -> PotExecutionArtifact:
    """从 PoT 过程阶段文本中抽取程序并执行。"""

    reasoning = _first_non_empty_text(assistant_text, provider_reasoning_text)
    if not reasoning:
        raise ValueError("PoT 过程输出为空。")
    program_text = extract_python_program(reasoning)
    execution_result, execution_status, execution_resolution, execution_error = _execute_program_if_present(program_text)
    return PotExecutionArtifact(
        reasoning=reasoning,
        final_answer="",
        program_text=program_text,
        execution_result=execution_result,
        execution_status=execution_status,
        execution_resolution=execution_resolution,
        execution_error=execution_error,
    )


def build_pot_answer_artifact(
    assistant_text: str,
    provider_reasoning_text: str,
    *,
    dataset: str,
) -> PotExecutionArtifact:
    """从 PoT 最终回答中抽取 JSON、程序和最终答案。"""

    raw_text = _first_non_empty_text(assistant_text, provider_reasoning_text)
    if not raw_text:
        raise ValueError("PoT 最终回答为空。")
    payload = _try_decode_json_object(raw_text) or {}
    answer_payload = validate_or_recover_structured_output(
        raw_text,
        SCHEMA_ANSWER_CORE,
        dataset=dataset,
        provider_reasoning_text=provider_reasoning_text,
    )
    reasoning = str(payload.get("reasoning") or answer_payload.get("reasoning") or "").strip()
    program_text = _clean_program_text(payload.get("python_program")) or extract_python_program(raw_text)
    final_answer = str(answer_payload.get("final_answer") or "").strip()
    execution_result, execution_status, execution_resolution, execution_error = _execute_program_if_present(program_text)
    if execution_status == "ok" and execution_result:
        final_answer = execution_result
    if execution_status != "ok" and final_answer:
        execution_result = final_answer
        execution_status = "ok"
        execution_resolution = "answer_field"
        execution_error = None
    if not final_answer:
        raise ValueError("PoT 最终回答缺少可用答案。")
    return PotExecutionArtifact(
        reasoning=reasoning,
        final_answer=final_answer,
        program_text=program_text,
        execution_result=execution_result,
        execution_status=execution_status,
        execution_resolution=execution_resolution,
        execution_error=execution_error,
    )


def extract_python_program(raw_text: str) -> str | None:
    """从模型文本中尽量稳定地提取 Python 程序。"""

    text = str(raw_text or "").strip()
    if not text:
        return None
    text = _strip_open_code_fence(text)
    for match in _CODE_BLOCK_RE.finditer(text):
        program = _clean_program_text(match.group(1))
        if program:
            return program
    if "ans" not in text:
        if not _contains_python_signal(text):
            return None
    candidate_lines = text.splitlines()
    start_index = None
    for index, line in enumerate(candidate_lines):
        stripped = line.strip()
        if not stripped:
            continue
        if _looks_like_code_line(stripped) or stripped.startswith("#"):
            start_index = index
            break
    if start_index is None:
        return None
    program = _clean_program_text("\n".join(candidate_lines[start_index:]))
    return program or None


def execute_pot_program(program_text: str, *, timeout_seconds: int = 5) -> PotExecutionArtifact:
    """对外暴露的安全执行入口，便于测试。"""

    cleaned = _clean_program_text(program_text)
    if not cleaned:
        return PotExecutionArtifact(
            reasoning="",
            final_answer="",
            program_text=None,
            execution_result=None,
            execution_status="missing_program",
            execution_resolution=None,
            execution_error="missing_program",
        )
    execution_result, execution_status, execution_resolution, execution_error = _execute_program_if_present(
        cleaned,
        timeout_seconds=timeout_seconds,
    )
    return PotExecutionArtifact(
        reasoning="",
        final_answer=execution_result or "",
        program_text=cleaned,
        execution_result=execution_result,
        execution_status=execution_status,
        execution_resolution=execution_resolution,
        execution_error=execution_error,
    )


def _execute_program_if_present(
    program_text: str | None,
    *,
    timeout_seconds: int = 5,
) -> tuple[str | None, str, str | None]:
    if not program_text:
        return None, "missing_program", None, "missing_program"
    try:
        _validate_program_safety(program_text)
    except Exception as exc:
        return None, "unsafe_program", None, str(exc)
    try:
        execution_result, execution_resolution = _run_program(program_text, timeout_seconds=timeout_seconds)
    except Exception as exc:
        error_text = str(exc)
        if "missing_result" in error_text:
            return None, "missing_result", None, "missing_result"
        return None, "runtime_error", None, error_text
    if not execution_result:
        return None, "missing_result", None, "missing_result"
    return execution_result, "ok", execution_resolution, None


def _validate_program_safety(program_text: str) -> None:
    tree = ast.parse(program_text, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODE_TYPES):
            raise ValueError(f"Unsupported Python construct: {type(node).__name__}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                _validate_import_name(alias.name)
        if isinstance(node, ast.ImportFrom):
            if node.module is None:
                raise ValueError("Relative imports are not allowed.")
            _validate_import_name(node.module)
        if isinstance(node, ast.Name):
            _validate_name_part(node.id)
        if isinstance(node, ast.Attribute):
            _validate_name_part(node.attr)
        if isinstance(node, ast.Call):
            target_name = _call_target_name(node.func)
            if target_name:
                for part in target_name.split("."):
                    _validate_name_part(part)


def _validate_import_name(import_name: str) -> None:
    root = str(import_name or "").split(".", 1)[0]
    if root not in ALLOWED_IMPORT_ROOTS:
        raise ValueError(f"Import root {root!r} is not allowed.")


def _validate_name_part(name_part: str) -> None:
    normalized = str(name_part or "").strip()
    if not normalized:
        return
    if normalized.startswith("__") or normalized in _FORBIDDEN_NAME_PARTS:
        raise ValueError(f"Name {normalized!r} is not allowed.")


def _call_target_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_target_name(node.value)
        return node.attr if not parent else f"{parent}.{node.attr}"
    return None


def _run_program(program_text: str, *, timeout_seconds: int) -> tuple[str | None, str]:
    wrapper = f"""
import json

_PROGRAM = {json.dumps(program_text, ensure_ascii=False)}
_ALLOWED_IMPORT_ROOTS = {sorted(ALLOWED_IMPORT_ROOTS)!r}
_PRINT_BUFFER = []

def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = str(name).split(".", 1)[0]
    if root not in _ALLOWED_IMPORT_ROOTS:
        raise ImportError(f"Import root {{root!r}} is not allowed.")
    return __import__(name, globals, locals, fromlist, level)

SAFE_BUILTINS = {{
    "__import__": _safe_import,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "pow": pow,
    "print": lambda *args, **kwargs: _PRINT_BUFFER.append((" ".join(str(item) for item in args) + str(kwargs.get("end", "\\n"))).rstrip()),
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}}

def _stringify(value):
    if isinstance(value, dict):
        return json.dumps({{str(key): _stringify(item) for key, item in value.items()}}, ensure_ascii=False, sort_keys=True)
    if isinstance(value, (list, tuple, set)):
        items = [_stringify(item) for item in value]
        if isinstance(value, set):
            items = sorted(items)
        return ", ".join(item for item in items if item != "")
    return str(value)

def _recover_execution_value(namespace):
    if "x1" in namespace and "x2" in namespace:
        return [namespace["x1"], namespace["x2"]]
    for key in ["solutions", "solution", "real_solutions", "roots", "result", "results", "count", "total", "answer"]:
        if key in namespace:
            return namespace[key]
    if _PRINT_BUFFER:
        return _PRINT_BUFFER[-1]
    return None

namespace = {{"__builtins__": SAFE_BUILTINS}}
exec(compile(_PROGRAM, "<pot>", "exec"), namespace, namespace)
status = "ok"
value = namespace.get("ans")
if value is None:
    value = _recover_execution_value(namespace)
    if value is None:
        raise RuntimeError("missing_result")
    resolution = "recovered_variables"
else:
    resolution = "direct"
print(json.dumps({{"resolution": resolution, "value": _stringify(value)}}, ensure_ascii=False))
"""
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".py", delete=False) as handle:
        handle.write(wrapper)
        wrapper_path = handle.name
    try:
        completed = subprocess.run(
            [sys.executable, "-I", wrapper_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    finally:
        try:
            import os

            os.unlink(wrapper_path)
        except OSError:
            pass
    if completed.returncode != 0:
        stderr = (completed.stderr or completed.stdout or "").strip() or "program_failed"
        raise RuntimeError(stderr)
    payload = json.loads((completed.stdout or "").strip() or "{}")
    return str(payload.get("value") or "").strip() or None, str(payload.get("resolution") or "direct")


def _try_decode_json_object(raw_text: str) -> dict[str, object] | None:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _clean_program_text(program_text: object) -> str | None:
    if program_text is None:
        return None
    cleaned = str(program_text).strip()
    if not cleaned:
        return None
    if cleaned.startswith("```"):
        match = _CODE_BLOCK_RE.search(cleaned)
        if match:
            cleaned = match.group(1).strip()
        else:
            cleaned = _strip_open_code_fence(cleaned)
    return cleaned or None


def _first_non_empty_text(*candidates: str) -> str:
    for item in candidates:
        cleaned = str(item or "").strip()
        if cleaned:
            return cleaned
    return ""


def _looks_like_code_line(line: str) -> bool:
    return bool(
        re.match(
            r"^(ans\s*=|from\s+\w+\s+import|import\s+\w+|for\s+.+:|while\s+.+:|if\s+.+:|def\s+\w+\(|[A-Za-z_]\w*\s*=)",
            line,
        )
    )


def _strip_open_code_fence(text: str) -> str:
    stripped = re.sub(r"^```(?:python)?\s*\n?", "", text, count=1, flags=re.IGNORECASE)
    if stripped.endswith("```"):
        stripped = stripped[:-3]
    return stripped.strip()


def _contains_python_signal(text: str) -> bool:
    lowered = text.lower()
    return any(signal in lowered for signal in ["import ", "from ", "def ", "for ", "while ", "if ", "ans =", "print(", "# "])
