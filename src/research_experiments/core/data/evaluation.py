"""预测归一化与任务级打分逻辑。

本模块刻意保持轻量，只实现当前仓库需要的答案归一化和精确匹配打分，
以便所有实验线共享同一套“答案如何比较”的基础规则。
"""

from __future__ import annotations

import ast
import json
import math
import re
import string
import subprocess
import sys
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy

if TYPE_CHECKING:
    from research_experiments.core.data.datasets import DatasetSample


@dataclass(frozen=True)
class AnswerCanonicalization:
    """The sample-aware, conservative result used by scoring and voting."""

    key: str
    valid: bool
    invalid_reason: str | None = None


def normalize_prediction(dataset: str, final_answer: str) -> str:
    """按数据集类型把模型答案归一化为可比较的形式。"""
    if dataset == "gsm8k":
        return normalize_number(final_answer)
    if dataset in {"math500", "competition_math", "omni_math", "omni_math_2_filtered"}:
        return normalize_math_expression(final_answer)
    if dataset == "strategyqa":
        return normalize_yes_no(final_answer)
    if dataset == "commongen_hard":
        return normalize_commongen_sentence(final_answer)
    if dataset in {
        "realmistake_math_problem_generation",
        "realmistake_fine_grained_fact_verification",
        "realmistake_answerability_classification",
    }:
        return normalize_error_detection_verdict(final_answer)
    if dataset in {
        "hotpotqa",
        "webquestions",
    }:
        return normalize_text(final_answer)
    if dataset == "bbeh":
        return normalize_bbeh_prediction(final_answer)
    if dataset in {"mmlu_pro", "gpqa_diamond", "mmlu_abstract_algebra", "musr"}:
        return normalize_multiple_choice(final_answer)
    if dataset == "mmlu":
        return normalize_multiple_choice(final_answer)
    if dataset == "humaneval":
        return normalize_code_completion(final_answer)
    if dataset == "seqbench":
        return _canonical_sequence_key(final_answer).key
    raise ValueError(f"Unsupported dataset {dataset}")


def normalize_gold(dataset: str, answer: str) -> str:
    """对金标答案沿用与预测值一致的归一化规则。"""
    if dataset == "webquestions":
        answers = _decode_text_answer_set_gold(answer)
        return normalize_text(answers[0]) if answers else ""
    if dataset == "bbeh":
        return normalize_bbeh_reference(answer)
    return normalize_prediction(dataset, answer)


def canonicalize_answer(sample: DatasetSample, raw_answer: str) -> AnswerCanonicalization:
    """Canonicalize an answer using the current sample's exact answer contract.

    BBEH includes both free-form tasks and multiple-choice tasks.  For the
    latter, a label only has meaning in the context of that sample's option
    table, so callers must use this entry point rather than normalizing a
    string in isolation.  Invalid answers deliberately have no key: they
    cannot create a vote class, trigger an intervention, or be promoted.
    """

    value = str(raw_answer or "").strip()
    if not value:
        return AnswerCanonicalization("", False, "empty_answer")
    contract = sample.metadata.get("answer_contract")
    contract_kind = str(contract.get("kind") or "") if isinstance(contract, dict) else ""
    contract_options = contract.get("options") if isinstance(contract, dict) else None
    if contract_kind == "ordered_sequence" or sample.dataset == "seqbench":
        return _canonical_sequence_key(value)
    if contract_kind in {"single_choice", "multi_choice"}:
        options = contract_options if isinstance(contract_options, list) else sample.metadata.get("options")
        if not isinstance(options, list) or not options:
            return AnswerCanonicalization("", False, "invalid_option_metadata")
        selection_mode = str(contract.get("selection_mode") or "single")
        if selection_mode == "concatenated":
            return _canonicalize_bbeh_multi_select(value, options)
        return _canonicalize_bbeh_multiple_choice(value, options)
    if sample.dataset != "bbeh":
        key = normalize_prediction(sample.dataset, value)
        return AnswerCanonicalization(key, bool(key), None if key else "empty_normalized_answer")

    options = sample.metadata.get("options")
    if not isinstance(options, list) or not options:
        key = _bbeh_format_key(value)
        return AnswerCanonicalization(key, bool(key), None if key else "empty_normalized_answer")
    contract = sample.metadata.get("answer_contract")
    selection_mode = (
        str(contract.get("selection_mode") or "")
        if isinstance(contract, dict)
        else str(sample.metadata.get("option_selection_mode") or "")
    )
    if selection_mode == "concatenated":
        return _canonicalize_bbeh_multi_select(value, options)
    return _canonicalize_bbeh_multiple_choice(value, options)


def answer_class_key(dataset: str, answer: str, *, sample: DatasetSample | None = None) -> str:
    """Return the conservative, label-free equivalence key used by voting and scoring.

    BBEH's official scorer accepts a small set of formatting variants.  Keeping
    those variants as separate vote classes creates artificial disagreement, so
    the exact same conservative key is now shared by aggregation and scoring.
    This intentionally does not attempt semantic or model-based equivalence.
    """

    if sample is not None:
        if sample.dataset != dataset:
            raise ValueError(f"Sample dataset {sample.dataset!r} does not match {dataset!r}.")
        return canonicalize_answer(sample, answer).key
    if dataset != "bbeh":
        return normalize_prediction(dataset, answer)
    return _bbeh_format_key(answer)


def _bbeh_format_key(answer: str) -> str:
    """Legacy conservative BBEH formatting key for non-option samples."""

    value = unicodedata.normalize("NFKC", normalize_bbeh_prediction(answer)).strip()
    quote_pairs = {('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’")}
    while len(value) >= 2 and (value[0], value[-1]) in quote_pairs:
        value = value[1:-1].strip()
    if len(value) == 3 and value[0] == "(" and value[-1] == ")" and value[1].isalnum():
        value = value[1]
    if len(value) >= 2 and value[0] == "[" and value[-1] == "]" and "[" not in value[1:-1]:
        value = value[1:-1].strip()
    value = value.replace("'", "")
    if value.endswith("?"):
        value = value[:-1].rstrip()
    try:
        numeric = float(value)
    except ValueError:
        return value
    if not math.isfinite(numeric):
        return value
    if math.isclose(numeric, round(numeric)):
        return str(int(round(numeric)))
    return format(numeric, ".15g")


def _canonicalize_bbeh_multiple_choice(
    raw_answer: str,
    options: list[object],
) -> AnswerCanonicalization:
    option_by_label: dict[str, str] = {}
    labels_by_text: dict[str, list[str]] = {}
    for option in options:
        if not isinstance(option, dict):
            continue
        label = str(option.get("label") or "").strip().upper()
        text = _exact_option_text(str(option.get("text") or ""))
        if not re.fullmatch(r"[A-Z]", label) or not text or label in option_by_label:
            continue
        option_by_label[label] = text
        labels_by_text.setdefault(text, []).append(label)
    if not option_by_label:
        return AnswerCanonicalization("", False, "invalid_option_metadata")

    value = _strip_bbeh_answer_wrapper(raw_answer)
    label_match = _match_bbeh_option_label(value)
    if label_match is not None:
        label, raw_remainder = label_match
        remainder = _exact_option_text(raw_remainder)
        if label not in option_by_label:
            return AnswerCanonicalization("", False, "unknown_option_label")
        if not remainder:
            return AnswerCanonicalization(label, True)
        if remainder == option_by_label[label]:
            return AnswerCanonicalization(label, True)
        return AnswerCanonicalization("", False, "label_text_conflict")

    exact_text = _exact_option_text(value)
    labels = labels_by_text.get(exact_text, [])
    if len(labels) == 1:
        return AnswerCanonicalization(labels[0], True)
    if len(labels) > 1:
        return AnswerCanonicalization("", False, "ambiguous_option_text")
    return AnswerCanonicalization("", False, "unmapped_option_answer")


def _canonicalize_bbeh_multi_select(
    raw_answer: str,
    options: list[object],
) -> AnswerCanonicalization:
    option_by_label: dict[str, str] = {}
    labels_by_text: dict[str, list[str]] = {}
    for option in options:
        if not isinstance(option, dict):
            continue
        label = str(option.get("label") or "").strip().upper()
        text = _exact_option_text(str(option.get("text") or ""))
        if not re.fullmatch(r"[A-Z]", label) or not text or label in option_by_label:
            continue
        option_by_label[label] = text
        labels_by_text.setdefault(text, []).append(label)
    if not option_by_label:
        return AnswerCanonicalization("", False, "invalid_option_metadata")

    value = _strip_bbeh_answer_wrapper(raw_answer)
    exact_labels = labels_by_text.get(_exact_option_text(value), [])
    if len(exact_labels) == 1:
        return AnswerCanonicalization(exact_labels[0], True)
    if len(exact_labels) > 1:
        return AnswerCanonicalization("", False, "ambiguous_option_text")

    labelled = _match_bbeh_option_label(value)
    if labelled is not None and labelled[1]:
        label, remainder = labelled
        if label not in option_by_label:
            return AnswerCanonicalization("", False, "unknown_option_label")
        if _exact_option_text(remainder) == option_by_label[label]:
            return AnswerCanonicalization(label, True)
        return AnswerCanonicalization("", False, "label_text_conflict")

    if re.fullmatch(r"[A-Za-z]+", value) is None:
        return AnswerCanonicalization("", False, "invalid_multi_option_format")
    labels = value.upper()
    if len(set(labels)) != len(labels):
        return AnswerCanonicalization("", False, "duplicate_multi_option_label")
    if any(label not in option_by_label for label in labels):
        return AnswerCanonicalization("", False, "unknown_option_label")
    canonical = "".join(label for label in option_by_label if label in set(labels))
    if labels != canonical:
        return AnswerCanonicalization("", False, "multi_option_labels_out_of_order")
    return AnswerCanonicalization(canonical, True)


def _match_bbeh_option_label(value: str) -> tuple[str, str] | None:
    parenthesized = re.fullmatch(
        r"\(\s*([A-Za-z])\s*\)(?:\s+(.+))?",
        value,
        flags=re.DOTALL,
    )
    if parenthesized is not None:
        return parenthesized.group(1).upper(), str(parenthesized.group(2) or "")
    bare = re.fullmatch(
        r"([A-Za-z])(?:[\]\).:]?)(?:\s+(.+))?",
        value,
        flags=re.DOTALL,
    )
    if bare is None:
        return None
    return bare.group(1).upper(), str(bare.group(2) or "")


def _strip_bbeh_answer_wrapper(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).replace("\r\n", "\n").strip()
    normalized = re.sub(r"^(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*", "", normalized, flags=re.IGNORECASE)
    return normalized.strip()


def _exact_option_text(value: str) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).replace("\r\n", "\n").strip()


def _canonical_sequence_key(raw_answer: str) -> AnswerCanonicalization:
    """Strictly canonicalize an ordered list of seqBench actions."""

    value = unicodedata.normalize("NFKC", str(raw_answer or "")).replace("\r\n", "\n").replace("\r", "\n").strip()
    if not value:
        return AnswerCanonicalization("", False, "empty_answer")
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError, TypeError):
            return AnswerCanonicalization("", False, "invalid_sequence_syntax")
    if not isinstance(parsed, list) or not parsed:
        return AnswerCanonicalization("", False, "sequence_must_be_nonempty_list")
    if not all(isinstance(item, str) and item.strip() for item in parsed):
        return AnswerCanonicalization("", False, "sequence_elements_must_be_nonempty_strings")
    normalized = [
        unicodedata.normalize("NFKC", item).replace("\r\n", "\n").replace("\r", "\n").strip()
        for item in parsed
    ]
    return AnswerCanonicalization(json.dumps(normalized, ensure_ascii=False, separators=(",", ":")), True)


def canonical_reference_key(sample: DatasetSample) -> AnswerCanonicalization:
    """Resolve gold through the same sample-level contract used for predictions."""

    contract = sample.metadata.get("answer_contract")
    kind = str(contract.get("kind") or "") if isinstance(contract, dict) else ""
    if kind in {"single_choice", "multi_choice"}:
        answer_label = str(sample.metadata.get("answer_letter") or "").strip()
        if answer_label:
            return canonicalize_answer(sample, answer_label)
    return canonicalize_answer(sample, sample.reference_answer)


def score_prediction(
    dataset: str,
    predicted: str,
    gold: str,
    *,
    sample: DatasetSample | None = None,
) -> float:
    """计算单题得分。

    当前仓库统一采用精确匹配：归一化后完全一致记为 `1.0`，否则记为 `0.0`。
    """
    if sample is not None and (
        sample.dataset == "seqbench"
        or isinstance(sample.metadata.get("answer_contract"), dict)
        and str(sample.metadata["answer_contract"].get("kind") or "") in {"single_choice", "multi_choice", "ordered_sequence"}
    ):
        predicted_key = canonicalize_answer(sample, predicted)
        gold_key = canonical_reference_key(sample)
        return 1.0 if predicted_key.valid and gold_key.valid and predicted_key.key == gold_key.key else 0.0
    if dataset in {"mmlu_pro", "gpqa_diamond", "mmlu_abstract_algebra", "musr"}:
        return score_multiple_choice(predicted, gold)
    if dataset == "webquestions":
        return score_text_answer_set(predicted, gold)
    if dataset == "commongen_hard":
        return score_commongen_hard(predicted, gold)
    if dataset == "bbeh":
        return score_bbeh(predicted, gold, sample=sample)
    if dataset in {
        "realmistake_math_problem_generation",
        "realmistake_fine_grained_fact_verification",
        "realmistake_answerability_classification",
    }:
        return 1.0 if normalize_error_detection_verdict(predicted) == normalize_error_detection_verdict(gold) else 0.0
    if dataset == "humaneval":
        return score_humaneval(predicted, gold)
    if dataset == "mmlu":
        return score_multiple_choice(predicted, gold)
    return 1.0 if normalize_prediction(dataset, predicted) == normalize_gold(dataset, gold) else 0.0


def aggregate_majority(candidates: Iterable[str]) -> tuple[str, dict[str, int]]:
    """聚合同一题的多次回答，并在平票时保持“先出现者优先”。"""
    ordered = [candidate for candidate in candidates if candidate]
    counts = Counter(ordered)
    if not counts:
        return "", {}
    winner = max(counts.items(), key=lambda item: (item[1], -ordered.index(item[0])))[0]
    return winner, dict(counts)


def normalize_number(value: str) -> str:
    """把数值答案清洗成稳定字符串，避免 `1` 与 `1.0` 被视为不同。"""
    match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", value.replace(",", ""))
    if not match:
        return value.strip().lower()
    numeric = match.group(0).replace(",", "")
    try:
        as_float = float(numeric)
    except ValueError:
        return numeric
    if math.isclose(as_float, round(as_float)):
        return str(int(round(as_float)))
    return str(as_float).rstrip("0").rstrip(".")


def normalize_yes_no(value: str) -> str:
    """把多种 `yes / no` 变体收敛成标准二元标签。"""
    lowered = value.strip().lower()
    if lowered.startswith("yes"):
        return "yes"
    if lowered.startswith("no"):
        return "no"
    return lowered


def normalize_error_detection_verdict(value: str) -> str:
    """把错误检测标签归一成 `contains_error / contains_no_error`。"""

    lowered = normalize_text(value)
    if lowered in {
        "error",
        "contains error",
        "containserror",
        "contains an error",
        "the model response contains an error",
        "therefore the model response contains an error",
        "incorrect",
        "has error",
    }:
        return "contains_error"
    if lowered in {
        "no error",
        "noerror",
        "contains no error",
        "containsnoerror",
        "contains no errors",
        "contains no mistakes",
        "the model response contains no error",
        "therefore the model response contains no error",
        "correct",
    }:
        return "contains_no_error"
    return lowered


def normalize_text(value: str) -> str:
    """对文本答案做轻量归一化，近似常见 QA 任务的 EM 预处理。"""
    lowered = value.lower()
    lowered = re.sub(r"\b(a|an|the)\b", " ", lowered)
    lowered = lowered.translate(str.maketrans("", "", string.punctuation))
    lowered = " ".join(lowered.split())
    return lowered


def normalize_bbeh_prediction(value: str) -> str:
    answer = str(value or "").strip()
    for prefix in ("The answer is:", "The final answer is ", "The final answer is: ", "The answer is "):
        if prefix in answer:
            answer = answer.split(prefix)[-1].strip()
    if answer.endswith("."):
        answer = answer[:-1]
    answer = _strip_bbeh_latex(answer).lower()
    answer = answer.replace(", ", ",").replace("**", "")
    answer = answer.split("\n", 1)[0]
    return answer[:-1] if answer.endswith(".") else answer


def normalize_bbeh_reference(value: str) -> str:
    return str(value or "").strip().lower().replace(", ", ",")


def score_bbeh(predicted: str, gold: str, *, sample: DatasetSample | None = None) -> float:
    predicted_key = answer_class_key("bbeh", predicted, sample=sample)
    gold_key = answer_class_key("bbeh", gold, sample=sample)
    return 1.0 if predicted_key and predicted_key == gold_key else 0.0


def _strip_bbeh_latex(value: str) -> str:
    answer = value
    if answer.startswith("$") and answer.endswith("$"):
        answer = answer[1:-1]
    for marker in ("boxed{", "text{", "texttt{"):
        if marker in answer and answer.endswith("}"):
            answer = answer[:-1].split(marker)[-1]
    return answer


def normalize_multiple_choice(value: str) -> str:
    """把选择题答案归一成选项字母或标准化后的选项文本。"""
    normalized = value.strip()
    if not normalized:
        return ""
    match = re.search(r"\b([A-J])\b", normalized.upper())
    if match:
        return match.group(1)
    return normalize_text(normalized)


def normalize_commongen_sentence(value: str) -> str:
    """把 CommonGen 类生成题的答案清洗成稳定句子。"""

    cleaned = re.sub(r"\s+", " ", _strip_code_fences(value).strip())
    return cleaned.strip().strip("\"'")


def normalize_code_completion(value: str) -> str:
    """把 HumanEval 代码补全整理成可执行的补全文本。"""

    cleaned = _strip_code_fences(str(value or ""))
    return cleaned.replace("\r\n", "\n").rstrip() + ("\n" if cleaned.strip() else "")


def normalize_math_expression(value: str) -> str:
    """对短数学表达式做轻量归一化，尽量保留判题所需语义。"""
    normalized = str(value or "").strip().lower()
    normalized = _unwrap_latex_named_command_contents(normalized, commands=("boxed", "text", "textrm", "mathrm", "mbox"))
    normalized = normalized.replace("$", "")
    normalized = normalized.replace("\\left", "").replace("\\right", "")
    normalized = normalized.replace("\\!", "")
    normalized = re.sub(r"^[a-z]\s*(?:\\in|∈)\s*", "", normalized)
    normalized = re.sub(r"(?<=\d),\s*(?=\d{3}(?:\D|$))", "", normalized)
    normalized = normalized.rstrip(".")

    if _looks_like_textual_math_answer(normalized):
        return _normalize_textual_math_answer(normalized)

    top_level_parts = _split_math_top_level(normalized)
    if len(top_level_parts) > 1:
        canonical_parts = sorted(_normalize_math_atom(part) for part in top_level_parts)
        return ",".join(canonical_parts)

    return _normalize_math_atom(normalized)


def _normalize_math_atom(value: str) -> str:
    trimmed = value.strip()
    if not trimmed:
        return ""

    wrapped_sequence = _normalize_wrapped_math_sequence(trimmed)
    if wrapped_sequence is not None:
        return wrapped_sequence

    prepared = _prepare_math_expression(trimmed)
    numeric_value = _safe_numeric_evaluate(prepared)
    if numeric_value is not None:
        return _format_numeric_math_value(numeric_value)
    symbolic_value = _safe_symbolic_canonicalize(prepared)
    if symbolic_value is not None:
        return symbolic_value
    return prepared


def _normalize_wrapped_math_sequence(value: str) -> str | None:
    if len(value) < 2:
        return None
    wrapper_map = {"(": ")", "[": "]", "{": "}"}
    opener = value[0]
    closer = wrapper_map.get(opener)
    if closer is None or value[-1] != closer:
        return None
    inner = value[1:-1]
    parts = _split_math_top_level(inner)
    if len(parts) <= 1:
        return None
    normalized_parts = [_normalize_math_atom(part) for part in parts]
    return opener + ",".join(normalized_parts) + closer


def _split_math_top_level(value: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for character in value:
        if character in "([{":
            depth += 1
        elif character in ")]}":
            depth = max(0, depth - 1)
        if character == "," and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        current.append(character)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _prepare_math_expression(value: str) -> str:
    compact = value.replace(" ", "")
    compact = compact.replace("\\cdot", "*").replace("\\times", "*")
    compact = compact.replace("\\pi", "pi")
    compact = compact.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    compact = _normalize_latex_function_commands(compact)
    compact = compact.replace("^{\\circ}", "").replace("^\\circ", "")
    compact = _replace_latex_command(compact, command="frac", binary=True)
    compact = _replace_latex_command(compact, command="sqrt", binary=False)
    compact = compact.replace("{", "(").replace("}", ")")
    compact = compact.replace("^", "**")
    # Preserve idempotence when a canonicalized expression is normalized a
    # second time for scoring.  LaTeX juxtaposition such as ``3\sqrt{21}``
    # becomes ``3sqrt(21)`` above and must be made explicit before either the
    # numeric or symbolic safe evaluator sees it.
    # Exclude e/E so scientific notation such as ``1.2e+43`` remains numeric.
    compact = re.sub(r"(?<=[0-9)])(?=[a-df-zA-DF-Z])", "*", compact)
    compact = re.sub(r"(?<=\))(?=[a-zA-Z0-9(])", "*", compact)
    return compact


def _normalize_latex_function_commands(value: str) -> str:
    for command in ("sin", "cos", "tan", "cot", "sec", "csc", "log", "ln"):
        value = value.replace(f"\\{command}", command)
    return value


def _unwrap_latex_named_command_contents(value: str, *, commands: tuple[str, ...]) -> str:
    unwrapped = value
    for command in commands:
        token = f"\\{command}"
        pieces: list[str] = []
        cursor = 0
        while cursor < len(unwrapped):
            start = unwrapped.find(token, cursor)
            if start < 0:
                pieces.append(unwrapped[cursor:])
                break
            pieces.append(unwrapped[cursor:start])
            cursor = start + len(token)
            argument, cursor = _read_latex_argument(unwrapped, cursor)
            pieces.append(argument)
        unwrapped = "".join(pieces)
    return unwrapped


def _looks_like_textual_math_answer(value: str) -> bool:
    candidate = str(value or "").strip()
    if not candidate:
        return False
    if any(character.isdigit() for character in candidate):
        return False
    if any(operator in candidate for operator in ("+", "*", "/", "^", "_", "=")):
        return False
    return any(character.isalpha() for character in candidate)


def _normalize_textual_math_answer(value: str) -> str:
    stripped = str(value or "").strip()
    option_candidate = normalize_text(stripped)
    if re.fullmatch(r"[a-j]", option_candidate):
        return option_candidate.upper()
    return option_candidate


def _replace_latex_command(value: str, *, command: str, binary: bool) -> str:
    token = f"\\{command}"
    pieces: list[str] = []
    cursor = 0
    while cursor < len(value):
        start = value.find(token, cursor)
        if start < 0:
            pieces.append(value[cursor:])
            break
        pieces.append(value[cursor:start])
        cursor = start + len(token)
        first_argument, cursor = _read_latex_argument(value, cursor)
        if binary:
            second_argument, cursor = _read_latex_argument(value, cursor)
            pieces.append(f"(({first_argument})/({second_argument}))")
            continue
        pieces.append(f"sqrt({first_argument})")
    return "".join(pieces)


def _read_latex_argument(value: str, cursor: int) -> tuple[str, int]:
    if cursor >= len(value):
        return "", cursor
    if value[cursor] == "{":
        depth = 1
        index = cursor + 1
        collected: list[str] = []
        while index < len(value):
            character = value[index]
            if character == "{":
                depth += 1
                collected.append(character)
            elif character == "}":
                depth -= 1
                if depth == 0:
                    return "".join(collected), index + 1
                collected.append(character)
            else:
                collected.append(character)
            index += 1
        return "".join(collected), index
    token_end = cursor + 1
    if value[cursor] == "\\":
        while token_end < len(value) and value[token_end].isalpha():
            token_end += 1
        return value[cursor:token_end], token_end
    while token_end < len(value) and value[token_end].isalnum():
        token_end += 1
    return value[cursor:token_end], token_end


def _safe_numeric_evaluate(expression: str) -> float | None:
    if not expression:
        return None
    try:
        node = ast.parse(expression, mode="eval")
        value = _eval_math_ast(node.body)
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return value


def _eval_math_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _eval_math_ast(node.operand)
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
        left = _eval_math_ast(node.left)
        right = _eval_math_ast(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        return left**right
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sqrt" and len(node.args) == 1:
        return math.sqrt(_eval_math_ast(node.args[0]))
    if isinstance(node, ast.Name) and node.id == "pi":
        return math.pi
    raise ValueError(f"Unsupported math expression AST: {ast.dump(node)}")


def _format_numeric_math_value(value: float) -> str:
    rounded = round(value, 12)
    if abs(rounded) < 1e-12:
        rounded = 0.0
    return f"{rounded:.12g}"


def _safe_symbolic_canonicalize(expression: str) -> str | None:
    """Canonicalize small algebraic expressions using a strict AST whitelist.

    This intentionally handles only arithmetic over symbolic names.  It fixes
    exact-answer false negatives such as ``2015+2x+y`` versus ``2x+y+2015``
    without passing model text to ``sympify``/``eval`` or accepting equations,
    inequalities, indexing, attributes, or arbitrary function calls.
    """

    candidate = str(expression or "").strip()
    if not candidate or len(candidate) > 256:
        return None
    if not re.fullmatch(r"[a-zA-Z0-9_+\-*/().]+", candidate):
        return None
    if not re.search(r"[a-zA-Z]", candidate):
        return None
    candidate = re.sub(r"(?<=\d)(?=[a-zA-Z])", "*", candidate)
    candidate = re.sub(r"(?<=\))(?=[a-zA-Z0-9(])", "*", candidate)
    try:
        node = ast.parse(candidate, mode="eval")
        symbols: dict[str, sympy.Symbol] = {}
        value = _eval_symbolic_ast(node.body, symbols)
        canonical = sympy.factor_terms(sympy.expand(value))
    except Exception:
        return None
    rendered = str(canonical).replace(" ", "")
    return rendered if rendered and len(rendered) <= 512 else None


def _eval_symbolic_ast(node: ast.AST, symbols: dict[str, sympy.Symbol]) -> sympy.Expr:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return sympy.Rational(str(node.value))
    if isinstance(node, ast.Name):
        if node.id == "pi":
            return sympy.pi
        if node.id not in symbols:
            symbols[node.id] = sympy.Symbol(node.id, real=True)
        return symbols[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _eval_symbolic_ast(node.operand, symbols)
        return operand if isinstance(node.op, ast.UAdd) else -operand
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
        left = _eval_symbolic_ast(node.left, symbols)
        right = _eval_symbolic_ast(node.right, symbols)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if not right.is_number or abs(float(right)) > 32:
            raise ValueError("Symbolic exponent is outside the safe canonicalization bound.")
        return left**right
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sqrt" and len(node.args) == 1:
        return sympy.sqrt(_eval_symbolic_ast(node.args[0], symbols))
    raise ValueError(f"Unsupported symbolic math AST: {ast.dump(node)}")


def score_multiple_choice(predicted: str, gold: str) -> float:
    """选择题允许命中字母选项，或命中对应选项文本。"""
    predicted_norm = normalize_multiple_choice(predicted)
    gold_letter, gold_text = _decode_multiple_choice_gold(gold)
    accepted = {gold_letter}
    if gold_text:
        accepted.add(normalize_text(gold_text))
    return 1.0 if predicted_norm in accepted else 0.0


def _decode_multiple_choice_gold(gold: str) -> tuple[str, str]:
    if "|||" in gold:
        letter, text = gold.split("|||", 1)
        return normalize_multiple_choice(letter), text.strip()
    normalized = normalize_multiple_choice(gold)
    return normalized, ""


def score_commongen_hard(predicted: str, gold: str) -> float:
    """用概念覆盖率作为 CommonGen-Hard 的稳定主指标。"""

    normalized_prediction = normalize_commongen_sentence(predicted).lower()
    if not normalized_prediction:
        return 0.0
    try:
        payload = json.loads(gold)
    except json.JSONDecodeError:
        return 0.0
    concept_set = payload.get("concept_set") or []
    normalized_concepts = [_normalize_commongen_concept(item) for item in concept_set]
    normalized_concepts = [item for item in normalized_concepts if item]
    if not normalized_concepts:
        return 0.0
    hits = sum(1 for concept in normalized_concepts if concept in normalized_prediction)
    return round(hits / len(normalized_concepts), 6)


def score_humaneval(predicted: str, gold: str) -> float:
    """在本地 Python 子进程里执行 HumanEval 用例，返回 pass@1。"""

    try:
        payload = json.loads(gold)
    except json.JSONDecodeError:
        return 0.0

    prompt = str(payload.get("prompt") or "")
    test_code = str(payload.get("test") or "")
    entry_point = str(payload.get("entry_point") or "").strip()
    completion = normalize_code_completion(predicted)
    if not prompt or not test_code or not entry_point or not completion.strip():
        return 0.0

    program = "\n".join(
        [
            prompt.rstrip("\n"),
            completion.rstrip("\n"),
            "",
            test_code.strip("\n"),
            "",
            f"check({entry_point})",
            "",
        ]
    )
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".py", delete=False) as handle:
        handle.write(program)
        temp_path = handle.name
    try:
        completed = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return 0.0
    finally:
        try:
            import os

            os.unlink(temp_path)
        except OSError:
            pass
    return 1.0 if completed.returncode == 0 else 0.0


def score_text_answer_set(predicted: str, gold: str) -> float:
    """在多个文本别名上取最大 token-F1，更贴近图问答常见评测口径。"""
    predicted_norm = normalize_text(predicted)
    if not predicted_norm:
        return 0.0
    gold_aliases = [normalize_text(answer) for answer in _decode_text_answer_set_gold(gold) if normalize_text(answer)]
    if not gold_aliases:
        return 0.0
    return round(max(_token_f1(predicted_norm, gold_alias) for gold_alias in gold_aliases), 6)


def _decode_text_answer_set_gold(gold: str) -> list[str]:
    stripped = str(gold or "").strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
        if isinstance(payload, list):
            return [str(item).strip() for item in payload if str(item).strip()]
    return [stripped]


def _token_f1(predicted: str, gold: str) -> float:
    predicted_tokens = predicted.split()
    gold_tokens = gold.split()
    if not predicted_tokens or not gold_tokens:
        return 0.0
    common = Counter(predicted_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _normalize_commongen_concept(value: str) -> str:
    concept = str(value or "").strip().lower()
    concept = re.sub(r"_[nv]$", "", concept)
    concept = concept.replace("_", " ")
    return normalize_text(concept)


def _strip_code_fences(value: str) -> str:
    text = str(value or "")
    fence_match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip("\r\n")
    return text.strip("\r\n")
