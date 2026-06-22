"""MiMo 兼容的自由文本标签行协议辅助工具。"""

from __future__ import annotations

import re
from typing import Any

from research_experiments.core.data.evaluation import normalize_prediction
from research_experiments.core.prompts.dataset_contracts import build_tagged_lines_system_prompt

FREE_TEXT_ANSWER_PROTOCOL_V1 = "free_text_answer_v1"

_MULTIPLE_CHOICE_DATASETS = {
    "gpqa_diamond",
    "mmlu",
    "mmlu_abstract_algebra",
    "mmlu_pro",
}
_MATH_DATASETS = {"gsm8k", "math500", "competition_math"}
_SHORT_SPAN_DATASETS = {"hotpotqa", "webquestions"}


def build_free_text_system_prompt(
    role_description: str,
    *,
    extra_rules: list[str] | None = None,
) -> str:
    """Build a system prompt for a short tagged-line answer protocol."""

    return build_tagged_lines_system_prompt(
        role_description,
        extra_rules=[
            "Return only the requested tagged lines.",
            "Do not add markdown fences or prose before or after the tagged lines.",
            *(extra_rules or []),
        ],
    )


def adapt_messages_for_free_text_protocol(
    messages: list[dict[str, str]],
    *,
    dataset: str,
    role: str,
) -> list[dict[str, str]]:
    """Rewrite an existing JSON-oriented prompt into a tagged-line protocol."""

    if len(messages) < 2:
        raise ValueError("Expected at least a system message and a user message.")

    adapted = [dict(item) for item in messages]
    adapted[0]["content"] = build_free_text_system_prompt(_role_description(role))

    user_message = dict(adapted[-1])
    base_user = _strip_json_contract(user_message.get("content", ""))
    protocol_block = (
        _answer_protocol_instruction(dataset)
        if role == "debate"
        else _answer_protocol_instruction(dataset)
    )
    user_message["content"] = f"{base_user.rstrip()}\n\n{protocol_block}"
    adapted[-1] = user_message
    return adapted


def parse_free_text_answer_output(
    raw_text: str,
    *,
    dataset: str,
) -> dict[str, Any]:
    """Parse a tagged-line free-text answer into a compact structured payload."""

    cleaned = _strip_code_fences(str(raw_text or "").strip())
    if not cleaned:
        raise ValueError("Assistant output is empty.")

    reasoning_match = _extract_labeled_match(cleaned, ["REASONING", "REASON", "RATIONALE", "WHY"])
    if reasoning_match is None:
        raise ValueError("Missing REASONING line.")
    final_match = _extract_labeled_match(cleaned, ["FINAL_ANSWER", "ANSWER", "FINAL"])
    if final_match is None:
        raise ValueError("Missing FINAL_ANSWER line.")
    if reasoning_match["line_index"] >= final_match["line_index"]:
        raise ValueError("REASONING must appear before FINAL_ANSWER.")

    reasoning = _collapse_whitespace(reasoning_match["value"])
    if not reasoning:
        raise ValueError("REASONING must be non-empty.")

    final_answer = str(final_match["value"]).strip()
    if not final_answer:
        raise ValueError("FINAL_ANSWER must be non-empty.")

    payload: dict[str, Any] = {
        "final_answer": final_answer,
        "reasoning": reasoning,
    }
    majority_error_match = _extract_labeled_match(
        cleaned,
        ["MAJORITY_ERROR", "MAJORITY_ERR", "ERROR_CERTIFICATE"],
    )
    if majority_error_match is not None:
        payload["majority_error"] = str(majority_error_match["value"]).strip()

    format_warning = _task_format_warning(dataset, final_answer)
    if format_warning is not None:
        payload["format_warning"] = format_warning
    return payload


def task_format_ok(dataset: str, final_answer: str) -> bool:
    """Check whether a free-text answer appears to stay inside the task slot."""

    answer = str(final_answer or "").strip()
    if not answer:
        return False
    if dataset in _MULTIPLE_CHOICE_DATASETS:
        return re.fullmatch(r"[A-J]", answer) is not None
    if dataset in _MATH_DATASETS:
        return "\\" not in answer
    return True


def normalized_answer_for_task(dataset: str, final_answer: str) -> str:
    """Normalize a parsed final answer for scoring or comparison."""

    answer = str(final_answer or "").strip()
    return normalize_prediction(dataset, answer) if answer else ""


def _role_description(role: str) -> str:
    if role == "debate":
        return "You are one reasoning agent revising an answer after peer feedback."
    if role == "initial":
        return "You are one reasoning agent producing an initial answer."
    return "You are a reasoning assistant producing a final answer for a research experiment."


def _strip_json_contract(user_content: str) -> str:
    markers = [
        "Return exactly one JSON object",
        "Return exactly one JSON object like",
        "Do not emit markdown fences or any extra text.",
        "Do not output JSON.",
    ]
    cut_index = len(user_content)
    for marker in markers:
        idx = user_content.find(marker)
        if idx >= 0:
            cut_index = min(cut_index, idx)
    return user_content[:cut_index].rstrip()


def build_free_text_answer_instruction(dataset: str) -> str:
    return _answer_protocol_instruction(dataset)


def _answer_protocol_instruction(dataset: str) -> str:
    lines = [
        "Return only the following two lines, in this exact order, with no markdown fences:",
        "REASONING: <required concise reasoning>",
        "FINAL_ANSWER: <canonical answer>",
        "Rules:",
        "- REASONING is required.",
        "- Keep REASONING concise, but include enough detail to justify or revise the answer.",
        "- If your reasoning changes the answer, rewrite FINAL_ANSWER to the corrected answer.",
        "- Use plain text only in REASONING. Do not use LaTeX commands, backslashes, or markdown.",
        "- FINAL_ANSWER must contain only the final answer and nothing else.",
    ]
    lines.extend(_dataset_specific_rules(dataset))
    return "\n".join(lines)


def _dataset_specific_rules(dataset: str) -> list[str]:
    if dataset in _MULTIPLE_CHOICE_DATASETS:
        return ['- FINAL_ANSWER must be exactly one option letter such as "A" or "B".']
    if dataset in _MATH_DATASETS:
        return [
            "- FINAL_ANSWER must use plain ASCII math only.",
            "- Do not use LaTeX commands such as \\frac, \\sqrt, \\left, or \\right.",
        ]
    if dataset in _SHORT_SPAN_DATASETS:
        return [
            "- FINAL_ANSWER must be the shortest judgeable text span.",
            "- Do not add category words, explanations, or extra qualifiers.",
        ]
    return []


def _extract_labeled_value(text: str, labels: list[str]) -> str | None:
    match = _extract_labeled_match(text, labels)
    if match is None:
        return None
    return str(match["value"])


def _extract_labeled_match(text: str, labels: list[str]) -> dict[str, Any] | None:
    lines = str(text or "").splitlines()
    for line_index, line in enumerate(lines):
        for label in labels:
            pattern = rf"(?i)^\s*{re.escape(label)}\s*:\s*(.+?)\s*$"
            match = re.match(pattern, line)
            if match is None:
                continue
            value = _clean_extracted_value(match.group(1))
            if value:
                return {"label": label, "value": value, "line_index": line_index}
    return None


def _clean_extracted_value(value: str) -> str:
    cleaned = str(value or "").strip().strip("\"'`")
    return _collapse_whitespace(cleaned)


def _collapse_whitespace(value: str) -> str:
    return " ".join(str(value or "").split())


def _strip_code_fences(text: str) -> str:
    trimmed = text.strip()
    if not trimmed.startswith("```"):
        return trimmed
    trimmed = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", trimmed)
    trimmed = re.sub(r"\s*```$", "", trimmed)
    return trimmed.strip()


def _task_format_warning(dataset: str, final_answer: str) -> str | None:
    if task_format_ok(dataset, final_answer):
        return None
    if dataset in _MULTIPLE_CHOICE_DATASETS:
        return "multiple_choice_answer_not_single_letter"
    if dataset in _MATH_DATASETS:
        return "math_answer_contains_non_ascii_math_markup"
    return "answer_outside_expected_slot"
