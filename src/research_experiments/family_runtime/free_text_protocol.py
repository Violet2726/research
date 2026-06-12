"""MiMo 兼容的自由文本标签行协议辅助工具。"""

from __future__ import annotations

import re
from typing import Any

from research_experiments.core.data.evaluation import normalize_prediction
from research_experiments.core.prompts.dataset_contracts import build_tagged_lines_system_prompt

FREE_TEXT_ANSWER_PROTOCOL_V1 = "free_text_answer_v1"
FREE_TEXT_DEBATE_UPDATE_PROTOCOL_V1 = "free_text_debate_update_v1"

_MULTIPLE_CHOICE_DATASETS = {
    "gpqa_diamond",
    "mmlu",
    "mmlu_abstract_algebra",
    "mmlu_pro",
}
_MATH_DATASETS = {"gsm8k", "math500", "competition_math"}
_SHORT_SPAN_DATASETS = {"hotpotqa", "webquestions"}


def build_free_text_system_prompt(role_description: str) -> str:
    """Build a system prompt for a short tagged-line answer protocol."""

    return build_tagged_lines_system_prompt(
        role_description,
        extra_rules=[
            "Return only the requested tagged lines.",
            "Do not add markdown fences or prose before or after the tagged lines.",
            "Keep the reason short and plain-text.",
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
        _debate_protocol_instruction(dataset)
        if role == "debate"
        else _answer_protocol_instruction(dataset, debate=False)
    )
    user_message["content"] = f"{base_user.rstrip()}\n\n{protocol_block}"
    adapted[-1] = user_message
    return adapted


def parse_free_text_answer_output(
    raw_text: str,
    *,
    dataset: str,
    require_decision: bool,
) -> dict[str, Any]:
    """Parse a tagged-line free-text answer into a compact structured payload."""

    cleaned = _strip_code_fences(str(raw_text or "").strip())
    if not cleaned:
        raise ValueError("Assistant output is empty.")

    final_answer = _extract_labeled_value(cleaned, ["FINAL_ANSWER", "ANSWER", "FINAL"])
    if not final_answer:
        raise ValueError("Missing FINAL_ANSWER line.")
    final_answer = final_answer.strip()
    if not final_answer:
        raise ValueError("FINAL_ANSWER must be non-empty.")

    reasoning = _extract_labeled_value(cleaned, ["REASON", "REASONING_BRIEF", "WHY", "RATIONALE"])
    if not reasoning:
        raise ValueError("Missing REASON line.")
    reasoning = _collapse_whitespace(reasoning)
    if not reasoning:
        raise ValueError("REASON must be non-empty.")

    decision = _extract_labeled_value(cleaned, ["DECISION", "CHANGED_ANSWER", "CHANGE"])
    normalized_decision = _normalize_decision(decision)
    if require_decision and normalized_decision is None:
        raise ValueError("Missing DECISION line.")

    payload: dict[str, Any] = {
        "final_answer": final_answer,
        "reasoning": reasoning,
    }
    if normalized_decision is not None:
        payload["decision"] = normalized_decision
        payload["changed_answer"] = normalized_decision == "revise"

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


def _answer_protocol_instruction(dataset: str, *, debate: bool) -> str:
    del debate
    lines = [
        "Return only the following two lines, in this exact order, with no markdown fences:",
        "FINAL_ANSWER: <answer only>",
        "REASON: <one short plain-text sentence>",
        "Rules:",
        "- FINAL_ANSWER must contain only the final answer and nothing else.",
        "- REASON must be one short sentence under 160 characters.",
        "- Use plain text only in REASON. Do not use LaTeX commands, backslashes, or markdown.",
    ]
    lines.extend(_dataset_specific_rules(dataset))
    return "\n".join(lines)


def _debate_protocol_instruction(dataset: str) -> str:
    lines = [
        "Return only the following three lines, in this exact order, with no markdown fences:",
        "DECISION: <keep or revise>",
        "FINAL_ANSWER: <answer only>",
        "REASON: <one short plain-text sentence>",
        "Rules:",
        "- Use DECISION: keep when peer feedback does not change your answer.",
        "- Use DECISION: revise only when peer feedback changes your final answer.",
        "- FINAL_ANSWER must contain only the final answer and nothing else.",
        "- REASON must be one short sentence under 160 characters.",
        "- Use plain text only in REASON. Do not use LaTeX commands, backslashes, or markdown.",
        "- Do not restate the full question or copy long peer passages.",
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
    for label in labels:
        escaped = re.escape(label).replace("\\ ", r"[\s_]+")
        pattern = rf"(?im)^\s*{escaped}\s*:\s*(.+?)\s*$"
        matches = list(re.finditer(pattern, text))
        for match in reversed(matches):
            value = _clean_extracted_value(match.group(1))
            if value:
                return value
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


def _normalize_decision(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = _collapse_whitespace(value).lower()
    if normalized in {"keep", "no", "unchanged", "same"}:
        return "keep"
    if normalized in {"revise", "yes", "changed", "change"}:
        return "revise"
    if normalized in {"initial", "n/a", "na"}:
        return "initial"
    return normalized or None


def _task_format_warning(dataset: str, final_answer: str) -> str | None:
    if task_format_ok(dataset, final_answer):
        return None
    if dataset in _MULTIPLE_CHOICE_DATASETS:
        return "multiple_choice_answer_not_single_letter"
    if dataset in _MATH_DATASETS:
        return "math_answer_contains_non_ascii_math_markup"
    return "answer_outside_expected_slot"
