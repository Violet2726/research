"""MiMo 兼容的自由文本标签行协议辅助工具。"""

from __future__ import annotations

import re
from typing import Any

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.core.data.evaluation import canonicalize_answer, normalize_prediction
from research_experiments.core.prompts.dataset_contracts import build_tagged_lines_system_prompt

FREE_TEXT_ANSWER_PROTOCOL_V1 = "free_text_answer_v1"

_MULTIPLE_CHOICE_DATASETS = {
    "gpqa_diamond",
    "mmlu",
    "mmlu_abstract_algebra",
    "mmlu_pro",
}
_MATH_DATASETS = {
    "gsm8k",
    "math500",
    "competition_math",
    "omni_math",
}
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

    strict_error: ValueError | None = None
    try:
        return _parse_strict_free_text_answer(cleaned, dataset=dataset)
    except ValueError as exc:
        strict_error = exc

    fallback = _recover_embedded_final_answer(cleaned, dataset=dataset)
    if fallback is not None:
        return fallback

    fallback = _recover_mc_tail_answer_phrase(cleaned, dataset=dataset)
    if fallback is not None:
        return fallback

    raise strict_error


def parse_sample_answer_output(
    sample: DatasetSample,
    raw_text: str,
) -> dict[str, Any]:
    """Parse every explicit final marker and canonicalize them as one answer.

    Some OpenAI-compatible transports have returned a complete tagged answer,
    a ``</think>`` separator, and a second complete answer in the same text.
    Selecting the first marker can therefore absorb the second reasoning trace
    into the answer.  This parser accepts duplication only when every explicit
    answer is valid for this sample and resolves to the same canonical class.
    """

    cleaned = _strip_code_fences(str(raw_text or "").strip())
    if not cleaned:
        return _invalid_sample_answer_output("empty_assistant_output")
    raw_answers = _explicit_final_answers(cleaned)
    if not raw_answers:
        try:
            parsed = parse_free_text_answer_output(cleaned, dataset=sample.dataset)
        except ValueError:
            return _invalid_sample_answer_output("missing_final_answer")
        raw_answers = [str(parsed.get("final_answer") or "").strip()]
        reasoning = str(parsed.get("reasoning") or "").strip()
    else:
        reasoning_matches = [
            match.group(1).strip()
            for match in re.finditer(
                r"(?im)^\s*(?:REASONING|REASONNING|REASON|RATIONALE|WHY)\s*:\s*(.*?)\s*$",
                cleaned,
            )
            if match.group(1).strip()
        ]
        reasoning = _collapse_whitespace(reasoning_matches[-1]) if reasoning_matches else _fallback_reasoning(cleaned)

    canonical = [canonicalize_answer(sample, answer) for answer in raw_answers]
    invalid = next((item.invalid_reason for item in canonical if not item.valid), None)
    keys = {item.key for item in canonical if item.valid}
    if invalid is not None or len(keys) != 1:
        reason = invalid or "conflicting_duplicate_final_answer"
        return {
            "final_answer": "",
            "reasoning": reasoning,
            "raw_final_answers": raw_answers,
            "canonical_answer_key": "",
            "canonicalization_valid": False,
            "canonicalization_invalid_reason": reason,
            "canonical_key": "",
            "canonical_valid": False,
            "canonical_invalid_reason": reason,
        }
    key = next(iter(keys))
    return {
        "final_answer": raw_answers[-1],
        "reasoning": reasoning,
        "raw_final_answers": raw_answers,
        "canonical_answer_key": key,
        "canonicalization_valid": True,
        "canonicalization_invalid_reason": None,
        "canonical_key": key,
        "canonical_valid": True,
        "canonical_invalid_reason": None,
    }


def _invalid_sample_answer_output(reason: str) -> dict[str, Any]:
    return {
        "final_answer": "",
        "reasoning": "",
        "raw_final_answers": [],
        "canonical_answer_key": "",
        "canonicalization_valid": False,
        "canonicalization_invalid_reason": reason,
        "canonical_key": "",
        "canonical_valid": False,
        "canonical_invalid_reason": reason,
    }


def _explicit_final_answers(cleaned: str) -> list[str]:
    values: list[str] = []
    for match in re.finditer(r"(?im)^\s*FINAL_ANSWER\s*:\s*(.*?)\s*$", cleaned):
        value = re.split(
            r"(?i)</?think>|\b(?:REASONING|REASONNING|REASON|RATIONALE|WHY|FINAL_ANSWER)\s*:",
            match.group(1),
            maxsplit=1,
        )[0]
        value = _clean_extracted_value(value)
        if value:
            values.append(value)
    return values


def _parse_strict_free_text_answer(cleaned: str, *, dataset: str) -> dict[str, Any]:
    reasoning_match = _extract_labeled_match(cleaned, ["REASONING", "REASONNING", "REASON", "RATIONALE", "WHY"])
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


def _recover_embedded_final_answer(cleaned: str, *, dataset: str) -> dict[str, Any] | None:
    markers = [
        marker
        for marker in re.finditer(r"(?i)\bFINAL_ANSWER\s*:\s*", cleaned)
        if not _marker_starts_labeled_line(cleaned, marker.start())
    ]
    if not markers:
        return None
    marker = markers[-1]
    answer_line = cleaned[marker.end() :].splitlines()[0] if cleaned[marker.end() :] else ""
    final_answer = _clean_extracted_value(answer_line)
    if not final_answer:
        return None
    reasoning = _fallback_reasoning(cleaned[: marker.start()])
    return _fallback_payload(
        dataset=dataset,
        final_answer=final_answer,
        reasoning=reasoning,
        recovery="embedded_final_answer",
    )


def _recover_mc_tail_answer_phrase(cleaned: str, *, dataset: str) -> dict[str, Any] | None:
    if dataset not in _MULTIPLE_CHOICE_DATASETS:
        return None
    match = re.search(r"(?is)\b((?:final\s+answer)|answer)\s+is\s+([A-J])\b\s*[.。!！]?\s*$", cleaned)
    if match is None:
        return None
    final_answer = match.group(2).upper()
    reasoning = _fallback_reasoning(cleaned[: match.start()])
    return _fallback_payload(
        dataset=dataset,
        final_answer=final_answer,
        reasoning=reasoning,
        recovery="mc_tail_answer_phrase",
    )


def _fallback_payload(*, dataset: str, final_answer: str, reasoning: str, recovery: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "final_answer": final_answer,
        "reasoning": reasoning,
        "protocol_recovery": recovery,
    }
    format_warning = _task_format_warning(dataset, final_answer)
    if format_warning is not None:
        payload["format_warning"] = format_warning
    return payload


def _fallback_reasoning(prefix: str) -> str:
    reasoning = _collapse_whitespace(prefix)
    if not reasoning or reasoning in {"</think>", "<think>"}:
        return "answer recovered from explicit final answer marker"
    return reasoning


def _marker_starts_labeled_line(text: str, start_index: int) -> bool:
    line_start = str(text or "").rfind("\n", 0, start_index) + 1
    return not str(text or "")[line_start:start_index].strip()


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
    leak_warning = _reasoning_leak_warning(final_answer)
    if leak_warning is not None:
        return leak_warning
    if task_format_ok(dataset, final_answer):
        return None
    if dataset in _MULTIPLE_CHOICE_DATASETS:
        return "multiple_choice_answer_not_single_letter"
    if dataset in _MATH_DATASETS:
        return "math_answer_contains_non_ascii_math_markup"
    return "answer_outside_expected_slot"


def _reasoning_leak_warning(final_answer: str) -> str | None:
    answer = str(final_answer or "").strip()
    lowered = answer.lower()
    if any(marker in lowered for marker in ("</think>", "<think>", "reasoning:", "final_answer:", "final answer:", "therefore ", "because ")):
        return "answer_contains_reasoning_leak"
    if "\n" in answer or "\r" in answer:
        return "answer_contains_reasoning_leak"
    if len(answer) > 160:
        return "answer_too_long_for_final_slot"
    return None
