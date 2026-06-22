"""自由文本推理 + 末尾小 JSON 答案块协议。"""

from __future__ import annotations

import json
import re
from typing import Any

from research_experiments.family_runtime.free_text_protocol import task_format_ok

JSON_TAIL_ANSWER_PROTOCOL_V1 = "json_tail_answer_v1"


def build_json_tail_answer_instruction(dataset: str, *, extra_json_keys: list[str] | None = None) -> str:
    """Return the shared CRED-MAD output contract."""

    keys = ["answer", "confidence", *(extra_json_keys or [])]
    lines = [
        "Write concise natural-language reasoning first.",
        "End with exactly one final JSON answer block in this format:",
        "[FINAL]",
        json.dumps({key: _example_value_for_key(key) for key in keys}, ensure_ascii=False),
        "[/FINAL]",
        "Rules:",
        "- The JSON block must be the final content in your response.",
        "- The JSON must be one small object only.",
        "- confidence must be a number from 0.0 to 1.0.",
        "- answer must contain only the canonical final answer.",
    ]
    if dataset in {"gpqa_diamond", "mmlu", "mmlu_abstract_algebra", "mmlu_pro"}:
        lines.append('- answer must be exactly one option letter such as "A" or "B".')
    elif dataset in {"gsm8k", "math500", "competition_math"}:
        lines.append("- answer must use plain ASCII math only; do not use LaTeX commands or backslashes.")
    elif dataset in {"hotpotqa", "webquestions"}:
        lines.append("- answer must be the shortest judgeable text span.")
    return "\n".join(lines)


def parse_json_tail_answer_output(raw_text: str, *, dataset: str) -> dict[str, Any]:
    """Parse a free-text response ending with [FINAL] JSON [/FINAL]."""

    cleaned = _strip_code_fences(str(raw_text or "").strip())
    if not cleaned:
        raise ValueError("Assistant output is empty.")

    match = re.search(r"\[FINAL\]\s*(\{.*?\})\s*\[/FINAL\]\s*$", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if match is None:
        raise ValueError("Missing final [FINAL] JSON block.")

    reasoning = cleaned[: match.start()].strip()
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ValueError("Final JSON block is invalid.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Final JSON block must contain one object.")

    answer = str(payload.get("answer") or payload.get("final_answer") or "").strip()
    if not answer:
        raise ValueError("Final JSON must include a non-empty answer.")
    confidence = _parse_confidence(payload.get("confidence"))
    if not reasoning:
        reasoning = str(payload.get("reasoning") or payload.get("rationale") or "").strip()

    parsed = dict(payload)
    parsed["answer"] = answer
    parsed["final_answer"] = answer
    parsed["confidence"] = confidence
    parsed["reasoning"] = reasoning
    if not task_format_ok(dataset, answer):
        parsed["format_warning"] = _task_format_warning(dataset)
    return parsed


def _example_value_for_key(key: str) -> object:
    if key == "answer":
        return "..."
    if key == "confidence":
        return 0.0
    if key == "changed":
        return False
    if key == "source":
        return "A|B|mixed"
    return "..."


def _parse_confidence(value: object) -> float:
    if value is None:
        raise ValueError("Final JSON must include confidence.")
    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be numeric.") from exc
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError("confidence must be between 0 and 1.")
    return round(confidence, 6)


def _strip_code_fences(text: str) -> str:
    trimmed = text.strip()
    if not trimmed.startswith("```"):
        return trimmed
    trimmed = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", trimmed)
    trimmed = re.sub(r"\s*```$", "", trimmed)
    return trimmed.strip()


def _task_format_warning(dataset: str) -> str:
    if dataset in {"gpqa_diamond", "mmlu", "mmlu_abstract_algebra", "mmlu_pro"}:
        return "multiple_choice_answer_not_single_letter"
    if dataset in {"gsm8k", "math500", "competition_math"}:
        return "math_answer_contains_non_ascii_math_markup"
    return "answer_outside_expected_slot"
