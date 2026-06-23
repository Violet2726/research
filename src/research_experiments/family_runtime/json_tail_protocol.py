"""自由文本推理 + 末尾单个 JSON 对象协议。"""

from __future__ import annotations

import json
import re
from typing import Any

from research_experiments.family_runtime.free_text_protocol import task_format_ok

JSON_OBJECT_TAIL_PROTOCOL_V2 = "json_object_tail_v2"
_RISK_LEVELS = {"none", "low", "medium", "high"}


def build_json_tail_answer_instruction(dataset: str, *, extra_json_keys: list[str] | None = None) -> str:
    """Return the shared CRED-MAD JSON-object-tail contract."""

    keys = _dedupe_keys(["answer", "confidence", "key_evidence", "risk_level", "risk_summary", *(extra_json_keys or [])])
    example = {key: _example_value_for_key(key) for key in keys}
    lines = [
        "Write concise natural-language reasoning first, then write this JSON object as the final content:",
        json.dumps(example, ensure_ascii=False),
        "Field guide:",
        *(_field_guide_line(key) for key in keys),
        "- Use valid JSON with double-quoted keys and strings.",
        "- Write JSON string values in plain text; use ASCII math forms such as sqrt(3), pi/6, and x^2.",
    ]
    if dataset in {"gpqa_diamond", "mmlu", "mmlu_abstract_algebra", "mmlu_pro"}:
        lines.append('- For this dataset, answer is exactly one option letter such as "A" or "B".')
    elif dataset in {"gsm8k", "math500", "competition_math"}:
        lines.append("- For this dataset, answer uses plain ASCII math, such as sqrt(3), pi/6, x^2, [a,b], or (a,b).")
    elif dataset in {"hotpotqa", "webquestions"}:
        lines.append("- For this dataset, answer is the shortest judgeable text span.")
    return "\n".join(lines)


def _dedupe_keys(keys: list[str]) -> list[str]:
    return list(dict.fromkeys(key for key in keys if key))


def _field_guide_line(key: str) -> str:
    descriptions = {
        "answer": "the canonical final answer only.",
        "confidence": "a number from 0.0 to 1.0.",
        "key_evidence": "one concrete plain-text calculation, option clue, or context span supporting answer.",
        "risk_level": 'one of "none", "low", "medium", "high"; use medium/high only for a concrete unresolved risk.',
        "risk_summary": "one short phrase explaining the risk_level.",
        "answer_type": "a short label for the answer form, such as expression, option, span, or yes_no.",
        "changed": "true when answer replaces the prior leading answer; false when the prior answer survives.",
        "attack_type": "a short label for the tested issue, such as contradiction, constraint_miss, slot_error, or calculation_error.",
        "attack_strength": 'one of "none", "weak", "medium", "high" for the strongest surviving attack.',
        "defense_status": "a short label for the result, such as defended, corrected, or unresolved.",
        "source": "the decisive support source: stage_a, refutation, defense, or mixed.",
    }
    return f"- {key}: {descriptions.get(key, 'a short task-specific value.')}"


def parse_json_object_tail_answer_output(raw_text: str, *, dataset: str) -> dict[str, Any]:
    """Parse a response whose final content is one JSON object."""

    cleaned = _strip_code_fences(str(raw_text or "").strip())
    if not cleaned:
        raise ValueError("Assistant output is empty.")

    reasoning, payload = _extract_tail_json_object(cleaned)
    answer = str(payload.get("answer") or payload.get("final_answer") or "").strip()
    if not answer:
        raise ValueError("Final JSON object must include a non-empty answer.")
    confidence = _parse_confidence(payload.get("confidence"))
    risk_level = _parse_risk_level(payload.get("risk_level"))
    if not str(payload.get("key_evidence") or payload.get("evidence") or "").strip():
        raise ValueError("Final JSON object must include key_evidence.")
    if not reasoning:
        reasoning = str(payload.get("reasoning") or payload.get("rationale") or "").strip()

    parsed = dict(payload)
    parsed["answer"] = answer
    parsed["final_answer"] = answer
    parsed["confidence"] = confidence
    parsed["risk_level"] = risk_level
    parsed["reasoning"] = reasoning
    if not task_format_ok(dataset, answer):
        parsed["format_warning"] = _task_format_warning(dataset)
    return parsed


def _extract_tail_json_object(text: str) -> tuple[str, dict[str, Any]]:
    decoder = json.JSONDecoder()
    saw_tail_candidate = False
    for match in reversed(list(re.finditer(r"\{", text))):
        start = match.start()
        candidate = text[start:]
        try:
            payload, end = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            saw_tail_candidate = True
            continue
        if candidate[end:].strip():
            continue
        if not isinstance(payload, dict):
            raise ValueError("Final JSON content must be one object.")
        return text[:start].strip(), payload
    if saw_tail_candidate or text.endswith("}"):
        raise ValueError("Final JSON object is invalid.")
    raise ValueError("Missing final JSON object.")


def _example_value_for_key(key: str) -> object:
    if key == "answer":
        return "..."
    if key == "confidence":
        return 0.0
    if key == "risk_level":
        return "none"
    if key == "risk_summary":
        return "short reason for the risk level"
    if key == "changed":
        return False
    if key == "source":
        return "stage_a|refutation|defense|mixed"
    if key == "attack_strength":
        return "none|weak|medium|high"
    return "..."


def _parse_confidence(value: object) -> float:
    if value is None:
        raise ValueError("Final JSON object must include confidence.")
    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be numeric.") from exc
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError("confidence must be between 0 and 1.")
    return round(confidence, 6)


def _parse_risk_level(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in _RISK_LEVELS:
        raise ValueError('risk_level must be one of "none", "low", "medium", or "high".')
    return normalized


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
