"""辩论实验共用的严格 JSON 对象答案协议。"""

from __future__ import annotations

import json
import re
from typing import Any

from research_experiments.family_runtime.free_text_protocol import task_format_ok

JSON_OBJECT_ANSWER_PROTOCOL_V3 = "json_object_answer_v3"
_RISK_LEVELS = {"none", "low", "medium", "high"}


def build_json_object_answer_instruction(dataset: str, *, extra_json_keys: list[str] | None = None) -> str:
    """Return the shared JSON object answer contract."""

    keys = _dedupe_keys(
        ["reasoning", "answer", "confidence", "key_evidence", "risk_level", "risk_summary", *(extra_json_keys or [])]
    )
    example = {key: _example_value_for_key(key) for key in keys}
    lines = [
        "Return one compact JSON answer card with these fields:",
        json.dumps(example, ensure_ascii=False),
        "Field guide:",
        *(_field_guide_line(key) for key in keys),
        "- Use JSON strings for text fields, JSON booleans for true/false fields, and a JSON number for confidence.",
        "- reasoning is a compact three-part verification trace: slot; decisive check; answer fit.",
        "- key_evidence is one short concrete clue, equation, option contrast, or context span.",
        "- risk_summary is a short phrase naming the remaining uncertainty.",
    ]
    if dataset in {"gpqa_diamond", "mmlu", "mmlu_abstract_algebra", "mmlu_pro"}:
        lines.append('- For this dataset, answer is exactly one option letter such as "A" or "B".')
    elif dataset in {"gsm8k", "math500", "competition_math"}:
        lines.append("- For this dataset, answer uses plain ASCII math, such as sqrt(3), pi/6, x^2, [a,b], or (a,b).")
    elif dataset in {"hotpotqa", "webquestions"}:
        lines.append("- For this dataset, answer is the complete judgeable answer span.")
    return "\n".join(lines)


def parse_json_object_answer_output(raw_text: str, *, dataset: str) -> dict[str, Any]:
    """Parse a response whose complete answer contract is one JSON object."""

    cleaned = _strip_code_fences(str(raw_text or "").strip())
    if not cleaned:
        raise ValueError("Assistant output is empty.")
    payload = _decode_json_object(cleaned)
    answer = _require_text(payload.get("answer") or payload.get("final_answer"), "answer")
    reasoning = _require_text(payload.get("reasoning") or payload.get("rationale"), "reasoning")
    confidence = _parse_confidence(payload.get("confidence"))
    key_evidence = _require_text(payload.get("key_evidence") or payload.get("evidence"), "key_evidence")
    risk_level = _parse_risk_level(payload.get("risk_level"))

    parsed = dict(payload)
    parsed["answer"] = answer
    parsed["final_answer"] = answer
    parsed["reasoning"] = reasoning
    parsed["confidence"] = confidence
    parsed["key_evidence"] = key_evidence
    parsed["risk_level"] = risk_level
    if not task_format_ok(dataset, answer):
        parsed["format_warning"] = _task_format_warning(dataset)
    return parsed


def _decode_json_object(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as strict_exc:
        payload = _recover_first_json_object(text, strict_exc)
    if not isinstance(payload, dict):
        raise ValueError("Assistant output must be one JSON object.")
    return payload


def _recover_first_json_object(text: str, strict_exc: json.JSONDecodeError) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            payload, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    raise ValueError("Assistant output must be valid JSON.") from strict_exc


def _dedupe_keys(keys: list[str]) -> list[str]:
    return list(dict.fromkeys(key for key in keys if key))


def _field_guide_line(key: str) -> str:
    descriptions = {
        "reasoning": "three compact verification clauses that justify the answer.",
        "answer": "the canonical final answer only.",
        "confidence": "a number from 0.0 to 1.0.",
        "key_evidence": "one concrete plain-text calculation, option clue, or context span supporting answer.",
        "risk_level": 'one of "none", "low", "medium", "high"; medium/high means a concrete unresolved risk remains.',
        "risk_summary": "one short phrase explaining the risk_level.",
        "answer_type": "a short label for the answer form, such as expression, option, span, or yes_no.",
        "promote": "true when a challenger should replace the current leading answer.",
        "leader_pass": "true when the current leader passes the decisive verification check.",
        "challenger_pass": "true when the challenger passes the decisive verification check.",
        "verification_type": "a short label for the decisive task check, such as option_concept, calculation, span_match, or factual_mapping.",
        "leader_score": "a number from 0.0 to 1.0 for how well the current leader passes the decisive check.",
        "challenger_score": "a number from 0.0 to 1.0 for how well the challenger passes the decisive check.",
        "leader_failure": "one concrete failed check or missing support for the current leader.",
        "challenger_support": "one concrete passed check or support source for the challenger.",
        "verdict": "one compact sentence explaining why the selected answer survives.",
        "changed": "true when answer replaces the prior leading answer; false when the prior answer survives.",
        "attack_type": "a short label for the tested issue, such as contradiction, constraint_miss, slot_error, or calculation_error.",
        "attack_strength": 'one of "none", "weak", "medium", "high" for the strongest surviving attack.',
        "defense_status": "a short label for the result, such as defended, corrected, or unresolved.",
        "source": "the decisive support source: stage_a, refutation, defense, or mixed.",
    }
    return f"- {key}: {descriptions.get(key, 'a short task-specific value.')}"


def _example_value_for_key(key: str) -> object:
    if key == "reasoning":
        return "slot identified; decisive check made; answer fits"
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
    if key == "promote":
        return False
    if key in {"leader_pass", "challenger_pass"}:
        return False
    if key in {"leader_score", "challenger_score"}:
        return 0.0
    if key == "source":
        return "stage_a|refutation|defense|mixed"
    if key == "attack_strength":
        return "none|weak|medium|high"
    return "..."


def _require_text(value: object, field_name: str) -> str:
    if value is None or isinstance(value, bool):
        raise ValueError(f"JSON object must include a non-empty {field_name}.")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"JSON object must include a non-empty {field_name}.")
    return normalized


def _parse_confidence(value: object) -> float:
    if value is None:
        raise ValueError("JSON object must include confidence.")
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
