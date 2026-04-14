from __future__ import annotations

import json
import re
from typing import Any


def parse_model_output(raw_text: str) -> tuple[dict[str, Any], str]:
    cleaned = _strip_code_fences(raw_text.strip())
    try:
        return json.loads(cleaned), "direct_json"
    except json.JSONDecodeError:
        pass

    merged_objects = _decode_multiple_objects(cleaned)
    if merged_objects is not None:
        return merged_objects

    candidate = _extract_json_object(cleaned)
    if candidate is not None:
        try:
            return json.loads(candidate), "substring_json"
        except json.JSONDecodeError:
            repaired = _light_repair(candidate)
            if repaired is not None:
                try:
                    return json.loads(repaired), "repaired_json"
                except json.JSONDecodeError:
                    pass

    regex_payload = _extract_fields_via_regex(cleaned)
    if regex_payload is not None:
        return regex_payload, "regex_fields"

    tail_payload = _extract_tail_value(cleaned)
    if tail_payload is not None:
        return tail_payload, "tail_value"

    raise ValueError("Unable to parse model output into JSON.")


def _strip_code_fences(text: str) -> str:
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_json_object(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _light_repair(text: str) -> str | None:
    candidate = text.strip().replace("\r", "")
    while candidate.endswith("}") and candidate.count("{") < candidate.count("}"):
        candidate = candidate[:-1].rstrip()
    if candidate.count("{") != candidate.count("}"):
        return None
    return candidate


def _decode_multiple_objects(text: str) -> tuple[dict[str, Any], str] | None:
    decoder = json.JSONDecoder()
    index = 0
    objects: list[dict[str, Any]] = []

    while index < len(text):
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text):
            break
        if text[index] != "{":
            next_start = text.find("{", index)
            if next_start == -1:
                break
            index = next_start
        try:
            parsed, next_index = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            break
        if isinstance(parsed, dict):
            objects.append(parsed)
        index = next_index

    if not objects:
        return None

    merged: dict[str, Any] = {}
    for obj in objects:
        for key in ("reasoning", "final_answer"):
            if key in obj and obj[key] not in (None, ""):
                merged[key] = obj[key]

    if "final_answer" not in merged and len(objects) == 1:
        return None
    return merged, "multi_json_merge" if len(objects) > 1 else "raw_decode_json"


def _extract_fields_via_regex(text: str) -> dict[str, Any] | None:
    reasoning_match = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.S)
    final_match = re.search(r'"final_answer"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.S)
    if final_match is None:
        return None

    payload: dict[str, Any] = {}
    if reasoning_match is not None:
        payload["reasoning"] = json.loads(f'"{reasoning_match.group(1)}"')
    payload["final_answer"] = json.loads(f'"{final_match.group(1)}"')
    return payload


def _extract_tail_value(text: str) -> dict[str, Any] | None:
    if '"final_answer"' in text or '"reasoning"' not in text:
        return None
    match = re.search(r'"\s*:\s*"((?:[^"\\]|\\.)*)"\s*}\s*$', text, re.S)
    if match is None:
        return None
    final_answer = json.loads(f'"{match.group(1)}"')
    if final_answer.strip().lower() == "final_answer":
        return None
    payload: dict[str, Any] = {"final_answer": final_answer}
    reasoning_match = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.S)
    if reasoning_match is not None:
        payload["reasoning"] = json.loads(f'"{reasoning_match.group(1)}"')
    return payload
