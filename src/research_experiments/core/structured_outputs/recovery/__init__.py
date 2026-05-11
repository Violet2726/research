"""按语义 schema 拆分的轻量恢复逻辑。"""

from __future__ import annotations

from typing import Any


def structured_output_candidates(
    *,
    assistant_text: str,
    provider_reasoning_text: str | None,
    schema_id: str,
) -> list[tuple[str, str]]:
    assistant = str(assistant_text or "").strip()
    reasoning = str(provider_reasoning_text or "").strip()
    if schema_id == "answer_with_proxy_signals.selective" and _looks_like_placeholder_visible_output(assistant):
        ordered = [(reasoning, "reasoning_fallback"), (assistant, "assistant_text")]
    else:
        ordered = [(assistant, "assistant_text"), (reasoning, "reasoning_fallback")]
    return [(text, source) for text, source in ordered if text]


def normalize_selective_json_payload(payload: dict[str, Any]) -> dict[str, Any]:
    key_aliases = {
        "final_answer": "final_answer",
        "answer": "final_answer",
        "final": "final_answer",
        "reasoning": "reasoning",
        "reason": "reasoning",
        "rationale": "reasoning",
        "why": "reasoning",
        "confidence_raw": "confidence_raw",
        "confidence": "confidence_raw",
        "conf": "confidence_raw",
        "claim_span": "claim_span",
        "claim": "claim_span",
        "span": "claim_span",
        "uncertainty_type": "uncertainty_type",
        "uncertainty": "uncertainty_type",
        "type": "uncertainty_type",
        "key_evidence": "key_evidence",
        "evidence": "key_evidence",
        "uncertain_point": "uncertain_point",
        "uncertainty_point": "uncertain_point",
    }
    normalized_payload: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(key, str):
            candidate = key.strip().lower().replace("-", "_").replace(" ", "_")
            mapped_key = key_aliases.get(candidate)
            if mapped_key is not None:
                if mapped_key not in normalized_payload or candidate == mapped_key:
                    normalized_payload[mapped_key] = value
                continue
        normalized_payload[key] = value
    return normalized_payload


def parse_proxy_signal_answer_payload(raw_text: str, dataset: str) -> dict[str, Any]:
    cleaned = raw_text.strip()
    if not cleaned:
        raise ValueError("Assistant output is empty.")
    if cleaned.startswith("{"):
        try:
            return normalize_selective_json_payload(_decode_json_object(cleaned))
        except Exception:
            pass
    return _coerce_selective_text_output(cleaned, dataset)


def recover_answer_from_reasoning_text(reasoning_text: str, dataset: str) -> dict[str, Any]:
    import re

    cleaned = " ".join(reasoning_text.split())
    if not cleaned:
        raise ValueError("Provider reasoning text is empty.")
    answer_phrase = _extract_answer_phrase(cleaned)
    final_answer = ""
    if dataset in {"gsm8k", "gsm_symbolic", "math500"}:
        if answer_phrase:
            match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", answer_phrase.replace(",", ""))
            if match:
                final_answer = match.group(0).replace(",", "")
        if not final_answer:
            raise ValueError("Could not recover numeric answer from reasoning text.")
    elif dataset == "strategyqa":
        if answer_phrase:
            lowered = answer_phrase.lower()
            if lowered.startswith("yes"):
                final_answer = "yes"
            elif lowered.startswith("no"):
                final_answer = "no"
        if not final_answer:
            matches = re.findall(r"\b(?:yes|no)\b", cleaned.lower())
            if matches:
                final_answer = matches[-1]
        if not final_answer:
            raise ValueError("Could not recover yes/no answer from reasoning text.")
    elif dataset == "hotpotqa":
        if answer_phrase:
            final_answer = answer_phrase.strip().strip("\"'")
        if not final_answer:
            raise ValueError("Could not recover text answer from reasoning text.")
    else:
        raise ValueError(f"Unsupported dataset for reasoning-text recovery: {dataset}")
    return {
        "final_answer": final_answer,
        "confidence_raw": None,
        "reasoning": _clip_selective_field(cleaned, 180),
        "claim_span": _clip_selective_field(final_answer, 160),
        "uncertainty_type": _default_uncertainty_type(dataset),
        "key_evidence": _clip_selective_field(final_answer, 160),
        "uncertain_point": None,
    }


def recover_partial_payload(raw_text: str, schema_id: str) -> dict[str, Any] | None:
    if schema_id == "answer_core":
        return _recover_answer_core_payload(raw_text)
    if schema_id == "answer_with_proxy_signals.budget":
        return _recover_budget_proxy_signal_payload(raw_text)
    if schema_id == "answer_with_proxy_signals.deliberation":
        return _recover_deliberation_proxy_signal_payload(raw_text)
    if schema_id == "deliberation_packet":
        return _recover_deliberation_packet_payload(raw_text)
    if schema_id == "belief_update_delta":
        return _recover_belief_update_delta_payload(raw_text)
    if schema_id == "audit_verdict":
        return _recover_audit_verdict_payload(raw_text)
    if schema_id in {"split_context_solver", "split_context_belief"}:
        return _recover_split_context_payload(raw_text, schema_id=schema_id)
    if schema_id == "cue_blackbox_packet":
        return _recover_cue_blackbox_payload(raw_text)
    return None


def recover_soft_rejection_payload(schema_id: str, *, dataset: str | None) -> dict[str, Any] | None:
    if schema_id == "answer_core":
        return {"final_answer": "unknown", "reasoning": "provider_soft_rejection"}
    if schema_id == "answer_with_proxy_signals.selective":
        return {
            "final_answer": "unknown",
            "confidence_raw": None,
            "reasoning": "provider_soft_rejection",
            "claim_span": "provider_soft_rejection",
            "uncertainty_type": _default_uncertainty_type(dataset),
            "key_evidence": "provider_soft_rejection",
            "uncertain_point": None,
        }
    if schema_id == "answer_with_proxy_signals.budget":
        return {
            "final_answer": "unknown",
            "reasoning_trace": "provider_soft_rejection",
            "claim_span": "provider_soft_rejection",
            "key_evidence": "provider_soft_rejection",
            "keyword_clues": ["provider_soft_rejection"],
            "confidence_raw": 0.0,
            "uncertain_point": None,
        }
    if schema_id == "answer_with_proxy_signals.deliberation":
        return {
            "final_answer": "unknown",
            "reasoning_trace": "provider_soft_rejection",
            "claim_span": "provider_soft_rejection",
            "confidence_raw": 0.0,
            "uncertain_point": None,
            "key_evidence": "provider_soft_rejection",
        }
    if schema_id == "deliberation_packet":
        return {
            "final_answer": "unknown",
            "reasoning_trace": "provider_soft_rejection",
            "confidence_raw": 0.0,
            "claim_span": "provider_soft_rejection",
            "key_evidence": "provider_soft_rejection",
        }
    if schema_id == "belief_update_delta":
        return {
            "changed_answer": False,
            "new_answer": None,
            "confidence_delta": None,
            "reason_for_change": "provider_soft_rejection",
            "remaining_disagreement": None,
        }
    if schema_id == "audit_verdict":
        return {
            "decision": "abstain",
            "verified_answer": None,
            "rationale": "provider_soft_rejection",
        }
    if schema_id == "cue_blackbox_packet":
        return {
            "final_answer": "unknown",
            "confidence": 0.0,
            "reasoning_sketch": "provider_soft_rejection",
            "uncertain_point": None,
            "top_claims": ["provider_soft_rejection"],
            "evidence_items": [],
            "counter_answer": None,
        }
    if schema_id == "split_context_solver":
        return {
            "final_answer": "unknown",
            "reasoning_trace": "provider_soft_rejection",
            "evidence_summary": "provider_soft_rejection",
            "supporting_facts": [],
            "confidence_raw": 0.0,
        }
    if schema_id == "split_context_belief":
        return {
            "changed_answer": False,
            "final_answer": "unknown",
            "reasoning_trace": "provider_soft_rejection",
            "evidence_summary": "provider_soft_rejection",
            "supporting_facts": [],
            "confidence_raw": 0.0,
        }
    return None


def looks_like_soft_rejection_text(text: str) -> bool:
    normalized = " ".join(str(text or "").split()).lower()
    if not normalized:
        return False
    return normalized in {
        "content omitted due to provider safety policy",
        "provider_refused_due_to_policy",
        "the request was rejected because it was considered high risk",
        "request rejected because it was considered high risk",
    }


def _recover_answer_core_payload(raw_text: str) -> dict[str, Any] | None:
    final_answer = _extract_json_answer_field(raw_text, "final_answer")
    reasoning = _extract_json_string_field(raw_text, "reasoning")
    if not final_answer:
        final_answer = _extract_answer_phrase(raw_text)
    if not final_answer:
        final_answer = _extract_answer_guess_from_text(raw_text)
    if not final_answer:
        return None
    payload: dict[str, Any] = {"final_answer": final_answer}
    if reasoning:
        payload["reasoning"] = reasoning
    return payload


def _recover_budget_proxy_signal_payload(raw_text: str) -> dict[str, Any] | None:
    final_answer = _extract_json_answer_field(raw_text, "final_answer")
    if final_answer is None and '"final_answer":""' in raw_text.replace(" ", ""):
        final_answer = "unknown"
    if not final_answer:
        return None
    claim_span = _extract_json_string_field(raw_text, "claim_span")
    keyword_clues = _extract_json_string_list_field(raw_text, "keyword_clues")
    if not keyword_clues:
        keyword_clues = [claim_span or final_answer]
    confidence_raw = _extract_json_number_field(raw_text, "confidence_raw")
    return {
        "final_answer": final_answer,
        "reasoning_trace": _extract_json_string_field(raw_text, "reasoning_trace"),
        "claim_span": claim_span,
        "key_evidence": _extract_json_string_field(raw_text, "key_evidence"),
        "keyword_clues": keyword_clues,
        "confidence_raw": confidence_raw if confidence_raw is not None else 0.5,
        "uncertain_point": _extract_json_string_field(raw_text, "uncertain_point"),
    }


def _recover_split_context_payload(raw_text: str, *, schema_id: str) -> dict[str, Any] | None:
    final_answer = _extract_json_answer_field(raw_text, "final_answer") or _extract_json_answer_field(raw_text, "new_answer")
    reasoning_trace = _extract_json_string_field(raw_text, "reasoning_trace") or _extract_json_string_field(raw_text, "reasoning")
    evidence_summary = _extract_json_string_field(raw_text, "evidence_summary") or _extract_json_string_field(raw_text, "key_evidence")
    supporting_facts = _extract_json_array_field(raw_text, "supporting_facts") or []
    confidence_raw = _extract_json_number_field(raw_text, "confidence_raw")
    if confidence_raw is None:
        confidence_raw = _extract_json_string_field(raw_text, "confidence_raw")
    changed_answer = _extract_json_bool_field(raw_text, "changed_answer")
    has_recoverable_content = bool(final_answer or reasoning_trace or evidence_summary or supporting_facts or confidence_raw is not None or changed_answer is not None)
    if not has_recoverable_content:
        return None
    payload: dict[str, Any] = {
        "final_answer": final_answer or "",
        "reasoning_trace": reasoning_trace,
        "evidence_summary": evidence_summary,
        "supporting_facts": supporting_facts,
        "confidence_raw": confidence_raw,
    }
    if schema_id == "split_context_belief":
        payload["changed_answer"] = bool(changed_answer) if final_answer else False
    return payload


def _recover_belief_update_delta_payload(raw_text: str) -> dict[str, Any] | None:
    changed_answer = _extract_json_bool_field(raw_text, "changed_answer")
    if changed_answer is None:
        return None
    new_answer = _extract_json_answer_field(raw_text, "new_answer")
    if changed_answer and new_answer is None:
        return None
    return {
        "changed_answer": changed_answer,
        "new_answer": new_answer,
        "confidence_delta": _extract_json_number_field(raw_text, "confidence_delta"),
        "reason_for_change": _extract_json_string_field(raw_text, "reason_for_change"),
        "remaining_disagreement": _extract_json_string_field(raw_text, "remaining_disagreement"),
    }


def _recover_deliberation_proxy_signal_payload(raw_text: str) -> dict[str, Any] | None:
    final_answer = _extract_json_answer_field(raw_text, "final_answer")
    if not final_answer:
        return None
    return {
        "final_answer": final_answer,
        "reasoning_trace": _extract_json_string_field(raw_text, "reasoning_trace")
        or _extract_json_string_field(raw_text, "reasoning_sketch")
        or _extract_json_string_field(raw_text, "reasoning"),
        "claim_span": _extract_json_string_field(raw_text, "claim_span"),
        "confidence_raw": _extract_json_number_field(raw_text, "confidence_raw")
        or _extract_json_string_field(raw_text, "confidence_raw"),
        "uncertain_point": _extract_json_string_field(raw_text, "uncertain_point"),
        "key_evidence": _extract_json_string_field(raw_text, "key_evidence"),
    }


def _recover_deliberation_packet_payload(raw_text: str) -> dict[str, Any] | None:
    final_answer = _extract_json_answer_field(raw_text, "final_answer")
    if not final_answer:
        return None
    return {
        "final_answer": final_answer,
        "reasoning_trace": _extract_json_string_field(raw_text, "reasoning_trace")
        or _extract_json_string_field(raw_text, "reasoning"),
        "confidence_raw": _extract_json_number_field(raw_text, "confidence_raw")
        or _extract_json_string_field(raw_text, "confidence_raw"),
        "claim_span": _extract_json_string_field(raw_text, "claim_span"),
        "key_evidence": _extract_json_string_field(raw_text, "key_evidence"),
    }


def _recover_audit_verdict_payload(raw_text: str) -> dict[str, Any] | None:
    decision = _extract_json_string_field(raw_text, "decision")
    if not decision:
        return None
    return {
        "decision": decision,
        "verified_answer": _extract_json_answer_field(raw_text, "verified_answer"),
        "rationale": _extract_json_string_field(raw_text, "rationale") or "Recovered partial rationale.",
    }


def _recover_cue_blackbox_payload(raw_text: str) -> dict[str, Any] | None:
    final_answer = _extract_json_answer_field(raw_text, "final_answer")
    if not final_answer:
        return None
    confidence = _extract_json_number_field(raw_text, "confidence")
    if confidence is None:
        confidence = _extract_json_string_field(raw_text, "confidence")
    return {
        "final_answer": final_answer,
        "confidence": confidence,
        "reasoning_sketch": _extract_json_string_field(raw_text, "reasoning_sketch") or final_answer,
        "uncertain_point": _extract_json_string_field(raw_text, "uncertain_point"),
        "top_claims": _extract_json_string_list_field(raw_text, "top_claims"),
        "evidence_items": _extract_json_string_list_field(raw_text, "evidence_items"),
        "counter_answer": _extract_json_answer_field(raw_text, "counter_answer"),
    }


def _coerce_selective_text_output(cleaned: str, dataset: str) -> dict[str, Any]:
    final_answer = _extract_selective_final_answer(cleaned, dataset)
    if final_answer is None:
        raise ValueError("Proxy-signal output must contain a recoverable final answer.")
    claim_span = _extract_labeled_value(
        cleaned,
        ["CLAIM_SPAN", "CLAIM SPAN", "CLAIMED_SPAN", "CLAIMS SPAN", "SUPPORTING SPAN", "CLAIM", "SPAN"],
    )
    confidence_raw = _extract_labeled_value(cleaned, ["CONFIDENCE", "CONF"])
    uncertainty_type = _extract_labeled_value(
        cleaned,
        ["UNCERTAINTY_TYPE", "UNCERTAINTY TYPE", "UNCERTAINTY", "TYPE"],
    )
    reasoning = _extract_labeled_value(cleaned, ["REASON", "KEY REASON", "RATIONALE", "WHY"])
    normalized_uncertainty = (
        _normalize_uncertainty_type_candidate(uncertainty_type)
        if uncertainty_type is not None
        else _default_uncertainty_type(dataset)
    )
    normalized_claim = _clip_selective_field(claim_span or _infer_claim_span(cleaned, final_answer), 160)
    normalized_reasoning = _clip_selective_field(reasoning or _infer_reasoning(cleaned, final_answer), 180)
    normalized_answer = _clip_selective_field(final_answer, 160)
    return {
        "final_answer": normalized_answer,
        "confidence_raw": _optional_confidence_raw(confidence_raw),
        "reasoning": normalized_reasoning,
        "claim_span": normalized_claim,
        "uncertainty_type": normalized_uncertainty,
        "key_evidence": normalized_claim,
        "uncertain_point": None,
    }


def _decode_json_object(cleaned: str) -> dict[str, Any]:
    import json

    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("Assistant output must be a JSON object.")
    return payload


def _looks_like_placeholder_visible_output(text: str) -> bool:
    import re

    trimmed = text.strip()
    if not trimmed:
        return False
    return re.fullmatch(r"\[\s*[\w\s,.\-]*\s*\]", trimmed, re.S) is not None


def _extract_answer_phrase(text: str) -> str | None:
    import re

    patterns = [
        r"(?i)\b(?:therefore|thus|so)?\s*,?\s*(?:the\s+)?(?:final\s+)?answer\s+(?:should\s+be|is)\s*[:\-]?\s*([^\n.]+)",
        r"(?i)\b(?:the\s+)?correct\s+answer\s+(?:should\s+be|is)\s*[:\-]?\s*([^\n.]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        candidate = _clean_extracted_value(match.group(1))
        if candidate and not _looks_like_placeholder_text(candidate):
            return candidate
    return None


def _optional_confidence_raw(value: object) -> float | str | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return None
        if normalized.lower() in {"na", "n/a", "none", "null"}:
            return None
    return _require_confidence_raw(value)


def _require_confidence_raw(value: object) -> float | str:
    if isinstance(value, bool) or value is None:
        raise ValueError("confidence_raw must be a numeric value or numeric string.")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise ValueError("confidence_raw must be non-empty when provided as a string.")
        try:
            float(normalized.rstrip("%"))
        except ValueError as exc:
            raise ValueError("confidence_raw string must be numeric.") from exc
        return normalized
    raise ValueError("confidence_raw must be a numeric value or numeric string.")


def _extract_labeled_value(text: str, labels: list[str]) -> str | None:
    import re

    for label in labels:
        escaped = re.escape(label).replace("\\ ", r"[\s_]+")
        pattern = rf"(?im)^[\s`>*\-\{{\[\(]*[\"']?\s*{escaped}[\"']?\s*(?:is|:)\s*[\"']?(.+?)[\"']?\s*[,}}\]\)]?\s*$"
        matches = list(re.finditer(pattern, text))
        for match in reversed(matches):
            value = _clean_extracted_value(match.group(1))
            if value and not _looks_like_label_placeholder(value):
                return value
    return None


def _extract_selective_final_answer(text: str, dataset: str) -> str | None:
    import re

    labeled = _extract_labeled_value(text, ["FINAL_ANSWER", "ANSWER", "FINAL"])
    if labeled:
        return labeled
    if dataset in {"gsm8k", "gsm_symbolic", "math500"}:
        match = re.search(r"(?i)(?:final answer\s*(?:is|[:：])|answer is)\s*([^\n.]+)", text)
        if match:
            return match.group(1).strip()
        cutoff_text = re.split(r"(?i)\b(?:the output must have|return only the following|final_answer\s*:)\b", text, maxsplit=1)[0]
        equals_matches = re.findall(r"=\s*([-+]?\d[\d,]*(?:\.\d+)?)", cutoff_text.replace(",", ""))
        if equals_matches:
            return equals_matches[-1]
        numeric_matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", cutoff_text.replace(",", ""))
        if numeric_matches:
            return numeric_matches[-1]
        return None
    if dataset == "strategyqa":
        match = re.search(r"\b(yes|no)\b", text.strip().lower())
        return match.group(1) if match else None
    if dataset == "hotpotqa":
        match = re.search(r"(?i)(?:final answer\s*(?:is|[:：])|answer is)\s*([^\n.]+)", text)
        if match:
            return match.group(1).strip().strip("\"'")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            return lines[-1].strip().strip("\"'")
        return None
    return None


def _infer_claim_span(text: str, final_answer: str) -> str:
    return final_answer.strip() if final_answer else ""


def _infer_reasoning(text: str, final_answer: str) -> str:
    trimmed = " ".join(text.split())
    if not trimmed:
        return final_answer
    if len(trimmed) <= 160:
        return trimmed
    return trimmed[:157] + "..."


def _default_uncertainty_type(dataset: str | None) -> str:
    if dataset in {"gsm8k", "gsm_symbolic", "math500"}:
        return "calculation"
    if dataset == "hotpotqa":
        return "multi_hop"
    if dataset == "strategyqa":
        return "commonsense_gap"
    return "other"


def _normalize_uncertainty_type_candidate(value: object) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return normalized if normalized in {
        "none",
        "calculation",
        "evidence_selection",
        "entity_linking",
        "multi_hop",
        "commonsense_gap",
        "format_extraction",
        "other",
    } else "other"


def _clip_selective_field(value: str, max_chars: int) -> str:
    trimmed = " ".join(str(value).split())
    if len(trimmed) <= max_chars:
        return trimmed
    return trimmed[: max_chars - 3] + "..."


def _clean_extracted_value(value: str) -> str:
    normalized = value.strip().strip("`")
    normalized = normalized.strip("\"'")
    normalized = normalized.rstrip(")}]")
    normalized = normalized.rstrip(".")
    return normalized.strip()


def _looks_like_label_placeholder(value: str) -> bool:
    normalized = value.strip()
    if not normalized:
        return True
    if normalized.startswith("<") and normalized.endswith(">"):
        return True
    return normalized.lower() in {
        "answer",
        "short supporting span",
        "one short sentence",
        "0-1 number or na",
    }


def _looks_like_placeholder_text(text: str) -> bool:
    import re

    trimmed = text.strip()
    if not trimmed or "[" not in trimmed or "]" not in trimmed:
        return False
    return re.fullmatch(r"[\[\]\(\)\s,\d.\-]+", trimmed) is not None


def _extract_json_string_field(raw_text: str, field_name: str) -> str | None:
    import re

    pattern = rf'"{re.escape(field_name)}"\s*:\s*"((?:\\.|[^"\\])*)"'
    match = re.search(pattern, raw_text, flags=re.DOTALL)
    if not match:
        return None
    return bytes(match.group(1), "utf-8").decode("unicode_escape").strip() or None


def _extract_json_string_list_field(raw_text: str, field_name: str) -> list[str]:
    import re

    pattern = rf'"{re.escape(field_name)}"\s*:\s*\[(.*?)\]'
    match = re.search(pattern, raw_text, flags=re.DOTALL)
    if not match:
        return []
    list_body = match.group(1)
    items = re.findall(r'"((?:\\.|[^"\\])*)"', list_body)
    result: list[str] = []
    for item in items[:3]:
        normalized = bytes(item, "utf-8").decode("unicode_escape").strip()
        if normalized:
            result.append(normalized)
    return result


def _extract_json_array_field(raw_text: str, field_name: str) -> object | None:
    import json
    import re

    match = re.search(rf'"{re.escape(field_name)}"\s*:\s*\[', raw_text, flags=re.DOTALL)
    if not match:
        return None
    start = match.end() - 1
    depth = 0
    in_string = False
    escape = False
    end = None
    for index in range(start, len(raw_text)):
        char = raw_text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "[":
            depth += 1
            continue
        if char == "]":
            depth -= 1
            if depth == 0:
                end = index
                break
    if end is None:
        return None
    try:
        return json.loads(raw_text[start : end + 1])
    except Exception:
        return None


def _extract_json_number_field(raw_text: str, field_name: str) -> float | None:
    import re

    pattern = rf'"{re.escape(field_name)}"\s*:\s*(-?\d+(?:\.\d+)?)'
    match = re.search(pattern, raw_text)
    if not match:
        return None
    return float(match.group(1))


def _extract_json_bool_field(raw_text: str, field_name: str) -> bool | None:
    import re

    pattern = rf'"{re.escape(field_name)}"\s*:\s*(true|false)'
    match = re.search(pattern, raw_text, flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).lower() == "true"


def _extract_json_answer_field(raw_text: str, field_name: str) -> str | None:
    import re

    string_value = _extract_json_string_field(raw_text, field_name)
    if string_value is not None:
        return string_value
    pattern = rf'"{re.escape(field_name)}"\s*:\s*([-+]?\d[\d,]*(?:\.\d+)?)'
    match = re.search(pattern, raw_text)
    if not match:
        return None
    return match.group(1).replace(",", "")


def _extract_answer_guess_from_text(raw_text: str) -> str | None:
    import re

    yes_no = re.findall(r"\b(?:yes|no)\b", raw_text.lower())
    if yes_no:
        return yes_no[-1]
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", raw_text.replace(",", ""))
    if numbers:
        return numbers[-1].replace(",", "")
    return None
