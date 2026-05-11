"""按语义 schema 拆分的结构化输出校验器。"""

from __future__ import annotations

from typing import Any


UNCERTAINTY_TYPE_CHOICES = {
    "none",
    "calculation",
    "evidence_selection",
    "entity_linking",
    "multi_hop",
    "commonsense_gap",
    "format_extraction",
    "other",
}


def validate_answer_core_payload(payload: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {"final_answer", "reasoning"}
    actual_keys = set(payload)
    if "final_answer" not in payload:
        raise ValueError('Assistant output must include "final_answer".')
    if not actual_keys.issubset(allowed_keys):
        raise ValueError(
            f"Assistant output may only include keys {sorted(allowed_keys)}; got {sorted(actual_keys)}."
        )
    validated = {"final_answer": _require_answer_value(payload.get("final_answer"), "final_answer")}
    if "reasoning" in payload:
        validated["reasoning"] = _require_non_empty_string(payload.get("reasoning"), "reasoning")
    return validated


def validate_proxy_signal_answer_payload(
    payload: dict[str, Any],
    *,
    dataset: str | None,
    profile: str,
) -> dict[str, Any]:
    if profile == "selective":
        return _validate_selective_proxy_signal_payload(payload, dataset=dataset)
    if profile == "deliberation":
        return _validate_deliberation_proxy_signal_payload(payload)
    if profile == "budget":
        return _validate_budget_proxy_signal_payload(payload)
    raise ValueError(f"Unsupported proxy-signal profile: {profile}")


def validate_deliberation_packet_payload(payload: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "final_answer",
        "reasoning_trace",
        "confidence_raw",
        "claim_span",
        "key_evidence",
    }
    actual_keys = set(payload)
    if "final_answer" not in payload:
        raise ValueError('Assistant output must include "final_answer".')
    if not actual_keys.issubset(allowed_keys):
        raise ValueError(
            f"Assistant output may only include keys {sorted(allowed_keys)}; got {sorted(actual_keys)}."
        )
    return {
        "final_answer": _require_answer_value(payload.get("final_answer"), "final_answer"),
        "reasoning_trace": _require_nullable_hint(payload.get("reasoning_trace"), "reasoning_trace"),
        "confidence_raw": _optional_confidence_raw(payload.get("confidence_raw")),
        "claim_span": _require_nullable_hint(payload.get("claim_span"), "claim_span"),
        "key_evidence": _require_nullable_hint(payload.get("key_evidence"), "key_evidence"),
    }


def validate_belief_update_delta_payload(payload: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "changed_answer",
        "new_answer",
        "confidence_delta",
        "reason_for_change",
        "remaining_disagreement",
    }
    actual_keys = set(payload)
    required_keys = {"changed_answer"}
    missing = sorted(required_keys - actual_keys)
    if missing:
        raise ValueError(f"Assistant output is missing required keys: {missing}.")
    if not actual_keys.issubset(allowed_keys):
        raise ValueError(
            f"Assistant output may only include keys {sorted(allowed_keys)}; got {sorted(actual_keys)}."
        )
    changed_answer = _require_bool(payload.get("changed_answer"), "changed_answer")
    new_answer = _optional_answer_value(payload.get("new_answer"), "new_answer")
    if changed_answer and new_answer is None:
        raise ValueError("new_answer must be present when changed_answer is true.")
    return {
        "changed_answer": changed_answer,
        "new_answer": new_answer,
        "confidence_delta": _optional_float(payload.get("confidence_delta"), "confidence_delta"),
        "reason_for_change": _require_nullable_hint(payload.get("reason_for_change"), "reason_for_change"),
        "remaining_disagreement": _require_nullable_hint(payload.get("remaining_disagreement"), "remaining_disagreement"),
    }


def validate_audit_verdict_payload(payload: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {"decision", "verified_answer", "rationale"}
    actual_keys = set(payload)
    required_keys = {"decision", "rationale"}
    missing = sorted(required_keys - actual_keys)
    if missing:
        raise ValueError(f"Assistant output is missing required keys: {missing}.")
    if not actual_keys.issubset(allowed_keys):
        raise ValueError(
            f"Assistant output may only include keys {sorted(allowed_keys)}; got {sorted(actual_keys)}."
        )
    return {
        "decision": _require_enum(
            payload.get("decision"),
            "decision",
            {"resolve_for_a", "resolve_for_b", "abstain"},
        ),
        "verified_answer": _optional_answer_value(payload.get("verified_answer"), "verified_answer"),
        "rationale": _require_non_empty_string(payload.get("rationale"), "rationale"),
    }


def validate_split_context_solver_payload(payload: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "final_answer",
        "reasoning_trace",
        "reasoning",
        "evidence_summary",
        "key_evidence",
        "supporting_facts",
        "confidence_raw",
    }
    actual_keys = set(payload)
    if "final_answer" not in payload:
        raise ValueError('Assistant output must include "final_answer".')
    if not actual_keys.issubset(allowed_keys):
        raise ValueError(
            f"Assistant output may only include keys {sorted(allowed_keys)}; got {sorted(actual_keys)}."
        )
    return {
        "final_answer": _optional_answer_value(payload.get("final_answer"), "final_answer"),
        "reasoning_trace": _require_nullable_hint(
            payload.get("reasoning_trace", payload.get("reasoning")),
            "reasoning_trace",
        ),
        "evidence_summary": _require_nullable_hint(
            payload.get("evidence_summary", payload.get("key_evidence")),
            "evidence_summary",
        ),
        "supporting_facts": _normalize_supporting_facts_payload(payload.get("supporting_facts")),
        "confidence_raw": _optional_confidence_raw(payload.get("confidence_raw")),
    }


def validate_split_context_belief_payload(payload: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "changed_answer",
        "final_answer",
        "new_answer",
        "reasoning_trace",
        "reasoning",
        "evidence_summary",
        "key_evidence",
        "supporting_facts",
        "confidence_raw",
    }
    actual_keys = set(payload)
    changed_answer = payload.get("changed_answer")
    if changed_answer is not None and not isinstance(changed_answer, bool):
        raise ValueError("changed_answer must be a boolean when present.")
    if "final_answer" not in payload and "new_answer" not in payload:
        raise ValueError('Assistant output must include "final_answer" or "new_answer".')
    if not actual_keys.issubset(allowed_keys):
        raise ValueError(
            f"Assistant output may only include keys {sorted(allowed_keys)}; got {sorted(actual_keys)}."
        )
    normalized_answer = _optional_answer_value(
        payload.get("final_answer", payload.get("new_answer")),
        "final_answer",
    )
    normalized_changed_answer = bool(changed_answer)
    if normalized_answer is None:
        normalized_changed_answer = False
    return {
        "changed_answer": normalized_changed_answer,
        "final_answer": normalized_answer,
        "reasoning_trace": _require_nullable_hint(
            payload.get("reasoning_trace", payload.get("reasoning")),
            "reasoning_trace",
        ),
        "evidence_summary": _require_nullable_hint(
            payload.get("evidence_summary", payload.get("key_evidence")),
            "evidence_summary",
        ),
        "supporting_facts": _normalize_supporting_facts_payload(payload.get("supporting_facts")),
        "confidence_raw": _optional_confidence_raw(payload.get("confidence_raw")),
    }


def validate_cue_blackbox_packet_payload(payload: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "final_answer",
        "confidence",
        "reasoning_sketch",
        "uncertain_point",
        "top_claims",
        "evidence_items",
        "counter_answer",
    }
    actual_keys = set(payload)
    if "final_answer" not in payload:
        raise ValueError('Assistant output must include "final_answer".')
    if not actual_keys.issubset(allowed_keys):
        raise ValueError(
            f"Assistant output may only include keys {sorted(allowed_keys)}; got {sorted(actual_keys)}."
        )
    return {
        "final_answer": _require_answer_value(payload.get("final_answer"), "final_answer"),
        "confidence": _optional_float(payload.get("confidence"), "confidence"),
        "reasoning_sketch": _require_nullable_hint(payload.get("reasoning_sketch"), "reasoning_sketch")
        or _require_answer_value(payload.get("final_answer"), "final_answer"),
        "uncertain_point": _require_nullable_hint(payload.get("uncertain_point"), "uncertain_point"),
        "top_claims": _require_short_string_list(payload.get("top_claims"), "top_claims"),
        "evidence_items": _require_short_string_list(payload.get("evidence_items"), "evidence_items"),
        "counter_answer": _optional_answer_value(payload.get("counter_answer"), "counter_answer"),
    }


def _validate_selective_proxy_signal_payload(payload: dict[str, Any], *, dataset: str | None) -> dict[str, Any]:
    allowed_keys = {
        "final_answer",
        "reasoning",
        "confidence_raw",
        "claim_span",
        "uncertainty_type",
        "key_evidence",
        "uncertain_point",
    }
    actual_keys = set(payload)
    required_keys = {"final_answer"}
    missing = sorted(required_keys - actual_keys)
    if missing:
        raise ValueError(f"Assistant output is missing required keys: {missing}.")
    if not actual_keys.issubset(allowed_keys):
        raise ValueError(
            f"Assistant output may only include keys {sorted(allowed_keys)}; got {sorted(actual_keys)}."
        )
    validated: dict[str, Any] = {
        "final_answer": _require_answer_value(payload.get("final_answer"), "final_answer"),
        "confidence_raw": _optional_confidence_raw(payload.get("confidence_raw")),
    }
    validated["reasoning"] = (
        _require_non_empty_string(payload.get("reasoning"), "reasoning")
        if "reasoning" in payload
        else validated["final_answer"]
    )
    validated["claim_span"] = (
        _require_nullable_hint(payload.get("claim_span"), "claim_span")
        if "claim_span" in payload
        else validated["final_answer"]
    )
    validated["uncertainty_type"] = (
        _require_uncertainty_type(payload.get("uncertainty_type"))
        if "uncertainty_type" in payload
        else _default_uncertainty_type(dataset)
    )
    validated["key_evidence"] = (
        _require_nullable_hint(payload.get("key_evidence"), "key_evidence")
        if "key_evidence" in payload
        else validated["claim_span"]
    )
    validated["uncertain_point"] = (
        _require_nullable_hint(payload.get("uncertain_point"), "uncertain_point")
        if "uncertain_point" in payload
        else None
    )
    return validated


def _validate_deliberation_proxy_signal_payload(payload: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "final_answer",
        "reasoning_trace",
        "reasoning_sketch",
        "claim_span",
        "confidence_raw",
        "uncertain_point",
        "key_evidence",
    }
    actual_keys = set(payload)
    if "final_answer" not in payload:
        raise ValueError('Assistant output must include "final_answer".')
    if not actual_keys.issubset(allowed_keys):
        raise ValueError(
            f"Assistant output may only include keys {sorted(allowed_keys)}; got {sorted(actual_keys)}."
        )
    return {
        "final_answer": _require_answer_value(payload.get("final_answer"), "final_answer"),
        "reasoning_trace": _require_nullable_hint(
            payload.get("reasoning_trace", payload.get("reasoning_sketch")),
            "reasoning_trace",
        ),
        "claim_span": _require_nullable_hint(payload.get("claim_span"), "claim_span"),
        "confidence_raw": _optional_confidence_raw(payload.get("confidence_raw")),
        "uncertain_point": _require_nullable_hint(payload.get("uncertain_point"), "uncertain_point"),
        "key_evidence": _require_nullable_hint(payload.get("key_evidence"), "key_evidence"),
    }


def _validate_budget_proxy_signal_payload(payload: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "final_answer",
        "reasoning_trace",
        "reasoning_sketch",
        "claim_span",
        "key_evidence",
        "keyword_clues",
        "confidence_raw",
        "uncertain_point",
    }
    actual_keys = set(payload)
    required_keys = {"final_answer", "confidence_raw", "keyword_clues"}
    missing = sorted(required_keys - actual_keys)
    if missing:
        raise ValueError(f"Assistant output is missing required keys: {missing}.")
    if not actual_keys.issubset(allowed_keys):
        raise ValueError(
            f"Assistant output may only include keys {sorted(allowed_keys)}; got {sorted(actual_keys)}."
        )
    return {
        "final_answer": _require_answer_value(payload.get("final_answer"), "final_answer"),
        "reasoning_trace": _require_nullable_hint(
            payload.get("reasoning_trace", payload.get("reasoning_sketch")),
            "reasoning_trace",
        ),
        "claim_span": _require_nullable_hint(payload.get("claim_span"), "claim_span"),
        "key_evidence": _require_nullable_hint(payload.get("key_evidence"), "key_evidence"),
        "keyword_clues": _require_keyword_clues(payload.get("keyword_clues")),
        "confidence_raw": _require_confidence_raw(payload.get("confidence_raw")),
        "uncertain_point": _require_nullable_hint(payload.get("uncertain_point"), "uncertain_point"),
    }


def _require_non_empty_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


def _require_answer_value(value: object, field_name: str) -> str:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{field_name} must be a string or numeric value.")
    if isinstance(value, (int, float)):
        return str(value)
    return _require_non_empty_string(value, field_name)


def _optional_answer_value(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return _require_answer_value(value, field_name)


def _require_nullable_hint(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string or null.")
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.lower() in {"none", "null", "n/a", "na"}:
        return None
    return normalized


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


def _require_uncertainty_type(value: object) -> str:
    normalized = _require_non_empty_string(value, "uncertainty_type").lower()
    if normalized not in UNCERTAINTY_TYPE_CHOICES:
        raise ValueError(f"uncertainty_type must be one of {sorted(UNCERTAINTY_TYPE_CHOICES)}.")
    return normalized


def _default_uncertainty_type(dataset: str | None) -> str:
    if dataset in {"gsm8k", "gsm_symbolic", "math500"}:
        return "calculation"
    if dataset == "hotpotqa":
        return "multi_hop"
    if dataset == "strategyqa":
        return "commonsense_gap"
    return "other"


def _optional_float(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a numeric value or null.")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return None
        try:
            return float(normalized)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be numeric when provided as a string.") from exc
    raise ValueError(f"{field_name} must be a numeric value or null.")


def _require_keyword_clues(value: object) -> list[str]:
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise ValueError("keyword_clues must be non-empty.")
        return [normalized]
    if not isinstance(value, list):
        raise ValueError("keyword_clues must be a string or a list of strings.")
    normalized_items = [_require_non_empty_string(item, "keyword_clues[]") for item in value]
    if not normalized_items:
        raise ValueError("keyword_clues must contain at least one clue.")
    return normalized_items


def _require_short_string_list(value: object, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        normalized = value.strip()
        return [normalized] if normalized else []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a string, list of strings, or null.")
    normalized_items = [_require_non_empty_string(item, f"{field_name}[]") for item in value]
    return normalized_items[:3]


def _normalize_supporting_facts_payload(value: object) -> list[dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []
    if value is None:
        return normalized_rows
    raw_items = value if isinstance(value, list) else [value]
    for item in raw_items:
        title: object | None = None
        sent_id: object | None = None
        if isinstance(item, dict):
            title = item.get("title")
            sent_id = item.get("sent_id", item.get("sentence_id"))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            title = item[0]
            sent_id = item[1]
        normalized_title = str(title or "").strip()
        if not normalized_title:
            continue
        try:
            normalized_sent_id = int(str(sent_id).strip())
        except (TypeError, ValueError):
            continue
        normalized_rows.append({"title": normalized_title, "sent_id": normalized_sent_id})
    return normalized_rows[:6]


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean.")
    return value


def _require_enum(value: object, field_name: str, allowed: set[str]) -> str:
    normalized = _require_non_empty_string(value, field_name)
    if normalized not in allowed:
        raise ValueError(f"{field_name} must be one of {sorted(allowed)}.")
    return normalized
