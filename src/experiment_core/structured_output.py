"""Minimal model-facing output validators plus framework artifact version."""

from __future__ import annotations

from typing import Any, Literal


ARTIFACT_VERSION = "v5"
OUTPUT_MODE_CORE = "core"
OUTPUT_MODE_SELECTIVE_COMM = "selective_comm"
OUTPUT_MODE_SPARC_SOLVER = "sparc_solver"
OUTPUT_MODE_SPARC_MESSAGE = "sparc_message"
OUTPUT_MODE_SPARC_BELIEF_UPDATE = "sparc_belief_update"
OUTPUT_MODE_SPARC_AUDIT = "sparc_audit"
OutputMode = Literal[
    "core",
    "selective_comm",
    "sparc_solver",
    "sparc_message",
    "sparc_belief_update",
    "sparc_audit",
]


def validate_structured_output(raw_text: str, output_mode: OutputMode) -> dict[str, Any]:
    """Validate one assistant response against the minimal model-facing contract."""
    cleaned = raw_text.strip()
    if not cleaned:
        raise ValueError("Assistant output is empty.")
    if cleaned.startswith("```"):
        raise ValueError("Markdown code fences are not allowed.")

    payload = _decode_json_object(cleaned)
    if output_mode == OUTPUT_MODE_CORE:
        return _validate_core_output(payload)
    if output_mode == OUTPUT_MODE_SELECTIVE_COMM:
        return _validate_selective_output(payload)
    if output_mode == OUTPUT_MODE_SPARC_SOLVER:
        return _validate_sparc_solver_output(payload)
    if output_mode == OUTPUT_MODE_SPARC_MESSAGE:
        return _validate_sparc_message_output(payload)
    if output_mode == OUTPUT_MODE_SPARC_BELIEF_UPDATE:
        return _validate_sparc_belief_update_output(payload)
    if output_mode == OUTPUT_MODE_SPARC_AUDIT:
        return _validate_sparc_audit_output(payload)
    raise ValueError(f"Unsupported output mode: {output_mode}")


def _validate_core_output(payload: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {"final_answer", "reasoning"}
    actual_keys = set(payload)
    if "final_answer" not in payload:
        raise ValueError('Assistant output must include "final_answer".')
    if not actual_keys.issubset(allowed_keys):
        raise ValueError(
            f"Assistant output may only include keys {sorted(allowed_keys)}; got {sorted(actual_keys)}."
        )

    validated: dict[str, Any] = {
        "final_answer": _require_answer_value(payload.get("final_answer"), "final_answer"),
    }
    if "reasoning" in payload:
        validated["reasoning"] = _require_non_empty_string(payload.get("reasoning"), "reasoning")
    return validated


def _validate_selective_output(payload: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {"final_answer", "reasoning", "confidence_raw", "key_evidence", "uncertain_point"}
    actual_keys = set(payload)
    required_keys = {"final_answer", "confidence_raw"}
    missing = sorted(required_keys - actual_keys)
    if missing:
        raise ValueError(f"Assistant output is missing required keys: {missing}.")
    if not actual_keys.issubset(allowed_keys):
        raise ValueError(
            f"Assistant output may only include keys {sorted(allowed_keys)}; got {sorted(actual_keys)}."
        )

    validated: dict[str, Any] = {
        "final_answer": _require_answer_value(payload.get("final_answer"), "final_answer"),
        "confidence_raw": _require_confidence_raw(payload.get("confidence_raw")),
    }
    if "reasoning" in payload:
        validated["reasoning"] = _require_non_empty_string(payload.get("reasoning"), "reasoning")
    if "key_evidence" in payload:
        validated["key_evidence"] = _require_nullable_hint(payload.get("key_evidence"), "key_evidence")
    else:
        validated["key_evidence"] = None
    if "uncertain_point" in payload:
        validated["uncertain_point"] = _require_nullable_hint(payload.get("uncertain_point"), "uncertain_point")
    else:
        validated["uncertain_point"] = None
    return validated


def _validate_sparc_solver_output(payload: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "final_answer",
        "reasoning_trace",
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
    validated: dict[str, Any] = {
        "final_answer": _require_answer_value(payload.get("final_answer"), "final_answer"),
        "reasoning_trace": _require_nullable_hint(payload.get("reasoning_trace"), "reasoning_trace"),
        "claim_span": _require_nullable_hint(payload.get("claim_span"), "claim_span"),
        "confidence_raw": _optional_confidence_raw(payload.get("confidence_raw")),
        "uncertain_point": _require_nullable_hint(payload.get("uncertain_point"), "uncertain_point"),
        "key_evidence": _require_nullable_hint(payload.get("key_evidence"), "key_evidence"),
    }
    return validated


def _validate_sparc_message_output(payload: dict[str, Any]) -> dict[str, Any]:
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


def _validate_sparc_belief_update_output(payload: dict[str, Any]) -> dict[str, Any]:
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


def _validate_sparc_audit_output(payload: dict[str, Any]) -> dict[str, Any]:
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


def _decode_json_object(cleaned: str) -> dict[str, Any]:
    import json

    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("Assistant output must be a JSON object.")
    return payload


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


def _require_nullable_string(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string or null.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty when present.")
    return normalized


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
    if isinstance(value, str) and not value.strip():
        return None
    return _require_confidence_raw(value)


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


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean.")
    return value


def _require_enum(value: object, field_name: str, allowed: set[str]) -> str:
    normalized = _require_non_empty_string(value, field_name)
    if normalized not in allowed:
        raise ValueError(f"{field_name} must be one of {sorted(allowed)}.")
    return normalized
