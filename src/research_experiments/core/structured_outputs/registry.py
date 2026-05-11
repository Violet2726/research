"""结构化输出 schema 注册表与公共 API。"""

from __future__ import annotations

from typing import Any, Literal
import json

from research_experiments.core.structured_outputs.recovery import (
    looks_like_soft_rejection_text,
    parse_proxy_signal_answer_payload,
    recover_answer_from_reasoning_text,
    recover_partial_payload,
    recover_soft_rejection_payload,
    structured_output_candidates,
)
from research_experiments.core.structured_outputs.validators import (
    validate_answer_core_payload,
    validate_audit_verdict_payload,
    validate_belief_update_delta_payload,
    validate_cue_blackbox_packet_payload,
    validate_deliberation_packet_payload,
    validate_proxy_signal_answer_payload,
    validate_split_context_belief_payload,
    validate_split_context_solver_payload,
)


ARTIFACT_VERSION = "v5"
SCHEMA_ANSWER_CORE = "answer_core"
SCHEMA_ANSWER_WITH_PROXY_SIGNALS_SELECTIVE = "answer_with_proxy_signals.selective"
SCHEMA_ANSWER_WITH_PROXY_SIGNALS_DELIBERATION = "answer_with_proxy_signals.deliberation"
SCHEMA_ANSWER_WITH_PROXY_SIGNALS_BUDGET = "answer_with_proxy_signals.budget"
SCHEMA_DELIBERATION_PACKET = "deliberation_packet"
SCHEMA_BELIEF_UPDATE_DELTA = "belief_update_delta"
SCHEMA_AUDIT_VERDICT = "audit_verdict"
SCHEMA_SPLIT_CONTEXT_SOLVER = "split_context_solver"
SCHEMA_SPLIT_CONTEXT_BELIEF = "split_context_belief"
SCHEMA_CUE_BLACKBOX_PACKET = "cue_blackbox_packet"

SchemaId = Literal[
    "answer_core",
    "answer_with_proxy_signals.selective",
    "answer_with_proxy_signals.deliberation",
    "answer_with_proxy_signals.budget",
    "deliberation_packet",
    "belief_update_delta",
    "audit_verdict",
    "split_context_solver",
    "split_context_belief",
    "cue_blackbox_packet",
]


def validate_structured_output(
    raw_text: str,
    schema_id: SchemaId,
    *,
    dataset: str | None = None,
) -> dict[str, Any]:
    cleaned = raw_text.strip()
    if not cleaned:
        raise ValueError("Assistant output is empty.")
    if cleaned.startswith("```"):
        raise ValueError("Markdown code fences are not allowed.")
    payload = _decode_json_object(cleaned)
    return _validate_payload(payload, schema_id, dataset=dataset)


def validate_or_recover_structured_output(
    assistant_text: str,
    schema_id: SchemaId,
    *,
    dataset: str | None = None,
    provider_reasoning_text: str | None = None,
) -> dict[str, Any]:
    candidate_errors: list[str] = []
    for candidate_text, source in structured_output_candidates(
        assistant_text=assistant_text,
        provider_reasoning_text=provider_reasoning_text,
        schema_id=schema_id,
    ):
        if schema_id == SCHEMA_ANSWER_WITH_PROXY_SIGNALS_SELECTIVE:
            if dataset is None:
                raise ValueError("dataset is required for proxy-signal selective recovery.")
            try:
                payload = (
                    recover_answer_from_reasoning_text(candidate_text, dataset)
                    if source == "reasoning_fallback"
                    else parse_proxy_signal_answer_payload(candidate_text, dataset)
                )
                return _validate_payload(payload, schema_id, dataset=dataset)
            except Exception as exc:
                candidate_errors.append(f"{source}: {exc}")
                continue
        try:
            return validate_structured_output(candidate_text, schema_id, dataset=dataset)
        except Exception as exc:
            candidate_errors.append(f"{source}: {exc}")
            recovered = recover_partial_payload(candidate_text, schema_id)
            if recovered is not None:
                return _validate_payload(recovered, schema_id, dataset=dataset)
    soft_rejection_fallback = _recover_soft_rejection_output(
        assistant_text=assistant_text,
        provider_reasoning_text=provider_reasoning_text,
        schema_id=schema_id,
        dataset=dataset,
    )
    if soft_rejection_fallback is not None:
        return soft_rejection_fallback
    raise ValueError("Structured output recovery failed: " + " | ".join(candidate_errors[-4:]))


def parse_proxy_signal_answer(raw_text: str, dataset: str) -> dict[str, Any]:
    payload = parse_proxy_signal_answer_payload(raw_text, dataset)
    return _validate_payload(payload, SCHEMA_ANSWER_WITH_PROXY_SIGNALS_SELECTIVE, dataset=dataset)


def _validate_payload(payload: dict[str, Any], schema_id: SchemaId, *, dataset: str | None) -> dict[str, Any]:
    if schema_id == SCHEMA_ANSWER_CORE:
        return validate_answer_core_payload(payload)
    if schema_id == SCHEMA_ANSWER_WITH_PROXY_SIGNALS_SELECTIVE:
        return validate_proxy_signal_answer_payload(payload, dataset=dataset, profile="selective")
    if schema_id == SCHEMA_ANSWER_WITH_PROXY_SIGNALS_DELIBERATION:
        return validate_proxy_signal_answer_payload(payload, dataset=dataset, profile="deliberation")
    if schema_id == SCHEMA_ANSWER_WITH_PROXY_SIGNALS_BUDGET:
        return validate_proxy_signal_answer_payload(payload, dataset=dataset, profile="budget")
    if schema_id == SCHEMA_DELIBERATION_PACKET:
        return validate_deliberation_packet_payload(payload)
    if schema_id == SCHEMA_BELIEF_UPDATE_DELTA:
        return validate_belief_update_delta_payload(payload)
    if schema_id == SCHEMA_AUDIT_VERDICT:
        return validate_audit_verdict_payload(payload)
    if schema_id == SCHEMA_SPLIT_CONTEXT_SOLVER:
        return validate_split_context_solver_payload(payload)
    if schema_id == SCHEMA_SPLIT_CONTEXT_BELIEF:
        return validate_split_context_belief_payload(payload)
    if schema_id == SCHEMA_CUE_BLACKBOX_PACKET:
        return validate_cue_blackbox_packet_payload(payload)
    raise ValueError(f"Unsupported schema_id: {schema_id}")


def _recover_soft_rejection_output(
    *,
    assistant_text: str,
    provider_reasoning_text: str | None,
    schema_id: SchemaId,
    dataset: str | None,
) -> dict[str, Any] | None:
    candidate_texts = [str(assistant_text or ""), str(provider_reasoning_text or "")]
    if not any(looks_like_soft_rejection_text(text) for text in candidate_texts):
        return None
    payload = recover_soft_rejection_payload(schema_id, dataset=dataset)
    if payload is None:
        return None
    return _validate_payload(payload, schema_id, dataset=dataset)


def _decode_json_object(cleaned: str) -> dict[str, Any]:
    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("Assistant output must be a JSON object.")
    return payload
