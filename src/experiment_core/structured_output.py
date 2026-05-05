"""面向模型输出的最小结构化校验器。

本模块定义仓库内各类 prompt 对应的最小 JSON 契约，并在模型返回文本后做统一校验。
设计目标不是做语义纠错，而是尽早发现 schema 偏移、字段缺失和类型错误，
从而让 runner 可以稳定地区分“请求失败”“格式失败”和“答案内容本身错误”。
"""

from __future__ import annotations

from typing import Any, Literal


ARTIFACT_VERSION = "v5"
OUTPUT_MODE_CORE = "core"
OUTPUT_MODE_SELECTIVE_COMM = "selective_comm"
OUTPUT_MODE_BUDGET_SOLVER = "budget_solver"
OUTPUT_MODE_BUDGET_BELIEF_UPDATE = "budget_belief_update"
OUTPUT_MODE_SPARC_SOLVER = "sparc_solver"
OUTPUT_MODE_SPARC_MESSAGE = "sparc_message"
OUTPUT_MODE_SPARC_BELIEF_UPDATE = "sparc_belief_update"
OUTPUT_MODE_SPARC_AUDIT = "sparc_audit"
OUTPUT_MODE_CUE_SOLVER = "cue_solver"
OUTPUT_MODE_CUE_BELIEF_UPDATE = "cue_belief_update"
OUTPUT_MODE_CUE_AUDIT = "cue_audit"
OUTPUT_MODE_COMM_NECESSARY_SOLVER = "comm_necessary_solver"
OUTPUT_MODE_COMM_NECESSARY_BELIEF = "comm_necessary_belief"
OutputMode = Literal[
    "core",
    "selective_comm",
    "budget_solver",
    "budget_belief_update",
    "sparc_solver",
    "sparc_message",
    "sparc_belief_update",
    "sparc_audit",
    "cue_solver",
    "cue_belief_update",
    "cue_audit",
    "comm_necessary_solver",
    "comm_necessary_belief",
]

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


def validate_structured_output(raw_text: str, output_mode: OutputMode) -> dict[str, Any]:
    """按指定输出模式校验一条模型响应。

    返回值是已经过字段级校验和轻度归一化的字典；
    如果结构不满足约定，则抛出 `ValueError`，由上层 runner 记录为 schema 失败。
    """
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
    if output_mode == OUTPUT_MODE_BUDGET_SOLVER:
        return _validate_budget_solver_output(payload)
    if output_mode == OUTPUT_MODE_BUDGET_BELIEF_UPDATE:
        return _validate_budget_belief_update_output(payload)
    if output_mode == OUTPUT_MODE_SPARC_SOLVER:
        return _validate_sparc_solver_output(payload)
    if output_mode == OUTPUT_MODE_SPARC_MESSAGE:
        return _validate_sparc_message_output(payload)
    if output_mode == OUTPUT_MODE_SPARC_BELIEF_UPDATE:
        return _validate_sparc_belief_update_output(payload)
    if output_mode == OUTPUT_MODE_SPARC_AUDIT:
        return _validate_sparc_audit_output(payload)
    if output_mode == OUTPUT_MODE_CUE_SOLVER:
        return _validate_cue_solver_output(payload)
    if output_mode == OUTPUT_MODE_CUE_BELIEF_UPDATE:
        return _validate_cue_belief_update_output(payload)
    if output_mode == OUTPUT_MODE_CUE_AUDIT:
        return _validate_cue_audit_output(payload)
    if output_mode == OUTPUT_MODE_COMM_NECESSARY_SOLVER:
        return _validate_comm_necessary_solver_output(payload)
    if output_mode == OUTPUT_MODE_COMM_NECESSARY_BELIEF:
        return _validate_comm_necessary_belief_output(payload)
    raise ValueError(f"Unsupported output mode: {output_mode}")


def validate_or_recover_structured_output(
    assistant_text: str,
    output_mode: OutputMode,
    *,
    dataset: str | None = None,
    provider_reasoning_text: str | None = None,
) -> dict[str, Any]:
    """Validate one response, then fall back to shared recovery heuristics when possible."""
    candidate_errors: list[str] = []
    for candidate_text, source in _structured_output_candidates(
        assistant_text=assistant_text,
        provider_reasoning_text=provider_reasoning_text,
        output_mode=output_mode,
    ):
        if output_mode == OUTPUT_MODE_SELECTIVE_COMM:
            if dataset is None:
                raise ValueError("dataset is required for selective output recovery.")
            try:
                if source == "reasoning_fallback":
                    return _recover_selective_output_from_reasoning_text(candidate_text, dataset)
                return parse_selective_output(candidate_text, dataset)
            except Exception as exc:
                candidate_errors.append(f"{source}: {exc}")
                continue
        try:
            return validate_structured_output(candidate_text, output_mode)
        except Exception as exc:
            candidate_errors.append(f"{source}: {exc}")
            recovered = _recover_partial_json_output(candidate_text, output_mode)
            if recovered is not None:
                return recovered
    soft_rejection_fallback = _recover_soft_rejection_output(
        assistant_text=assistant_text,
        provider_reasoning_text=provider_reasoning_text,
        output_mode=output_mode,
        dataset=dataset,
    )
    if soft_rejection_fallback is not None:
        return soft_rejection_fallback
    raise ValueError("Structured output recovery failed: " + " | ".join(candidate_errors[-4:]))


def parse_selective_output(raw_text: str, dataset: str) -> dict[str, Any]:
    """解析 selective_comm 输出。

    新协议优先支持简短标签行格式，并在可能时接受 JSON；
    对不完全遵守格式但仍能识别答案的文本，做最小化结构抽取。
    """
    cleaned = raw_text.strip()
    if not cleaned:
        raise ValueError("Assistant output is empty.")
    if cleaned.startswith("{"):
        try:
            payload = _normalize_selective_json_payload(_decode_json_object(cleaned))
            return _validate_selective_output(payload, dataset=dataset)
        except Exception:
            pass
    return _coerce_selective_text_output(cleaned, dataset)


def _normalize_selective_json_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize provider JSON keys like `FINAL_ANSWER` into the selective schema."""
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


def _validate_core_output(payload: dict[str, Any]) -> dict[str, Any]:
    """校验最基础的 `final_answer [+ reasoning]` 输出模式。"""
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


def _validate_selective_output(payload: dict[str, Any], *, dataset: str | None = None) -> dict[str, Any]:
    """校验选择性通信实验使用的回答与置信度输出。"""
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
    if "reasoning" in payload:
        validated["reasoning"] = _require_non_empty_string(payload.get("reasoning"), "reasoning")
    else:
        validated["reasoning"] = validated["final_answer"]
    if "claim_span" in payload:
        validated["claim_span"] = _require_nullable_hint(payload.get("claim_span"), "claim_span")
    else:
        validated["claim_span"] = validated["final_answer"]
    if "uncertainty_type" in payload:
        validated["uncertainty_type"] = _require_uncertainty_type(payload.get("uncertainty_type"))
    else:
        validated["uncertainty_type"] = _default_uncertainty_type(dataset)
    if "key_evidence" in payload:
        validated["key_evidence"] = _require_nullable_hint(payload.get("key_evidence"), "key_evidence")
    else:
        validated["key_evidence"] = validated["claim_span"]
    if "uncertain_point" in payload:
        validated["uncertain_point"] = _require_nullable_hint(payload.get("uncertain_point"), "uncertain_point")
    else:
        validated["uncertain_point"] = None
    return validated


def _coerce_selective_text_output(cleaned: str, dataset: str) -> dict[str, Any]:
    """从标签行或自由文本中抽取最小 selective 结构。"""
    final_answer = _extract_selective_final_answer(cleaned, dataset)
    if final_answer is None:
        raise ValueError("Selective output must contain a recoverable final answer.")

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


def _validate_sparc_solver_output(payload: dict[str, Any]) -> dict[str, Any]:
    """校验 SPARC Stage A solver 的结构化输出。"""
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
    validated: dict[str, Any] = {
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
    return validated


def _validate_budget_solver_output(payload: dict[str, Any]) -> dict[str, Any]:
    """校验 `budget_comm` Stage A solver 的结构化输出。"""
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


def _validate_sparc_message_output(payload: dict[str, Any]) -> dict[str, Any]:
    """校验 SPARC 消息包投影前的候选输出。"""
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
    """校验 belief update 阶段的变更说明输出。"""
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


def _validate_budget_belief_update_output(payload: dict[str, Any]) -> dict[str, Any]:
    """`budget_comm` 与 SPARC 共用同一套 belief update 校验规则。"""
    return _validate_sparc_belief_update_output(payload)


def _validate_sparc_audit_output(payload: dict[str, Any]) -> dict[str, Any]:
    """校验 SPARC 局部审计器的裁决输出。"""
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


def _validate_cue_solver_output(payload: dict[str, Any]) -> dict[str, Any]:
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


def _validate_cue_belief_update_output(payload: dict[str, Any]) -> dict[str, Any]:
    return _validate_sparc_belief_update_output(payload)


def _validate_cue_audit_output(payload: dict[str, Any]) -> dict[str, Any]:
    return _validate_sparc_audit_output(payload)


def _validate_comm_necessary_solver_output(payload: dict[str, Any]) -> dict[str, Any]:
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
        "final_answer": _require_answer_value(payload.get("final_answer"), "final_answer"),
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


def _validate_comm_necessary_belief_output(payload: dict[str, Any]) -> dict[str, Any]:
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
    candidate_answer = payload.get("final_answer", payload.get("new_answer"))
    if candidate_answer is None:
        raise ValueError('Assistant output must include "final_answer" or "new_answer".')
    if not actual_keys.issubset(allowed_keys):
        raise ValueError(
            f"Assistant output may only include keys {sorted(allowed_keys)}; got {sorted(actual_keys)}."
        )
    return {
        "changed_answer": bool(changed_answer),
        "final_answer": _require_answer_value(candidate_answer, "final_answer"),
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


def _decode_json_object(cleaned: str) -> dict[str, Any]:
    """把模型文本解码为 JSON 对象，并拒绝非对象顶层结构。"""
    import json

    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("Assistant output must be a JSON object.")
    return payload


def _structured_output_candidates(
    *,
    assistant_text: str,
    provider_reasoning_text: str | None,
    output_mode: OutputMode,
) -> list[tuple[str, str]]:
    assistant = str(assistant_text or "").strip()
    reasoning = str(provider_reasoning_text or "").strip()
    if output_mode == OUTPUT_MODE_SELECTIVE_COMM and _looks_like_placeholder_visible_output(assistant):
        ordered = [(reasoning, "reasoning_fallback"), (assistant, "assistant_text")]
    else:
        ordered = [(assistant, "assistant_text"), (reasoning, "reasoning_fallback")]
    return [(text, source) for text, source in ordered if text]


def _looks_like_placeholder_visible_output(text: str) -> bool:
    import re

    trimmed = text.strip()
    if not trimmed:
        return False
    return re.fullmatch(r"\[\s*[\w\s,.\-]*\s*\]", trimmed, re.S) is not None


def _clip_reasoning_text(value: str, max_chars: int = 180) -> str:
    text = " ".join(value.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _extract_answer_phrase_from_reasoning(text: str) -> str | None:
    import re

    patterns = [
        r"(?i)\b(?:therefore|thus|so)?\s*,?\s*(?:the\s+)?(?:final\s+)?answer\s+(?:should\s+be|is)\s*[:\-]?\s*([^\n.]+)",
        r"(?i)\b(?:the\s+)?correct\s+answer\s+(?:should\s+be|is)\s*[:\-]?\s*([^\n.]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        candidate = match.group(1).strip().strip("\"'`").rstrip(".")
        if candidate and not _looks_like_placeholder_visible_output(candidate):
            return candidate
    return None


def _recover_selective_output_from_reasoning_text(reasoning_text: str, dataset: str) -> dict[str, Any]:
    import re

    cleaned = " ".join(reasoning_text.split())
    if not cleaned:
        raise ValueError("Provider reasoning text is empty.")

    answer_phrase = _extract_answer_phrase_from_reasoning(cleaned)
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
        "reasoning": _clip_reasoning_text(cleaned),
        "claim_span": final_answer,
        "uncertainty_type": _default_uncertainty_type(dataset),
        "key_evidence": final_answer,
        "uncertain_point": None,
    }


def _recover_partial_json_output(raw_text: str, output_mode: OutputMode) -> dict[str, Any] | None:
    if output_mode == OUTPUT_MODE_CORE:
        return _recover_core_output(raw_text)
    if output_mode == OUTPUT_MODE_BUDGET_SOLVER:
        return _recover_budget_solver_output(raw_text)
    if output_mode in {OUTPUT_MODE_BUDGET_BELIEF_UPDATE, OUTPUT_MODE_SPARC_BELIEF_UPDATE, OUTPUT_MODE_CUE_BELIEF_UPDATE}:
        return _recover_belief_update_output(raw_text, output_mode)
    if output_mode in {OUTPUT_MODE_COMM_NECESSARY_SOLVER, OUTPUT_MODE_COMM_NECESSARY_BELIEF}:
        return _recover_comm_necessary_output(raw_text, output_mode)
    if output_mode == OUTPUT_MODE_SPARC_SOLVER:
        return _recover_sparc_solver_output(raw_text)
    if output_mode == OUTPUT_MODE_SPARC_MESSAGE:
        return _recover_sparc_message_output(raw_text)
    if output_mode in {OUTPUT_MODE_SPARC_AUDIT, OUTPUT_MODE_CUE_AUDIT}:
        return _recover_audit_output(raw_text, output_mode)
    if output_mode == OUTPUT_MODE_CUE_SOLVER:
        return _recover_cue_solver_output(raw_text)
    return None


def _recover_core_output(raw_text: str) -> dict[str, Any] | None:
    final_answer = _extract_json_answer_field(raw_text, "final_answer")
    reasoning = _extract_json_string_field(raw_text, "reasoning")
    if not final_answer:
        final_answer = _extract_answer_phrase_from_reasoning(raw_text)
    if not final_answer:
        final_answer = _extract_answer_guess_from_text(raw_text)
    if not final_answer:
        return None
    payload: dict[str, Any] = {"final_answer": final_answer}
    if reasoning:
        payload["reasoning"] = reasoning
    return validate_structured_output(_encode_recovered_payload(payload), OUTPUT_MODE_CORE)


def _recover_budget_solver_output(raw_text: str) -> dict[str, Any] | None:
    final_answer = _extract_json_answer_field(raw_text, "final_answer")
    if not final_answer:
        return None
    confidence_raw = _extract_json_number_field(raw_text, "confidence_raw")
    if confidence_raw is None:
        confidence_raw = _extract_json_string_field(raw_text, "confidence_raw")
    if confidence_raw is None:
        confidence_raw = 0.5
    claim_span = _extract_json_string_field(raw_text, "claim_span")
    key_evidence = _extract_json_string_field(raw_text, "key_evidence")
    keyword_clues = _extract_json_string_list_field(raw_text, "keyword_clues")
    if not keyword_clues:
        fallback_clue = claim_span or key_evidence or final_answer
        keyword_clues = [fallback_clue]
    payload = {
        "final_answer": final_answer,
        "reasoning_trace": _extract_json_string_field(raw_text, "reasoning_trace")
        or _extract_json_string_field(raw_text, "reasoning_sketch")
        or _extract_json_string_field(raw_text, "reasoning")
        or final_answer,
        "claim_span": claim_span,
        "key_evidence": key_evidence,
        "keyword_clues": keyword_clues,
        "confidence_raw": confidence_raw,
        "uncertain_point": _extract_json_string_field(raw_text, "uncertain_point"),
    }
    return validate_structured_output(_encode_recovered_payload(payload), OUTPUT_MODE_BUDGET_SOLVER)


def _recover_comm_necessary_output(raw_text: str, output_mode: OutputMode) -> dict[str, Any] | None:
    final_answer = _extract_json_answer_field(raw_text, "final_answer") or _extract_json_answer_field(raw_text, "new_answer")
    if not final_answer:
        return None
    payload = {
        "final_answer": final_answer,
        "reasoning_trace": _extract_json_string_field(raw_text, "reasoning_trace")
        or _extract_json_string_field(raw_text, "reasoning"),
        "evidence_summary": _extract_json_string_field(raw_text, "evidence_summary")
        or _extract_json_string_field(raw_text, "key_evidence"),
        "supporting_facts": _extract_json_array_field(raw_text, "supporting_facts") or [],
        "confidence_raw": _extract_json_number_field(raw_text, "confidence_raw")
        or _extract_json_string_field(raw_text, "confidence_raw"),
    }
    if output_mode == OUTPUT_MODE_COMM_NECESSARY_BELIEF:
        payload["changed_answer"] = _extract_json_bool_field(raw_text, "changed_answer") or False
    return validate_structured_output(_encode_recovered_payload(payload), output_mode)


def _recover_belief_update_output(raw_text: str, output_mode: OutputMode) -> dict[str, Any] | None:
    changed_answer = _extract_json_bool_field(raw_text, "changed_answer")
    if changed_answer is None:
        return None
    new_answer = _extract_json_answer_field(raw_text, "new_answer")
    if changed_answer and new_answer is None:
        return None
    payload = {
        "changed_answer": changed_answer,
        "new_answer": new_answer,
        "confidence_delta": _extract_json_number_field(raw_text, "confidence_delta"),
        "reason_for_change": _extract_json_string_field(raw_text, "reason_for_change"),
        "remaining_disagreement": _extract_json_string_field(raw_text, "remaining_disagreement"),
    }
    return validate_structured_output(_encode_recovered_payload(payload), output_mode)


def _recover_sparc_solver_output(raw_text: str) -> dict[str, Any] | None:
    final_answer = _extract_json_answer_field(raw_text, "final_answer")
    if not final_answer:
        return None
    payload = {
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
    return validate_structured_output(_encode_recovered_payload(payload), OUTPUT_MODE_SPARC_SOLVER)


def _recover_sparc_message_output(raw_text: str) -> dict[str, Any] | None:
    final_answer = _extract_json_answer_field(raw_text, "final_answer")
    if not final_answer:
        return None
    payload = {
        "final_answer": final_answer,
        "reasoning_trace": _extract_json_string_field(raw_text, "reasoning_trace")
        or _extract_json_string_field(raw_text, "reasoning"),
        "confidence_raw": _extract_json_number_field(raw_text, "confidence_raw")
        or _extract_json_string_field(raw_text, "confidence_raw"),
        "claim_span": _extract_json_string_field(raw_text, "claim_span"),
        "key_evidence": _extract_json_string_field(raw_text, "key_evidence"),
    }
    return validate_structured_output(_encode_recovered_payload(payload), OUTPUT_MODE_SPARC_MESSAGE)


def _recover_audit_output(raw_text: str, output_mode: OutputMode) -> dict[str, Any] | None:
    decision = _extract_json_string_field(raw_text, "decision")
    if not decision:
        return None
    payload = {
        "decision": decision,
        "verified_answer": _extract_json_answer_field(raw_text, "verified_answer"),
        "rationale": _extract_json_string_field(raw_text, "rationale") or "Recovered partial rationale.",
    }
    return validate_structured_output(_encode_recovered_payload(payload), output_mode)


def _recover_cue_solver_output(raw_text: str) -> dict[str, Any] | None:
    final_answer = _extract_json_answer_field(raw_text, "final_answer")
    if not final_answer:
        return None
    confidence = _extract_json_number_field(raw_text, "confidence")
    if confidence is None:
        confidence = _extract_json_string_field(raw_text, "confidence")
    payload = {
        "final_answer": final_answer,
        "confidence": confidence,
        "reasoning_sketch": _extract_json_string_field(raw_text, "reasoning_sketch") or final_answer,
        "uncertain_point": _extract_json_string_field(raw_text, "uncertain_point"),
        "top_claims": _extract_json_string_list_field(raw_text, "top_claims"),
        "evidence_items": _extract_json_string_list_field(raw_text, "evidence_items"),
        "counter_answer": _extract_json_answer_field(raw_text, "counter_answer"),
    }
    return validate_structured_output(_encode_recovered_payload(payload), OUTPUT_MODE_CUE_SOLVER)


def _encode_recovered_payload(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False)


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


def _require_non_empty_string(value: object, field_name: str) -> str:
    """要求字段是非空字符串。"""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


def _require_answer_value(value: object, field_name: str) -> str:
    """要求答案字段是非空字符串或数值，并统一转成字符串。"""
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{field_name} must be a string or numeric value.")
    if isinstance(value, (int, float)):
        return str(value)
    return _require_non_empty_string(value, field_name)


def _optional_answer_value(value: object, field_name: str) -> str | None:
    """读取可选答案字段；空串按缺失处理。"""
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return _require_answer_value(value, field_name)


def _require_nullable_string(value: object, field_name: str) -> str | None:
    """要求字段为字符串或 `null`，且出现时不能为空串。"""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string or null.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty when present.")
    return normalized


def _require_nullable_hint(value: object, field_name: str) -> str | None:
    """读取可选提示字段，并把常见“空语义”文本折叠为 `None`。"""
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
    """要求 `confidence_raw` 是数值或可解析为数值的字符串。"""
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


def _require_uncertainty_type(value: object) -> str:
    """要求 `uncertainty_type` 落在固定枚举内。"""
    normalized = _require_non_empty_string(value, "uncertainty_type").lower()
    if normalized not in UNCERTAINTY_TYPE_CHOICES:
        raise ValueError(f"uncertainty_type must be one of {sorted(UNCERTAINTY_TYPE_CHOICES)}.")
    return normalized


def _normalize_uncertainty_type_candidate(value: object) -> str:
    """宽松归一化 uncertainty type 文本。"""
    if value is None:
        return "other"
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in UNCERTAINTY_TYPE_CHOICES:
        return normalized
    return "other"


def _require_keyword_clues(value: object) -> list[str]:
    """要求 `keyword_clues` 至少包含一个非空线索。"""
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise ValueError("keyword_clues must be non-empty.")
        return [normalized]
    if not isinstance(value, list):
        raise ValueError("keyword_clues must be a string or a list of strings.")
    normalized_items: list[str] = []
    for item in value:
        normalized_items.append(_require_non_empty_string(item, "keyword_clues[]"))
    if not normalized_items:
        raise ValueError("keyword_clues must contain at least one clue.")
    return normalized_items


def _optional_confidence_raw(value: object) -> float | str | None:
    """读取可选的 `confidence_raw`；空串按缺失处理。"""
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return None
        if normalized.lower() in {"na", "n/a", "none", "null"}:
            return None
    return _require_confidence_raw(value)


def _extract_labeled_value(text: str, labels: list[str]) -> str | None:
    """提取 `LABEL: value` 形式的短字段。"""
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


def _looks_like_placeholder_text(text: str) -> bool:
    import re

    trimmed = text.strip()
    if not trimmed or "[" not in trimmed or "]" not in trimmed:
        return False
    return re.fullmatch(r"[\[\]\(\)\s,\d.\-]+", trimmed) is not None


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


def _extract_selective_final_answer(text: str, dataset: str) -> str | None:
    """按数据集类型从文本中恢复答案。"""
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
        if match:
            return match.group(1)
        return None

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
    """生成最小 claim span。"""
    if final_answer:
        return final_answer.strip()
    return ""


def _infer_reasoning(text: str, final_answer: str) -> str:
    """生成最小 reasoning 摘要。"""
    trimmed = " ".join(text.split())
    if not trimmed:
        return final_answer
    if len(trimmed) <= 160:
        return trimmed
    return trimmed[:157] + "..."


def _default_uncertainty_type(dataset: str | None) -> str:
    """为缺省场景提供稳定 uncertainty 类型。"""
    if dataset in {"gsm8k", "gsm_symbolic", "math500"}:
        return "calculation"
    if dataset == "hotpotqa":
        return "multi_hop"
    if dataset == "strategyqa":
        return "commonsense_gap"
    return "other"


def _clean_extracted_value(value: str) -> str:
    """清理从标签行中提取出的值。"""
    normalized = value.strip().strip("`")
    normalized = normalized.strip("\"'")
    normalized = normalized.rstrip(")}]")
    normalized = normalized.rstrip(".")
    return normalized.strip()


def _looks_like_label_placeholder(value: str) -> bool:
    """识别提示词模板中的占位符，避免把示例字段当成真实输出。"""
    normalized = value.strip()
    if not normalized:
        return True
    if normalized.startswith("<") and normalized.endswith(">"):
        return True
    lowered = normalized.lower()
    return lowered in {
        "answer",
        "short supporting span",
        "one short sentence",
        "0-1 number or na",
    }


def _clip_selective_field(value: str, max_chars: int) -> str:
    """限制进入 selective 协议字段的文本长度。"""
    trimmed = " ".join(str(value).split())
    if len(trimmed) <= max_chars:
        return trimmed
    return trimmed[: max_chars - 3] + "..."


def _optional_float(value: object, field_name: str) -> float | None:
    """读取可选浮点字段。"""
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


def _require_short_string_list(value: object, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        normalized = value.strip()
        return [normalized] if normalized else []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a string, list of strings, or null.")
    normalized_items: list[str] = []
    for item in value:
        normalized = _require_non_empty_string(item, f"{field_name}[]")
        normalized_items.append(normalized)
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


def _recover_soft_rejection_output(
    *,
    assistant_text: str,
    provider_reasoning_text: str | None,
    output_mode: OutputMode,
    dataset: str | None,
) -> dict[str, Any] | None:
    candidate_texts = [str(assistant_text or ""), str(provider_reasoning_text or "")]
    if not any(_looks_like_soft_rejection_text(text) for text in candidate_texts):
        return None
    if output_mode == OUTPUT_MODE_CORE:
        return {"final_answer": "unknown", "reasoning": "provider_soft_rejection"}
    if output_mode == OUTPUT_MODE_SELECTIVE_COMM:
        return {
            "final_answer": "unknown",
            "confidence_raw": None,
            "reasoning": "provider_soft_rejection",
            "claim_span": "provider_soft_rejection",
            "uncertainty_type": _default_uncertainty_type(dataset),
            "key_evidence": "provider_soft_rejection",
            "uncertain_point": None,
        }
    if output_mode == OUTPUT_MODE_BUDGET_SOLVER:
        return {
            "final_answer": "unknown",
            "reasoning_trace": "provider_soft_rejection",
            "claim_span": "provider_soft_rejection",
            "key_evidence": "provider_soft_rejection",
            "keyword_clues": ["provider_soft_rejection"],
            "confidence_raw": 0.0,
            "uncertain_point": None,
        }
    if output_mode in {OUTPUT_MODE_BUDGET_BELIEF_UPDATE, OUTPUT_MODE_SPARC_BELIEF_UPDATE, OUTPUT_MODE_CUE_BELIEF_UPDATE}:
        return {
            "changed_answer": False,
            "new_answer": None,
            "confidence_delta": None,
            "reason_for_change": "provider_soft_rejection",
            "remaining_disagreement": None,
        }
    if output_mode == OUTPUT_MODE_SPARC_SOLVER:
        return {
            "final_answer": "unknown",
            "reasoning_trace": "provider_soft_rejection",
            "claim_span": "provider_soft_rejection",
            "confidence_raw": 0.0,
            "uncertain_point": None,
            "key_evidence": "provider_soft_rejection",
        }
    if output_mode == OUTPUT_MODE_SPARC_MESSAGE:
        return {
            "final_answer": "unknown",
            "reasoning_trace": "provider_soft_rejection",
            "confidence_raw": 0.0,
            "claim_span": "provider_soft_rejection",
            "key_evidence": "provider_soft_rejection",
        }
    if output_mode in {OUTPUT_MODE_SPARC_AUDIT, OUTPUT_MODE_CUE_AUDIT}:
        return {
            "decision": "abstain",
            "verified_answer": None,
            "rationale": "provider_soft_rejection",
        }
    if output_mode == OUTPUT_MODE_CUE_SOLVER:
        return {
            "final_answer": "unknown",
            "confidence": 0.0,
            "reasoning_sketch": "provider_soft_rejection",
            "uncertain_point": None,
            "top_claims": ["provider_soft_rejection"],
            "evidence_items": [],
            "counter_answer": None,
        }
    if output_mode == OUTPUT_MODE_COMM_NECESSARY_SOLVER:
        return {
            "final_answer": "unknown",
            "reasoning_trace": "provider_soft_rejection",
            "evidence_summary": "provider_soft_rejection",
            "supporting_facts": [],
            "confidence_raw": 0.0,
        }
    if output_mode == OUTPUT_MODE_COMM_NECESSARY_BELIEF:
        return {
            "changed_answer": False,
            "final_answer": "unknown",
            "reasoning_trace": "provider_soft_rejection",
            "evidence_summary": "provider_soft_rejection",
            "supporting_facts": [],
            "confidence_raw": 0.0,
        }
    return None


def _looks_like_soft_rejection_text(text: str) -> bool:
    normalized = " ".join(str(text or "").split()).lower()
    if not normalized:
        return False
    return normalized in {
        "the request was rejected because it was considered high risk",
        "request rejected because it was considered high risk",
    }


def _require_bool(value: object, field_name: str) -> bool:
    """要求字段是布尔值。"""
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean.")
    return value


def _require_enum(value: object, field_name: str, allowed: set[str]) -> str:
    """要求字段属于给定枚举集合。"""
    normalized = _require_non_empty_string(value, field_name)
    if normalized not in allowed:
        raise ValueError(f"{field_name} must be one of {sorted(allowed)}.")
    return normalized
