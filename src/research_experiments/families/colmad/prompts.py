"""ColMAD 提示词与结构化输出解析。"""

from __future__ import annotations

from typing import Any
import json
import re

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.core.data.evaluation import normalize_error_detection_verdict


DEFAULT_PROMPT_VERSION = "colmad_paper_v1"
FAILURE_MODES = ("fake_evidence", "overconfident_claim", "fallacious_argument")


def build_single_agent_messages(sample: DatasetSample) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are a careful error-detection assistant."},
        {
            "role": "user",
            "content": (
                f"{_task_block(sample)}\n\n"
                "Judge whether the candidate response contains an error.\n"
                "Return a JSON object with keys:\n"
                "- `verdict`: `contains_error` or `contains_no_error`\n"
                "- `rationale`: short explanation\n"
                "- `evidence_points`: list of concrete checks\n"
                "- `confidence`: number between 0 and 1\n"
            ),
        },
    ]


def build_debater_opening_messages(
    sample: DatasetSample,
    *,
    debate_protocol: str,
    debater_name: str,
) -> list[dict[str, str]]:
    protocol_instruction = _protocol_instruction(debate_protocol, opening=True)
    return [
        {"role": "system", "content": f"You are {debater_name}, an LLM error-detection debater."},
        {
            "role": "user",
            "content": (
                f"{_task_block(sample)}\n\n"
                f"{protocol_instruction}\n"
                "Return a JSON object with keys:\n"
                "- `verdict`: `contains_error` or `contains_no_error`\n"
                "- `rationale`: short explanation\n"
                "- `evidence_points`: list of concrete checks\n"
                "- `confidence`: number between 0 and 1\n"
            ),
        },
    ]


def build_debater_reply_messages(
    sample: DatasetSample,
    *,
    debate_protocol: str,
    debater_name: str,
    own_opening: dict[str, Any],
    peer_opening: dict[str, Any],
) -> list[dict[str, str]]:
    protocol_instruction = _protocol_instruction(debate_protocol, opening=False)
    peer_block = json.dumps(peer_opening, ensure_ascii=False, indent=2)
    own_block = json.dumps(own_opening, ensure_ascii=False, indent=2)
    return [
        {"role": "system", "content": f"You are {debater_name}, an LLM error-detection debater."},
        {
            "role": "user",
            "content": (
                f"{_task_block(sample)}\n\n"
                f"Your opening analysis:\n{own_block}\n\n"
                f"Peer opening analysis:\n{peer_block}\n\n"
                f"{protocol_instruction}\n"
                "Return a JSON object with keys:\n"
                "- `verdict`: `contains_error` or `contains_no_error`\n"
                "- `rationale`: short explanation\n"
                "- `evidence_points`: list of concrete checks\n"
                "- `confidence`: number between 0 and 1\n"
                "- `observed_failure_modes`: subset of [fake_evidence, overconfident_claim, fallacious_argument]\n"
                "- `supportive_critique`: short text, empty string if not collaborative\n"
                "- `complemented_peer_points`: list of peer blind spots you helped cover\n"
                "- `attack_points`: list of direct attacks, empty if not competitive\n"
            ),
        },
    ]


def build_judge_messages(
    sample: DatasetSample,
    *,
    debate_protocol: str,
    transcript_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    transcript = "\n\n".join(
        [
            f"{row['role']} ({row['turn_stage']}):\n"
            f"{json.dumps(row.get('validated_output') or {}, ensure_ascii=False, indent=2)}"
            for row in transcript_rows
        ]
    )
    return [
        {"role": "system", "content": "You are the final judge in an error-detection debate."},
        {
            "role": "user",
            "content": (
                f"{_task_block(sample)}\n\n"
                f"Debate protocol: {debate_protocol}\n\n"
                f"Transcript:\n{transcript}\n\n"
                "Decide whether the candidate response contains an error.\n"
                "Return a JSON object with keys:\n"
                "- `final_verdict`: `contains_error` or `contains_no_error`\n"
                "- `confidence`: number between 0 and 1\n"
                "- `rationale`: short explanation grounded in the transcript\n"
                "- `observed_failure_modes`: subset of [fake_evidence, overconfident_claim, fallacious_argument]\n"
                "- `supportive_critique_observed`: true or false\n"
                "- `evidence_complementarity_observed`: true or false\n"
            ),
        },
    ]


def validate_detector_output(assistant_text: str, provider_reasoning_text: str) -> dict[str, Any]:
    payload = _parse_payload(assistant_text)
    return {
        "verdict": _normalize_verdict(payload.get("verdict") or assistant_text),
        "rationale": _clean_text(payload.get("rationale") or assistant_text),
        "evidence_points": _normalize_string_list(payload.get("evidence_points")),
        "confidence": _normalize_confidence(payload.get("confidence")),
        "observed_failure_modes": _normalize_failure_modes(payload.get("observed_failure_modes")),
        "supportive_critique": "",
        "complemented_peer_points": [],
        "attack_points": [],
    }


def validate_debater_output(
    assistant_text: str,
    provider_reasoning_text: str,
    *,
    debate_protocol: str,
) -> dict[str, Any]:
    payload = _parse_payload(assistant_text)
    supportive_text = _clean_text(payload.get("supportive_critique"))
    complemented_points = _normalize_string_list(payload.get("complemented_peer_points"))
    attack_points = _normalize_string_list(payload.get("attack_points"))
    if debate_protocol == "competitive":
        supportive_text = ""
        complemented_points = []
    else:
        attack_points = []
    return {
        "verdict": _normalize_verdict(payload.get("verdict") or assistant_text),
        "rationale": _clean_text(payload.get("rationale") or assistant_text),
        "evidence_points": _normalize_string_list(payload.get("evidence_points")),
        "confidence": _normalize_confidence(payload.get("confidence")),
        "observed_failure_modes": _normalize_failure_modes(payload.get("observed_failure_modes")),
        "supportive_critique": supportive_text,
        "complemented_peer_points": complemented_points,
        "attack_points": attack_points,
    }


def validate_judge_output(assistant_text: str, provider_reasoning_text: str) -> dict[str, Any]:
    payload = _parse_payload(assistant_text)
    return {
        "final_verdict": _normalize_verdict(payload.get("final_verdict") or assistant_text),
        "confidence": _normalize_confidence(payload.get("confidence")),
        "rationale": _clean_text(payload.get("rationale") or assistant_text),
        "observed_failure_modes": _normalize_failure_modes(payload.get("observed_failure_modes")),
        "supportive_critique_observed": _normalize_bool(payload.get("supportive_critique_observed")),
        "evidence_complementarity_observed": _normalize_bool(payload.get("evidence_complementarity_observed")),
    }


def _task_block(sample: DatasetSample) -> str:
    candidate_response = str(sample.metadata.get("candidate_response") or "").strip()
    return (
        f"Task name: {sample.metadata.get('task_name')}\n"
        f"Candidate response model: {sample.metadata.get('candidate_response_model')}\n\n"
        f"Task input:\n{sample.question}\n\n"
        f"Candidate response:\n{candidate_response}"
    )


def _protocol_instruction(debate_protocol: str, *, opening: bool) -> str:
    if debate_protocol == "competitive":
        if opening:
            return "You are in a competitive debate. Defend your own verdict and prepare to win the judge."
        return "You are in a competitive debate. Attack weak reasoning from your peer and persuade the judge to adopt your verdict."
    if opening:
        return "You are in a collaborative debate. State your verdict clearly and surface the strongest available evidence."
    return "You are in a collaborative debate. Supportively criticize your peer, fill missing evidence, and help the judge reach a better verdict."


def _parse_payload(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    candidate = match.group(0)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return {}


def _normalize_verdict(value: Any) -> str:
    normalized = normalize_error_detection_verdict(str(value or ""))
    if normalized == "contains_error":
        return "contains_error"
    if normalized == "contains_no_error":
        return "contains_no_error"
    return "contains_error" if "error" in normalized and "no" not in normalized else "contains_no_error"


def _normalize_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return items
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _normalize_failure_modes(value: Any) -> list[str]:
    normalized: list[str] = []
    for item in _normalize_string_list(value):
        text = item.strip().lower().replace("-", "_").replace(" ", "_")
        for failure_mode in FAILURE_MODES:
            if failure_mode in text:
                normalized.append(failure_mode)
                break
    return sorted(set(normalized))


def _normalize_confidence(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, round(numeric, 6)))


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"true", "1", "yes", "y"}


def _clean_text(value: Any) -> str:
    text = str(value or "").strip()
    return text or ""

