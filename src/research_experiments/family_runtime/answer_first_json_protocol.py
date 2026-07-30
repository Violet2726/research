"""D4 严格 answer-first JSON 输出协议（strict answer-first JSON protocol）。

The protocol deliberately rejects recovery heuristics.  A response is usable
only when it is one complete JSON object, has the two keys in the frozen
answer-first order, contains no duplicate keys, and canonicalizes to one
answer under the sample's answer contract.
"""

from __future__ import annotations

import json
from typing import Any

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.core.data.evaluation import canonicalize_answer

ANSWER_FIRST_JSON_PROTOCOL_V1 = "answer_first_json_v1"
ANSWER_FIRST_JSON_PROMPT_V1 = "single_agent_answer_first_json_v1"
ANSWER_FIRST_JSON_PROMPT_V2 = "single_agent_answer_first_json_v2"
ANSWER_FIRST_JSON_PROMPT_VERSION = ANSWER_FIRST_JSON_PROMPT_V1
REASONING_FIRST_JSON_PROTOCOL_V1 = "reasoning_first_json_v1"


def build_answer_first_json_instruction(
    dataset: str,
    *,
    prompt_version: str = ANSWER_FIRST_JSON_PROMPT_VERSION,
) -> str:
    lines = [
        "Return exactly one JSON object and no other text.",
        'The first key must be "final_answer" and the second key must be "reasoning".',
        'Use exactly this shape: {"final_answer":"canonical answer","reasoning":"concise verification"}',
        "Do not repeat either key. Both values must be non-empty JSON strings.",
    ]
    if prompt_version == ANSWER_FIRST_JSON_PROMPT_V2:
        lines.append(
            "reasoning must be plain text of at most 80 words; do not use Markdown, LaTeX, or backslashes."
        )
    elif prompt_version != ANSWER_FIRST_JSON_PROMPT_VERSION:
        raise ValueError(f"Unsupported answer-first JSON prompt version: {prompt_version!r}")
    if dataset in {"gpqa_diamond", "musr", "musr_x", "supergpqa", "supergpqa_science"}:
        lines.append('final_answer must be exactly one option letter such as "A" or "B".')
    elif dataset in {"bbeh", "bbeh_extension"}:
        lines.append("final_answer must contain only the exact answer requested by the task.")
    return "\n".join(lines)


def parse_answer_first_json_output(sample: DatasetSample, raw_text: str) -> dict[str, Any]:
    return _parse_two_key_json(sample, raw_text, expected_keys=["final_answer", "reasoning"])


def parse_reasoning_first_json_output(sample: DatasetSample, raw_text: str) -> dict[str, Any]:
    return _parse_two_key_json(sample, raw_text, expected_keys=["reasoning", "final_answer"])


def _parse_two_key_json(
    sample: DatasetSample,
    raw_text: str,
    *,
    expected_keys: list[str],
) -> dict[str, Any]:
    cleaned = str(raw_text or "").strip()
    if not cleaned:
        raise ValueError("empty_assistant_output")
    if cleaned.startswith("```") or not cleaned.startswith("{") or not cleaned.endswith("}"):
        raise ValueError("answer_first_json_not_one_bare_object")

    pairs: list[tuple[str, Any]] = []

    def _pairs_hook(items: list[tuple[str, Any]]) -> dict[str, Any]:
        pairs.extend(items)
        return dict(items)

    try:
        payload = json.loads(cleaned, object_pairs_hook=_pairs_hook)
    except json.JSONDecodeError as exc:
        raise ValueError("answer_first_json_invalid_or_truncated") from exc
    if not isinstance(payload, dict):
        raise ValueError("answer_first_json_not_object")
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("answer_first_json_duplicate_key")
    if keys != expected_keys or set(payload) != {"final_answer", "reasoning"}:
        raise ValueError("two_key_json_keys_or_order_invalid")
    final_answer = payload["final_answer"]
    reasoning = payload["reasoning"]
    if not isinstance(final_answer, str) or not final_answer.strip():
        raise ValueError("answer_first_json_final_answer_invalid")
    if not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("answer_first_json_reasoning_invalid")

    canonical = canonicalize_answer(sample, final_answer.strip())
    if not canonical.valid or not canonical.key:
        raise ValueError(canonical.invalid_reason or "answer_first_json_answer_contract_invalid")
    return {
        "final_answer": final_answer.strip(),
        "reasoning": reasoning.strip(),
        "canonical_answer_key": canonical.key,
        "canonicalization_valid": True,
        "canonicalization_invalid_reason": None,
        "canonical_key": canonical.key,
        "canonical_valid": True,
        "canonical_invalid_reason": None,
    }
