"""选择性通信实验的提示词构造器。"""

from __future__ import annotations

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.core.prompts.dataset_contracts import (
    build_json_system_prompt,
    build_tagged_lines_system_prompt,
    dataset_instruction_for_sample,
)


DEFAULT_PROMPT_VERSION = "selective_comm_trigger_json"
VOC_PROMPT_VERSION = "selective_comm_voc_json_v2"


def build_initial_messages(
    sample: DatasetSample,
    agent_id: int,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    """按 prompt_version 分派 Stage A 初始消息模板。"""
    if prompt_version == DEFAULT_PROMPT_VERSION:
        return _build_initial_messages(sample, agent_id)
    if prompt_version == VOC_PROMPT_VERSION:
        return _build_initial_messages_v2(sample, agent_id)
    raise ValueError(f"Unsupported selective prompt_version: {prompt_version}")


def build_debate_messages(
    sample: DatasetSample,
    agent_id: int,
    round_index: int,
    previous_reasoning: str,
    previous_answer: str,
    previous_confidence_raw: object,
    peer_messages: list[dict[str, str]],
    previous_claim_span: str | None = None,
    previous_uncertainty_type: str | None = None,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    """按 prompt_version 分派 Stage B 讨论消息模板。"""
    if prompt_version == DEFAULT_PROMPT_VERSION:
        return _build_debate_messages(
            sample,
            agent_id,
            round_index,
            previous_reasoning,
            previous_answer,
            previous_confidence_raw,
            peer_messages,
        )
    if prompt_version == VOC_PROMPT_VERSION:
        return _build_debate_messages_v2(
            sample,
            agent_id,
            round_index,
            previous_reasoning,
            previous_answer,
            previous_confidence_raw,
            previous_claim_span,
            previous_uncertainty_type,
            peer_messages,
        )
    raise ValueError(f"Unsupported selective prompt_version: {prompt_version}")


def _build_initial_messages(sample: DatasetSample, agent_id: int) -> list[dict[str, str]]:
    user_prompt = (
        f"You are agent_{agent_id} in Stage A of a selective-communication reasoning experiment.\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        "Return exactly one JSON object with keys "
        '{"reasoning":"brief reasoning","final_answer":"answer","confidence_raw":0.0,'
        '"uncertain_point":"short point","key_evidence":"short evidence"}.\n'
        "Use concise text. confidence_raw should be a numeric confidence."
    )
    return [
        {"role": "system", "content": _json_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def _build_initial_messages_v2(sample: DatasetSample, agent_id: int) -> list[dict[str, str]]:
    uncertainty_choices = " | ".join(
        [
            "none",
            "calculation",
            "evidence_selection",
            "entity_linking",
            "multi_hop",
            "commonsense_gap",
            "format_extraction",
            "other",
        ]
    )
    user_prompt = (
        f"You are agent_{agent_id} in Stage A of a selective-communication reasoning experiment.\n"
        f"{_dataset_instruction(sample)}\n"
        "Extract a short, comparable black-box proxy of your reasoning state.\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        "Return only the following five lines, in this order, with no markdown fences:\n"
        "FINAL_ANSWER: <answer>\n"
        "CLAIM_SPAN: <short supporting span>\n"
        f"UNCERTAINTY_TYPE: <one of {uncertainty_choices}>\n"
        "CONFIDENCE: <0-1 number or NA>\n"
        "REASON: <one short sentence>\n"
        "Rules:\n"
        "- FINAL_ANSWER must be the final answer only.\n"
        "- CLAIM_SPAN should be short and comparable across agents.\n"
        "- If confidence is unavailable, write NA.\n"
        "- Do not output JSON."
    )
    return [
        {"role": "system", "content": _tagged_lines_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def _build_debate_messages(
    sample: DatasetSample,
    agent_id: int,
    round_index: int,
    previous_reasoning: str,
    previous_answer: str,
    previous_confidence_raw: object,
    peer_messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    peer_block = "\n\n".join(
        f"{item['agent']} answer: {item['answer']}\n"
        f"{item['agent']} confidence_raw: {item['confidence_raw']}\n"
        f"{item['agent']} short_reasoning: {item['reasoning']}"
        for item in peer_messages
    ) or "No peer feedback."
    user_prompt = (
        f"You are agent_{agent_id} in Stage B debate round {round_index}.\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        f"Your previous reasoning: {previous_reasoning}\n"
        f"Your previous final_answer: {previous_answer}\n"
        f"Your previous confidence_raw: {previous_confidence_raw}\n\n"
        f"Peer feedback:\n{peer_block}\n\n"
        f"{_revision_instruction()}\n"
        "Return exactly one JSON object with keys reasoning, final_answer, confidence_raw, "
        "uncertain_point, and key_evidence."
    )
    return [
        {"role": "system", "content": _json_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def _build_debate_messages_v2(
    sample: DatasetSample,
    agent_id: int,
    round_index: int,
    previous_reasoning: str,
    previous_answer: str,
    previous_confidence_raw: object,
    previous_claim_span: str | None,
    previous_uncertainty_type: str | None,
    peer_messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    del previous_reasoning
    uncertainty_choices = " | ".join(
        [
            "none",
            "calculation",
            "evidence_selection",
            "entity_linking",
            "multi_hop",
            "commonsense_gap",
            "format_extraction",
            "other",
        ]
    )
    peer_block = "\n".join(
        f"{item['agent']} answer: {_compact_text(item['answer'], 80)}; "
        f"claim: {_compact_text(item.get('claim_span', ''), 100)}; "
        f"type: {_compact_text(item.get('uncertainty_type', ''), 24)}"
        for item in peer_messages
    ) or "No peer feedback."
    user_prompt = (
        f"You are agent_{agent_id} in Stage B debate round {round_index}.\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    user_prompt += (
        f"Your previous final_answer: {_compact_text(previous_answer, 80)}\n"
        f"Your previous claim_span: {_compact_text(previous_claim_span or '', 100)}\n"
        f"Your previous uncertainty_type: {_compact_text(previous_uncertainty_type or '', 24)}\n"
        f"Your previous confidence_raw: {_compact_text(str(previous_confidence_raw or ''), 24)}\n\n"
        f"Peer feedback:\n{peer_block}\n\n"
        "Use the earlier context only implicitly through these prior claims; do not restate long passages.\n"
        f"{_revision_instruction()}\n"
        "Return only the following five lines, in this order, with no markdown fences:\n"
        "FINAL_ANSWER: <answer>\n"
        "CLAIM_SPAN: <short supporting span>\n"
        f"UNCERTAINTY_TYPE: <one of {uncertainty_choices}>\n"
        "CONFIDENCE: <0-1 number or NA>\n"
        "REASON: <one short sentence>\n"
        "Do not output JSON."
    )
    return [
        {"role": "system", "content": _tagged_lines_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def _json_system_prompt() -> str:
    return build_json_system_prompt(
        "You are one reasoning agent in a trigger and early-exit experiment.",
        extra_rules=[
            "Prefer a short final answer and a short supporting span.",
            "If confidence is unavailable, you may write NA.",
        ],
    )


def _tagged_lines_system_prompt() -> str:
    return build_tagged_lines_system_prompt(
        "You are one reasoning agent in a trigger and early-exit experiment.",
        extra_rules=[
            "Prefer a short final answer and a short supporting span.",
            "If confidence is unavailable, you may write NA.",
        ],
    )


def _dataset_instruction(sample: DatasetSample) -> str:
    return dataset_instruction_for_sample(sample, hotpot_style="shortest_span_copy")


def _revision_instruction() -> str:
    return (
        "Revise only if peer arguments reveal a concrete mistake or stronger evidence. "
        "Keep your message short and focused on answer, confidence, and the key reason."
    )


def _compact_text(value: str, max_chars: int) -> str:
    text = " ".join(str(value).split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


