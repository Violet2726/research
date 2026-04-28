"""选择性通信实验提示词构造。

本模块负责构造共享前缀实验中的 Stage A 与 Stage B prompt。
当前主协议采用“标签行输出”，避免数学模型在 JSON 模式下漂移成大段自由文本。
"""

from __future__ import annotations

from experiment_core.datasets import DatasetSample


DEFAULT_PROMPT_VERSION = "selective_comm_trigger_json"
VOC_PROMPT_VERSION = "selective_comm_voc_json_v2"


def build_initial_messages(
    sample: DatasetSample,
    agent_id: int,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
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
        {"role": "system", "content": _system_prompt()},
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
        {"role": "system", "content": _system_prompt()},
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
        {"role": "system", "content": _system_prompt()},
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
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def _system_prompt() -> str:
    return (
        "You are one reasoning agent in a trigger and early-exit experiment.\n"
        "Return compact tagged lines only.\n"
        "Do not use markdown fences.\n"
        "Prefer a short final answer and a short supporting span.\n"
        "If confidence is unavailable, you may write NA."
    )


def _dataset_instruction(sample: DatasetSample) -> str:
    if sample.dataset == "gsm8k":
        return (
            "Solve the math word problem carefully. "
            "The final_answer must be only the final numeric answer without commas or units."
        )
    if sample.dataset == "gsm_symbolic":
        return (
            "Solve the math problem carefully. "
            "The final_answer must be only the final numeric answer without commas or units."
        )
    if sample.dataset == "math500":
        return (
            "Solve the math problem carefully. "
            "The final_answer must be only the final mathematical expression, with no explanation."
        )
    if sample.dataset == "strategyqa":
        return (
            'Answer with exactly "yes" or "no". '
            'The final_answer must be exactly "yes" or "no".'
        )
    if sample.dataset == "hotpotqa":
        return (
            "Answer the multi-hop question using only the provided context. "
            "The final_answer must be the shortest judgeable text span. "
            "Prefer exact wording from the context when possible."
        )
    if sample.dataset in {"mmlu_pro", "gpqa_diamond"}:
        return (
            "Choose the single best option. "
            'The final_answer must be only the option letter, such as "A" or "B".'
        )
    raise ValueError(f"Unsupported dataset: {sample.dataset}")


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
