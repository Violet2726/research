"""Selective-communication prompting."""

from __future__ import annotations

from experiment_core.datasets import DatasetSample


DEFAULT_PROMPT_VERSION = "selective_comm_trigger_json"


def build_initial_messages(
    sample: DatasetSample,
    agent_id: int,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    if prompt_version != DEFAULT_PROMPT_VERSION:
        raise ValueError(f"Unsupported selective prompt_version: {prompt_version}")
    return _build_initial_messages(sample, agent_id)


def build_debate_messages(
    sample: DatasetSample,
    agent_id: int,
    round_index: int,
    previous_reasoning: str,
    previous_answer: str,
    previous_confidence_raw: object,
    peer_messages: list[dict[str, str]],
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    if prompt_version != DEFAULT_PROMPT_VERSION:
        raise ValueError(f"Unsupported selective prompt_version: {prompt_version}")
    return _build_debate_messages(
        sample,
        agent_id,
        round_index,
        previous_reasoning,
        previous_answer,
        previous_confidence_raw,
        peer_messages,
    )


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


def _system_prompt() -> str:
    return (
        "You are one reasoning agent in a trigger and early-exit experiment.\n"
        "Always return strict JSON.\n"
        "Keep reasoning concise.\n"
        "Do not use markdown fences.\n"
        "Do not add extra keys.\n"
        "When you provide confidence_raw, prefer a numeric value in [0, 1]."
    )


def _dataset_instruction(sample: DatasetSample) -> str:
    if sample.dataset == "gsm8k":
        return (
            "Solve the math word problem carefully. "
            "The final_answer must be only the final numeric answer without commas or units."
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
    raise ValueError(f"Unsupported dataset: {sample.dataset}")


def _revision_instruction() -> str:
    return (
        "Revise only if peer arguments reveal a concrete mistake or stronger evidence. "
        "Keep your message short and focused on answer, confidence, and the key reason."
    )
