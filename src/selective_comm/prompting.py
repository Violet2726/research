"""选择性通信实验提示词构造。"""

from __future__ import annotations

from api_baselines.datasets import DatasetSample


DEFAULT_PROMPT_VERSION = "v1-trigger-json-short-reasoning"


def build_initial_messages(
    sample: DatasetSample,
    agent_id: int,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    """构造 Stage A 独立解题提示。"""
    user_prompt = (
        f"You are agent_{agent_id} in Stage A of a selective-communication reasoning experiment.\n"
        f"{_dataset_instruction(sample, prompt_version)}\n"
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
        {"role": "system", "content": _system_prompt(prompt_version)},
        {"role": "user", "content": user_prompt},
    ]


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
    """构造 Stage B 修订提示。"""
    peer_block = "\n\n".join(
        f"{item['agent']} answer: {item['answer']}\n"
        f"{item['agent']} confidence_raw: {item['confidence_raw']}\n"
        f"{item['agent']} short_reasoning: {item['reasoning']}"
        for item in peer_messages
    ) or "No peer feedback."
    user_prompt = (
        f"You are agent_{agent_id} in Stage B debate round {round_index}.\n"
        f"{_dataset_instruction(sample, prompt_version)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        f"Your previous reasoning: {previous_reasoning}\n"
        f"Your previous final_answer: {previous_answer}\n"
        f"Your previous confidence_raw: {previous_confidence_raw}\n\n"
        f"Peer feedback:\n{peer_block}\n\n"
        f"{_revision_instruction(sample, prompt_version)}\n"
        "Return exactly one JSON object with keys reasoning, final_answer, confidence_raw, "
        "uncertain_point, and key_evidence."
    )
    return [
        {"role": "system", "content": _system_prompt(prompt_version)},
        {"role": "user", "content": user_prompt},
    ]


def _system_prompt(prompt_version: str) -> str:
    """返回系统提示词。"""
    if prompt_version == "v1-trigger-json-short-reasoning":
        return (
            "You are one reasoning agent in a trigger and early-exit experiment.\n"
            "Always return strict JSON.\n"
            "Keep reasoning concise.\n"
            "Do not use markdown fences.\n"
            "Do not add extra keys.\n"
            "When you provide confidence_raw, prefer a numeric value in [0, 1]."
        )
    return (
        "You are one reasoning agent in a selective communication experiment.\n"
        "Always return strict JSON with concise reasoning."
    )


def _dataset_instruction(sample: DatasetSample, prompt_version: str) -> str:
    """返回数据集专属约束。"""
    del prompt_version
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


def _revision_instruction(sample: DatasetSample, prompt_version: str) -> str:
    """返回 Stage B 修订规则。"""
    del sample, prompt_version
    return (
        "Revise only if peer arguments reveal a concrete mistake or stronger evidence. "
        "Keep your message short and focused on answer, confidence, and the key reason."
    )
