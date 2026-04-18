"""多智能体提示词构造。"""

from __future__ import annotations

from api_baselines.datasets import DatasetSample


SYSTEM_PROMPT = """You are one reasoning agent in a controlled multi-agent debate experiment.
Solve the task carefully.
Always return strict JSON with keys reasoning and final_answer.
Keep reasoning concise and avoid markdown fences."""


def build_initial_messages(sample: DatasetSample, agent_id: int) -> list[dict[str, str]]:
    """构造 agent 的首轮独立求解提示。"""
    user_prompt = (
        f"You are agent_{agent_id}.\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        'Return exactly one JSON object like '
        '{"reasoning":"brief reasoning","final_answer":"answer"}'
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_debate_messages(
    sample: DatasetSample,
    agent_id: int,
    round_index: int,
    previous_reasoning: str,
    previous_answer: str,
    peer_messages: list[dict[str, str]],
) -> list[dict[str, str]]:
    """构造某一轮 debate 的修订提示。"""
    peer_block = "\n\n".join(
        f"{item['agent']} previous answer: {item['answer']}\n"
        f"{item['agent']} reasoning: {item['reasoning']}"
        for item in peer_messages
    ) or "No peer feedback."
    user_prompt = (
        f"You are agent_{agent_id} in debate round {round_index}.\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        f"Your previous reasoning: {previous_reasoning}\n"
        f"Your previous answer: {previous_answer}\n\n"
        f"Peer feedback:\n{peer_block}\n\n"
        "Revise your reasoning only if peer arguments reveal a concrete mistake or stronger evidence. "
        "Return strict JSON with keys reasoning and final_answer."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _dataset_instruction(sample: DatasetSample) -> str:
    """返回数据集专属回答约束。"""
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
            "The final_answer should be a short text span."
        )
    raise ValueError(f"Unsupported dataset: {sample.dataset}")
