"""多智能体辩论实验的提示词构造器。"""

from __future__ import annotations

from research_experiments.core.foundation.datasets import DatasetSample
from research_experiments.core.foundation.prompt_contracts import build_json_system_prompt, dataset_instruction_for_sample


DEFAULT_PROMPT_VERSION = "multi_agent_debate_json"
CONTROLLED_PROMPT_VERSION = "multi_agent_controlled_json"


def build_initial_messages(
    sample: DatasetSample,
    agent_id: int,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    """构造多智能体实验的初始独立作答消息。"""
    user_prompt = (
        f"You are agent_{agent_id}.\n"
        f"{_dataset_instruction(sample, prompt_version)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        'Return exactly one JSON object with key "final_answer". '
        'You may optionally include "reasoning". '
        'If you include reasoning, keep it under 60 words. '
        "Do not add any other keys. "
        "Return JSON only."
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
    peer_messages: list[dict[str, str]],
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    """构造辩论轮次中的复审消息。"""
    peer_block = "\n\n".join(
        f"{item['agent']} previous answer: {item['answer']}\n"
        f"{item['agent']} reasoning: {item['reasoning']}"
        for item in peer_messages
    ) or "No peer feedback."
    user_prompt = (
        f"You are agent_{agent_id} in debate round {round_index}.\n"
        f"{_dataset_instruction(sample, prompt_version)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        f"Your previous reasoning: {previous_reasoning}\n"
        f"Your previous final_answer: {previous_answer}\n\n"
        f"Peer feedback:\n{peer_block}\n\n"
        f"{_revision_instruction(sample, prompt_version)} "
        'Return exactly one JSON object with key "final_answer". '
        'You may optionally include "reasoning". '
        'If you include reasoning, keep it under 60 words. '
        "Do not add any other keys. "
        "Return JSON only."
    )
    return [
        {"role": "system", "content": _system_prompt(prompt_version)},
        {"role": "user", "content": user_prompt},
    ]


def _system_prompt(prompt_version: str) -> str:
    if prompt_version == CONTROLLED_PROMPT_VERSION:
        return build_json_system_prompt(
            "You are one reasoning agent in a controlled debate-vs-vote experiment.",
            extra_rules=[
                "Solve the task carefully using only the provided question and context.",
                "Keep optional reasoning compact and outcome-focused.",
                "Do not add natural-language text before or after the JSON object.",
                "Do not add labels, category words, or explanatory suffixes to final_answer.",
            ],
        )
    if prompt_version != DEFAULT_PROMPT_VERSION:
        raise ValueError(f"Unsupported multi-agent prompt_version: {prompt_version}")
    return build_json_system_prompt(
        "You are one reasoning agent in a controlled multi-agent debate experiment.",
        extra_rules=[
            "Solve the task carefully.",
            "Keep optional reasoning compact and outcome-focused.",
            "Do not add natural-language text before or after the JSON object.",
        ],
    )


def _dataset_instruction(sample: DatasetSample, prompt_version: str) -> str:
    if sample.dataset == "hotpotqa":
        if prompt_version == CONTROLLED_PROMPT_VERSION:
            return dataset_instruction_for_sample(sample, hotpot_style="shortest_span_copy")
        return dataset_instruction_for_sample(sample, hotpot_style="short_span")
    return dataset_instruction_for_sample(sample)


def _revision_instruction(sample: DatasetSample, prompt_version: str) -> str:
    if sample.dataset == "hotpotqa" and prompt_version == CONTROLLED_PROMPT_VERSION:
        return (
            "Revise your answer only if peer arguments reveal a concrete mistake or provide stronger textual evidence. "
            "If the peer answer differs only by added labels, category words, or formatting, prefer the shortest "
            "context-grounded span."
        )
    return "Revise your reasoning only if peer arguments reveal a concrete mistake or stronger evidence."

