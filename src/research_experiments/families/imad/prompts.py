"""iMAD 实验使用的提示词构造器。"""

from __future__ import annotations

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.core.prompts.dataset_contracts import build_json_system_prompt, dataset_instruction_for_sample


DEFAULT_PROMPT_VERSION = "imad_controlled_json"


def build_initial_messages(
    sample: DatasetSample,
    agent_id: int,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    """构造 iMAD 的初始独立作答消息。"""

    user_prompt = (
        f"You are agent_{agent_id}.\n"
        f"{_dataset_instruction(sample)}\n"
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
    """构造 iMAD 的第 N 轮 debate 消息。"""

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
        f"Your previous final_answer: {previous_answer}\n\n"
        f"Peer feedback:\n{peer_block}\n\n"
        "Revise your answer only if peer arguments reveal a concrete mistake or stronger evidence. "
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
    if prompt_version != DEFAULT_PROMPT_VERSION:
        raise ValueError(f"Unsupported iMAD prompt_version: {prompt_version}")
    return build_json_system_prompt(
        "You are one reasoning agent in an adaptive multi-agent debate experiment.",
        extra_rules=[
            "Solve the task carefully using only the provided question and context.",
            "Keep optional reasoning compact and outcome-focused.",
            "Do not add natural-language text before or after the JSON object.",
            "Do not add labels, category words, or explanatory suffixes to final_answer.",
        ],
    )


def _dataset_instruction(sample: DatasetSample) -> str:
    if sample.dataset == "hotpotqa":
        return dataset_instruction_for_sample(sample, hotpot_style="shortest_span_copy")
    return dataset_instruction_for_sample(sample)

