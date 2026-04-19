"""多智能体实验提示词构造。

当前支持两套稳定 prompt 版本：

- ``v1-mad-json-short-reasoning``：兼容 Vanilla MAD 复现实验。
- ``v2-controlled-span-strict``：用于 Debate vs Vote 对照分析，强化
  HotpotQA 的最短可判别 span 约束。
"""

from __future__ import annotations

from experiment_core.datasets import DatasetSample


DEFAULT_PROMPT_VERSION = "v1-mad-json-short-reasoning"


def build_initial_messages(
    sample: DatasetSample,
    agent_id: int,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    """构造 agent 首轮独立求解消息。"""
    user_prompt = (
        f"You are agent_{agent_id}.\n"
        f"{_dataset_instruction(sample, prompt_version)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        'Return exactly one JSON object like '
        '{"reasoning":"brief reasoning","final_answer":"answer"}'
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
    """构造某一轮 debate 的修订消息。"""
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
        f"Your previous answer: {previous_answer}\n\n"
        f"Peer feedback:\n{peer_block}\n\n"
        f"{_revision_instruction(sample, prompt_version)} "
        "Return strict JSON with keys reasoning and final_answer."
    )
    return [
        {"role": "system", "content": _system_prompt(prompt_version)},
        {"role": "user", "content": user_prompt},
    ]


def _system_prompt(prompt_version: str) -> str:
    """返回当前 prompt 版本对应的 system prompt。"""
    if prompt_version == "v2-controlled-span-strict":
        return (
            "You are one reasoning agent in a controlled debate-vs-vote experiment.\n"
            "Solve the task carefully using only the provided question and context.\n"
            "Always return strict JSON with keys reasoning and final_answer.\n"
            "Keep reasoning concise.\n"
            "Do not use markdown fences.\n"
            "Do not add labels, category words, or explanatory suffixes to final_answer."
        )
    return (
        "You are one reasoning agent in a controlled multi-agent debate experiment.\n"
        "Solve the task carefully.\n"
        "Always return strict JSON with keys reasoning and final_answer.\n"
        "Keep reasoning concise and avoid markdown fences."
    )


def _dataset_instruction(sample: DatasetSample, prompt_version: str) -> str:
    """返回数据集专属答题约束。"""
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
        if prompt_version == "v2-controlled-span-strict":
            return (
                "Answer the multi-hop question using only the provided context. "
                "The final_answer must be the shortest judgeable text span. "
                "Prefer copying the exact wording from the context when possible. "
                "Do not add category words, parentheses, explanations, or extra qualifiers."
            )
        return (
            "Answer the multi-hop question using only the provided context. "
            "The final_answer should be a short text span."
        )
    raise ValueError(f"Unsupported dataset: {sample.dataset}")


def _revision_instruction(sample: DatasetSample, prompt_version: str) -> str:
    """返回 debate 轮的修订规则。"""
    if sample.dataset == "hotpotqa" and prompt_version == "v2-controlled-span-strict":
        return (
            "Revise your answer only if peer arguments reveal a concrete mistake or provide stronger textual evidence. "
            "If the peer answer differs only by added labels, category words, or formatting, prefer the shortest "
            "context-grounded span."
        )
    return "Revise your reasoning only if peer arguments reveal a concrete mistake or stronger evidence."
