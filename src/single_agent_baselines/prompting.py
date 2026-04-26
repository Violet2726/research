"""单智能体基线提示词构造。

这里定义仓库中最基础的一类 prompt：单模型、单轮、单个 JSON 输出。
虽然结构简单，但它也是后续多数对照实验的公共参照，因此约束需要保持稳定。
"""

from __future__ import annotations

from experiment_core.datasets import DatasetSample


DEFAULT_PROMPT_VERSION = "single_agent_reasoning_json_v1"


def build_messages(
    sample: DatasetSample,
    method_family: str,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    """构造一次单智能体请求。"""
    del method_family
    if prompt_version != DEFAULT_PROMPT_VERSION:
        raise ValueError(f"Unsupported single-agent prompt_version: {prompt_version}")
    return [
        {"role": "system", "content": _system_prompt(prompt_version)},
        {"role": "user", "content": _user_prompt(sample, prompt_version)},
    ]


def _system_prompt(prompt_version: str) -> str:
    """返回单智能体实验的 system prompt。"""
    if prompt_version == DEFAULT_PROMPT_VERSION:
        return (
            "You are an expert reasoning assistant for controlled research experiments.\n"
            "Follow the task instruction carefully.\n"
            "Return strict JSON with keys reasoning and final_answer.\n"
            "Keep reasoning concise and under 120 tokens.\n"
            "Do not add markdown fences."
        )
    raise ValueError(f"Unsupported single-agent prompt_version: {prompt_version}")


def _user_prompt(sample: DatasetSample, prompt_version: str) -> str:
    """构造数据集相关的 user prompt。"""
    user_prompt = (
        f"{_dataset_instruction(sample, prompt_version)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"

    if prompt_version == DEFAULT_PROMPT_VERSION:
        user_prompt += (
            'Return exactly one JSON object like '
            '{"reasoning":"brief reasoning","final_answer":"answer"}'
        )
        return user_prompt

    raise ValueError(f"Unsupported single-agent prompt_version: {prompt_version}")


def _dataset_instruction(sample: DatasetSample, prompt_version: str) -> str:
    """返回数据集特定的答题约束。"""
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
            'Answer the question with yes or no. The final_answer must be exactly "yes" or "no".'
        )
    if sample.dataset == "hotpotqa":
        return (
            "Answer the multi-hop question using only the provided context. "
            "The final_answer should be a short text span."
        )
    if sample.dataset in {"mmlu_pro", "gpqa_diamond"}:
        return (
            "Choose the single best option. "
            'The final_answer must be only the option letter, such as "A" or "B".'
        )
    raise ValueError(f"Unsupported dataset: {sample.dataset}")
