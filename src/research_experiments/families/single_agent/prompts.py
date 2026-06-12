"""单智能体基线实验的 prompt 构造器。"""

from __future__ import annotations

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.core.prompts.dataset_contracts import dataset_instruction_for_sample
from research_experiments.family_runtime.free_text_protocol import (
    build_free_text_answer_instruction,
    build_free_text_system_prompt,
)
from research_experiments.family_runtime.reasoning_methods import resolve_reasoning_method

FREE_TEXT_V1_PROMPT_VERSION = "single_agent_free_text_v1"
DEFAULT_PROMPT_VERSION = FREE_TEXT_V1_PROMPT_VERSION
SUPPORTED_PROMPT_VERSIONS = (FREE_TEXT_V1_PROMPT_VERSION,)


def build_messages(
    sample: DatasetSample,
    method_family: str,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    _ensure_prompt_version(prompt_version)
    return [
        {"role": "system", "content": _system_prompt(prompt_version)},
        {"role": "user", "content": _user_prompt(sample, method_family, prompt_version)},
    ]


def _ensure_prompt_version(prompt_version: str) -> None:
    if prompt_version not in SUPPORTED_PROMPT_VERSIONS:
        raise ValueError(f"Unsupported single-agent prompt_version: {prompt_version}")


def _system_prompt(prompt_version: str) -> str:
    _ensure_prompt_version(prompt_version)
    return build_free_text_system_prompt(
        "You are an expert reasoning assistant for controlled research experiments.",
        extra_rules=[
            "Follow the task instruction carefully.",
            "Output the labels in exactly this order: REASONING, FINAL_ANSWER.",
            "REASONING is required.",
            "Keep REASONING concise, but include enough detail to justify or revise the answer.",
            "If your reasoning changes the answer, rewrite FINAL_ANSWER to the corrected answer.",
        ],
    )


def _user_prompt(sample: DatasetSample, method_family: str, prompt_version: str) -> str:
    reasoning_spec = resolve_reasoning_method(sample.dataset, _base_reasoning_method(method_family))
    user_prompt = (
        f"Reasoning method: {reasoning_spec.label}\n"
        f"Method summary: {reasoning_spec.summary}\n"
        f"Method guidance: {reasoning_spec.guidance}\n"
        f"Method checklist: {reasoning_spec.checklist}\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"

    if prompt_version != FREE_TEXT_V1_PROMPT_VERSION:
        raise ValueError(f"Unsupported single-agent prompt_version: {prompt_version}")
    user_prompt += build_free_text_answer_instruction(sample.dataset)
    return user_prompt


def _dataset_instruction(sample: DatasetSample) -> str:
    return dataset_instruction_for_sample(sample, hotpot_style="shortest_span_copy")


def _base_reasoning_method(method_family: str) -> str:
    normalized = str(method_family or "").strip().lower()
    if normalized in {"cot", "self_consistency", "majority_vote", "mv"}:
        return "cot"
    return normalized
