"""跨实验统一的控制方法 prompt 模板。"""

from __future__ import annotations

from dataclasses import dataclass

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.core.prompts.dataset_contracts import (
    build_json_system_prompt,
    dataset_instruction_for_sample,
)
from research_experiments.family_runtime.free_text_protocol import (
    build_free_text_answer_instruction,
    build_free_text_system_prompt,
)

CONTROL_PROMPT_VERSION = "unified_control_v1"
FREE_TEXT_V1_PROMPT_VERSION = "single_agent_free_text_v1"


@dataclass(frozen=True)
class _ReasoningMethodSpec:
    label: str
    summary: str
    guidance: str
    checklist: str


def _resolve_cot_method(dataset: str) -> _ReasoningMethodSpec:
    del dataset
    return _ReasoningMethodSpec(
        label="CoT",
        summary="Chain-of-Thought prompting: derive the answer through an explicit step-by-step solution.",
        guidance="Write one coherent derivation, keep every decisive inference explicit, and verify the final conclusion before committing.",
        checklist="state the decisive steps in order, justify the key transformation, and check that the conclusion answers the original question",
    )


def build_cot_messages(
    sample: DatasetSample,
    replicate_id: int,
    prompt_version: str | None = None,
) -> list[dict[str, str]]:
    del replicate_id
    return [
        {"role": "system", "content": _system_prompt(prompt_version)},
        {"role": "user", "content": _user_prompt(sample, prompt_version)},
    ]


def build_mv_messages(
    sample: DatasetSample,
    replicate_id: int,
    prompt_version: str | None = None,
) -> list[dict[str, str]]:
    del replicate_id
    return [
        {"role": "system", "content": _system_prompt(prompt_version)},
        {"role": "user", "content": _user_prompt(sample, prompt_version)},
    ]


def _system_prompt(prompt_version: str | None) -> str:
    if prompt_version == FREE_TEXT_V1_PROMPT_VERSION:
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
    return build_json_system_prompt(
        "You are an expert reasoning assistant for controlled research experiments.",
        extra_rules=[
            "Follow the task instruction carefully.",
            "Return exactly one JSON object with keys reasoning and final_answer.",
            "Keep reasoning concise and under 120 tokens.",
        ],
    )


def _user_prompt(sample: DatasetSample, prompt_version: str | None) -> str:
    reasoning_spec = _resolve_cot_method(sample.dataset)
    user_prompt = (
        f"Reasoning method: {reasoning_spec.label}\n"
        f"Method summary: {reasoning_spec.summary}\n"
        f"Method guidance: {reasoning_spec.guidance}\n"
        f"Method checklist: {reasoning_spec.checklist}\n"
        f"{dataset_instruction_for_sample(sample, hotpot_style='shortest_span_copy')}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"

    if prompt_version == FREE_TEXT_V1_PROMPT_VERSION:
        user_prompt += build_free_text_answer_instruction(sample.dataset)
        return user_prompt

    user_prompt += (
        'Return exactly one JSON object like '
        '{"reasoning":"brief reasoning","final_answer":"answer"}'
    )
    return user_prompt
