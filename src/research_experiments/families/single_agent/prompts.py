"""单智能体基线实验的 prompt 构造器。"""

from __future__ import annotations

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.core.prompts.dataset_contracts import build_json_system_prompt, dataset_instruction_for_sample
from research_experiments.family_runtime.reasoning_methods import resolve_reasoning_method

CONSISTENT_JSON_V2_PROMPT_VERSION = "single_agent_consistent_json_v2"
DEFAULT_PROMPT_VERSION = CONSISTENT_JSON_V2_PROMPT_VERSION
SUPPORTED_PROMPT_VERSIONS = (CONSISTENT_JSON_V2_PROMPT_VERSION,)


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
    if prompt_version == CONSISTENT_JSON_V2_PROMPT_VERSION:
        return build_json_system_prompt(
            "You are an expert reasoning assistant for controlled research experiments.",
            extra_rules=[
                "Follow the task instruction carefully.",
                "Return exactly one JSON object with keys reasoning and final_answer.",
                "Output the keys in exactly this order: reasoning, final_answer.",
                "reasoning is required.",
                "Keep reasoning concise, but include enough detail to justify or revise the answer.",
                "If your reasoning changes the answer, rewrite final_answer to the corrected answer.",
            ],
        )
    return build_json_system_prompt(
        "You are an expert reasoning assistant for controlled research experiments.",
        extra_rules=[
            "Follow the task instruction carefully.",
            "Return exactly one JSON object with keys final_answer, final_answer_check, and reasoning.",
            "Output the keys in exactly this order: final_answer, final_answer_check, reasoning.",
            "final_answer and final_answer_check must match exactly after normalization.",
            "reasoning is required and must stay under 120 tokens.",
            "If your reasoning changes the answer, rewrite both answer fields to the corrected answer.",
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

    if prompt_version == CONSISTENT_JSON_V2_PROMPT_VERSION:
        user_prompt += (
            'Return exactly one JSON object like '
            '{"reasoning":"brief reasoning","final_answer":"canonical answer"}'
        )
        return user_prompt
    raise ValueError(f"Unsupported single-agent prompt_version: {prompt_version}")


def _dataset_instruction(sample: DatasetSample) -> str:
    return dataset_instruction_for_sample(sample, hotpot_style="shortest_span_copy")


def _base_reasoning_method(method_family: str) -> str:
    normalized = str(method_family or "").strip().lower()
    if normalized in {"cot", "self_consistency", "majority_vote", "mv"}:
        return "cot"
    return normalized
