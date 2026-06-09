"""单智能体基线实验的提示词构造器。"""

from __future__ import annotations

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.core.prompts.dataset_contracts import build_json_system_prompt, dataset_instruction_for_sample
from research_experiments.family_runtime.reasoning_methods import resolve_reasoning_method

DEFAULT_PROMPT_VERSION = "single_agent_reasoning_json_v1"
SUPPORTED_PROMPT_VERSIONS = (DEFAULT_PROMPT_VERSION,)


def build_messages(
    sample: DatasetSample,
    method_family: str,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    """构造单智能体基线的一轮请求消息。"""

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
    return build_json_system_prompt(
        "You are an expert reasoning assistant for controlled research experiments.",
        extra_rules=[
            "Follow the task instruction carefully.",
            "Return exactly one JSON object with keys reasoning and final_answer.",
            "Keep reasoning concise and under 120 tokens.",
        ],
    )


def _user_prompt(sample: DatasetSample, method_family: str, prompt_version: str) -> str:
    reasoning_spec = resolve_reasoning_method(sample.dataset, _base_reasoning_method(method_family))
    user_prompt = (
        f"Reasoning method: {reasoning_spec.label}\n"
        f"Method summary: {reasoning_spec.summary}\n"
        f"Method guidance: {reasoning_spec.guidance}\n"
        f"Method checklist: {reasoning_spec.checklist}\n"
        f"{_dataset_instruction(sample, prompt_version)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"

    user_prompt += (
        'Return exactly one JSON object like '
        '{"reasoning":"brief reasoning","final_answer":"answer"}'
    )
    return user_prompt


def _dataset_instruction(sample: DatasetSample, prompt_version: str) -> str:
    del prompt_version
    return dataset_instruction_for_sample(sample, hotpot_style="short_span")


def _base_reasoning_method(method_family: str) -> str:
    normalized = str(method_family or "").strip().lower()
    if normalized in {"cot", "self_consistency", "majority_vote", "mv"}:
        return "cot"
    return normalized
