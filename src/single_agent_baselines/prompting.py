"""Single-agent prompting."""

from __future__ import annotations

from experiment_core.datasets import DatasetSample


DEFAULT_PROMPT_VERSION = "single_agent_reasoning_json_v1"


def build_messages(
    sample: DatasetSample,
    method_family: str,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    """Build one single-agent request."""
    del method_family
    if prompt_version != DEFAULT_PROMPT_VERSION:
        raise ValueError(f"Unsupported single-agent prompt_version: {prompt_version}")
    return [
        {"role": "system", "content": _system_prompt(prompt_version)},
        {"role": "user", "content": _user_prompt(sample, prompt_version)},
    ]


def _system_prompt(prompt_version: str) -> str:
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
    if sample.dataset == "gsm8k":
        return (
            "Solve the math word problem carefully. "
            "The final_answer must be only the final numeric answer without commas or units."
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
    raise ValueError(f"Unsupported dataset: {sample.dataset}")
