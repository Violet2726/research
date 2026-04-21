"""Single-agent prompting."""

from __future__ import annotations

from experiment_core.datasets import DatasetSample


DEFAULT_PROMPT_VERSION = "single_agent_baseline_json"


def build_messages(
    sample: DatasetSample,
    method_family: str,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    """Build one single-agent request."""
    del method_family
    if prompt_version != DEFAULT_PROMPT_VERSION:
        raise ValueError(f"Unsupported single-agent prompt_version: {prompt_version}")

    user_prompt = (
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        'Return exactly one JSON object with key "final_answer". '
        'You may optionally include "reasoning". '
        'Do not add any other keys. '
        'Return JSON only.'
    )
    return [
        {
            "role": "system",
            "content": (
                "You are an expert reasoning assistant for controlled research experiments.\n"
                "Follow the task instruction carefully.\n"
                "Return a single JSON object only.\n"
                "Do not use markdown fences.\n"
                "Do not add natural-language text before or after the JSON object."
            ),
        },
        {"role": "user", "content": user_prompt},
    ]


def _dataset_instruction(sample: DatasetSample) -> str:
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
