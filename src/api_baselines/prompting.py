from __future__ import annotations

from api_baselines.datasets import DatasetSample


SYSTEM_PROMPT = """You are an expert reasoning assistant for controlled research experiments.
Follow the task instruction carefully.
Return strict JSON with keys reasoning and final_answer.
Keep reasoning concise and under 120 tokens.
Do not add markdown fences."""


def build_messages(sample: DatasetSample, method_family: str) -> list[dict[str, str]]:
    # Keep the solver prompt identical across CoT, SC, and MV so the budget
    # comparison is not polluted by prompt-style differences.
    user_prompt = (
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        'Return exactly one JSON object like '
        '{"reasoning":"brief reasoning","final_answer":"answer"}'
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
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
            "Answer the question with yes or no. "
            'The final_answer must be exactly "yes" or "no".'
        )
    if sample.dataset == "hotpotqa":
        return (
            "Answer the multi-hop question using only the provided context. "
            "The final_answer should be a short text span."
        )
    raise ValueError(f"Unsupported dataset: {sample.dataset}")
