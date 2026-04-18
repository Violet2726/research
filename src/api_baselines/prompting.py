"""提示词构造。

这里刻意让 CoT、SC、MV 共享同一求解提示，只在采样与投票层面区分，
避免预算对比被 prompt 风格差异污染。
"""

from __future__ import annotations

from api_baselines.datasets import DatasetSample


SYSTEM_PROMPT = """You are an expert reasoning assistant for controlled research experiments.
Follow the task instruction carefully.
Return strict JSON with keys reasoning and final_answer.
Keep reasoning concise and under 120 tokens.
Do not add markdown fences."""


def build_messages(sample: DatasetSample, method_family: str) -> list[dict[str, str]]:
    """为单个样本构造标准 chat messages。"""

    # 保持不同方法族的求解提示一致，只让预算和聚合方式产生变量。
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
    """返回数据集专属的回答约束。"""
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
