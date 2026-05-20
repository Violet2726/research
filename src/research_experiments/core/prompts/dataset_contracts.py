"""跨实验共享的题型指令与系统提示词契约。"""

from __future__ import annotations

from typing import Literal

from research_experiments.core.data.datasets import DatasetSample


ContextScope = Literal["provided", "visible"]
HotpotStyle = Literal["short_span", "shortest_span", "shortest_span_copy"]
MultipleChoiceScope = Literal["general", "visible"]


def dataset_instruction_for_sample(
    sample: DatasetSample,
    *,
    context_scope: ContextScope = "provided",
    hotpot_style: HotpotStyle = "short_span",
    multiple_choice_scope: MultipleChoiceScope = "general",
) -> str:
    """根据样本对象生成与数据集匹配的任务指令。"""
    return dataset_instruction(
        sample.dataset,
        context_scope=context_scope,
        hotpot_style=hotpot_style,
        multiple_choice_scope=multiple_choice_scope,
    )


def dataset_instruction(
    dataset: str,
    *,
    context_scope: ContextScope = "provided",
    hotpot_style: HotpotStyle = "short_span",
    multiple_choice_scope: MultipleChoiceScope = "general",
) -> str:
    """按数据集与可见上下文范围生成标准化指令文本。"""
    if dataset in {"gsm8k", "gsm_symbolic"}:
        return (
            "Solve the math problem carefully. "
            "The final_answer must be only the final numeric answer without commas or units."
        )
    if dataset in {"math500", "competition_math"}:
        return (
            "Solve the math problem carefully. "
            "The final_answer must be only the final mathematical expression, with no explanation."
        )
    if dataset == "strategyqa":
        return 'Answer with exactly "yes" or "no". The final_answer must be exactly "yes" or "no".'
    if dataset == "hotpotqa":
        context_phrase = "provided context" if context_scope == "provided" else "context visible to you"
        if hotpot_style == "shortest_span_copy":
            return (
                f"Answer the multi-hop question using only the {context_phrase}. "
                "The final_answer must be the shortest judgeable text span. "
                "Prefer copying the exact wording from the context when possible. "
                "Do not add category words, parentheses, explanations, or extra qualifiers."
            )
        if hotpot_style == "shortest_span":
            return (
                f"Answer the multi-hop question using only the {context_phrase}. "
                "The final_answer must be the shortest judgeable text span."
            )
        return (
            f"Answer the multi-hop question using only the {context_phrase}. "
            "The final_answer must be a short text span."
        )
    if dataset in {"webquestions", "grailqa"}:
        context_phrase = "provided graph evidence" if context_scope == "provided" else "graph evidence visible to you"
        return (
            f"Answer the graph question using only the {context_phrase}. "
            "The final_answer must be the shortest judgeable entity span or literal answer. "
            "Do not add category words, explanations, or extra qualifiers."
        )
    if dataset in {"mmlu_pro", "gpqa_diamond", "mmlu_abstract_algebra"}:
        if multiple_choice_scope == "visible":
            return (
                "Choose the single best option using only the context visible to you. "
                'The final_answer must be only the option letter, such as "A" or "B".'
            )
        return (
            "Choose the single best option. "
            'The final_answer must be only the option letter, such as "A" or "B".'
        )
    raise ValueError(f"Unsupported dataset: {dataset}")


def build_json_system_prompt(
    role_description: str,
    *,
    extra_rules: list[str] | None = None,
) -> str:
    """拼装严格 JSON 输出模式下的 system prompt。"""
    lines = [
        role_description.strip(),
        "Return strict JSON only.",
        "Do not use markdown fences.",
    ]
    if extra_rules:
        lines.extend(rule.strip() for rule in extra_rules if rule.strip())
    return "\n".join(lines)


def build_tagged_lines_system_prompt(
    role_description: str,
    *,
    extra_rules: list[str] | None = None,
) -> str:
    """拼装“固定标签行”输出模式下的 system prompt。"""
    lines = [
        role_description.strip(),
        "Return only the requested tagged lines.",
        "Do not use markdown fences.",
    ]
    if extra_rules:
        lines.extend(rule.strip() for rule in extra_rules if rule.strip())
    return "\n".join(lines)


