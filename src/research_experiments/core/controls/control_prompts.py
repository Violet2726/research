"""跨实验统一的控制方法 prompt 模板。

本模块为 cot、majority_vote 等无通信控制方法提供统一的 prompt 构建函数，
确保同一控制方法在不同实验家族中使用完全相同的 prompt，实现公平对比。

设计原则：
- 复用 single_agent 的 JSON 输出格式（结构化强、解析可靠）
- 复用 core/prompts/dataset_contracts 的数据集指令
- 内联 CoT 推理方法定义，避免跨层导入
"""

from __future__ import annotations

from dataclasses import dataclass

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.core.prompts.dataset_contracts import (
    build_json_system_prompt,
    dataset_instruction_for_sample,
)

# 统一的 prompt 版本标识
CONTROL_PROMPT_VERSION = "unified_control_v1"


@dataclass(frozen=True)
class _ReasoningMethodSpec:
    """推理方法规格（内联定义，避免跨层导入）。"""
    label: str
    summary: str
    guidance: str
    checklist: str


def _resolve_cot_method(dataset: str) -> _ReasoningMethodSpec:
    """解析 CoT 推理方法规格。"""
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
    """构建统一的 CoT (Chain-of-Thought) 控制方法 prompt。

    此函数不依赖任何实验家族的 prompt 模板，确保所有实验使用完全相同的 prompt。
    """
    return [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": _user_prompt(sample, "cot")},
    ]


def build_mv_messages(
    sample: DatasetSample,
    replicate_id: int,
    prompt_version: str | None = None,
) -> list[dict[str, str]]:
    """构建统一的 Majority Vote 控制方法 prompt。

    每次调用使用不同的 replicate_id 来获取独立采样结果。
    """
    return [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": _user_prompt(sample, "cot")},
    ]


def _system_prompt() -> str:
    """统一的系统提示词。"""
    return build_json_system_prompt(
        "You are an expert reasoning assistant for controlled research experiments.",
        extra_rules=[
            "Follow the task instruction carefully.",
            "Return exactly one JSON object with keys reasoning and final_answer.",
            "Keep reasoning concise and under 120 tokens.",
        ],
    )


def _user_prompt(sample: DatasetSample, method_family: str) -> str:
    """统一的用户提示词。"""
    reasoning_spec = _resolve_cot_method(sample.dataset)
    user_prompt = (
        f"Reasoning method: {reasoning_spec.label}\n"
        f"Method summary: {reasoning_spec.summary}\n"
        f"Method guidance: {reasoning_spec.guidance}\n"
        f"Method checklist: {reasoning_spec.checklist}\n"
        f"{dataset_instruction_for_sample(sample, hotpot_style='short_span')}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"

    user_prompt += (
        'Return exactly one JSON object like '
        '{"reasoning":"brief reasoning","final_answer":"answer"}'
    )
    return user_prompt
