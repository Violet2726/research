"""MADJudge 实验的提示词构造器。

基于论文 "Multi-Agent Debate for LLM Judges with Adaptive Stability Detection" 的 prompt 设计。
统一使用 JSON 输出格式，包含 final_answer 和 reasoning 字段。
"""

from __future__ import annotations

from typing import Any

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.core.prompts.dataset_contracts import build_json_system_prompt, dataset_instruction_for_sample

DEFAULT_PROMPT_VERSION = "madjudge_v1"

# ── 论文 Section 3.1：初始响应生成 ──────────────────────────────────────────
_INITIAL_TEMPLATE = """{dataset_instruction}
{question}{context}

Provide your judgment/response. Think step by step, then give your final answer.

Return exactly one JSON object with keys "final_answer" and "reasoning".
- final_answer: your answer to the question
- reasoning: brief step-by-step explanation (under 120 tokens)
- Ensure final_answer exactly matches the conclusion stated in reasoning.

Return JSON only. Do not add text before or after the JSON object."""

# ── 论文 Section 3.1：多轮辩论 ──────────────────────────────────────────────
_DEBATE_TEMPLATE = """{dataset_instruction}
{question}{context}

Here are responses provided by other judges. Please review their reasoning and update your response if you find their arguments convincing. Maintain your position if you believe your original reasoning is correct.

{peer_block}

Provide your updated judgment/response. Think step by step, then give your final answer.

Return exactly one JSON object with keys "final_answer" and "reasoning".
- final_answer: your revised answer
- reasoning: brief explanation of what changed and why (under 120 tokens)
- Ensure final_answer exactly matches the conclusion stated in reasoning.

Return JSON only. Do not add text before or after the JSON object."""


def build_initial_messages(
    sample: DatasetSample,
    agent_id: int,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    *,
    persona_instruction: str = "",
) -> list[dict[str, str]]:
    """构造论文 Section 3.1 的初始响应生成消息。"""
    _assert_prompt_version(prompt_version)
    context_block = f"\n\nContext:\n{sample.prompt_context}" if sample.prompt_context else ""
    dataset_inst = _dataset_instruction(sample)
    user_prompt = _INITIAL_TEMPLATE.format(
        dataset_instruction=dataset_inst,
        question=sample.question.strip(),
        context=context_block,
    )
    return [
        {"role": "system", "content": _system_prompt(persona_instruction)},
        {"role": "user", "content": user_prompt},
    ]


def build_debate_messages(
    sample: DatasetSample,
    agent_id: int,
    round_index: int,
    previous_reasoning: str,
    previous_answer: str,
    peer_messages: list[dict[str, Any]],
    *,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    persona_instruction: str = "",
) -> list[dict[str, str]]:
    """构造论文 Section 3.1 的辩论轮次消息。"""
    _assert_prompt_version(prompt_version)
    context_block = f"\n\nContext:\n{sample.prompt_context}" if sample.prompt_context else ""
    dataset_inst = _dataset_instruction(sample)

    peer_block = _render_peer_messages(peer_messages)

    user_prompt = _DEBATE_TEMPLATE.format(
        dataset_instruction=dataset_inst,
        question=sample.question.strip(),
        context=context_block,
        peer_block=peer_block,
    )
    return [
        {"role": "system", "content": _system_prompt(persona_instruction)},
        {"role": "user", "content": user_prompt},
    ]


# ── 辅助函数 ────────────────────────────────────────────────────────────

def _system_prompt(persona_instruction: str = "") -> str:
    """构建系统提示。"""
    if persona_instruction:
        return persona_instruction
    return build_json_system_prompt(
        "You are a helpful judge providing accurate and well-reasoned responses.",
        extra_rules=[
            "Solve the task carefully using only the provided question and context.",
            "Keep reasoning concise and under 120 tokens.",
            "Before responding, verify that final_answer exactly matches the conclusion in reasoning.",
            "Do not add natural-language text before or after the JSON object.",
            "Do not add labels, category words, or explanatory suffixes to final_answer.",
        ],
    )


def _dataset_instruction(sample: DatasetSample) -> str:
    """按数据集类型选择 MADJudge 的任务说明模板。"""

    if sample.dataset == "hotpotqa":
        return dataset_instruction_for_sample(sample, hotpot_style="short_span")
    return dataset_instruction_for_sample(sample)


def _render_peer_messages(peer_messages: list[dict[str, Any]]) -> str:
    """渲染对等消息。"""
    if not peer_messages:
        return "(No peer messages yet.)"
    blocks = []
    for msg in peer_messages:
        agent = msg.get("agent", "unknown")
        answer = msg.get("answer", "")
        reasoning = msg.get("reasoning", "")
        blocks.append(
            f"{agent} said the answer is {answer} and their explanation is {reasoning}"
        )
    return "\n".join(blocks)


def _assert_prompt_version(prompt_version: str) -> None:
    """校验调用方使用的是当前支持的 MADJudge prompt 版本。"""

    if prompt_version != DEFAULT_PROMPT_VERSION:
        raise ValueError(f"Unsupported MADJudge prompt_version: {prompt_version}")
