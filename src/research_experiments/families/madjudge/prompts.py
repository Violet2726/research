"""MADJudge 实验的提示词构造器。

基于论文 "Multi-Agent Debate for LLM Judges with Adaptive Stability Detection" 的 prompt 设计。
论文使用简洁的辩论格式，让 judges 协作推理并迭代优化判断。
"""

from __future__ import annotations

from typing import Any

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.core.prompts.dataset_contracts import dataset_instruction_for_sample

DEFAULT_PROMPT_VERSION = "madjudge_v1"

# ── 论文 Section 3.1：初始响应生成 ──────────────────────────────────────────
_INITIAL_TEMPLATE = """{dataset_instruction}
{question}{context}

Provide your judgment/response. Think step by step, then give your final answer.

##Answer
<your answer here>

##Explanation
<your reasoning here>"""

# ── 论文 Section 3.1：多轮辩论 ──────────────────────────────────────────────
_DEBATE_TEMPLATE = """{dataset_instruction}
{question}{context}

Here are responses provided by other judges. Please review their reasoning and update your response if you find their arguments convincing. Maintain your position if you believe your original reasoning is correct.

{peer_block}

Provide your updated judgment/response. Think step by step, then give your final answer.

##Answer
<your answer here>

##Explanation
<your reasoning here>"""


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
    return "You are a helpful judge providing accurate and well-reasoned responses."


def _dataset_instruction(sample: DatasetSample) -> str:
    if sample.dataset == "hotpotqa":
        return dataset_instruction_for_sample(sample, hotpot_style="shortest_span_copy")
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
    if prompt_version != DEFAULT_PROMPT_VERSION:
        raise ValueError(f"Unsupported MADJudge prompt_version: {prompt_version}")
