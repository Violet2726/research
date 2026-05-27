"""CONSENSAGENT 实验的提示词构造器（对齐论文 Appendix L + Figure 4）。

模板采用论文的 ##Answer/##Explanation/##Confidence 标记格式，
Phase 3 使用 LLM in-context learning（含 1 个 few-shot 示例）替代微调。
"""

from __future__ import annotations

from typing import Any

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.core.prompts.dataset_contracts import dataset_instruction_for_sample


DEFAULT_PROMPT_VERSION = "consensagent_paper_v1"

# ── 论文 Phase 1：初始响应生成（Appendix L）─────────────────────────────
_INITIAL_TEMPLATE = """{dataset_instruction}
{question}{context}

Provide an explanation after '##Explanation'.
Evaluate your confidence (0.0 to 1.0) after '##Confidence'.

##Answer
<your answer here>

##Explanation
<your reasoning here>

##Confidence
<your confidence score>"""

# ── 论文 Phase 2：多轮辩论（Appendix L）─────────────────────────────────
_DEBATE_TEMPLATE = """{dataset_instruction}
{question}{context}

Here are responses provided by other agents. Please update your responses if necessary.
Clearly explain what you agree with and disagree with.

{peer_block}
##Answer
<your answer here>

##Explanation
<your reasoning here>

##Confidence
<your confidence score>"""

# ── 论文 Phase 3：Prompt 优化（in-context learning 替代微调）───────────
# 基于论文 Figure 4 / Appendix B 的优化示例
_DATASET_OPTIMIZATION_EXAMPLES: dict[str, dict[str, str]] = {
    "hotpotqa": {
        "original": "{question}\n\nContext: {context}",
        "refined": (
            "First, identify the entity mentioned in the first part of the question. "
            "Then, use that entity to answer the second part. "
            "Think step by step. Do NOT agree with other agents without reason. "
            "Output only the final answer, no extra text."
        ),
        "reasoning": "Clarified multi-hop instruction order, added anti-sycophancy warning, specified concise output.",
    },
    "gsm8k": {
        "original": "{question}",
        "refined": (
            "Solve this math word problem step by step.\n"
            "1. Identify the key quantities and what is being asked.\n"
            "2. Break the problem into smaller computational steps.\n"
            "3. Perform each calculation and verify intermediate results.\n"
            "4. State the final numerical answer clearly.\n"
            "Do NOT agree with other agents without verifying their calculations."
        ),
        "reasoning": "Added structured computational steps, verification instruction, and anti-sycophancy warning.",
    },
}

_OPTIMIZER_TEMPLATE = """You are a prompt optimization specialist. A multi-agent debate has stalled due to {trigger_type}.

Below is the original question and the debate history showing the problem.

Original question:
{question}

Debate history:
{debate_summary}

Here is ONE example of how a prompt was successfully optimized for the {dataset_name} dataset:

BEFORE optimization:
{example_original}

AFTER optimization:
{example_refined}

WHY this worked:
{example_reasoning}

---
Now, refine the original question prompt for this specific task. The refined prompt should:
1. Add clearer step-by-step instructions specific to this task type
2. Include guidance that discourages blindly agreeing with other agents
3. Specify the expected answer format clearly

Output ONLY the refined prompt text, nothing else. Do not include explanations or markdown formatting.

Refined prompt:"""


def build_initial_messages(
    sample: DatasetSample,
    agent_id: int,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    *,
    persona_instruction: str = "",
) -> list[dict[str, str]]:
    """构造论文 Phase 1 的初始响应生成消息。"""
    _assert_prompt_version(prompt_version)
    context_block = f"\n\nContext:\n{sample.prompt_context}" if sample.prompt_context else ""
    dataset_inst = _dataset_instruction(sample)
    user_prompt = _INITIAL_TEMPLATE.format(
        dataset_instruction=dataset_inst,
        question=sample.question.strip(),
        context=context_block,
    )
    return [
        {"role": "system", "content": _system_prompt("initial", persona_instruction)},
        {"role": "user", "content": user_prompt},
    ]


def build_debate_messages(
    sample: DatasetSample,
    agent_id: int,
    round_index: int,
    previous_reasoning: str,
    previous_answer: str,
    previous_confidence: float,
    peer_messages: list[dict[str, Any]],
    *,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    persona_instruction: str = "",
) -> list[dict[str, str]]:
    """构造论文 Phase 2 的辩论轮次消息。"""
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
        {"role": "system", "content": _system_prompt("debate", persona_instruction)},
        {"role": "user", "content": user_prompt},
    ]


def build_optimizer_messages(
    sample: DatasetSample,
    debate_history: list[dict[str, Any]],
    trigger_type: str,
    *,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    """构造论文 Phase 3 的 prompt 优化消息（in-context learning）。"""
    _assert_prompt_version(prompt_version)

    # 获取该数据集的 few-shot 示例
    dataset_key = sample.dataset
    example = _DATASET_OPTIMIZATION_EXAMPLES.get(
        dataset_key,
        _DATASET_OPTIMIZATION_EXAMPLES["gsm8k"],  # fallback
    )

    # 格式化辩论摘要
    debate_lines = []
    for entry in debate_history:
        r = entry.get("round", "?")
        answers = entry.get("answers", [])
        parts = [f"Agent {a['agent_id']}: {a['answer']} (conf={a['confidence']:.2f})" for a in answers]
        debate_lines.append(f"Round {r}: " + "; ".join(parts))
    debate_summary = "\n".join(debate_lines) if debate_lines else "(no debate rounds)"

    # 数据集显示名
    dataset_names = {
        "kitab": "KITAB (constraint satisfaction)",
        "clutrr": "CLUTRR (family relationship reasoning)",
        "hotpotqa": "HotpotQA (multi-hop QA)",
        "ethics": "Ethics (moral reasoning)",
        "gsm8k": "GSM8K (math word problems)",
        "triviaqa": "TriviaQA (open-domain QA)",
    }

    user_prompt = _OPTIMIZER_TEMPLATE.format(
        trigger_type=trigger_type,
        question=sample.question.strip() + (f"\nContext: {sample.prompt_context}" if sample.prompt_context else ""),
        debate_summary=debate_summary,
        dataset_name=dataset_names.get(dataset_key, dataset_key),
        example_original=example["original"],
        example_refined=example["refined"],
        example_reasoning=example["reasoning"],
    )
    return [
        {"role": "system", "content": "You are a prompt optimization specialist for multi-agent LLM debate systems."},
        {"role": "user", "content": user_prompt},
    ]


def build_team_answer_messages(
    sample: DatasetSample,
    agent_answers: list[dict[str, Any]],
    *,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    """构造论文 Phase 4 的团队答案生成消息。"""
    _assert_prompt_version(prompt_version)
    answers_block = _render_agent_answers(agent_answers)
    context_block = f"\nContext:\n{sample.prompt_context}" if sample.prompt_context else ""

    user_prompt = (
        f"You are the team answer synthesizer in a multi-agent debate.\n"
        f"Question:\n{sample.question.strip()}{context_block}\n\n"
        f"Agent answers with confidence scores:\n{answers_block}\n\n"
        "Generate the final team answer by considering:\n"
        "1. The confidence scores of each agent\n"
        "2. The consistency of answers across agents\n"
        "3. The quality of reasoning provided\n\n"
        "Output format:\n"
        "##Answer\n<your final answer>"
    )
    return [
        {"role": "system", "content": _system_prompt("team")},
        {"role": "user", "content": user_prompt},
    ]


# ── 辅助函数 ────────────────────────────────────────────────────────────

def _system_prompt(phase: str = "initial", persona_instruction: str = "") -> str:
    """构建系统提示。论文无显式 system prompt，保持最小化。"""
    if persona_instruction:
        return persona_instruction
    return "You are a helpful assistant."


def _dataset_instruction(sample: DatasetSample) -> str:
    if sample.dataset == "hotpotqa":
        return dataset_instruction_for_sample(sample, hotpot_style="shortest_span_copy")
    return dataset_instruction_for_sample(sample)


def _render_peer_messages(peer_messages: list[dict[str, Any]]) -> str:
    """按论文 Appendix L 格式渲染对等消息。"""
    if not peer_messages:
        return "(No peer messages yet.)"
    blocks = []
    for msg in peer_messages:
        agent = msg.get("agent", "unknown")
        answer = msg.get("answer", "")
        reasoning = msg.get("reasoning", "")
        confidence = msg.get("confidence", 0.0)
        blocks.append(
            f"{agent} said the answer is {answer} and their explanation is "
            f"{reasoning} with confidence {confidence:.2f}"
        )
    return "\n".join(blocks)


def _render_agent_answers(agent_answers: list[dict[str, Any]]) -> str:
    """渲染 agent 答案列表。"""
    blocks = []
    for i, ans in enumerate(agent_answers, 1):
        answer = ans.get("answer", "")
        reasoning = ans.get("reasoning", "")
        confidence = ans.get("confidence", 0.0)
        blocks.append(
            f"Agent {i}:\n"
            f"  answer: {answer}\n"
            f"  reasoning: {reasoning}\n"
            f"  confidence: {confidence:.2f}"
        )
    return "\n\n".join(blocks)


def _assert_prompt_version(prompt_version: str) -> None:
    if prompt_version != DEFAULT_PROMPT_VERSION:
        raise ValueError(f"Unsupported CONSENSAGENT prompt_version: {prompt_version}")
