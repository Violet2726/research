"""Free-MAD-lite 实验的提示词构造器。"""

from __future__ import annotations

from hashlib import sha256

from research_experiments.core.foundation.datasets import DatasetSample
from research_experiments.core.foundation.prompt_contracts import build_json_system_prompt, dataset_instruction_for_sample


DEFAULT_PROMPT_VERSION = "free_mad_lite_v1_json"


def build_initial_messages(
    sample: DatasetSample,
    agent_id: int,
    *,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    """构造 Stage A 的独立求解提示词。"""
    _ensure_prompt_version(prompt_version)
    user_prompt = (
        f"You are agent_{agent_id} in a Free-MAD-lite experiment.\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        'Return exactly one JSON object with keys "final_answer", "reasoning", and "confidence_raw". '
        "Keep reasoning concise. confidence_raw should be numeric in [0, 1]. Return JSON only."
    )
    return [
        {"role": "system", "content": _base_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def build_debate_messages(
    sample: DatasetSample,
    agent_id: int,
    *,
    mode: str,
    previous_answer: str,
    previous_reasoning: str,
    peer_messages: list[dict[str, str]],
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    """构造单轮 vanilla / anti-conformity 辩论提示词。"""
    _ensure_prompt_version(prompt_version)
    if mode not in {"vanilla", "anti_conformity"}:
        raise ValueError(f"Unsupported Free-MAD-lite debate mode: {mode}")
    peer_block = "\n\n".join(
        f"{item['agent']} answer: {item['answer']}\n{item['agent']} reasoning: {item['reasoning']}"
        for item in peer_messages
    ) or "No peer feedback."
    instruction = _vanilla_revision_instruction() if mode == "vanilla" else _anti_conformity_instruction()
    user_prompt = (
        f"You are agent_{agent_id} in one-round {mode} debate.\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        f"Your previous final_answer: {previous_answer}\n"
        f"Your previous reasoning: {previous_reasoning}\n\n"
        f"Peer responses:\n{peer_block}\n\n"
        f"{instruction}\n"
        'Return exactly one JSON object with keys "final_answer", "reasoning", "changed_answer", and "confidence_raw". '
        "Return JSON only."
    )
    return [
        {"role": "system", "content": _base_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def build_trajectory_judge_messages(
    sample: DatasetSample,
    trajectories: list[dict[str, object]],
    *,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    """构造轨迹级裁判的评审提示词。"""
    _ensure_prompt_version(prompt_version)
    trajectory_block = "\n\n".join(
        f"Agent {item['agent_id']}:\n"
        f"initial_answer={item['initial_answer']}\n"
        f"initial_reasoning={item['initial_reasoning']}\n"
        f"anti_answer={item['anti_answer']}\n"
        f"anti_reasoning={item['anti_reasoning']}\n"
        f"changed_answer={item['changed_answer']}"
        for item in trajectories
    )
    user_prompt = (
        "Choose the best final answer by evaluating the whole trajectory, not only the final majority.\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        f"Trajectories:\n{trajectory_block}\n\n"
        'Return exactly one JSON object with keys "final_answer", "selected_agent_id", and "rationale". '
        "selected_agent_id must be 1, 2, or 3 when possible. Return JSON only."
    )
    return [
        {"role": "system", "content": _judge_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def anti_conformity_prompt_hash() -> str:
    """返回 anti-conformity 指令文本的稳定哈希。"""
    return sha256(_anti_conformity_instruction().encode("utf-8")).hexdigest()


def _ensure_prompt_version(prompt_version: str) -> None:
    if prompt_version != DEFAULT_PROMPT_VERSION:
        raise ValueError(f"Unsupported Free-MAD-lite prompt_version: {prompt_version}")


def _base_system_prompt() -> str:
    return build_json_system_prompt(
        "You are one reasoning agent in a controlled multi-agent debate experiment.",
        extra_rules=["Do not add extra keys."],
    )


def _judge_system_prompt() -> str:
    return build_json_system_prompt(
        "You are a trajectory-level judge for a controlled Free-MAD-lite experiment.",
        extra_rules=["Prefer evidence and reasoning stability over conformity."],
    )


def _vanilla_revision_instruction() -> str:
    return "Revise your answer only if peer reasoning reveals a concrete mistake or stronger evidence."


def _anti_conformity_instruction() -> str:
    return (
        "Use anti-conformity: actively look for flaws in peer answers, especially majority answers. "
        "Do not change your answer merely because most peers disagree. Change only when the evidence or reasoning is stronger."
    )


def _dataset_instruction(sample: DatasetSample) -> str:
    return dataset_instruction_for_sample(sample, hotpot_style="shortest_span")

