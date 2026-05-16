"""ECON 实验的提示词构造器。"""

from __future__ import annotations

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.core.prompts.dataset_contracts import build_json_system_prompt, dataset_instruction_for_sample


DEFAULT_PROMPT_VERSION = "econ_bne_v1"


def build_single_agent_messages(
    sample: DatasetSample,
    *,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    """构造单智能体 CoT 基线提示词。"""

    if prompt_version != DEFAULT_PROMPT_VERSION:
        raise ValueError(f"Unsupported econ prompt_version: {prompt_version}")
    user_prompt = (
        "You are the single strong baseline in a same-context reasoning experiment.\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context.strip():
        user_prompt += f"Context:\n{sample.prompt_context.strip()}\n\n"
    user_prompt += (
        "Return exactly one JSON object with keys "
        '{"final_answer":"answer","reasoning_trace":"brief trace","claim_span":"one short claim",'
        '"key_evidence":"one short evidence span","keyword_clues":["kw1","kw2"],'
        '"confidence_raw":0.0,"uncertain_point":"one short uncertainty"}.\n'
        "Keep the reasoning compact and concrete. Return JSON only."
    )
    return [
        {"role": "system", "content": _solver_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def build_agent_messages(
    sample: DatasetSample,
    agent_id: int,
    *,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    """构造 same-context agent 的独立初答提示词。"""

    if prompt_version != DEFAULT_PROMPT_VERSION:
        raise ValueError(f"Unsupported econ prompt_version: {prompt_version}")
    user_prompt = (
        f"You are agent_{agent_id} in a low-communication coordination experiment.\n"
        "Work independently before any coordination.\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context.strip():
        user_prompt += f"Context:\n{sample.prompt_context.strip()}\n\n"
    user_prompt += (
        "Return exactly one JSON object with keys "
        '{"final_answer":"answer","reasoning_trace":"brief trace","claim_span":"one short claim",'
        '"key_evidence":"one short evidence span","keyword_clues":["kw1","kw2"],'
        '"confidence_raw":0.0,"uncertain_point":"one short uncertainty"}.\n'
        "Keep every field concise. Return JSON only."
    )
    return [
        {"role": "system", "content": _solver_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def build_belief_update_messages(
    sample: DatasetSample,
    agent_id: int,
    *,
    previous_answer: str,
    previous_reasoning_trace: str,
    previous_confidence_raw: object,
    selected_action: str,
    selected_peer_packets: list[dict[str, object]],
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    """构造受控 belief update 提示词。"""

    if prompt_version != DEFAULT_PROMPT_VERSION:
        raise ValueError(f"Unsupported econ prompt_version: {prompt_version}")
    peer_block = "\n\n".join(
        (
            f"{item['agent']} confidence={item['confidence']}\n"
            f"answer={item['final_answer']}\n"
            f"claim={item['claim_span']}\n"
            f"evidence={item['key_evidence']}\n"
            f"packet={item['packet_text']}"
        )
        for item in selected_peer_packets
    ) or "No peer packets were selected."
    user_prompt = (
        f"You are agent_{agent_id} in a belief-driven coordination experiment.\n"
        f"Selected coordination action: {selected_action}.\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context.strip():
        user_prompt += f"Context:\n{sample.prompt_context.strip()}\n\n"
    user_prompt += (
        f"Your previous final_answer: {previous_answer}\n"
        f"Your previous reasoning_trace: {previous_reasoning_trace}\n"
        f"Your previous confidence_raw: {previous_confidence_raw}\n\n"
        f"Selected peer packets:\n{peer_block}\n\n"
        "Only change your answer if the packets expose a concrete mistake or provide missing evidence.\n"
        "If the disagreement is only stylistic or does not improve correctness, keep your previous answer.\n"
        "Return exactly one JSON object with keys "
        '{"changed_answer":true,"new_answer":"answer","confidence_delta":-0.05,'
        '"reason_for_change":"short reason","remaining_disagreement":"short unresolved point"}.\n'
        "If you keep your answer, set changed_answer to false and repeat the previous answer in new_answer.\n"
        "Return JSON only."
    )
    return [
        {"role": "system", "content": _belief_update_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def _solver_system_prompt() -> str:
    return build_json_system_prompt(
        "You are one reasoning agent in a low-communication multi-agent coordination experiment.",
        extra_rules=[
            "Do not add extra keys.",
            "Keep reasoning_trace short and evidence-grounded.",
            "confidence_raw should prefer a numeric value in [0, 1].",
        ],
    )


def _belief_update_system_prompt() -> str:
    return build_json_system_prompt(
        "You are one reasoning agent receiving a small number of peer packets.",
        extra_rules=[
            "Do not rewrite the answer unless the peer evidence is concrete.",
            "Do not add extra keys.",
        ],
    )


def _dataset_instruction(sample: DatasetSample) -> str:
    return dataset_instruction_for_sample(
        sample,
        context_scope="visible",
        hotpot_style="shortest_span",
        multiple_choice_scope="visible",
    )

