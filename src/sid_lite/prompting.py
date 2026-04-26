"""SID-lite 提示词构造。

本模块为 SID-lite 的两个阶段构造结构化 prompt：
Stage A 负责产出答案、证据与置信度，
Stage B 在读取压缩消息后仅输出“是否改答”的最小更新信息。
"""

from __future__ import annotations

from experiment_core.datasets import DatasetSample


DEFAULT_PROMPT_VERSION = "sid_lite_v1_json"


def build_solver_messages(
    sample: DatasetSample,
    agent_id: int,
    *,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    """构造 Stage A 初始解题提示词。"""
    _ensure_prompt_version(prompt_version)
    user_prompt = (
        f"You are agent_{agent_id} in Stage A of a SID-lite experiment.\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        "Return exactly one JSON object with keys "
        '{"final_answer":"answer","reasoning_trace":"brief trace","claim_span":"one short disputable step",'
        '"key_evidence":"one short evidence span","uncertain_point":"one short uncertainty",'
        '"confidence_raw":0.0}.\n'
        "Keep all fields concise. confidence_raw should be numeric in [0, 1]. Return JSON only."
    )
    return [
        {"role": "system", "content": _solver_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def build_belief_update_messages(
    sample: DatasetSample,
    agent_id: int,
    *,
    previous_packet: dict[str, object],
    peer_packets: list[dict[str, object]],
    packet_mode: str,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    """构造一轮 belief update 提示词。"""
    _ensure_prompt_version(prompt_version)
    peer_block = "\n\n".join(
        f"{item['agent']} packet_mode={item['packet_mode']}\npacket={item['packet_text']}"
        for item in peer_packets
    ) or "No peer packets."
    user_prompt = (
        f"You are agent_{agent_id} in Stage B of SID-lite.\n"
        f"Visible packet mode: {packet_mode}.\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        f"Your previous packet:\n{previous_packet['packet_text']}\n\n"
        f"Peer packets:\n{peer_block}\n\n"
        "Revise only if peer packets reveal a concrete mistake or stronger evidence. "
        "If you keep your answer, set changed_answer to false and repeat your previous answer in new_answer.\n"
        'Return exactly one JSON object with keys "changed_answer", "new_answer", '
        '"confidence_delta", "reason_for_change", and "remaining_disagreement". Return JSON only.'
    )
    return [
        {"role": "system", "content": _belief_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def _ensure_prompt_version(prompt_version: str) -> None:
    if prompt_version != DEFAULT_PROMPT_VERSION:
        raise ValueError(f"Unsupported SID-lite prompt_version: {prompt_version}")


def _solver_system_prompt() -> str:
    return (
        "You are one reasoning agent in a controlled SID-lite mechanism experiment.\n"
        "Return strict JSON only. Do not use markdown fences. Do not add extra keys.\n"
        "Confidence is a self-reported black-box proxy because logits and attention are unavailable."
    )


def _belief_system_prompt() -> str:
    return (
        "You are one reasoning agent receiving SID-lite peer messages.\n"
        "Return strict JSON only. Do not restate the whole solution. Only update for concrete reasons."
    )


def _dataset_instruction(sample: DatasetSample) -> str:
    if sample.dataset == "gsm8k":
        return (
            "Solve the math word problem carefully. "
            "The final_answer must be only the final numeric answer without commas or units."
        )
    if sample.dataset == "strategyqa":
        return 'Answer with exactly "yes" or "no". The final_answer must be exactly "yes" or "no".'
    if sample.dataset == "hotpotqa":
        return (
            "Answer the multi-hop question using only the provided context. "
            "The final_answer must be the shortest judgeable text span."
        )
    raise ValueError(f"Unsupported dataset: {sample.dataset}")
