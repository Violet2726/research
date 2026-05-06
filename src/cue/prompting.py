"""CUE 实验的提示词构造器。"""

from __future__ import annotations

import json

from experiment_core.datasets import DatasetSample
from experiment_core.prompt_contracts import build_json_system_prompt, dataset_instruction_for_sample


DEFAULT_PROMPT_VERSION = "cue_v1_json"


def build_solver_messages(
    sample: DatasetSample,
    agent_id: int,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    """构造 CUE Stage A 的独立求解消息。"""
    if prompt_version != DEFAULT_PROMPT_VERSION:
        raise ValueError(f"Unsupported cue prompt_version: {prompt_version}")
    user_prompt = (
        f"You are agent_{agent_id} in a CUE black-box multi-agent reasoning experiment.\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        "First solve independently. Then return exactly one JSON object with keys "
        '{"final_answer":"answer","confidence":0.0,"reasoning_sketch":"one short summary",'
        '"uncertain_point":"one short uncertainty or null","top_claims":["short claim"],'
        '"evidence_items":["short evidence"],"counter_answer":"best alternative answer or null"}.\n'
        "Keep all fields extremely concise.\n"
        "reasoning_sketch must be at most 25 words.\n"
        "uncertain_point must be at most 12 words.\n"
        "Each item in top_claims and evidence_items must be at most 8 words.\n"
        "Do not explain outside the JSON object.\n"
        "Return JSON only."
    )
    return [
        {"role": "system", "content": _solver_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def build_communication_messages(
    sample: DatasetSample,
    agent_id: int,
    round_index: int,
    previous_packet: dict[str, object],
    peer_packets: list[dict[str, object]],
    conflict_object: dict[str, object],
    message_type: str,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    """构造 CUE 通信轮中围绕 conflict object 的更新消息。"""
    if prompt_version != DEFAULT_PROMPT_VERSION:
        raise ValueError(f"Unsupported cue prompt_version: {prompt_version}")
    peer_block = "\n\n".join(
        f"{item['agent']} packet:\n{item['packet_text']}"
        for item in peer_packets
    ) or "No peer packets."
    conflict_text = json.dumps(conflict_object, ensure_ascii=False, sort_keys=True)
    user_prompt = (
        f"You are agent_{agent_id} in CUE communication round {round_index}.\n"
        f"Message type: {message_type}.\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        f"Your previous packet:\n{json.dumps(previous_packet, ensure_ascii=False, sort_keys=True)}\n\n"
        f"Conflict object:\n{conflict_text}\n\n"
        f"Peer packets:\n{peer_block}\n\n"
        "Revise only if the peer packets or conflict object expose a concrete mistake or stronger support.\n"
        "Return exactly one JSON object with keys "
        '{"changed_answer":true,"new_answer":"answer","confidence_delta":-0.05,'
        '"reason_for_change":"short reason","remaining_disagreement":"short unresolved point or null"}.\n'
        "Keep reason_for_change and remaining_disagreement each under 20 words.\n"
        "Return JSON only."
    )
    return [
        {"role": "system", "content": _communication_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def build_audit_messages(
    sample: DatasetSample,
    candidate_a: dict[str, object],
    candidate_b: dict[str, object],
    conflict_object: dict[str, object],
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    """构造 CUE 本地审计器比较候选解的提示词。"""
    if prompt_version != DEFAULT_PROMPT_VERSION:
        raise ValueError(f"Unsupported cue prompt_version: {prompt_version}")
    user_prompt = (
        "You are a local auditor in the CUE framework.\n"
        "Judge only the provided local conflict. Do not reconstruct the whole debate.\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        f"Conflict object:\n{json.dumps(conflict_object, ensure_ascii=False, sort_keys=True)}\n\n"
        f"Candidate A:\n{json.dumps(candidate_a, ensure_ascii=False, sort_keys=True)}\n\n"
        f"Candidate B:\n{json.dumps(candidate_b, ensure_ascii=False, sort_keys=True)}\n\n"
        'Return exactly one JSON object with keys "decision", "verified_answer", and "rationale".\n'
        'decision must be one of "resolve_for_a", "resolve_for_b", or "abstain".\n'
        "rationale must be one short sentence under 20 words.\n"
        "Return JSON only."
    )
    return [
        {"role": "system", "content": _audit_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def _solver_system_prompt() -> str:
    return build_json_system_prompt(
        "You are one reasoning agent in a controlled CUE experiment.",
        extra_rules=["confidence should prefer a numeric value in [0, 1]."],
    )


def _communication_system_prompt() -> str:
    return build_json_system_prompt(
        "You are one reasoning agent receiving compact peer evidence.",
        extra_rules=[
            "Do not restate the full solution.",
            "Only report whether you change the answer and why.",
        ],
    )


def _audit_system_prompt() -> str:
    return build_json_system_prompt(
        "You are a cautious local auditor.",
        extra_rules=["Use abstain when the local evidence is insufficient."],
    )


def _dataset_instruction(sample: DatasetSample) -> str:
    return dataset_instruction_for_sample(sample, hotpot_style="shortest_span")
