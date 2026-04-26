"""SPARC 提示词构造。

本模块为 solver、debate、single judge 与 auditor 分别构造 prompt，
显式区分内容压缩、触发通信和局部审计几个阶段的输出契约。
"""

from __future__ import annotations

from experiment_core.datasets import DatasetSample


DEFAULT_PROMPT_VERSION = "sparc_v1_json"


def build_solver_messages(
    sample: DatasetSample,
    agent_id: int,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    if prompt_version != DEFAULT_PROMPT_VERSION:
        raise ValueError(f"Unsupported sparc prompt_version: {prompt_version}")
    user_prompt = (
        f"You are agent_{agent_id} in Stage A of a SPARC selective communication experiment.\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        "Return exactly one JSON object with keys "
        '{"final_answer":"answer","reasoning_trace":"short trace","claim_span":"one short disputable step",'
        '"confidence_raw":0.0,"uncertain_point":"one short uncertainty","key_evidence":"one short evidence"}.\n'
        "Keep reasoning_trace concise, claim_span very short, and key_evidence to one short string.\n"
        "Return JSON only."
    )
    return [
        {"role": "system", "content": _solver_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def build_debate_messages(
    sample: DatasetSample,
    agent_id: int,
    round_index: int,
    previous_packet: dict[str, object],
    peer_packets: list[dict[str, object]],
    requested_message_mode: str,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    if prompt_version != DEFAULT_PROMPT_VERSION:
        raise ValueError(f"Unsupported sparc prompt_version: {prompt_version}")
    peer_block = "\n\n".join(
        f"{item['agent']} shared_mode: {item['effective_message_mode']}\n"
        f"{item['agent']} packet: {item['packet_text']}"
        for item in peer_packets
    ) or "No peer packets."
    user_prompt = (
        f"You are agent_{agent_id} in Stage B debate round {round_index}.\n"
        f"Requested message mode: {requested_message_mode}.\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        f"Your previous packet:\n{previous_packet['packet_text']}\n\n"
        f"Peer packets:\n{peer_block}\n\n"
        "Revise only if peer packets expose a concrete mistake or stronger evidence.\n"
        "If you do not change the answer, set changed_answer to false and repeat your previous final answer in new_answer.\n"
        "If you change the answer, set changed_answer to true and provide the revised final answer in new_answer.\n"
        "Return exactly one JSON object with keys "
        '{"changed_answer":true,"new_answer":"answer","confidence_delta":-0.05,'
        '"reason_for_change":"short reason","remaining_disagreement":"short unresolved point"}.\n'
        "Return JSON only."
    )
    return [
        {"role": "system", "content": _debate_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def build_single_judge_messages(
    sample: DatasetSample,
    candidates: list[dict[str, object]],
) -> list[dict[str, str]]:
    candidate_block = "\n\n".join(
        f"Candidate {index}:\n{candidate}"
        for index, candidate in enumerate(candidates, start=1)
    )
    user_prompt = (
        "Select the best final answer among the candidate packets.\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        f"Candidate packets:\n{candidate_block}\n\n"
        'Return exactly one JSON object with key "final_answer". '
        'You may optionally include "reasoning". '
        "Return JSON only."
    )
    return [
        {"role": "system", "content": _judge_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def build_audit_messages(
    sample: DatasetSample,
    candidate_a: dict[str, object],
    candidate_b: dict[str, object],
) -> list[dict[str, str]]:
    user_prompt = (
        "You are a local auditor. Compare only the two candidate packets below.\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        f"Candidate A:\n{candidate_a}\n\n"
        f"Candidate B:\n{candidate_b}\n\n"
        "Do not reconstruct the whole debate. Judge only this local disagreement.\n"
        'Return exactly one JSON object with keys "decision", "verified_answer", and "rationale".\n'
        'decision must be one of "resolve_for_a", "resolve_for_b", or "abstain".\n'
        "Return JSON only."
    )
    return [
        {"role": "system", "content": _audit_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def _solver_system_prompt() -> str:
    return (
        "You are one reasoning agent in a controlled SPARC experiment.\n"
        "Return strict JSON only.\n"
        "Do not use markdown fences.\n"
        "Keep every field concise.\n"
        "confidence_raw should prefer a numeric value in [0, 1]."
    )


def _debate_system_prompt() -> str:
    return (
        "You are one reasoning agent receiving compressed peer packets.\n"
        "Return strict JSON only.\n"
        "Do not use markdown fences.\n"
        "Do not restate the whole solution.\n"
        "Only report whether your answer changed and why."
    )


def _judge_system_prompt() -> str:
    return (
        "You are a concise judge for controlled research experiments.\n"
        "Return strict JSON only.\n"
        "Do not add explanations outside the JSON object."
    )


def _audit_system_prompt() -> str:
    return (
        "You are a local auditor for SPARC.\n"
        "You only see two candidate packets and must not imagine missing debate context.\n"
        "Return strict JSON only.\n"
        "Use abstain when the local evidence is insufficient."
    )


def _dataset_instruction(sample: DatasetSample) -> str:
    if sample.dataset == "gsm8k":
        return (
            "Solve the math word problem carefully. "
            "The final answer must be only the final numeric answer without commas or units."
        )
    if sample.dataset == "strategyqa":
        return (
            'Answer with exactly "yes" or "no". '
            'The final answer must be exactly "yes" or "no".'
        )
    if sample.dataset == "hotpotqa":
        return (
            "Answer the multi-hop question using only the provided context. "
            "The final answer must be the shortest judgeable text span."
        )
    raise ValueError(f"Unsupported dataset: {sample.dataset}")
