"""预算约束通信实验的提示词构造器。"""

from __future__ import annotations

from budget_comm.dataset_views import ContextView
from experiment_core.datasets import DatasetSample
from experiment_core.prompt_contracts import build_json_system_prompt, dataset_instruction_for_sample


DEFAULT_PROMPT_VERSION = "budget_comm_dala_lite_v1"


def build_solver_messages(
    sample: DatasetSample,
    view: ContextView,
    *,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    """构造预算通信实验 Stage A 的独立求解提示词。"""
    if prompt_version != DEFAULT_PROMPT_VERSION:
        raise ValueError(f"Unsupported budget_comm prompt_version: {prompt_version}")
    user_prompt = (
        f"You are agent_{view.agent_id} in Stage A of a budget-aware communication experiment.\n"
        f"Track: {view.track_name}. View kind: {view.view_kind}.\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if view.context_text:
        user_prompt += f"Context available to you:\n{view.context_text}\n\n"
    user_prompt += (
        "Return exactly one JSON object with keys "
        '{"final_answer":"answer","reasoning_trace":"brief trace","claim_span":"one short disputable claim",'
        '"key_evidence":"one short evidence span","keyword_clues":["kw1","kw2"],'
        '"confidence_raw":0.0,"uncertain_point":"one short uncertainty"}.\n'
        "Keep every field concise. keyword_clues should be a short list of clue strings.\n"
        "Return JSON only."
    )
    return [
        {"role": "system", "content": _solver_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def build_belief_update_messages(
    sample: DatasetSample,
    view: ContextView,
    *,
    previous_answer: str,
    previous_reasoning_trace: str,
    previous_confidence_raw: object,
    selected_peer_packets: list[dict[str, object]],
    method_name: str,
    round_budget_tokens: int | None,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    """构造在预算约束下吸收 peer packet 的 Stage B 提示词。"""
    if prompt_version != DEFAULT_PROMPT_VERSION:
        raise ValueError(f"Unsupported budget_comm prompt_version: {prompt_version}")
    peer_block = "\n\n".join(
        f"{item['agent']} mode={item['packet_mode']}\npacket={item['packet_text']}"
        for item in selected_peer_packets
    ) or "No selected peer packets were broadcast to you."
    budget_text = f"{round_budget_tokens}" if round_budget_tokens is not None else "not_applicable"
    user_prompt = (
        f"You are agent_{view.agent_id} in Stage B of method {method_name}.\n"
        f"Track: {view.track_name}. View kind: {view.view_kind}.\n"
        f"Round communication budget tokens: {budget_text}.\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if view.context_text:
        user_prompt += f"Your private context:\n{view.context_text}\n\n"
    user_prompt += (
        f"Your previous final_answer: {previous_answer}\n"
        f"Your previous reasoning_trace: {previous_reasoning_trace}\n"
        f"Your previous confidence_raw: {previous_confidence_raw}\n\n"
        f"Selected peer packets:\n{peer_block}\n\n"
        "Revise only if the selected peer packets reveal a concrete mistake or supply missing evidence.\n"
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
        "You are one reasoning agent in a controlled DALA-lite experiment.",
        extra_rules=[
            "Do not add extra keys.",
            "Keep keyword_clues short and concrete.",
            "confidence_raw should prefer a numeric value in [0, 1].",
        ],
    )


def _belief_update_system_prompt() -> str:
    return build_json_system_prompt(
        "You are one reasoning agent receiving budget-selected peer packets.",
        extra_rules=[
            "Do not restate the full solution.",
            "Only update the answer if there is a concrete reason.",
        ],
    )


def _dataset_instruction(sample: DatasetSample) -> str:
    return dataset_instruction_for_sample(
        sample,
        context_scope="visible",
        hotpot_style="shortest_span",
        multiple_choice_scope="visible",
    )
