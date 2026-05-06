"""split-context communication-necessary 实验的提示词构造器。"""

from __future__ import annotations

from comm_necessary.dataset_views import HotpotView
from experiment_core.datasets import DatasetSample
from experiment_core.prompt_contracts import build_json_system_prompt, dataset_instruction_for_sample


DEFAULT_PROMPT_VERSION = "comm_necessary_hotpotqa_v1"


def build_solver_messages(
    sample: DatasetSample,
    view: HotpotView,
    *,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    """构造 split-context HotpotQA 的 Stage A 求解提示词。"""
    _assert_prompt_version(prompt_version)
    user_prompt = (
        f"You are agent_{view.agent_id} in a controlled HotpotQA communication experiment.\n"
        f"View kind: {view.view_kind}.\n"
        f"{dataset_instruction_for_sample(sample, context_scope='visible', hotpot_style='shortest_span')}\n"
        "If you cite evidence, use the exact paragraph title and sentence id shown as (0), (1), ... .\n\n"
        f"Question:\n{sample.question.strip()}\n\n"
        f"Visible context:\n{view.context_text}\n\n"
        "Return exactly one JSON object with keys "
        '{"final_answer":"short answer","reasoning_trace":"brief private reasoning",'
        '"evidence_summary":"short evidence summary",'
        '"supporting_facts":[{"title":"Paragraph title","sent_id":0}],'
        '"confidence_raw":0.0}.\n'
        "If evidence is incomplete, still answer with your best short span and cite only visible sentence ids.\n"
        "Return JSON only."
    )
    return [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def build_belief_update_messages(
    sample: DatasetSample,
    view: HotpotView,
    *,
    previous_answer: str,
    previous_reasoning_trace: str,
    previous_evidence_summary: str,
    peer_packets: list[dict[str, object]],
    method_name: str,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    """构造基于选中 peer packet 的 Stage B 更新提示词。"""
    _assert_prompt_version(prompt_version)
    peer_block = "\n\n".join(
        f"{item['agent']} packet_mode={item['packet_mode']}\npacket={item['packet_text']}"
        for item in peer_packets
    ) or "No peer packet is visible."
    user_prompt = (
        f"You are agent_{view.agent_id} in Stage B of method {method_name}.\n"
        "Use your private context plus selected peer packets. Revise only if peer evidence supplies a missing hop or reveals a mistake.\n"
        "Do not cite sentence ids that are absent from your private context or peer packets.\n\n"
        f"Question:\n{sample.question.strip()}\n\n"
        f"Your private context:\n{view.context_text}\n\n"
        f"Your previous final_answer: {previous_answer}\n"
        f"Your previous reasoning_trace: {previous_reasoning_trace}\n"
        f"Your previous evidence_summary: {previous_evidence_summary}\n\n"
        f"Peer packets:\n{peer_block}\n\n"
        "Return exactly one JSON object with keys "
        '{"changed_answer":true,"final_answer":"short answer","reasoning_trace":"brief revised reasoning",'
        '"evidence_summary":"short revised evidence summary",'
        '"supporting_facts":[{"title":"Paragraph title","sent_id":0}],'
        '"confidence_raw":0.0}.\n'
        "Return JSON only."
    )
    return [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def _system_prompt() -> str:
    return build_json_system_prompt(
        "You are a careful HotpotQA reasoning agent.",
        extra_rules=[
            "Do not add extra keys.",
            "final_answer must be a short answer span.",
            "supporting_facts must use exact visible titles and integer sent_id values.",
            "confidence_raw should be a number in [0, 1].",
        ],
    )


def _assert_prompt_version(prompt_version: str) -> None:
    if prompt_version != DEFAULT_PROMPT_VERSION:
        raise ValueError(f"Unsupported prompt_version: {prompt_version}")
