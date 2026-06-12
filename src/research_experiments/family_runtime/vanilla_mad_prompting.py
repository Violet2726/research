"""共享 vanilla MAD comparator 的强化 JSON prompt。"""

from __future__ import annotations

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.core.prompts.dataset_contracts import build_json_system_prompt, dataset_instruction_for_sample
from research_experiments.family_runtime.free_text_protocol import build_free_text_system_prompt

CONSISTENT_FREE_TEXT_PROMPT_VERSION = "multi_agent_free_text_v1"
CONTROLLED_PROMPT_VERSION = CONSISTENT_FREE_TEXT_PROMPT_VERSION
DEFAULT_PROMPT_VERSION = CONSISTENT_FREE_TEXT_PROMPT_VERSION
SUPPORTED_SHARED_PROMPT_VERSIONS = (CONSISTENT_FREE_TEXT_PROMPT_VERSION,)


def build_initial_messages(
    sample: DatasetSample,
    agent_id: int,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    _ensure_prompt_version(prompt_version)
    user_prompt = (
        f"You are agent_{agent_id}.\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += _json_output_contract_instruction()
    return [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def build_debate_messages(
    sample: DatasetSample,
    agent_id: int,
    round_index: int,
    previous_reasoning: str,
    previous_answer: str,
    peer_messages: list[dict[str, str]],
    previous_response_text: str = "",
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    del previous_response_text
    _ensure_prompt_version(prompt_version)
    peer_block = (
        "\n\n".join(
            f"{item['agent']} previous final_answer: {item['answer']}\n"
            f"{item['agent']} previous reasoning: {item['reasoning'] or '[missing reasoning]'}"
            for item in peer_messages
        )
        or "No peer feedback."
    )
    user_prompt = (
        f"You are agent_{agent_id} in debate round {round_index}.\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        f"Your previous reasoning: {previous_reasoning or '[missing reasoning]'}\n"
        f"Your previous final_answer: {previous_answer or '[missing answer]'}\n\n"
        f"Peer feedback:\n{peer_block}\n\n"
        f"{_revision_instruction(sample)}\n\n"
        f"{_debate_output_contract_instruction(dataset=sample.dataset)}"
    )
    return [
        {"role": "system", "content": _system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def prompt_version_uses_json_response_format(prompt_version: str) -> bool:
    _ensure_prompt_version(prompt_version)
    return False


def _ensure_prompt_version(prompt_version: str) -> None:
    if prompt_version not in SUPPORTED_SHARED_PROMPT_VERSIONS:
        raise ValueError(f"Unsupported multi-agent prompt_version: {prompt_version}")


def _system_prompt() -> str:
    return build_free_text_system_prompt(
        "You are one reasoning agent in a controlled multi-agent debate experiment.",
    )


def _dataset_instruction(sample: DatasetSample) -> str:
    if sample.dataset == "hotpotqa":
        return dataset_instruction_for_sample(sample, hotpot_style="shortest_span_copy")
    return dataset_instruction_for_sample(sample)


def _revision_instruction(sample: DatasetSample) -> str:
    if sample.dataset == "hotpotqa":
        return (
            "Revise your answer only if peer arguments reveal a concrete mistake or stronger textual evidence. "
            "Prefer the shortest judgeable span copied from context, and avoid replacing a nationality adjective "
            "with a country name or other higher-level category."
        )
    return (
        "Revise your answer only if peer arguments reveal a concrete mistake or stronger evidence. "
        "If you fix the reasoning, you must also rewrite final_answer."
    )


def _json_output_contract_instruction() -> str:
    return (
        "Return only the following two lines, in this exact order, with no markdown fences:\n"
        "FINAL_ANSWER: <answer only>\n"
        "REASON: <one short plain-text sentence>\n"
        "Rules:\n"
        "- FINAL_ANSWER must contain only the final answer.\n"
        "- REASON must be one short plain-text sentence.\n"
        "- Do not use LaTeX commands or backslashes in REASON."
    )


def _debate_output_contract_instruction(*, dataset: str) -> str:
    del dataset
    return (
        "Return only the following three lines, in this exact order, with no markdown fences:\n"
        "DECISION: <keep or revise>\n"
        "FINAL_ANSWER: <answer only>\n"
        "REASON: <one short plain-text sentence>\n"
        "Rules:\n"
        "- Use DECISION: keep when peer feedback does not change your answer.\n"
        "- Use DECISION: revise only when peer feedback changes your final answer.\n"
        "- FINAL_ANSWER must contain only the final answer.\n"
        "- REASON must be one short plain-text sentence.\n"
        "- Do not use LaTeX commands or backslashes in REASON."
    )
