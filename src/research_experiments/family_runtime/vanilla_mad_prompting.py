"""共享 vanilla MAD comparator 的 SC 对齐自由文本 prompt。"""

from __future__ import annotations

from research_experiments.core.controls.control_prompts import FREE_TEXT_V1_PROMPT_VERSION, build_cot_messages
from research_experiments.core.data.datasets import DatasetSample
from research_experiments.core.prompts.dataset_contracts import dataset_instruction_for_sample
from research_experiments.family_runtime.free_text_protocol import build_free_text_system_prompt

SC_ALIGNED_MAD_FREE_TEXT_PROMPT_VERSION = "sc_aligned_mad_free_text_v1"
CONSISTENT_FREE_TEXT_PROMPT_VERSION = SC_ALIGNED_MAD_FREE_TEXT_PROMPT_VERSION
CONTROLLED_PROMPT_VERSION = SC_ALIGNED_MAD_FREE_TEXT_PROMPT_VERSION
DEFAULT_PROMPT_VERSION = SC_ALIGNED_MAD_FREE_TEXT_PROMPT_VERSION
SUPPORTED_SHARED_PROMPT_VERSIONS = (SC_ALIGNED_MAD_FREE_TEXT_PROMPT_VERSION,)


def build_initial_messages(
    sample: DatasetSample,
    agent_id: int,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    _ensure_prompt_version(prompt_version)
    return build_cot_messages(sample, agent_id, FREE_TEXT_V1_PROMPT_VERSION)


def build_debate_messages(
    sample: DatasetSample,
    agent_id: int,
    round_index: int,
    previous_reasoning: str,
    previous_answer: str,
    peer_messages: list[dict[str, str]],
    previous_response_text: str = "",
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    stage_a_majority_answer: str = "",
    stage_a_vote_counts: dict[str, int] | None = None,
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
    stage_a_majority = str(stage_a_majority_answer or "").strip() or "[unknown]"
    vote_counts = stage_a_vote_counts or {}
    vote_count_text = ", ".join(f"{answer}:{count}" for answer, count in sorted(vote_counts.items()))
    vote_count_text = vote_count_text or "[unknown]"
    user_prompt = (
        f"You are agent_{agent_id} in debate round {round_index}.\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        f"Stage A majority answer: {stage_a_majority}\n"
        f"Stage A vote counts: {vote_count_text}\n\n"
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
        raise ValueError(f"Unsupported shared MAD prompt_version: {prompt_version}")


def _system_prompt() -> str:
    return build_free_text_system_prompt(
        "You are one reasoning agent in a controlled multi-agent debate experiment.",
        extra_rules=[
            "Solve the task carefully using only the provided question and context.",
            "Output the labels in exactly this order: REASONING, MAJORITY_ERROR, FINAL_ANSWER.",
            "REASONING is required.",
            "MAJORITY_ERROR must be none unless you identify a concrete error in the Stage A majority answer.",
            "Keep REASONING concise, but include enough detail to justify or revise the answer.",
            "If your reasoning changes the answer, rewrite FINAL_ANSWER to the corrected answer.",
            "Do not add natural-language text before or after the tagged lines.",
        ],
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
            "with a country name or other higher-level category. "
            "If you move away from the Stage A majority, MAJORITY_ERROR must name the concrete mistake."
        )
    return (
        "Revise your answer only if peer arguments reveal a concrete mistake or stronger evidence. "
        "If you move away from the Stage A majority, MAJORITY_ERROR must name the concrete mistake. "
        "If you fix the reasoning, you must also rewrite FINAL_ANSWER."
    )


def _debate_output_contract_instruction(*, dataset: str) -> str:
    lines = [
        "Return only the following three lines, in this exact order, with no markdown fences:",
        "REASONING: <required concise reasoning>",
        "MAJORITY_ERROR: <concrete Stage A majority error, or none>",
        "FINAL_ANSWER: <canonical answer>",
        "Rules:",
        "- REASONING is required.",
        "- MAJORITY_ERROR is required; use exactly none when retaining the Stage A majority.",
        "- Keep REASONING concise, but include enough detail to justify or revise the answer.",
        "- If your reasoning changes the answer, rewrite FINAL_ANSWER to the corrected answer.",
        "- Use plain text only in REASONING. Do not use LaTeX commands, backslashes, or markdown.",
        "- FINAL_ANSWER must contain only the final answer and nothing else.",
    ]
    if dataset in {"gpqa_diamond", "mmlu", "mmlu_abstract_algebra", "mmlu_pro"}:
        lines.append('- FINAL_ANSWER must be exactly one option letter such as "A" or "B".')
    elif dataset in {"gsm8k", "math500", "competition_math"}:
        lines.extend(
            [
                "- FINAL_ANSWER must use plain ASCII math only.",
                "- Do not use LaTeX commands such as \\frac, \\sqrt, \\left, or \\right.",
            ]
        )
    elif dataset in {"hotpotqa", "webquestions"}:
        lines.extend(
            [
                "- FINAL_ANSWER must be the shortest judgeable text span.",
                "- Do not add category words, explanations, or extra qualifiers.",
            ]
        )
    return "\n".join(lines)
