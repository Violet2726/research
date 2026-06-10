"""共享 vanilla MAD comparator 的标准 prompt。"""

from __future__ import annotations

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.core.prompts.dataset_contracts import build_json_system_prompt, dataset_instruction_for_sample

DEFAULT_PROMPT_VERSION = "multi_agent_debate_json"
CONTROLLED_PROMPT_VERSION = "multi_agent_controlled_json"
PAPER_PROMPT_VERSION = "multi_agent_paper_text"
SUPPORTED_SHARED_PROMPT_VERSIONS = (
    CONTROLLED_PROMPT_VERSION,
    PAPER_PROMPT_VERSION,
)


def build_initial_messages(
    sample: DatasetSample,
    agent_id: int,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    if prompt_version == PAPER_PROMPT_VERSION:
        return [
            {"role": "system", "content": _system_prompt(prompt_version)},
            {"role": "user", "content": _paper_initial_prompt(sample, agent_id)},
        ]
    user_prompt = (
        f"You are agent_{agent_id}.\n"
        f"{_dataset_instruction(sample, prompt_version)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        'Return exactly one JSON object with key "final_answer". '
        'You may optionally include "reasoning". '
        "If you include reasoning, keep it under 60 words. "
        "Do not add any other keys. "
        "Return JSON only."
    )
    return [
        {"role": "system", "content": _system_prompt(prompt_version)},
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
    if prompt_version == PAPER_PROMPT_VERSION:
        return [
            {"role": "system", "content": _system_prompt(prompt_version)},
            {
                "role": "user",
                "content": _paper_debate_prompt(
                    sample,
                    agent_id,
                    round_index,
                    previous_response_text=previous_response_text,
                    peer_messages=peer_messages,
                ),
            },
        ]
    peer_block = (
        "\n\n".join(
            f"{item['agent']} previous answer: {item['answer']}\n{item['agent']} reasoning: {item['reasoning']}"
            for item in peer_messages
        )
        or "No peer feedback."
    )
    user_prompt = (
        f"You are agent_{agent_id} in debate round {round_index}.\n"
        f"{_dataset_instruction(sample, prompt_version)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        f"Your previous reasoning: {previous_reasoning}\n"
        f"Your previous final_answer: {previous_answer}\n\n"
        f"Peer feedback:\n{peer_block}\n\n"
        f"{_revision_instruction(sample, prompt_version)} "
        'Return exactly one JSON object with key "final_answer". '
        'You may optionally include "reasoning". '
        "If you include reasoning, keep it under 60 words. "
        "Do not add any other keys. "
        "Return JSON only."
    )
    return [
        {"role": "system", "content": _system_prompt(prompt_version)},
        {"role": "user", "content": user_prompt},
    ]


def _system_prompt(prompt_version: str) -> str:
    if prompt_version == CONTROLLED_PROMPT_VERSION:
        return build_json_system_prompt(
            "You are one reasoning agent in a controlled debate-vs-vote experiment.",
            extra_rules=[
                "Solve the task carefully using only the provided question and context.",
                "Keep optional reasoning compact and outcome-focused.",
                "Do not add natural-language text before or after the JSON object.",
                "Do not add labels, category words, or explanatory suffixes to final_answer.",
            ],
        )
    if prompt_version == PAPER_PROMPT_VERSION:
        return (
            "You are one reasoning agent in a multi-agent debate experiment.\n"
            "Solve the task carefully.\n"
            "Explain your reasoning before committing to the final answer.\n"
            "Do not use markdown fences unless you are quoting another agent response."
        )
    if prompt_version != DEFAULT_PROMPT_VERSION:
        raise ValueError(f"Unsupported multi-agent prompt_version: {prompt_version}")
    return build_json_system_prompt(
        "You are one reasoning agent in a controlled multi-agent debate experiment.",
        extra_rules=[
            "Solve the task carefully.",
            "Keep optional reasoning compact and outcome-focused.",
            "Do not add natural-language text before or after the JSON object.",
        ],
    )


def _dataset_instruction(sample: DatasetSample, prompt_version: str) -> str:
    if sample.dataset == "hotpotqa":
        return dataset_instruction_for_sample(sample, hotpot_style="short_span")
    return dataset_instruction_for_sample(sample)


def _revision_instruction(sample: DatasetSample, prompt_version: str) -> str:
    if sample.dataset == "hotpotqa" and prompt_version == CONTROLLED_PROMPT_VERSION:
        return (
            "Revise your answer only if peer arguments reveal a concrete mistake or provide stronger textual evidence. "
            "If the peer answer differs only by added labels, category words, or formatting, prefer the shortest "
            "context-grounded span."
        )
    return "Revise your reasoning only if peer arguments reveal a concrete mistake or stronger evidence."


def prompt_version_uses_json_response_format(prompt_version: str) -> bool:
    if prompt_version == PAPER_PROMPT_VERSION:
        return False
    if prompt_version in {CONTROLLED_PROMPT_VERSION, DEFAULT_PROMPT_VERSION}:
        return True
    raise ValueError(f"Unsupported multi-agent prompt_version: {prompt_version}")


def _paper_initial_prompt(sample: DatasetSample, agent_id: int) -> str:
    parts = [
        f"You are agent_{agent_id}.",
        _paper_task_instruction(sample),
        f"Question:\n{sample.question.strip()}",
    ]
    if sample.prompt_context:
        parts.append(f"Context:\n{sample.prompt_context}")
    parts.append(_paper_answer_format_instruction(sample))
    return "\n\n".join(parts)


def _paper_debate_prompt(
    sample: DatasetSample,
    agent_id: int,
    round_index: int,
    *,
    previous_response_text: str,
    peer_messages: list[dict[str, str]],
) -> str:
    peer_block = "\n\n".join(
        f"One agent solution: ```{item.get('response_text', '').strip()}```"
        for item in peer_messages
        if str(item.get("response_text", "")).strip()
    )
    parts = [
        f"You are agent_{agent_id} in debate round {round_index}.",
        _paper_task_instruction(sample),
        f"Question:\n{sample.question.strip()}",
    ]
    if sample.prompt_context:
        parts.append(f"Context:\n{sample.prompt_context}")
    if previous_response_text.strip():
        parts.append(f"Your previous solution: ```{previous_response_text.strip()}```")
    if peer_block:
        parts.append(f"These are the solutions to the problem from other agents:\n\n{peer_block}")
    parts.append(
        "Using the reasoning from other agents as additional advice, can you give an updated answer? "
        "Examine your solution and the other agents step by step."
    )
    parts.append(_paper_answer_format_instruction(sample))
    return "\n\n".join(parts)


def _paper_task_instruction(sample: DatasetSample) -> str:
    if sample.dataset in {"gsm8k", "math500", "competition_math"}:
        return "Can you solve the following math problem? Explain your reasoning."
    if sample.dataset in {"mmlu", "mmlu_abstract_algebra", "mmlu_pro", "gpqa_diamond"}:
        return "Can you answer the following question as accurately as possible? Explain your answer."
    if sample.dataset == "hotpotqa":
        return (
            "Can you answer the following multi-hop question using only the provided context? Explain your reasoning."
        )
    if sample.dataset == "strategyqa":
        return "Can you answer the following yes-or-no question as accurately as possible? Explain your reasoning."
    return "Can you solve the following problem carefully? Explain your reasoning."


def _paper_answer_format_instruction(sample: DatasetSample) -> str:
    if sample.dataset == "gsm8k":
        return (
            "Your final answer should be a single numerical number, in the form \\boxed{answer}, "
            "at the end of your response."
        )
    if sample.dataset in {"math500", "competition_math"}:
        return (
            "Your final answer should be a single mathematical expression, in the form \\boxed{answer}, "
            "at the end of your response."
        )
    if sample.dataset in {"mmlu", "mmlu_abstract_algebra", "mmlu_pro", "gpqa_diamond"}:
        return "Put your final answer in the form (X) at the end of your response."
    if sample.dataset == "strategyqa":
        return 'State your final answer as "yes" or "no" at the end of your response.'
    if sample.dataset == "hotpotqa":
        return "Put your final answer in the form Answer: <short span> at the end of your response."
    return "State your final answer explicitly at the end of your response."
