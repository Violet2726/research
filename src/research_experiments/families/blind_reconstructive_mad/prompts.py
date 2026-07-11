"""BRD-MAD 审查提示词，与 Stage-A 提示词刻意隔离。"""

from __future__ import annotations

from research_experiments.core.controls.control_prompts import build_cot_messages
from research_experiments.core.data.datasets import DatasetSample
from research_experiments.family_runtime.free_text_protocol import (
    build_free_text_answer_instruction,
    build_free_text_system_prompt,
)

BRD_PROMPT_VERSION = "brd_blind_reconstruct_v1"
SGSA_PROMPT_VERSION = "sgsa_concise_synthesis_v1"


def build_stage_a_messages(sample: DatasetSample, agent_id: int) -> list[dict[str, str]]:
    """Exactly reuse the sc_5 free-CoT construction (including its contract)."""

    from research_experiments.core.controls.control_prompts import FREE_TEXT_V1_PROMPT_VERSION

    return build_cot_messages(sample, replicate_id=agent_id, prompt_version=FREE_TEXT_V1_PROMPT_VERSION)


def build_reviewer_messages(
    sample: DatasetSample,
    *,
    candidate_board: str | None,
    method_name: str,
) -> list[dict[str, str]]:
    if method_name == "conditional_resample_3":
        task = (
            "Independently solve the problem from first principles. Do not assume an earlier answer exists. "
            "Check the decisive inference before committing. Do all extended work silently; the visible REASONING "
            "line must be one sentence of at most 40 words."
        )
    elif method_name in {"gsa_quorum_3", "sgsa_unanimous_3", "sgsa_visible_support_3"}:
        task = (
            "Independently synthesize the anonymous candidate derivations and solve the problem yourself. "
            "Do not infer popularity from candidate labels or their order. Identify only the single decisive check "
            "needed to distinguish the candidates. Do all extended work silently; the visible REASONING line "
            "must be one sentence of at most 40 words."
        )
    else:
        task = (
            "For every anonymous candidate, look for its first decisive mathematical or factual error. "
            "Then discard all positions and reconstruct the solution independently before choosing a final answer. "
            "Do not infer popularity from candidate labels or their order. Do all extended work silently; the "
            "visible REASONING line must be one sentence of at most 40 words."
        )
    user = f"{task}\n\nQuestion:\n{sample.question.strip()}\n"
    if sample.prompt_context:
        user += f"\nContext:\n{sample.prompt_context}\n"
    if candidate_board:
        user += f"\nAnonymous candidate board:\n{candidate_board}\n"
    user += "\n" + build_free_text_answer_instruction(sample.dataset)
    return [
        {
            "role": "system",
            "content": build_free_text_system_prompt(
                "You are an independent reviewer in a blinded reasoning experiment.",
                extra_rules=[
                    "Do not report confidence scores or vote counts.",
                    "Output labels in exactly this order: REASONING, FINAL_ANSWER.",
                    "REASONING must state the decisive check or reconstruction.",
                    "Solve silently before writing. REASONING must be exactly one sentence and at most 40 words.",
                    "After that one sentence, immediately write FINAL_ANSWER; never continue the derivation.",
                ],
            ),
        },
        {"role": "user", "content": user},
    ]
