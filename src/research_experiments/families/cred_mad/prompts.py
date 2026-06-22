"""CRED-MAD 提示词构造。"""

from __future__ import annotations

from typing import Any

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.core.prompts.dataset_contracts import dataset_instruction_for_sample
from research_experiments.family_runtime.json_tail_protocol import build_json_tail_answer_instruction

CRED_PROMPT_VERSION = "cred_mad_json_tail_v1"

AGENT_ROLES = (
    "cot_builder",
    "decomposer",
    "constraint_skeptic",
    "evidence_slot_verifier",
    "counterfactual_falsifier",
)

_ROLE_GUIDANCE = {
    "cot_builder": "Solve directly with a compact chain of reasoning and a final self-check.",
    "decomposer": "Break the problem into subclaims or substeps, solve each, then compose the answer.",
    "constraint_skeptic": "Focus on legal answer format, units, options, hidden constraints, and common traps.",
    "evidence_slot_verifier": "Anchor the answer to the exact requested slot, evidence span, option, or calculation.",
    "counterfactual_falsifier": "Try to refute the most obvious answer; if it survives, state why it survives.",
}


def build_stage_a_messages(sample: DatasetSample, *, agent_id: int, agent_role: str) -> list[dict[str, str]]:
    user_prompt = (
        f"You are CRED-MAD agent_{agent_id}.\n"
        f"Role: {agent_role}\n"
        f"Contract: {_ROLE_GUIDANCE[agent_role]}\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        "Before the final JSON block, include concise reasoning plus these contract fields in prose: "
        "answer contract, key evidence, and failure risk.\n"
        + build_json_tail_answer_instruction(
            sample.dataset,
            extra_json_keys=["answer_type", "key_evidence", "failure_risk"],
        )
    )
    return [
        {"role": "system", "content": "You are an expert reasoning agent in a controlled CRED-MAD experiment."},
        {"role": "user", "content": user_prompt},
    ]


def build_refutation_messages(
    sample: DatasetSample,
    *,
    leading_answer: str,
    target_row: dict[str, Any],
    stage_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    user_prompt = (
        "You are the CRED-MAD refuter. Your job is not to chat; it is to produce one checkable attack.\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        f"Current leading answer: `{leading_answer or 'unknown'}`.\n"
        f"Best challenger packet: {_format_packet(target_row)}\n"
        "Stage A board:\n"
        f"{_format_board(stage_rows)}\n"
        "Attack the leading answer only if you can give a concrete contradiction, constraint miss, slot error, or calculation error. "
        "Otherwise defend the leading answer and keep it.\n"
        + build_json_tail_answer_instruction(
            sample.dataset,
            extra_json_keys=["changed", "attack_type", "key_evidence"],
        )
    )
    return [
        {"role": "system", "content": "You are a precise refutation agent for controlled CRED-MAD experiments."},
        {"role": "user", "content": user_prompt},
    ]


def build_defense_messages(
    sample: DatasetSample,
    *,
    leading_answer: str,
    refutation_row: dict[str, Any],
    stage_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    user_prompt = (
        "You are the CRED-MAD defender. Test whether the refutation really defeats the leading answer.\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        f"Leading answer to defend: `{leading_answer or 'unknown'}`.\n"
        f"Refutation packet: {_format_packet(refutation_row)}\n"
        "Stage A board:\n"
        f"{_format_board(stage_rows)}\n"
        "If the refutation is stronger, accept the corrected answer. If not, keep the leading answer.\n"
        + build_json_tail_answer_instruction(
            sample.dataset,
            extra_json_keys=["changed", "defense_status", "key_evidence"],
        )
    )
    return [
        {"role": "system", "content": "You are a precise defense agent for controlled CRED-MAD experiments."},
        {"role": "user", "content": user_prompt},
    ]


def build_judge_messages(
    sample: DatasetSample,
    *,
    leading_answer: str,
    stage_rows: list[dict[str, Any]],
    refutation_rows: list[dict[str, Any]],
    defense_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    user_prompt = (
        "You are the CRED-MAD judge. Choose the answer that survives the strongest concrete attacks.\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        f"Pre-debate leading answer: `{leading_answer or 'unknown'}`.\n"
        "Stage A board:\n"
        f"{_format_board(stage_rows)}\n"
        "Refutations:\n"
        f"{_format_board(refutation_rows)}\n"
        "Defenses:\n"
        f"{_format_board(defense_rows)}\n"
        "Prefer the pre-debate leading answer unless a concrete attack remains undefeated.\n"
        + build_json_tail_answer_instruction(
            sample.dataset,
            extra_json_keys=["source", "key_evidence"],
        )
    )
    return [
        {"role": "system", "content": "You are a compact final judge for controlled CRED-MAD experiments."},
        {"role": "user", "content": user_prompt},
    ]


def _dataset_instruction(sample: DatasetSample) -> str:
    return dataset_instruction_for_sample(sample, hotpot_style="shortest_span_copy")


def _format_board(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "- none\n"
    return "\n".join(f"- {_format_packet(row)}" for row in rows) + "\n"


def _format_packet(row: dict[str, Any]) -> str:
    payload = row.get("validated_output") if isinstance(row.get("validated_output"), dict) else {}
    answer = str(row.get("normalized_answer") or row.get("prediction") or payload.get("answer") or "unknown")
    return (
        f"{row.get('agent_role') or row.get('role') or 'agent'}: answer=`{answer}`, "
        f"confidence={row.get('confidence_value') if row.get('confidence_value') is not None else payload.get('confidence', 'unknown')}, "
        f"evidence=`{payload.get('key_evidence') or row.get('key_evidence') or 'n/a'}`, "
        f"risk=`{payload.get('failure_risk') or row.get('failure_risk') or 'n/a'}`"
    )
