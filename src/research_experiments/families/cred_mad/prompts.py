"""CRED-MAD 提示词构造。"""

from __future__ import annotations

from typing import Any

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.core.prompts.dataset_contracts import dataset_instruction_for_sample
from research_experiments.family_runtime.json_tail_protocol import build_json_tail_answer_instruction

CRED_PROMPT_VERSION = "cred_mad_strong_solver_audit_v3"

AGENT_ROLES = (
    "cot_builder",
    "decomposer",
    "constraint_skeptic",
    "evidence_slot_verifier",
    "counterfactual_falsifier",
)

_ROLE_AUDIT_LENS = {
    "cot_builder": "Check that the final answer follows from one coherent derivation.",
    "decomposer": "Check that the subclaims or substeps are solved and recomposed correctly.",
    "constraint_skeptic": "Check the requested answer format, units, options, hidden constraints, and common traps.",
    "evidence_slot_verifier": "Check the exact requested slot, evidence span, option clue, or calculation that supports the answer.",
    "counterfactual_falsifier": "Check the selected answer against the strongest plausible alternative and keep the answer that survives.",
}


def build_stage_a_messages(sample: DatasetSample, *, agent_id: int, agent_role: str) -> list[dict[str, str]]:
    user_prompt = (
        f"You are CRED-MAD agent_{agent_id}.\n"
        "Primary contract: first solve as a strong independent single-agent reasoner; then apply your audit lens.\n"
        f"Audit lens: {agent_role} - {_ROLE_AUDIT_LENS[agent_role]}\n"
        f"{_strong_solver_workflow(sample.dataset)}\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        "Write the concise reasoning that leads to your answer, then state how the audit lens affected the commitment. "
        "In the JSON object, answer is the committed answer after the audit lens; key_evidence is the decisive support; "
        "risk_level is the remaining unresolved risk in that committed answer.\n"
        + build_json_tail_answer_instruction(
            sample.dataset,
            extra_json_keys=["answer_type"],
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
        "You are the CRED-MAD refuter. Produce one checkable attack after solving the task independently.\n"
        f"{_strong_solver_workflow(sample.dataset)}\n"
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
        "Solve independently, compare the leading answer with the challenger packet, and test the strongest concrete "
        "contradiction, constraint miss, slot error, or calculation error. In the final JSON, answer is the surviving "
        "answer. Set changed=true only when answer replaces the leading answer; set attack_strength=none when the "
        "leading answer survives.\n"
        + build_json_tail_answer_instruction(
            sample.dataset,
            extra_json_keys=["changed", "attack_type", "attack_strength"],
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
        "You are the CRED-MAD defender. Solve independently, then test whether the refutation defeats the leading answer.\n"
        f"{_strong_solver_workflow(sample.dataset)}\n"
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
        "Compare the refutation with the leading answer. In the final JSON, answer is the surviving answer. "
        "Set changed=true only when the refutation defeats the leading answer; use risk_level for the remaining risk.\n"
        + build_json_tail_answer_instruction(
            sample.dataset,
            extra_json_keys=["changed", "defense_status"],
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
        "You are the CRED-MAD judge. Solve independently, then choose the answer that survives the strongest concrete attacks.\n"
        f"{_strong_solver_workflow(sample.dataset)}\n"
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
        "Choose the answer with the strongest surviving concrete evidence. In the final JSON, answer is that final answer; "
        "source identifies where the decisive support came from.\n"
        + build_json_tail_answer_instruction(
            sample.dataset,
            extra_json_keys=["source"],
        )
    )
    return [
        {"role": "system", "content": "You are a compact final judge for controlled CRED-MAD experiments."},
        {"role": "user", "content": user_prompt},
    ]


def _strong_solver_workflow(dataset: str) -> str:
    return "\n".join(
        [
            "Strong solver workflow:",
            "- Build one coherent derivation before committing to an answer.",
            "- Keep every decisive inference explicit enough to verify.",
            f"- {_dataset_solver_focus(dataset)}",
            "- Check that the final answer fills exactly the requested answer slot.",
        ]
    )


def _dataset_solver_focus(dataset: str) -> str:
    if dataset in {"gpqa_diamond", "mmlu", "mmlu_abstract_algebra", "mmlu_pro"}:
        return "For multiple-choice tasks, identify the decisive concept, compare plausible options, and commit to one option letter."
    if dataset in {"gsm8k", "math500", "competition_math"}:
        return "For math tasks, carry out the calculation symbolically or numerically and sanity-check the result."
    if dataset in {"hotpotqa", "webquestions"}:
        return "For short-span tasks, connect the evidence hops and commit to the shortest judgeable span."
    if dataset == "strategyqa":
        return 'For yes/no tasks, resolve the needed factual subclaims and map them to exactly "yes" or "no".'
    return "Use the task instruction to decide the answer form."


def _dataset_instruction(sample: DatasetSample) -> str:
    instruction = dataset_instruction_for_sample(sample, hotpot_style="shortest_span_copy")
    return instruction.replace("The final_answer must", "The JSON answer field must").replace(
        "final_answer",
        "JSON answer field",
    )


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
        f"risk_level={payload.get('risk_level') or row.get('risk_level') or 'unknown'}, "
        f"risk_summary=`{payload.get('risk_summary') or row.get('failure_risk') or 'n/a'}`"
    )
