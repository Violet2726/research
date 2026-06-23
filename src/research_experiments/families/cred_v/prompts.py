"""CRED-V 提示词构造。"""

from __future__ import annotations

from typing import Any

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.family_runtime.json_object_protocol import build_json_object_answer_instruction

CRED_PROMPT_VERSION = "cred_v_selective_verify_v1"

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
        f"You are CRED-V agent_{agent_id}.\n"
        "Primary contract: solve independently, audit once, then commit a compact answer card.\n"
        f"Audit lens: {agent_role} - {_ROLE_AUDIT_LENS[agent_role]}\n"
        f"{_strong_solver_workflow(sample.dataset)}\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        "Use the workflow to produce the answer card. "
        "reasoning is three compact clauses: requested slot; decisive check; answer fit. "
        "answer is the committed answer after audit. "
        "key_evidence is the single strongest support. "
        "risk_level names the remaining unresolved risk in that committed answer.\n"
        + build_json_object_answer_instruction(
            sample.dataset,
            extra_json_keys=["answer_type"],
        )
    )
    return [
        {"role": "system", "content": "You are an expert reasoning agent in a controlled CRED-V experiment. Return a JSON answer object."},
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
        "You are the CRED-V refutation verifier. Test whether the challenger defeats the leading answer with one checkable error certificate.\n"
        f"{_compact_audit_workflow(sample.dataset)}\n"
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
        "Test the strongest concrete contradiction, constraint miss, slot error, or calculation error. "
        "In the JSON object, answer is the surviving answer; changed=true when the challenger defeats the leading answer; "
        "attack_strength is high or medium for a checkable defeat and weak or none for a surviving leading answer.\n"
        + build_json_object_answer_instruction(
            sample.dataset,
            extra_json_keys=["changed", "attack_type", "attack_strength"],
        )
    )
    return [
        {"role": "system", "content": "You are a precise refutation agent for controlled CRED-V experiments."},
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
        "You are the CRED-V defense verifier. Decide whether the refutation really defeats the leading answer.\n"
        f"{_compact_audit_workflow(sample.dataset)}\n"
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
        "Compare the refutation with the leading answer. In the JSON object, answer is the surviving answer. "
        "changed=true when the refutation defeats the leading answer; defense_status names defended, corrected, or unresolved.\n"
        + build_json_object_answer_instruction(
            sample.dataset,
            extra_json_keys=["changed", "defense_status"],
        )
    )
    return [
        {"role": "system", "content": "You are a precise defense agent for controlled CRED-V experiments."},
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
        "You are the CRED-V final verifier. Choose the answer that survives the strongest concrete attacks.\n"
        f"{_compact_audit_workflow(sample.dataset)}\n"
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
        "Prefer the pre-debate leading answer. Switch to a challenger when a concrete refutation and the verification packets support that challenger. "
        "In the JSON object, answer is the final answer and source identifies the decisive support.\n"
        + build_json_object_answer_instruction(
            sample.dataset,
            extra_json_keys=["source"],
        )
    )
    return [
        {"role": "system", "content": "You are a compact final judge for controlled CRED-V experiments."},
        {"role": "user", "content": user_prompt},
    ]


def _strong_solver_workflow(dataset: str) -> str:
    return "\n".join(
        [
            "Strong solver workflow:",
            "- Build one coherent derivation.",
            "- Select the decisive check that makes the answer verifiable.",
            f"- {_dataset_solver_focus(dataset)}",
            "- Commit the exact requested answer slot.",
        ]
    )


def _compact_audit_workflow(dataset: str) -> str:
    return "\n".join(
        [
            "Compact audit workflow:",
            "- Identify the requested answer slot.",
            "- Compare the leading answer and challenger against the decisive check.",
            f"- {_dataset_solver_focus(dataset)}",
            "- Commit to the answer that survives the concrete check.",
        ]
    )


def _dataset_solver_focus(dataset: str) -> str:
    if dataset in {"gpqa_diamond", "mmlu", "mmlu_abstract_algebra", "mmlu_pro"}:
        return "For multiple-choice tasks, identify the decisive concept, compare plausible options, and commit to one option letter."
    if dataset in {"gsm8k", "math500", "competition_math"}:
        return "For math tasks, carry out the calculation symbolically or numerically and sanity-check the result."
    if dataset in {"hotpotqa", "webquestions"}:
        return "For short-span tasks, connect the evidence hops and commit to the complete judgeable answer span."
    if dataset == "strategyqa":
        return 'For yes/no tasks, resolve the needed factual subclaims and map them to exactly "yes" or "no".'
    return "Use the task instruction to decide the answer form."


def _dataset_instruction(sample: DatasetSample) -> str:
    if sample.dataset == "strategyqa":
        return 'Answer with exactly "yes" or "no". The JSON answer field is exactly "yes" or "no".'
    if sample.dataset in {"gpqa_diamond", "mmlu", "mmlu_abstract_algebra", "mmlu_pro"}:
        return 'Choose the single best option. The JSON answer field is exactly the option letter, such as "A" or "B".'
    if sample.dataset in {"math500", "competition_math"}:
        return "Solve the math problem carefully. The JSON answer field is only the final mathematical expression."
    if sample.dataset == "gsm8k":
        return "Solve the math problem carefully. The JSON answer field is only the final numeric answer without commas or units."
    if sample.dataset == "hotpotqa":
        return (
            "Answer the multi-hop question using only the provided context. "
            "The JSON answer field is the complete judgeable text span. "
            "Prefer copying exact wording from the context when possible. "
            "Include essential type or unit words such as language, students, episodes, title, city, or organization when they identify the answer span."
        )
    if sample.dataset == "webquestions":
        return (
            "Answer the graph question using only the provided graph evidence. "
            "The JSON answer field is the complete judgeable entity span or literal answer."
        )
    return "Use the task instruction to decide the JSON answer field."


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
        f"evidence=`{_clip(payload.get('key_evidence') or row.get('key_evidence') or 'n/a', 180)}`, "
        f"risk_level={payload.get('risk_level') or row.get('risk_level') or 'unknown'}, "
        f"risk_summary=`{_clip(payload.get('risk_summary') or row.get('failure_risk') or 'n/a', 120)}`"
    )


def _clip(value: object, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."
