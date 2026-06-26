"""CRED-V 任务验证提示词构造。"""

from __future__ import annotations

from typing import Any

from research_experiments.core.controls.control_prompts import FREE_TEXT_V1_PROMPT_VERSION, build_cot_messages
from research_experiments.core.data.datasets import DatasetSample
from research_experiments.family_runtime.free_text_protocol import FREE_TEXT_ANSWER_PROTOCOL_V1
from research_experiments.family_runtime.json_object_protocol import build_json_object_answer_instruction

CRED_PROMPT_VERSION = "cred_verify_safe_v1"

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


def build_stage_a_messages(
    sample: DatasetSample,
    *,
    agent_id: int,
    agent_role: str,
    output_protocol: str | None = None,
) -> list[dict[str, str]]:
    if output_protocol == FREE_TEXT_ANSWER_PROTOCOL_V1:
        del agent_role
        return build_cot_messages(sample, agent_id, FREE_TEXT_V1_PROMPT_VERSION)

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


def build_task_verifier_messages(
    sample: DatasetSample,
    *,
    leading_answer: str,
    target_row: dict[str, Any],
    stage_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    challenger_answer = _row_answer(target_row)
    user_prompt = (
        "You are the CRED-V task verifier. Run one decisive task check that compares the current leader with one challenger.\n"
        f"{_task_verifier_workflow(sample.dataset)}\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        f"Current leader answer: `{leading_answer or 'unknown'}`.\n"
        f"Challenger answer: `{challenger_answer or 'unknown'}`.\n"
        f"Challenger packet: {_format_packet(target_row)}\n"
        "Stage A board:\n"
        f"{_format_board(stage_rows)}\n"
        "Promotion certificate: promote=true when the challenger passes the decisive task check and the leader fails that same check. "
        "promote=false when the leader passes, both answers remain unresolved, or both answers pass equivalently. "
        "answer is the selected final answer. leader_score and challenger_score are 0.0 to 1.0 scores for the decisive check. "
        "confidence is confidence in the selected final answer. key_evidence names the concrete check result that decides the certificate. "
        "verification_type is a short label such as option_concept, calculation, span_match, factual_mapping, or format_slot. "
        "verdict is one compact sentence explaining why the selected answer survives.\n"
        + build_json_object_answer_instruction(
            sample.dataset,
            extra_json_keys=["promote", "verification_type", "leader_score", "challenger_score", "verdict"],
        )
    )
    return [
        {"role": "system", "content": "You are a precise task verifier for controlled CRED-V experiments. Return one JSON answer object."},
        {"role": "user", "content": user_prompt},
    ]


def build_safe_hetero_verifier_messages(
    sample: DatasetSample,
    *,
    leading_answer: str,
    target_row: dict[str, Any],
    stage_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    challenger_answer = _row_answer(target_row)
    user_prompt = (
        "You are the CRED-V independent verifier. Produce a verification certificate for one leader/challenger pair.\n"
        f"{_safe_verifier_workflow(sample.dataset)}\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        f"Leader answer: `{leading_answer or 'unknown'}`.\n"
        f"Challenger answer: `{challenger_answer or 'unknown'}`.\n"
        f"Challenger packet: {_format_packet(target_row)}\n"
        "Stage A board:\n"
        f"{_format_board(stage_rows)}\n"
        "Certificate fields: leader_pass is true when the leader satisfies the decisive check. "
        "challenger_pass is true when the challenger satisfies that same check. "
        "promote is true when challenger_pass is true and leader_pass is false. "
        "answer is the answer that passes the certificate. "
        "leader_failure names the concrete failed check for the leader. "
        "challenger_support names the concrete support for the challenger. "
        "key_evidence is the decisive external clue, equation, context span, or option contrast. "
        "confidence is confidence in this certificate, not confidence in Stage A popularity.\n"
        + build_json_object_answer_instruction(
            sample.dataset,
            extra_json_keys=[
                "promote",
                "leader_pass",
                "challenger_pass",
                "verification_type",
                "leader_failure",
                "challenger_support",
                "verdict",
            ],
        )
    )
    return [
        {"role": "system", "content": "You are an independent verifier in a controlled CRED-V experiment. Return one JSON answer object."},
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


def _task_verifier_workflow(dataset: str) -> str:
    return "\n".join(
        [
            "Task verifier workflow:",
            "- Identify the requested answer slot.",
            "- Apply one decisive check to the leader and the challenger.",
            f"- {_dataset_verifier_focus(dataset)}",
            "- Score each answer by how well it passes that check.",
            "- Select the answer with the stronger verified fit.",
        ]
    )


def _safe_verifier_workflow(dataset: str) -> str:
    return "\n".join(
        [
            "Independent verifier workflow:",
            "- Identify the requested answer slot.",
            "- Run one decisive check against the leader and the challenger.",
            f"- {_dataset_verifier_focus(dataset)}",
            "- Mark leader_pass and challenger_pass from the same check.",
            "- Select the answer that is independently verified by the certificate.",
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


def _dataset_verifier_focus(dataset: str) -> str:
    if dataset in {"gpqa_diamond", "mmlu", "mmlu_abstract_algebra", "mmlu_pro"}:
        return "For multiple-choice tasks, compare the leader letter and challenger letter by the decisive concept or option clue."
    if dataset in {"gsm8k", "math500", "competition_math"}:
        return "For math tasks, recompute the critical step, check equivalence, and score the requested final expression."
    if dataset in {"hotpotqa", "webquestions"}:
        return "For short-span tasks, match each answer to the necessary evidence span and score completeness of the judgeable span."
    if dataset == "strategyqa":
        return 'For yes/no tasks, verify the needed factual mapping and score the exact "yes" or "no" answer.'
    return "Use the task instruction to compare answer fit."


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
    answer = _row_answer(row) or str(payload.get("answer") or "unknown")
    return (
        f"{row.get('agent_role') or row.get('role') or 'agent'}: answer=`{answer}`, "
        f"confidence={row.get('confidence_value') if row.get('confidence_value') is not None else payload.get('confidence', 'unknown')}, "
        f"evidence=`{_clip(payload.get('key_evidence') or row.get('key_evidence') or payload.get('reasoning') or 'n/a', 220)}`, "
        f"risk_level={payload.get('risk_level') or row.get('risk_level') or 'unknown'}, "
        f"risk_summary=`{_clip(payload.get('risk_summary') or row.get('failure_risk') or 'n/a', 120)}`"
    )


def _row_answer(row: dict[str, Any]) -> str:
    return str(row.get("normalized_answer") or row.get("prediction") or "").strip()


def _clip(value: object, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."
