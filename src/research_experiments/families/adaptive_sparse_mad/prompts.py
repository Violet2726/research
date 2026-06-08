"""A-SMAD prompts."""

from __future__ import annotations

from research_experiments.core.controls.control_prompts import build_cot_messages
from research_experiments.core.data.datasets import DatasetSample
from research_experiments.core.prompts.dataset_contracts import build_json_system_prompt, dataset_instruction_for_sample
from research_experiments.family_runtime.reasoning_methods import resolve_reasoning_method

STAGE_A_V2_PROMPT_VERSION = "adaptive_sparse_mad_v2_task_schema"
STAGE_A_V4_PROMPT_VERSION = "adaptive_sparse_mad_v4_evidence_gate"
DEFAULT_PROMPT_VERSION = STAGE_A_V2_PROMPT_VERSION
_SUPPORTED_PROMPT_VERSIONS = {
    STAGE_A_V2_PROMPT_VERSION,
    STAGE_A_V4_PROMPT_VERSION,
}
SOLVER_MODES = ("solver_cot", "solver_l2m", "solver_skeptic")
ADAPTIVE_ADDON_SOLVER_MODES = ("solver_verify", "solver_option_elim", "solver_evidence", "solver_slot_contrast")


def build_stage_a_messages(
    sample: DatasetSample,
    *,
    solver_mode: str,
    agent_id: int,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    _ensure_prompt_version(prompt_version)
    if solver_mode == "solver_cot" and prompt_version == STAGE_A_V2_PROMPT_VERSION:
        return build_cot_messages(sample, agent_id, None)
    if prompt_version == STAGE_A_V4_PROMPT_VERSION:
        return _build_stage_a_v4_messages(sample, solver_mode=solver_mode, agent_id=agent_id)
    return _build_stage_a_v2_messages(sample, solver_mode=solver_mode, agent_id=agent_id)


def build_adaptive_addon_messages(
    sample: DatasetSample,
    *,
    solver_mode: str,
    agent_id: int,
    stage_a_rows: list[dict[str, object]],
    prompt_version: str = STAGE_A_V4_PROMPT_VERSION,
) -> list[dict[str, str]]:
    _ensure_prompt_version(prompt_version)
    if prompt_version != STAGE_A_V4_PROMPT_VERSION:
        raise ValueError("Adaptive V4 add-on solvers require the v4 prompt version.")
    if solver_mode not in ADAPTIVE_ADDON_SOLVER_MODES:
        raise ValueError(f"Unsupported adaptive add-on solver_mode: {solver_mode}")
    return _build_adaptive_addon_v4_messages(
        sample,
        solver_mode=solver_mode,
        agent_id=agent_id,
        stage_a_rows=stage_a_rows,
    )


def _build_stage_a_v2_messages(
    sample: DatasetSample,
    *,
    solver_mode: str,
    agent_id: int,
) -> list[dict[str, str]]:
    instruction = _stage_a_v2_instruction(sample.dataset, solver_mode)
    user_prompt = (
        f"You are agent_{agent_id} in Stage A of a heterogeneous same-context reasoning experiment.\n"
        f"Solver role: {instruction['label']}\n"
        f"Role summary: {instruction['summary']}\n"
        f"Role guidance: {instruction['guidance']}\n"
        f"Role checklist: {instruction['checklist']}\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        "Return exactly one JSON object with keys "
        '{"reasoning":"brief reasoning","final_answer":"answer","answer_type":"type",'
        '"key_constraints":"short constraints","failure_risk":"short risk"}.\n'
        "The final_answer must obey the dataset instruction exactly. "
        "For multiple-choice tasks, final_answer must be exactly one visible option letter, never the option text."
    )
    return [
        {"role": "system", "content": _schema_solver_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def _build_stage_a_v4_messages(
    sample: DatasetSample,
    *,
    solver_mode: str,
    agent_id: int,
) -> list[dict[str, str]]:
    instruction = _stage_a_v2_instruction(sample.dataset, solver_mode)
    user_prompt = (
        f"You are agent_{agent_id} in Stage A of an adaptive heterogeneous reasoning experiment.\n"
        f"Solver role: {instruction['label']}\n"
        f"Role summary: {instruction['summary']}\n"
        f"Role guidance: {instruction['guidance']}\n"
        f"Role checklist: {instruction['checklist']}\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        "Return exactly one JSON object with keys "
        '{"reasoning":"brief reasoning","final_answer":"answer","answer_type":"type","key_constraints":"short constraints",'
        '"failure_risk":"short risk","confidence_raw":0.0,"claim_span":"exact answer span or canonical slot",'
        '"key_evidence":"short supporting snippet","uncertainty_type":"short label"}.\n'
        "Use confidence_raw on a 0 to 1 scale. The claim_span should be the shortest exact span that supports the final answer when a span exists. "
        "For multiple-choice tasks, final_answer must be exactly one visible option letter, never the option text."
    )
    return [
        {"role": "system", "content": _schema_solver_v4_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def _build_adaptive_addon_v4_messages(
    sample: DatasetSample,
    *,
    solver_mode: str,
    agent_id: int,
    stage_a_rows: list[dict[str, object]],
) -> list[dict[str, str]]:
    instruction = _adaptive_addon_instruction(sample.dataset, solver_mode)
    user_prompt = (
        f"You are agent_{agent_id} in the adaptive verification step of a same-context reasoning experiment.\n"
        f"Verifier role: {instruction['label']}\n"
        f"Role summary: {instruction['summary']}\n"
        f"Role guidance: {instruction['guidance']}\n"
        f"Role checklist: {instruction['checklist']}\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += "Stage A candidate summary:\n"
    user_prompt += _format_stage_a_candidate_summary(stage_a_rows)
    user_prompt += (
        "\nRe-check the answer slot carefully. You may confirm one candidate or produce a corrected answer if every candidate fails the constraints.\n"
        'Return exactly one JSON object with keys '
        '{"reasoning":"brief reasoning","final_answer":"answer","answer_type":"type","key_constraints":"short constraints",'
        '"failure_risk":"short risk","confidence_raw":0.0,"claim_span":"exact answer span or canonical slot",'
        '"key_evidence":"short supporting snippet","uncertainty_type":"short label","selected_candidate":"solver label or novel_answer"}.\n'
        "Use confidence_raw on a 0 to 1 scale."
    )
    return [
        {"role": "system", "content": _schema_solver_v4_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def build_stage_a_safe_retry_messages(
    sample: DatasetSample,
    *,
    agent_id: int,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    _ensure_prompt_version(prompt_version)
    user_prompt = (
        f"You are agent_{agent_id} in a fallback Stage A reasoning pass.\n"
        f"{_dataset_instruction(sample)}\n"
        "Focus only on the requested answer slot. Prefer the shortest exact answer span supported by the context.\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        'Return exactly one JSON object like '
        '{"reasoning":"brief reasoning","final_answer":"answer"}.\n'
        "Keep reasoning under 80 tokens."
    )
    return [
        {"role": "system", "content": _solver_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def _stage_a_v2_instruction(dataset: str, solver_mode: str) -> dict[str, str]:
    if solver_mode == "solver_cot":
        return {
            "label": "Direct Solver",
            "summary": "Solve the task directly with concise reasoning and strict answer-slot control.",
            "guidance": "Identify the requested output type first, solve the task, and emit only the answer format requested by the dataset.",
            "checklist": "answer slot, decisive evidence or calculation, final format",
        }
    if solver_mode == "solver_l2m":
        spec = resolve_reasoning_method(dataset, "pot_l2m")
        return {
            "label": f"Decomposition Solver ({spec.label})",
            "summary": spec.summary,
            "guidance": spec.guidance,
            "checklist": spec.checklist,
        }
    if solver_mode == "solver_skeptic":
        return {
            "label": "Constraint Solver",
            "summary": "Solve by checking answer type, visible choices, units, bounds, and the most likely failure mode before committing.",
            "guidance": (
                "First determine what kind of answer is legal, then test the candidate against constraints. "
                "For multiple-choice tasks, choose the option letter whose text best satisfies the constraints."
            ),
            "checklist": "legal answer type, visible option or unit constraints, strongest counterexample, final format",
        }
    raise ValueError(f"Unsupported solver_mode: {solver_mode}")


def _solver_system_prompt() -> str:
    return build_json_system_prompt(
        "You are an expert reasoning assistant for controlled research experiments.",
        extra_rules=[
            "Follow the task instruction carefully.",
            "Return exactly one JSON object with keys reasoning and final_answer.",
            "Keep reasoning concise and under 120 tokens.",
            "Do not add extra keys.",
        ],
    )


def _schema_solver_system_prompt() -> str:
    return build_json_system_prompt(
        "You are an expert reasoning assistant for controlled research experiments.",
        extra_rules=[
            "Follow the task instruction carefully.",
            "Return exactly one JSON object with keys reasoning, final_answer, answer_type, key_constraints, and failure_risk.",
            "Keep reasoning concise and under 120 tokens.",
            "Do not add extra keys.",
        ],
    )


def _schema_solver_v4_system_prompt() -> str:
    return build_json_system_prompt(
        "You are an expert reasoning assistant for controlled research experiments.",
        extra_rules=[
            "Follow the task instruction carefully.",
            "Return exactly one JSON object with keys reasoning, final_answer, answer_type, key_constraints, failure_risk, confidence_raw, claim_span, key_evidence, uncertainty_type, and optional selected_candidate.",
            "Keep reasoning concise and under 160 tokens.",
            "Do not add extra keys other than selected_candidate.",
        ],
    )


def _dataset_instruction(sample: DatasetSample) -> str:
    base = dataset_instruction_for_sample(sample, hotpot_style="short_span")
    if sample.dataset == "hotpotqa":
        return (
            f"{base} "
            "Return the target answer slot asked by the question, not the anchor entity used for comparison. "
            "If the answer is a named type such as a language, film, court, or designation, include the exact type words when they appear in the context. "
            "Prefer the shortest exact span copied from the context."
        )
    return base


def _ensure_prompt_version(prompt_version: str) -> None:
    if prompt_version not in _SUPPORTED_PROMPT_VERSIONS:
        raise ValueError(f"Unsupported adaptive_sparse_mad prompt_version: {prompt_version}")


def _adaptive_addon_instruction(dataset: str, solver_mode: str) -> dict[str, str]:
    del dataset
    if solver_mode == "solver_verify":
        return {
            "label": "Independent Verifier",
            "summary": "Re-derive or re-check the candidate answers independently, then keep only the answer that survives the strongest verification.",
            "guidance": "Use a fresh line of reasoning, test the leading candidate against its constraints, and repair the answer if the original candidates are all inconsistent.",
            "checklist": "fresh verification path, answer-slot legality, strongest failure case, final exact answer",
        }
    if solver_mode == "solver_option_elim":
        return {
            "label": "Option Elimination Verifier",
            "summary": "Eliminate inconsistent answer options and commit to the single option letter best supported by the prompt and context.",
            "guidance": "Test the visible options one by one, reject options that violate the question or context, and return only the final option letter.",
            "checklist": "option legality, elimination evidence, surviving option, final letter",
        }
    if solver_mode == "solver_evidence":
        return {
            "label": "Evidence Span Verifier",
            "summary": "Find the exact supporting span for the requested answer slot and use it to repair underspecified or anchor-biased answers.",
            "guidance": "Locate the shortest span that directly answers the question, distinguish anchor entities from the requested slot, and normalize the answer only after the span is fixed.",
            "checklist": "requested slot, exact evidence span, answer normalization, final exact answer",
        }
    if solver_mode == "solver_slot_contrast":
        return {
            "label": "Slot Contrast Verifier",
            "summary": "Compare competing candidate answer families and choose the one whose exact wording best matches the requested answer slot.",
            "guidance": "When candidates differ by year, title words, units, type words, or specificity, prefer the candidate whose evidence span most literally answers the question.",
            "checklist": "candidate family contrast, answer-slot wording, exact evidence span, final exact answer",
        }
    raise ValueError(f"Unsupported adaptive add-on solver_mode: {solver_mode}")


def _format_stage_a_candidate_summary(stage_a_rows: list[dict[str, object]]) -> str:
    lines = []
    for row in stage_a_rows:
        validated_output = row.get("validated_output") if isinstance(row.get("validated_output"), dict) else {}
        solver_name = str(row.get("solver_mode") or row.get("method_name") or "solver")
        answer = str(row.get("normalized_answer") or row.get("prediction") or "unknown")
        confidence = row.get("confidence_value")
        answer_type = str(row.get("answer_type") or "") or str(validated_output.get("answer_type") or "")
        constraints = str(row.get("key_constraints") or "") or str(validated_output.get("key_constraints") or "")
        evidence = str(row.get("key_evidence") or "") or str(row.get("claim_span") or "")
        risk = str(row.get("failure_risk") or "") or str(row.get("uncertainty_type") or "")
        lines.append(
            f"- {solver_name}: answer=`{answer}`, confidence={confidence if confidence is not None else 'null'}, "
            f"answer_type=`{answer_type or 'unknown'}`, constraints=`{constraints or 'n/a'}`, "
            f"evidence=`{evidence or 'n/a'}`, risk=`{risk or 'n/a'}`"
        )
    return "\n".join(lines) + "\n"


def _sample_is_multiple_choice(sample: DatasetSample) -> bool:
    raw_options = sample.metadata.get("options") or sample.metadata.get("choices") or []
    return bool(raw_options) or sample.dataset in {"mmlu_pro", "gpqa_diamond", "mmlu", "mmlu_abstract_algebra"}
