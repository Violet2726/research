"""A-SMAD 提示词构造。

本模块负责把样本、solver 角色和版本号转换成模型消息。
提示词正文保持英文，以便和既有实验版本、缓存键以及历史结果一致。
"""

from __future__ import annotations

import json
import re
from typing import Any

from research_experiments.core.controls.control_prompts import build_cot_messages
from research_experiments.core.data.datasets import DatasetSample
from research_experiments.core.prompts.dataset_contracts import build_json_system_prompt, dataset_instruction_for_sample
from research_experiments.family_runtime.free_text_protocol import build_free_text_system_prompt, task_format_ok
from research_experiments.family_runtime.reasoning_methods import resolve_reasoning_method

STAGE_A_V2_PROMPT_VERSION = "adaptive_sparse_mad_v2_task_schema"
STAGE_A_V4_PROMPT_VERSION = "adaptive_sparse_mad_v4_evidence_gate"
FREE_TEXT_DEBATE_PROMPT_VERSION = "adaptive_sparse_mad_free_text_debate_v1"
DEFAULT_PROMPT_VERSION = STAGE_A_V2_PROMPT_VERSION
_SUPPORTED_PROMPT_VERSIONS = {
    STAGE_A_V2_PROMPT_VERSION,
    STAGE_A_V4_PROMPT_VERSION,
    FREE_TEXT_DEBATE_PROMPT_VERSION,
}
SOLVER_MODES = ("solver_cot", "solver_l2m", "solver_skeptic")
ADAPTIVE_ADDON_SOLVER_MODES = (
    "solver_verify",
    "solver_option_elim",
    "solver_evidence",
    "solver_slot_contrast",
    "solver_counterfactual",
    "solver_disconfirm",
)
META_ROUTER_ERROR_MODES = (
    "clean_consensus",
    "pseudo_majority",
    "false_consensus",
    "all_three_wrong_suspect",
)
META_ROUTER_NO_CONFIDENT_CANDIDATE = "no_confident_candidate"
_MULTIPLE_CHOICE_DATASETS = {"mmlu_pro", "gpqa_diamond", "mmlu", "mmlu_abstract_algebra"}


def build_stage_a_messages(
    sample: DatasetSample,
    *,
    solver_mode: str,
    agent_id: int,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
) -> list[dict[str, str]]:
    """构造核心 Stage A solver 消息，并按版本选择 v2 或 v4 schema。"""
    _ensure_prompt_version(prompt_version)
    if prompt_version == FREE_TEXT_DEBATE_PROMPT_VERSION:
        return build_stage_a_free_text_messages(sample, solver_mode=solver_mode, agent_id=agent_id)
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
    """构造自适应追加 solver 消息，只允许使用 v4 evidence gate schema。"""
    _ensure_prompt_version(prompt_version)
    if prompt_version == FREE_TEXT_DEBATE_PROMPT_VERSION:
        if solver_mode not in ADAPTIVE_ADDON_SOLVER_MODES:
            raise ValueError(f"Unsupported adaptive add-on solver_mode: {solver_mode}")
        return build_adaptive_addon_free_text_messages(
            sample,
            solver_mode=solver_mode,
            agent_id=agent_id,
            stage_a_rows=stage_a_rows,
        )
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


def build_stage_a_free_text_messages(
    sample: DatasetSample,
    *,
    solver_mode: str,
    agent_id: int,
) -> list[dict[str, str]]:
    """Build Stage A messages for the enhanced free-text A-SMAD protocol."""
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
    user_prompt += _enhanced_free_text_protocol_instruction(
        sample.dataset,
        selected_candidate=False,
        revision_note=False,
    )
    return [
        {"role": "system", "content": _free_text_solver_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def build_adaptive_addon_free_text_messages(
    sample: DatasetSample,
    *,
    solver_mode: str,
    agent_id: int,
    stage_a_rows: list[dict[str, object]],
) -> list[dict[str, str]]:
    """Build adaptive add-on verifier messages for the enhanced free-text protocol."""
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
    if solver_mode in {"solver_counterfactual", "solver_disconfirm"}:
        dominant_answer = _dominant_candidate_answer(stage_a_rows)
        if dominant_answer:
            user_prompt += (
                f"\nCurrent leading candidate family: `{dominant_answer}`.\n"
                "Your final answer must not be a trivial restatement, formatting variant, or same answer family as that leading candidate.\n"
            )
    user_prompt += (
        "\nRe-check the answer slot carefully. Confirm one candidate or produce a corrected answer only when it is better grounded.\n"
        + _enhanced_free_text_protocol_instruction(
            sample.dataset,
            selected_candidate=True,
            revision_note=False,
        )
    )
    return [
        {"role": "system", "content": _free_text_solver_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def build_meta_router_head_messages(
    sample: DatasetSample,
    *,
    stage_a_rows: list[dict[str, object]],
) -> list[dict[str, str]]:
    """Build the strict-JSON V7 meta-router prompt from the three Stage A packets."""
    allowed_candidates = ", ".join([*SOLVER_MODES, META_ROUTER_NO_CONFIDENT_CANDIDATE])
    allowed_error_modes = ", ".join(META_ROUTER_ERROR_MODES)
    allowed_addons = ", ".join(ADAPTIVE_ADDON_SOLVER_MODES)
    user_prompt = (
        "You are the meta-router head for A-SMAD V7.\n"
        "Read the Stage A candidate packets and decide whether this sample is already clean, is a pseudo-majority,"
        " hides a false consensus, or looks like an all-three-wrong capacity failure.\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += "Stage A candidate summary:\n"
    user_prompt += _format_stage_a_candidate_summary(stage_a_rows)
    user_prompt += (
        "\nReturn exactly one JSON object with keys "
        '{"selected_candidate":"solver label","error_mode":"mode","should_trigger":true,'
        '"recommended_solver_sequence":["solver_a","solver_b"],"router_confidence":0.0,"reasoning_short":"brief rationale"}.\n'
        f"- selected_candidate must be one of: {allowed_candidates}.\n"
        f"- error_mode must be one of: {allowed_error_modes}.\n"
        f"- recommended_solver_sequence may only use: {allowed_addons}.\n"
        "- Keep reasoning_short under 40 words.\n"
        "- Do not propose a new final answer. Only pick an existing candidate label or no_confident_candidate.\n"
        "- If error_mode is clean_consensus, should_trigger should usually be false and recommended_solver_sequence should be []."
    )
    return [
        {
            "role": "system",
            "content": build_json_system_prompt(
                "You are a precise routing assistant for controlled reasoning experiments.",
                extra_rules=[
                    "Return exactly one JSON object.",
                    "Do not add markdown, code fences, or extra commentary.",
                    "Use only the allowed enum values provided by the user.",
                ],
            ),
        },
        {"role": "user", "content": user_prompt},
    ]


def build_capacity5_arbiter_messages(
    sample: DatasetSample,
    *,
    candidate_rows: list[dict[str, object]],
) -> list[dict[str, str]]:
    """Build the compact V8 arbiter prompt constrained to the current top-2 families."""
    if len(candidate_rows) != 2:
        raise ValueError("Capacity5 arbiter requires exactly two candidate families.")
    allowed_families: list[str] = []
    user_prompt = (
        "You are the compact champion-challenger arbiter for A-SMAD V8.\n"
        "You must choose between exactly two existing answer families and may not invent a third family.\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += "Candidate families:\n"
    for row in candidate_rows:
        family_key = str(row.get("family_key") or "").strip() or "unknown_family"
        allowed_families.append(family_key)
        user_prompt += (
            f"- {family_key}: answer=`{str(row.get('representative_answer') or 'unknown')}`, "
            f"family_score={row.get('family_score')}, clean_support={row.get('clean_support')}, "
            f"evidence_count={row.get('evidence_count')}, solvers=[{', '.join(str(item) for item in (row.get('solver_modes') or [])) or 'unknown'}]\n"
        )
        for member_row in row.get("rows") or []:
            user_prompt += (
                f"  - {str(member_row.get('solver_mode') or 'solver')}: "
                f"answer=`{str(member_row.get('normalized_answer') or member_row.get('prediction') or 'unknown')}`, "
                f"confidence={member_row.get('confidence_value') if member_row.get('confidence_value') is not None else 'unknown'}, "
                f"evidence=`{_row_evidence(member_row) or 'n/a'}`, "
                f"constraints=`{str(member_row.get('key_constraints') or '') or 'n/a'}`\n"
            )
    user_prompt += (
        "\nReturn exactly one JSON object with keys "
        '{"selected_family":"family_key","selected_answer":"representative answer","confidence_raw":0.0,"reasoning_short":"brief rationale"}.\n'
        f"- selected_family must be one of: {', '.join(allowed_families)}.\n"
        "- selected_answer must restate the representative answer of the selected family, not a new answer.\n"
        "- confidence_raw must be between 0 and 1.\n"
        "- Keep reasoning_short under 40 words.\n"
        "- Do not output any family not listed above."
    )
    return [
        {
            "role": "system",
            "content": build_json_system_prompt(
                "You are a precise binary arbiter for controlled reasoning experiments.",
                extra_rules=[
                    "Return exactly one JSON object.",
                    "Do not add markdown, code fences, or extra commentary.",
                    "Choose only between the two provided candidate families.",
                ],
            ),
        },
        {"role": "user", "content": user_prompt},
    ]


def build_sparse_debate_messages(
    sample: DatasetSample,
    *,
    agent_id: int,
    round_index: int,
    own_row: dict[str, object],
    peer_rows: list[dict[str, object]],
    gate_decision: dict[str, object],
    leading_answer: str,
    prompt_version: str = FREE_TEXT_DEBATE_PROMPT_VERSION,
) -> list[dict[str, str]]:
    """Build one cross-examination revision prompt for trigger-only A-SMAD debate."""
    _ensure_prompt_version(prompt_version)
    if prompt_version == STAGE_A_V4_PROMPT_VERSION:
        user_prompt = (
            f"You are agent_{agent_id} in debate revision round {round_index} of A-SMAD.\n"
            f"{_dataset_instruction(sample)}\n"
            f"Question:\n{sample.question.strip()}\n\n"
        )
        if sample.prompt_context:
            user_prompt += f"Context:\n{sample.prompt_context}\n\n"
        user_prompt += (
            "Your prior answer packet:\n"
            f"- answer=`{_row_answer(own_row) or 'unknown'}`\n"
            f"- reasoning=`{_row_reasoning(own_row) or 'n/a'}`\n"
            f"- evidence=`{_row_evidence(own_row) or 'n/a'}`\n"
            f"- confidence=`{own_row.get('confidence_value') if own_row.get('confidence_value') is not None else 'unknown'}`\n\n"
            "Peer answers and evidence:\n"
            f"{_format_debate_peer_summary(peer_rows)}\n"
            f"Gate reasons: {', '.join(str(item) for item in (gate_decision.get('trigger_reasons') or [])) or 'none'}.\n"
            f"Current leading answer family: `{leading_answer or 'unknown'}`.\n\n"
            "Cross-examine the peers and your own prior answer. Preserve the correct answer slot, and revise only when another candidate is better grounded.\n"
            'Return exactly one JSON object with keys '
            '{"reasoning":"brief reasoning","final_answer":"answer","answer_type":"type","key_constraints":"short constraints",'
            '"failure_risk":"short risk","confidence_raw":0.0,"claim_span":"exact answer span or canonical slot",'
            '"key_evidence":"short supporting snippet","uncertainty_type":"short label","selected_candidate":"solver label or novel_answer","revision_note":"short change note"}.\n'
            "Use confidence_raw on a 0 to 1 scale.\n"
            "If you keep an existing candidate family, selected_candidate should name that solver. "
            "If every existing candidate is wrong and you propose a new answer family, selected_candidate should be `novel_answer`."
        )
        return [
            {"role": "system", "content": _schema_solver_v4_system_prompt()},
            {"role": "user", "content": user_prompt},
        ]
    if prompt_version != FREE_TEXT_DEBATE_PROMPT_VERSION:
        raise ValueError("Sparse debate messages require the free-text debate or v4 structured prompt version.")
    user_prompt = (
        f"You are agent_{agent_id} in debate revision round {round_index} of A-SMAD.\n"
        f"{_dataset_instruction(sample)}\n"
        f"Question:\n{sample.question.strip()}\n\n"
    )
    if sample.prompt_context:
        user_prompt += f"Context:\n{sample.prompt_context}\n\n"
    user_prompt += (
        "Your prior answer packet:\n"
        f"- answer=`{_row_answer(own_row) or 'unknown'}`\n"
        f"- reasoning=`{_row_reasoning(own_row) or 'n/a'}`\n"
        f"- evidence=`{_row_evidence(own_row) or 'n/a'}`\n"
        f"- confidence=`{own_row.get('confidence_value') if own_row.get('confidence_value') is not None else 'unknown'}`\n\n"
        "Peer answers and evidence:\n"
        f"{_format_debate_peer_summary(peer_rows)}\n"
        f"Gate reasons: {', '.join(str(item) for item in (gate_decision.get('trigger_reasons') or [])) or 'none'}.\n"
        f"Current leading answer family: `{leading_answer or 'unknown'}`.\n\n"
        "Cross-examine the peers and your own prior answer. Defend your answer if it still best fits the evidence, or revise to the best supported legal answer.\n"
        + _enhanced_free_text_protocol_instruction(
            sample.dataset,
            selected_candidate=True,
            revision_note=True,
        )
    )
    return [
        {"role": "system", "content": _free_text_debate_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def parse_adaptive_sparse_mad_free_text_output(raw_text: str, *, dataset: str) -> dict[str, Any]:
    """Parse enhanced tagged-line A-SMAD free text into the Stage A structured payload."""
    cleaned = _strip_code_fences(str(raw_text or "").strip())
    if not cleaned:
        raise ValueError("Assistant output is empty.")
    values = _extract_tagged_values(cleaned)
    required_labels = (
        "REASONING",
        "FINAL_ANSWER",
        "CONFIDENCE",
        "ANSWER_TYPE",
        "KEY_CONSTRAINTS",
        "KEY_EVIDENCE",
        "FAILURE_RISK",
    )
    missing = [label for label in required_labels if not values.get(label)]
    if missing:
        raise ValueError(f"Missing required tagged line(s): {', '.join(missing)}.")
    confidence = _parse_confidence(values["CONFIDENCE"])
    final_answer = _normalize_free_text_final_answer(values["FINAL_ANSWER"], dataset=dataset)
    if not final_answer:
        raise ValueError("FINAL_ANSWER must be non-empty.")
    payload: dict[str, Any] = {
        "final_answer": final_answer,
        "reasoning": values["REASONING"],
        "confidence_raw": confidence,
        "answer_type": values["ANSWER_TYPE"],
        "key_constraints": values["KEY_CONSTRAINTS"],
        "key_evidence": values["KEY_EVIDENCE"],
        "claim_span": values["KEY_EVIDENCE"] or final_answer,
        "failure_risk": values["FAILURE_RISK"],
        "selected_candidate": values.get("SELECTED_CANDIDATE"),
        "revision_note": values.get("REVISION_NOTE"),
        "output_protocol": FREE_TEXT_DEBATE_PROMPT_VERSION,
    }
    if not task_format_ok(dataset, final_answer):
        payload["format_warning"] = "free_text_answer_outside_task_format"
    return payload


def parse_meta_router_head_output(raw_text: str) -> dict[str, Any]:
    """Parse and normalize the strict-JSON V7 meta-router output."""
    cleaned = _strip_code_fences(str(raw_text or "").strip())
    if not cleaned:
        raise ValueError("Meta-router output is empty.")
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError("Meta-router output must be valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Meta-router output must be a JSON object.")
    selected_candidate = _normalize_meta_router_selected_candidate(payload.get("selected_candidate"))
    error_mode = _normalize_meta_router_error_mode(payload.get("error_mode"))
    should_trigger = payload.get("should_trigger")
    if not isinstance(should_trigger, bool):
        raise ValueError("Meta-router should_trigger must be boolean.")
    recommended_solver_sequence = _normalize_meta_router_solver_sequence(payload.get("recommended_solver_sequence"))
    router_confidence = _parse_confidence(str(payload.get("router_confidence") or ""))
    reasoning_short = _collapse_whitespace(str(payload.get("reasoning_short") or "").strip())
    if not reasoning_short:
        raise ValueError("Meta-router reasoning_short must be non-empty.")
    return {
        "selected_candidate": selected_candidate,
        "error_mode": error_mode,
        "should_trigger": should_trigger,
        "recommended_solver_sequence": recommended_solver_sequence,
        "router_confidence": router_confidence,
        "reasoning_short": reasoning_short,
    }


def parse_capacity5_arbiter_output(raw_text: str, *, allowed_families: list[str]) -> dict[str, Any]:
    """Parse and normalize the strict-JSON V8 arbiter output."""
    cleaned = _strip_code_fences(str(raw_text or "").strip())
    if not cleaned:
        raise ValueError("Capacity5 arbiter output is empty.")
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError("Capacity5 arbiter output must be valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Capacity5 arbiter output must be a JSON object.")
    selected_family = _collapse_whitespace(str(payload.get("selected_family") or "").strip())
    if selected_family not in allowed_families:
        raise ValueError("Capacity5 arbiter selected_family must be one of the provided family labels.")
    selected_answer = _collapse_whitespace(str(payload.get("selected_answer") or "").strip())
    if not selected_answer:
        raise ValueError("Capacity5 arbiter selected_answer must be non-empty.")
    confidence_raw = _parse_confidence(str(payload.get("confidence_raw") or ""))
    reasoning_short = _collapse_whitespace(str(payload.get("reasoning_short") or "").strip())
    if not reasoning_short:
        raise ValueError("Capacity5 arbiter reasoning_short must be non-empty.")
    return {
        "selected_family": selected_family,
        "selected_answer": selected_answer,
        "confidence_raw": confidence_raw,
        "reasoning_short": reasoning_short,
    }


def _build_stage_a_v2_messages(
    sample: DatasetSample,
    *,
    solver_mode: str,
    agent_id: int,
) -> list[dict[str, str]]:
    """构造 v2 任务 schema 的 Stage A 消息。"""
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
    """构造带证据字段与置信度字段的 v4 Stage A 消息。"""
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
    """构造自适应验证步骤的 v4 消息，并附上 Stage A 候选摘要。"""
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
    if solver_mode in {"solver_counterfactual", "solver_disconfirm"}:
        dominant_answer = _dominant_candidate_answer(stage_a_rows)
        if dominant_answer:
            user_prompt += (
                f"\nCurrent leading candidate family: `{dominant_answer}`.\n"
                "Your final_answer must not be a trivial restatement, formatting variant, or same answer family as that leading candidate.\n"
            )
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
    """构造 Stage A 兜底重试消息，优先恢复最短合法答案槽。"""
    _ensure_prompt_version(prompt_version)
    if prompt_version == FREE_TEXT_DEBATE_PROMPT_VERSION:
        user_prompt = (
            f"You are agent_{agent_id} in a fallback Stage A reasoning pass.\n"
            f"{_dataset_instruction(sample)}\n"
            "Focus only on the requested answer slot. Prefer the shortest legal answer supported by the prompt.\n"
            f"Question:\n{sample.question.strip()}\n\n"
        )
        if sample.prompt_context:
            user_prompt += f"Context:\n{sample.prompt_context}\n\n"
        user_prompt += _enhanced_free_text_protocol_instruction(
            sample.dataset,
            selected_candidate=False,
            revision_note=False,
        )
        return [
            {"role": "system", "content": _free_text_solver_system_prompt()},
            {"role": "user", "content": user_prompt},
        ]
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
    """根据 solver 模式返回 Stage A 角色说明。"""
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
    if solver_mode in ADAPTIVE_ADDON_SOLVER_MODES:
        return _adaptive_addon_instruction(dataset, solver_mode)
    raise ValueError(f"Unsupported solver_mode: {solver_mode}")


def _solver_system_prompt() -> str:
    """生成最小 answer_core 输出格式的系统提示词。"""
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
    """生成 v2 结构化 Stage A 输出格式的系统提示词。"""
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
    """生成 v4 证据增强输出格式的系统提示词。"""
    return build_json_system_prompt(
        "You are an expert reasoning assistant for controlled research experiments.",
        extra_rules=[
            "Follow the task instruction carefully.",
            "Return exactly one JSON object with keys reasoning, final_answer, answer_type, key_constraints, failure_risk, confidence_raw, claim_span, key_evidence, uncertainty_type, and optional selected_candidate.",
            "Keep reasoning concise and under 160 tokens.",
            "Do not add extra keys other than selected_candidate.",
        ],
    )


def _free_text_solver_system_prompt() -> str:
    return build_free_text_system_prompt(
        "You are an expert reasoning assistant for controlled research experiments.",
        extra_rules=[
            "Follow the task instruction carefully.",
            "Use the exact tag names requested by the user.",
            "Keep REASONING concise and evidence-aware.",
        ],
    )


def _free_text_debate_system_prompt() -> str:
    return build_free_text_system_prompt(
        "You are one reasoning agent revising an answer after peer cross-examination.",
        extra_rules=[
            "Use the exact tag names requested by the user.",
            "Ground revisions in task constraints and peer evidence.",
            "Do not change the answer unless the evidence or constraints justify it.",
        ],
    )


def _enhanced_free_text_protocol_instruction(
    dataset: str,
    *,
    selected_candidate: bool,
    revision_note: bool,
) -> str:
    lines = [
        "Return only tagged lines in this exact order:",
        "REASONING: <required concise reasoning>",
        "FINAL_ANSWER: <canonical final answer only>",
        "CONFIDENCE: <number from 0.0 to 1.0>",
        "ANSWER_TYPE: <short answer slot type>",
        "KEY_CONSTRAINTS: <short task-format and semantic constraints>",
        "KEY_EVIDENCE: <short decisive evidence, calculation, or constraint check>",
        "FAILURE_RISK: <main remaining risk or 'none'>",
    ]
    if selected_candidate:
        lines.append("SELECTED_CANDIDATE: <source candidate label or novel_answer>")
    if revision_note:
        lines.append("REVISION_NOTE: <defend_or_revise plus why>")
    lines.extend(
        [
            "Rules:",
            "- Every required tag must appear exactly once.",
            "- FINAL_ANSWER must contain only the answer, with no explanation.",
            "- CONFIDENCE must be calibrated on a 0 to 1 scale.",
            "- Keep KEY_EVIDENCE short but specific enough for a deterministic resolver.",
        ]
    )
    if dataset in _MULTIPLE_CHOICE_DATASETS:
        lines.append('- FINAL_ANSWER must be exactly one visible option letter such as "A" or "B".')
    elif dataset in {"gsm8k", "math500", "competition_math"}:
        lines.append("- FINAL_ANSWER must use plain ASCII math only; do not use LaTeX commands or backslashes.")
    elif dataset in {"hotpotqa", "webquestions"}:
        lines.append("- FINAL_ANSWER must be the shortest judgeable text span.")
    return "\n".join(lines)


def _format_debate_peer_summary(peer_rows: list[dict[str, object]]) -> str:
    if not peer_rows:
        return "- no peer packets available\n"
    lines = []
    for row in peer_rows:
        agent_id = row.get("agent_id")
        solver = str(row.get("solver_mode") or row.get("method_name") or "solver")
        confidence = row.get("confidence_value")
        lines.append(
            f"- agent_{agent_id} {solver}: answer=`{_row_answer(row) or 'unknown'}`, "
            f"confidence={confidence if confidence is not None else 'unknown'}, "
            f"reasoning=`{_row_reasoning(row) or 'n/a'}`, evidence=`{_row_evidence(row) or 'n/a'}`"
        )
    return "\n".join(lines) + "\n"


def _row_answer(row: dict[str, object]) -> str:
    return str(row.get("normalized_answer") or row.get("prediction") or "").strip()


def _row_reasoning(row: dict[str, object]) -> str:
    return str(row.get("reasoning") or "").strip()


def _row_evidence(row: dict[str, object]) -> str:
    return str(row.get("key_evidence") or row.get("claim_span") or "").strip()


def _normalize_meta_router_selected_candidate(value: object) -> str:
    normalized = str(value or "").strip().lower().replace("agent_", "solver_")
    aliases = {
        "cot": "solver_cot",
        "l2m": "solver_l2m",
        "skeptic": "solver_skeptic",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized in SOLVER_MODES:
        return normalized
    if normalized == META_ROUTER_NO_CONFIDENT_CANDIDATE:
        return normalized
    raise ValueError("Meta-router selected_candidate must be a known Stage A solver label.")


def _normalize_meta_router_error_mode(value: object) -> str:
    normalized = _collapse_whitespace(str(value or "").strip().lower())
    if normalized not in META_ROUTER_ERROR_MODES:
        raise ValueError("Meta-router error_mode must use a supported enum value.")
    return normalized


def _normalize_meta_router_solver_sequence(value: object) -> list[str]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise ValueError("Meta-router recommended_solver_sequence must be a JSON array.")
    normalized: list[str] = []
    for item in value:
        solver_name = _collapse_whitespace(str(item or "").strip())
        if solver_name not in ADAPTIVE_ADDON_SOLVER_MODES:
            raise ValueError("Meta-router recommended_solver_sequence contains an unsupported solver.")
        normalized.append(solver_name)
    return normalized


def _extract_tagged_values(text: str) -> dict[str, str]:
    labels = {
        "REASONING",
        "FINAL_ANSWER",
        "CONFIDENCE",
        "ANSWER_TYPE",
        "KEY_CONSTRAINTS",
        "KEY_EVIDENCE",
        "FAILURE_RISK",
        "SELECTED_CANDIDATE",
        "REVISION_NOTE",
    }
    values: dict[str, str] = {}
    for raw_line in str(text or "").splitlines():
        match = re.match(r"^\s*([A-Z_]+)\s*:\s*(.*?)\s*$", raw_line, flags=re.IGNORECASE)
        if match is None:
            continue
        label = match.group(1).upper()
        if label not in labels or label in values:
            continue
        value = _collapse_whitespace(match.group(2).strip().strip("\"'`"))
        if value:
            values[label] = value
    return values


def _parse_confidence(value: str) -> float:
    raw = str(value or "").strip()
    if raw.endswith("%"):
        try:
            numeric = float(raw[:-1].strip()) / 100.0
        except ValueError as exc:
            raise ValueError("CONFIDENCE must be numeric.") from exc
    else:
        try:
            numeric = float(raw)
        except ValueError as exc:
            raise ValueError("CONFIDENCE must be numeric.") from exc
        if numeric > 1.0 and numeric <= 100.0:
            numeric = numeric / 100.0
    if numeric < 0.0 or numeric > 1.0:
        raise ValueError("CONFIDENCE must be between 0 and 1.")
    return round(numeric, 6)


def _normalize_free_text_final_answer(value: str, *, dataset: str) -> str:
    answer = str(value or "").strip().strip("\"'`")
    if dataset not in _MULTIPLE_CHOICE_DATASETS:
        return _collapse_whitespace(answer)
    exact = answer.upper().strip()
    if re.fullmatch(r"[A-J]", exact):
        return exact
    match = re.match(r"^\(?([A-J])\)?(?:[.)]|:|,|-)?(?:\s|$)", exact)
    if match:
        return match.group(1)
    option_match = re.search(r"\b(?:OPTION|CHOICE|ANSWER)\s*(?:IS|:)?\s*([A-J])\b", exact)
    if option_match:
        return option_match.group(1)
    return _collapse_whitespace(answer)


def _strip_code_fences(text: str) -> str:
    trimmed = str(text or "").strip()
    if not trimmed.startswith("```"):
        return trimmed
    trimmed = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", trimmed)
    trimmed = re.sub(r"\s*```$", "", trimmed)
    return trimmed.strip()


def _collapse_whitespace(value: str) -> str:
    return " ".join(str(value or "").split())


def _dataset_instruction(sample: DatasetSample) -> str:
    """生成数据集任务说明，并为 HotpotQA 补充答案槽约束。"""
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
    """拒绝未登记的提示词版本，避免误用历史缓存。"""
    if prompt_version not in _SUPPORTED_PROMPT_VERSIONS:
        raise ValueError(f"Unsupported adaptive_sparse_mad prompt_version: {prompt_version}")


def _adaptive_addon_instruction(dataset: str, solver_mode: str) -> dict[str, str]:
    """根据追加 solver 模式返回自适应验证角色说明。"""
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
    if solver_mode == "solver_counterfactual":
        return {
            "label": "Counterfactual Candidate Generator",
            "summary": "Deliberately search for a clean alternative answer family when the current candidate set may have collapsed onto the same wrong answer.",
            "guidance": (
                "Do not paraphrase the leading candidate family. Re-open the reasoning from the question and constraints, then produce a genuinely different candidate "
                "only if it is better grounded in the evidence or task structure."
            ),
            "checklist": "leading family to avoid, fresh candidate family, typed answer-slot check, exact supporting evidence",
        }
    if solver_mode == "solver_disconfirm":
        return {
            "label": "Disconfirmation Verifier",
            "summary": "Actively try to refute the current leading family and only replace it when a different answer family is better supported.",
            "guidance": (
                "Do not restate the current family. Search for the strongest contradiction, then propose a different family only if it has clearer evidence, "
                "stronger slot fit, or stricter constraint consistency."
            ),
            "checklist": "refutation attempt, alternative family, stronger evidence or constraints, final exact answer",
        }
    raise ValueError(f"Unsupported adaptive add-on solver_mode: {solver_mode}")


def _format_stage_a_candidate_summary(stage_a_rows: list[dict[str, object]]) -> str:
    """把 Stage A 候选压缩成提示词可读的逐行摘要。"""
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
    """根据元数据和数据集名判断样本是否为多选题。"""
    raw_options = sample.metadata.get("options") or sample.metadata.get("choices") or []
    return bool(raw_options) or sample.dataset in _MULTIPLE_CHOICE_DATASETS


def _dominant_candidate_answer(stage_a_rows: list[dict[str, object]]) -> str:
    """返回当前 Stage A 候选集中出现最多的答案族。"""
    counts: dict[str, int] = {}
    for row in stage_a_rows:
        answer = str(row.get("normalized_answer") or row.get("prediction") or "").strip()
        if not answer:
            continue
        counts[answer] = counts.get(answer, 0) + 1
    if not counts:
        return ""
    return max(counts, key=lambda answer: (counts[answer], len(answer), answer))
