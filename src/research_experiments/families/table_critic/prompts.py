"""Table-Critic 提示词构造与结构化输出解析。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from research_experiments.core.data.datasets import DatasetSample


DEFAULT_PROMPT_VERSION = "table_critic_paper_v1"

_TABFACT_EXAMPLE = """Example Table:
/*
col : player | goals | club
row 1 : Alice | 10 | Red FC
row 2 : Bob | 8 | Blue FC
*/
Statement: Alice scored more goals than Bob.
Return JSON:
{"reasoning":"Alice has 10 goals while Bob has 8, so the statement is supported.","final_answer":"true"}"""

_WIKITQ_EXAMPLE = """Example Table:
/*
col : country | medals
row 1 : USA | 10
row 2 : China | 8
*/
Question: which country won the most medals?
Return JSON:
{"reasoning":"USA has 10 medals, which is the maximum.","final_answer":"USA"}"""

_JUDGE_FEWSHOT = """Example 1:
Original Table:
/*
col : country | medals
row 1 : USA | 10
row 2 : China | 8
*/
Question:
which country won the most medals?
Prediction Answer:
USA
Explanation:
USA has 10 medals, which is higher than China's 8, so the answer is correct.
Conclusion: [Correct]

Example 2:
Original Table:
/*
col : player | goals | club
row 1 : Alice | 10 | Red FC
row 2 : Bob | 8 | Blue FC
*/
Question:
who scored fewer goals?
Prediction Answer:
Alice
Explanation:
Alice scored 10 goals while Bob scored 8, so Bob scored fewer goals.
Conclusion: [Incorrect]
"""

_CRITIC_FEWSHOT = """Example:
Original Table:
/*
col : player | goals | club
row 1 : Alice | 10 | Red FC
row 2 : Bob | 8 | Blue FC
*/
Question:
who scored fewer goals?
Reasoning Steps:
Step 1: Read the goals for Alice and Bob.
Step 2: Compare 10 and 8.
Step 3: Claim Alice scored fewer goals.
Prediction Answer:
Alice
Critique:
Step 1 and Step 2 are correct. Step 3 is incorrect because 8 is fewer than 10, so Bob scored fewer goals.
Conclusion: [Incorrect] Step 3
"""


@dataclass(frozen=True)
class TemplateHint:
    """供 critic / refiner 参考的模板摘要。"""

    template_id: str
    path: str
    pattern_summary: str
    reuse_hint: str


def build_direct_messages(sample: DatasetSample) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are a careful table reasoning assistant."},
        {
            "role": "user",
            "content": (
                f"{_task_header(sample)}\n"
                f"{sample.prompt_context}\n\n"
                f"{_task_instruction(sample)}\n"
                "Return a JSON object with keys `reasoning` and `final_answer`."
            ),
        },
    ]


def build_few_shot_messages(sample: DatasetSample) -> list[dict[str, str]]:
    example = _TABFACT_EXAMPLE if sample.dataset == "tabfact" else _WIKITQ_EXAMPLE
    return [
        {"role": "system", "content": "You are a careful table reasoning assistant."},
        {
            "role": "user",
            "content": (
                f"{example}\n\n"
                f"Now solve the next example.\n"
                f"{_task_header(sample)}\n"
                f"{sample.prompt_context}\n\n"
                f"{_task_instruction(sample)}\n"
                "Return a JSON object with keys `reasoning` and `final_answer`."
            ),
        },
    ]


def build_chain_of_table_messages(sample: DatasetSample) -> list[dict[str, str]]:
    compact_instruction = _compact_chain_of_table_instruction(sample)
    return [
        {"role": "system", "content": "You are a Chain-of-Table style table reasoning assistant."},
        {
            "role": "user",
            "content": (
                f"{_task_header(sample)}\n"
                f"{sample.prompt_context}\n\n"
                "Reason with an explicit sub-table style process.\n"
                "Use short numbered steps such as selecting rows, selecting columns, comparing values, or aggregating values.\n"
                f"{compact_instruction}"
                f"{_task_instruction(sample)}\n"
                "Return a JSON object with keys `reasoning` and `final_answer`."
            ),
        },
    ]


def build_judge_messages(
    sample: DatasetSample,
    *,
    current_reasoning: str,
    current_answer: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are the judge in a table reasoning correction pipeline."},
        {
            "role": "user",
            "content": (
                "You are an intelligent judge tasked with determining whether the given Prediction Answer is correct or incorrect.\n\n"
                f"{_JUDGE_FEWSHOT}\n"
                "Now evaluate the next case.\n"
                f"Original Table:\n{sample.prompt_context}\n\n"
                f"Question:\n{sample.question}\n\n"
                f"Prediction Answer:\n{current_answer or '<empty>'}\n\n"
                "Decide whether the current answer should pass.\n"
                "Explanation:\n"
            ),
        },
    ]


def build_critic_messages(
    sample: DatasetSample,
    *,
    current_reasoning: str,
    current_answer: str,
    judge_payload: dict[str, Any],
    template_hints: list[TemplateHint],
) -> list[dict[str, str]]:
    template_block = _render_template_block(template_hints)
    return [
        {"role": "system", "content": "You are the critic in a table reasoning correction pipeline."},
        {
            "role": "user",
            "content": (
                "You are an intelligent critic tasked with determining which step of the table reasoning is incorrect.\n\n"
                f"{_CRITIC_FEWSHOT}\n"
                f"{template_block}\n\n"
                f"Original Table:\n{sample.prompt_context}\n\n"
                f"Question:\n{sample.question}\n\n"
                f"Reasoning Steps:\n{_render_reasoning_steps(current_reasoning)}\n\n"
                f"Prediction Answer:\n{current_answer or '<empty>'}\n\n"
                f"Judge Signal:\nerror_step={judge_payload.get('error_step')} ; rationale={judge_payload.get('rationale')}\n\n"
                "Give a structured critique that points to the first wrong step.\n"
                "Critique:"
            ),
        },
    ]


def build_refiner_messages(
    sample: DatasetSample,
    *,
    current_reasoning: str,
    current_answer: str,
    judge_payload: dict[str, Any],
    critic_payload: dict[str, Any],
    template_hints: list[TemplateHint],
) -> list[dict[str, str]]:
    template_block = _render_template_block(template_hints)
    evidence = "\n".join(f"- {item}" for item in critic_payload.get("conflicting_evidence", [])) or "- None"
    answer_format_guard = _answer_format_refiner_instruction(judge_payload)
    return [
        {"role": "system", "content": "You are the refiner in a table reasoning correction pipeline."},
        {
            "role": "user",
            "content": (
                "We previously answered a table question but received a critique.\n\n"
                f"{template_block}\n\n"
                f"Original Table:\n{sample.prompt_context}\n\n"
                f"Question:\n{sample.question}\n\n"
                f"Previous reasoning:\n{current_reasoning or '<empty>'}\n\n"
                f"Previous answer:\n{current_answer or '<empty>'}\n\n"
                f"Judge signal:\nerror_step={judge_payload.get('error_step')} ; rationale={judge_payload.get('rationale')}\n\n"
                f"Critique:\n{critic_payload.get('critic_feedback') or '<empty>'}\n"
                f"Conflicting evidence:\n{evidence}\n"
                f"Repair hint:\n{critic_payload.get('repair_hint') or '<empty>'}\n\n"
                f"{answer_format_guard}"
                f"{_task_instruction(sample)}\n"
                "Please reproduce a better explanation and answer.\n"
                "Use the format:\nExplanation: ...\nAnswer: ..."
            ),
        },
    ]


def build_curator_messages(
    sample: DatasetSample,
    *,
    judge_payload: dict[str, Any],
    critic_payload: dict[str, Any],
    improved: bool,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "You are the curator maintaining a reusable table-reasoning critique tree."},
        {
            "role": "user",
            "content": (
                f"Dataset: {sample.dataset}\n"
                f"Question type: {sample.metadata.get('question_type')}\n"
                f"Judge path: {' / '.join(judge_payload.get('node_path') or ['ROOT'])}\n"
                f"Judge rationale: {judge_payload.get('rationale') or ''}\n"
                f"Critic feedback: {critic_payload.get('critic_feedback') or ''}\n"
                f"Repair hint: {critic_payload.get('repair_hint') or ''}\n"
                f"Improved after refinement: {str(bool(improved)).lower()}\n\n"
                "Summarize this correction pattern for reuse.\n"
                "Return a JSON object with keys:\n"
                "- `pattern_summary`: short reusable summary\n"
                "- `reuse_hint`: one short hint future critics can copy\n"
                "- `template_title`: short label"
            ),
        },
    ]


def validate_reasoning_answer_output(assistant_text: str, provider_reasoning_text: str) -> dict[str, Any]:
    raw_text = assistant_text or provider_reasoning_text
    try:
        payload = _decode_json_object(raw_text)
        reasoning_value = payload.get("reasoning", payload.get("explanation", payload.get("Explanation")))
        answer_value = payload.get("final_answer", payload.get("answer", payload.get("Answer")))
        return {
            "reasoning": _require_textish(reasoning_value, "reasoning"),
            "final_answer": _require_textish(answer_value, "final_answer"),
        }
    except ValueError:
        explanation_match = re.search(r"Explanation\s*:\s*(?P<text>.*?)(?:Answer\s*:|$)", str(raw_text), flags=re.IGNORECASE | re.DOTALL)
        answer_match = re.search(r"Answer\s*:\s*(?P<text>.+)$", str(raw_text), flags=re.IGNORECASE | re.DOTALL)
        if not answer_match:
            raise
        reasoning = explanation_match.group("text").strip() if explanation_match else str(raw_text).strip()
        final_answer = answer_match.group("text").strip()
        return {
            "reasoning": _require_textish(reasoning, "reasoning"),
            "final_answer": _require_textish(final_answer, "final_answer"),
        }


def validate_refiner_output_with_fallback(
    assistant_text: str,
    provider_reasoning_text: str,
    *,
    previous_reasoning: str,
    previous_answer: str,
) -> dict[str, Any]:
    """优先解析 refiner 输出；若模型重新退化成 critic 风格，则保守沿用旧答案。"""

    try:
        return validate_reasoning_answer_output(assistant_text, provider_reasoning_text)
    except ValueError:
        return {
            "reasoning": _require_textish(previous_reasoning or "previous reasoning retained", "reasoning"),
            "final_answer": _require_textish(previous_answer or "unknown", "final_answer"),
        }


def validate_judge_output(assistant_text: str, provider_reasoning_text: str) -> dict[str, Any]:
    raw_text = assistant_text or provider_reasoning_text
    try:
        payload = _decode_json_object(raw_text)
        if "passed" in payload:
            error_step = _normalize_error_step(payload.get("error_step"))
            passed = _coerce_bool(payload.get("passed"))
            if passed is None:
                raise ValueError("Judge output requires boolean `passed`.")
            error_detected = _coerce_bool(payload.get("error_detected"))
            if error_detected is None:
                error_detected = not passed
            node_path = _normalize_node_path(payload.get("node_path"), error_step=error_step)
            rationale = _require_textish(payload.get("rationale"), "rationale")
        elif "conclusion" in payload:
            conclusion = str(payload.get("conclusion") or "").strip().lower()
            passed = "correct" in conclusion and "incorrect" not in conclusion
            error_detected = not passed
            error_step = "None" if passed else _infer_error_step_from_text(payload.get("explanation") or raw_text)
            node_path = _normalize_node_path(payload.get("node_path"), error_step=error_step)
            rationale = _require_textish(payload.get("explanation") or payload.get("rationale"), "rationale")
        else:
            raise ValueError("Judge output is missing pass/conclusion fields.")
    except ValueError:
        conclusion = _extract_conclusion_text(raw_text)
        passed = "[correct]" in conclusion.lower()
        error_detected = not passed
        error_step = "None" if passed else _infer_error_step_from_text(raw_text)
        node_path = _normalize_node_path(_extract_json_array_field(raw_text, "node_path"), error_step=error_step)
        rationale = _extract_explanation_text(raw_text)
    return {
        "passed": passed,
        "error_detected": error_detected,
        "error_step": error_step,
        "node_path": node_path,
        "rationale": rationale,
    }


def validate_critic_output(assistant_text: str, provider_reasoning_text: str) -> dict[str, Any]:
    raw_text = assistant_text or provider_reasoning_text
    try:
        payload = _decode_json_object(raw_text)
        if "critic_feedback" in payload:
            evidence = payload.get("conflicting_evidence") or []
            if not isinstance(evidence, list):
                raise ValueError("`conflicting_evidence` must be a list.")
            return {
                "critic_feedback": _require_textish(payload.get("critic_feedback"), "critic_feedback"),
                "conflicting_evidence": [str(item).strip() for item in evidence if str(item).strip()],
                "repair_hint": _require_textish(payload.get("repair_hint"), "repair_hint"),
                "error_category": _require_textish(payload.get("error_category"), "error_category"),
            }
        if "critique" in payload:
            conclusion = str(payload.get("conclusion") or "").strip()
            return {
                "critic_feedback": _require_textish(payload.get("critique"), "critic_feedback"),
                "conflicting_evidence": [],
                "repair_hint": _derive_repair_hint_from_critique(str(payload.get("critique") or "")),
                "error_category": conclusion or "free_text_critic",
            }
        raise ValueError("Critic output missing structured critique fields.")
    except ValueError:
        critique, conclusion = _split_critique_and_conclusion(raw_text)
        return {
            "critic_feedback": _require_textish(critique, "critic_feedback"),
            "conflicting_evidence": [],
            "repair_hint": _derive_repair_hint_from_critique(critique),
            "error_category": _require_textish(conclusion or "free_text_critic", "error_category"),
        }


def validate_curator_output(assistant_text: str, provider_reasoning_text: str) -> dict[str, Any]:
    payload = _decode_json_object(assistant_text or provider_reasoning_text)
    return {
        "pattern_summary": _require_textish(payload.get("pattern_summary"), "pattern_summary"),
        "reuse_hint": _require_textish(payload.get("reuse_hint"), "reuse_hint"),
        "template_title": _require_textish(payload.get("template_title"), "template_title"),
    }


def _task_header(sample: DatasetSample) -> str:
    if sample.dataset == "tabfact":
        return f"Statement: {sample.question}"
    return f"Question: {sample.question}"


def _task_instruction(sample: DatasetSample) -> str:
    if sample.dataset == "tabfact":
        return "Decide whether the statement is supported by the table. Use `true` if supported and `false` if refuted as the final answer."
    return "Answer the question using only the table. If there are multiple equally valid answers, join them with ` | ` in the final answer."


def _render_template_block(template_hints: list[TemplateHint]) -> str:
    if not template_hints:
        return "No prior reusable critique templates are available."
    lines = ["Reusable critique templates:"]
    for hint in template_hints:
        lines.append(
            f"- [{hint.template_id}] {hint.path}: {hint.pattern_summary} | reuse hint: {hint.reuse_hint}"
        )
    return "\n".join(lines)


def _render_reasoning_steps(reasoning: str) -> str:
    text = str(reasoning or "").strip()
    if not text:
        return "Step 1: No reasoning was provided."
    if "Step 1" in text or "step 1" in text.lower():
        return text
    if text.startswith("[") and text.endswith("]"):
        try:
            payload = json.loads(text.replace("'", '"'))
            if isinstance(payload, list) and payload:
                return "\n".join(f"Step {index}: {str(item).strip()}" for index, item in enumerate(payload, start=1))
        except Exception:
            pass
    lines = [item.strip("- ").strip() for item in text.splitlines() if item.strip()]
    if len(lines) <= 1:
        return f"Step 1: {text}"
    return "\n".join(f"Step {index}: {line}" for index, line in enumerate(lines, start=1))


def _compact_chain_of_table_instruction(sample: DatasetSample) -> str:
    if len(sample.prompt_context) <= 6000:
        return ""
    return (
        "This table is large. Keep the reasoning compact.\n"
        "Use at most 6 short numbered steps.\n"
        "Only cite the few rows or values needed for the decision.\n"
        "Do not enumerate every row, every candidate, or every numeric value.\n"
    )


def _answer_format_refiner_instruction(judge_payload: dict[str, Any]) -> str:
    if str(judge_payload.get("error_step") or "") != "Answer Format Error":
        return ""
    return (
        "The judge only flagged an Answer Format Error.\n"
        "Preserve the original answer content unless it is empty.\n"
        "Focus on fixing the explanation or the answer format, not on changing the answer semantics.\n\n"
    )


def _decode_json_object(raw_text: str) -> dict[str, Any]:
    cleaned = str(raw_text or "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Assistant output must contain a JSON object.")
    payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Assistant output must be a JSON object.")
    return payload


def _require_textish(value: object, field_name: str) -> str:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{field_name} is required.")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


def _coerce_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"true", "yes", "1", "pass", "passed"}:
        return True
    if normalized in {"false", "no", "0", "fail", "failed"}:
        return False
    return None


def _normalize_error_step(value: object) -> str:
    normalized = str(value or "None").strip().lower()
    mapping = {
        "none": "None",
        "row error": "Row Error",
        "row_error": "Row Error",
        "column error": "Column Error",
        "column_error": "Column Error",
        "calculation error": "Calculation Error",
        "calculation_error": "Calculation Error",
        "logical error": "Logical Error",
        "logical_error": "Logical Error",
        "answer format error": "Answer Format Error",
        "answer_format_error": "Answer Format Error",
    }
    return mapping.get(normalized, "Logical Error")


def _normalize_node_path(value: object, *, error_step: str) -> list[str]:
    if isinstance(value, list) and value:
        path = [str(item).strip() for item in value if str(item).strip()]
        if path and path[0] != "ROOT":
            path.insert(0, "ROOT")
        return path
    if error_step == "Row Error":
        return ["ROOT", "Sub-table Error", "Row Error"]
    if error_step == "Column Error":
        return ["ROOT", "Sub-table Error", "Column Error"]
    if error_step == "Calculation Error":
        return ["ROOT", "Final Query Error", "Calculation Error"]
    if error_step == "Answer Format Error":
        return ["ROOT", "Answer Format Error"]
    if error_step == "None":
        return ["ROOT"]
    return ["ROOT", "Final Query Error", "Logical Error"]


def _extract_json_bool_field(text: str, field_name: str) -> str | None:
    match = re.search(rf'"{re.escape(field_name)}"\s*:\s*(true|false)', str(text or ""), flags=re.IGNORECASE)
    if not match:
        return None
    return match.group(1).lower()


def _extract_json_string_field(text: str, field_name: str) -> str | None:
    match = re.search(rf'"{re.escape(field_name)}"\s*:\s*"(?P<value>.*?)"', str(text or ""), flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return match.group("value").strip()


def _extract_json_array_field(text: str, field_name: str) -> list[str] | None:
    match = re.search(rf'"{re.escape(field_name)}"\s*:\s*\[(?P<value>.*?)\]', str(text or ""), flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    return [item.strip().strip('"').strip("'") for item in match.group("value").split(",") if item.strip()]


def _extract_fallback_rationale(text: str) -> str:
    stripped = " ".join(str(text or "").split())
    if not stripped:
        raise ValueError("rationale is required.")
    return stripped[:300]


def _infer_error_step_from_text(text: str) -> str:
    lowered = str(text or "").lower()
    if "row error" in lowered:
        return "Row Error"
    if "column error" in lowered:
        return "Column Error"
    if "calculation error" in lowered:
        return "Calculation Error"
    if "answer format error" in lowered or "format" in lowered:
        return "Answer Format Error"
    return "Logical Error"


def _extract_conclusion_text(text: str) -> str:
    match = re.search(r"Conclusion\s*:\s*(?P<value>.+)$", str(text or ""), flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return " ".join(match.group("value").split())


def _extract_explanation_text(text: str) -> str:
    match = re.search(r"Explanation\s*:\s*(?P<value>.*?)(?:Conclusion\s*:|$)", str(text or ""), flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return _extract_fallback_rationale(text)
    return _require_textish(match.group("value").strip(), "rationale")


def _split_critique_and_conclusion(text: str) -> tuple[str, str]:
    match = re.search(r"Critique\s*:\s*(?P<critique>.*?)(?:Conclusion\s*:|$)", str(text or ""), flags=re.IGNORECASE | re.DOTALL)
    critique = match.group("critique").strip() if match else str(text or "").strip()
    return critique, _extract_conclusion_text(text)


def _derive_repair_hint_from_critique(critique: str) -> str:
    text = " ".join(str(critique or "").split())
    if not text:
        return "Re-evaluate the wrong step and regenerate the answer conservatively."
    return text[:220]
