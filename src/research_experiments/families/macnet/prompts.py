"""MacNet 的提示词与轻量输出校验。"""

from __future__ import annotations

import json
import re
from typing import Any

from research_experiments.core.data.datasets import DatasetSample


def build_single_agent_messages(sample: DatasetSample, *, prompt_version: str) -> list[dict[str, str]]:
    """构建单智能体基线提示。"""

    return build_actor_messages(
        sample,
        node_id=0,
        topology_type="single",
        direction_mode="none",
        profile_text="You are a strong single-agent baseline. Solve the task directly and return one final answer.",
        parent_artifacts=[],
        inbound_instructions=[],
        prompt_version=prompt_version,
        terminal_fuse=False,
    )


def build_actor_messages(
    sample: DatasetSample,
    *,
    node_id: int,
    topology_type: str,
    direction_mode: str,
    profile_text: str,
    parent_artifacts: list[dict[str, Any]],
    inbound_instructions: list[dict[str, Any]],
    prompt_version: str,
    terminal_fuse: bool,
) -> list[dict[str, str]]:
    """构建 actor 节点提示。"""

    task_block = _task_block(sample)
    parent_block = _parent_artifact_block(parent_artifacts)
    instruction_block = _instruction_block(inbound_instructions)
    role_line = "terminal_fuse_actor" if terminal_fuse else f"actor_node_{node_id}"
    guidance = (
        "Merge upstream artifacts and only revise what is necessary."
        if parent_artifacts or inbound_instructions
        else "Produce a strong first artifact for the task."
    )
    user = "\n\n".join(
        [
            f"Prompt version: {prompt_version}",
            f"Role: {role_line}",
            f"Topology: {topology_type}",
            f"Direction mode: {direction_mode}",
            guidance,
            task_block,
            parent_block,
            instruction_block,
            _output_contract(sample.dataset),
        ]
    )
    return [
        {"role": "system", "content": profile_text},
        {"role": "user", "content": user},
    ]


def build_instruction_messages(
    sample: DatasetSample,
    *,
    source_node_id: int,
    target_node_id: int,
    source_artifact: str,
    source_answer: str,
    profile_text: str,
    prompt_version: str,
) -> list[dict[str, str]]:
    """构建边 critic/supervisor 提示。"""

    user = "\n\n".join(
        [
            f"Prompt version: {prompt_version}",
            f"Source node: {source_node_id}",
            f"Target node: {target_node_id}",
            _task_block(sample),
            "Upstream artifact:",
            _clip_text(source_artifact, 900),
            f"Upstream final answer: {_clip_text(source_answer, 240)}",
            (
                'Return exactly one JSON object with keys '
                '"instruction", "focus_risk", and "preserve_strength". '
                "The instruction must be actionable and concise."
            ),
        ]
    )
    return [
        {"role": "system", "content": profile_text},
        {"role": "user", "content": user},
    ]


def validate_actor_output(raw_text: str, dataset: str) -> dict[str, Any]:
    """解析 actor 输出。"""

    payload = _parse_json_object(raw_text)
    if payload is None:
        fallback = _fallback_actor_answer(raw_text, dataset)
        return {
            "artifact": fallback,
            "final_answer": fallback,
            "reasoning_trace": "",
            "confidence_raw": None,
        }
    artifact = str(payload.get("artifact") or payload.get("analysis") or payload.get("draft") or "").strip()
    final_answer = str(payload.get("final_answer") or payload.get("answer") or "").strip()
    reasoning_trace = str(payload.get("reasoning_trace") or payload.get("reasoning") or "").strip()
    if not final_answer:
        final_answer = _fallback_actor_answer(artifact or raw_text, dataset)
    if not artifact:
        artifact = final_answer
    return {
        "artifact": artifact,
        "final_answer": final_answer,
        "reasoning_trace": reasoning_trace,
        "confidence_raw": payload.get("confidence_raw"),
    }


def validate_instruction_output(raw_text: str) -> dict[str, Any]:
    """解析 edge instruction 输出。"""

    payload = _parse_json_object(raw_text)
    if payload is None:
        instruction = _clip_text(raw_text.strip(), 240)
        return {
            "instruction": instruction or "Keep strong parts and revise the weakest step.",
            "focus_risk": "",
            "preserve_strength": "",
        }
    instruction = str(payload.get("instruction") or "").strip()
    if not instruction:
        instruction = "Keep strong parts and revise the weakest step."
    return {
        "instruction": instruction,
        "focus_risk": str(payload.get("focus_risk") or "").strip(),
        "preserve_strength": str(payload.get("preserve_strength") or "").strip(),
    }


def _task_block(sample: DatasetSample) -> str:
    lines = [f"Dataset: {sample.dataset}", "Task:", sample.question.strip()]
    if sample.prompt_context:
        lines.extend(["Context:", sample.prompt_context.strip()])
    return "\n".join(lines)


def _parent_artifact_block(parent_artifacts: list[dict[str, Any]]) -> str:
    if not parent_artifacts:
        return "Parent artifacts:\n<none>"
    lines = ["Parent artifacts:"]
    for item in parent_artifacts:
        lines.append(f"- node {item['node_id']} artifact: {_clip_text(str(item['artifact']), 320)}")
        lines.append(f"  node {item['node_id']} answer: {_clip_text(str(item['final_answer']), 120)}")
    return "\n".join(lines)


def _instruction_block(inbound_instructions: list[dict[str, Any]]) -> str:
    if not inbound_instructions:
        return "Inbound instructions:\n<none>"
    lines = ["Inbound instructions:"]
    for item in inbound_instructions:
        lines.append(
            f"- from node {item['source_node_id']}: {_clip_text(str(item['instruction']), 220)}"
        )
    return "\n".join(lines)


def _output_contract(dataset: str) -> str:
    if dataset == "mmlu":
        answer_hint = 'a single option letter like "A"'
    elif dataset == "humaneval":
        answer_hint = "the Python completion only"
    else:
        answer_hint = "one short natural-language answer"
    return (
        'Return exactly one JSON object with keys "artifact", "final_answer", '
        '"reasoning_trace", and "confidence_raw". '
        f'For this dataset, final_answer should be {answer_hint}.'
    )


def _fallback_actor_answer(raw_text: str, dataset: str) -> str:
    text = _strip_code_fences(raw_text).strip()
    if dataset == "mmlu":
        match = re.search(r"\b([A-D])\b", text.upper())
        return match.group(1) if match else text
    if dataset == "humaneval":
        return text
    sentence = text.splitlines()[0].strip() if text.splitlines() else text
    return sentence


def _parse_json_object(raw_text: str) -> dict[str, Any] | None:
    text = str(raw_text or "").strip()
    if not text:
        return None
    code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if code_block:
        text = code_block.group(1)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return payload if isinstance(payload, dict) else None


def _clip_text(value: str, limit: int) -> str:
    stripped = " ".join(str(value or "").split())
    if len(stripped) <= limit:
        return stripped
    return stripped[: max(0, limit - 3)] + "..."


def _strip_code_fences(value: str) -> str:
    text = str(value or "").strip()
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text
