"""跨实验共享的题型指令与系统提示词契约。"""

from __future__ import annotations

from typing import Literal

from research_experiments.core.data.datasets import DatasetSample

ContextScope = Literal["provided", "visible"]
HotpotStyle = Literal["short_span", "shortest_span", "shortest_span_copy"]
MultipleChoiceScope = Literal["general", "visible"]


def dataset_instruction_for_sample(
    sample: DatasetSample,
    *,
    context_scope: ContextScope = "provided",
    hotpot_style: HotpotStyle = "short_span",
    multiple_choice_scope: MultipleChoiceScope = "general",
) -> str:
    """根据样本对象生成与数据集匹配的任务指令。"""
    return dataset_instruction(
        sample.dataset,
        context_scope=context_scope,
        hotpot_style=hotpot_style,
        multiple_choice_scope=multiple_choice_scope,
    )


def dataset_instruction(
    dataset: str,
    *,
    context_scope: ContextScope = "provided",
    hotpot_style: HotpotStyle = "short_span",
    multiple_choice_scope: MultipleChoiceScope = "general",
) -> str:
    """按数据集与可见上下文范围生成标准化指令文本。"""
    if dataset == "gsm8k":
        return (
            "Solve the math problem carefully. "
            "The final_answer must be only the final numeric answer without commas or units."
        )
    if dataset in {"math500", "competition_math", "omni_math", "omni_math_2_filtered"}:
        return (
            "Solve the math problem carefully. "
            "The final_answer must be only the final mathematical expression, with no explanation."
        )
    if dataset == "strategyqa":
        return 'Answer with exactly "yes" or "no". The final_answer must be exactly "yes" or "no".'
    if dataset == "hotpotqa":
        context_phrase = "provided context" if context_scope == "provided" else "context visible to you"
        if hotpot_style == "shortest_span_copy":
            return (
                f"Answer the multi-hop question using only the {context_phrase}. "
                "The final_answer must be the shortest judgeable text span. "
                "Prefer copying the exact wording from the context when possible. "
                "Return the entity or literal answer span itself, with qualifiers only when they are part of that span."
            )
        if hotpot_style == "shortest_span":
            return (
                f"Answer the multi-hop question using only the {context_phrase}. "
                "The final_answer must be the shortest judgeable text span."
            )
        return (
            f"Answer the multi-hop question using only the {context_phrase}. "
            "The final_answer must be a short text span."
        )
    if dataset == "webquestions":
        context_phrase = "provided graph evidence" if context_scope == "provided" else "graph evidence visible to you"
        return (
            f"Answer the graph question using only the {context_phrase}. "
            "The final_answer must be the shortest judgeable entity span or literal answer. "
            "Return the entity or literal answer span itself, with qualifiers only when they are part of that span."
        )
    if dataset in {"mmlu_pro", "gpqa_diamond", "mmlu_abstract_algebra", "musr"}:
        if multiple_choice_scope == "visible":
            return (
                "Choose the single best option using only the context visible to you. "
                'The final_answer must be only the option letter, such as "A" or "B".'
            )
        return (
            "Choose the single best option. "
            'The final_answer must be only the option letter, such as "A" or "B".'
        )
    if dataset == "mmlu":
        if multiple_choice_scope == "visible":
            return (
                "Choose the single best option using only the context visible to you. "
                'The final_answer must be only the option letter, such as "A" or "B".'
            )
        return (
            "Choose the single best option. "
            'The final_answer must be only the option letter, such as "A" or "B".'
        )
    if dataset == "bbeh":
        return "Solve the reasoning task carefully. The final_answer must contain only the exact answer requested by the task."
    if dataset == "seqbench":
        return (
            "Navigate the described maze from the agent's start room to the target and rescue the target. "
            "Use only these exact action strings: 'start: ROOM' as the first action; 'move_to: ROOM' for an "
            "adjacent room through an open door; 'pick_up_key: KEY' when standing in the key's room; "
            "'use_key: KEY' before opening the door that key unlocks; 'unlock_and_open_door_to: ROOM' while "
            "standing next to the matching locked door; and 'rescue: NAME' as the final action in the target's room. "
            "Never omit start, key-use, door-unlock, movement, or rescue actions.\n\n"
            "Official-format example 1 (simple navigation): Room C4 and C3 are connected by an open door. "
            "Room C3 and D3 are connected by an open door. Room D5 and E5 are connected by an open door. "
            "Room A2 and A1 are connected by an open door. Room A3 and B3 are connected by an open door. "
            "Room A1 and B1 are connected by an open door. Room A4 and A3 are connected by an open door. "
            "Room E5 and E4 are connected by an open door. Room D4 and D3 are connected by an open door. "
            "Room A5 and B5 are connected by an open door. Room D4 and E4 are connected by an open door. "
            "Bob is in room D5. Alice is in room C4. Output "
            '["start: D5","move_to: E5","move_to: E4","move_to: D4","move_to: D3",'
            '"move_to: C3","move_to: C4","rescue: Alice"].\n\n'
            "Official-format example 2 (single key): Room A1 and A2 are connected by an open door. "
            "Room A2 and B2 are connected by an open door. Room B1 and B2 are connected by an open door. "
            "Room B1 and C1 are connected by an open door. Room C1 and C2 are connected by a closed and "
            "locked door. Door between C1 and C2 requires key 1. Key 1 is in room A2. Bob is in room A1. "
            "Alice is in room C2. Output "
            '["start: A1","move_to: A2","pick_up_key: 1","move_to: B2","move_to: B1",'
            '"move_to: C1","use_key: 1","unlock_and_open_door_to: C2","move_to: C2","rescue: Alice"].\n\n'
            "Official-format example 3 (multiple keys): Room B5 and B4 are connected by a closed and locked door. "
            "The locked door between B5 and B4 requires key 3. Key 3 is in room B5. Room B5 and C5 are "
            "connected by a closed and locked door. The locked door between B5 and C5 requires key 16. "
            "Key 16 is in room C5. Room B4 and C4 are connected by an open door. Room C4 and C3 are "
            "connected by an open door. Room C3 and D3 are connected by a closed and locked door. The locked "
            "door between C3 and D3 requires key 10. Key 10 is in room C4. Room D5 and D4 are connected by "
            "an open door. Room D4 and D3 are connected by an open door. Room A5 and B5 are connected by an "
            "open door. Bob is in room C5. Alice is in room D5. Output "
            '["start: C5","pick_up_key: 16","use_key: 16","unlock_and_open_door_to: B5",'
            '"move_to: B5","pick_up_key: 3","use_key: 3","unlock_and_open_door_to: B4",'
            '"move_to: B4","move_to: C4","pick_up_key: 10","move_to: C3","use_key: 10",'
            '"unlock_and_open_door_to: D3","move_to: D3","move_to: D4","move_to: D5","rescue: Alice"].\n\n'
            "Return the complete ordered action sequence. The final_answer must be only a JSON or Python-style "
            "list of exact action strings in execution order."
        )
    raise ValueError(f"Unsupported dataset: {dataset}")


def build_json_system_prompt(
    role_description: str,
    *,
    extra_rules: list[str] | None = None,
) -> str:
    """拼装严格 JSON 输出模式下的 system prompt。"""
    lines = [
        role_description.strip(),
        "Return strict JSON only.",
        "Do not use markdown fences.",
    ]
    if extra_rules:
        lines.extend(rule.strip() for rule in extra_rules if rule.strip())
    return "\n".join(lines)


def build_tagged_lines_system_prompt(
    role_description: str,
    *,
    extra_rules: list[str] | None = None,
) -> str:
    """拼装“固定标签行”输出模式下的 system prompt。"""
    lines = [
        role_description.strip(),
        "Return only the requested tagged lines.",
        "Do not use markdown fences.",
    ]
    if extra_rules:
        lines.extend(rule.strip() for rule in extra_rules if rule.strip())
    return "\n".join(lines)


