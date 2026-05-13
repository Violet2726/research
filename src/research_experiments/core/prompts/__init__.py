"""共享题型指令与提示词契约入口。"""

from __future__ import annotations

from research_experiments.core.prompts.dataset_contracts import (
    ContextScope,
    HotpotStyle,
    MultipleChoiceScope,
    build_json_system_prompt,
    build_tagged_lines_system_prompt,
    dataset_instruction,
    dataset_instruction_for_sample,
)

__all__ = [
    "ContextScope",
    "HotpotStyle",
    "MultipleChoiceScope",
    "dataset_instruction_for_sample",
    "dataset_instruction",
    "build_json_system_prompt",
    "build_tagged_lines_system_prompt",
]
