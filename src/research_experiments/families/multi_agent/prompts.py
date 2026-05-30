"""多智能体 vanilla MAD 实验的标准提示词入口。"""

from __future__ import annotations

from research_experiments.families.shared.vanilla_mad_prompting import (
    CONTROLLED_PROMPT_VERSION,
    DEFAULT_PROMPT_VERSION,
    build_debate_messages,
    build_initial_messages,
)

__all__ = [
    "DEFAULT_PROMPT_VERSION",
    "CONTROLLED_PROMPT_VERSION",
    "build_initial_messages",
    "build_debate_messages",
]
