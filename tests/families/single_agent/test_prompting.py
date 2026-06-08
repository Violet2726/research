"""覆盖单智能体基线提示词构造约束的测试。"""

from __future__ import annotations

from research_experiments.core.controls.control_prompts import build_cot_messages, build_mv_messages
from research_experiments.core.data.datasets import DatasetSample
from research_experiments.families.single_agent.prompts import (
    DEFAULT_PROMPT_VERSION,
    UNIFIED_CONTROL_PORT_PROMPT_VERSION,
    ZERO_SHOT_COT_PROMPT_VERSION,
    build_messages,
)


def _sample(dataset: str) -> DatasetSample:
    return DatasetSample(
        dataset=dataset,
        sample_id=f"{dataset}-00001",
        question="Example question?",
        reference_answer="42",
        prompt_context="Example context." if dataset == "hotpotqa" else "",
        metadata={},
    )


def test_build_messages_supports_all_single_agent_prompt_versions() -> None:
    sample = _sample("gsm8k")
    for prompt_version in (
        DEFAULT_PROMPT_VERSION,
        UNIFIED_CONTROL_PORT_PROMPT_VERSION,
        ZERO_SHOT_COT_PROMPT_VERSION,
    ):
        messages = build_messages(sample, method_family="cot", prompt_version=prompt_version)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"


def test_single_prompt_requires_reasoning_key_in_user_instruction() -> None:
    sample = _sample("hotpotqa")
    messages = build_messages(sample, method_family="cot", prompt_version=DEFAULT_PROMPT_VERSION)
    assert '"reasoning"' in messages[1]["content"]
    assert '"final_answer"' in messages[1]["content"]
    assert "Return exactly one JSON object like" in messages[1]["content"]


def test_unified_control_port_matches_shared_cot_prompt() -> None:
    sample = _sample("gsm8k")
    expected = build_cot_messages(sample, replicate_id=1, prompt_version=None)
    actual = build_messages(sample, method_family="cot", prompt_version=UNIFIED_CONTROL_PORT_PROMPT_VERSION)
    assert actual == expected


def test_unified_control_port_matches_shared_mv_prompt() -> None:
    sample = _sample("mmlu_pro")
    expected = build_mv_messages(sample, replicate_id=1, prompt_version=None)
    actual = build_messages(sample, method_family="majority_vote", prompt_version=UNIFIED_CONTROL_PORT_PROMPT_VERSION)
    assert actual == expected


def test_zero_shot_cot_prompt_includes_step_by_step_instruction() -> None:
    sample = _sample("math500")
    messages = build_messages(sample, method_family="self_consistency", prompt_version=ZERO_SHOT_COT_PROMPT_VERSION)
    assert "Let's think step by step" in messages[1]["content"]
    assert '"reasoning"' in messages[1]["content"]
    assert '"final_answer"' in messages[1]["content"]
