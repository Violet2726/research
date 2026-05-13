"""覆盖单智能体基线提示词构造约束的测试。"""

from __future__ import annotations

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.families.single_agent.prompts import DEFAULT_PROMPT_VERSION, build_messages


def _sample(dataset: str) -> DatasetSample:
    return DatasetSample(
        dataset=dataset,
        sample_id=f"{dataset}-00001",
        question="Example question?",
        reference_answer="42",
        prompt_context="Example context." if dataset == "hotpotqa" else "",
        metadata={},
    )


def test_build_messages_supports_the_single_agent_prompt_version() -> None:
    sample = _sample("gsm8k")
    messages = build_messages(sample, method_family="cot", prompt_version=DEFAULT_PROMPT_VERSION)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_single_prompt_requires_reasoning_key_in_user_instruction() -> None:
    sample = _sample("hotpotqa")
    messages = build_messages(sample, method_family="cot", prompt_version=DEFAULT_PROMPT_VERSION)
    assert '"reasoning"' in messages[1]["content"]
    assert '"final_answer"' in messages[1]["content"]
    assert "Return exactly one JSON object like" in messages[1]["content"]


