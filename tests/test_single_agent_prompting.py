from __future__ import annotations

from experiment_core.datasets import DatasetSample
from single_agent.prompting import DEFAULT_PROMPT_VERSION, build_messages


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
