"""验证统一控制方法 prompt 的正确性。

此测试确保所有实验家族的 cot_1 控制方法使用完全相同的 prompt，
与 single_agent 的最优实现对齐。
"""

from __future__ import annotations

from research_experiments.core.controls.control_prompts import (
    build_cot_messages,
    build_mv_messages,
)
from research_experiments.core.data.datasets import DatasetSample


def _make_gsm8k_sample() -> DatasetSample:
    """构造一个 GSM8K 测试样本。"""
    return DatasetSample(
        dataset="gsm8k",
        sample_id="test-001",
        question="Jared is trying to increase his typing speed. He starts with 47 words per minute (WPM). After some lessons the next time he tests his typing speed it has increased to 52 WPM. If he continues to increase his typing speed once more by 5 words, what will be the average of the three measurements?",
        reference_answer="52",
        prompt_context="",
        metadata={},
    )


def test_cot_prompt_matches_single_agent() -> None:
    """验证 cot_1 的 prompt 与 single_agent 的实现一致。"""
    sample = _make_gsm8k_sample()
    messages = build_cot_messages(sample, 1, None)

    # 验证 system prompt
    system_msg = messages[0]
    assert system_msg["role"] == "system"
    assert "expert reasoning assistant" in system_msg["content"]
    assert "JSON" in system_msg["content"]
    assert "reasoning" in system_msg["content"]
    assert "final_answer" in system_msg["content"]

    # 验证 user prompt
    user_msg = messages[1]
    assert user_msg["role"] == "user"
    assert "CoT" in user_msg["content"]
    assert "Chain-of-Thought" in user_msg["content"]
    assert "GSM8K" in user_msg["content"] or "math problem" in user_msg["content"]
    assert "Jared" in user_msg["content"]
    assert '{"reasoning"' in user_msg["content"]


def test_mv_prompt_matches_single_agent() -> None:
    """验证 mv 的 prompt 与 single_agent 的实现一致。"""
    sample = _make_gsm8k_sample()
    messages = build_mv_messages(sample, 1, None)

    # mv 应该使用与 cot 相同的 prompt
    system_msg = messages[0]
    assert system_msg["role"] == "system"
    assert "expert reasoning assistant" in system_msg["content"]


def test_prompt_consistency_across_calls() -> None:
    """验证多次调用生成的 prompt 完全一致。"""
    sample = _make_gsm8k_sample()

    messages1 = build_cot_messages(sample, 1, None)
    messages2 = build_cot_messages(sample, 2, None)

    # 不同 replicate_id 应该生成相同的 prompt
    assert messages1 == messages2


def test_prompt_includes_context_when_present() -> None:
    """验证当有 context 时，prompt 包含 context。"""
    sample = DatasetSample(
        dataset="hotpotqa",
        sample_id="test-002",
        question="What is the capital of France?",
        reference_answer="Paris",
        prompt_context="France is a country in Europe.",
        metadata={},
    )

    messages = build_cot_messages(sample, 1, None)
    user_msg = messages[1]

    assert "France is a country in Europe" in user_msg["content"]


def test_prompt_excludes_context_when_empty() -> None:
    """验证当没有 context 时，prompt 不包含 Context 部分。"""
    sample = _make_gsm8k_sample()
    messages = build_cot_messages(sample, 1, None)
    user_msg = messages[1]

    # 不应该有 Context: 部分
    assert "Context:\n" not in user_msg["content"]


if __name__ == "__main__":
    test_cot_prompt_matches_single_agent()
    test_mv_prompt_matches_single_agent()
    test_prompt_consistency_across_calls()
    test_prompt_includes_context_when_present()
    test_prompt_excludes_context_when_empty()
    print("All tests passed!")
