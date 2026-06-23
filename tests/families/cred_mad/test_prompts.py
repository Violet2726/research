from __future__ import annotations

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.families.cred_mad.prompts import build_judge_messages, build_stage_a_messages


def _sample(dataset: str = "gpqa_diamond") -> DatasetSample:
    return DatasetSample(
        dataset=dataset,
        sample_id="s1",
        question="Which option is best?",
        reference_answer="A",
        prompt_context="A. alpha\nB. beta",
        metadata={},
    )


def test_stage_a_prompt_keeps_role_as_audit_lens_after_strong_solving() -> None:
    content = build_stage_a_messages(_sample(), agent_id=1, agent_role="counterfactual_falsifier")[1]["content"]

    assert "first solve as a strong independent single-agent reasoner" in content
    assert "Audit lens: counterfactual_falsifier" in content
    assert "Strong solver workflow:" in content
    assert "compare plausible options" in content
    assert "Reason briefly according to your role" not in content


def test_judge_prompt_solves_independently_before_using_board() -> None:
    content = build_judge_messages(
        _sample(dataset="strategyqa"),
        leading_answer="yes",
        stage_rows=[],
        refutation_rows=[],
        defense_rows=[],
    )[1]["content"]

    assert "Solve independently" in content
    assert 'exactly "yes" or "no"' in content
