from __future__ import annotations

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.families.cred_v.prompts import build_stage_a_messages


def test_stage_a_free_text_prompt_is_reasoning_first_not_json_card() -> None:
    sample = DatasetSample(
        dataset="mmlu_pro",
        sample_id="mmlu-1",
        question="Which option is best?",
        reference_answer="A",
        prompt_context="",
        metadata={"options": ["alpha", "beta"]},
    )

    messages = build_stage_a_messages(
        sample,
        agent_id=1,
        agent_role="constraint_skeptic",
        output_protocol="free_text_answer_v1",
    )

    combined = "\n".join(message["content"] for message in messages)
    assert "Assigned lens: constraint_skeptic" in combined
    assert "REASONING:" in combined
    assert "FINAL_ANSWER:" in combined
    assert "JSON answer object" not in combined
    assert "answer card" not in combined
