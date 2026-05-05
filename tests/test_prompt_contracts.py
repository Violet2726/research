from __future__ import annotations

from experiment_core.prompt_contracts import dataset_instruction


def test_dataset_instruction_uses_shortest_span_copy_for_controlled_hotpot() -> None:
    instruction = dataset_instruction(
        "hotpotqa",
        context_scope="provided",
        hotpot_style="shortest_span_copy",
    )

    assert "shortest judgeable text span" in instruction
    assert "Prefer copying the exact wording" in instruction


def test_dataset_instruction_uses_visible_context_for_split_context_tracks() -> None:
    instruction = dataset_instruction(
        "hotpotqa",
        context_scope="visible",
        hotpot_style="shortest_span",
    )

    assert "context visible to you" in instruction
    assert "shortest judgeable text span" in instruction


def test_dataset_instruction_unifies_multiple_choice_answer_format() -> None:
    instruction = dataset_instruction("mmlu_pro")
    assert 'only the option letter' in instruction
