from research_experiments.core.data.datasets import (
    DatasetSample,
    _parse_bbeh_answer_contract,
    _parse_bbeh_options,
    question_without_answer_contract,
    question_without_bbeh_options,
)
from research_experiments.core.data.evaluation import canonicalize_answer, score_prediction


def _sample() -> DatasetSample:
    return DatasetSample(
        dataset="bbeh",
        sample_id="bbeh-test-0000",
        question="Question body\nOptions:\n(A) Alice\n(B) Bob\n(D) Rodrigo",
        reference_answer="(D)",
        prompt_context="",
        metadata={"task": "test", "options": [{"label": "A", "text": "Alice"}, {"label": "B", "text": "Bob"}, {"label": "D", "text": "Rodrigo"}]},
    )


def test_sample_aware_bbeh_option_canonicalization_and_scoring() -> None:
    sample = _sample()
    for answer in ("D", "(D)", "(D) Rodrigo", "Rodrigo", "The final answer is: D"):
        result = canonicalize_answer(sample, answer)
        assert result.valid and result.key == "D"
        assert score_prediction("bbeh", answer, sample.reference_answer, sample=sample) == 1.0


def test_label_text_conflict_and_unmapped_answers_are_invalid() -> None:
    sample = _sample()
    assert canonicalize_answer(sample, "(D) Bob").invalid_reason == "label_text_conflict"
    assert canonicalize_answer(sample, "Rodriguez").invalid_reason == "unmapped_option_answer"
    assert canonicalize_answer(sample, "(Z)").invalid_reason == "unknown_option_label"
    assert score_prediction("bbeh", "(D) Bob", sample.reference_answer, sample=sample) == 0.0


def test_single_choice_option_ordinals_are_unambiguous_contract_aliases() -> None:
    sample = _sample()
    assert canonicalize_answer(sample, "option 3").key == "D"
    assert canonicalize_answer(sample, "3").key == "D"
    assert canonicalize_answer(sample, "4").invalid_reason == "unmapped_option_answer"


def test_non_option_bbeh_uses_conservative_legacy_format_key() -> None:
    sample = DatasetSample("bbeh", "plain", "x", "(b)", "", {"task": "plain", "options": []})
    result = canonicalize_answer(sample, "b")
    assert result.valid and result.key == "b"


def test_bbeh_options_metadata_is_taken_only_from_a_terminal_options_block() -> None:
    question = "Options: mentioned in the stem only.\nQuestion body\nOptions:\n(A) Alpha\n(B) Beta"
    assert _parse_bbeh_options(question) == [{"label": "A", "text": "Alpha"}, {"label": "B", "text": "Beta"}]
    assert _parse_bbeh_options(question + "\nExplanation after choices") == []


def test_bbeh_geometric_shapes_rounding_note_is_part_of_terminal_option_region() -> None:
    question = (
        "SVG geometry question\nOptions:\n(A) triangle\n(B) square\n"
        "Coordinates have been rounded to 5 decimal places so ignore slight differences."
    )
    options = _parse_bbeh_options(question)
    sample = DatasetSample("bbeh", "geo", question, "A", "", {"options": options})

    assert options == [{"label": "A", "text": "triangle"}, {"label": "B", "text": "square"}]
    assert question_without_bbeh_options(sample) == "SVG geometry question"


def test_inline_answer_contract_and_multi_choice_are_strict() -> None:
    question = "Choose the concatenation of all the correct choices.\n(A) first\n(B) second\n(C) third"
    contract = _parse_bbeh_answer_contract(question)
    sample = DatasetSample(
        "bbeh", "multi", question, "BC", "", {"answer_contract": contract, "options": contract["options"]}
    )

    assert contract["kind"] == "multi_choice"
    assert contract["source_style"] == "inline"
    assert canonicalize_answer(sample, "bc").key == "BC"
    assert canonicalize_answer(sample, "CB").invalid_reason == "multi_option_labels_out_of_order"
    assert canonicalize_answer(sample, "B,C").invalid_reason == "invalid_multi_option_format"
    assert question_without_answer_contract(sample) == "Choose the concatenation of all the correct choices."


def test_labeled_exact_text_keeps_terminal_punctuation() -> None:
    contract = {
        "kind": "single_choice",
        "options": [{"label": "D", "text": "Ambiguous."}],
        "block_start": 1,
        "block_end": 2,
        "source_style": "inline",
        "selection_mode": "single",
    }
    sample = DatasetSample("bbeh", "punct", "q", "D", "", {"answer_contract": contract, "options": contract["options"]})
    assert canonicalize_answer(sample, "(D) Ambiguous.").key == "D"
