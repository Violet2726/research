from research_experiments.core.data.datasets import (
    DatasetSample,
    _parse_bbeh_options,
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
