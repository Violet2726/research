import pytest

from research_experiments.core.data.evaluation import answer_class_key, score_bbeh


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("(b)", "b"),
        ("[answer]", "answer"),
        ("1.0", "1"),
        ("'quoted'", "quoted"),
        ("“quoted”", "quoted"),
        ("Ｂ", "b"),
        ("answer?", "answer"),
    ],
)
def test_bbeh_format_equivalence_key_matches_scorer(left, right) -> None:
    assert answer_class_key("bbeh", left) == answer_class_key("bbeh", right)
    assert score_bbeh(left, right) == 1.0


@pytest.mark.parametrize(("left", "right"), [("(ab)", "ab"), ("[a[b]]", "a[b]"), ("1", "2")])
def test_bbeh_key_does_not_overmerge_unsafe_forms(left, right) -> None:
    assert answer_class_key("bbeh", left) != answer_class_key("bbeh", right)
