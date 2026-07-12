import pytest

from research_experiments.families.risk_controlled_trace_mad.certificates import verify_evidence


def _verify(question: str, answer: str, test_type: str, payload: dict, claim_kind: str = "support"):
    return verify_evidence(
        question=question,
        target_answer=answer,
        evidence={
            "target_answer": answer,
            "claim_kind": claim_kind,
            "test_type": test_type,
            "payload": payload,
        },
    )


@pytest.mark.parametrize(
    ("question", "answer", "kind", "payload"),
    [
        ("Compute 2 + 3.", "5", "arithmetic", {"left": "2+3", "right": "5", "relation": "eq"}),
        (
            "For x, compare x + x and 2*x. Answer 2*x.",
            "2*x",
            "symbolic",
            {"left": "x+x", "right": "2*x", "relation": "eq"},
        ),
        (
            "Arrange red, blue.",
            "red,blue",
            "collection",
            {"items": ["red", "blue"], "expected_items": ["red", "blue"], "mode": "ordered", "relation": "eq"},
        ),
        ("p is true. Is p true?", "true", "boolean", {"expression": "p", "variables": {"p": True}, "expected": True}),
        (
            "A links B and B links C. Is C reachable from A?",
            "yes",
            "graph",
            {"edges": [["A", "B"], ["B", "C"]], "source": "A", "target": "C", "reachable": True, "directed": True},
        ),
    ],
)
def test_safe_dsl_positive_cases(question, answer, kind, payload) -> None:
    assert _verify(question, answer, kind, payload)["status"] == "pass"


def test_dangerous_or_unbound_expressions_are_unsupported() -> None:
    result = _verify(
        "Compute 2 + 3.", "5", "arithmetic", {"left": "__import__('os').system('x')", "right": "5", "relation": "eq"}
    )
    assert result["status"] == "unsupported"
    unbound = _verify("Compute 2 + 3.", "99", "arithmetic", {"left": "2+3", "right": "99", "relation": "eq"})
    assert unbound["status"] == "fail"


def test_falsification_cannot_masquerade_as_a_passing_equality() -> None:
    mislabeled = _verify(
        "Compute 2 + 3.",
        "5",
        "arithmetic",
        {"left": "2+3", "right": "5", "relation": "eq"},
        claim_kind="falsify",
    )
    assert mislabeled["status"] == "unsupported"
    actual = _verify(
        "Compute 2 + 3.",
        "4",
        "arithmetic",
        {"left": "2+3", "right": "4", "relation": "ne"},
        claim_kind="falsify",
    )
    assert actual["status"] == "pass"
