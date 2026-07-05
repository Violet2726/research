from __future__ import annotations

from research_experiments.families.cred_v.certificates import (
    compile_math_check_spec,
    verify_hotpot_certificate,
    verify_math_certificate,
)


def test_math_certificate_verifies_challenger_and_rejects_leader() -> None:
    result = verify_math_certificate(
        question="Evaluate 1/2 + 1/3.",
        leader_answer="1/2",
        payload={
            "answer": "5/6",
            "certificate_type": "expression_evaluation",
            "problem_expression": "1/2 + 1/3",
            "problem_constants": ["1", "2", "1", "3"],
            "problem_variables": [],
            "unit": "",
        },
    )

    assert result.valid is True
    assert result.challenger_pass is True
    assert result.leader_pass is False
    assert result.normalized_answer == "5/6"


def test_math_certificate_compiler_only_accepts_locally_parseable_question_sources() -> None:
    compiled = compile_math_check_spec("Evaluate $1/2 + 1/3$.")
    plain = compile_math_check_spec("Compute the value of 2^3 + 1.")
    geometric = compile_math_check_spec("Find the angle between two lines in degrees.")

    assert compiled is not None
    assert compiled.problem_expression == "1/2 + 1/3"
    assert compiled.problem_constants == ("1", "2", "1", "3")
    assert plain is not None
    assert plain.problem_expression == "2^3 + 1"
    assert geometric is None


def test_math_certificate_compiler_rejects_unrelated_math_spans_in_word_problems() -> None:
    questions = (
        "Positive integers $a$, $b$, and $2009$ form a geometric sequence. What is $a$?",
        "What is the remainder when $129^{34}+96^{38}$ is divided by $11$?",
        "If each island has a $1/5$ chance, what is the probability exactly four have treasure?",
        "Two sides are each $8$ units long. What is the greatest possible perimeter?",
    )

    assert all(compile_math_check_spec(question) is None for question in questions)


def test_math_certificate_rejects_unknown_operations_and_foreign_constants() -> None:
    unsafe = verify_math_certificate(
        question="Evaluate 2 + __import__('os').",
        leader_answer="5",
        payload={
            "answer": "5",
            "certificate_type": "expression_evaluation",
            "problem_expression": "2 + __import__('os')",
            "problem_constants": ["2"],
            "problem_variables": [],
            "unit": "",
        },
    )
    foreign = verify_math_certificate(
        question="Evaluate 2 + 3.",
        leader_answer="5",
        payload={
            "answer": "7",
            "certificate_type": "expression_evaluation",
            "problem_expression": "2 + 3",
            "problem_constants": ["2", "3", "7"],
            "problem_variables": [],
            "unit": "",
        },
    )

    assert unsafe.valid is False
    assert unsafe.failure_reason == "uncompilable_question"
    assert foreign.valid is False
    assert foreign.failure_reason == "problem_signature_mismatch"


def test_math_interval_certificate_preserves_open_and_closed_bounds() -> None:
    accepted = verify_math_certificate(
        question="Return the interval (2, infinity).",
        leader_answer="[2,infinity)",
        payload={
            "answer": "(2, infinity)",
            "certificate_type": "interval_equivalence",
            "problem_expression": "(2, infinity)",
            "problem_constants": ["2"],
            "problem_variables": [],
            "unit": "",
        },
    )

    assert accepted.valid is True
    assert accepted.challenger_pass is True
    assert accepted.leader_pass is False


def test_math_certificate_rejects_self_declared_target_not_bound_to_question() -> None:
    result = verify_math_certificate(
        question="Evaluate 1/2 + 1/3.",
        leader_answer="1/2",
        payload={
            "answer": "7/8",
            "certificate_type": "expression_evaluation",
            "problem_expression": "7/8",
            "problem_constants": ["7", "8"],
            "problem_variables": [],
            "unit": "",
        },
    )

    assert result.valid is False
    assert result.failure_reason == "certificate_spec_mismatch"


def test_hotpot_certificate_requires_unique_context_span_and_strict_completion() -> None:
    result = verify_hotpot_certificate(
        context="[Expedition] The expedition was led by Captain John Underhill before command changed.",
        leader_answer="John Underhill",
        payload={
            "answer": "Captain John Underhill",
            "certificate_type": "context_span_completion",
            "source_title": "Expedition",
            "source_sentence_index": 0,
            "evidence_span": "Captain John Underhill",
            "missing_tokens": ["Captain"],
        },
    )
    unsupported = verify_hotpot_certificate(
        context="[Expedition] John Underhill led the expedition.",
        leader_answer="John Underhill",
        payload={
            "answer": "Captain John Underhill",
            "certificate_type": "context_span_completion",
            "source_title": "Expedition",
            "source_sentence_index": 0,
            "evidence_span": "Captain John Underhill",
            "missing_tokens": ["Captain"],
        },
    )

    assert result.valid is True
    assert result.challenger_pass is True
    assert result.leader_pass is False
    assert unsupported.valid is False


def test_hotpot_certificate_checks_declared_title_and_sentence_index() -> None:
    raw_context = {
        "title": ["Expedition", "Other"],
        "sentences": [
            ["The expedition was led by Captain John Underhill.", "It ended in spring."],
            ["John Underhill appears here without the title."],
        ],
    }
    payload = {
        "answer": "Captain John Underhill",
        "certificate_type": "context_span_completion",
        "source_title": "Expedition",
        "source_sentence_index": 0,
        "evidence_span": "Captain John Underhill",
        "missing_tokens": ["Captain"],
    }

    accepted = verify_hotpot_certificate(
        context="[Expedition] The expedition was led by Captain John Underhill.\n[Other] John Underhill appears here.",
        raw_context=raw_context,
        leader_answer="John Underhill",
        payload=payload,
    )
    wrong_index = verify_hotpot_certificate(
        context="[Expedition] The expedition was led by Captain John Underhill.\n[Other] John Underhill appears here.",
        raw_context=raw_context,
        leader_answer="John Underhill",
        payload={**payload, "source_sentence_index": 1},
    )

    assert accepted.valid is True
    assert wrong_index.valid is False
    assert wrong_index.failure_reason == "declared_source_mismatch"
