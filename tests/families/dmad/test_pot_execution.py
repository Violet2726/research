from __future__ import annotations

from research_experiments.families.shared.pot_execution import (
    build_pot_answer_artifact,
    execute_pot_program,
    extract_python_program,
)


def test_extract_python_program_prefers_fenced_block() -> None:
    program = extract_python_program(
        'Reasoning first.\n```python\nx = 2 + 2\nans = x\n```\nFinal answer: 4'
    )

    assert program == "x = 2 + 2\nans = x"


def test_extract_python_program_accepts_unclosed_fence() -> None:
    program = extract_python_program("```python\nx = 2 + 2\nans = x")

    assert program == "x = 2 + 2\nans = x"


def test_execute_pot_program_returns_ans_value() -> None:
    artifact = execute_pot_program("x = 2 + 2\nans = x")

    assert artifact.execution_status == "ok"
    assert artifact.execution_resolution == "direct"
    assert artifact.execution_result == "4"
    assert artifact.final_answer == "4"


def test_execute_pot_program_rejects_unsafe_import() -> None:
    artifact = execute_pot_program("import os\nans = 4")

    assert artifact.execution_status == "unsafe_program"
    assert artifact.execution_error is not None


def test_execute_pot_program_reports_missing_result() -> None:
    artifact = execute_pot_program("x = 2 + 2")

    assert artifact.execution_status == "missing_result"
    assert artifact.execution_result is None


def test_execute_pot_program_recovers_common_result_variables() -> None:
    artifact = execute_pot_program("x1 = 2 + 2\nx2 = 5 + 1")

    assert artifact.execution_status == "ok"
    assert artifact.execution_resolution == "recovered_variables"
    assert artifact.execution_result == "4, 6"


def test_execute_pot_program_recovers_last_printed_value() -> None:
    artifact = execute_pot_program("print(2 + 2)")

    assert artifact.execution_status == "ok"
    assert artifact.execution_resolution == "recovered_variables"
    assert artifact.execution_result == "4"


def test_execute_pot_program_times_out() -> None:
    artifact = execute_pot_program("while True:\n    pass", timeout_seconds=1)

    assert artifact.execution_status == "runtime_error"
    assert artifact.execution_error is not None


def test_build_pot_answer_artifact_uses_executed_result_as_final_answer() -> None:
    artifact = build_pot_answer_artifact(
        '{"final_answer":"5","reasoning":"Use Python.","python_program":"x = 2 + 2\\nans = x"}',
        "",
        dataset="competition_math",
    )

    assert artifact.final_answer == "4"
    assert artifact.execution_status == "ok"
    assert artifact.execution_resolution == "direct"
    assert artifact.execution_result == "4"


def test_build_pot_answer_artifact_recovers_truncated_json_answer() -> None:
    artifact = build_pot_answer_artifact(
        '{"final_answer":"45","reasoning":"Count carefully.","python_program":"from math import comb\\nans = comb(10,2) - 15',
        "",
        dataset="competition_math",
    )

    assert artifact.final_answer == "45"
    assert artifact.execution_status == "ok"
    assert artifact.execution_resolution == "answer_field"
