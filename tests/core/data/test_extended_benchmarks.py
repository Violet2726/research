"""覆盖扩展 benchmark 配置与数据装载约束的测试。"""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

from research_experiments.core.config import load_benchmark_config
from research_experiments.core.data.datasets import load_samples
from research_experiments.core.data.evaluation import normalize_prediction, score_prediction


def test_math500_loader_reads_problem_and_unique_id() -> None:
    benchmark = load_benchmark_config("configs/core/shared/benchmarks/math500/test.toml")
    samples = load_samples(benchmark)
    assert samples
    sample = samples[0]
    assert sample.dataset == "math500"
    assert sample.question
    assert sample.reference_answer
    assert "unique_id" in sample.metadata


def test_competition_math_loader_reads_subject_and_solution(tmp_path: Path) -> None:
    zip_path = tmp_path / "MATH.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(
            "MATH/test/algebra/example.json",
            json.dumps(
                {
                    "problem": "What is 2 + 2?",
                    "answer": "4",
                    "solution": "2 + 2 = 4",
                    "level": "Level 1",
                    "type": "algebra",
                },
                ensure_ascii=False,
            ),
        )
    benchmark_path = tmp_path / "competition_math.toml"
    benchmark_path.write_text(
        "\n".join(
            [
                'name = "MATH"',
                'slug = "competition_math"',
                'loader = "competition_math_zip"',
                f'source_path = "{zip_path.as_posix()}"',
                'source_split = "test"',
                'sample_id_prefix = "competition_math"',
                'question_field = "problem"',
                'answer_field = "answer"',
                "smoke_size = 1",
                "pilot_size = 1",
                "main_size = 1",
                "random_seed = 0",
                'notes = ""',
            ]
        ),
        encoding="utf-8",
    )
    benchmark = load_benchmark_config(benchmark_path)
    samples = load_samples(benchmark)
    assert samples
    sample = samples[0]
    assert sample.dataset == "competition_math"
    assert sample.question
    assert sample.reference_answer
    assert sample.metadata.get("subject")
    assert "solution" in sample.metadata


def test_mmlu_pro_loader_renders_options_and_mcq_gold() -> None:
    benchmark = load_benchmark_config("configs/core/shared/benchmarks/mmlu-pro/test.toml")
    samples = load_samples(benchmark)
    assert samples
    sample = samples[0]
    assert sample.dataset == "mmlu_pro"
    assert sample.prompt_context.startswith("Options:")
    assert "|||" in sample.reference_answer
    assert "options" in sample.metadata


def test_mmlu_abstract_algebra_benchmark_filters_to_single_subject() -> None:
    benchmark = load_benchmark_config("configs/core/shared/benchmarks/mmlu/abstract_algebra.toml")
    samples = load_samples(benchmark)
    assert samples
    assert len(samples) == 100
    assert all(sample.dataset == "mmlu_abstract_algebra" for sample in samples)
    assert {sample.metadata.get("subject") for sample in samples} == {"abstract_algebra"}


def test_gpqa_loader_renders_options_and_mcq_gold() -> None:
    benchmark = load_benchmark_config("configs/core/shared/benchmarks/gpqa/dataset.toml")
    samples = load_samples(benchmark)
    assert samples
    sample = samples[0]
    assert sample.dataset == "gpqa_diamond"
    assert sample.prompt_context.startswith("Options:")
    assert "|||" in sample.reference_answer
    assert sample.metadata["answer_letter"] in {"A", "B", "C", "D"}


def test_math500_expression_normalization_is_whitespace_and_left_right_insensitive() -> None:
    gold = r"\left( 3, \frac{\pi}{2} \right)"
    predicted = r"(3,\frac{\pi}{2})"
    assert normalize_prediction("math500", gold) == normalize_prediction("math500", predicted)
    assert score_prediction("math500", predicted, gold) == 1.0
    assert normalize_prediction("competition_math", gold) == normalize_prediction("competition_math", predicted)
    assert score_prediction("competition_math", predicted, gold) == 1.0


def test_math_expression_normalization_handles_fraction_notation_and_unordered_solution_lists() -> None:
    gold_fraction = r"(-\frac{3}{2},6)"
    predicted_fraction = r"(-3/2, 6)"
    assert normalize_prediction("competition_math", gold_fraction) == normalize_prediction("competition_math", predicted_fraction)
    assert score_prediction("competition_math", predicted_fraction, gold_fraction) == 1.0

    gold_roots = r"2+\sqrt{3},-2+\sqrt{3}"
    predicted_roots = r"-2+\sqrt3,2+\sqrt{3}"
    assert normalize_prediction("competition_math", gold_roots) == normalize_prediction("competition_math", predicted_roots)
    assert score_prediction("competition_math", predicted_roots, gold_roots) == 1.0

    gold_integer = "17700"
    predicted_with_comma = "17,700"
    assert normalize_prediction("competition_math", gold_integer) == normalize_prediction("competition_math", predicted_with_comma)
    assert score_prediction("competition_math", predicted_with_comma, gold_integer) == 1.0


def test_multiple_choice_scoring_accepts_letter_or_option_text() -> None:
    gold = "B|||polyA tail"
    assert score_prediction("gpqa_diamond", "B", gold) == 1.0
    assert score_prediction("gpqa_diamond", "polyA tail", gold) == 1.0
    assert score_prediction("gpqa_diamond", "A", gold) == 0.0
    assert score_prediction("mmlu_abstract_algebra", "B", gold) == 1.0


