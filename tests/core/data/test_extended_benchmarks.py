"""覆盖扩展 benchmark 配置与数据装载约束的测试。"""

from __future__ import annotations

from research_experiments.core.config import load_benchmark_config
from research_experiments.core.data.datasets import load_samples
from research_experiments.core.data.evaluation import normalize_prediction, score_prediction


def test_math500_loader_reads_problem_and_unique_id() -> None:
    benchmark = load_benchmark_config("configs/core/shared/benchmarks/math500.toml")
    samples = load_samples(benchmark)
    assert samples
    sample = samples[0]
    assert sample.dataset == "math500"
    assert sample.question
    assert sample.reference_answer
    assert "unique_id" in sample.metadata


def test_mmlu_pro_loader_renders_options_and_mcq_gold() -> None:
    benchmark = load_benchmark_config("configs/core/shared/benchmarks/mmlu_pro.toml")
    samples = load_samples(benchmark)
    assert samples
    sample = samples[0]
    assert sample.dataset == "mmlu_pro"
    assert sample.prompt_context.startswith("Options:")
    assert "|||" in sample.reference_answer
    assert "options" in sample.metadata


def test_gpqa_loader_renders_options_and_mcq_gold() -> None:
    benchmark = load_benchmark_config("configs/core/shared/benchmarks/gpqa_diamond.toml")
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


def test_multiple_choice_scoring_accepts_letter_or_option_text() -> None:
    gold = "B|||polyA tail"
    assert score_prediction("gpqa_diamond", "B", gold) == 1.0
    assert score_prediction("gpqa_diamond", "polyA tail", gold) == 1.0
    assert score_prediction("gpqa_diamond", "A", gold) == 0.0


