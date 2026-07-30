"""覆盖扩展 benchmark 配置与数据装载约束的测试。"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from research_experiments.core.config import BenchmarkConfig, load_benchmark_config
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
                "random_seed = 42",
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


def test_bbeh_extension_loader_requires_provenance_and_preserves_task(tmp_path: Path) -> None:
    source = tmp_path / "extension.jsonl"
    source.write_text(
        json.dumps(
            {
                "record_id": "ext-001",
                "task": "shuffled_objects",
                "input": "At the start there are A and B. Swap A and B. What is at the end?\n(A) A\n(B) B",
                "target": "(A)",
                "provenance": {
                    "source_id": "generator-001",
                    "source_sha256": "a" * 64,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    samples = load_samples(_d4_config("bbeh_extension", "bbeh_extension_jsonl", source))
    assert len(samples) == 1
    assert samples[0].sample_id == "ext-001"
    assert samples[0].metadata["task"] == "shuffled_objects"
    assert samples[0].metadata["provenance"]["source_sha256"] == "a" * 64
    assert samples[0].metadata["extension_schema"] == "catch_d4_bbeh_extension_record_v1"


def test_bbeh_extension_loader_rejects_missing_or_duplicate_provenance(tmp_path: Path) -> None:
    source = tmp_path / "extension.jsonl"
    base = {
        "record_id": "ext-001",
        "task": "shuffled_objects",
        "input": "Question",
        "target": "A",
    }
    source.write_text(json.dumps(base) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        load_samples(_d4_config("bbeh_extension", "bbeh_extension_jsonl", source))
    base["provenance"] = {"source_id": "generator", "source_sha256": "a" * 64}
    source.write_text(json.dumps(base) + "\n" + json.dumps(base) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_samples(_d4_config("bbeh_extension", "bbeh_extension_jsonl", source))


def test_supergpqa_science_loader_matches_official_schema_and_gold(tmp_path: Path) -> None:
    source = tmp_path / "science.jsonl"
    source.write_text(
        json.dumps(
            {
                "uuid": "a" * 32,
                "question": "Which field studies stars?",
                "options": ["Physics", "Biology", "Chemistry", "History"],
                "answer": "Physics",
                "answer_letter": "A",
                "discipline": "Science",
                "field": "Physics",
                "subfield": "Astrophysics",
                "difficulty": "middle",
                "is_calculation": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    samples = load_samples(_d4_config("supergpqa_science", "supergpqa_science_jsonl", source))
    assert len(samples) == 1
    sample = samples[0]
    assert sample.reference_answer == "A|||Physics"
    assert sample.metadata["domain"] == "Physics"
    assert sample.metadata["subfield"] == "Astrophysics"
    assert sample.metadata["dataset_schema"] == "supergpqa_official_science_v1"


def test_supergpqa_science_loader_rejects_non_science_and_gold_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "science.jsonl"
    record = {
        "uuid": "b" * 32,
        "question": "Question",
        "options": ["one", "two"],
        "answer": "two",
        "answer_letter": "A",
        "discipline": "Science",
        "field": "Physics",
        "subfield": "Mechanics",
    }
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="disagree"):
        load_samples(_d4_config("supergpqa_science", "supergpqa_science_jsonl", source))
    record["answer_letter"] = "B"
    record["discipline"] = "Engineering"
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        load_samples(_d4_config("supergpqa_science", "supergpqa_science_jsonl", source))


def test_musr_x_loader_preserves_latent_identity_and_deterministic_gold(tmp_path: Path) -> None:
    source = tmp_path / "musr_x.jsonl"
    source.write_text(
        json.dumps(
            {
                "record_id": "musr-x-001",
                "task": "object_placements",
                "latent_graph_sha256": "c" * 64,
                "narrative": "A key moved from the table to the drawer.",
                "question": "Where is the key?",
                "choices": ["table", "drawer", "shelf"],
                "answer_index": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    samples = load_samples(_d4_config("musr_x", "musr_x_jsonl", source))
    assert len(samples) == 1
    sample = samples[0]
    assert sample.metadata["task"] == "object_placements"
    assert sample.metadata["latent_graph_sha256"] == "c" * 64
    assert sample.metadata["answer_text"] == "drawer"
    assert sample.reference_answer.endswith("|||drawer")


def test_musr_x_loader_rejects_duplicate_latent_graphs(tmp_path: Path) -> None:
    source = tmp_path / "musr_x.jsonl"
    record = {
        "record_id": "musr-x-001",
        "task": "team_allocation",
        "latent_graph_sha256": "d" * 64,
        "narrative": "Two people must perform two tasks.",
        "question": "Which allocation works?",
        "choices": ["first", "second"],
        "answer_index": 0,
    }
    second = {**record, "record_id": "musr-x-002"}
    source.write_text(json.dumps(record) + "\n" + json.dumps(second) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="latent_graph_sha256"):
        load_samples(_d4_config("musr_x", "musr_x_jsonl", source))


def _d4_config(slug: str, loader: str, source: Path) -> BenchmarkConfig:
    return BenchmarkConfig(
        name=slug,
        slug=slug,
        loader=loader,
        source_path=str(source),
        source_split="sealed",
        sample_id_prefix=slug,
        question_field="question",
        answer_field="answer",
        smoke_size=1,
        pilot_size=1,
        main_size=1,
        random_seed=42,
        notes="D4 loader fixture",
    )


def test_math_expression_normalization_handles_latex_trig_function_commands() -> None:
    assert normalize_prediction("math500", r"\cotx") == normalize_prediction("math500", "cotx")
    assert score_prediction("math500", "cotx", r"\cotx") == 1.0


def test_omni_math_2_normalization_canonicalizes_simple_symbolic_arithmetic() -> None:
    predicted = "2015+2x+y"
    gold = "2x+y+2015"

    assert normalize_prediction("omni_math_2_filtered", predicted) == normalize_prediction(
        "omni_math_2_filtered", gold
    )
    assert score_prediction("omni_math_2_filtered", predicted, gold) == 1.0


def test_math_normalization_is_idempotent_for_implicit_latex_multiplication() -> None:
    raw = r"3\sqrt{21}"
    normalized = normalize_prediction("omni_math_2_filtered", raw)

    assert normalize_prediction("omni_math_2_filtered", normalized) == normalized
    assert score_prediction("omni_math_2_filtered", normalized, raw) == 1.0


def test_math_normalization_is_idempotent_for_scientific_notation_lists() -> None:
    raw = "5^{56}, 31^{28}, 17^{35}, 10^{51}"
    normalized = normalize_prediction("omni_math_2_filtered", raw)

    assert normalize_prediction("omni_math_2_filtered", normalized) == normalized
    assert score_prediction("omni_math_2_filtered", normalized, raw) == 1.0


def test_math500_normalization_handles_textual_answers_interval_prefixes_and_dfrac() -> None:
    assert score_prediction("math500", "[-2,7]", r"x \in [-2,7]") == 1.0
    assert score_prediction("math500", "0.34", r"\dfrac{17}{50}") == 1.0
    assert score_prediction("math500", "evelyn", r"\text{Evelyn}") == 1.0
    assert score_prediction("math500", "ellipse", r"\text{ellipse}") == 1.0
    assert score_prediction("math500", "c", r"\text{(C)}") == 1.0
    assert score_prediction("math500", "11,111,111,100", r"11,\! 111,\! 111,\! 100") == 1.0


def test_multiple_choice_scoring_accepts_letter_or_option_text() -> None:
    gold = "B|||polyA tail"
    assert score_prediction("gpqa_diamond", "B", gold) == 1.0
    assert score_prediction("gpqa_diamond", "polyA tail", gold) == 1.0
    assert score_prediction("gpqa_diamond", "A", gold) == 0.0
    assert score_prediction("mmlu_abstract_algebra", "B", gold) == 1.0


