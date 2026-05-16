from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from research_experiments.core.config import load_benchmark_config
from research_experiments.core.data.datasets import load_samples
from research_experiments.core.data.evaluation import normalize_prediction, score_prediction


def test_mmlu_loader_renders_options_and_mcq_gold(tmp_path: Path) -> None:
    dataset_root = tmp_path / "local" / "datasets" / "mmlu"
    dataset_root.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "question": "2 + 2 = ?",
                    "subject": "elementary_mathematics",
                    "choices": ["1", "2", "4", "8"],
                    "answer": 2,
                }
            ]
        ),
        dataset_root / "test.parquet",
    )
    benchmark_path = tmp_path / "mmlu.toml"
    benchmark_path.write_text(
        "\n".join(
            [
                'name = "MMLU"',
                'slug = "mmlu"',
                'loader = "mmlu_parquet"',
                f'source_path = "{(dataset_root / "test.parquet").as_posix()}"',
                'source_split = "test"',
                'sample_id_prefix = "mmlu"',
                'question_field = "question"',
                'answer_field = "answer"',
                'options_field = "choices"',
                'answer_index_field = "answer"',
                "smoke_size = 20",
                "pilot_size = 100",
                "main_size = 300",
                "random_seed = 42",
                'notes = ""',
            ]
        ),
        encoding="utf-8",
    )
    sample = load_samples(load_benchmark_config(benchmark_path))[0]

    assert sample.dataset == "mmlu"
    assert sample.prompt_context.startswith("Options:")
    assert sample.reference_answer == "C|||4"
    assert sample.metadata["subject"] == "elementary_mathematics"


def test_humaneval_loader_keeps_prompt_and_test_contract(tmp_path: Path) -> None:
    dataset_root = tmp_path / "local" / "datasets" / "humaneval"
    dataset_root.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "task_id": "HumanEval/0",
                    "prompt": "def add(a, b):\n",
                    "canonical_solution": "    return a + b\n",
                    "test": "def check(candidate):\n    assert candidate(1, 2) == 3\n",
                    "entry_point": "add",
                }
            ]
        ),
        dataset_root / "test.parquet",
    )
    benchmark_path = tmp_path / "humaneval.toml"
    benchmark_path.write_text(
        "\n".join(
            [
                'name = "HumanEval"',
                'slug = "humaneval"',
                'loader = "humaneval_parquet"',
                f'source_path = "{(dataset_root / "test.parquet").as_posix()}"',
                'source_split = "test"',
                'sample_id_prefix = "humaneval"',
                'question_field = "prompt"',
                'answer_field = "canonical_solution"',
                "smoke_size = 20",
                "pilot_size = 100",
                "main_size = 164",
                "random_seed = 42",
                'notes = ""',
            ]
        ),
        encoding="utf-8",
    )
    sample = load_samples(load_benchmark_config(benchmark_path))[0]
    gold = json.loads(sample.reference_answer)

    assert sample.dataset == "humaneval"
    assert gold["entry_point"] == "add"
    assert "check(candidate)" in gold["test"]


def test_commongen_hard_scoring_uses_concept_coverage() -> None:
    gold = json.dumps({"concept_set": ["catch_V", "dog_N", "frisbee_N", "throw_V"]}, ensure_ascii=False)
    assert score_prediction(
        "commongen_hard",
        "The dog catches the frisbee when a child throws it.",
        gold,
    ) == 1.0
    assert 0.0 < score_prediction("commongen_hard", "The dog catches it.", gold) < 1.0


def test_humaneval_scoring_executes_python_tests() -> None:
    gold = json.dumps(
        {
            "prompt": "def add(a, b):\n",
            "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n",
            "entry_point": "add",
        },
        ensure_ascii=False,
    )
    assert score_prediction("humaneval", "    return a + b\n", gold) == 1.0
    assert score_prediction("humaneval", "    return a - b\n", gold) == 0.0


def test_mmlu_scoring_accepts_letter_and_text() -> None:
    gold = "C|||4"
    assert normalize_prediction("mmlu", "C") == "C"
    assert score_prediction("mmlu", "C", gold) == 1.0
    assert score_prediction("mmlu", "4", gold) == 1.0
