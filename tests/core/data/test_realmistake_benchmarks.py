"""覆盖 ReaLMistake benchmark 的 loader 与二值错误检测评分。"""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

from research_experiments.core.config import load_benchmark_config
from research_experiments.core.data.datasets import load_samples
from research_experiments.core.data.evaluation import normalize_prediction, score_prediction


def test_realmistake_loader_reads_both_model_shards_from_password_zip(tmp_path: Path) -> None:
    dataset_root = tmp_path / "local" / "datasets" / "realmistake"
    dataset_root.mkdir(parents=True)
    zip_path = dataset_root / "data.zip"
    payloads = {
        "data/math_word_problem_generation/gpt-4-0613.jsonl": [
            {
                "input": "Task input A",
                "llm_response": "Candidate response A",
                "error_label": "error",
                "human_explanation": "A has an arithmetic issue.",
                "error_categories": ["Reasoning Correctness"],
                "metadata": {
                    "id": "math_a",
                    "task_name": "math_problem_generation",
                    "llm_response_model": "gpt-4-0613",
                },
            }
        ],
        "data/math_word_problem_generation/Llama-2-70b-chat-hf.jsonl": [
            {
                "input": "Task input B",
                "llm_response": "Candidate response B",
                "error_label": "no_error",
                "human_explanation": "",
                "error_categories": [],
                "metadata": {
                    "id": "math_b",
                    "task_name": "math_problem_generation",
                    "llm_response_model": "Llama-2-70b-chat-hf",
                },
            }
        ],
    }
    with zipfile.ZipFile(zip_path, "w") as archive:
        for member_name, rows in payloads.items():
            archive.writestr(
                member_name,
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                compress_type=zipfile.ZIP_DEFLATED,
            )
    benchmark_path = tmp_path / "realmistake_math.toml"
    benchmark_path.write_text(
        "\n".join(
            [
                'name = "ReaLMistake Math Problem Generation"',
                'slug = "realmistake_math_problem_generation"',
                'loader = "realmistake_error_detection_zip"',
                f'source_path = "{zip_path.as_posix()}"',
                'source_split = "math_word_problem_generation"',
                'sample_id_prefix = "realmistake_math"',
                'question_field = "input"',
                'answer_field = "error_label"',
                "smoke_size = 20",
                "pilot_size = 100",
                "main_size = 300",
                "random_seed = 42",
                'notes = ""',
                'archive_member = "data/math_word_problem_generation"',
            ]
        ),
        encoding="utf-8",
    )

    samples = load_samples(load_benchmark_config(benchmark_path))

    assert [sample.sample_id for sample in samples] == ["math_b", "math_a"]
    assert samples[0].metadata["candidate_response_model"] == "Llama-2-70b-chat-hf"
    assert samples[1].reference_answer == "error"
    assert samples[1].metadata["error_categories"] == ["Reasoning Correctness"]


def test_realmistake_verdict_normalization_supports_binary_labels() -> None:
    assert normalize_prediction("realmistake_math_problem_generation", "contains an error") == "contains_error"
    assert normalize_prediction("realmistake_answerability_classification", "no_error") == "contains_no_error"
    assert score_prediction("realmistake_fine_grained_fact_verification", "contains_no_error", "no_error") == 1.0
    assert score_prediction("realmistake_math_problem_generation", "contains_error", "no_error") == 0.0
