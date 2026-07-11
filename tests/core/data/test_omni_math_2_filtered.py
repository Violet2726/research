from __future__ import annotations

import json

from research_experiments.core.config import BenchmarkConfig
from research_experiments.core.data.datasets import load_samples, load_split_ids


def test_omni_math_2_loader_excludes_every_tagged_record(tmp_path) -> None:
    source = tmp_path / "Omni-Math-2.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {"id": "clean", "problem": "1+1", "answer": "2", "domain": ["algebra"], "difficulty": 1, "tags": []},
                {"id": "proof", "problem": "Prove", "answer": "", "tags": ["proof"]},
                {"id": "image", "problem": "See image", "answer": "x", "tags": "image"},
                {"id": "empty", "problem": "2+2", "answer": "4", "tags": "[]"},
            ]
        ),
        encoding="utf-8",
    )
    config = BenchmarkConfig(
        name="test",
        slug="omni_math_2_filtered",
        loader="omni_math_2_filtered_jsonl",
        source_path=str(source),
        source_split="clean_exact_answer",
        sample_id_prefix="omni_math_2",
        question_field="problem",
        answer_field="answer",
        smoke_size=20,
        pilot_size=100,
        main_size=1000,
        random_seed=1,
        notes="",
    )

    samples = load_samples(config)

    assert [sample.sample_id for sample in samples] == ["clean", "empty"]


def test_omni_math_2_standard_counts_are_nested() -> None:
    count20 = load_split_ids("omni-math-2/Omni-Math-2", "count20_seed42")
    count100 = load_split_ids("omni-math-2/Omni-Math-2", "count100_seed42")
    count300 = load_split_ids("omni-math-2/Omni-Math-2", "count300_seed42")

    assert count20 == count100[:20]
    assert count100 == count300[:100]
