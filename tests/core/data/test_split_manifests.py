
"""覆盖冻结 split 生成与样本读取行为。"""

from __future__ import annotations

from pathlib import Path
import json

import pytest

from research_experiments.core.config import load_benchmark_config
from research_experiments.core.data.datasets import generate_split_manifests, load_split_ids, select_samples


def test_generate_and_load_split_manifests(tmp_path: Path) -> None:
    source_path = tmp_path / "gsm8k.jsonl"
    source_path.write_text(
        "\n".join(
            [
                json.dumps({"question": "1+1?", "answer": "#### 2"}, ensure_ascii=False),
                json.dumps({"question": "2+2?", "answer": "#### 4"}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    benchmark_path = tmp_path / "benchmark.toml"
    benchmark_path.write_text(
        "\n".join(
            [
                'name = "Toy GSM8K"',
                'slug = "toy_gsm8k"',
                'loader = "gsm8k_jsonl"',
                f'source_path = "{source_path.as_posix()}"',
                'source_split = "test"',
                'sample_id_prefix = "toy"',
                'question_field = "question"',
                'answer_field = "answer"',
                'smoke_size = 1',
                'pilot_size = 2',
                'main_size = 2',
                'random_seed = 42',
                'notes = ""',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    benchmark = load_benchmark_config(benchmark_path)
    created = generate_split_manifests([benchmark], tmp_path / "splits")
    assert created
    smoke_ids = load_split_ids(benchmark.cache_namespace or benchmark.slug, "count20_seed42", tmp_path / "splits")
    samples = select_samples(benchmark, "count20_seed42", tmp_path / "splits")
    assert len(smoke_ids) == 1
    assert [sample.sample_id for sample in samples] == smoke_ids


def test_select_samples_raises_on_missing_manifest_sample_ids(tmp_path: Path) -> None:
    source_path = tmp_path / "gsm8k.jsonl"
    source_path.write_text(
        json.dumps({"question": "1+1?", "answer": "#### 2"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    benchmark_path = tmp_path / "benchmark.toml"
    benchmark_path.write_text(
        "\n".join(
            [
                'name = "Toy GSM8K"',
                'slug = "toy_gsm8k"',
                'loader = "gsm8k_jsonl"',
                f'source_path = "{source_path.as_posix()}"',
                'source_split = "test"',
                'sample_id_prefix = "toy"',
                'question_field = "question"',
                'answer_field = "answer"',
                'smoke_size = 1',
                'pilot_size = 1',
                'main_size = 1',
                'random_seed = 42',
                'notes = ""',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    benchmark = load_benchmark_config(benchmark_path)
    dataset_key = benchmark.cache_namespace or benchmark.slug
    split_path = tmp_path / "splits" / "count20" / f"{dataset_key}-seed42.json"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    split_path.write_text(
        json.dumps(
            {
                "dataset": dataset_key,
                "split_name": "count20_seed42",
                "source_split": "test",
                "sample_count": 1,
                "sample_ids": ["missing-id"],
                "random_seed": 42,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(KeyError, match="missing sample_id"):
        select_samples(benchmark, "count20_seed42", tmp_path / "splits")

