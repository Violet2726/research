
"""覆盖冻结 split 生成与样本读取行为。"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

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


def test_default_count100_manifest_uses_true_count_target(tmp_path: Path) -> None:
    source_path = tmp_path / "gsm8k.jsonl"
    source_path.write_text(
        "\n".join(
            json.dumps({"question": f"{index}+1?", "answer": f"#### {index + 1}"}, ensure_ascii=False)
            for index in range(150)
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
                'smoke_size = 20',
                'pilot_size = 150',
                'main_size = 150',
                'random_seed = 42',
                'notes = ""',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    benchmark = load_benchmark_config(benchmark_path)

    generate_split_manifests([benchmark], tmp_path / "splits")
    sample_ids = load_split_ids(benchmark.cache_namespace or benchmark.slug, "count100_seed42", tmp_path / "splits")

    assert len(sample_ids) == 100


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


def test_count_split_covering_full_dataset_uses_full_manifest_name(tmp_path: Path) -> None:
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
                "",
                "[[split_presets]]",
                'name = "count300_seed42"',
                'strategy = "shuffle"',
                "size = 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    benchmark = load_benchmark_config(benchmark_path)

    created = generate_split_manifests([benchmark], tmp_path / "splits")
    assert [path.parent.name for path in created] == ["full"]
    assert json.loads(created[0].read_text(encoding="utf-8"))["split_name"] == "full2_seed42"

    sample_ids = load_split_ids(benchmark.cache_namespace or benchmark.slug, "count300_seed42", tmp_path / "splits")
    samples = select_samples(benchmark, "count300_seed42", tmp_path / "splits")

    assert len(sample_ids) == 2
    assert [sample.sample_id for sample in samples] == sample_ids


def test_generate_stratified_split_manifest_for_competition_math(tmp_path: Path) -> None:
    zip_path = tmp_path / "MATH.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for subject, index in [("algebra", 1), ("geometry", 2)]:
            archive.writestr(
                f"MATH/test/{subject}/{index}.json",
                json.dumps(
                    {
                        "problem": f"{subject} problem",
                        "answer": str(index),
                        "solution": str(index),
                        "level": "Level 1",
                        "type": subject,
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
                "smoke_size = 2",
                "pilot_size = 2",
                "main_size = 2",
                "random_seed = 42",
                'notes = ""',
                "",
                "[[split_presets]]",
                'name = "count20_seed42"',
                'strategy = "stratified"',
                'field = "subject"',
                "size = 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    benchmark = load_benchmark_config(benchmark_path)

    generate_split_manifests([benchmark], tmp_path / "splits")
    sample_ids = load_split_ids(benchmark.cache_namespace or benchmark.slug, "count20_seed42", tmp_path / "splits")
    samples = select_samples(benchmark, "count20_seed42", tmp_path / "splits")

    assert len(sample_ids) == 2
    assert len(samples) == 2
    assert {sample.metadata["subject"] for sample in samples} == {"algebra", "geometry"}


def test_shuffle_window_split_is_disjoint_from_development_prefix(tmp_path: Path) -> None:
    source_path = tmp_path / "gsm8k.jsonl"
    source_path.write_text(
        "\n".join(json.dumps({"question": f"{index}+1?", "answer": f"#### {index + 1}"}) for index in range(20)) + "\n",
        encoding="utf-8",
    )
    benchmark_path = tmp_path / "benchmark.toml"
    benchmark_path.write_text(
        "\n".join(
            [
                'name = "Toy"',
                'slug = "toy_gsm8k"',
                'loader = "gsm8k_jsonl"',
                f'source_path = "{source_path.as_posix()}"',
                'source_split = "test"',
                'sample_id_prefix = "toy"',
                'question_field = "question"',
                'answer_field = "answer"',
                "smoke_size = 5",
                "pilot_size = 10",
                "main_size = 10",
                "random_seed = 42",
                'notes = ""',
                "[[split_presets]]",
                'name = "dev10_seed42"',
                'strategy = "shuffle_window"',
                "offset = 0",
                "size = 10",
                "[[split_presets]]",
                'name = "locked10_seed42"',
                'strategy = "shuffle_window"',
                "offset = 10",
                "size = 10",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    benchmark = load_benchmark_config(benchmark_path)

    generate_split_manifests([benchmark], tmp_path / "splits")
    dataset_key = benchmark.cache_namespace or benchmark.slug
    dev = set(load_split_ids(dataset_key, "dev10_seed42", tmp_path / "splits"))
    locked = set(load_split_ids(dataset_key, "locked10_seed42", tmp_path / "splits"))

    assert len(dev) == 10
    assert len(locked) == 10
    assert dev.isdisjoint(locked)


def test_stratified_window_split_is_balanced_and_disjoint(tmp_path: Path) -> None:
    source_path = tmp_path / "omni.jsonl"
    source_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": f"item-{index}",
                    "problem": f"Compute {index}+1.",
                    "answer": str(index + 1),
                    "domain": ["algebra" if index % 2 == 0 else "geometry"],
                    "difficulty": 2,
                }
            )
            for index in range(40)
        )
        + "\n",
        encoding="utf-8",
    )
    benchmark_path = tmp_path / "omni.toml"
    benchmark_path.write_text(
        "\n".join(
            [
                'name = "Omni toy"',
                'slug = "omni_toy"',
                'loader = "omni_math_jsonl"',
                f'source_path = "{source_path.as_posix()}"',
                'source_split = "test"',
                'sample_id_prefix = "omni"',
                'question_field = "problem"',
                'answer_field = "answer"',
                "smoke_size = 10",
                "pilot_size = 20",
                "main_size = 20",
                "random_seed = 42",
                'notes = ""',
                "[[split_presets]]",
                'name = "pilot20_seed42"',
                'strategy = "stratified_window"',
                'field = "primary_domain"',
                "offset = 0",
                "size = 20",
                "[[split_presets]]",
                'name = "locked20_seed42"',
                'strategy = "stratified_window"',
                'field = "primary_domain"',
                "offset = 20",
                "size = 20",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    benchmark = load_benchmark_config(benchmark_path)

    generate_split_manifests([benchmark], tmp_path / "splits")
    pilot = select_samples(benchmark, "pilot20_seed42", tmp_path / "splits")
    locked = select_samples(benchmark, "locked20_seed42", tmp_path / "splits")

    assert {sample.sample_id for sample in pilot}.isdisjoint(sample.sample_id for sample in locked)
    assert {sample.metadata["primary_domain"] for sample in pilot} == {"algebra", "geometry"}
    assert {sample.metadata["primary_domain"] for sample in locked} == {"algebra", "geometry"}

