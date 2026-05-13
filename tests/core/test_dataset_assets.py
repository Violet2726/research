"""覆盖项目数据集资产发现与说明文档写出的测试。"""

from __future__ import annotations

import json
from pathlib import Path

from research_experiments.core.config import load_benchmark_config
from research_experiments.workspace.dataset_assets import (
    build_supplementary_dataset_specs,
    discover_used_benchmark_config_paths,
    write_dataset_inventory_files,
)
from research_experiments.core.data.datasets import DatasetSample


def test_discover_used_benchmark_config_paths_scans_experiment_configs(tmp_path: Path) -> None:
    shared_benchmarks = tmp_path / "configs" / "core" / "shared" / "benchmarks"
    shared_benchmarks.mkdir(parents=True)
    (shared_benchmarks / "gsm8k.toml").write_text(
        "\n".join(
            [
                'name = "GSM8K"',
                'slug = "gsm8k"',
                'loader = "gsm8k_jsonl"',
                'source_path = "gsm8k/test.jsonl"',
                'source_split = "test"',
                'sample_id_prefix = "gsm8k"',
                'question_field = "question"',
                'answer_field = "answer"',
                "smoke_size = 20",
                "pilot_size = 100",
                "main_size = 300",
                "random_seed = 42",
                'notes = ""',
            ]
        ),
        encoding="utf-8",
    )
    (shared_benchmarks / "strategyqa.toml").write_text(
        "\n".join(
            [
                'name = "StrategyQA"',
                'slug = "strategyqa"',
                'loader = "strategyqa_json"',
                'source_path = "strategyqa/dev.json"',
                'source_split = "dev"',
                'sample_id_prefix = "strategyqa"',
                'question_field = "question"',
                'answer_field = "answer"',
                "smoke_size = 20",
                "pilot_size = 100",
                "main_size = 229",
                "random_seed = 42",
                'notes = ""',
            ]
        ),
        encoding="utf-8",
    )

    experiment_dir = tmp_path / "configs" / "families" / "single_agent" / "experiments"
    experiment_dir.mkdir(parents=True)
    (experiment_dir / "demo.toml").write_text(
        "\n".join(
            [
                'name = "demo"',
                'benchmark_configs = [',
                '  "configs/core/shared/benchmarks/gsm8k.toml",',
                '  "configs/core/shared/benchmarks/strategyqa.toml",',
                '  "configs/core/shared/benchmarks/gsm8k.toml",',
                ']',
            ]
        ),
        encoding="utf-8",
    )

    discovered = discover_used_benchmark_config_paths(tmp_path / "configs")
    assert [path.name for path in discovered] == ["gsm8k.toml", "strategyqa.toml"]


def test_write_dataset_inventory_files_writes_local_manifest_and_repo_readme(tmp_path: Path, monkeypatch) -> None:
    benchmark_path = tmp_path / "math500.toml"
    benchmark_path.write_text(
        "\n".join(
            [
                'name = "MATH500"',
                'slug = "math500"',
                'loader = "math500_jsonl"',
                f'source_path = "{(tmp_path / "local" / "datasets" / "math500" / "test.jsonl").as_posix()}"',
                'source_split = "test"',
                'sample_id_prefix = "math500"',
                'question_field = "problem"',
                'answer_field = "answer"',
                "smoke_size = 20",
                "pilot_size = 100",
                "main_size = 500",
                "random_seed = 42",
                'notes = ""',
            ]
        ),
        encoding="utf-8",
    )
    benchmark = load_benchmark_config(benchmark_path)
    dataset_file = tmp_path / "local" / "datasets" / "math500" / "test.jsonl"
    dataset_file.parent.mkdir(parents=True)
    dataset_file.write_text('{"problem":"1+1","answer":"2"}\n', encoding="utf-8")
    splits_root = tmp_path / "splits"
    (splits_root / "count20").mkdir(parents=True)
    (splits_root / "count20" / "math500-seed42.json").write_text(
        json.dumps({"sample_ids": ["math500-00000"]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "research_experiments.workspace.dataset_assets.load_samples",
        lambda _benchmark: [
            DatasetSample(
                dataset="math500",
                sample_id="math500-00000",
                question="1+1",
                reference_answer="2",
                prompt_context="",
                metadata={},
            )
        ],
    )

    paths = write_dataset_inventory_files(
        [benchmark],
        datasets_root=tmp_path / "local" / "datasets",
        docs_root=tmp_path / "datasets",
        splits_root=splits_root,
    )
    readme = (tmp_path / "datasets" / "README.md").read_text(encoding="utf-8")
    manifest = json.loads((tmp_path / "local" / "datasets" / "manifest.json").read_text(encoding="utf-8"))

    assert paths["readme"].name == "README.md"
    assert paths["manifest"].name == "manifest.json"
    assert "local/datasets" in readme
    assert "不再承载正式数据文件" in readme
    assert "uv run research_cli tools dataset-assets prepare-all-sources" in readme
    assert manifest["dataset_count"] == 1
    assert manifest["primary_assets"][0]["slug"] == "math500"


def test_build_supplementary_dataset_specs_covers_train_and_validation_assets(tmp_path: Path) -> None:
    benchmark_dir = tmp_path / "benchmarks"
    benchmark_dir.mkdir()
    (benchmark_dir / "gsm8k.toml").write_text(
        "\n".join(
            [
                'name = "GSM8K"',
                'slug = "gsm8k"',
                'loader = "gsm8k_jsonl"',
                'source_path = "gsm8k/test.jsonl"',
                'source_split = "test"',
                'sample_id_prefix = "gsm8k"',
                'question_field = "question"',
                'answer_field = "answer"',
                "smoke_size = 20",
                "pilot_size = 100",
                "main_size = 300",
                "random_seed = 42",
                'notes = ""',
            ]
        ),
        encoding="utf-8",
    )
    (benchmark_dir / "mmlu_pro.toml").write_text(
        "\n".join(
            [
                'name = "MMLU-Pro"',
                'slug = "mmlu_pro"',
                'loader = "mmlu_pro_parquet"',
                'source_path = "mmlu-pro/test.parquet"',
                'source_split = "test"',
                'sample_id_prefix = "mmlu_pro"',
                'question_field = "question"',
                'answer_field = "answer"',
                "smoke_size = 20",
                "pilot_size = 100",
                "main_size = 1200",
                "random_seed = 42",
                'notes = ""',
            ]
        ),
        encoding="utf-8",
    )
    benchmarks = [
        load_benchmark_config(benchmark_dir / "gsm8k.toml"),
        load_benchmark_config(benchmark_dir / "mmlu_pro.toml"),
    ]

    specs = build_supplementary_dataset_specs(benchmarks)
    assert {(spec.slug, spec.asset_id, spec.purpose) for spec in specs} == {
        ("gsm8k", "train", "train"),
        ("mmlu_pro", "validation", "validation"),
    }

