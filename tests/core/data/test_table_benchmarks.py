"""覆盖结构化表推理 benchmark 的 loader 与评分约束。"""

from __future__ import annotations

import json
from pathlib import Path

from research_experiments.core.config import load_benchmark_config
from research_experiments.core.data.datasets import load_samples
from research_experiments.core.data.evaluation import normalize_prediction, score_prediction


def test_wikitq_loader_reads_processed_table_critic_jsonl(tmp_path: Path) -> None:
    dataset_root = tmp_path / "local" / "datasets" / "wikitq"
    dataset_root.mkdir(parents=True)
    (dataset_root / "test_lower.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "statement": "which country had the most cyclists finish within the top 10?",
                        "table_text": [
                            ["rank", "cyclist", "team"],
                            ["1", "alejandro valverde (esp)", "caisse d'epargne"],
                            ["2", "davide rebellin (ita)", "gerolsteiner"],
                        ],
                        "answer": ["ESP", "ITA"],
                        "ids": "nu-0",
                    },
                    ensure_ascii=False,
                )
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    benchmark_path = tmp_path / "wikitq.toml"
    benchmark_path.write_text(
        "\n".join(
            [
                'name = "WikiTQ"',
                'slug = "wikitq"',
                'loader = "wikitq_jsonl"',
                f'source_path = "{(dataset_root / "test_lower.jsonl").as_posix()}"',
                'source_split = "test"',
                'sample_id_prefix = "wikitq"',
                'question_field = "statement"',
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

    sample = load_samples(load_benchmark_config(benchmark_path))[0]

    assert sample.dataset == "wikitq"
    assert json.loads(sample.reference_answer) == ["ESP", "ITA"]
    assert "Table:" in sample.prompt_context
    assert sample.metadata["question_type"] in {"lookup", "table_qa", "superlative", "count"}


def test_tabfact_loader_reads_processed_table_critic_jsonl_and_cleaned_statement(tmp_path: Path) -> None:
    dataset_root = tmp_path / "local" / "datasets" / "tabfact"
    dataset_root.mkdir(parents=True)
    (dataset_root / "test.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "statement": "the wildcats kept the opposing team scoreless in four games",
                        "label": 1,
                        "table_caption": "1947 kentucky wildcats football team",
                        "table_text": [
                            ["game", "opponent", "opponents"],
                            ["1", "ole miss", "14"],
                            ["2", "cincinnati", "0"],
                            ["3", "georgia", "0"],
                        ],
                        "table_id": "1-24560733-1.html.csv",
                    },
                    ensure_ascii=False,
                )
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (dataset_root / "raw2clean.jsonl").write_text(
        json.dumps(
            {
                "statement": "the wildcats kept the opposing team scoreless in four games",
                "cleaned_statement": "the wildcat keep the oppose team scoreless in 4 game",
                "label": 1,
                "table_caption": "1947 kentucky wildcats football team",
                "table_text": [],
                "table_id": "1-24560733-1.html.csv",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    benchmark_path = tmp_path / "tabfact.toml"
    benchmark_path.write_text(
        "\n".join(
            [
                'name = "TabFact"',
                'slug = "tabfact"',
                'loader = "tabfact_jsonl"',
                f'source_path = "{(dataset_root / "test.jsonl").as_posix()}"',
                'source_split = "test"',
                'sample_id_prefix = "tabfact"',
                'question_field = "statement"',
                'answer_field = "label"',
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

    assert sample.dataset == "tabfact"
    assert sample.reference_answer == "entailed"
    assert sample.metadata["cleaned_statement"] == "the wildcat keep the oppose team scoreless in 4 game"
    assert sample.metadata["question_type"] == "verification"


def test_wikitq_scoring_uses_answer_set_equality() -> None:
    gold = json.dumps(["ESP", "ITA"], ensure_ascii=False)
    assert score_prediction("wikitq", "ESP | ITA", gold) == 1.0
    assert score_prediction("wikitq", "ESP", gold) == 0.0


def test_tabfact_label_normalization_accepts_boolean_synonyms() -> None:
    assert normalize_prediction("tabfact", "YES") == "entailed"
    assert normalize_prediction("tabfact", "false") == "refuted"
    assert score_prediction("tabfact", "yes", "entailed") == 1.0
    assert score_prediction("tabfact", "refuted", "entailed") == 0.0
