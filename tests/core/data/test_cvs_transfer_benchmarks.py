from __future__ import annotations

import json
from types import SimpleNamespace

from research_experiments.core.data.datasets import load_samples
from research_experiments.core.data.evaluation import normalize_prediction, score_prediction


def test_omni_math_jsonl_loader_preserves_domain_and_difficulty(tmp_path) -> None:
    source = tmp_path / "Omni-Math.jsonl"
    source.write_text(
        json.dumps(
            {
                "problem": "Compute 1+1.",
                "answer": "2",
                "solution": "It is 2.",
                "domain": ["Mathematics -> Algebra"],
                "difficulty": 7.0,
                "source": "test",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    samples = load_samples(_config("omni_math", "omni_math_jsonl", source))

    assert len(samples) == 1
    assert samples[0].reference_answer == "2"
    assert samples[0].metadata["domain"] == ["Mathematics -> Algebra"]
    assert samples[0].metadata["difficulty"] == 7.0


def test_bbeh_directory_loader_keeps_task_identity(tmp_path) -> None:
    task_dir = tmp_path / "bbeh" / "benchmark_tasks" / "bbeh_boolean_expressions"
    task_dir.mkdir(parents=True)
    (task_dir / "task.json").write_text(
        json.dumps({"examples": [{"input": "Which expression?", "target": "(E)"}]}),
        encoding="utf-8",
    )

    samples = load_samples(_config("bbeh", "bbeh_json_bundle", tmp_path))

    assert len(samples) == 1
    assert samples[0].reference_answer == "(E)"
    assert samples[0].metadata["task"] == "boolean_expressions"
    assert samples[0].sample_id == "bbeh-boolean_expressions-0000"


def test_transfer_benchmark_scoring_uses_math_and_exact_text_contracts() -> None:
    assert normalize_prediction("omni_math", "\\boxed{2}") == "2"
    assert score_prediction("omni_math", "2", "2") == 1.0
    assert score_prediction("bbeh", "(E)", "(E)") == 1.0
    assert score_prediction("bbeh", "D", "(E)") == 0.0
    assert score_prediction("bbeh", "Ok The answer is: (A)", "a") == 1.0
    assert score_prediction("bbeh", "The final answer is: **25**\nExplanation", "25.0") == 1.0


def _config(slug: str, loader: str, source_path) -> SimpleNamespace:
    return SimpleNamespace(
        slug=slug,
        loader=loader,
        source_path=str(source_path),
        sample_id_prefix=slug,
        random_seed=42,
        record_filters=[],
    )
