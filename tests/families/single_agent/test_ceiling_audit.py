from __future__ import annotations

import json
from pathlib import Path

from research_experiments.families.single_agent.ceiling_audit import rebaseline_core_conclusions
from research_experiments.families.single_agent.config import load_experiment_config
from research_experiments.families.single_agent.run.sample import _reruns_for_method
from research_experiments.family_runtime.method_catalog import load_method_catalog
from tests.testsupport.filesystem import write_json


def test_canonical_cot_reruns_are_enabled_for_count100() -> None:
    canonical = load_experiment_config("configs/families/single_agent/experiments/canonical_simple_baselines.toml")
    canonical_catalog = load_method_catalog(canonical.method_catalog)

    assert _reruns_for_method(canonical, "count100", canonical_catalog["cot_1"]) == 3
    assert _reruns_for_method(canonical, "count100", canonical_catalog["mv_3"]) == 3
    assert _reruns_for_method(canonical, "count100", canonical_catalog["sc_5"]) == 3


def test_rebaseline_core_conclusions_classifies_methods_against_canonical_best(tmp_path: Path) -> None:
    canonical_summary = tmp_path / "ceiling_summary.json"
    write_json(
        canonical_summary,
        {
            "per_dataset_rows": [
                {
                    "base_method": "cot_1",
                    "dataset": "math500",
                    "reference_accuracy": 0.60,
                    "optimized_mean_accuracy": 0.68,
                },
                {
                    "base_method": "mv_3",
                    "dataset": "math500",
                    "reference_accuracy": 0.66,
                    "optimized_mean_accuracy": 0.72,
                },
                {
                    "base_method": "sc_5",
                    "dataset": "math500",
                    "reference_accuracy": 0.70,
                    "optimized_mean_accuracy": 0.75,
                },
                {
                    "base_method": "cot_1",
                    "dataset": "hotpotqa",
                    "reference_accuracy": 0.55,
                    "optimized_mean_accuracy": 0.65,
                },
                {
                    "base_method": "mv_3",
                    "dataset": "hotpotqa",
                    "reference_accuracy": 0.57,
                    "optimized_mean_accuracy": 0.66,
                },
                {
                    "base_method": "sc_5",
                    "dataset": "hotpotqa",
                    "reference_accuracy": 0.59,
                    "optimized_mean_accuracy": 0.67,
                },
            ],
        },
    )
    run_dir = tmp_path / "method_run"
    write_json(
        run_dir / "manifest.json",
        {
            "family_name": "adaptive_sparse_mad",
            "experiment_name": "demo_main",
            "phase_name": "count100",
            "prompt_version": "demo",
            "controls": {},
        },
    )
    write_json(
        run_dir / "views" / "metrics.json",
        {
            "summary": [
                {
                    "dataset": "math500",
                    "method_name": "cot_1",
                    "method_kind": "control",
                    "question_count": 100,
                    "accuracy_mean": 0.62,
                },
                {
                    "dataset": "math500",
                    "method_name": "method_only_old_cot",
                    "method_kind": "aggregate",
                    "question_count": 100,
                    "accuracy_mean": 0.66,
                },
                {
                    "dataset": "hotpotqa",
                    "method_name": "method_strong",
                    "method_kind": "aggregate",
                    "question_count": 100,
                    "accuracy_mean": 0.70,
                },
            ],
        },
    )

    payload = rebaseline_core_conclusions(
        canonical_summary_json=canonical_summary,
        run_dirs=[run_dir],
        output_dir=tmp_path / "out",
        focus_datasets=("math500", "hotpotqa"),
    )

    result = json.loads(Path(payload["json_path"]).read_text(encoding="utf-8"))
    assert result["baseline_mainline"]["config"] == "configs/families/single_agent/experiments/canonical_simple_baselines.toml"
    judgements = {row["method_name"]: row["judgement"] for row in result["per_dataset_rows"]}
    assert judgements["method_only_old_cot"] == "only_beats_old_official_cot1"
    assert judgements["method_strong"] == "holds_vs_canonical_best"
    markdown = Path(payload["markdown_path"]).read_text(encoding="utf-8")
    assert "Canonical Baseline Recheck" in markdown
    assert "canonical_simple_baselines.toml" in markdown
    assert "method_only_old_cot" in markdown
