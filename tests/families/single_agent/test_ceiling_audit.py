from __future__ import annotations

import json
from pathlib import Path

from research_experiments.families.single_agent.ceiling_audit import (
    rebaseline_core_conclusions,
    select_screening_candidates,
)
from research_experiments.families.single_agent.config import load_experiment_config
from research_experiments.families.single_agent.run.sample import _reruns_for_method
from research_experiments.family_runtime.method_catalog import load_method_catalog
from tests.testsupport.filesystem import write_json, write_jsonl


def _candidate_spec(*, family: str, budget_calls: int, temperature: float) -> dict[str, object]:
    return {
        "family": family,
        "budget_calls": budget_calls,
        "temperature": temperature,
        "top_p": 1.0,
        "max_output_tokens": 256,
    }


def _write_screen_run(
    root: Path,
    *,
    prompt_version: str,
    method_scores: dict[str, list[int]],
    method_specs: dict[str, dict[str, object]],
) -> None:
    write_json(
        root / "manifest.json",
        {
            "family_name": "single_agent",
            "experiment_name": "baseline_ceiling_v1_test",
            "phase_name": "count20",
            "prompt_version": prompt_version,
            "methods": method_specs,
        },
    )
    rows = []
    for method_name, scores in method_scores.items():
        for index, score in enumerate(scores):
            rows.append(
                {
                    "dataset": "gsm8k" if index < 3 else "hotpotqa",
                    "sample_id": f"{method_name}-{index}",
                    "method_name": method_name,
                    "score": float(score),
                    "rerun_index": 0,
                    "total_tokens_per_question": 100.0 + index,
                }
            )
    write_jsonl(root / "views" / "predictions.jsonl", rows)


def test_select_screening_candidates_keeps_top2_and_near_best(tmp_path: Path) -> None:
    current_run = tmp_path / "current"
    unified_run = tmp_path / "unified"
    zero_shot_run = tmp_path / "zero"
    method_specs = {
        "cot_1_temp_0p0": _candidate_spec(family="cot", budget_calls=1, temperature=0.0),
        "cot_1_temp_0p7": _candidate_spec(family="cot", budget_calls=1, temperature=0.7),
        "mv_3_temp_0p5": _candidate_spec(family="majority_vote", budget_calls=3, temperature=0.5),
        "mv_3_temp_0p7": _candidate_spec(family="majority_vote", budget_calls=3, temperature=0.7),
        "sc_5_temp_0p5": _candidate_spec(family="self_consistency", budget_calls=5, temperature=0.5),
        "sc_5_temp_0p7": _candidate_spec(family="self_consistency", budget_calls=5, temperature=0.7),
    }
    _write_screen_run(
        current_run,
        prompt_version="single_agent_reasoning_json_v1",
        method_scores={
            "cot_1_temp_0p0": [1, 1, 1, 1, 1, 1],
            "cot_1_temp_0p7": [1, 1, 1, 1, 1, 0],
            "mv_3_temp_0p5": [1, 1, 1, 1, 0, 0],
            "mv_3_temp_0p7": [1, 1, 1, 1, 1, 0],
            "sc_5_temp_0p5": [1, 1, 1, 1, 1, 1],
            "sc_5_temp_0p7": [1, 1, 1, 1, 1, 0],
        },
        method_specs=method_specs,
    )
    _write_screen_run(
        unified_run,
        prompt_version="unified_control_v1_port",
        method_scores={
            "cot_1_temp_0p0": [1, 1, 1, 1, 1, 0],
            "cot_1_temp_0p7": [1, 1, 1, 1, 0, 0],
            "mv_3_temp_0p5": [1, 1, 1, 1, 1, 1],
            "mv_3_temp_0p7": [1, 1, 1, 1, 1, 0],
            "sc_5_temp_0p5": [1, 1, 1, 1, 1, 0],
            "sc_5_temp_0p7": [1, 1, 1, 1, 0, 0],
        },
        method_specs=method_specs,
    )
    _write_screen_run(
        zero_shot_run,
        prompt_version="zero_shot_cot_v1",
        method_scores={
            "cot_1_temp_0p0": [1, 1, 1, 1, 1, 0],
            "cot_1_temp_0p7": [1, 1, 1, 1, 1, 0],
            "mv_3_temp_0p5": [1, 1, 1, 1, 1, 0],
            "mv_3_temp_0p7": [1, 1, 1, 1, 0, 0],
            "sc_5_temp_0p5": [1, 1, 1, 1, 1, 1],
            "sc_5_temp_0p7": [1, 1, 1, 1, 1, 0],
        },
        method_specs=method_specs,
    )

    payload = select_screening_candidates(
        run_dirs=[current_run, unified_run, zero_shot_run],
        output_dir=tmp_path / "out",
    )

    selection = json.loads(Path(payload["selection_json"]).read_text(encoding="utf-8"))
    cot_selected = [item for item in selection["selected_candidates"] if item["base_method"] == "cot_1"]
    assert len(cot_selected) >= 3
    assert any(item["method_name"] == "cot_1_temp_0p0" for item in cot_selected)
    assert any(item["prompt_version"] == "zero_shot_cot_v1" for item in cot_selected)
    generated_configs = selection["generated_count100_configs"]
    assert any(item["prompt_version"] == "single_agent_reasoning_json_v1" for item in generated_configs)
    assert any(item["prompt_version"] == "unified_control_v1_port" for item in generated_configs)
    for item in generated_configs:
        config_text = Path(item["experiment_config"]).read_text(encoding="utf-8")
        assert 'competition_math = "count100_total_seed0"' in config_text


def test_cot_reruns_can_be_enabled_for_ceiling_configs() -> None:
    experiment = load_experiment_config("configs/families/single_agent/experiments/baseline_ceiling_v1_current_prompt.toml")
    catalog = load_method_catalog(experiment.method_catalog)
    assert _reruns_for_method(experiment, "count100", catalog["cot_1_temp_0p0"]) == 3

    canonical = load_experiment_config("configs/families/single_agent/experiments/canonical_simple_baselines.toml")
    canonical_catalog = load_method_catalog(canonical.method_catalog)
    assert _reruns_for_method(canonical, "count100", canonical_catalog["cot_1"]) == 3


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
    judgements = {row["method_name"]: row["judgement"] for row in result["per_dataset_rows"]}
    assert judgements["method_only_old_cot"] == "only_beats_old_official_cot1"
    assert judgements["method_strong"] == "holds_vs_canonical_best"
    markdown = Path(payload["markdown_path"]).read_text(encoding="utf-8")
    assert "Canonical Baseline Recheck" in markdown
    assert "method_only_old_cot" in markdown
