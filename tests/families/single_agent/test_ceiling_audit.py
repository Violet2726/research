from __future__ import annotations

import json
from pathlib import Path

from research_experiments.families.single_agent.ceiling_audit import select_screening_candidates
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

    legacy = load_experiment_config("configs/families/single_agent/experiments/same_context_main_table.toml")
    legacy_catalog = load_method_catalog(legacy.method_catalog)
    assert _reruns_for_method(legacy, "count100", legacy_catalog["cot_1"]) == 1
