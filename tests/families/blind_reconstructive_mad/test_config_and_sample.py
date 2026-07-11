from __future__ import annotations

import json
from types import SimpleNamespace

from research_experiments.core.controls.control_prompts import FREE_TEXT_V1_PROMPT_VERSION, build_cot_messages
from research_experiments.core.data.datasets import DatasetSample
from research_experiments.families.blind_reconstructive_mad.config import (
    BrdMadExperimentConfig,
    load_experiment_config,
    load_protocol_config,
    runtime_for_provider,
)
from research_experiments.families.blind_reconstructive_mad.prompts import build_stage_a_messages
from research_experiments.families.blind_reconstructive_mad.run import execute as execute_runner
from research_experiments.families.blind_reconstructive_mad.run import sample as sample_runner
from research_experiments.family_runtime.config_helpers import resolve_model


def test_frozen_v1_config_and_mimo_profile() -> None:
    experiment = load_experiment_config("configs/families/blind_reconstructive_mad/experiments/brd_mad_pilot.toml")
    protocol = load_protocol_config(experiment.protocol)

    assert protocol.stage_a_candidates == 5
    assert protocol.reviewer_count == 3
    assert protocol.hide_vote_counts is True
    assert protocol.strong_majority_quorum == 3
    assert protocol.default_quorum == 2
    assert protocol.novel_answer_mode == "shadow"
    assert runtime_for_provider(experiment, "xiaomimimo").requests_per_minute_limit == 18
    assert runtime_for_provider(experiment, "xiaomimimo").max_concurrent_requests == 8


def test_stage_a_prompt_is_exactly_sc5_control_prompt() -> None:
    sample = DatasetSample("strategyqa", "s", "Is this true?", "yes", "", {})

    assert build_stage_a_messages(sample, 4) == build_cot_messages(sample, 4, FREE_TEXT_V1_PROMPT_VERSION)


def test_fake_provider_sample_smoke_uses_shared_stage_and_safe_four_one_gate(monkeypatch) -> None:
    monkeypatch.setattr(sample_runner, "_execute_turn", _fake_turn)
    experiment = load_experiment_config("configs/families/blind_reconstructive_mad/experiments/brd_mad_pilot.toml")
    protocol = load_protocol_config(experiment.protocol)
    sample = DatasetSample("strategyqa", "s", "Is this true?", "yes", "", {})

    turns, messages, routers, predictions = sample_runner._run_brd_sample(
        sample,
        run_id="r",
        benchmark_slug="strategyqa",
        split_name="pilot",
        experiment=experiment,
        protocol=protocol,
        active_methods=["conditional_resample_3", "brd_quorum_3"],
        backbone=SimpleNamespace(name="fake"),
        provider=None,
        cache=None,
        throttle=None,
    )
    by_method = {row["method_name"]: row for row in predictions}

    assert sum(row["method_name"] == "brd_stage_a_shared" for row in turns) == 5
    assert len(messages) == 6
    assert routers[0]["disagreement_pattern"] == "4-1"
    assert by_method["brd_quorum_3"]["prediction"] == "yes"
    assert by_method["brd_quorum_3"]["quorum_required"] == 3
    assert by_method["brd_quorum_3"]["calls_per_question"] == 8
    assert by_method["brd_quorum_3"]["label_permutations"]


def test_fake_provider_end_to_end_smoke_writes_and_validates_artifacts(monkeypatch, tmp_path) -> None:
    benchmark = tmp_path / "bbeh.toml"
    benchmark.write_text(
        "\n".join(
            [
                'name = "BBEH smoke"',
                'slug = "bbeh"',
                'loader = "bbeh_json_bundle"',
                'source_path = "unused.zip"',
                'source_split = "test"',
                'sample_id_prefix = "bbeh"',
                'question_field = "input"',
                'answer_field = "target"',
                "smoke_size = 1",
                "pilot_size = 1",
                "main_size = 1",
                "random_seed = 1",
                'notes = "烟雾测试"',
            ]
        ),
        encoding="utf-8",
    )
    experiment = BrdMadExperimentConfig(
        name="smoke",
        description="fake provider smoke",
        benchmark_configs=[benchmark],
        protocol="configs/families/blind_reconstructive_mad/protocols/brd_v1.toml",
        control_catalog="configs/families/single_agent/methods/common.toml",
        control_methods=["cot_1", "sc_3", "sc_5"],
        brd_methods=["conditional_resample_3", "brd_quorum_3"],
        method_order=["cot_1", "sc_3", "sc_5", "conditional_resample_3", "brd_quorum_3"],
        global_seed=7,
        control_prompt_version="single_agent_free_text_v1",
        output_protocol="free_text_answer_v1",
        primary_model_ref="dashscope/qwen-flash",
        max_concurrent_requests=1,
        requests_per_minute_limit=18,
        runtime_profiles={},
        raw={"phases": {"smoke": {"split_overrides": {"bbeh": "smoke"}}}},
    )
    sample = DatasetSample("bbeh", "s", "Is this true?", "yes", "", {"task": "task_a"})
    monkeypatch.setattr(sample_runner, "_execute_turn", _fake_turn)
    monkeypatch.setattr(execute_runner, "_execute_turn", _fake_turn)
    monkeypatch.setattr(execute_runner, "load_selected_samples", lambda benchmark, split_name: [sample])
    monkeypatch.setattr(execute_runner, "estimate_work", lambda *args: (20, 5))
    monkeypatch.setattr(execute_runner, "OpenAICompatibleProvider", _FakeProvider)
    monkeypatch.setenv("RESEARCH_REPORTS_ROOT", str(tmp_path / "reports"))

    run_dir = execute_runner.run_experiment(
        experiment,
        "smoke",
        resolve_model("dashscope/qwen-flash"),
        run_root=tmp_path / "runs",
        cache_root=tmp_path / "cache",
    )

    validation = json.loads((run_dir / "run_validation.json").read_text(encoding="utf-8"))
    assert validation["passed"]
    assert (run_dir / "diagnostics" / "pilot_gate.json").exists()
    assert (run_dir / "exports" / "brd_comparison.json").exists()


def _fake_turn(**kwargs):
    method = kwargs["method_name"]
    agent_id = kwargs["agent_id"]
    answer = "no" if method == "brd_stage_a_shared" and agent_id < 5 else "yes"
    return {
        "method_name": method,
        "agent_id": agent_id,
        "normalized_answer": answer,
        "prediction": answer,
        "assistant_text": f"REASONING: fake {method}\nFINAL_ANSWER: {answer}",
        "protocol_parse_status": "ok",
        "reason_present": True,
        "prompt_tokens": 10.0,
        "completion_tokens": 5.0,
        "total_tokens": 15.0,
        "latency_ms": 2.0,
    }


class _FakeProvider:
    def __init__(self, backbone) -> None:
        del backbone

    def close(self) -> None:
        return None
