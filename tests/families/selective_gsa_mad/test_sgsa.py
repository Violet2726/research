from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from research_experiments.core.data.datasets import DatasetSample, load_split_ids
from research_experiments.families.blind_reconstructive_mad.config import (
    load_experiment_config,
    load_protocol_config,
    runtime_for_provider,
)
from research_experiments.families.blind_reconstructive_mad.run import sample as sample_runner
from research_experiments.families.blind_reconstructive_mad.run import report as report_runner
from research_experiments.families.blind_reconstructive_mad.run.execute import _use_bbeh_harmonic
from research_experiments.families.selective_gsa_mad.count100_gate import evaluate_count100_gate
from research_experiments.family_runtime.config_helpers import resolve_model


def test_sgsa_config_uses_qwen_throughput_and_long_reviewer_ceiling() -> None:
    experiment = load_experiment_config(
        "configs/families/selective_gsa_mad/experiments/sgsa_mad.toml"
    )
    protocol = load_protocol_config(experiment.protocol)
    runtime = runtime_for_provider(experiment, "dashscope")

    assert runtime.max_concurrent_requests == 1000
    assert runtime.requests_per_minute_limit == 1000
    assert protocol.reviewer_max_tokens == 8192
    assert resolve_model("dashscope/qwen-flash").timeout_seconds == 300
    assert "sgsa_unanimous_3" in experiment.brd_methods
    assert set(experiment.raw["phases"]) == {"count20_seed42", "count100_seed42", "full_seed42"}
    assert experiment.raw["phases"]["full_seed42"]["full_after_count100_gate"] is True


def test_bbeh_count_uses_micro_and_full_uses_task_harmonic() -> None:
    experiment = load_experiment_config(
        "configs/families/selective_gsa_mad/experiments/sgsa_mad.toml"
    )

    assert not _use_bbeh_harmonic(experiment, "count20_seed42")
    assert not _use_bbeh_harmonic(experiment, "count100_seed42")
    assert _use_bbeh_harmonic(experiment, "full_seed42")


def test_sgsa_report_uses_sgsa_method_and_count_micro_labels() -> None:
    manifest = {
        "experiment": "sgsa_mad",
        "phase": "count100_seed42",
        "method_order": ["sc_5", "sgsa_unanimous_3"],
        "dataset_order": ["bbeh"],
        "resolved_model": {"name": "dashscope/qwen-flash"},
        "output_protocol": "free_text_answer_v1",
    }
    summary = [
        {
            "dataset": dataset,
            "method_name": method,
            "accuracy_mean": score,
            "micro_accuracy_mean": score,
        }
        for dataset in ("overall", "bbeh")
        for method, score in (("sc_5", 0.3), ("sgsa_unanimous_3", 0.31))
    ]
    metrics = {"summary": summary, "bbeh_metric": {"primary": "micro_accuracy", "secondary": None}}
    diagnostics = {"summary_rows": []}
    paired = {
        "reference_method": "sgsa_unanimous_3",
        "tests": [
            {
                "dataset": "bbeh",
                "model_name": "dashscope/qwen-flash",
                "comparison_method": "sc_5",
                "absolute_accuracy_delta": 0.01,
                "bootstrap_ci_95": [-0.01, 0.03],
                "mcnemar_exact_p": 1.0,
                "holm_adjusted_p_within_dataset": 1.0,
            }
        ],
    }
    comparison = report_runner._comparison(
        manifest,
        metrics,
        diagnostics,
        paired,
        family_name="selective_gsa_mad",
    )
    markdown = report_runner._markdown(
        manifest,
        metrics,
        diagnostics,
        paired,
        {"rows": []},
        {"conditions": {}},
        comparison,
        Path("run"),
        "SGSA-MAD",
        family_name="selective_gsa_mad",
    )

    assert "blinded generative synthesis" in markdown
    assert "sgsa_unanimous_3 − sc_5" in markdown
    assert "BBEH metrics (primary: micro_accuracy)" in markdown
    assert "BRD reviewers falsify" not in markdown


def test_gsa_two_thirds_and_sgsa_unanimous_share_three_physical_calls(monkeypatch) -> None:
    calls: list[tuple[str, int]] = []

    def fake_turn(**kwargs):
        method = kwargs["method_name"]
        agent_id = kwargs["agent_id"]
        calls.append((method, agent_id))
        answer = "no" if method == "brd_stage_a_shared" and agent_id < 4 else "yes"
        return {
            "method_name": method,
            "agent_id": agent_id,
            "normalized_answer": answer,
            "prediction": answer,
            "assistant_text": f"REASONING: check\nFINAL_ANSWER: {answer}",
            "protocol_parse_status": "ok",
            "reason_present": True,
            "prompt_tokens": 2.0,
            "completion_tokens": 2.0,
            "total_tokens": 4.0,
            "latency_ms": 1.0,
        }

    monkeypatch.setattr(sample_runner, "_execute_turn", fake_turn)
    experiment = load_experiment_config(
        "configs/families/selective_gsa_mad/experiments/sgsa_mad.toml"
    )
    protocol = load_protocol_config(experiment.protocol)
    sample = DatasetSample("strategyqa", "s", "Is this true?", "yes", "", {})
    turns, _, _, predictions = sample_runner._run_brd_sample(
        sample,
        run_id="r",
        benchmark_slug="strategyqa",
        split_name="count100_seed42",
        experiment=experiment,
        protocol=protocol,
        active_methods=["gsa_quorum_3", "sgsa_unanimous_3"],
        backbone=SimpleNamespace(name="fake"),
        provider=None,
        cache=None,
        throttle=None,
    )

    assert sum(method == "gsa_shared_panel" for method, _ in calls) == 3
    assert len(turns) == 8
    by_method = {row["method_name"]: row for row in predictions}
    assert by_method["gsa_quorum_3"]["quorum_required"] == 2
    assert by_method["sgsa_unanimous_3"]["quorum_required"] == 3
    assert by_method["gsa_quorum_3"]["calls_per_question"] == 8
    assert by_method["sgsa_unanimous_3"]["calls_per_question"] == 8


def test_conditional_resample_reuses_sc_prompt_without_reviewer_token_cap(monkeypatch) -> None:
    captured: list[dict] = []

    def fake_turn(**kwargs):
        captured.append(kwargs)
        method = kwargs["method_name"]
        agent_id = kwargs["agent_id"]
        answer = "no" if method == "brd_stage_a_shared" and agent_id < 4 else "yes"
        return {
            "method_name": method,
            "agent_id": agent_id,
            "normalized_answer": answer,
            "prediction": answer,
            "assistant_text": f"REASONING: check\nFINAL_ANSWER: {answer}",
            "protocol_parse_status": "ok",
            "reason_present": True,
            "prompt_tokens": 2.0,
            "completion_tokens": 2.0,
            "total_tokens": 4.0,
            "latency_ms": 1.0,
        }

    monkeypatch.setattr(sample_runner, "_execute_turn", fake_turn)
    experiment = load_experiment_config(
        "configs/families/selective_gsa_mad/experiments/sgsa_mad.toml"
    )
    protocol = load_protocol_config(experiment.protocol)
    sample = DatasetSample("strategyqa", "s", "Is this true?", "yes", "", {})

    sample_runner._run_brd_sample(
        sample,
        run_id="r",
        benchmark_slug="strategyqa",
        split_name="count100_seed42",
        experiment=experiment,
        protocol=protocol,
        active_methods=["conditional_resample_3"],
        backbone=SimpleNamespace(name="fake"),
        provider=None,
        cache=None,
        throttle=None,
    )

    resamples = [item for item in captured if item["method_name"] == "conditional_resample_3"]
    assert len(resamples) == 3
    for request in resamples:
        assert request["messages"] == sample_runner.build_stage_a_messages(sample, request["agent_id"])
        assert request["max_tokens"] is None
        assert request["agent_role"] == "conditional_sc_resample"


def test_sgsa_uses_shared_canonical_splits() -> None:
    omni20 = load_split_ids("omni-math-2/Omni-Math-2", "count20_seed42")
    omni100 = load_split_ids("omni-math-2/Omni-Math-2", "count100_seed42")
    bbeh20 = load_split_ids("bbeh/bbeh-main", "count20_seed42")
    bbeh100 = load_split_ids("bbeh/bbeh-main", "count100_seed42")

    assert omni20 == omni100[:20]
    assert bbeh20 == bbeh100[:20]


def test_strict_gate_requires_each_dataset_to_improve() -> None:
    rows = []
    for dataset in ("omni_math_2_filtered", "bbeh"):
        for index in range(12):
            sample_id = f"{dataset}-{index}"
            base = 0.0
            sgsa = 1.0
            for method, score in (
                ("sc_5", base),
                ("conditional_resample_3", base),
                ("sgsa_unanimous_3", sgsa),
            ):
                rows.append(
                    {
                        "dataset": dataset,
                        "sample_id": sample_id,
                        "method_name": method,
                        "score": score,
                        "initial_vote_score": base,
                        "candidate_oracle_correct": True,
                        "override_accepted": method == "sgsa_unanimous_3",
                        "corrected_by_debate": method == "sgsa_unanimous_3",
                        "harmed_by_debate": False,
                    }
                )
    gate = evaluate_count100_gate(
        prediction_rows=rows,
        turn_rows=[],
        diagnostics={},
        model_name="dashscope/qwen-flash",
    )

    assert gate["passed"]
    assert gate["evidence"]["override_count"] == 24

