from __future__ import annotations

from types import SimpleNamespace

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.families.cred_v.run import sample as sample_runner


def test_gpqa_two_of_three_pairwise_triggers_two_shadow_retries(monkeypatch) -> None:
    monkeypatch.setattr(sample_runner, "_execute_turn", _fake_pairwise_turn)
    sample = _mc_sample("gpqa_diamond")
    protocol = SimpleNamespace(
        selection_modes=("gpqa_unanimous_pairwise_duel",),
        shadow_selection_modes=("gpqa_2of3_retry_shadow",),
        pairwise_allowed_datasets=("gpqa_diamond",),
        shadow_pairwise_allowed_datasets=("gpqa_diamond", "mmlu_pro"),
        pairwise_option_count_max=4,
        pairwise_duel_replicates=3,
        shadow_pairwise_retry_replicates=2,
        expansion_model_refs=(),
        verifier_temperature=0.0,
        top_p=1.0,
        verifier_max_tokens=128,
        adaptive_extra_solver_calls=2,
        stage_a_temperature=0.7,
        stage_a_max_tokens=0,
    )

    rows = sample_runner._run_rfs_pairwise_selection(
        run_id="r",
        benchmark_slug="gpqa_diamond",
        split_name="s",
        sample=sample,
        experiment=SimpleNamespace(global_seed=42, verifier_model_refs=["pro"], cred_verification_output_protocol="json_object_answer_v3", cred_output_protocol="free_text_answer_v1"),
        protocol=protocol,
        stage_rows=[],
        vote_decision=SimpleNamespace(final_answer="A"),
        backbone=SimpleNamespace(name="main"),
        provider=None,
        cache=None,
        throttle=None,
        verifier_runtimes=[_runtime()],
        router_bucket="weak_split_select",
        targets=[{"normalized_answer": "B"}],
    )

    assert [row["expansion_mode"] for row in rows] == [
        "gpqa_unanimous_pairwise_duel",
        "gpqa_unanimous_pairwise_duel",
        "gpqa_unanimous_pairwise_duel",
        "gpqa_2of3_retry_shadow",
        "gpqa_2of3_retry_shadow",
    ]


def test_mmlu_shadow_pairwise_runs_without_actual_promotion_mode(monkeypatch) -> None:
    monkeypatch.setattr(sample_runner, "_execute_turn", _fake_pairwise_turn)
    sample = _mc_sample("mmlu_pro")
    protocol = SimpleNamespace(
        selection_modes=("gpqa_unanimous_pairwise_duel",),
        shadow_selection_modes=("mmlu_unanimous_pairwise_shadow",),
        pairwise_allowed_datasets=("gpqa_diamond",),
        shadow_pairwise_allowed_datasets=("gpqa_diamond", "mmlu_pro"),
        pairwise_option_count_max=4,
        pairwise_duel_replicates=3,
        shadow_pairwise_retry_replicates=2,
        expansion_model_refs=(),
        verifier_temperature=0.0,
        top_p=1.0,
        verifier_max_tokens=128,
        adaptive_extra_solver_calls=2,
        stage_a_temperature=0.7,
        stage_a_max_tokens=0,
    )

    rows = sample_runner._run_rfs_pairwise_selection(
        run_id="r",
        benchmark_slug="mmlu_pro",
        split_name="s",
        sample=sample,
        experiment=SimpleNamespace(global_seed=42, verifier_model_refs=["pro"], cred_verification_output_protocol="json_object_answer_v3", cred_output_protocol="free_text_answer_v1"),
        protocol=protocol,
        stage_rows=[],
        vote_decision=SimpleNamespace(final_answer="A"),
        backbone=SimpleNamespace(name="main"),
        provider=None,
        cache=None,
        throttle=None,
        verifier_runtimes=[_runtime()],
        router_bucket="weak_split_select",
        targets=[{"normalized_answer": "B"}],
    )

    assert len(rows) == 3
    assert {row["expansion_mode"] for row in rows} == {"mmlu_unanimous_pairwise_shadow"}


def test_strategyqa_shadow_resample_runs_on_weak_split(monkeypatch) -> None:
    monkeypatch.setattr(sample_runner, "_execute_turn", _fake_strategy_turn)
    protocol = SimpleNamespace(
        selection_modes=("gpqa_unanimous_pairwise_duel",),
        shadow_selection_modes=("strategyqa_resample_shadow",),
        pairwise_allowed_datasets=("gpqa_diamond",),
        shadow_pairwise_allowed_datasets=("gpqa_diamond", "mmlu_pro"),
        pairwise_option_count_max=4,
        pairwise_duel_replicates=3,
        shadow_pairwise_retry_replicates=2,
        expansion_model_refs=(),
        verifier_temperature=0.0,
        top_p=1.0,
        verifier_max_tokens=128,
        adaptive_extra_solver_calls=2,
        stage_a_temperature=0.7,
        stage_a_max_tokens=0,
    )

    rows = sample_runner._run_rfs_pairwise_selection(
        run_id="r",
        benchmark_slug="strategyqa",
        split_name="s",
        sample=DatasetSample("strategyqa", "id", "Question?", "no", "", {}),
        experiment=SimpleNamespace(global_seed=42, verifier_model_refs=["pro"], cred_verification_output_protocol="json_object_answer_v3", cred_output_protocol="free_text_answer_v1"),
        protocol=protocol,
        stage_rows=[],
        vote_decision=SimpleNamespace(final_answer="yes"),
        backbone=SimpleNamespace(name="main"),
        provider=None,
        cache=None,
        throttle=None,
        verifier_runtimes=[],
        router_bucket="weak_split_select",
        targets=[{"normalized_answer": "no"}],
    )

    assert len(rows) == 2
    assert {row["expansion_mode"] for row in rows} == {"strategyqa_resample_shadow"}
    assert all(row["expansion_validation_pass"] for row in rows)


def _mc_sample(dataset: str) -> DatasetSample:
    return DatasetSample(dataset, "id", "Question?", "B", "", {"options": ["a", "b", "c", "d"]})


def _runtime() -> SimpleNamespace:
    return SimpleNamespace(model_ref="pro", backbone=SimpleNamespace(name="pro"), provider=None, cache=None, throttle=None)


def _fake_pairwise_turn(**kwargs):
    agent_id = int(kwargs["agent_id"])
    if agent_id in {901, 902, 1004, 1005, 1101, 1102, 1103}:
        selected_side = "X" if agent_id % 2 == 1 else "Y"
    else:
        selected_side = "Y" if agent_id % 2 == 1 else "X"
    return {
        "dataset": kwargs["dataset"],
        "method_name": kwargs["method_name"],
        "prediction": selected_side,
        "normalized_answer": selected_side,
        "request_status": "ok",
        "output_status": "ok",
        "protocol_parse_status": "ok",
        "validated_output": {"answer": selected_side, "selected_side": selected_side},
        "evidence_quality": 0.0,
    }


def _fake_strategy_turn(**kwargs):
    return {
        "dataset": kwargs["dataset"],
        "method_name": kwargs["method_name"],
        "prediction": "no",
        "normalized_answer": "no",
        "request_status": "ok",
        "output_status": "ok",
        "protocol_parse_status": "ok",
        "validated_output": {"answer": "no", "final_answer": "no"},
        "evidence_quality": 0.0,
    }
