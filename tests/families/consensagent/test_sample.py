from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

import research_experiments.families.consensagent.run.sample as sample_mod
from research_experiments.core.data.datasets import DatasetSample
from research_experiments.families.consensagent.algorithms import TriggerState
from research_experiments.families.consensagent.config import (
    AgentProfile,
    ExperimentSetup,
    Phase3Config,
    ProtocolConfig,
    RosterConfig,
    TriggerConfig,
)


def test_run_consensagent_sample_keeps_last_debate_round_when_consensus_breaks(monkeypatch) -> None:
    sample = _demo_sample()
    setup = ExperimentSetup(name="consensagent_3a", protocol=Path("protocol.toml"), roster=Path("roster.toml"))
    protocol = ProtocolConfig(
        agent_count=3,
        max_debate_rounds=2,
        initial_temperature=0.3,
        debate_temperature=0.3,
        top_p=0.95,
        trigger=TriggerConfig(stagnation_threshold=1, sycophancy_consistency_threshold=0.67, check_sycophancy_on_consensus=True),
        phase3=Phase3Config(enabled=False),
    )
    roster = _demo_roster()
    backbone = SimpleNamespace(name="demo-model")

    monkeypatch.setattr(sample_mod, "build_initial_messages", lambda *args, **kwargs: [{"role": "user", "content": "initial"}])
    monkeypatch.setattr(sample_mod, "build_debate_messages", lambda *args, **kwargs: [{"role": "user", "content": "debate"}])
    monkeypatch.setattr(sample_mod, "_execute_turn", _fake_execute_turn)
    monkeypatch.setattr(sample_mod, "check_triggers", lambda **kwargs: TriggerState())

    turn_rows, debate_rows, prediction_row = sample_mod._run_consensagent_sample(
        sample,
        run_id="demo-run",
        benchmark_slug="gsm8k",
        split_name="count100_seed42",
        setup=setup,
        protocol=protocol,
        roster=roster,
        backbone=backbone,
        provider=None,
        cache=None,
        throttle=None,
        global_seed=7,
        prompt_version="consensagent_paper_v1",
    )

    assert prediction_row["prediction"] == "83"
    assert prediction_row["final_vote_prediction"] == "83"
    assert prediction_row["weighted_prediction"] == "83"
    assert prediction_row["actual_debate_rounds"] == 1
    assert prediction_row["calls_per_question"] == 6
    assert len(turn_rows) == 6
    assert len(debate_rows) == 6
    assert all(row["role"] != "optimizer" for row in turn_rows)


def test_run_consensagent_sample_logs_phase3_rounds_and_uses_latest_round_for_final_vote(monkeypatch) -> None:
    sample = _demo_sample()
    setup = ExperimentSetup(name="consensagent_3a", protocol=Path("protocol.toml"), roster=Path("roster.toml"))
    protocol = ProtocolConfig(
        agent_count=3,
        max_debate_rounds=2,
        initial_temperature=0.3,
        debate_temperature=0.3,
        top_p=0.95,
        trigger=TriggerConfig(stagnation_threshold=1, sycophancy_consistency_threshold=0.67, check_sycophancy_on_consensus=True),
        phase3=Phase3Config(enabled=True, post_optimization_rounds=1),
    )
    roster = _demo_roster()
    backbone = SimpleNamespace(name="demo-model")

    monkeypatch.setattr(sample_mod, "build_initial_messages", lambda *args, **kwargs: [{"role": "user", "content": "initial"}])
    monkeypatch.setattr(sample_mod, "build_debate_messages", lambda *args, **kwargs: [{"role": "user", "content": "debate"}])
    monkeypatch.setattr(sample_mod, "build_optimizer_messages", lambda *args, **kwargs: [{"role": "user", "content": "optimizer"}])
    monkeypatch.setattr(sample_mod, "_optimized_system_prompt", lambda persona_instruction="": "optimized-system")
    monkeypatch.setattr(sample_mod, "_format_optimized_debate_prompt", lambda **kwargs: f"PREV={kwargs['previous_answer']}")
    monkeypatch.setattr(sample_mod, "_execute_turn", _fake_execute_turn)
    monkeypatch.setattr(
        sample_mod,
        "check_triggers",
        lambda **kwargs: TriggerState(sycophancy_triggered=True, trigger_round=1, trigger_type="copycat"),
    )

    turn_rows, debate_rows, prediction_row = sample_mod._run_consensagent_sample(
        sample,
        run_id="demo-run",
        benchmark_slug="gsm8k",
        split_name="count100_seed42",
        setup=setup,
        protocol=protocol,
        roster=roster,
        backbone=backbone,
        provider=None,
        cache=None,
        throttle=None,
        global_seed=7,
        prompt_version="consensagent_paper_v1",
    )

    role_counts = {}
    for row in turn_rows:
        role_counts[row["role"]] = role_counts.get(row["role"], 0) + 1

    assert prediction_row["prediction"] == "83"
    assert prediction_row["final_vote_prediction"] == "83"
    assert prediction_row["weighted_prediction"] == "83"
    assert prediction_row["actual_debate_rounds"] == 2
    assert prediction_row["calls_per_question"] == 10
    assert prediction_row["total_tokens_per_question"] == 10.0
    assert prediction_row["debate_total_tokens_per_question"] == 7.0
    assert role_counts == {"initial": 3, "debate": 3, "optimizer": 1, "debate_optimized": 3}
    assert len(debate_rows) == 12


def test_validate_consensagent_output_recovers_answer_from_malformed_json() -> None:
    text = """{
  "final_answer": "1954",
  "reasoning": "The context says Matt Groening was born in 1954, and the title includes "Matt" in quotes.",
  "confidence": 0.82
}"""

    payload = sample_mod._validate_consensagent_output(text, "")

    assert payload["final_answer"] == "1954"
    assert payload["confidence"] == 0.82
    assert "Matt Groening" in payload["reasoning"]


def test_validate_consensagent_output_rejects_provider_soft_rejection() -> None:
    with pytest.raises(ValueError, match="soft rejection"):
        sample_mod._validate_consensagent_output("The request was rejected because it was considered high risk", "")


def test_validate_optimizer_output_accepts_plain_refined_prompt() -> None:
    payload = sample_mod._validate_optimizer_output(
        "Solve the problem step by step and output only the final number.",
        "",
    )

    assert payload == {
        "final_answer": "Solve the problem step by step and output only the final number.",
        "reasoning": "",
        "confidence": 1.0,
    }


def _demo_sample() -> DatasetSample:
    return DatasetSample(
        dataset="gsm8k",
        sample_id="gsm8k-00057",
        question="demo question",
        reference_answer="83",
        prompt_context="",
        metadata={},
    )


def _demo_roster() -> RosterConfig:
    return RosterConfig(
        agents=[
            AgentProfile(agent_id=1, persona_name="a1", persona_instruction="persona-1"),
            AgentProfile(agent_id=2, persona_name="a2", persona_instruction="persona-2"),
            AgentProfile(agent_id=3, persona_name="a3", persona_instruction="persona-3"),
        ]
    )


def _fake_execute_turn(
    *,
    run_id: str,
    dataset: str,
    split_name: str,
    sample: DatasetSample,
    method_name: str,
    method_type: str,
    round_index: int,
    agent_id: int,
    role: str,
    visible_peer_count: int,
    messages: list[dict[str, str]],
    **_: object,
) -> dict[str, object]:
    if role == "initial":
        answer = "82"
    elif role == "debate":
        answer = "83"
    elif role == "optimizer":
        answer = "optimized_prompt"
    elif role == "debate_optimized":
        user_content = next(msg["content"] for msg in messages if msg["role"] == "user")
        match = re.search(r"PREV=(\S+)", user_content)
        answer = match.group(1) if match else "missing"
    else:  # pragma: no cover - defensive
        raise AssertionError(f"Unexpected role: {role}")

    return {
        "run_id": run_id,
        "dataset": dataset,
        "split": split_name,
        "sample_id": sample.sample_id,
        "method_name": method_name,
        "method_type": method_type,
        "round_index": round_index,
        "agent_id": agent_id,
        "role": role,
        "prompt_hash": f"{role}-{agent_id}-{round_index}",
        "prediction": answer,
        "confidence": 0.9,
        "output_status": "ok",
        "prompt_tokens": 1.0,
        "completion_tokens": 0.0,
        "total_tokens": 1.0,
        "latency_ms": 1.0,
        "cache_hit": False,
        "request_error": None,
        "visible_peer_count": visible_peer_count,
        "payload": {"messages": messages},
        "assistant_text": "",
        "provider_reasoning_text": "",
        "validated_output": {
            "final_answer": answer,
            "reasoning": f"{role}:{answer}",
            "confidence": 0.9,
        },
        "normalized_answer": answer,
    }
