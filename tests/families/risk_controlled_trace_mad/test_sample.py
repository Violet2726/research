from __future__ import annotations

import json
from types import SimpleNamespace

from research_experiments.core.controls.no_comm_controls import resolve_unified_control_seed
from research_experiments.core.data.datasets import DatasetSample
from research_experiments.core.execution.runner_common import CachedRequestResult
from research_experiments.families.risk_controlled_trace_mad.run import sample as sample_runner


def _sample() -> DatasetSample:
    return DatasetSample(dataset="bbeh", sample_id="s1", question="Choose A or B.", reference_answer="A", prompt_context="", metadata={"task": "x"})


def _stage_row(agent_id: int, answer: str) -> dict:
    return {"agent_id": agent_id, "normalized_answer": answer, "prediction": answer, "validated_output": {"reasoning": f"r{agent_id}"}, "assistant_text": f"r{agent_id}", "prompt_hash": f"h{agent_id}", "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "latency_ms": 1, "protocol_parse_status": "ok", "reason_present": True, "network_attempt_count": 0}


def test_fake_provider_disagreement_uses_one_shared_synthesizer(monkeypatch) -> None:
    stage = [_stage_row(1, "A"), _stage_row(2, "A"), _stage_row(3, "A"), _stage_row(4, "B"), _stage_row(5, "B"), *_stage_row_tail()]
    monkeypatch.setattr(sample_runner, "_run_stage_pool", lambda **_: stage)
    calls = []
    def synth(**kwargs):
        calls.append(kwargs)
        payload = {"reasoning_summary": "minority has the decisive check", "final_answer": "B", "source_trace_ids": ["T1"], "decisive_claim": "check", "certificate_type": "unsupported", "certificate_payload": {}}
        return {"output_status": "ok", "validated_output": payload, "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "latency_ms": 1, "protocol_parse_status": "ok", "reason_present": True, "network_attempt_count": 1, "normalized_answer": "B"}
    monkeypatch.setattr(sample_runner, "_execute_synthesis_turn", synth)
    experiment = SimpleNamespace(global_seed=42, control_methods=["sc_5", "sc_9"], control_prompt_version="single_agent_free_text_v1")
    protocol = SimpleNamespace(trace_max_chars=1200, board_max_chars=7000, synthesis_temperature=0.7, top_p=1.0, synthesis_max_tokens=2048, reasoning_word_limit=120)
    _, _, _, predictions = sample_runner.run_sample(_sample(), run_id="r", dataset="bbeh", split_name="count20_seed42", experiment=experiment, protocol=protocol, active_methods=["gsa_trace_1", "rcta_certificate_shadow_1"], backbone=SimpleNamespace(name="fake"), provider=None, cache=None, throttle=None, router=None)
    assert len(calls) == 1
    assert {row["method_name"] for row in predictions} == {"sc_5", "sc_9", "gsa_trace_1", "rcta_certificate_shadow_1"}
    assert next(row for row in predictions if row["method_name"] == "gsa_trace_1")["logical_calls_per_question"] == 6


def test_fake_provider_five_zero_does_not_synthesize(monkeypatch) -> None:
    monkeypatch.setattr(sample_runner, "_run_stage_pool", lambda **_: [_stage_row(i, "A") for i in range(1, 10)])
    monkeypatch.setattr(sample_runner, "_execute_synthesis_turn", lambda **_: (_ for _ in ()).throw(AssertionError("must not synthesize")))
    experiment = SimpleNamespace(global_seed=42, control_methods=["sc_5"], control_prompt_version="single_agent_free_text_v1")
    protocol = SimpleNamespace(trace_max_chars=1200, board_max_chars=7000, synthesis_temperature=0.7, top_p=1.0, synthesis_max_tokens=2048, reasoning_word_limit=120)
    _, _, _, predictions = sample_runner.run_sample(_sample(), run_id="r", dataset="bbeh", split_name="count20_seed42", experiment=experiment, protocol=protocol, active_methods=["gsa_trace_1"], backbone=SimpleNamespace(name="fake"), provider=None, cache=None, throttle=None, router=None)
    gsa = next(row for row in predictions if row["method_name"] == "gsa_trace_1")
    assert gsa["logical_calls_per_question"] == 5
    assert gsa["triggered"] is False


def test_non_length_json_protocol_failure_replays_once(monkeypatch) -> None:
    valid = {"reasoning_summary": "check", "final_answer": "A", "source_trace_ids": ["T1"], "decisive_claim": "x", "certificate_type": "unsupported", "certificate_payload": {}}
    responses = ["not json", json.dumps(valid)]
    calls = []
    def execute(**kwargs):
        calls.append(kwargs)
        text = responses.pop(0)
        return CachedRequestResult(payload={"seed": kwargs["seed"]}, prompt_hash="p", cache_key="k", cache_hit=False, response_payload={"assistant_text": text, "finish_reason": "stop", "latency_ms": 1, "network_attempt_count": 1, "request_started_at": "2026-01-01T00:00:00+00:00"}, request_error=None, usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2})
    monkeypatch.setattr(sample_runner, "execute_cached_request", execute)
    cache = SimpleNamespace(delete=lambda key: None)
    row = sample_runner._execute_synthesis_turn(run_id="r", dataset="bbeh", split_name="count20_seed42", sample=_sample(), board="board", backbone=None, provider=None, cache=cache, throttle=None, temperature=0.7, top_p=1.0, seed=42, max_tokens=128, reasoning_word_limit=120)
    assert row["output_status"] == "ok"
    assert row["protocol_attempt_count"] == 2
    assert row["network_attempt_count"] == 2
    assert len(row["request_started_at_events"]) == 2
    assert calls[0]["seed"] == calls[1]["seed"] == 42


def test_stage_a_first_five_match_shared_sc_seed_and_uncapped_request(monkeypatch) -> None:
    captured = []

    def execute(**kwargs):
        captured.append(kwargs)
        return _stage_row(kwargs["agent_id"], "A")

    monkeypatch.setattr(sample_runner, "_execute_free_text_turn", execute)
    protocol = SimpleNamespace(sc_ceiling_candidates=9, stage_a_temperature=0.7, top_p=1.0)
    experiment = SimpleNamespace(global_seed=42, control_prompt_version="single_agent_free_text_v1")
    sample_runner._run_stage_pool(
        run_id="r",
        dataset="bbeh",
        split_name="count20_seed42",
        sample=_sample(),
        experiment=experiment,
        protocol=protocol,
        backbone=None,
        provider=None,
        cache=None,
        throttle=None,
    )
    first_five = sorted(captured, key=lambda item: item["agent_id"])[:5]
    assert [item["seed"] for item in first_five] == [
        resolve_unified_control_seed(global_seed=42, method_family="self_consistency", replicate_id=index)
        for index in range(5)
    ]
    assert all(item["max_tokens"] is None for item in first_five)


def _stage_row_tail() -> list[dict]:
    return [_stage_row(6, "A"), _stage_row(7, "B"), _stage_row(8, "C"), _stage_row(9, "D")]
