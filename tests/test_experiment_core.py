from __future__ import annotations

from pathlib import Path
import json
import time

import pytest

from experiment_core.cache import CachedResponse, RequestCache, json_dump
from experiment_core.config import load_benchmark_config, load_model_catalog, parse_model_ref, resolve_model_ref
from experiment_core.datasets import generate_split_manifests, load_split_ids, select_samples
from experiment_core.providers import _extract_message_channels, build_payload
from experiment_core.rate_limits import SlidingWindowRateLimiter
from experiment_core.selective_signals import decide_trigger, summarize_confidence_rows
from experiment_core.structured_output import (
    OUTPUT_MODE_BUDGET_BELIEF_UPDATE,
    OUTPUT_MODE_BUDGET_SOLVER,
    OUTPUT_MODE_CORE,
    OUTPUT_MODE_SELECTIVE_COMM,
    OUTPUT_MODE_SPARC_AUDIT,
    OUTPUT_MODE_SPARC_BELIEF_UPDATE,
    OUTPUT_MODE_SPARC_MESSAGE,
    OUTPUT_MODE_SPARC_SOLVER,
    validate_structured_output,
)


def test_parse_model_ref() -> None:
    assert parse_model_ref("dashscope/qwen-turbo-1101") == ("dashscope", "qwen-turbo-1101")


def test_load_model_catalog() -> None:
    catalog = load_model_catalog()
    assert catalog
    assert all("/" in key for key in catalog)


def test_resolve_local_ollama_model_ref() -> None:
    resolved = resolve_model_ref("local_ollama/qwen3:4b")
    assert resolved.provider == "local_ollama"
    assert resolved.model_id == "qwen3:4b"
    assert resolved.base_url == "http://127.0.0.1:11434/v1"
    assert resolved.api_key_env == "OLLAMA_API_KEY"
    assert resolved.reasoning_effort == "none"
    assert resolved.supports_response_format is True


def test_build_payload_maps_thinking_control_by_provider() -> None:
    local_model = resolve_model_ref("local_ollama/qwen3:4b")
    local_payload = build_payload(
        local_model,
        [{"role": "user", "content": "hi"}],
        temperature=0.0,
        top_p=1.0,
        max_output_tokens=16,
        seed=42,
    )
    assert local_payload["reasoning_effort"] == "none"
    assert "enable_thinking" not in local_payload

    dashscope_model = resolve_model_ref("dashscope/qwen-turbo-1101")
    dashscope_payload = build_payload(
        dashscope_model,
        [{"role": "user", "content": "hi"}],
        temperature=0.0,
        top_p=1.0,
        max_output_tokens=16,
        seed=42,
    )
    assert dashscope_payload["enable_thinking"] is False
    assert "reasoning_effort" not in dashscope_payload


def test_validate_core_structured_output() -> None:
    payload = validate_structured_output(
        json.dumps(
            {
                "final_answer": "yes",
                "reasoning": "short",
            }
        ),
        OUTPUT_MODE_CORE,
    )
    assert payload["final_answer"] == "yes"
    assert payload["reasoning"] == "short"


def test_validate_selective_structured_output() -> None:
    payload = validate_structured_output(
        json.dumps(
            {
                "final_answer": "yes",
                "confidence_raw": 0.4,
                "reasoning": "short",
                "key_evidence": "one clue",
                "uncertain_point": None,
            }
        ),
        OUTPUT_MODE_SELECTIVE_COMM,
    )
    assert payload["confidence_raw"] == 0.4
    assert payload["key_evidence"] == "one clue"
    assert payload["uncertain_point"] is None


def test_validate_budget_solver_structured_output() -> None:
    payload = validate_structured_output(
        json.dumps(
            {
                "final_answer": "yes",
                "reasoning_trace": "short trace",
                "claim_span": "one claim",
                "key_evidence": "one evidence",
                "keyword_clues": ["alpha", "beta"],
                "confidence_raw": 0.8,
                "uncertain_point": None,
            }
        ),
        OUTPUT_MODE_BUDGET_SOLVER,
    )
    assert payload["keyword_clues"] == ["alpha", "beta"]
    assert payload["confidence_raw"] == 0.8


def test_validate_budget_belief_update_structured_output() -> None:
    payload = validate_structured_output(
        json.dumps(
            {
                "changed_answer": False,
                "new_answer": "no",
                "confidence_delta": 0.1,
                "reason_for_change": "peer confirms it",
                "remaining_disagreement": None,
            }
        ),
        OUTPUT_MODE_BUDGET_BELIEF_UPDATE,
    )
    assert payload["changed_answer"] is False
    assert payload["new_answer"] == "no"


def test_validate_sparc_solver_structured_output() -> None:
    payload = validate_structured_output(
        json.dumps(
            {
                "final_answer": "42",
                "reasoning_trace": "short trace",
                "claim_span": "2+40",
                "confidence_raw": 0.7,
                "uncertain_point": "unit conversion",
                "key_evidence": "5 groups of 8 and 2 extra",
            }
        ),
        OUTPUT_MODE_SPARC_SOLVER,
    )
    assert payload["final_answer"] == "42"
    assert payload["confidence_raw"] == 0.7


def test_validate_sparc_message_structured_output() -> None:
    payload = validate_structured_output(
        json.dumps(
            {
                "final_answer": "yes",
                "confidence_raw": "0.8",
                "claim_span": "the key factual claim",
            }
        ),
        OUTPUT_MODE_SPARC_MESSAGE,
    )
    assert payload["claim_span"] == "the key factual claim"


def test_validate_sparc_belief_update_structured_output() -> None:
    payload = validate_structured_output(
        json.dumps(
            {
                "changed_answer": True,
                "new_answer": "no",
                "confidence_delta": -0.1,
                "reason_for_change": "peer evidence is stronger",
                "remaining_disagreement": None,
            }
        ),
        OUTPUT_MODE_SPARC_BELIEF_UPDATE,
    )
    assert payload["changed_answer"] is True
    assert payload["new_answer"] == "no"


def test_validate_sparc_audit_structured_output() -> None:
    payload = validate_structured_output(
        json.dumps(
            {
                "decision": "resolve_for_a",
                "verified_answer": "48",
                "rationale": "candidate A matches the evidence",
            }
        ),
        OUTPUT_MODE_SPARC_AUDIT,
    )
    assert payload["decision"] == "resolve_for_a"


def test_selective_signal_summary_and_decision() -> None:
    summary = summarize_confidence_rows(
        [
            {"agent_id": 1, "confidence_valid": True, "confidence_value": 0.9},
            {"agent_id": 2, "confidence_valid": True, "confidence_value": 0.4},
            {"agent_id": 3, "confidence_valid": False, "confidence_value": None},
        ]
    )
    assert summary.mean_confidence == 0.65
    assert summary.confidence_spread == 0.5
    assert summary.invalid_agent_ids == [3]
    decision = decide_trigger(
        trigger_type="hybrid_trigger",
        initial_disagreement=False,
        mean_confidence=summary.mean_confidence,
        confidence_spread=summary.confidence_spread,
        any_invalid_confidence=summary.any_invalid_confidence,
        fail_open_to_always=False,
    )
    assert decision.triggered is True


@pytest.mark.parametrize(
    ("raw_text", "mode"),
    [
        ('```json\n{"final_answer":"yes","reasoning":"short"}\n```', OUTPUT_MODE_CORE),
        ("The answer is yes.", OUTPUT_MODE_CORE),
        ('{"final_answer":"yes","confidence_raw":"high","reasoning":"short","key_evidence":null,"uncertain_point":null}', OUTPUT_MODE_SELECTIVE_COMM),
        ('{"final_answer":"yes","reasoning_trace":"short","claim_span":"claim","key_evidence":"evidence","keyword_clues":[],"confidence_raw":0.8,"uncertain_point":null}', OUTPUT_MODE_BUDGET_SOLVER),
        ('{"final_answer":"yes","reasoning":"short"}{"final_answer":"no","reasoning":"alt"}', OUTPUT_MODE_CORE),
        ('{"reasoning":"short"}', OUTPUT_MODE_CORE),
        ('{"final_answer":"yes"}', OUTPUT_MODE_SELECTIVE_COMM),
        ('{"final_answer":"yes","unexpected":"field"}', OUTPUT_MODE_CORE),
    ],
)
def test_validate_structured_output_rejects_malformed_payloads(raw_text: str, mode: str) -> None:
    with pytest.raises(ValueError):
        validate_structured_output(raw_text, mode)  # type: ignore[arg-type]


def test_extract_message_channels_supports_reasoning_metadata() -> None:
    assistant_text, provider_reasoning_text = _extract_message_channels(
        {
            "choices": [
                {
                    "message": {
                        "content": [{"type": "text", "text": '{"final_answer":"yes","reasoning":"short"}'}],
                        "reasoning": "hidden provider reasoning",
                    }
                }
            ]
        }
    )
    assert assistant_text.startswith('{"final_answer":"yes"')
    assert provider_reasoning_text == "hidden provider reasoning"


def test_generate_and_load_split_manifests(tmp_path: Path) -> None:
    source_path = tmp_path / "gsm8k.jsonl"
    source_path.write_text(
        "\n".join(
            [
                json.dumps({"question": "1+1?", "answer": "#### 2"}, ensure_ascii=False),
                json.dumps({"question": "2+2?", "answer": "#### 4"}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    benchmark_path = tmp_path / "benchmark.toml"
    benchmark_path.write_text(
        "\n".join(
            [
                'name = "Toy GSM8K"',
                'slug = "toy_gsm8k"',
                'loader = "gsm8k_jsonl"',
                f'source_path = "{source_path.as_posix()}"',
                'source_split = "test"',
                'sample_id_prefix = "toy"',
                'question_field = "question"',
                'answer_field = "answer"',
                'smoke_size = 1',
                'pilot_size = 2',
                'main_size = 2',
                'random_seed = 42',
                'notes = ""',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    benchmark = load_benchmark_config(benchmark_path)
    created = generate_split_manifests([benchmark], tmp_path / "splits")
    assert created
    smoke_ids = load_split_ids("toy_gsm8k", "smoke20_seed42", tmp_path / "splits")
    samples = select_samples(benchmark, "smoke20_seed42", tmp_path / "splits")
    assert len(smoke_ids) == 1
    assert [sample.sample_id for sample in samples] == smoke_ids


def test_request_cache_round_trip(tmp_path: Path) -> None:
    cache = RequestCache(tmp_path / "requests.sqlite")
    record = CachedResponse(
        cache_key="abc",
        payload_json=json_dump({"a": 1}),
        response_json=json_dump({"b": 2}),
        http_status=200,
        latency_ms=12.5,
        provider_request_id="req_1",
    )
    cache.put(record)
    loaded = cache.get("abc")
    cache.close()
    assert loaded == record


def test_rate_limiter_without_waiting() -> None:
    limiter = SlidingWindowRateLimiter(requests_per_minute=100, tokens_per_minute=1000)
    started = time.monotonic()
    limiter.acquire(10)
    limiter.acquire(10)
    assert time.monotonic() - started < 1.0
