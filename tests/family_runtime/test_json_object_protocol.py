from __future__ import annotations

import pytest

from research_experiments.core.config import resolve_model_ref
from research_experiments.core.execution.cache import RequestCache
from research_experiments.family_runtime.json_object_protocol import parse_json_object_answer_output
from research_experiments.family_runtime.output_protocols import execute_output_protocol_turn


def test_parse_json_object_answer_output_parses_full_contract() -> None:
    payload = parse_json_object_answer_output(
        (
            '{"reasoning":"Check 6*7.","answer":"42","confidence":0.8,'
            '"risk_level":"low","risk_summary":"arithmetic only","key_evidence":"6*7=42"}'
        ),
        dataset="gsm8k",
    )

    assert payload["final_answer"] == "42"
    assert payload["answer"] == "42"
    assert payload["confidence"] == 0.8
    assert payload["risk_level"] == "low"
    assert payload["reasoning"] == "Check 6*7."


def test_parse_json_object_answer_output_requires_reasoning() -> None:
    with pytest.raises(ValueError, match="reasoning"):
        parse_json_object_answer_output(
            '{"answer":"42","confidence":0.8,"risk_level":"low","key_evidence":"6*7=42"}',
            dataset="gsm8k",
        )


def test_parse_json_object_answer_output_flags_task_format_warning() -> None:
    payload = parse_json_object_answer_output(
        (
            '{"reasoning":"Option C text is supported.","answer":"Option C","confidence":0.7,'
            '"risk_level":"medium","key_evidence":"option text"}'
        ),
        dataset="gpqa_diamond",
    )

    assert payload["format_warning"] == "multiple_choice_answer_not_single_letter"


def test_parse_json_object_answer_output_recovers_pairwise_selected_side_from_balanced_invalid_json() -> None:
    payload = parse_json_object_answer_output(
        (
            '{"reasoning":"The phrase "chemically distinct" points to side Y.",'
            '"answer":"Y","confidence":0.84,"risk_level":"low",'
            '"key_evidence":"distinct chemical clue","selected_side":"Y"}'
        ),
        dataset="gpqa_diamond",
    )

    assert payload["answer"] == "Y"
    assert payload["selected_side"] == "Y"
    assert payload["protocol_recovery"] == "pairwise_selected_side_fallback"


def test_parse_json_object_answer_output_does_not_recover_truncated_pairwise_json() -> None:
    with pytest.raises(ValueError, match="valid JSON"):
        parse_json_object_answer_output(
            '{"reasoning":"long unfinished trace","answer":"Y","confidence":0.84,"selected_side":"Y"',
            dataset="gpqa_diamond",
        )


def test_parse_json_object_answer_output_does_not_recover_conflicting_pairwise_side() -> None:
    with pytest.raises(ValueError, match="valid JSON"):
        parse_json_object_answer_output(
            (
                '{"reasoning":"The phrase "chemically distinct" points to side Y.",'
                '"answer":"X","confidence":0.84,"risk_level":"low",'
                '"key_evidence":"distinct chemical clue","selected_side":"Y"}'
            ),
            dataset="gpqa_diamond",
        )


def test_execute_json_object_answer_protocol_uses_provider_response_format(tmp_path) -> None:
    model = resolve_model_ref("xiaomimimo/mimo-v2.5")
    cache = RequestCache(tmp_path / "json-object.sqlite")
    captured_payloads = []

    def request_executor(payload, provider, throttle):
        del provider, throttle
        captured_payloads.append(payload)
        return {
            "assistant_text": (
                '{"reasoning":"Option B best matches the clue.","answer":"B","confidence":0.9,'
                '"risk_level":"low","risk_summary":"direct option match","key_evidence":"option B clue"}'
            ),
            "provider_reasoning_text": "",
            "finish_reason": "stop",
            "usage_reported": {"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22},
            "usage_estimated": {"prompt_tokens": 10, "completion_tokens": 12, "total_tokens": 22},
            "usage_source": "reported",
            "latency_ms": 11.0,
            "provider_request_id": "req",
            "response_id": "resp",
            "request_started_at": "2026-06-23T00:00:00+00:00",
            "request_error": None,
        }

    result = execute_output_protocol_turn(
        backbone=model,
        provider=object(),
        cache=cache,
        throttle=None,
        sample=None,
        messages=[{"role": "user", "content": "Return JSON."}],
        temperature=0.0,
        top_p=1.0,
        seed=42,
        dataset="gpqa_diamond",
        role="judge",
        output_protocol="json_object_answer_v3",
        max_tokens=512,
        request_executor=request_executor,
    )
    cache.close()

    assert result.output_status == "ok"
    assert captured_payloads[0]["response_format"] == {"type": "json_object"}
    assert captured_payloads[0]["max_tokens"] == 512
