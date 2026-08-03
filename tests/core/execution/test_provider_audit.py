import json

import pytest

from research_experiments.core.execution import provider_audit as provider_audit_module
from research_experiments.core.execution.provider_audit import evaluate_mimo_provider_audit
from research_experiments.families.disagreement_guided_crux_reconstruction.run.execute import (
    _require_passing_provider_audit,
)
from research_experiments.family_runtime.config_helpers import resolve_model


def test_provider_audit_requires_documented_fields() -> None:
    rows = [
        {
            "prompt_kind": "one" if index < 8 else "two",
            "payload": {
                "thinking": {"type": "disabled"},
                "max_completion_tokens": 2048,
                "temperature": 0.7,
                "top_p": 1.0,
                "seed": 42,
            },
            "seed": 42,
            "cap": 2048,
            "thinking_disabled": True,
            "cap_field_correct": True,
            "finish_reason": "stop",
            "reasoning_tokens": 0,
            "network_attempt_count": 1,
            "attempt_timeline": [
                {
                    "attempt_index": 1,
                    "queued_at": "q",
                    "rate_admitted_at": "a",
                    "network_started_at": "s",
                    "network_finished_at": "f",
                }
            ],
            "request_error": None,
            "assistant_text": "same",
            "usage_reported": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "cache_policy": "live_only",
            "request_source": "live_uncached_provider_audit",
            "cache_hit": False,
        }
        for index in range(10)
    ]
    assert evaluate_mimo_provider_audit(rows, expected_cache_policy="live_only")["passed"] is True
    rows[0]["reasoning_tokens"] = None
    assert evaluate_mimo_provider_audit(rows)["passed"] is False


def test_gate_rejects_missing_or_unsuccessful_provider_audit(tmp_path) -> None:
    path = tmp_path / "provider_audit.json"
    with pytest.raises(RuntimeError, match="missing"):
        _require_passing_provider_audit(path, expected_cache_policy="live_only")

    rows = [
        {
            "prompt_kind": "one" if index < 8 else "two",
            "payload": {
                "thinking": {"type": "disabled"},
                "max_completion_tokens": 2048,
                "temperature": 0.7,
                "top_p": 1.0,
                "seed": 42,
            },
            "seed": 42,
            "cap": 2048,
            "thinking_disabled": True,
            "cap_field_correct": True,
            "finish_reason": "stop",
            "reasoning_tokens": 0,
            "network_attempt_count": 1,
            "attempt_timeline": [
                {
                    "attempt_index": 1,
                    "queued_at": "q",
                    "rate_admitted_at": "a",
                    "network_started_at": "s",
                    "network_finished_at": "f",
                }
            ],
            "request_error": None,
            "assistant_text": "same",
            "usage_reported": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "cache_policy": "live_only",
            "request_source": "live_uncached_provider_audit",
            "cache_hit": False,
        }
        for index in range(10)
    ]
    payload = evaluate_mimo_provider_audit(rows, expected_cache_policy="live_only")
    payload.update({"provider": "xiaomimimo", "model_id": "mimo-v2.5"})
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert _require_passing_provider_audit(path, expected_cache_policy="live_only")["passed"] is True


def test_live_audit_matrix_covers_catch_role_and_solver_caps_without_cache(monkeypatch) -> None:
    def fake_request(*, payload, **_kwargs):
        return {
            "assistant_text": payload["messages"][0]["content"],
            "finish_reason": "stop",
            "network_attempt_count": 1,
            "request_started_at_events": ["now"],
            "attempt_timeline": [
                {
                    "attempt_index": 1,
                    "queued_at": "q",
                    "rate_admitted_at": "a",
                    "network_started_at": "s",
                    "network_finished_at": "f",
                }
            ],
            "request_error": None,
            "usage_reported": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
        }

    monkeypatch.setattr(provider_audit_module, "_execute_request_with_retries", fake_request)
    report = provider_audit_module.run_mimo_provider_audit(
        backbone=resolve_model("xiaomimimo/mimo-v2.5"),
        provider=object(),
    )

    assert report["passed"] is True
    assert {row["cap"] for row in report["records"]} == {2_048, 4_096, 32_768, 65_536}
    assert all(row["cache_hit"] is False for row in report["records"])
