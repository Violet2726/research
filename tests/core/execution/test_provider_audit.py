import json

import pytest

from research_experiments.core.execution.provider_audit import evaluate_mimo_provider_audit
from research_experiments.families.disagreement_guided_crux_reconstruction.run.execute import (
    _require_passing_provider_audit,
)


def test_provider_audit_requires_documented_fields() -> None:
    rows = [
        {
            "prompt_kind": "one" if index < 8 else "two",
            "payload": {"thinking": {"type": "disabled"}, "max_completion_tokens": 2048},
            "seed": 42,
            "cap": 2048,
            "thinking_disabled": True,
            "cap_field_correct": True,
            "finish_reason": "stop",
            "reasoning_tokens": 0,
            "network_attempt_count": 1,
            "request_error": None,
            "assistant_text": "same",
            "usage_reported": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "cache_namespace": "dgcr-provider-audit-v1",
            "request_source": "live_uncached_provider_audit",
            "cache_hit": False,
        }
        for index in range(10)
    ]
    assert evaluate_mimo_provider_audit(rows, expected_cache_namespace="dgcr-provider-audit-v1")["passed"] is True
    rows[0]["reasoning_tokens"] = None
    assert evaluate_mimo_provider_audit(rows)["passed"] is False


def test_gate_rejects_missing_or_unsuccessful_provider_audit(tmp_path) -> None:
    path = tmp_path / "provider_audit.json"
    with pytest.raises(RuntimeError, match="missing"):
        _require_passing_provider_audit(path, expected_cache_namespace="dgcr-provider-audit-v1")

    rows = [
        {
            "prompt_kind": "one" if index < 8 else "two",
            "payload": {"thinking": {"type": "disabled"}, "max_completion_tokens": 2048},
            "seed": 42,
            "cap": 2048,
            "thinking_disabled": True,
            "cap_field_correct": True,
            "finish_reason": "stop",
            "reasoning_tokens": 0,
            "network_attempt_count": 1,
            "request_error": None,
            "assistant_text": "same",
            "usage_reported": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "cache_namespace": "dgcr-provider-audit-v1",
            "request_source": "live_uncached_provider_audit",
            "cache_hit": False,
        }
        for index in range(10)
    ]
    payload = evaluate_mimo_provider_audit(rows, expected_cache_namespace="dgcr-provider-audit-v1")
    payload.update({"provider": "xiaomimimo", "model_id": "mimo-v2.5"})
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert _require_passing_provider_audit(path, expected_cache_namespace="dgcr-provider-audit-v1")["passed"] is True
