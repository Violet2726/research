"""供确认实验使用的、仅真实网络调用的 provider 合约审计。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from research_experiments.core.execution.providers import build_payload
from research_experiments.core.execution.runner_common import _execute_request_with_retries

AUDIT_PROMPTS = (
    "Return exactly the JSON object {\"audit\":\"one\"}.",
    "Return exactly the JSON object {\"audit\":\"two\"}.",
)


def run_mimo_provider_audit(
    *,
    backbone,
    provider,
    cache_namespace: str = "dgcr-provider-audit-v1",
) -> dict[str, Any]:
    """Run ten uncached requests and report, rather than assume, seed behavior.

    The caller intentionally supplies no cache.  Identical payloads must reach
    the provider so this audit can observe repeatability rather than cache hits.
    """

    specs: list[tuple[str, int, int]] = [
        (AUDIT_PROMPTS[0], 2_048, 42),
        (AUDIT_PROMPTS[0], 4_096, 42),
        (AUDIT_PROMPTS[1], 16_384, 42),
        *[(AUDIT_PROMPTS[0], 2_048, 42) for _ in range(4)],
        (AUDIT_PROMPTS[0], 2_048, 41),
        (AUDIT_PROMPTS[0], 2_048, 43),
        (AUDIT_PROMPTS[0], 2_048, 44),
    ]
    records: list[dict[str, Any]] = []
    for index, (prompt, cap, seed) in enumerate(specs, start=1):
        payload = build_payload(
            backbone,
            [{"role": "user", "content": prompt}],
            temperature=0.7,
            top_p=1.0,
            seed=seed,
            max_tokens=cap,
        )
        # This is deliberately uncached, but it still records every retry so
        # a provider-side failure cannot be mistaken for one logical attempt.
        response = _execute_request_with_retries(
            payload=payload,
            provider=provider,
            throttle=None,
            request_executor=None,
        )
        records.append(
            _audit_record(
                index=index,
                payload=payload,
                response=response,
                cap=cap,
                seed=seed,
                cache_namespace=cache_namespace,
            )
        )
    report = evaluate_mimo_provider_audit(records, expected_cache_namespace=cache_namespace)
    report["provider"] = backbone.provider
    report["model_id"] = backbone.model_id
    return report


def evaluate_mimo_provider_audit(
    records: Iterable[dict[str, Any]],
    *,
    expected_cache_namespace: str | None = None,
) -> dict[str, Any]:
    """Evaluate audit records; suitable for deterministic unit tests."""

    rows = [dict(row) for row in records]
    same_seed_texts = [
        str(row.get("assistant_text") or "")
        for row in rows
        if row.get("seed") == 42 and row.get("cap") == 2_048 and row.get("prompt_kind") == "one"
    ]
    same_seed_consistent = len(set(same_seed_texts)) <= 1 if same_seed_texts else False
    conditions = {
        "ten_live_requests": len(rows) == 10,
        "all_requests_succeeded": bool(rows) and all(not row.get("request_error") for row in rows),
        "all_records_capture_full_payload": bool(rows) and all(isinstance(row.get("payload"), dict) for row in rows),
        "all_payloads_disable_thinking": bool(rows) and all(
            row.get("thinking_disabled") and row.get("payload", {}).get("thinking") == {"type": "disabled"}
            for row in rows
        ),
        "all_payloads_use_max_completion_tokens": bool(rows) and all(
            row.get("cap_field_correct")
            and row.get("payload", {}).get("max_completion_tokens") == row.get("cap")
            and "max_tokens" not in row.get("payload", {})
            for row in rows
        ),
        "all_responses_report_finish_reason": bool(rows) and all(row.get("finish_reason") for row in rows),
        "all_responses_report_usage": bool(rows) and all(isinstance(row.get("usage_reported"), dict) for row in rows),
        "reported_reasoning_tokens_zero": bool(rows) and all(row.get("reasoning_tokens") == 0 for row in rows),
        "network_attempts_recorded": bool(rows) and all(int(row.get("network_attempt_count") or 0) >= 1 for row in rows),
        "all_requests_are_uncached_live_audit_calls": bool(rows) and all(
            row.get("request_source") == "live_uncached_provider_audit" and not row.get("cache_hit")
            for row in rows
        ),
    }
    if expected_cache_namespace is not None:
        conditions["all_records_use_expected_audit_namespace"] = bool(rows) and all(
            row.get("cache_namespace") == expected_cache_namespace for row in rows
        )
    return {
        "audit_kind": "mimo_live_provider_contract_v1",
        "cache_mode": "bypassed",
        "passed": all(conditions.values()),
        "conditions": conditions,
        "seed_observation": {
            "same_seed_repeat_count": len(same_seed_texts),
            "same_seed_outputs_identical": same_seed_consistent,
            "claim": "observational_only_not_a_determinism_guarantee",
        },
        "records": rows,
    }


def _audit_record(
    *,
    index: int,
    payload: dict[str, Any],
    response: dict[str, Any],
    cap: int,
    seed: int,
    cache_namespace: str,
) -> dict[str, Any]:
    usage = dict(response.get("usage_reported") or {})
    completion_details = usage.get("completion_tokens_details") or {}
    reasoning_tokens = usage.get("reasoning_tokens", completion_details.get("reasoning_tokens"))
    return {
        "index": index,
        "payload": payload,
        "cache_namespace": cache_namespace,
        "request_source": "live_uncached_provider_audit",
        "cache_hit": False,
        "prompt_kind": "one" if "one" in str(payload["messages"][0]["content"]) else "two",
        "seed": seed,
        "cap": cap,
        "thinking_disabled": payload.get("thinking") == {"type": "disabled"},
        "cap_field_correct": payload.get("max_completion_tokens") == cap and "max_tokens" not in payload,
        "finish_reason": response.get("finish_reason"),
        "reasoning_tokens": reasoning_tokens,
        "actual_prompt_tokens": usage.get("prompt_tokens"),
        "actual_completion_tokens": usage.get("completion_tokens"),
        "actual_total_tokens": usage.get("total_tokens"),
        "network_attempt_count": int(response.get("network_attempt_count") or 1),
        "request_started_at_events": list(response.get("request_started_at_events") or []),
        "request_error": response.get("request_error"),
        "assistant_text": str(response.get("assistant_text") or ""),
        "usage_reported": usage,
    }
