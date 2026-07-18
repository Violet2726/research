from __future__ import annotations

from research_experiments.core.execution import runner_common


def test_retryable_request_replays_same_payload_and_stops_after_success(monkeypatch) -> None:
    payload = {"model": "m", "messages": []}
    seen = []
    sleeps = []

    def executor(value, provider, throttle):
        del provider, throttle
        seen.append(value)
        if len(seen) < 3:
            return {"http_status": 429, "request_error": "limited", "retry_after_seconds": 0}
        return {"http_status": 200, "request_error": None}

    monkeypatch.setattr(runner_common.time, "sleep", sleeps.append)
    result = runner_common._execute_request_with_retries(
        payload=payload,
        provider=None,
        throttle=None,
        request_executor=executor,
    )

    assert seen == [payload, payload, payload]
    assert result["network_attempt_count"] == 3
    assert sleeps == [0.0, 0.0]
    assert [item["attempt_index"] for item in result["attempt_timeline"]] == [1, 2, 3]
    assert [item["retry_delay_seconds"] for item in result["attempt_timeline"]] == [0.0, 0.0, None]
