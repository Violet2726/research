from __future__ import annotations

import requests

from research_experiments.workspace.hf import common


def test_run_hf_request_falls_back_to_direct_connection_on_proxy_error(monkeypatch) -> None:
    configured_modes: list[bool] = []
    call_count = 0

    monkeypatch.setattr(common, "_HF_HTTP_BACKEND_CONFIGURED", False)
    monkeypatch.setattr(common, "_HF_HTTP_TRUST_ENV", True)
    monkeypatch.setattr(
        common,
        "hf_configure_http_backend",
        lambda backend_factory: configured_modes.append(backend_factory().trust_env),
    )

    def _operation() -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise requests.exceptions.ProxyError("proxy failed")
        return "ok"

    payload = common.run_hf_request(_operation)

    assert payload == "ok"
    assert call_count == 2
    assert configured_modes == [True, False]


def test_run_hf_request_reuses_direct_mode_after_proxy_failure(monkeypatch) -> None:
    configured_modes: list[bool] = []

    monkeypatch.setattr(common, "_HF_HTTP_BACKEND_CONFIGURED", False)
    monkeypatch.setattr(common, "_HF_HTTP_TRUST_ENV", True)
    monkeypatch.setattr(
        common,
        "hf_configure_http_backend",
        lambda backend_factory: configured_modes.append(backend_factory().trust_env),
    )

    first_calls = 0

    def _first_operation() -> str:
        nonlocal first_calls
        first_calls += 1
        if first_calls == 1:
            raise requests.exceptions.ProxyError("proxy failed")
        return "ok"

    second_calls = 0

    def _second_operation() -> str:
        nonlocal second_calls
        second_calls += 1
        return "still-ok"

    assert common.run_hf_request(_first_operation) == "ok"
    assert common.run_hf_request(_second_operation) == "still-ok"
    assert second_calls == 1
    assert configured_modes == [True, False]

