
"""????????????????"""

from __future__ import annotations

from research_experiments.core.config import load_model_catalog, parse_model_ref, resolve_model_ref


def test_parse_model_ref() -> None:
    assert parse_model_ref("deepseek/deepseek-v4-flash") == ("deepseek", "deepseek-v4-flash")

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

def test_resolve_deepseek_model_ref() -> None:
    resolved = resolve_model_ref("deepseek/deepseek-v4-flash")
    assert resolved.provider == "deepseek"
    assert resolved.model_id == "deepseek-v4-flash"
    assert resolved.reasoning_effort == "none"
    assert resolved.supports_response_format is True

def test_resolve_xiaomimimo_model_ref() -> None:
    resolved = resolve_model_ref("xiaomimimo/mimo-v2.5")
    assert resolved.provider == "xiaomimimo"
    assert resolved.model_id == "mimo-v2.5"
    assert resolved.reasoning_effort == "none"
    assert resolved.supports_response_format is True

