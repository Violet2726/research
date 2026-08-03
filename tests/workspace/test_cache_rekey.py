from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from research_experiments.core.execution.cache import build_request_cache_key
from research_experiments.workspace.cache_rekey import apply_cache_rekey, inspect_cache_rekey

LEGACY_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    cache_key TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    response_json TEXT NOT NULL,
    http_status INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    provider_request_id TEXT
)
"""


def _write_row(
    database: Path,
    *,
    key: str,
    created_at: str,
    payload: dict,
    response: dict,
) -> None:
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    try:
        connection.execute(LEGACY_SCHEMA)
        connection.execute(
            """
            INSERT INTO requests (
                cache_key, created_at, payload_json, response_json, http_status, latency_ms, provider_request_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                key,
                created_at,
                json.dumps(payload, sort_keys=True),
                json.dumps(response, sort_keys=True),
                int(response.get("http_status") or 0),
                1.0,
                "req",
            ),
        )
        connection.commit()
    finally:
        connection.close()


def _tagged_payload(cap: int) -> dict:
    return {
        "model": "mimo-v2.5",
        "messages": [
            {"role": "system", "content": "Output the labels in exactly this order: REASONING, FINAL_ANSWER."},
            {"role": "user", "content": "Question"},
        ],
        "temperature": 0.7,
        "max_completion_tokens": cap,
    }


def _valid_response(answer: str) -> dict:
    return {
        "http_status": 200,
        "finish_reason": "stop",
        "assistant_text": f"REASONING: because\nFINAL_ANSWER: {answer}",
    }


def test_cache_rekey_dry_run_drops_invalid_rows_and_uses_earliest_valid_collision(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    database = root / "namespaces" / "catch-dev-kernel-d4-v1" / "providers" / "xiaomimimo" / "mimo-v2-5" / "bbeh" / "requests.sqlite"
    _write_row(database, key="old-1", created_at="2026-01-01T00:00:00Z", payload=_tagged_payload(16_384), response=_valid_response("A"))
    _write_row(database, key="old-2", created_at="2026-01-02T00:00:00Z", payload=_tagged_payload(65_536), response=_valid_response("B"))
    _write_row(
        database,
        key="old-length",
        created_at="2026-01-03T00:00:00Z",
        payload={**_tagged_payload(65_536), "seed": 7},
        response={**_valid_response("C"), "finish_reason": "length"},
    )
    _write_row(
        database,
        key="old-json",
        created_at="2026-01-04T00:00:00Z",
        payload={
            **_tagged_payload(65_536),
            "messages": [{"role": "system", "content": "Return strict JSON only."}],
        },
        response=_valid_response("D"),
    )
    _write_row(
        database,
        key="old-compiler",
        created_at="2026-01-05T00:00:00Z",
        payload={
            **_tagged_payload(65_536),
            "messages": [
                {
                    "role": "system",
                    "content": "You are a candidate-blind source compiler. Compiler protocol: catch_d4_candidate_blind_compiler_v1",
                }
            ],
            "seed": 8,
        },
        response=_valid_response("E"),
    )

    report = inspect_cache_rekey(root)

    assert report["mode"] == "dry_run"
    assert report["input_row_count"] == 5
    assert report["output_row_count"] == 1
    assert report["collision_count"] == 1
    assert report["dropped_by_reason"] == {
        "finish_reason_length": 1,
        "retired_d4_protocol": 2,
    }


def test_cache_rekey_does_not_drop_current_d4_compiler_protocol(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    database = (
        root
        / "namespaces"
        / "catch-dev-kernel-d4-v1"
        / "providers"
        / "xiaomimimo"
        / "mimo-v2-5"
        / "bbeh"
        / "requests.sqlite"
    )
    payload = {
        **_tagged_payload(65_536),
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a candidate-blind source compiler. Compiler protocol: "
                    "catch_d4_candidate_blind_compiler_v2"
                ),
            }
        ],
    }
    _write_row(
        database,
        key="current-compiler",
        created_at="2026-08-01T00:00:00Z",
        payload=payload,
        response=_valid_response("A"),
    )

    report = inspect_cache_rekey(root)

    assert report["output_row_count"] == 1
    assert report["dropped_by_reason"] == {}


def test_cache_rekey_apply_replaces_root_and_preserves_earliest_valid_response(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    database = root / "namespaces" / "catch-dev-kernel-d4-v1" / "providers" / "xiaomimimo" / "mimo-v2-5" / "bbeh" / "requests.sqlite"
    first = _tagged_payload(16_384)
    second = _tagged_payload(65_536)
    _write_row(database, key="old-1", created_at="2026-01-01T00:00:00Z", payload=first, response=_valid_response("A"))
    _write_row(database, key="old-2", created_at="2026-01-02T00:00:00Z", payload=second, response=_valid_response("B"))

    report = apply_cache_rekey(root)

    assert report["mode"] == "applied"
    assert Path(report["backup_root"]).is_dir()
    assert root.is_dir()
    assert report["validation"]["passed"] is True
    target_database = root / "providers" / "xiaomimimo" / "mimo-v2-5" / "bbeh" / "requests.sqlite"
    assert not (root / "namespaces").exists()
    connection = sqlite3.connect(target_database)
    try:
        rows = connection.execute("SELECT cache_key, payload_json, response_json FROM requests").fetchall()
    finally:
        connection.close()
    assert len(rows) == 1
    key, payload_json, response_json = rows[0]
    assert key == build_request_cache_key(provider="xiaomimimo", request_model="mimo-v2.5", payload=second)
    assert "max_completion_tokens" not in json.loads(payload_json)
    assert "FINAL_ANSWER: A" in json.loads(response_json)["assistant_text"]


def test_cache_rekey_rolls_back_if_atomic_activation_fails(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "cache"
    database = root / "providers" / "xiaomimimo" / "mimo-v2-5" / "bbeh" / "requests.sqlite"
    _write_row(
        database,
        key="old-1",
        created_at="2026-01-01T00:00:00Z",
        payload=_tagged_payload(16_384),
        response=_valid_response("A"),
    )
    original_replace = Path.replace

    def fail_activation(self: Path, target: Path):
        if self.name.startswith("cache.global-rekeying-") and target.name == "cache":
            raise OSError("simulated activation failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_activation)
    try:
        apply_cache_rekey(root)
    except OSError as exc:
        assert "simulated activation failure" in str(exc)
    else:
        raise AssertionError("expected simulated activation failure")
    assert root.is_dir()
    assert database.is_file()
    assert any(path.name.startswith("cache.global-rekeying-") for path in tmp_path.iterdir())
