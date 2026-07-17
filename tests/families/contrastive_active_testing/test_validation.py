from __future__ import annotations

import json
from pathlib import Path

from research_experiments.families.contrastive_active_testing.run.validate import validate_run


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, list):
        path.write_text("".join(json.dumps(row) + "\n" for row in payload), encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")


def test_validation_separates_scientific_usage_failure_from_artifact_failure(tmp_path: Path) -> None:
    _write(
        tmp_path / "manifest.json",
        {
            "family_name": "contrastive_active_testing",
            "cache_namespace": "catch-dev-v1",
            "request_source": "fresh_catch_confirmation_cache",
        },
    )
    turn = {
        "cache_namespace": "catch-dev-v1",
        "request_source": "catch_confirmation_cache",
        "payload": {"max_completion_tokens": 4096},
        "cache_key": "key",
        "raw_finish_reason": "stop",
        "network_attempt_count": 1,
        "request_error": None,
        "usage_source": "estimated",
        "actual_total_tokens": None,
        "reasoning_tokens": None,
    }
    prediction = {"logical_calls_per_question": 8}
    _write(tmp_path / "turns" / "agent_turns.jsonl", [turn])
    _write(tmp_path / "turns" / "router_decisions.jsonl", [{}])
    _write(tmp_path / "views" / "predictions.jsonl", [prediction])
    _write(tmp_path / "views" / "metrics.json", {"summary": []})
    _write(tmp_path / "views" / "run_summary.json", {})
    _write(tmp_path / "diagnostics" / "gate.json", {"passed": False})

    result = validate_run(tmp_path)
    assert result["artifact_violations"] == []
    assert result["scientific_violations"] == ["missing_actual_or_reasoning_tokens"]
    assert result["passed"] is False
    assert result["performance_gate_passed"] is False

