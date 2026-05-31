from __future__ import annotations

from pathlib import Path

from testsupport.filesystem import (
    touch_figure_contract,
    write_json,
    write_jsonl,
    write_registered_family_manifest,
)

from research_experiments.families.single_agent.run.validate import validate_run as validate_single_agent
from research_experiments.family_runtime.validation import validate_rate_limit_check


def test_validate_rate_limit_check_detects_window_violation(tmp_path: Path) -> None:
    progress_path = tmp_path / "progress.json"
    write_json(progress_path, {"rate_limit_429_count": 0})

    turn_rows = [
        {
            "dataset": "gsm8k",
            "sample_id": "s1",
            "method_name": "cot_1",
            "agent_id": None,
            "cache_hit": False,
            "request_started_at": "2026-04-24T00:00:00+00:00",
            "estimated_request_tokens": 80,
        },
        {
            "dataset": "gsm8k",
            "sample_id": "s2",
            "method_name": "cot_1",
            "agent_id": None,
            "cache_hit": False,
            "request_started_at": "2026-04-24T00:00:01+00:00",
            "estimated_request_tokens": 80,
        },
    ]

    result = validate_rate_limit_check(
        progress_path,
        turn_rows,
        requests_per_minute_limit=1,
        tokens_per_minute_limit=100,
    )

    assert result["passed"] is False
    assert result["network_event_count"] == 2
    assert result["violation_count"] == 2


def test_single_agent_validator_fails_when_progress_records_429(tmp_path: Path) -> None:
    write_registered_family_manifest(
        tmp_path / "manifest.json",
        family_name="single_agent",
        payload={
            "requests_per_minute_limit": 95,
            "tokens_per_minute_limit": 9000000,
        },
    )
    write_jsonl(
        tmp_path / "turns" / "raw_responses.jsonl",
        [
            {
                "dataset": "gsm8k",
                "sample_id": "gsm8k-00001",
                "method_name": "cot_1",
                "rerun_index": 0,
                "output_status": "ok",
            }
        ],
    )
    write_jsonl(
        tmp_path / "views" / "predictions.jsonl",
        [
            {"dataset": "gsm8k", "method_name": "cot_1", "rerun_index": 0}
        ],
    )
    write_json(
        tmp_path / "views" / "metrics.json",
        {
            "summary": [
                {
                    "dataset": "gsm8k",
                    "model_name": "m",
                    "method_name": "cot_1",
                    "total_tokens_mean": 10.0,
                }
            ]
        },
    )
    write_json(tmp_path / "progress.json", {"rate_limit_429_count": 1})
    (tmp_path / "report.md").write_text("# report\n", encoding="utf-8")
    touch_figure_contract(tmp_path)

    payload = validate_single_agent(tmp_path)

    assert payload["passed"] is False
    assert payload["rate_limit_check"]["progress_429_count"] == 1
