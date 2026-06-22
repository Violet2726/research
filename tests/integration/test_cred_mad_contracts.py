from __future__ import annotations

from pathlib import Path

from testsupport.filesystem import touch_figure_contract, write_json, write_jsonl, write_registered_family_manifest

from research_experiments.families.cred_mad.run.validate import validate_run


def test_validate_run_accepts_minimal_cred_contract(tmp_path: Path) -> None:
    write_registered_family_manifest(
        tmp_path / "manifest.json",
        family_name="cred_mad",
        payload={"method_order": ["cot_1", "cred_vote_5"], "cred_methods": ["cred_vote_5"]},
    )
    write_json(tmp_path / "progress.json", {"rate_limit_429_count": 0})
    write_json(
        tmp_path / "views" / "metrics.json",
        {
            "summary": [
                {"dataset": "overall", "method_name": "cot_1", "accuracy_mean": 0.8, "total_tokens_mean": 100.0},
                {"dataset": "overall", "method_name": "cred_vote_5", "accuracy_mean": 0.85, "total_tokens_mean": 500.0},
            ]
        },
    )
    write_json(tmp_path / "diagnostics" / "debate_diagnostics.json", {"sample_rows": [], "summary_rows": []})
    write_json(tmp_path / "diagnostics" / "router_eval.json", {"sample_rows": [], "summary_rows": []})
    write_json(tmp_path / "diagnostics" / "output_protocol_diagnostics.json", {"rows": []})
    write_json(tmp_path / "exports" / "cred_comparison.json", {"overall_macro_rows": []})
    (tmp_path / "exports").mkdir(exist_ok=True)
    (tmp_path / "exports" / "paper_summary.csv").write_text("dataset,method_name\n", encoding="utf-8")
    (tmp_path / "report.md").write_text("# report\n", encoding="utf-8")
    touch_figure_contract(tmp_path)
    write_jsonl(
        tmp_path / "turns" / "agent_turns.jsonl",
        [
            {
                "dataset": "gsm8k",
                "sample_id": "s1",
                "method_name": "cred_stage_a",
                "output_status": "ok",
                "request_status": "ok",
                "protocol_parse_status": "ok",
                "cache_hit": True,
                "request_started_at": "2026-06-05T00:00:00+00:00",
            }
        ],
    )
    write_jsonl(tmp_path / "turns" / "debate_messages.jsonl", [])
    write_jsonl(tmp_path / "turns" / "router_decisions.jsonl", [])
    write_jsonl(
        tmp_path / "views" / "predictions.jsonl",
        [
            {"dataset": "gsm8k", "sample_id": "s1", "method_name": "cot_1", "method_type": "control"},
            {"dataset": "gsm8k", "sample_id": "s1", "method_name": "cred_vote_5", "method_type": "mad"},
        ],
    )

    result = validate_run(tmp_path)

    assert result["passed"] is True
