"""覆盖 `multi_agent` family 的基本摘要链路。"""

from __future__ import annotations

from pathlib import Path

from testsupport.filesystem import write_json, write_registered_family_manifest

from research_experiments.families.multi_agent.run.report import summarize_run


def test_summarize_run_groups_rows_by_dataset(tmp_path: Path) -> None:
    write_registered_family_manifest(tmp_path / "manifest.json", family_name="multi_agent")
    write_json(
        tmp_path / "views" / "metrics.json",
        {
            "summary": [
                {"dataset": "gsm8k", "method_name": "mad_3a_r1", "accuracy_mean": 0.7},
                {"dataset": "overall", "method_name": "mad_3a_r1", "accuracy_mean": 0.7},
            ]
        },
    )

    payload = summarize_run(tmp_path)

    assert payload["row_count"] == 2
    assert payload["datasets"] == ["gsm8k", "overall"]
    assert "gsm8k" in payload["summary_by_dataset"]
