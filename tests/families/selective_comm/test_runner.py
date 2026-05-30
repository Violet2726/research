"""覆盖 `selective_comm` family 的基本摘要链路。"""

from __future__ import annotations

from pathlib import Path

from testsupport.filesystem import write_json, write_registered_family_manifest

from research_experiments.families.selective_comm.run.report import summarize_run


def test_summarize_run_reads_policy_metrics(tmp_path: Path) -> None:
    write_registered_family_manifest(tmp_path / "manifest.json", family_name="selective_comm")
    write_json(
        tmp_path / "views" / "metrics.json",
        {
            "summary": [
                {"dataset": "gsm8k", "method_name": "hybrid_trigger", "accuracy_mean": 0.8},
                {"dataset": "overall", "method_name": "hybrid_trigger", "accuracy_mean": 0.8},
            ]
        },
    )

    payload = summarize_run(tmp_path)

    assert payload["row_count"] == 2
    assert payload["datasets"] == ["gsm8k", "overall"]
