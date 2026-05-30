"""覆盖 `selective_comm` family 的基本摘要链路。"""

from __future__ import annotations

from pathlib import Path

from testsupport.filesystem import write_json

from research_experiments.families.selective_comm.run.report import summarize_run


def test_summarize_run_reads_policy_metrics(tmp_path: Path) -> None:
    write_json(
        tmp_path / "policy_metrics.json",
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
