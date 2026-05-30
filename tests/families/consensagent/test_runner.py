"""覆盖 `consensagent` family 的基本摘要链路。"""

from __future__ import annotations

from pathlib import Path

from testsupport.filesystem import write_json

from research_experiments.families.consensagent.run.report import summarize_run


def test_summarize_run_counts_methods_and_questions(tmp_path: Path) -> None:
    write_json(
        tmp_path / "metrics.json",
        {
            "summary": [
                {"method_name": "consensagent_3a", "prediction_rows": 5},
                {"method_name": "mad_3a_r1", "prediction_rows": 5},
            ]
        },
    )

    payload = summarize_run(tmp_path)

    assert payload["run_dir"] == str(tmp_path)
    assert payload["total_questions"] == 10
    assert payload["method_count"] == 2
