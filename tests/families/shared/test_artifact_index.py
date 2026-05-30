"""覆盖 family manifest 驱动的产物索引读取。"""

from __future__ import annotations

import json
from pathlib import Path

from testsupport.filesystem import write_json

from research_experiments.families.artifacts import (
    load_metrics_payload,
    load_prediction_records,
    resolve_run_artifact_index,
)


def test_resolve_run_artifact_index_uses_family_registration_schema(tmp_path: Path) -> None:
    write_json(tmp_path / "manifest.json", {"family_name": "selective_comm", "run_id": "demo"})

    index = resolve_run_artifact_index(tmp_path)

    assert index.family_name == "selective_comm"
    assert index.metrics_view_path.name == "policy_metrics.json"
    assert index.prediction_records_path.name == "policy_predictions.jsonl"


def test_load_metrics_payload_and_prediction_records_follow_family_contract(tmp_path: Path) -> None:
    write_json(tmp_path / "manifest.json", {"family_name": "single_agent", "run_id": "demo"})
    write_json(tmp_path / "metrics.json", {"summary": [{"dataset": "overall", "method_name": "cot_1"}]})
    (tmp_path / "predictions.jsonl").write_text(
        json.dumps(
            {
                "dataset": "gsm8k",
                "sample_id": "1",
                "method_name": "cot_1",
                "score": 1.0,
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = load_metrics_payload(tmp_path)
    rows = load_prediction_records(tmp_path)

    assert metrics["summary"][0]["method_name"] == "cot_1"
    assert rows[0].sample_id == "1"
