"""覆盖 family manifest 驱动的产物索引读取。"""

from __future__ import annotations

import json
from pathlib import Path

from testsupport.filesystem import write_json

from research_experiments.core.contracts import FamilyArtifactSchema
from research_experiments.family_runtime.artifact_index import (
    load_metrics_payload,
    load_prediction_records,
    resolve_run_artifact_index,
)


def test_resolve_run_artifact_index_uses_family_registration_schema(tmp_path: Path) -> None:
    write_json(
        tmp_path / "manifest.json",
        {
            "family_name": "selective_comm",
            "prototype": "shared_stage_policy",
            "run_id": "demo",
            "artifact_schema": FamilyArtifactSchema(
                metrics_view_path="views/metrics.json",
                prediction_records_path="views/predictions.jsonl",
                turn_record_paths=("turns/stage_a_turns.jsonl",),
                diagnostic_paths=("diagnostics/policy_diagnostics.json",),
                export_paths=("exports/oracle_trigger_eval.json",),
            ).to_manifest_payload(),
        },
    )

    index = resolve_run_artifact_index(tmp_path)

    assert index.family_name == "selective_comm"
    assert index.metrics_view_path.as_posix().endswith("views/metrics.json")
    assert index.prediction_records_path.as_posix().endswith("views/predictions.jsonl")


def test_load_metrics_payload_and_prediction_records_follow_family_contract(tmp_path: Path) -> None:
    write_json(
        tmp_path / "manifest.json",
        {
            "family_name": "single_agent",
            "prototype": "independent_sampling",
            "run_id": "demo",
            "artifact_schema": FamilyArtifactSchema(
                metrics_view_path="views/metrics.json",
                prediction_records_path="views/predictions.jsonl",
            ).to_manifest_payload(),
        },
    )
    write_json(tmp_path / "views" / "metrics.json", {"summary": [{"dataset": "overall", "method_name": "cot_1"}]})
    (tmp_path / "views" / "predictions.jsonl").write_text(
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

