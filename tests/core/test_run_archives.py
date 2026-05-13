"""run 归档与恢复的共享测试。"""

from __future__ import annotations

from pathlib import Path
import json
import shutil

from research_experiments.workspace.run_archives import (
    _build_run_publish_commit_message,
    extract_run_archives,
    pack_run_artifacts,
    validate_archive_contract,
)


def test_pack_run_artifacts_creates_split_archives(tmp_path: Path) -> None:
    _seed_run_dir(tmp_path)

    payload = pack_run_artifacts(tmp_path, runs_root=tmp_path.parent)

    assert payload["archive_count"] == 2
    assert (tmp_path / "traces.tar.zst").exists()
    assert (tmp_path / "predictions.tar.zst").exists()
    assert validate_archive_contract(tmp_path, verify_sha256=True)["passed"] is True

    archive_manifest = json.loads((tmp_path / "archive_manifest.json").read_text(encoding="utf-8"))
    assert archive_manifest["remote_prefix"] == tmp_path.name
    assert "report.md" in archive_manifest["visible_files"]
    assert "raw_responses.jsonl" not in archive_manifest["visible_files"]

    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["archive_manifest_path"] == "archive_manifest.json"
    assert manifest["artifacts_packaged"] is True


def test_extract_run_archives_restores_removed_files(tmp_path: Path) -> None:
    _seed_run_dir(tmp_path)
    pack_run_artifacts(tmp_path, runs_root=tmp_path.parent)

    (tmp_path / "raw_responses.jsonl").unlink()
    (tmp_path / "final_predictions.jsonl").unlink()
    shutil.rmtree(tmp_path / "hotpot_predictions")

    restored = extract_run_archives(tmp_path)

    assert "raw_responses.jsonl" in restored
    assert "final_predictions.jsonl" in restored
    assert "hotpot_predictions/method_a.json" in restored
    assert (tmp_path / "raw_responses.jsonl").exists()
    assert (tmp_path / "final_predictions.jsonl").exists()
    assert (tmp_path / "hotpot_predictions" / "method_a.json").exists()


def test_build_run_publish_commit_message_is_human_readable() -> None:
    message = _build_run_publish_commit_message(
        "comm_necessary/hotpotqa_split_context_communication_necessity/count300/20260510T045655Z-xiaomimimo-mimo-v2.5"
    )
    assert (
        message
        == "发布 run [comm_necessary] hotpotqa_split_context_communication_necessity | count300 | 20260510T045655Z-xiaomimimo-mimo-v2.5"
    )

    matrix_message = _build_run_publish_commit_message("faithful_matrix/20260510T045449Z-count20-xiaomimimo-mimo-v2.5")
    assert matrix_message == "发布 run [faithful_matrix] 20260510T045449Z-count20-xiaomimimo-mimo-v2.5"


def _seed_run_dir(root: Path) -> None:
    (root / "manifest.json").write_text(json.dumps({"run_id": "test-run"}, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "metrics.json").write_text(json.dumps({"summary": []}, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "run_validation.json").write_text(json.dumps({"passed": True}, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "report.md").write_text("# report\n\n![Frontier](figures/frontier_overall.svg)\n", encoding="utf-8")
    (root / "diagnostics.json").write_text(json.dumps({"rows": []}, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "raw_responses.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "stage_a_turns.jsonl").write_text("{}\n", encoding="utf-8")
    (root / "final_predictions.jsonl").write_text("{}\n", encoding="utf-8")
    figures_dir = root / "figures"
    figures_dir.mkdir()
    for figure_id in ["frontier_overall", "efficiency_rank_overall", "score_by_dataset"]:
        (figures_dir / f"{figure_id}.svg").write_text("<svg/>\n", encoding="utf-8")
        (figures_dir / f"{figure_id}.csv").write_text("label,value\nx,1\n", encoding="utf-8")
    (root / "figure_manifest.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-01-01T00:00:00+00:00",
                "figure_count": 3,
                "figures": [
                    {
                        "figure_id": figure_id,
                        "title": figure_id,
                        "caption": "test figure",
                        "svg_path": f"figures/{figure_id}.svg",
                        "csv_path": f"figures/{figure_id}.csv",
                        "source_kind": "test",
                        "dataset_scope": "overall",
                        "primary_metric": "Accuracy",
                    }
                    for figure_id in ["frontier_overall", "efficiency_rank_overall", "score_by_dataset"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    predictions_dir = root / "hotpot_predictions"
    predictions_dir.mkdir()
    (predictions_dir / "method_a.json").write_text(json.dumps({"answer": {}}, ensure_ascii=False, indent=2), encoding="utf-8")

