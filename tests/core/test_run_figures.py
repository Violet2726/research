from __future__ import annotations

import json
from pathlib import Path

from experiment_core.reporting.run_figures import (
    append_figure_gallery_markdown,
    build_scatter_figure_spec,
    validate_figure_contract,
    write_figure_bundle,
)


def test_write_figure_bundle_writes_manifest_svg_and_csv(tmp_path: Path) -> None:
    bundle = write_figure_bundle(
        tmp_path,
        [
            build_scatter_figure_spec(
                figure_id="frontier_overall",
                title="Frontier",
                caption="Accuracy versus cost.",
                primary_metric="Accuracy",
                data=[{"label": "cot_1", "short_label": "cot_1", "x": 100.0, "y": 0.8, "value": 0.8}],
                x_label="Tokens",
                y_label="Accuracy",
                source_kind="metrics.summary",
                dataset_scope="overall",
                note="test note",
            ),
            build_scatter_figure_spec(
                figure_id="efficiency_rank_overall",
                title="Efficiency",
                caption="Efficiency proxy.",
                primary_metric="Accuracy",
                data=[{"label": "cot_1", "short_label": "cot_1", "x": 1.0, "y": 0.8, "value": 0.8}],
                x_label="x",
                y_label="y",
                source_kind="metrics.summary",
                dataset_scope="overall",
                note="test note",
            ),
            build_scatter_figure_spec(
                figure_id="score_by_dataset",
                title="By dataset",
                caption="Dataset scores.",
                primary_metric="Accuracy",
                data=[{"label": "cot_1", "short_label": "cot_1", "x": 1.0, "y": 0.8, "value": 0.8}],
                x_label="x",
                y_label="y",
                source_kind="metrics.summary",
                dataset_scope="overall",
                note="test note",
            ),
        ],
    )

    manifest_path = Path(bundle["figure_manifest"])
    assert manifest_path.exists()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["figure_count"] == 3
    assert (tmp_path / "figures" / "frontier_overall.svg").exists()
    assert (tmp_path / "figures" / "frontier_overall.csv").exists()


def test_validate_figure_contract_checks_manifest_and_report_reference(tmp_path: Path) -> None:
    bundle = write_figure_bundle(
        tmp_path,
        [
            build_scatter_figure_spec(
                figure_id="frontier_overall",
                title="Frontier",
                caption="Accuracy versus cost.",
                primary_metric="Accuracy",
                data=[{"label": "cot_1", "short_label": "cot_1", "x": 100.0, "y": 0.8, "value": 0.8}],
                x_label="Tokens",
                y_label="Accuracy",
                source_kind="metrics.summary",
                dataset_scope="overall",
                note="test note",
            ),
            build_scatter_figure_spec(
                figure_id="efficiency_rank_overall",
                title="Efficiency",
                caption="Efficiency proxy.",
                primary_metric="Accuracy",
                data=[{"label": "cot_1", "short_label": "cot_1", "x": 1.0, "y": 0.8, "value": 0.8}],
                x_label="x",
                y_label="y",
                source_kind="metrics.summary",
                dataset_scope="overall",
                note="test note",
            ),
            build_scatter_figure_spec(
                figure_id="score_by_dataset",
                title="By dataset",
                caption="Dataset scores.",
                primary_metric="Accuracy",
                data=[{"label": "cot_1", "short_label": "cot_1", "x": 1.0, "y": 0.8, "value": 0.8}],
                x_label="x",
                y_label="y",
                source_kind="metrics.summary",
                dataset_scope="overall",
                note="test note",
            ),
        ],
    )
    report_text = append_figure_gallery_markdown("# Report\n", bundle["figures"], run_dir=tmp_path)
    (tmp_path / "report.md").write_text(report_text, encoding="utf-8")

    contract = validate_figure_contract(tmp_path)
    assert contract["passed"] is True
    assert contract["report_references_count"] >= 1
