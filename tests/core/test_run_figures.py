from __future__ import annotations

import json
from pathlib import Path

from experiment_core.reporting.report_views import (
    DiagnosticTableView,
    MatrixAnalysisTableView,
    MatrixStateEntryView,
    StatisticComparisonTableView,
    SummaryRowView,
    SummaryTableView,
    load_json_payload,
    load_jsonl_rows,
)
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
    assert payload["figures"][0]["takeaway"]
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
    assert "要点：" in report_text


def test_summary_row_view_exposes_stable_label_and_numeric_access() -> None:
    row = SummaryRowView.from_row(
        {
            "dataset": "overall",
            "method_name": "sc_5",
            "display_name": "SC 5",
            "accuracy_mean": "0.81",
            "total_tokens_mean": 320,
        }
    )

    assert row.dataset == "overall"
    assert row.method_name == "sc_5"
    assert row.label() == "SC 5"
    assert row.short_label() == "SC 5"
    assert row.number("accuracy_mean") == 0.81
    assert row.number("total_tokens_mean") == 320.0


def test_summary_table_view_groups_and_selects_best_rows() -> None:
    table = SummaryTableView.from_rows(
        [
            {"dataset": "overall", "method_name": "cot_1", "accuracy_mean": 0.70, "acc_per_1k_tokens": 6.2},
            {"dataset": "overall", "method_name": "sc_5", "accuracy_mean": 0.78, "acc_per_1k_tokens": 2.7},
            {"dataset": "gsm8k", "method_name": "cot_1", "accuracy_mean": 0.69, "acc_per_1k_tokens": 6.0},
        ]
    )

    assert table.dataset_names() == ["gsm8k"]
    assert len(table.overall_rows()) == 2
    assert table.best_by("accuracy_mean", rows=table.overall_rows()).method_name == "sc_5"
    assert table.best_by("acc_per_1k_tokens", rows=table.overall_rows()).method_name == "cot_1"


def test_diagnostic_table_view_orders_and_selects_best_rows() -> None:
    table = DiagnosticTableView.from_rows(
        [
            {"dataset": "overall", "policy_name": "always_communicate", "accuracy_mean": 0.80, "trigger_rate": 1.0},
            {"dataset": "overall", "policy_name": "hybrid_trigger", "accuracy_mean": 0.82, "trigger_rate": 0.4},
        ]
    )

    assert [row.method_name for row in table.overall_rows()] == ["always_communicate", "hybrid_trigger"]
    assert table.best_by("accuracy_mean", rows=table.overall_rows()).method_name == "hybrid_trigger"


def test_shared_report_loaders_read_json_and_jsonl(tmp_path: Path) -> None:
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps({"summary": [{"dataset": "overall"}]}, ensure_ascii=False), encoding="utf-8")
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_text('{"row": 1}\n{"row": 2}\n', encoding="utf-8")

    assert load_json_payload(payload_path)["summary"][0]["dataset"] == "overall"
    assert load_jsonl_rows(rows_path) == [{"row": 1}, {"row": 2}]


def test_matrix_analysis_table_view_filters_by_tier_and_track() -> None:
    table = MatrixAnalysisTableView.from_rows(
        [
            {
                "family": "selective_comm",
                "experiment_name": "trigger_early_exit_main",
                "evaluation_track": "same_context",
                "evidence_tier": "headline",
                "dataset": "overall",
                "primary_method_name": "hybrid_trigger",
                "faithful_score": 0.84,
            },
            {
                "family": "cue",
                "experiment_name": "cue_black_box_utility_main",
                "evaluation_track": "same_context",
                "evidence_tier": "diagnostic",
                "dataset": "overall",
                "primary_method_name": "cue_v1",
                "faithful_score": 0.58,
            },
        ]
    )

    headline = table.by_tier("headline", track="same_context")
    assert len(headline) == 1
    assert headline[0].experiment_name == "trigger_early_exit_main"
    assert table.overall_rows()[1].evidence_tier == "diagnostic"


def test_matrix_state_entry_and_statistic_comparison_views() -> None:
    entry = MatrixStateEntryView.from_row(
        {
            "family": "selective_comm",
            "config_path": "configs/selective_comm/experiments/trigger_early_exit_main.toml",
            "experiment_name": "trigger_early_exit_main",
            "run_dir": "local/runs/selective_comm/demo",
            "status": "completed",
        }
    )
    stats = StatisticComparisonTableView.from_rows(
        [
            {"comparison_id": "a_vs_b", "status": "completed", "paired_n": 20, "mean_delta": 0.03},
            {"comparison_id": "c_vs_d", "status": "skipped", "paired_n": 0, "mean_delta": 0.0},
        ]
    )

    assert entry.family == "selective_comm"
    assert entry.status == "completed"
    assert stats.rows[0].comparison_id == "a_vs_b"
    assert stats.rows[0].paired_n == 20
