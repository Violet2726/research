"""统一 MAD 创新主线的 EVF 科研报告。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from research_experiments.family_runtime.artifact_index import (
    load_metrics_payload,
    named_diagnostic_paths,
    resolve_run_artifact_index,
)
from research_experiments.family_runtime.report_bundle import (
    render_family_report_bundle,
    render_family_scientific_report,
)
from research_experiments.reporting.report_views import SummaryTableView, load_json_payload
from research_experiments.reporting.run_figures import (
    build_efficiency_rank_figure_spec,
    build_frontier_figure_spec,
    build_score_by_dataset_figure_spec,
)
from research_experiments.reporting.scientific_report import format_float, render_run_reproducibility_section
from research_experiments.workspace.layout import default_reports_root

FAMILY_NAME = "risk_controlled_trace_mad"


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    metrics = load_metrics_payload(run_dir, family_name=FAMILY_NAME)
    summary = SummaryTableView.from_metrics_payload(metrics)
    grouped = summary.grouped_by_dataset()
    return {
        "run_dir": str(Path(run_dir)),
        "row_count": len(summary.rows),
        "datasets": sorted(grouped),
        "summary_by_dataset": {key: [row.raw for row in values] for key, values in grouped.items()},
        "progression_gate": metrics.get("progression_gate"),
    }


def render_report(run_dir: str | Path, publish_dir: str | Path | None = None) -> dict[str, Any]:
    index = resolve_run_artifact_index(run_dir, family_name=FAMILY_NAME)
    root = index.run_dir
    manifest = load_json_payload(index.manifest_path)
    metrics = load_json_payload(index.metrics_view_path)
    diagnostics = load_json_payload(named_diagnostic_paths(root, family_name=FAMILY_NAME)["evf_diagnostics.json"])
    summary = SummaryTableView.from_metrics_payload(metrics)
    rows = [row.raw for row in summary.rows]
    overall = summary.overall_rows()
    best = summary.best_by("accuracy_mean", rows=overall)
    gate = dict(metrics.get("progression_gate") or {})
    markdown = render_family_scientific_report(
        title="MAD Innovation / EVF-MAD research report",
        abstract=[
            "EVF-MAD uses a fixed three-Qwen/two-MiMo solver roster and only overrides its heterogeneous majority with independently executable falsification evidence.",
            "Retired BRD, SGSA and RCTA versions remain historical evidence; this run does not revive their selection logic.",
            f"Current-run best method: `{best.method_name}` ({format_float(best.accuracy_mean)})"
            if best
            else "No complete result is available.",
            f"Progression gate: `{gate.get('passed')}`; failures={gate.get('failures', [])}.",
        ],
        overview_items=[
            ("Experiment", str(manifest.get("experiment_name"))),
            ("Version", str(manifest.get("active_version"))),
            ("Phase", str(manifest.get("phase_name"))),
            ("Model roster", "3×Qwen-Flash + 2×MiMo-v2.5"),
            ("Run directory", root.as_posix()),
        ],
        sections=[
            {
                "title": "Frozen mechanism",
                "bullets": [
                    "Five heterogeneous trajectories form the anchor; a 5–0 result exits immediately.",
                    "A blind selector can choose only an existing challenger. Qwen and MiMo audit the same pair under independent label permutations.",
                    "An override requires two passing challenger checks, one passing anchor falsification, auditor agreement, and no failed challenger evidence.",
                    "The DSL rejects arbitrary code, ungrounded literals, filesystem access and network access.",
                ],
            },
            {
                "title": "Overall results",
                "table": {
                    "headers": ["method", "accuracy", "tokens", "logical calls", "corrected", "harmed"],
                    "rows": [
                        [
                            f"`{row.method_name}`",
                            format_float(row.accuracy_mean),
                            format_float(row.total_tokens_mean, 2),
                            format_float(row.calls_per_question_mean, 2),
                            str(row.corrected_count or 0),
                            str(row.harmed_count or 0),
                        ]
                        for row in overall
                    ],
                },
            },
            {
                "title": "Evidence and replacement diagnostics",
                "bullets": [
                    f"`{method}`: {payload}" for method, payload in sorted((diagnostics.get("methods") or {}).items())
                ],
            },
            {
                "title": "Claim boundary",
                "bullets": [
                    "A positive result may only support a fixed Qwen-Flash + MiMo-v2.5 compound system under at most ten logical model calls.",
                    "The Minority Sentinel entry is a non-official rule reproduction, not author code.",
                    "No unconditional global-SOTA or fixed-single-backbone claim is permitted.",
                ],
            },
            render_run_reproducibility_section(
                run_dir=root,
                artifact_items=[
                    "Core evidence: predictions.jsonl, evf_diagnostics.json, paired_statistics.json, output_protocol_diagnostics.json, paper_summary.csv."
                ],
            ),
        ],
    )
    figures = [
        build_frontier_figure_spec(
            rows,
            title="EVF accuracy/token frontier",
            caption="Accuracy versus actual mean token use.",
            score_field="accuracy_mean",
            primary_metric="accuracy",
            method_label_field="method_name",
        ),
        build_efficiency_rank_figure_spec(
            rows,
            title="EVF efficiency rank",
            caption="Accuracy per thousand actual tokens.",
            efficiency_field="accuracy_per_1k_tokens",
            primary_metric="accuracy per 1k tokens",
            method_label_field="method_name",
        ),
        build_score_by_dataset_figure_spec(
            rows,
            title="EVF score by dataset",
            caption="Primary dataset score for every pre-registered method.",
            score_field="accuracy_mean",
            primary_metric="primary accuracy",
            method_label_field="method_name",
        ),
    ]
    return render_family_report_bundle(
        family_name=FAMILY_NAME,
        run_dir=root,
        publish_dir=publish_dir or default_reports_root(FAMILY_NAME),
        manifest=manifest,
        base_markdown=markdown,
        figure_specs=figures,
    )
