"""由 manifest 驱动的当前与历史 MAD 科研报告。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from research_experiments.family_runtime.artifact_index import load_metrics_payload, resolve_run_artifact_index
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
    is_hsgsa = str(manifest.get("active_version")) == "v5_hsgsa"
    diagnostic_name = "hsgsa_diagnostics.json" if is_hsgsa else "evf_diagnostics.json"
    diagnostics = load_json_payload(root / "diagnostics" / diagnostic_name)
    summary = SummaryTableView.from_metrics_payload(metrics)
    rows = [row.raw for row in summary.rows]
    overall = summary.overall_rows()
    best = summary.best_by("accuracy_mean", rows=overall)
    gate = dict(metrics.get("progression_gate") or {})
    descriptions = dict(manifest.get("method_descriptions") or {})
    protocol = dict(manifest.get("protocol") or {})
    mechanism = (
        [
            f"One frozen model produces {protocol.get('stage_candidates')} Stage-A candidates with identical base prompts.",
            "All aggregation, triggers, and scores use the same conservative answer-class key.",
            f"Disagreement adds {protocol.get('resample_candidates')} independent resamples and {protocol.get('reviewer_count')} independently permuted support-blind reviews.",
            "Only three valid, unanimous picks of one existing non-anchor answer class can override SC5; generated reviewer answers are shadow data only.",
        ]
        if is_hsgsa
        else ["This is a historical EVF artifact. Its model roster and mechanisms are read from the preserved manifest."]
    )
    markdown = render_family_scientific_report(
        title="H-SGSA homogeneous support-blind adjudication report" if is_hsgsa else "Historical EVF-MAD report",
        abstract=[
            str(manifest.get("description") or ""),
            f"Current-run best method: `{best.method_name}` ({format_float(best.accuracy_mean)})"
            if best
            else "No complete result is available.",
            f"Pre-registered progression gate: `{gate.get('passed')}`; failures={gate.get('failures', [])}.",
        ],
        overview_items=[
            ("Experiment", str(manifest.get("experiment_name"))),
            ("Version", str(manifest.get("active_version"))),
            ("Phase", str(manifest.get("phase_name"))),
            ("Resolved models", str(manifest.get("resolved_models"))),
            ("Run directory", root.as_posix()),
        ],
        sections=[
            {"title": "Frozen mechanism", "bullets": mechanism},
            {
                "title": "Manifest-derived method definitions",
                "table": {
                    "headers": ["method", "definition"],
                    "rows": [[f"`{method}`", descriptions.get(method, "Recorded in historical manifest")]
                             for method in manifest.get("method_order", [])],
                },
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
                "title": "Selection diagnostics",
                "bullets": [f"`{method}`: {payload}" for method, payload in sorted((diagnostics.get("methods") or {}).items())],
            },
            {
                "title": "Claim boundary",
                "bullets": [
                    str(manifest.get("claim_scope") or "No claim scope recorded."),
                    "A failed confirmation gate forbids a SOTA or successful-subset claim.",
                    "No cross-model or global leaderboard SOTA claim is licensed by this experiment.",
                ],
            },
            render_run_reproducibility_section(
                run_dir=root,
                artifact_items=[
                    f"Core evidence: predictions.jsonl, {diagnostic_name}, paired_statistics.json, output_protocol_diagnostics.json, and paper_summary.csv."
                ],
            ),
        ],
    )
    prefix = "H-SGSA" if is_hsgsa else "EVF"
    figures = [
        build_frontier_figure_spec(
            rows,
            title=f"{prefix} accuracy/token frontier",
            caption="Accuracy versus actual mean token use.",
            score_field="accuracy_mean",
            primary_metric="accuracy",
            method_label_field="method_name",
        ),
        build_efficiency_rank_figure_spec(
            rows,
            title=f"{prefix} efficiency rank",
            caption="Accuracy per thousand actual tokens.",
            efficiency_field="accuracy_per_1k_tokens",
            primary_metric="accuracy per 1k tokens",
            method_label_field="method_name",
        ),
        build_score_by_dataset_figure_spec(
            rows,
            title=f"{prefix} score by dataset",
            caption="Pre-registered score for every method and dataset.",
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
