"""RCTA 科研报告和标准图资产。"""

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


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    metrics = load_metrics_payload(run_dir, family_name="risk_controlled_trace_mad")
    summary = SummaryTableView.from_metrics_payload(metrics)
    grouped = summary.grouped_by_dataset()
    return {"run_dir": str(Path(run_dir)), "row_count": len(summary.rows), "datasets": sorted(grouped), "summary_by_dataset": {key: [row.raw for row in values] for key, values in grouped.items()}}


def render_report(run_dir: str | Path, publish_dir: str | Path | None = None) -> dict[str, Any]:
    index = resolve_run_artifact_index(run_dir, family_name="risk_controlled_trace_mad")
    root = index.run_dir
    manifest = load_json_payload(index.manifest_path)
    metrics = load_json_payload(index.metrics_view_path)
    diagnostics = load_json_payload(named_diagnostic_paths(root, family_name="risk_controlled_trace_mad")["rcta_diagnostics.json"])
    summary = SummaryTableView.from_metrics_payload(metrics)
    rows = [row.raw for row in summary.rows]
    overall = summary.overall_rows()
    best = summary.best_by("accuracy_mean", rows=overall)
    markdown = render_family_scientific_report(
        title="RCTA-MAD research report",
        abstract=[
            "RCTA-MAD tests whether one trace synthesizer plus a frozen global replacement-risk router can outperform self-consistency and debate under at most ten logical calls.",
            "This report never emits an unconditional global-SOTA claim; the pre-registered gate must pass on both datasets and backbones.",
            f"Current-run best method: `{best.method_name}` ({format_float(best.accuracy_mean)})" if best else "No complete method result is available.",
        ],
        overview_items=[("Experiment", str(manifest.get("experiment_name") or manifest.get("experiment"))), ("Phase", str(manifest.get("phase_name") or manifest.get("phase"))), ("Backbone", str((manifest.get("resolved_model") or {}).get("name") or "unknown")), ("Run directory", root.as_posix())],
        sections=[
            {"title": "Frozen mechanism", "bullets": [
                "The first five trajectories are exactly aligned to sc_5 prompts and seeds; trajectories six through nine exist only for stronger SC controls.",
                "On disagreement, one JSON trace synthesizer proposes an answer and a bounded executable certificate; unsupported certificates are never treated as truth.",
                "The router uses only the frozen dataset/model-independent feature whitelist and one global threshold.",
            ]},
            {"title": "Overall results", "table": {"headers": ["method", "accuracy", "tokens", "logical calls", "corrected", "harmed"], "rows": [[f"`{row.method_name}`", format_float(row.accuracy_mean), format_float(row.total_tokens_mean, 2), format_float(row.calls_per_question_mean, 2), str(row.corrected_count or 0), str(row.harmed_count or 0)] for row in overall]}},
            {"title": "Certificate and replacement diagnostics", "bullets": [f"`{method}`: {payload}" for method, payload in sorted((diagnostics.get("methods") or {}).items())]},
            render_run_reproducibility_section(run_dir=root, artifact_items=["Core evidence: predictions.jsonl, rcta_diagnostics.json, paired_statistics.json, output_protocol_diagnostics.json, paper_summary.csv."]),
        ],
    )
    figure_specs = [
        build_frontier_figure_spec(rows, title="RCTA accuracy/token frontier", caption="Accuracy versus actual mean token use.", score_field="accuracy_mean", primary_metric="accuracy", method_label_field="method_name"),
        build_efficiency_rank_figure_spec(rows, title="RCTA efficiency rank", caption="Accuracy per thousand actual tokens.", efficiency_field="accuracy_per_1k_tokens", primary_metric="accuracy per 1k tokens", method_label_field="method_name"),
        build_score_by_dataset_figure_spec(rows, title="RCTA score by dataset", caption="Primary dataset score for every pre-registered method.", score_field="accuracy_mean", primary_metric="primary accuracy", method_label_field="method_name"),
    ]
    return render_family_report_bundle(family_name="risk_controlled_trace_mad", run_dir=root, publish_dir=publish_dir or default_reports_root("risk_controlled_trace_mad"), manifest=manifest, base_markdown=markdown, figure_specs=figure_specs)

