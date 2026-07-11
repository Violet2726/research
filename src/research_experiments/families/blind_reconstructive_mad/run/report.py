"""BRD-MAD 的科研报告渲染。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from research_experiments.core.io import write_json
from research_experiments.family_runtime.artifact_index import load_metrics_payload, resolve_run_artifact_index
from research_experiments.family_runtime.report_bundle import (
    render_family_report_bundle,
    render_family_scientific_report,
)
from research_experiments.reporting.report_views import SummaryTableView, load_json_payload
from research_experiments.reporting.reporting_utils import resolve_manifest_model_name
from research_experiments.reporting.run_figures import (
    build_efficiency_rank_figure_spec,
    build_frontier_figure_spec,
    build_grouped_bar_figure_spec,
    build_score_by_dataset_figure_spec,
)
from research_experiments.reporting.scientific_report import format_float, render_run_reproducibility_section
from research_experiments.workspace.layout import default_reports_root


def load_metrics(run_dir: str | Path, *, family_name: str = "blind_reconstructive_mad") -> dict[str, Any]:
    return load_metrics_payload(run_dir, family_name=family_name)


def summarize_run(run_dir: str | Path, *, family_name: str = "blind_reconstructive_mad") -> dict[str, Any]:
    summary = SummaryTableView.from_metrics_payload(load_metrics(run_dir, family_name=family_name))
    return {
        "run_dir": str(Path(run_dir)),
        "row_count": len(summary.rows),
        "datasets": sorted(summary.grouped_by_dataset()),
        "summary_by_dataset": {dataset: [row.raw for row in rows] for dataset, rows in summary.grouped_by_dataset().items()},
    }


def render_report(
    run_dir: str | Path,
    publish_dir: str | Path | None = None,
    *,
    family_name: str = "blind_reconstructive_mad",
    display_name: str = "BRD-MAD",
) -> dict[str, Any]:
    index = resolve_run_artifact_index(run_dir, family_name=family_name)
    root = index.run_dir
    manifest = load_json_payload(index.manifest_path)
    metrics = load_metrics(root, family_name=family_name)
    diagnostics = load_json_payload(root / "diagnostics" / "brd_diagnostics.json")
    paired = load_json_payload(root / "diagnostics" / "paired_statistics.json")
    protocol = load_json_payload(root / "diagnostics" / "output_protocol_diagnostics.json")
    gate_filename = "count100_gate.json" if family_name == "selective_gsa_mad" else "pilot_gate.json"
    promotion_gate = load_json_payload(root / "diagnostics" / gate_filename)
    comparison = _comparison(manifest, metrics, diagnostics, paired, family_name=family_name)
    comparison_path = root / "exports" / "brd_comparison.json"
    write_json(comparison_path, comparison)
    csv_path = root / "exports" / "paper_summary.csv"
    _write_csv(csv_path, metrics.get("summary", []))
    markdown = _markdown(
        manifest,
        metrics,
        diagnostics,
        paired,
        protocol,
        promotion_gate,
        comparison,
        root,
        display_name,
        family_name=family_name,
    )
    payload = render_family_report_bundle(
        family_name=family_name,
        run_dir=root,
        publish_dir=publish_dir or default_reports_root(family_name),
        manifest=manifest,
        base_markdown=markdown,
        figure_specs=_figures(metrics, diagnostics, display_name),
    )
    payload["brd_comparison"] = str(comparison_path)
    payload["paper_summary"] = str(csv_path)
    return payload


def _comparison(manifest, metrics, diagnostics, paired, *, family_name: str) -> dict[str, Any]:
    order = list(manifest.get("method_order") or [])
    summary = list(metrics.get("summary") or [])
    reference_method = str(paired.get("reference_method") or "brd_quorum_3")
    return {
        "experiment": manifest.get("experiment"),
        "phase": manifest.get("phase"),
        "method_order": order,
        "dataset_order": list(manifest.get("dataset_order") or []),
        "overall_macro_rows": _ordered(summary, "overall", order),
        "overall_micro_rows": _ordered(summary, "overall_micro", order),
        "per_dataset_rows": {dataset: _ordered(summary, dataset, order) for dataset in manifest.get("dataset_order") or []},
        "brd_diagnostics": diagnostics.get("summary_rows", []),
        "paired_statistics": paired,
        "sota_claim_rule": (
            f"Use fixed-backbone, fixed-reasoning-budget SOTA only if {reference_method} beats "
            "conditional_resample_3 with Holm-corrected "
            f"positive 95% CIs on both primary datasets, no backbone point estimate is negative, and {reference_method} has the best accuracy "
            "without more mean tokens than the strongest competitor."
        ),
        "family_name": family_name,
    }


def _markdown(
    manifest,
    metrics,
    diagnostics,
    paired,
    protocol,
    promotion_gate,
    comparison,
    run_dir: Path,
    display_name: str,
    *,
    family_name: str,
) -> str:
    overall = comparison["overall_macro_rows"]
    brd = [row for row in diagnostics.get("summary_rows", []) if row.get("dataset") == "overall"]
    protocol_rows = [row for row in protocol.get("rows", []) if row.get("dataset") == "overall"]
    bbeh_rows = comparison["per_dataset_rows"].get("bbeh", [])
    stats = paired.get("tests", [])
    reference_method = str(paired.get("reference_method") or "brd_quorum_3")
    is_sgsa = family_name == "selective_gsa_mad"
    bbeh_primary = str((metrics.get("bbeh_metric") or {}).get("primary") or "micro_accuracy")
    mechanism_bullets = (
        [
            "Stage A is exactly five free-CoT calls aligned to sc_5 prompts and seeds. Candidate families are normalized final answers.",
            "On disagreement, each reviewer receives the same complete candidate set with a fair per-candidate rationale budget, independently permuted anonymous labels, and hidden support counts.",
            "One shared generative-synthesis panel supplies both the 2/3 GSA counterfactual and SGSA's 3/3 existing-candidate promotion rule. New answers remain shadow-only.",
            "Reviewer agreement is not assumed independent; the report estimates error correlation and effective panel size.",
        ]
        if is_sgsa
        else [
            "Stage A is exactly five free-CoT calls aligned to sc_5 prompts and seeds. Candidate families are normalized final answers.",
            "On disagreement, each reviewer receives the same complete candidate set with a fair per-candidate rationale budget, independently permuted anonymous labels, and hidden support counts.",
            "Only existing Stage-A candidates can be promoted. A 4-1 split requires 3/3 minority support; other disagreement patterns require 2/3. New answers are shadow-only.",
            "The IID 2/3 quorum expression is a reference calculation, not a guarantee; reviewer correlation and effective panel size are reported.",
        ]
    )
    sections = [
        {
            "title": "Pre-registered mechanism",
            "bullets": mechanism_bullets,
        },
        {
            "title": "Overall results (macro accuracy)",
            "table": {
                "headers": ["method", "accuracy", "vs best no-comm", "initial vote", "debate gain", "tokens", "calls"],
                "rows": [[
                    f"`{row.get('method_name')}`",
                    format_float(row.get("accuracy_mean")),
                    _signed(row.get("accuracy_delta_vs_best_no_comm")),
                    format_float(row.get("initial_vote_accuracy_mean")),
                    _signed(row.get("debate_gain_over_initial_vote")),
                    format_float(row.get("total_tokens_mean"), 2),
                    format_float(row.get("calls_per_question_mean"), 2),
                ] for row in overall],
            },
        },
        {
            "title": f"{display_name} safety and coverage diagnostics",
            "table": {
                "headers": ["method", "oracle gap", "overrides", "precision", "recall", "corrected", "harmed", "shadows", "review corr."],
                "rows": [[
                    f"`{row.get('method_name')}`",
                    _signed(row.get("candidate_oracle_gap_over_anchor")),
                    str(row.get("override_count")),
                    format_float(row.get("override_precision")),
                    format_float(row.get("override_recall_on_oracle_opportunities")),
                    str(row.get("corrected_count")),
                    str(row.get("harmed_count")),
                    str(row.get("shadow_novel_answer_count")),
                    format_float((row.get("reviewer_error_correlation") or {}).get("mean_error_correlation")),
                ] for row in brd],
            },
        },
        {
            "title": f"BBEH metrics (primary: {bbeh_primary})",
            "table": {
                "headers": ["method", "primary accuracy", "micro accuracy"],
                "rows": [[
                    f"`{row.get('method_name')}`",
                    format_float(row.get("accuracy_mean")),
                    format_float(row.get("micro_accuracy_mean", row.get("accuracy_mean"))),
                ] for row in bbeh_rows],
            },
        },
        {
            "title": "Paired inference",
            "table": {
                "headers": ["dataset", "model", "comparison", "delta", "95% bootstrap CI", "McNemar p", "Holm p"],
                "rows": [[
                    str(row.get("dataset")), str(row.get("model_name")),
                    f"{reference_method} − {row.get('comparison_method')}",
                    _signed(row.get("absolute_accuracy_delta")),
                    _ci(row.get("bootstrap_ci_95")),
                    format_float(row.get("mcnemar_exact_p")),
                    format_float(row.get("holm_adjusted_p_within_dataset")),
                ] for row in stats],
            },
        },
        {
            "title": "Output protocol reliability",
            "table": {
                "headers": ["method/stage", "protocol failure rate", "missing reasoning rate"],
                "rows": [[f"`{row.get('method_name')}`", format_float(row.get("protocol_failure_rate")), format_float(row.get("reason_missing_rate"))] for row in protocol_rows],
            },
        },
        {
            "title": "Promotion gate",
            "table": {
                "headers": ["condition", "passed"],
                "rows": [[name, str(value)] for name, value in (promotion_gate.get("conditions") or {}).items()],
            },
        },
        render_run_reproducibility_section(
            run_dir=run_dir,
            artifact_items=[
                "Core evidence: metrics.json, brd_diagnostics.json, paired_statistics.json, brd_comparison.json, paper_summary.csv.",
                "Candidate oracle, label permutations, reviewer selections, shadow answers, calls, tokens, and latency are retained at sample level.",
            ],
        ),
    ]
    return render_family_scientific_report(
        title=f"{display_name} research report",
        abstract=[
            (
                "SGSA-MAD tests whether blinded generative synthesis with unanimous promotion can safely recover correct Stage-A minorities under a fixed reasoning budget."
                if is_sgsa
                else "BRD-MAD tests whether blinded, independently reconstructive review can safely recover correct Stage-A minorities under a fixed reasoning budget."
            ),
            "This report does not make a SOTA claim automatically; the pre-registered claim rule is exported with the comparison artifact.",
        ],
        overview_items=[
            ("Experiment", str(manifest.get("experiment"))),
            ("Phase", str(manifest.get("phase"))),
            ("Backbone", resolve_manifest_model_name(manifest)),
            ("Output protocol", str(manifest.get("output_protocol"))),
            ("Run directory", run_dir.as_posix()),
        ],
        sections=sections,
    )


def _figures(metrics, diagnostics, display_name: str) -> list[dict[str, Any]]:
    summary = [row for row in metrics.get("summary", []) if row.get("dataset") != "overall_micro"]
    overall = [row for row in diagnostics.get("summary_rows", []) if row.get("dataset") == "overall"]
    bbeh_primary = str((metrics.get("bbeh_metric") or {}).get("primary") or "micro_accuracy")
    return [
        build_frontier_figure_spec(
            summary,
            title=f"{display_name}: accuracy/token frontier",
            caption=f"Fixed-backbone results; {display_name} token accounting includes shared Stage A plus conditional review calls.",
            score_field="accuracy_mean",
            primary_metric="accuracy",
            method_label_field="method_name",
        ),
        build_efficiency_rank_figure_spec(
            summary,
            title=f"{display_name}: efficiency rank",
            caption="Accuracy per thousand tokens under the same backbone and run phase.",
            efficiency_field="accuracy_per_1k_tokens",
            primary_metric="accuracy per 1k tokens",
            method_label_field="method_name",
        ),
        build_score_by_dataset_figure_spec(
            [row for row in summary if row.get("dataset") not in {"overall", "overall_micro"}],
            title=f"{display_name}: score by dataset",
            caption=f"BBEH primary score for this run is {bbeh_primary}.",
            score_field="accuracy_mean",
            primary_metric="primary accuracy",
            method_label_field="method_name",
        ),
        build_grouped_bar_figure_spec(
            figure_id="brd_safety_diagnostics",
            title=f"{display_name}: coverage and safety",
            caption="Positive net correction must be accompanied by safe existing-candidate coverage.",
            primary_metric="rate or count",
            data=[{
                "label": row["method_name"], "short_label": row["method_name"],
                "oracle_gap": row.get("candidate_oracle_gap_over_anchor", 0.0),
                "precision": row.get("override_precision", 0.0),
                "net_corrected": row.get("net_corrected", 0.0),
            } for row in overall],
            series=[("oracle_gap", "candidate-oracle gap"), ("precision", "override precision"), ("net_corrected", "net corrected")],
            x_label="value",
            source_kind="diagnostics.brd_diagnostics",
            dataset_scope="overall",
            note="A zero precision is reported when no override occurs; inspect override_count before interpreting it.",
        ),
    ]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["dataset", "aggregate_kind", "model_name", "method_name", "method_type", "question_count", "accuracy_mean", "accuracy_delta_vs_cot_1", "accuracy_delta_vs_best_no_comm", "initial_vote_accuracy_mean", "debate_gain_over_initial_vote", "trigger_rate", "corrected_rate", "harmed_rate", "total_tokens_mean", "calls_per_question_mean"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)


def _ordered(rows: list[dict[str, Any]], dataset: str, order: list[str]) -> list[dict[str, Any]]:
    rank = {method: index for index, method in enumerate(order)}
    return sorted((row for row in rows if row.get("dataset") == dataset), key=lambda row: rank.get(str(row.get("method_name")), 999))


def _signed(value: Any) -> str:
    return "" if value is None else f"{float(value):+.4f}"


def _ci(value: Any) -> str:
    if not isinstance(value, list) or len(value) != 2:
        return ""
    return f"[{float(value[0]):+.4f}, {float(value[1]):+.4f}]"
