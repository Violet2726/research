"""Render paper-facing reports and figure bundles for faithful matrix runs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import os
from statistics import mean
from typing import Any

from experiment_core.foundation.workspace import default_reports_root
from experiment_core.matrix.matrix_specs import (
    EVIDENCE_DIAGNOSTIC,
    EVIDENCE_HEADLINE,
    EVIDENCE_REFERENCE,
    EVIDENCE_SUPPORTING,
    TRACK_SAME_CONTEXT,
    TRACK_SPLIT_CONTEXT,
    get_experiment_matrix_spec,
)
from experiment_core.reporting.paper_statistics import render_paper_statistics
from experiment_core.reporting.report_views import (
    HelpfulHarmfulTableView,
    MatrixAnalysisRowView,
    MatrixAnalysisTableView,
    MatrixStateEntryView,
    StatisticComparisonTableView,
    SummaryRowView,
    load_json_payload,
    load_jsonl_rows,
)
from experiment_core.reporting.run_figures import (
    _render_points_csv,
    _render_svg,
    append_figure_gallery_markdown,
    build_grouped_bar_figure_spec,
    build_interval_figure_spec,
    build_scatter_figure_spec,
    write_figure_bundle,
)


PREDICTION_FILE_CANDIDATES = (
    "policy_predictions.jsonl",
    "final_predictions.jsonl",
    "predictions.jsonl",
)

FIGURE_LABEL_OVERRIDES = {
    "local_auditing_ablation": "SPARC audit",
    "content_ablation": "SPARC content",
    "cue_black_box_utility_main": "CUE",
    "dala_lite_same_context_main": "DALA same",
    "dala_lite_split_context_main": "DALA split",
    "free_mad_lite_mechanism_validation": "Free-MAD-lite",
    "hotpotqa_split_context_communication_necessity": "Hotpot split",
    "same_context_controlled_debate": "Vanilla MAD",
    "cross_provider_robustness": "Robustness",
    "sid_lite_mechanism_validation": "SID-lite",
    "end_to_end_main": "SPARC",
    "trigger_early_exit_main": "Trigger early-exit",
    "voc_trigger_main": "Trigger VOC",
}


def render_paper_package(
    state_path_or_root: str | Path,
    *,
    output_root: str | Path | None = None,
    published_path: str | Path | None = None,
    figures_root: str | Path | None = None,
) -> dict[str, str]:
    """Render a paper package from an existing matrix run."""
    state_path = _resolve_state_path(state_path_or_root)
    root = Path(output_root) if output_root is not None else state_path.parent
    root.mkdir(parents=True, exist_ok=True)

    analysis_path = root / "faithful_analysis.json"
    if not analysis_path.exists():
        from experiment_core.matrix.faithful_analysis import render_faithful_analysis

        render_faithful_analysis(state_path)

    statistics_path = root / "paper_statistics.json"
    if not statistics_path.exists():
        render_paper_statistics(state_path, output_root=root)

    state_payload = load_json_payload(state_path)
    analysis = load_json_payload(analysis_path)
    statistics = load_json_payload(statistics_path)
    package = build_paper_package_payload(state_payload, analysis, statistics)

    run_id = state_path.parent.name
    figure_dir = Path(figures_root) if figures_root is not None else root / "figures"
    figure_specs = _build_figure_specs(package)
    figure_bundle = (
        write_figure_bundle(root, figure_specs)
        if figures_root is None
        else _write_external_figure_bundle(root, figure_dir, figure_specs)
    )
    package["figure_paths"] = {
        row["figure_id"]: row["svg_path"]
        for row in figure_bundle["figures"]
    }

    package_json = root / "paper_package.json"
    package_md = root / "paper_package.md"
    published = (
        Path(published_path)
        if published_path is not None
        else Path(default_reports_root("faithful_matrix")) / f"{run_id}-paper_package.md"
    )

    markdown_body = render_paper_package_markdown(package)
    markdown = append_figure_gallery_markdown(markdown_body, figure_bundle["figures"], run_dir=root)
    published_markdown = append_figure_gallery_markdown(
        markdown_body,
        figure_bundle["figures"],
        run_dir=root,
        published_path=published,
    )

    package_json.write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")
    package_md.write_text(markdown, encoding="utf-8")
    published.parent.mkdir(parents=True, exist_ok=True)
    published.write_text(published_markdown, encoding="utf-8")
    return {
        "package_json": package_json.as_posix(),
        "package_markdown": package_md.as_posix(),
        "published_path": published.as_posix(),
        "figures_root": figure_dir.as_posix(),
        "figure_manifest": figure_bundle["figure_manifest"],
    }


def build_paper_package_payload(
    state_payload: dict[str, Any],
    analysis: dict[str, Any],
    statistics: dict[str, Any],
) -> dict[str, Any]:
    """Convert matrix artifacts into stable paper-facing sections."""
    rows = MatrixAnalysisTableView.from_analysis_payload(analysis)
    sections = {
        "same_context_main_table": [row.to_dict() for row in rows.by_tier(EVIDENCE_HEADLINE, track=TRACK_SAME_CONTEXT)],
        "split_context_main_table": [row.to_dict() for row in rows.by_tier(EVIDENCE_HEADLINE, track=TRACK_SPLIT_CONTEXT)],
        "supporting_evidence_table": [row.to_dict() for row in rows.by_tier(EVIDENCE_SUPPORTING)],
        "diagnostic_evidence_table": [row.to_dict() for row in rows.by_tier(EVIDENCE_DIAGNOSTIC)],
        "reference_table": [row.to_dict() for row in rows.by_tier(EVIDENCE_REFERENCE)],
    }
    state_entries = [
        entry
        for entry in (MatrixStateEntryView.from_row(item) for item in state_payload.get("semantic_entries", []))
        if entry.status == "completed" and entry.run_dir
    ]
    budget_references = _build_budget_matched_single_agent_references(state_entries, rows.overall_rows())
    helpful_harmful = _build_helpful_harmful_breakdown(state_entries)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase_name": state_payload.get("overrides", {}).get("phase_name"),
        "model_ref": state_payload.get("overrides", {}).get("model_ref"),
        "counts": state_payload.get("counts", {}),
        "sections": sections,
        "statistics": statistics,
        "budget_matched_single_agent_references": budget_references,
        "helpful_harmful_communication": helpful_harmful,
    }


def render_paper_package_markdown(package: dict[str, Any]) -> str:
    """Render the package payload as a compact paper-oriented Markdown report."""
    lines = [
        "# 论文产物包",
        "",
        f"- generated_at: `{package.get('generated_at')}`",
        f"- phase_name: `{package.get('phase_name')}`",
        f"- model_ref: `{package.get('model_ref')}`",
        f"- counts: `{json.dumps(package.get('counts', {}), ensure_ascii=False)}`",
        "",
        "## 同上下文主结果表",
        "",
    ]
    lines.extend(_render_result_table(package["sections"]["same_context_main_table"]))
    lines.extend(["", "## 分视角主结果表", ""])
    lines.extend(_render_result_table(package["sections"]["split_context_main_table"]))
    lines.extend(["", "## 支撑证据", ""])
    lines.extend(_render_result_table(package["sections"]["supporting_evidence_table"]))
    lines.extend(["", "## 诊断证据", ""])
    lines.extend(_render_result_table(package["sections"]["diagnostic_evidence_table"]))
    lines.extend(["", "## 等预算单智能体参照", ""])
    lines.extend(_render_budget_baseline_table(package.get("budget_matched_single_agent_references", [])))
    lines.extend(["", "## 固定统计比较", ""])
    lines.extend(_render_statistics_table(package.get("statistics", {}).get("comparisons", []), package.get("statistics", {})))
    lines.extend(["", "## 有益 / 有害通信", ""])
    lines.extend(_render_helpful_table(package.get("helpful_harmful_communication", [])))

    figure_paths = package.get("figure_paths", {})
    if figure_paths:
        lines.extend(["", "## 图表索引", ""])
        for name, path in sorted(figure_paths.items()):
            lines.append(f"- `{name}`: `{path}`")

    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 主结果表中的 headline 行才用于正文主结论。",
            "- supporting 和 diagnostic 行用于解释机制、限制和消融，不直接替代主结论。",
            "- 等预算单智能体行是评测控制组，不代表新增方法步骤。",
            "- 如果 confirmatory 结果削弱了 headline 结论，应下调结论强度，而不是事后改写方法图。",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_budget_matched_single_agent_references(
    entries: list[MatrixStateEntryView],
    overall_rows: list[MatrixAnalysisRowView],
) -> list[dict[str, Any]]:
    single_agent_rows = _collect_single_agent_summary_rows(entries)
    headline_rows = [row for row in overall_rows if row.evidence_tier == EVIDENCE_HEADLINE]
    baselines: list[dict[str, Any]] = []
    for row in headline_rows:
        target_tokens = _as_float(row.total_tokens_mean)
        target_calls = _as_float(row.calls_per_question_mean)
        for baseline_name, method_name in (
            ("budget_matched_long_cot", "cot_1"),
            ("budget_matched_sc", "sc_5"),
        ):
            reference = _nearest_single_agent_row(
                single_agent_rows,
                method_name=method_name,
                target_tokens=target_tokens,
            )
            baselines.append(
                _baseline_record(
                    source_row=row,
                    baseline_name=baseline_name,
                    reference=reference,
                    target_calls=target_calls,
                    target_tokens=target_tokens,
                )
            )
        if row.evaluation_track == TRACK_SPLIT_CONTEXT and row.text("full_context_reference"):
            raw = row.to_dict()
            baselines.append(
                {
                    "experiment_name": row.experiment_name,
                    "baseline_name": "budget_matched_full_context_single",
                    "matched_budget_source": raw.get("full_context_reference"),
                    "matched_calls_per_q": None,
                    "matched_total_tokens_mean": None,
                    "score": raw.get("full_context_score"),
                    "target_calls_per_q": target_calls,
                    "target_total_tokens_mean": target_tokens,
                    "budget_match_status": "same_experiment_full_context_reference",
                }
            )
    return baselines


def _baseline_record(
    *,
    source_row: MatrixAnalysisRowView,
    baseline_name: str,
    reference: SummaryRowView | None,
    target_calls: float,
    target_tokens: float,
) -> dict[str, Any]:
    if reference is None:
        return {
            "experiment_name": source_row.experiment_name,
            "baseline_name": baseline_name,
            "matched_budget_source": None,
            "matched_calls_per_q": None,
            "matched_total_tokens_mean": None,
            "score": None,
            "target_calls_per_q": target_calls,
            "target_total_tokens_mean": target_tokens,
            "budget_match_status": "missing_reference_run",
        }

    matched_tokens = _as_float(reference.total_tokens_mean)
    return {
        "experiment_name": source_row.experiment_name,
        "baseline_name": baseline_name,
        "matched_budget_source": (
            f"{reference.text('experiment_name')}/{reference.method_name}/{reference.dataset}"
        ),
        "matched_calls_per_q": _as_float(reference.calls_per_question_mean),
        "matched_total_tokens_mean": matched_tokens,
        "score": _as_float(reference.accuracy_mean),
        "target_calls_per_q": target_calls,
        "target_total_tokens_mean": target_tokens,
        "token_ratio_to_target": None if target_tokens <= 0 else round(matched_tokens / target_tokens, 6),
        "budget_match_status": "available_proxy_not_exact_budget",
    }


def _collect_single_agent_summary_rows(entries: list[MatrixStateEntryView]) -> list[SummaryRowView]:
    rows: list[SummaryRowView] = []
    for entry in entries:
        if entry.family != "single_agent":
            continue
        run_dir = Path(entry.run_dir)
        payload = load_json_payload(run_dir / "metrics.json")
        for row in payload.get("summary", []) if isinstance(payload, dict) else []:
            if isinstance(row, dict):
                enriched = dict(row)
                enriched["experiment_name"] = entry.experiment_name
                rows.append(SummaryRowView.from_row(enriched))
    return rows + _synthesize_single_agent_overall_rows(rows)


def _nearest_single_agent_row(
    rows: list[SummaryRowView],
    *,
    method_name: str,
    target_tokens: float,
) -> SummaryRowView | None:
    candidates = [
        row
        for row in rows
        if row.method_name == method_name and row.dataset == "overall"
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            _single_agent_reference_rank(row.text("experiment_name")),
            abs(_as_float(row.total_tokens_mean) - target_tokens),
        ),
    )


def _single_agent_reference_rank(experiment_name: str) -> int:
    if experiment_name == "same_context_core_benchmarks":
        return 0
    if experiment_name == "same_context_main_table":
        return 1
    return 2


def _synthesize_single_agent_overall_rows(rows: list[SummaryRowView]) -> list[SummaryRowView]:
    direct_keys = {
        (row.text("experiment_name"), row.method_name)
        for row in rows
        if row.dataset == "overall"
    }
    grouped: dict[tuple[Any, Any], list[SummaryRowView]] = {}
    for row in rows:
        if row.dataset == "overall":
            continue
        key = (row.text("experiment_name"), row.method_name)
        if key in direct_keys:
            continue
        grouped.setdefault(key, []).append(row)

    synthesized: list[SummaryRowView] = []
    for (experiment_name, method_name), items in grouped.items():
        if not items:
            continue
        synthesized.append(
            SummaryRowView.from_row({
                "experiment_name": experiment_name,
                "method_name": method_name,
                "dataset": "overall",
                "accuracy_mean": round(mean(_as_float(item.accuracy_mean) for item in items), 6),
                "total_tokens_mean": round(mean(_as_float(item.total_tokens_mean) for item in items), 6),
                "calls_per_question_mean": round(
                    mean(_as_float(item.calls_per_question_mean) for item in items),
                    6,
                ),
            })
        )
    return synthesized


def _build_helpful_harmful_breakdown(entries: list[MatrixStateEntryView]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        spec = get_experiment_matrix_spec(entry.config_path)
        if spec.evidence_tier != EVIDENCE_HEADLINE:
            continue
        prediction_rows = [
            row
            for row in _load_prediction_rows(Path(entry.run_dir))
            if row.get("method_name") == spec.primary_method_name
        ]
        if not prediction_rows:
            continue

        helpful = 0
        harmful = 0
        neutral = 0
        for row in prediction_rows:
            if bool(row.get("corrected_by_method")):
                helpful += 1
                continue
            if bool(row.get("harmed_by_method")):
                harmful += 1
                continue

            stage_a = _as_optional_float(row.get("stage_a_score"))
            final = _as_optional_float(row.get("score"))
            if stage_a is None or final is None:
                neutral += 1
            elif final > stage_a:
                helpful += 1
            elif final < stage_a:
                harmful += 1
            else:
                neutral += 1

        rows.append(
            {
                "experiment_name": entry.experiment_name,
                "method_name": spec.primary_method_name,
                "sample_method_rows": len(prediction_rows),
                "helpful": helpful,
                "harmful": harmful,
                "neutral": neutral,
                "helpful_rate": round(helpful / len(prediction_rows), 6),
                "harmful_rate": round(harmful / len(prediction_rows), 6),
            }
        )
    return rows


def _build_figure_specs(package: dict[str, Any]) -> list[dict[str, Any]]:
    sections = package.get("sections", {})
    stage_points = _stage_gap_points(sections)
    helpful_points = _helpful_points(package.get("helpful_harmful_communication", []))
    return [
        build_scatter_figure_spec(
            figure_id="budget_frontier_same_context",
            title="预算前沿：同上下文 headline 方法",
            caption="Faithful score 相对于 full communication token 比率的位置关系。",
            primary_metric="Faithful score",
            data=_frontier_points(sections.get("same_context_main_table", [])),
            x_label="相对 full communication 的 token 比率",
            y_label="Faithful score",
            source_kind="faithful_analysis",
            dataset_scope="same_context_overall",
            note="横轴越小表示成本越低；参考线 x=1 表示 full communication 基线。",
            reference_x=1.0,
        ),
        build_scatter_figure_spec(
            figure_id="budget_frontier_split_context",
            title="预算前沿：分视角 headline 方法",
            caption="Faithful score 相对于 full communication 或匹配参照 token 比率的位置关系。",
            primary_metric="Faithful score",
            data=_frontier_points(sections.get("split_context_main_table", [])),
            x_label="相对 full communication 的 token 比率",
            y_label="Faithful score",
            source_kind="faithful_analysis",
            dataset_scope="split_context_overall",
            note="横轴越小表示成本越低；参考线 x=1 表示 full communication 基线。",
            reference_x=1.0,
        ),
        build_scatter_figure_spec(
            figure_id="trigger_utility",
            title="Trigger 收益",
            caption="trigger 类方法相对于最优无通信基线的收益。",
            primary_metric="相对最优无通信基线的差值",
            data=_trigger_utility_points(sections.get("same_context_main_table", [])),
            x_label="相对 full communication 的 token 比率",
            y_label="相对最优无通信基线的差值",
            source_kind="faithful_analysis",
            dataset_scope="same_context_overall",
            note="纵轴为正表示相对于最强无通信控制组有提升。",
            reference_x=1.0,
            reference_y=0.0,
        ),
        build_interval_figure_spec(
            figure_id="stage_ceiling_gap",
            title="距 stage ceiling 的差距",
            caption="headline 与 supporting 运行中 faithful score 到 stage ceiling 的绝对差距。",
            primary_metric="Stage ceiling gap",
            data=[
                {
                    "label": str(point["short_label"]),
                    "short_label": str(point["short_label"]),
                    "value": float(point["value"]),
                    "low": 0.0,
                    "high": float(point["value"]),
                    "track": str(point.get("track") or ""),
                }
                for point in stage_points
            ],
            x_label="到 stage ceiling 的差距（越低越好）",
            source_kind="faithful_analysis",
            dataset_scope="matrix_overall",
            note="差距越小，说明当前方法距离该阶段可达到的上界越近。",
        ),
        build_grouped_bar_figure_spec(
            figure_id="helpful_harmful_comm",
            title="有益与有害通信对比",
            caption="各通信实验中有益通信率与有害通信率的并列比较。",
            primary_metric="通信样本上的比例",
            data=helpful_points,
            series=[("helpful_rate", "Helpful"), ("harmful_rate", "Harmful")],
            x_label="比例",
            source_kind="prediction_rows",
            dataset_scope="matrix_overall",
            note="有益与有害比例分开呈现，避免净值抵消带来的误读。",
        ),
    ]


def _write_external_figure_bundle(
    run_root: Path,
    figure_dir: Path,
    figure_specs: list[dict[str, Any]],
) -> dict[str, Any]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    relative_prefix = Path(os.path.relpath(figure_dir, run_root)).as_posix()
    for spec in figure_specs:
        figure_id = str(spec["figure_id"])
        normalized = {
            "figure_id": figure_id,
            "title": str(spec["title"]),
            "caption": str(spec["caption"]),
            "svg_path": f"{relative_prefix}/{figure_id}.svg",
            "csv_path": f"{relative_prefix}/{figure_id}.csv",
            "source_kind": str(spec.get("source_kind") or ""),
            "dataset_scope": str(spec.get("dataset_scope") or ""),
            "primary_metric": str(spec.get("primary_metric") or ""),
        }
        (figure_dir / f"{figure_id}.svg").write_text(_render_svg(spec), encoding="utf-8")
        (figure_dir / f"{figure_id}.csv").write_text(_render_points_csv(spec.get("data", [])), encoding="utf-8")
        rows.append(normalized)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "figure_count": len(rows),
        "figures": rows,
    }
    manifest_path = run_root / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "figures_dir": figure_dir.as_posix(),
        "figure_manifest": manifest_path.as_posix(),
        "figures": rows,
    }


def _frontier_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for row in MatrixAnalysisTableView.from_rows(rows).rows:
        score = _as_optional_float(row.faithful_score)
        if score is None:
            continue
        x = _as_optional_float(row.token_ratio_vs_full_comm)
        if x is None:
            x = _as_optional_float(row.token_ratio_vs_best_no_comm)
        x = 1.0 if x is None else x
        points.append(
            {
                "label": row.experiment_name,
                "short_label": _short_figure_label(row.experiment_name),
                "family": row.family,
                "track": row.evaluation_track,
                "method_name": row.primary_method_name,
                "x": x,
                "y": score,
                "value": score,
            }
        )
    return sorted(points, key=lambda point: (float(point["x"]), -float(point["y"]), str(point["label"])))


def _trigger_utility_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for row in MatrixAnalysisTableView.from_rows(rows).rows:
        experiment_name = row.experiment_name
        if "trigger" not in experiment_name:
            continue
        delta = _as_optional_float(row.delta_vs_best_no_comm)
        if delta is None:
            continue
        token_ratio = _as_optional_float(row.token_ratio_vs_full_comm)
        points.append(
            {
                "label": experiment_name,
                "short_label": _short_figure_label(experiment_name),
                "x": 1.0 if token_ratio is None else token_ratio,
                "y": delta,
                "value": delta,
                "direction": "improved" if delta >= 0 else "regressed",
            }
        )
    return sorted(points, key=lambda point: (float(point["x"]), -float(point["y"]), str(point["label"])))


def _stage_gap_points(sections: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in (
        "same_context_main_table",
        "split_context_main_table",
        "supporting_evidence_table",
        "diagnostic_evidence_table",
    ):
        rows.extend(sections.get(key, []))

    points: list[dict[str, Any]] = []
    for row in MatrixAnalysisTableView.from_rows(rows).rows:
        gap = _as_optional_float(row.stage_ceiling_gap)
        if gap is None:
            continue
        experiment_name = row.experiment_name
        track = row.evaluation_track or "other"
        points.append(
            {
                "label": experiment_name,
                "short_label": _short_figure_label(experiment_name),
                "track": track,
                "track_label": _track_display_label(track),
                "x": gap,
                "y": gap,
                "value": gap,
                "family": row.family,
            }
        )
    return sorted(points, key=lambda point: (float(point["value"]), str(point["track"]), str(point["label"])))


def _helpful_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for row in HelpfulHarmfulTableView.from_rows(rows).rows:
        helpful_rate = _as_optional_float(row.helpful_rate)
        harmful_rate = _as_optional_float(row.harmful_rate)
        experiment_name = row.experiment_name
        points.append(
            {
                "label": experiment_name,
                "short_label": _short_figure_label(experiment_name),
                "method_name": row.method_name,
                "sample_method_rows": row.sample_method_rows,
                "helpful_rate": 0.0 if helpful_rate is None else helpful_rate,
                "harmful_rate": 0.0 if harmful_rate is None else harmful_rate,
                "net_gain": (0.0 if helpful_rate is None else helpful_rate)
                - (0.0 if harmful_rate is None else harmful_rate),
                "value": 0.0 if helpful_rate is None else helpful_rate,
            }
        )
    return sorted(points, key=lambda point: (-float(point["net_gain"]), str(point["label"])))


def _render_result_table(rows: list[dict[str, Any]]) -> list[str]:
    headers = [
        "experiment_name",
        "primary_method_name",
        "faithful_score",
        "delta_vs_best_no_comm",
        "delta_vs_full_comm",
        "delta_vs_full_context",
        "token_ratio_vs_full_comm",
        "calls_per_question_mean",
    ]
    return _render_table(rows, headers)


def _render_budget_baseline_table(rows: list[dict[str, Any]]) -> list[str]:
    headers = [
        "experiment_name",
        "baseline_name",
        "matched_budget_source",
        "matched_calls_per_q",
        "matched_total_tokens_mean",
        "score",
        "budget_match_status",
    ]
    return _render_table(rows, headers)


def _render_statistics_table(
    comparisons: list[dict[str, Any]],
    statistics: dict[str, Any],
) -> list[str]:
    rows: list[dict[str, Any]] = []
    bootstrap = statistics.get("bootstrap_ci", {})
    paired = statistics.get("paired_win_loss", {})
    mcnemar = statistics.get("mcnemar_tests", {})
    for comparison in StatisticComparisonTableView.from_rows(comparisons).rows:
        comparison_id = comparison.comparison_id
        ci = bootstrap.get(comparison_id, {})
        pair = paired.get(comparison_id, {})
        test = mcnemar.get(comparison_id, {})
        rows.append(
            {
                "comparison_id": comparison_id,
                "status": comparison.status,
                "paired_n": comparison.paired_n,
                "mean_delta": comparison.mean_delta,
                "delta_ci95": _format_ci(ci.get("delta_ci95")),
                "wins": pair.get("wins"),
                "losses": pair.get("losses"),
                "ties": pair.get("ties"),
                "mcnemar_p": test.get("p_value_chi_square_cc"),
            }
        )
    return _render_table(
        rows,
        ["comparison_id", "status", "paired_n", "mean_delta", "delta_ci95", "wins", "losses", "ties", "mcnemar_p"],
    )


def _render_helpful_table(rows: list[dict[str, Any]]) -> list[str]:
    return _render_table(
        rows,
        ["experiment_name", "method_name", "sample_method_rows", "helpful", "harmful", "neutral", "helpful_rate", "harmful_rate"],
    )


def _render_table(rows: list[dict[str, Any]], headers: list[str]) -> list[str]:
    if not rows:
        return ["无可用数据。"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_display(row.get(header)) for header in headers) + " |")
    return lines


def _format_ci(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    return f"[{_display(value.get('low'))}, {_display(value.get('high'))}]"


def _display(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _load_prediction_rows(run_dir: Path) -> list[dict[str, Any]]:
    for filename in PREDICTION_FILE_CANDIDATES:
        path = run_dir / filename
        if path.exists():
            return load_jsonl_rows(path)
    return []


def _resolve_state_path(state_path_or_root: str | Path) -> Path:
    path = Path(state_path_or_root)
    if path.is_dir():
        candidate = path / "state.json"
        if candidate.exists():
            return candidate
    return path


def _safe_load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return load_json_payload(path)
    except Exception:
        return {}


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _short_figure_label(value: Any) -> str:
    label = str(value or "")
    if label in FIGURE_LABEL_OVERRIDES:
        return FIGURE_LABEL_OVERRIDES[label]

    cleaned = label
    for fragment in ("_v1", "_v2", "_smoke", "_main"):
        cleaned = cleaned.replace(fragment, "")
    cleaned = cleaned.replace("same_context", "same")
    cleaned = cleaned.replace("split_context", "split")
    cleaned = cleaned.replace("_", " ").strip()
    return cleaned[:24].strip() or label[:24]


def _track_display_label(track: str) -> str:
    if track == TRACK_SAME_CONTEXT:
        return "same"
    if track == TRACK_SPLIT_CONTEXT:
        return "split"
    return "other"
