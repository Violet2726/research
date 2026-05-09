"""单智能体实验的报告与图资产生成。"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import json
from typing import Any

from experiment_core.foundation.workspace import default_reports_root
from experiment_core.reporting.reporting_utils import resolve_manifest_model_name
from experiment_core.reporting.run_figures import (
    append_figure_gallery_markdown,
    build_frontier_figure_spec,
    build_efficiency_rank_figure_spec,
    build_grouped_bar_figure_spec,
    build_score_by_dataset_figure_spec,
    build_scatter_figure_spec,
    write_figure_bundle,
)


def load_metrics(run_dir: str | Path) -> dict[str, Any]:
    """读取单次运行目录下的 `metrics.json`。"""
    return json.loads((Path(run_dir) / "metrics.json").read_text(encoding="utf-8"))


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    """按数据集聚合单智能体运行摘要。"""
    metrics = load_metrics(run_dir)
    summary_rows = metrics.get("summary", [])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        grouped[row["dataset"]].append(row)
    return {
        "run_dir": str(Path(run_dir)),
        "dataset_count": len(grouped),
        "rows": len(summary_rows),
        "by_dataset": {dataset: rows for dataset, rows in grouped.items()},
    }


def render_report(run_dir: str | Path, publish_dir: str | Path | None = None) -> dict[str, Any]:
    """生成正式 `report.md` 与 run 级图资产。"""
    publish_dir = publish_dir or default_reports_root("single_agent")
    root = Path(run_dir)
    manifest = _load_json(root / "manifest.json")
    metrics = load_metrics(root)
    summary_rows = metrics.get("summary", [])
    figure_bundle = write_figure_bundle(root, _build_figure_specs(summary_rows))

    base_markdown = _render_markdown(manifest, summary_rows, root)
    local_report = append_figure_gallery_markdown(base_markdown, figure_bundle["figures"], run_dir=root)
    report_path = root / "report.md"
    report_path.write_text(local_report, encoding="utf-8")

    publish_path = Path(publish_dir) / _published_report_name(manifest)
    publish_path.parent.mkdir(parents=True, exist_ok=True)
    published_report = append_figure_gallery_markdown(
        base_markdown,
        figure_bundle["figures"],
        run_dir=root,
        published_path=publish_path,
    )
    publish_path.write_text(published_report, encoding="utf-8")
    return {
        "run_dir": str(root),
        "local_report": str(report_path),
        "published_report": str(publish_path),
        "figure_manifest": str(root / "figure_manifest.json"),
        "figures_dir": str(root / "figures"),
    }


def export_paper_tables(run_dir: str | Path, output_path: str | Path) -> Path:
    """导出论文表格风格的 Markdown 摘要。"""
    metrics = load_metrics(run_dir)
    rows = metrics.get("summary", [])
    lower_bound_rows = [row for row in rows if row["method_name"] == "cot_1"]
    self_consistency_rows = [row for row in rows if row["method_name"].startswith("sc_")]

    lines: list[str] = []
    lines.append("# 论文表格导出")
    lines.append("")
    lines.append("## Lower Bound 表")
    lines.append("")
    lines.extend(
        _markdown_table(
            lower_bound_rows,
            [
                "dataset",
                "model_name",
                "method_name",
                "questions_per_rerun",
                "accuracy_mean",
                "prompt_tokens_mean",
                "completion_tokens_mean",
                "total_tokens_mean",
                "calls_per_question_mean",
                "latency_ms_mean",
            ],
        )
    )
    lines.append("")
    lines.append("## Self-Consistency 无通信对照表")
    lines.append("")
    lines.extend(
        _markdown_table(
            self_consistency_rows,
            [
                "dataset",
                "model_name",
                "method_name",
                "questions_per_rerun",
                "accuracy_mean",
                "accuracy_std",
                "total_tokens_mean",
                "calls_per_question_mean",
                "acc_per_1k_tokens",
            ],
        )
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _build_figure_specs(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    overall_rows = [row for row in summary_rows if row.get("dataset") == "overall"]
    sc_rows = [
        row for row in overall_rows
        if str(row.get("method_name") or "").startswith("sc_")
    ]
    return [
        build_frontier_figure_spec(
            summary_rows,
            title="Single-agent frontier",
            caption="Accuracy versus average total tokens across the overall summary rows.",
            score_field="accuracy_mean",
            primary_metric="Accuracy",
            method_label_field="method_name",
        ),
        build_efficiency_rank_figure_spec(
            summary_rows,
            title="Single-agent efficiency ranking",
            caption="Overall efficiency ranking measured by accuracy per 1K tokens.",
            efficiency_field="acc_per_1k_tokens",
            primary_metric="Accuracy per 1K tokens",
            method_label_field="method_name",
        ),
        build_score_by_dataset_figure_spec(
            summary_rows,
            title="Single-agent score by dataset",
            caption="Per-dataset accuracy map across all configured methods.",
            score_field="accuracy_mean",
            primary_metric="Accuracy",
            method_label_field="method_name",
        ),
        build_scatter_figure_spec(
            figure_id="self_consistency_scaling",
            title="Self-consistency scaling",
            caption="Scaling behavior of self-consistency under increasing call budgets.",
            primary_metric="Accuracy",
            data=[
                {
                    "label": str(row["method_name"]),
                    "short_label": str(row["method_name"]),
                    "x": float(row.get("calls_per_question_mean") or 0.0),
                    "y": float(row.get("accuracy_mean") or 0.0),
                    "value": float(row.get("accuracy_mean") or 0.0),
                }
                for row in sorted(sc_rows, key=lambda item: float(item.get("calls_per_question_mean") or 0.0))
            ],
            x_label="Calls per question",
            y_label="Accuracy",
            source_kind="metrics.summary",
            dataset_scope="overall",
            note="Only self-consistency rows are shown in this scaling plot.",
        ),
        build_grouped_bar_figure_spec(
            figure_id="method_budget_profile",
            title="Method budget profile",
            caption="Prompt and completion token components for each overall method.",
            primary_metric="Average tokens per question",
            data=[
                {
                    "label": str(row["method_name"]),
                    "short_label": str(row["method_name"]),
                    "prompt_tokens_mean": float(row.get("prompt_tokens_mean") or 0.0),
                    "completion_tokens_mean": float(row.get("completion_tokens_mean") or 0.0),
                }
                for row in sorted(overall_rows, key=lambda item: float(item.get("total_tokens_mean") or 0.0), reverse=True)
            ],
            series=[
                ("prompt_tokens_mean", "Prompt tokens"),
                ("completion_tokens_mean", "Completion tokens"),
            ],
            x_label="Average tokens per question",
            source_kind="metrics.summary",
            dataset_scope="overall",
            note="Token components are shown separately to expose whether budget growth comes from prompts or completions.",
        ),
    ]


def _render_markdown(manifest: dict[str, Any], summary_rows: list[dict[str, Any]], run_dir: Path) -> str:
    backbone = resolve_manifest_model_name(manifest)
    overall_rows = [row for row in summary_rows if row.get("dataset") == "overall"]
    per_dataset = sorted({str(row.get("dataset")) for row in summary_rows if row.get("dataset") != "overall"})
    lines = [
        "# Single-Agent Report",
        "",
        "## 1. 实验概览",
        "",
        f"- 实验名：`{manifest.get('experiment')}`",
        f"- Phase：`{manifest.get('phase') or manifest.get('phase_name')}`",
        f"- Backbone：`{backbone}`",
        f"- Prompt Version：`{manifest.get('prompt_version')}`",
        f"- 运行目录：`{run_dir.as_posix()}`",
        "",
        "## 2. Overall 结果",
        "",
        "| Method | Accuracy | Total Tokens | Calls / Q | Acc / 1K Tokens |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(overall_rows, key=lambda item: float(item.get("accuracy_mean") or 0.0), reverse=True):
        lines.append(
            f"| `{row['method_name']}` | {float(row.get('accuracy_mean') or 0.0):.4f} | "
            f"{float(row.get('total_tokens_mean') or 0.0):.2f} | {float(row.get('calls_per_question_mean') or 0.0):.2f} | "
            f"{float(row.get('acc_per_1k_tokens') or 0.0):.6f} |"
        )

    lines.extend(["", "## 3. 分数据集结果", ""])
    for dataset in per_dataset:
        lines.extend(
            [
                f"### {dataset}",
                "",
                "| Method | Accuracy | Total Tokens | Calls / Q | Acc / 1K Tokens |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        dataset_rows = [row for row in summary_rows if row.get("dataset") == dataset]
        for row in sorted(dataset_rows, key=lambda item: float(item.get("accuracy_mean") or 0.0), reverse=True):
            lines.append(
                f"| `{row['method_name']}` | {float(row.get('accuracy_mean') or 0.0):.4f} | "
                f"{float(row.get('total_tokens_mean') or 0.0):.2f} | {float(row.get('calls_per_question_mean') or 0.0):.2f} | "
                f"{float(row.get('acc_per_1k_tokens') or 0.0):.6f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 4. 说明",
            "",
            "- 本报告面向 run 级复核，图表资产与 CSV 源数据一并固化在当前运行目录。",
            "- `paper_tables.md` 继续保留为论文表格附录；本文件聚焦整体结果和图形化诊断。",
            "",
        ]
    )
    return "\n".join(lines)


def _markdown_table(rows: list[dict[str, Any]], headers: list[str]) -> list[str]:
    """把字典行渲染为简洁 Markdown 表格。"""
    if not rows:
        return ["暂无数据。"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return lines


def _published_report_name(manifest: dict[str, Any]) -> str:
    created_at = manifest.get("created_at")
    try:
        created_date = (
            datetime.fromisoformat(created_at).date().isoformat()
            if created_at
            else "unknown-date"
        )
    except ValueError:
        created_date = "unknown-date"
    experiment = str(manifest.get("experiment", "single-agent")).replace("/", "-")
    phase = str(manifest.get("phase") or manifest.get("phase_name") or "phase").replace("/", "-")
    backbone = resolve_manifest_model_name(manifest).replace("/", "-")
    return f"{created_date}-{experiment}-{phase}-{backbone}-report.md"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
