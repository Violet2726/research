from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import json
from typing import Any

from experiment_core.reporting.analysis_reports import render_frontier_report, write_report
from experiment_core.reporting.reporting_utils import resolve_manifest_model_name
from experiment_core.reporting.run_figures import (
    append_figure_gallery_markdown,
    build_efficiency_rank_figure_spec,
    build_frontier_figure_spec,
    build_grouped_bar_figure_spec,
    build_scatter_figure_spec,
    build_score_by_dataset_figure_spec,
    write_figure_bundle,
)
from experiment_core.foundation.workspace import default_reports_root


METHOD_ORDER = [
    "mv_3",
    "always_communicate",
    "disagreement_triggered",
    "consensus_freeze",
    "cue_v1",
    "mv_6",
    "sc_6",
]


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    metrics = _load_json(Path(run_dir) / "policy_metrics.json")
    rows = metrics.get("summary", [])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(row)
    return {
        "run_dir": str(Path(run_dir)),
        "row_count": len(rows),
        "datasets": sorted(grouped),
        "summary_by_dataset": grouped,
    }


def render_cue_report(run_dir: str | Path, publish_dir: str | Path | None = None) -> dict[str, Any]:
    publish_dir = publish_dir or default_reports_root("cue")
    root = Path(run_dir)
    manifest = _load_json(root / "manifest.json")
    metrics = _load_json(root / "policy_metrics.json")
    diagnostics = _load_json(root / "policy_diagnostics.json")
    oracle = _load_json(root / "oracle_trigger_eval.json")
    figure_bundle = write_figure_bundle(root, _build_figure_specs(metrics, diagnostics))
    base_markdown = _render_markdown(manifest, metrics, diagnostics, oracle, root)
    markdown = append_figure_gallery_markdown(base_markdown, figure_bundle["figures"], run_dir=root)
    local_report_path = root / "report.md"
    local_report_path.write_text(markdown, encoding="utf-8")
    write_report(root / "frontier_report.md", render_frontier_report(metrics.get("summary", []), title="CUE Frontier"))
    publish_path = Path(publish_dir) / _published_report_name(manifest)
    publish_path.parent.mkdir(parents=True, exist_ok=True)
    publish_path.write_text(
        append_figure_gallery_markdown(base_markdown, figure_bundle["figures"], run_dir=root, published_path=publish_path),
        encoding="utf-8",
    )
    return {
        "run_dir": str(root),
        "local_report": str(local_report_path),
        "published_report": str(publish_path),
        "frontier_report": str(root / "frontier_report.md"),
        "figure_manifest": str(root / "figure_manifest.json"),
    }


def _build_figure_specs(metrics: dict[str, Any], diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    summary_rows = metrics.get("summary", [])
    overall_policy_rows = [
        row for row in diagnostics.get("policy_rows", [])
        if row.get("dataset") == "overall"
    ]
    return [
        build_frontier_figure_spec(
            summary_rows,
            title="CUE frontier",
            caption="Overall accuracy versus average total tokens across CUE policies and controls.",
            score_field="accuracy_mean",
            primary_metric="Accuracy",
        ),
        build_efficiency_rank_figure_spec(
            summary_rows,
            title="CUE efficiency ranking",
            caption="Overall efficiency ranking measured by accuracy per 1K tokens.",
            efficiency_field="acc_per_1k_tokens",
            primary_metric="Accuracy per 1K tokens",
        ),
        build_score_by_dataset_figure_spec(
            summary_rows,
            title="CUE score by dataset",
            caption="Per-dataset accuracy map across CUE policies and controls.",
            score_field="accuracy_mean",
            primary_metric="Accuracy",
        ),
        build_scatter_figure_spec(
            figure_id="policy_tradeoff",
            title="Policy tradeoff",
            caption="Overall trigger rate versus accuracy across CUE policies.",
            primary_metric="Accuracy",
            data=[
                {
                    "label": str(row.get("display_name") or row.get("policy_name") or "unknown"),
                    "short_label": str(row.get("display_name") or row.get("policy_name") or "unknown"),
                    "x": float(row.get("trigger_rate") or 0.0),
                    "y": float(row.get("accuracy_mean") or 0.0),
                    "value": float(row.get("accuracy_mean") or 0.0),
                }
                for row in overall_policy_rows
            ],
            x_label="Trigger rate",
            y_label="Accuracy",
            source_kind="policy_diagnostics",
            dataset_scope="overall",
            note="Policies on the upper-left achieve higher accuracy with fewer trigger events.",
        ),
        build_grouped_bar_figure_spec(
            figure_id="oracle_alignment",
            title="Oracle alignment",
            caption="Overall oracle precision and recall across CUE policies.",
            primary_metric="Rate",
            data=[
                {
                    "label": str(row.get("display_name") or row.get("policy_name") or "unknown"),
                    "short_label": str(row.get("display_name") or row.get("policy_name") or "unknown"),
                    "precision": float(row.get("precision") or 0.0),
                    "recall": float(row.get("recall") or 0.0),
                }
                for row in overall_policy_rows
            ],
            series=[("precision", "Precision"), ("recall", "Recall")],
            x_label="Rate",
            source_kind="policy_diagnostics",
            dataset_scope="overall",
            note="Precision and recall are computed against the communication-benefit oracle approximation.",
        ),
    ]


def _render_markdown(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    oracle: dict[str, Any],
    run_dir: Path,
) -> str:
    backbone = {"name": resolve_manifest_model_name(manifest)}
    metric_rows = metrics.get("summary", [])
    policy_rows = diagnostics.get("policy_rows", [])
    overall_main_rows = _ordered_rows([row for row in metric_rows if row.get("dataset") == "overall"])
    per_dataset_rows = {
        dataset: _ordered_rows([row for row in metric_rows if row.get("dataset") == dataset])
        for dataset in sorted({row["dataset"] for row in metric_rows if row.get("dataset") != "overall"})
    }
    recommendation = diagnostics.get("recommended_next_default_policy", {})
    lines = [
        "# CUE 实验报告",
        "",
        "## 1. 实验概览",
        "",
        f"- 实验名：`{manifest.get('experiment')}`",
        f"- Phase：`{manifest.get('phase')}`",
        f"- Backbone：`{backbone.get('name')}`",
        f"- Prompt Version：`{manifest.get('prompt_version')}`",
        f"- 运行目录：`{run_dir.as_posix()}`",
        "- 方法主线：独立求解 -> utility 估计 -> 定向通信 -> 可选局部审计。",
        "",
        "## 2. Overall 主结果",
        "",
        "| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in overall_main_rows:
        lines.append(
            f"| `{row['display_name']}` | {row['accuracy_mean']:.4f} | {row['communication_tokens_mean']:.2f} | "
            f"{row['total_tokens_mean']:.2f} | {row['calls_per_question_mean']:.2f} | {row['acc_per_1k_tokens']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 3. Trigger 诊断",
            "",
            "| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall | False Trigger Rate | Missed Beneficial Comm Rate |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in _ordered_policy_rows([row for row in policy_rows if row.get("dataset") == "overall"]):
        lines.append(
            f"| `{row['display_name']}` | {row['trigger_rate']:.4f} | {row['early_exit_rate']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['false_trigger_rate']:.4f} | {row['missed_beneficial_comm_rate']:.4f} |"
        )
    lines.extend(["", "## 4. 分数据集结果", ""])
    for dataset, rows in per_dataset_rows.items():
        lines.extend(
            [
                f"### {dataset}",
                "",
                "| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            lines.append(
                f"| `{row['display_name']}` | {row['accuracy_mean']:.4f} | {row['communication_tokens_mean']:.2f} | "
                f"{row['total_tokens_mean']:.2f} | {row['calls_per_question_mean']:.2f} | {row['acc_per_1k_tokens']:.6f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## 5. 默认策略建议",
            "",
            f"- 推荐策略：`{recommendation.get('selected_policy', 'cue_v1')}`",
            f"- 相对 `always_communicate` 的准确率下降：`{recommendation.get('accuracy_drop_vs_always', 0.0)}`",
            f"- 相对 `always_communicate` 的总 token 下降比例：`{recommendation.get('token_drop_ratio_vs_always', 0.0)}`",
            "",
            "## 6. 局限",
            "",
            "- 当前报告主要用于首轮机制验证。",
            "- helpful / harmful 通信仍以 `always_communicate` 相对 `mv_3` 的变化作为 oracle 近似。",
            "- 更大规模结论仍需更长周期实验进一步确认。",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _ordered_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: METHOD_ORDER.index(row["method_name"]) if row["method_name"] in METHOD_ORDER else 999)


def _ordered_policy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: METHOD_ORDER.index(row["policy_name"]) if row["policy_name"] in METHOD_ORDER else 999)


def _published_report_name(manifest: dict[str, Any]) -> str:
    created_at = manifest.get("created_at")
    try:
        created_date = datetime.fromisoformat(created_at).date().isoformat() if created_at else "unknown-date"
    except ValueError:
        created_date = "unknown-date"
    experiment = str(manifest.get("experiment", "cue")).replace("/", "-")
    phase = str(manifest.get("phase", "phase")).replace("/", "-")
    backbone_name = str(manifest.get("backbone", {}).get("name", "backbone")).replace("/", "-")
    return f"{created_date}-{experiment}-{phase}-{backbone_name}-cue-report.md"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

