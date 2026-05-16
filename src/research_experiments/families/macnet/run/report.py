"""MacNet 报告与图表产物。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from research_experiments.families.shared.report_common import render_family_report_bundle
from research_experiments.reporting.report_views import SummaryTableView, load_json_payload
from research_experiments.reporting.run_figures import (
    build_efficiency_rank_figure_spec,
    build_frontier_figure_spec,
    build_grouped_bar_figure_spec,
    build_score_by_dataset_figure_spec,
)
from research_experiments.workspace.layout import default_reports_root


def load_metrics(run_dir: str | Path) -> dict[str, Any]:
    return load_json_payload(Path(run_dir) / "metrics.json")


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    summary = SummaryTableView.from_metrics_payload(load_metrics(run_dir))
    return {
        "run_dir": str(Path(run_dir)),
        "row_count": len(summary.rows),
        "summary_rows": [row.raw for row in summary.rows],
    }


def render_report(run_dir: str | Path, publish_dir: str | Path | None = None) -> dict[str, Any]:
    publish_dir = publish_dir or default_reports_root("macnet")
    root = Path(run_dir)
    manifest = load_json_payload(root / "manifest.json")
    metrics = load_metrics(root)
    topology_manifest = load_json_payload(root / "topology_manifest.json")
    scaling_summary = load_json_payload(root / "scaling_summary.json")
    summary = SummaryTableView.from_metrics_payload(metrics)
    rows = [row.raw for row in summary.rows]
    lines = [
        "# MacNet 拓扑协作复现报告",
        "",
        "## 摘要",
        "",
        f"- 实验名：`{manifest['experiment_name']}`",
        f"- 实验类型：`{manifest['experiment_kind']}`",
        f"- 模型：`{manifest['resolved_model']['name']}`",
        f"- prompt version：`{manifest['prompt_version']}`",
        "",
        "## 研究口径",
        "",
        "- 这条线是平行论文复现支线，不进入当前 faithful matrix。",
        "- canonical 目标是复现 DAG 拓扑协作、方向差异与规模曲线，而不是把 MacNet 改写成普通多轮辩论。",
        "- `SRDD_Profile` 在 v1 里作为官方角色库资产接入，用于 actor/critic profile 选择。",
        "",
        "## 汇总表",
        "",
    ]
    lines.extend(
        _render_table(
            rows,
            [
                "dataset",
                "method_name",
                "node_scale",
                "topology_direction_mode",
                "quality_mean",
                "total_tokens_mean",
                "communication_tokens_mean",
                "calls_per_question_mean",
                "artifact_revision_count_mean",
                "inbound_instruction_count_mean",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## 拓扑清单",
            "",
            f"- 拓扑条目数：`{len(topology_manifest.get('topologies', []))}`",
            "",
            "## Scaling 摘要",
            "",
        ]
    )
    for series in scaling_summary.get("series", []):
        scale_parts = ", ".join(
            f"{item['node_scale']}=>{item['quality_mean']:.4f}"
            for item in series.get("scales", [])
        )
        lines.append(
            f"- `{series['method_name']}` / `{series['topology_direction_mode']}`：{scale_parts}"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 如果不同拓扑之间只体现 token 单调增加、没有任务分化，这条线不应包装成论文级强复现。",
            "- 如果 random / layer / mesh 出现更高质量但明显成本上升，应和效率图一起解读，而不是单看分数表。",
        ]
    )
    return render_family_report_bundle(
        family_name="macnet",
        run_dir=root,
        publish_dir=publish_dir,
        manifest=manifest,
        base_markdown="\n".join(lines) + "\n",
        figure_specs=_build_figure_specs(rows),
    )


def _build_figure_specs(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        build_frontier_figure_spec(
            summary_rows,
            title="MacNet 成本-性能前沿",
            caption="比较不同拓扑在各数据集上的质量与 token 成本位置。",
            score_field="quality_mean",
            primary_metric="quality",
            method_label_field="method_name",
        ),
        build_efficiency_rank_figure_spec(
            summary_rows,
            title="MacNet 效率排序",
            caption="按每千 token 质量排序，观察拓扑协作是否只是单纯堆成本。",
            efficiency_field="quality_per_1k_tokens",
            primary_metric="每千 token 质量",
            method_label_field="method_name",
        ),
        build_score_by_dataset_figure_spec(
            summary_rows,
            title="MacNet 跨数据集表现",
            caption="比较拓扑方法在不同任务上的质量分布。",
            score_field="quality_mean",
            primary_metric="quality",
            method_label_field="method_name",
        ),
        build_grouped_bar_figure_spec(
            figure_id="macnet_revision_pressure",
            title="MacNet 修订压力",
            caption="比较各方法的 artifact revision 与 inbound instruction 平均规模。",
            primary_metric="平均次数",
            data=[
                {
                    "label": f"{row.get('method_name')}@{row.get('node_scale')}",
                    "short_label": str(row.get("method_name")),
                    "artifact_revision_count_mean": float(row.get("artifact_revision_count_mean") or 0.0),
                    "inbound_instruction_count_mean": float(row.get("inbound_instruction_count_mean") or 0.0),
                }
                for row in summary_rows
                if row.get("dataset") == "mmlu"
            ],
            series=[
                ("artifact_revision_count_mean", "artifact_revision_count_mean"),
                ("inbound_instruction_count_mean", "inbound_instruction_count_mean"),
            ],
            x_label="平均次数",
            source_kind="metrics.summary",
            dataset_scope="mmlu",
            note="用来观察拓扑协作的有效修订是否真的来自结构差异，而不是泛化噪声。",
        ),
    ]


def _render_table(rows: list[dict[str, Any]], headers: list[str]) -> list[str]:
    if not rows:
        return ["No rows."]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values: list[str] = []
        for header in headers:
            value = row.get(header)
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            elif value is None:
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines
