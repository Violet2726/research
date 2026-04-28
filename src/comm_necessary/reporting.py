"""`comm_necessary` 报告与摘要。

报告层重点展示不同通信强度对 HotpotQA answer、supporting facts 与 joint 指标的影响，
并帮助判断“通信是否必要、必要到什么程度”。
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import json
from typing import Any

from comm_necessary.logic import METHOD_ORDER
from experiment_core.analysis_reports import render_split_context_report, write_report
from experiment_core.workspace import default_files_root, default_reports_root


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    """输出简短运行摘要。"""
    metrics = _load_json(Path(run_dir) / "metrics.json")
    rows = metrics.get("summary", [])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("dataset"))].append(row)
    return {
        "run_dir": str(Path(run_dir)),
        "row_count": len(rows),
        "datasets": sorted(grouped),
        "summary_by_dataset": grouped,
    }


def render_report(
    run_dir: str | Path,
    publish_dir: str | Path | None = None,
) -> dict[str, Any]:
    """渲染中文 Markdown 报告，并同步写入 files 汇总。"""
    publish_dir = publish_dir or default_reports_root("comm_necessary")
    root = Path(run_dir)
    manifest = _load_json(root / "manifest.json")
    metrics = _load_json(root / "metrics.json")
    diagnostics = _load_json(root / "diagnostics.json")
    predictions = _load_jsonl(root / "final_predictions.jsonl")
    markdown = _render_markdown(manifest, metrics, diagnostics, predictions, root)
    local_report = root / "comm_necessary_report.md"
    local_report.write_text(markdown, encoding="utf-8")
    write_report(root / "split_context_report.md", render_split_context_report(metrics.get("summary", []), title="Communication-Necessary Split-Context Report"))

    publish_path = Path(publish_dir) / _published_report_name(manifest)
    publish_path.parent.mkdir(parents=True, exist_ok=True)
    publish_path.write_text(markdown, encoding="utf-8")

    summary_path = Path(default_files_root()) / "HotpotQA通信必要性_smoke20结果汇总.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(markdown, encoding="utf-8")
    return {
        "run_dir": str(root),
        "local_report": str(local_report),
        "published_report": str(publish_path),
        "summary_report": str(summary_path),
        "split_context_report": str(root / "split_context_report.md"),
    }


def _render_markdown(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    predictions: list[dict[str, Any]],
    run_dir: Path,
) -> str:
    backbone = manifest.get("backbone", {})
    overall_rows = _ordered_rows([row for row in metrics.get("summary", []) if row.get("dataset") == "overall"])
    lines = [
        "# HotpotQA 通信必要性 Smoke20 报告",
        "",
        "## 1. 实验概览",
        "",
        f"- 实验名：`{manifest.get('experiment')}`",
        f"- Phase：`{manifest.get('phase')}`",
        f"- Backbone：`{backbone.get('name')}`",
        f"- 运行目录：`{run_dir.as_posix()}`",
        "- 任务：HotpotQA split-context evidence exchange；smoke20 只作工程验证和方向性证据。",
        "- 方法：`full_context_single`、`split_no_comm_mv3`、`answer_only_exchange`、`evidence_exchange`、`full_packet_exchange`。",
        "",
        "## 2. 主结果表",
        "",
        "| Method | Ans EM | Ans F1 | Sup F1 | Joint F1 | Title Recall | Comm Tokens | Total Tokens | Calls / Q |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in overall_rows:
        lines.append(
            f"| `{row['method_name']}` | {row['answer_em_mean']:.4f} | {row['answer_f1_mean']:.4f} | "
            f"{row['supporting_f1_mean']:.4f} | {row['joint_f1_mean']:.4f} | {row['support_title_recall_mean']:.4f} | "
            f"{row['communication_tokens_mean']:.2f} | {row['total_tokens_mean']:.2f} | {row['calls_per_question_mean']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## 3. 关键 Delta",
            "",
            "| Comparison | Ans EM Δ | Sup F1 Δ | Joint F1 Δ | Comm Tokens Δ |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in diagnostics.get("key_deltas", []):
        lines.append(
            f"| `{row['comparison']}` | {row['answer_em_delta']:.4f} | {row['supporting_f1_delta']:.4f} | "
            f"{row['joint_f1_delta']:.4f} | {row['communication_tokens_delta']:.2f} |"
        )

    lines.extend(
        [
            "",
            "## 4. 机制与校验摘要",
            "",
            f"- 题数：`{_sample_count(predictions)}`",
            f"- split 视图数：`{diagnostics.get('split_view_count', 0)}`；full-context 参考视图数：`{diagnostics.get('full_context_view_count', 0)}`",
            "- 每个方法均导出 HotpotQA 官方预测文件格式：`hotpot_predictions/{method}.json`，包含 `answer` 与 `sp`。",
            "- 报告口径：Answer 使用 EM/F1；Supporting Facts 使用 sentence-level EM/F1；Joint 使用 answer 与 support 的联合指标。",
            "",
            "## 5. 分数据集表",
            "",
        ]
    )
    for dataset in sorted({row["dataset"] for row in metrics.get("summary", []) if row.get("dataset") != "overall"}):
        lines.extend(
            [
                f"### {dataset}",
                "",
                "| Method | Ans EM | Sup F1 | Joint F1 | Comm Tokens | Total Tokens |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in _ordered_rows([item for item in metrics.get("summary", []) if item.get("dataset") == dataset]):
            lines.append(
                f"| `{row['method_name']}` | {row['answer_em_mean']:.4f} | {row['supporting_f1_mean']:.4f} | "
                f"{row['joint_f1_mean']:.4f} | {row['communication_tokens_mean']:.2f} | {row['total_tokens_mean']:.2f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 6. 资料来源",
            "",
            "- HotpotQA official：https://hotpotqa.github.io/",
            "- HotpotQA GitHub eval format：https://github.com/hotpotqa/hotpot",
            "- HuggingFace HotpotQA dataset card：https://huggingface.co/datasets/hotpotqa/hotpot_qa",
            "- HotpotQA paper：https://arxiv.org/abs/1809.09600",
            "- AgentsNet OpenReview：https://openreview.net/forum?id=gsSIH0mZ0Y",
            "",
            "## 7. 局限",
            "",
            "- 当前只运行 smoke20，不作为全量显著性结论。",
            "- 本轮优先验证 HotpotQA evidence exchange，AgentsNet 拓扑协调留作后续扩展。",
            "",
        ]
    )
    return "\n".join(lines)


def _ordered_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: METHOD_ORDER.index(row["method_name"]) if row.get("method_name") in METHOD_ORDER else 999)


def _sample_count(predictions: list[dict[str, Any]]) -> int:
    return len({(row.get("dataset"), row.get("sample_id")) for row in predictions})


def _published_report_name(manifest: dict[str, Any]) -> str:
    created_at = manifest.get("created_at")
    try:
        created_date = datetime.fromisoformat(created_at).date().isoformat() if created_at else "unknown-date"
    except ValueError:
        created_date = "unknown-date"
    experiment = str(manifest.get("experiment", "comm-necessary")).replace("/", "-")
    phase = str(manifest.get("phase", "phase")).replace("/", "-")
    backbone = str(manifest.get("backbone", {}).get("name", "backbone")).replace("/", "-")
    return f"{created_date}-{experiment}-{phase}-{backbone}-report.md"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

