"""`budget_comm` 报告与摘要。

本模块负责把运行目录中的 JSON/JSONL 产物整理成面向研究阅读的中文 Markdown 报告，
重点突出预算校准、通信成本、方法对比和是否值得进入 full DALA 的门槛判断。
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import json
import math
import random
from typing import Any

from budget_comm.logic import METHOD_ORDER
from experiment_core.analysis_reports import render_frontier_report, write_report
from experiment_core.workspace import default_reports_root


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    """输出简短运行摘要。"""
    metrics = _load_json(Path(run_dir) / "metrics.json")
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


def render_report(
    run_dir: str | Path,
    publish_dir: str | Path | None = None,
) -> dict[str, Any]:
    """渲染并写出中文 Markdown 报告。"""
    publish_dir = publish_dir or default_reports_root("budget_comm")
    root = Path(run_dir)
    manifest = _load_json(root / "manifest.json")
    metrics = _load_json(root / "metrics.json")
    diagnostics = _load_json(root / "budget_diagnostics.json")
    predictions = _load_jsonl(root / "final_predictions.jsonl")
    markdown = _render_markdown(manifest, metrics, diagnostics, predictions, root)
    local_report_path = root / "report.md"
    local_report_path.write_text(markdown, encoding="utf-8")
    write_report(root / "frontier_report.md", render_frontier_report(metrics.get("summary", []), title="Budget Communication Frontier"))

    publish_path = Path(publish_dir) / _published_report_name(manifest)
    publish_path.parent.mkdir(parents=True, exist_ok=True)
    publish_path.write_text(markdown, encoding="utf-8")
    return {
        "run_dir": str(root),
        "local_report": str(local_report_path),
        "published_report": str(publish_path),
        "frontier_report": str(root / "frontier_report.md"),
    }


def _render_markdown(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    predictions: list[dict[str, Any]],
    run_dir: Path,
) -> str:
    """把 manifest、指标与诊断结果渲染成最终报告文本。"""
    backbone = manifest.get("backbone", {})
    track_name = manifest.get("context_view", {}).get("track_name", "unknown")
    calibration = diagnostics.get("calibration", {})
    full_gate = diagnostics.get("full_dala_gate", {})
    overall_rows = _ordered_rows([row for row in metrics.get("summary", []) if row.get("dataset") == "overall"])

    lines = [
        "# DALA-lite Smoke20 报告",
        "",
        "## 1. 实验概览",
        "",
        f"- 实验名：`{manifest.get('experiment')}`",
        f"- 轨道：`{track_name}`",
        f"- Phase：`{manifest.get('phase')}`",
        f"- Backbone：`{backbone.get('name')}`",
        f"- 运行目录：`{run_dir.as_posix()}`",
        "- 方法固定为：`mv_3`、`all_to_all_full`、`budget_random`、`budget_confidence`、`dala_lite`。",
        "",
        "## 2. 预算冻结",
        "",
    ]
    for dataset, row in calibration.items():
        lines.append(
            f"- `{dataset}`：校准样本数=`{row['sample_count']}`，"
            f"`p50(all_to_all_full_comm_tokens)`=`{row['p50_all_to_all_full_communication_tokens']}`，"
            f"`round_budget_tokens`=`{row['round_budget_tokens']}`。"
        )

    lines.extend(
        [
            "",
            "## 3. 主结果表",
            "",
            "| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in overall_rows:
        lines.append(
            f"| `{row['display_name']}` | {row['accuracy_mean']:.4f} | {row['communication_tokens_mean']:.2f} | "
            f"{row['total_tokens_mean']:.2f} | {row['calls_per_question_mean']:.2f} | {row['acc_per_1k_tokens']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## 4. 机制表",
            "",
            "| Method | Winner Set Size | Budget Utilization | Full Ratio | Summary Ratio | Keywords Ratio | Silence Ratio | Corrected | Harmed |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in overall_rows:
        utilization = "-" if row["method_name"] in {"mv_3", "all_to_all_full"} else f"{row['budget_utilization_mean']:.4f}"
        lines.append(
            f"| `{row['display_name']}` | {row['winner_set_size_mean']:.4f} | {utilization} | "
            f"{row['full_ratio_mean']:.4f} | {row['summary_ratio_mean']:.4f} | {row['keywords_ratio_mean']:.4f} | "
            f"{row['silence_ratio_mean']:.4f} | {row['corrected_count']} | {row['harmed_count']} |"
        )

    per_dataset = sorted(
        {
            row["dataset"]
            for row in metrics.get("summary", [])
            if row.get("dataset") != "overall"
        }
    )
    lines.extend(["", "## 5. 数据集分表", ""])
    for dataset in per_dataset:
        lines.extend(
            [
                f"### {dataset}",
                "",
                "| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in _ordered_rows([item for item in metrics.get("summary", []) if item.get("dataset") == dataset]):
            lines.append(
                f"| `{row['display_name']}` | {row['accuracy_mean']:.4f} | {row['communication_tokens_mean']:.2f} | "
                f"{row['total_tokens_mean']:.2f} | {row['acc_per_1k_tokens']:.6f} |"
            )
        lines.append("")

    lines.extend(["## 6. 失败案例", ""])
    failure_cases = _select_failure_cases(predictions)
    if not failure_cases:
        lines.append("- 当前 smoke20 下没有收集到稳定失败案例。")
        lines.append("")
    else:
        for index, case in enumerate(failure_cases, start=1):
            lines.extend(
                [
                    f"### Case {index}",
                    "",
                    f"- 数据集：`{case['dataset']}`",
                    f"- 样本：`{case['sample_id']}`",
                    f"- 问题预览：{case['question_preview']}",
                    f"- 金标：`{case['gold']}`",
                    f"- `all_to_all_full`：`{case['full_prediction']}` / score=`{case['full_score']}`",
                    f"- `dala_lite`：`{case['dala_prediction']}` / score=`{case['dala_score']}`",
                    f"- 说明：{case['reason']}",
                    "",
                ]
            )

    ci_text = _bootstrap_ci_text(predictions, "dala_lite", "all_to_all_full")
    lines.extend(
        [
            "## 7. 探索性区间",
            "",
            f"- `dala_lite` 相对 `all_to_all_full` 的 overall accuracy delta 95% bootstrap CI：{ci_text}",
            "",
            "## 8. Full DALA 进入门槛",
            "",
            f"- 是否满足进入条件：`{full_gate.get('ready_for_full_dala', False)}`",
            f"- 原因：`{full_gate.get('reason', 'unknown')}`",
        ]
    )
    for name, passed in sorted(full_gate.get("conditions", {}).items()):
        lines.append(f"- `{name}`：`{passed}`")
    if "accuracy_gap_vs_all_to_all_full" in full_gate:
        lines.append(f"- `accuracy_gap_vs_all_to_all_full`：`{full_gate['accuracy_gap_vs_all_to_all_full']}`")
    if "communication_ratio_vs_all_to_all_full" in full_gate:
        lines.append(f"- `communication_ratio_vs_all_to_all_full`：`{full_gate['communication_ratio_vs_all_to_all_full']}`")
    lines.extend(
        [
            "",
            "## 9. 局限",
            "",
            "- 当前只覆盖 smoke20，小样本结果仅用于机制验证与工程联调。",
            "- split-context 轨道与 same-context 轨道分开报告，不直接合并成统一 overall 结论。",
            "- 当前只实现 DALA-lite 的无训练近似，不包含 MAPPO/value network 学习版。",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _select_failure_cases(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """挑选少量具有解释价值的失败或优势样例。"""
    lookup = {
        (row["dataset"], row["sample_id"], row["method_name"]): row
        for row in predictions
    }
    cases: list[dict[str, Any]] = []
    sample_keys = sorted({(row["dataset"], row["sample_id"]) for row in predictions})
    for dataset, sample_id in sample_keys:
        full_row = lookup.get((dataset, sample_id, "all_to_all_full"))
        dala_row = lookup.get((dataset, sample_id, "dala_lite"))
        confidence_row = lookup.get((dataset, sample_id, "budget_confidence"))
        if full_row is None or dala_row is None:
            continue
        reason = None
        if float(dala_row["score"]) < float(full_row["score"]):
            reason = "dala_lite 在该题上弱于 all_to_all_full。"
        elif confidence_row is not None and float(dala_row["score"]) > float(confidence_row["score"]):
            reason = "dala_lite 在同预算下优于 budget_confidence。"
        if reason is None:
            continue
        cases.append(
            {
                "dataset": dataset,
                "sample_id": sample_id,
                "question_preview": dala_row["question_preview"],
                "gold": dala_row["gold"],
                "full_prediction": full_row["prediction"],
                "full_score": full_row["score"],
                "dala_prediction": dala_row["prediction"],
                "dala_score": dala_row["score"],
                "reason": reason,
            }
        )
        if len(cases) >= 5:
            break
    return cases


def _bootstrap_ci_text(predictions: list[dict[str, Any]], primary_method: str, reference_method: str) -> str:
    """生成两种方法 accuracy delta 的 bootstrap 置信区间文本。"""
    paired = _paired_rows(predictions, primary_method, reference_method)
    if not paired:
        return "未计算。"
    deltas = _bootstrap_accuracy_delta(paired, iterations=2000, seed=42)
    return f"[{_quantile(deltas, 0.025):.6f}, {_quantile(deltas, 0.975):.6f}]（探索性）"


def _paired_rows(predictions: list[dict[str, Any]], primary_method: str, reference_method: str) -> list[tuple[float, float]]:
    """按样本 ID 配对两种方法的分数。"""
    lookup = {
        (row["dataset"], row["sample_id"], row["method_name"]): row
        for row in predictions
    }
    paired: list[tuple[float, float]] = []
    sample_keys = sorted({(row["dataset"], row["sample_id"]) for row in predictions})
    for dataset, sample_id in sample_keys:
        primary = lookup.get((dataset, sample_id, primary_method))
        reference = lookup.get((dataset, sample_id, reference_method))
        if primary is None or reference is None:
            continue
        paired.append((float(primary["score"]), float(reference["score"])))
    return paired


def _bootstrap_accuracy_delta(
    paired_scores: list[tuple[float, float]],
    *,
    iterations: int,
    seed: int,
) -> list[float]:
    """对两种方法的准确率差做配对 bootstrap 采样。"""
    rng = random.Random(seed)
    rows = list(paired_scores)
    samples: list[float] = []
    for _ in range(iterations):
        picked = [rows[rng.randrange(len(rows))] for _ in range(len(rows))]
        primary_acc = sum(primary for primary, _ in picked) / len(picked)
        reference_acc = sum(reference for _, reference in picked) / len(picked)
        samples.append(round(primary_acc - reference_acc, 6))
    return samples


def _ordered_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按论文中约定的方法顺序排序。"""
    return sorted(rows, key=lambda row: METHOD_ORDER.index(row["method_name"]) if row["method_name"] in METHOD_ORDER else 999)


def _published_report_name(manifest: dict[str, Any]) -> str:
    """构造发布到 `reports/budget_comm` 的报告文件名。"""
    created_at = manifest.get("created_at")
    try:
        created_date = datetime.fromisoformat(created_at).date().isoformat() if created_at else "unknown-date"
    except ValueError:
        created_date = "unknown-date"
    experiment = str(manifest.get("experiment", "budget-comm")).replace("/", "-")
    phase = str(manifest.get("phase", "phase")).replace("/", "-")
    backbone_name = str(manifest.get("backbone", {}).get("name", "backbone")).replace("/", "-")
    return f"{created_date}-{experiment}-{phase}-{backbone_name}-report.md"


def _quantile(values: list[float], q: float) -> float:
    """计算线性插值分位数。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 6)


def _load_json(path: Path) -> dict[str, Any]:
    """读取 UTF-8 JSON 文件；不存在时返回空字典。"""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 UTF-8 JSONL 文件；不存在时返回空列表。"""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
