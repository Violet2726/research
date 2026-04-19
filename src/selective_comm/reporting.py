"""选择性通信实验报告。"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import json
from typing import Any


METHOD_ORDER = [
    "mv_3",
    "always_communicate",
    "disagreement_triggered",
    "confidence_triggered",
    "hybrid_trigger",
    "mv_6",
    "sc_6",
]


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    """输出简短运行摘要。"""
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


def render_trigger_report(
    run_dir: str | Path,
    publish_dir: str | Path = "reports/selective_comm",
) -> dict[str, Any]:
    """渲染并写出中文 trigger 报告。"""
    root = Path(run_dir)
    manifest = _load_json(root / "manifest.json")
    metrics = _load_json(root / "policy_metrics.json")
    diagnostics = _load_json(root / "policy_diagnostics.json")
    oracle = _load_json(root / "oracle_trigger_eval.json")
    predictions = _load_jsonl(root / "policy_predictions.jsonl")

    markdown = _render_markdown(manifest, metrics, diagnostics, oracle, predictions, root)
    local_report_path = root / "trigger_report.md"
    local_report_path.write_text(markdown, encoding="utf-8")

    publish_path = Path(publish_dir) / _published_report_name(manifest)
    publish_path.parent.mkdir(parents=True, exist_ok=True)
    publish_path.write_text(markdown, encoding="utf-8")
    return {
        "run_dir": str(root),
        "local_report": str(local_report_path),
        "published_report": str(publish_path),
    }


def _render_markdown(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    oracle: dict[str, Any],
    predictions: list[dict[str, Any]],
    run_dir: Path,
) -> str:
    """渲染中文 Markdown 报告。"""
    backbone = manifest.get("backbone", {})
    metric_rows = metrics.get("summary", [])
    policy_rows = diagnostics.get("policy_rows", [])
    shared_prefix_rows = diagnostics.get("shared_prefix_rows", [])
    recommendation = diagnostics.get("recommended_next_default_policy", {})

    overall_main_rows = _ordered_rows([row for row in metric_rows if row.get("dataset") == "overall"])
    per_dataset_rows = {
        dataset: _ordered_rows([row for row in metric_rows if row.get("dataset") == dataset])
        for dataset in sorted(
            {
                row["dataset"]
                for row in metric_rows
                if row.get("dataset") not in {"overall"}
            }
        )
    }

    lines = [
        "# Trigger / Early-exit 实验报告",
        "",
        "## 1. 实验范围与公平性说明",
        "",
        f"- 实验名：`{manifest.get('experiment')}`",
        f"- Phase：`{manifest.get('phase')}`",
        f"- Backbone：`{backbone.get('name')}`",
        f"- Prompt Version：`{manifest.get('prompt_version')}`",
        f"- 运行目录：`{run_dir.as_posix()}`",
        "- 数据集：`GSM8K + StrategyQA + HotpotQA`，当前轮次只解释 `smoke20`。",
        "- 共享前缀设计：4 个 trigger 策略共享同一份 Stage A 与同一份 Stage B，不重复发 4 套网络请求。",
        "- 方法边界：本轮只回答“何时通信 / 何时 early exit”，不混入消息内容压缩和局部审计。",
        "",
        "## 2. 共享前缀设计与预算节省说明",
        "",
    ]

    for row in shared_prefix_rows:
        lines.append(
            f"- `{row['dataset']}`：共享实际 token=`{row['shared_actual_tokens']:.2f}`；"
            f"若按 4 套 trigger 独立重跑则为 `{row['naive_independent_tokens']:.2f}`；"
            f"共享前缀节省比例=`{row['shared_prefix_savings_ratio']:.4f}`"
        )
    lines.extend(
        [
            "",
            "## 3. 主结果表",
            "",
            "| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in overall_main_rows:
        lines.append(
            f"| `{row['display_name']}` | {row['accuracy_mean']:.4f} | {row['communication_tokens_mean']:.2f} | "
            f"{row['total_tokens_mean']:.2f} | {row['acc_per_1k_tokens']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## 4. Trigger 诊断表",
            "",
            "| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall | False Trigger Rate | Missed Beneficial Comm Rate | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in _ordered_policy_rows([row for row in policy_rows if row.get("dataset") == "overall"]):
        lines.append(
            f"| `{row['display_name']}` | {row['trigger_rate']:.4f} | {row['early_exit_rate']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['false_trigger_rate']:.4f} | "
            f"{row['missed_beneficial_comm_rate']:.4f} | {row['communication_tokens_mean']:.2f} | "
            f"{row['total_tokens_mean']:.2f} | {row['acc_per_1k_tokens']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## 5. 数据集分表",
            "",
        ]
    )
    for dataset, rows in per_dataset_rows.items():
        lines.extend(
            [
                f"### {dataset}",
                "",
                "| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in rows:
            lines.append(
                f"| `{row['display_name']}` | {row['accuracy_mean']:.4f} | {row['communication_tokens_mean']:.2f} | "
                f"{row['total_tokens_mean']:.2f} | {row['acc_per_1k_tokens']:.6f} |"
            )
        dataset_policy_rows = _ordered_policy_rows([row for row in policy_rows if row.get("dataset") == dataset])
        lines.extend(
            [
                "",
                "| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in dataset_policy_rows:
            lines.append(
                f"| `{row['display_name']}` | {row['trigger_rate']:.4f} | {row['early_exit_rate']:.4f} | "
                f"{row['precision']:.4f} | {row['recall']:.4f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 6. 失败案例",
            "",
        ]
    )
    failure_cases = _select_failure_cases(oracle.get("sample_rows", []), predictions)
    if not failure_cases:
        lines.append("- 当前 smoke20 下没有收集到可稳定复述的失败案例。")
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
                    f"- `mv_3`：`{case['mv_3_prediction']}` / score=`{case['mv_3_score']}`",
                    f"- `always`：`{case['always_prediction']}` / score=`{case['always_score']}`",
                    f"- 说明：{case['reason']}",
                    "",
                ]
            )

    lines.extend(
        [
            "## 7. 下一轮默认 trigger 建议",
            "",
            f"- 推荐策略：`{recommendation.get('selected_policy', 'disagreement_triggered')}`",
            f"- 相对 `always_communicate` 的准确率下降：`{recommendation.get('accuracy_drop_vs_always', 0.0)}`",
            f"- 相对 `always_communicate` 的总 token 下降比例：`{recommendation.get('token_drop_ratio_vs_always', 0.0)}`",
            f"- 规则是否通过：`{recommendation.get('rule_passed', False)}`",
            "",
            "## 8. 局限",
            "",
            "- 本轮只覆盖 `smoke20`，因此只报告描述性结果，不做统计显著性结论。",
            "- 当前只比较 trigger / early-exit，不比较消息内容压缩和局部审计。",
            "- `mv_3` 直接复用共享 Stage A，因此它是本实验内部的无通信基线，不代表独立运行时的物理网络成本。",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _select_failure_cases(
    oracle_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """挑选 3 至 5 个有代表性的失败案例。"""
    pred_lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in predictions:
        pred_lookup[(row["dataset"], row["sample_id"], row["method_name"])] = row

    cases: list[dict[str, Any]] = []
    for row in oracle_rows:
        dataset = row["dataset"]
        sample_id = row["sample_id"]
        mv_3_row = pred_lookup.get((dataset, sample_id, "mv_3"))
        always_row = pred_lookup.get((dataset, sample_id, "always_communicate"))
        disagreement_row = pred_lookup.get((dataset, sample_id, "disagreement_triggered"))
        hybrid_row = pred_lookup.get((dataset, sample_id, "hybrid_trigger"))
        if mv_3_row is None or always_row is None:
            continue
        reason = None
        if row["beneficial_communication"] and hybrid_row and not hybrid_row.get("triggered"):
            reason = "always_communicate 能纠错，但 hybrid_trigger 在该题 early exit，漏掉了有益通信。"
        elif not row["beneficial_communication"] and disagreement_row and disagreement_row.get("triggered"):
            reason = "通信本身没有带来收益，但 disagreement_triggered 仍然进入了通信。"
        elif float(always_row["score"]) < float(mv_3_row["score"]):
            reason = "always_communicate 比无通信的 `mv_3` 更差，说明该题存在通信伤害。"
        if reason is None:
            continue
        cases.append(
            {
                "dataset": dataset,
                "sample_id": sample_id,
                "question_preview": row["question_preview"],
                "gold": mv_3_row["gold"],
                "mv_3_prediction": mv_3_row["prediction"],
                "mv_3_score": mv_3_row["score"],
                "always_prediction": always_row["prediction"],
                "always_score": always_row["score"],
                "reason": reason,
            }
        )
    return cases[:5]


def _ordered_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按固定方法顺序排序。"""
    return sorted(rows, key=lambda row: METHOD_ORDER.index(row["method_name"]) if row["method_name"] in METHOD_ORDER else 999)


def _ordered_policy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按 trigger 策略顺序排序。"""
    return sorted(rows, key=lambda row: METHOD_ORDER.index(row["policy_name"]) if row["policy_name"] in METHOD_ORDER else 999)


def _published_report_name(manifest: dict[str, Any]) -> str:
    """构造发布到 ``reports/selective_comm`` 的文件名。"""
    created_at = manifest.get("created_at")
    try:
        created_date = datetime.fromisoformat(created_at).date().isoformat() if created_at else "unknown-date"
    except ValueError:
        created_date = "unknown-date"
    experiment = str(manifest.get("experiment", "selective-comm")).replace("/", "-")
    phase = str(manifest.get("phase", "phase")).replace("/", "-")
    backbone_name = str(manifest.get("backbone", {}).get("name", "backbone")).replace("/", "-")
    return f"{created_date}-{experiment}-{phase}-{backbone_name}-trigger-report.md"


def _load_json(path: Path) -> dict[str, Any]:
    """读取 UTF-8 JSON 文件。"""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 UTF-8 JSONL 文件。"""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
