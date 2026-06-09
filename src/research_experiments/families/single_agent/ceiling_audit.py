"""Canonical simple-baseline recheck tool for single-agent baselines."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research_experiments.core.io import read_json, write_json, write_markdown
from research_experiments.family_runtime.comparators import canonical_standard_method_name

BASE_METHOD_ORDER = ("cot_1", "mv_3", "sc_5")
DEFAULT_REBASELINE_FOCUS_DATASETS = ("math500", "mmlu_pro", "hotpotqa")
CANONICAL_BASELINE_CONFIG = "configs/families/single_agent/experiments/canonical_simple_baselines.toml"


@dataclass(frozen=True)
class RunContext:
    run_dir: Path
    family_name: str
    experiment_name: str
    phase_name: str


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "rebaseline-conclusions":
        payload = rebaseline_core_conclusions(
            canonical_summary_json=Path(args.canonical_summary_json),
            run_dirs=[Path(item) for item in args.run_dir],
            output_dir=Path(args.output_dir),
            focus_datasets=tuple(args.focus_dataset or DEFAULT_REBASELINE_FOCUS_DATASETS),
        )
        print(payload["markdown_path"])
        return 0
    raise ValueError(f"Unsupported command: {args.command}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="用 canonical simple baseline 复核主方法结论。")
    subparsers = parser.add_subparsers(dest="command", required=True)
    rebaseline = subparsers.add_parser(
        "rebaseline-conclusions",
        help="对齐 canonical simple baseline 后重新判断主方法结论。",
    )
    rebaseline.add_argument(
        "--canonical-summary-json",
        required=True,
        help="canonical simple baseline 的 count100 summary JSON。",
    )
    rebaseline.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="需要复核的主方法 count100 run 目录，可重复传入。",
    )
    rebaseline.add_argument(
        "--focus-dataset",
        action="append",
        help="重点展示的数据集，可重复传入；默认展示 math500/mmlu_pro/hotpotqa。",
    )
    rebaseline.add_argument("--output-dir", required=True, help="输出目录。")
    return parser


def rebaseline_core_conclusions(
    *,
    canonical_summary_json: Path,
    run_dirs: list[Path],
    output_dir: Path,
    focus_datasets: tuple[str, ...] = DEFAULT_REBASELINE_FOCUS_DATASETS,
) -> dict[str, Any]:
    canonical_index = build_canonical_baseline_index(read_json(canonical_summary_json))
    focus_dataset_set = set(focus_datasets)
    per_dataset_rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        context = load_run_context(run_dir)
        metrics = read_json(run_dir / "views" / "metrics.json")
        summary_rows = metrics.get("summary") or []
        run_cot1_by_dataset = {
            str(row.get("dataset") or ""): float(row.get("accuracy_mean") or 0.0)
            for row in summary_rows
            if str(row.get("method_name") or "") == "cot_1" and str(row.get("dataset") or "") != "overall"
        }
        for row in summary_rows:
            dataset = str(row.get("dataset") or "")
            if dataset == "overall" or dataset not in canonical_index["per_dataset"]:
                continue
            method_name = str(row.get("method_name") or "")
            method_kind = str(row.get("method_kind") or "")
            if canonical_standard_method_name(method_name) is not None or method_kind in {"control", "baseline"}:
                continue
            baselines = canonical_index["per_dataset"][dataset]
            if "cot_1" not in baselines:
                continue
            canonical_best_method, canonical_best_accuracy = best_canonical_baseline(baselines)
            method_accuracy = float(row.get("accuracy_mean") or 0.0)
            old_official_cot1_accuracy = float(baselines["cot_1"]["reference_accuracy"])
            canonical_cot1_accuracy = float(baselines["cot_1"]["optimized_mean_accuracy"])
            question_count = int(row.get("question_count") or row.get("questions_per_rerun") or 0)
            per_dataset_rows.append(
                {
                    "experiment_name": context.experiment_name,
                    "family_name": context.family_name,
                    "run_dir": run_dir.as_posix(),
                    "dataset": dataset,
                    "method_name": method_name,
                    "method_kind": method_kind,
                    "question_count": question_count,
                    "method_accuracy": method_accuracy,
                    "run_cot1_accuracy": run_cot1_by_dataset.get(dataset),
                    "old_official_cot1_accuracy": old_official_cot1_accuracy,
                    "canonical_cot1_accuracy": canonical_cot1_accuracy,
                    "canonical_best_method": canonical_best_method,
                    "canonical_best_accuracy": canonical_best_accuracy,
                    "delta_vs_old_official_cot1": method_accuracy - old_official_cot1_accuracy,
                    "delta_vs_canonical_cot1": method_accuracy - canonical_cot1_accuracy,
                    "delta_vs_canonical_best": method_accuracy - canonical_best_accuracy,
                    "judgement": classify_rebaseline_result(
                        method_accuracy=method_accuracy,
                        old_official_cot1_accuracy=old_official_cot1_accuracy,
                        canonical_cot1_accuracy=canonical_cot1_accuracy,
                        canonical_best_accuracy=canonical_best_accuracy,
                    ),
                }
            )

    aggregate_rows = build_rebaseline_aggregate_rows(per_dataset_rows)
    payload = {
        "baseline_mainline": {
            "config": CANONICAL_BASELINE_CONFIG,
            "methods": list(BASE_METHOD_ORDER),
            "temperature": 0.7,
            "count100_reruns": 3,
            "note": "旧 screening/search 配置已退出正式主线；后续主结论统一对齐 canonical_simple_baselines。",
        },
        "canonical_summary_json": canonical_summary_json.as_posix(),
        "run_dirs": [path.as_posix() for path in run_dirs],
        "focus_datasets": list(focus_datasets),
        "canonical_baselines": canonical_index["per_dataset"],
        "aggregate_rows": aggregate_rows,
        "per_dataset_rows": per_dataset_rows,
        "focus_rows": [row for row in per_dataset_rows if str(row["dataset"]) in focus_dataset_set],
        "judgement_counts": judgement_counts(aggregate_rows),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = write_json(output_dir / "canonical_baseline_recheck.json", payload)
    markdown_path = write_markdown(output_dir / "canonical_baseline_recheck.md", render_rebaseline_markdown(payload))
    return {
        "json_path": json_path.as_posix(),
        "markdown_path": markdown_path.as_posix(),
    }


def build_canonical_baseline_index(payload: dict[str, Any]) -> dict[str, Any]:
    per_dataset: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in payload.get("per_dataset_rows") or []:
        dataset = str(row.get("dataset") or "")
        base_method = str(row.get("base_method") or "")
        if dataset and base_method:
            per_dataset[dataset][base_method] = dict(row)
    return {"per_dataset": dict(per_dataset)}


def build_rebaseline_aggregate_rows(per_dataset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_dataset_rows:
        grouped[(str(row["experiment_name"]), str(row["method_name"]), str(row["run_dir"]))].append(row)

    aggregate_rows: list[dict[str, Any]] = []
    for (experiment_name, method_name, run_dir), rows in sorted(grouped.items()):
        total_questions = sum(int(row["question_count"]) for row in rows)
        if total_questions <= 0:
            continue
        method_accuracy = weighted_accuracy(rows, "method_accuracy", total_questions)
        old_official_cot1_accuracy = weighted_accuracy(rows, "old_official_cot1_accuracy", total_questions)
        canonical_cot1_accuracy = weighted_accuracy(rows, "canonical_cot1_accuracy", total_questions)
        canonical_best_accuracy = weighted_accuracy(rows, "canonical_best_accuracy", total_questions)
        aggregate_rows.append(
            {
                "experiment_name": experiment_name,
                "method_name": method_name,
                "run_dir": run_dir,
                "datasets": sorted({str(row["dataset"]) for row in rows}),
                "dataset_count": len({str(row["dataset"]) for row in rows}),
                "question_count": total_questions,
                "method_accuracy": method_accuracy,
                "old_official_cot1_accuracy": old_official_cot1_accuracy,
                "canonical_cot1_accuracy": canonical_cot1_accuracy,
                "canonical_best_accuracy": canonical_best_accuracy,
                "delta_vs_old_official_cot1": method_accuracy - old_official_cot1_accuracy,
                "delta_vs_canonical_cot1": method_accuracy - canonical_cot1_accuracy,
                "delta_vs_canonical_best": method_accuracy - canonical_best_accuracy,
                "judgement": classify_rebaseline_result(
                    method_accuracy=method_accuracy,
                    old_official_cot1_accuracy=old_official_cot1_accuracy,
                    canonical_cot1_accuracy=canonical_cot1_accuracy,
                    canonical_best_accuracy=canonical_best_accuracy,
                ),
            }
        )
    return sorted(aggregate_rows, key=lambda row: (str(row["experiment_name"]), str(row["method_name"])))


def weighted_accuracy(rows: list[dict[str, Any]], field_name: str, total_questions: int) -> float:
    return sum(float(row[field_name]) * int(row["question_count"]) for row in rows) / total_questions


def best_canonical_baseline(baselines: dict[str, dict[str, Any]]) -> tuple[str, float]:
    best_method = ""
    best_accuracy = -1.0
    for base_method in BASE_METHOD_ORDER:
        row = baselines.get(base_method)
        if row is None:
            continue
        accuracy = float(row.get("optimized_mean_accuracy") or 0.0)
        if accuracy > best_accuracy:
            best_method = base_method
            best_accuracy = accuracy
    return best_method, max(best_accuracy, 0.0)


def classify_rebaseline_result(
    *,
    method_accuracy: float,
    old_official_cot1_accuracy: float,
    canonical_cot1_accuracy: float,
    canonical_best_accuracy: float,
) -> str:
    if method_accuracy - canonical_best_accuracy >= 0.01:
        return "holds_vs_canonical_best"
    if method_accuracy > canonical_best_accuracy:
        return "borderline_above_canonical_best"
    if method_accuracy - canonical_cot1_accuracy >= 0.01:
        return "beats_canonical_cot_not_best"
    if method_accuracy > canonical_cot1_accuracy:
        return "borderline_above_canonical_cot_not_best"
    if method_accuracy > old_official_cot1_accuracy:
        return "only_beats_old_official_cot1"
    return "does_not_beat_old_official_cot1"


def judgement_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row["judgement"])] += 1
    return dict(sorted(counts.items()))


def render_rebaseline_markdown(payload: dict[str, Any]) -> str:
    aggregate_rows = payload.get("aggregate_rows") or []
    focus_rows = payload.get("focus_rows") or []
    canonical_baselines = payload.get("canonical_baselines") or {}
    focus_datasets = [str(item) for item in payload.get("focus_datasets") or []]
    lines = [
        "# Canonical Baseline Recheck",
        "",
        "## 摘要",
        "",
        "- 正式 simple baseline 主线固定为 `canonical_simple_baselines.toml`。",
        "- 方法固定为 `cot_1 / mv_3 / sc_5`，全局 `temperature=0.7`，`count100` 使用 3 reruns。",
        "- 旧的 prompt/temperature screening 逻辑已退出正式主线；本工具只做 canonical baseline 复核。",
        "- `old official cot_1` 来自历史 ceiling summary；`run cot_1` 是被复核 run 自带的 control，二者可能不同。",
        "- `holds_vs_canonical_best` 表示方法超过同数据集上 `cot_1/mv_3/sc_5` 三者中最强的 canonical baseline。",
        "- `only_beats_old_official_cot1` 表示结论只相对旧 `cot_1` 成立，不能再包装成优于 strong simple baseline。",
        "",
        "## 输入 runs",
        "",
    ]
    for run_dir in payload.get("run_dirs") or []:
        lines.append(f"- `{run_dir}`")
    lines.extend(
        [
            "",
            "## Canonical Baseline 表",
            "",
            "| Dataset | Old official cot_1 | Canonical cot_1 | Canonical mv_3 | Canonical sc_5 | Canonical Best |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for dataset in sorted(canonical_baselines):
        baselines = canonical_baselines[dataset]
        best_method, best_accuracy = best_canonical_baseline(baselines)
        lines.append(
            "| {dataset} | {old_cot} | {new_cot} | {new_mv} | {new_sc} | `{best_method}` {best_accuracy:.4f} |".format(
                dataset=dataset,
                old_cot=format_accuracy(baselines.get("cot_1", {}).get("reference_accuracy")),
                new_cot=format_accuracy(baselines.get("cot_1", {}).get("optimized_mean_accuracy")),
                new_mv=format_accuracy(baselines.get("mv_3", {}).get("optimized_mean_accuracy")),
                new_sc=format_accuracy(baselines.get("sc_5", {}).get("optimized_mean_accuracy")),
                best_method=best_method,
                best_accuracy=best_accuracy,
            )
        )
    lines.extend(
        [
            "",
            "## Overlap Aggregate 结论",
            "",
            "| Experiment | Method | Datasets | Acc | Old cot_1 | New cot_1 | Canonical Best | Delta Best | Judgement |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in aggregate_rows:
        lines.append(
            "| `{experiment_name}` | `{method_name}` | {dataset_count} | {method_accuracy:.4f} | {old_official_cot1_accuracy:.4f} | {canonical_cot1_accuracy:.4f} | {canonical_best_accuracy:.4f} | {delta_vs_canonical_best:+.4f} | `{judgement}` |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            f"## Focus Dataset 结论: {', '.join(focus_datasets)}",
            "",
            "| Dataset | Experiment | Method | Acc | Run cot_1 | Old cot_1 | New cot_1 | Canonical Best | Delta Best | Judgement |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in sorted(focus_rows, key=lambda item: (str(item["dataset"]), str(item["experiment_name"]), str(item["method_name"]))):
        lines.append(
            "| `{dataset}` | `{experiment_name}` | `{method_name}` | {method_accuracy:.4f} | {run_cot1} | {old_official_cot1_accuracy:.4f} | {canonical_cot1_accuracy:.4f} | `{canonical_best_method}` {canonical_best_accuracy:.4f} | {delta_vs_canonical_best:+.4f} | `{judgement}` |".format(
                run_cot1=format_accuracy(row.get("run_cot1_accuracy")),
                **row,
            )
        )
    lines.extend(["", "## Judgement Counts", ""])
    for judgement, count in (payload.get("judgement_counts") or {}).items():
        lines.append(f"- `{judgement}`: {count}")
    lines.append("")
    return "\n".join(lines)


def format_accuracy(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.4f}"


def load_run_context(run_dir: Path) -> RunContext:
    manifest = read_json(run_dir / "manifest.json")
    return RunContext(
        run_dir=run_dir,
        family_name=str(manifest.get("family_name") or ""),
        experiment_name=str(manifest.get("experiment_name") or manifest.get("experiment") or ""),
        phase_name=str(manifest.get("phase_name") or manifest.get("phase") or ""),
    )


if __name__ == "__main__":
    raise SystemExit(main())
