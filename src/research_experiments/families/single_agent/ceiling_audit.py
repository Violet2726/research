"""固定预算 baseline ceiling 审计工具。"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research_experiments.core.io import read_json, read_jsonl, read_toml, write_json, write_markdown
from research_experiments.families.single_agent.config import load_experiment_config
from research_experiments.families.single_agent.prompts import (
    DEFAULT_PROMPT_VERSION,
    UNIFIED_CONTROL_PORT_PROMPT_VERSION,
    ZERO_SHOT_COT_PROMPT_VERSION,
)
from research_experiments.family_runtime.comparators import canonical_standard_method_name
from research_experiments.family_runtime.method_catalog import MethodConfig, load_method_catalog

REFERENCE_PROMPT_VERSIONS = (
    DEFAULT_PROMPT_VERSION,
    UNIFIED_CONTROL_PORT_PROMPT_VERSION,
    ZERO_SHOT_COT_PROMPT_VERSION,
)
BASE_METHOD_ORDER = ("cot_1", "mv_3", "sc_5")
DEFAULT_REBASELINE_FOCUS_DATASETS = ("math500", "mmlu_pro", "hotpotqa")
BENCHMARK_CONFIGS = (
    "configs/core/shared/benchmarks/competition_math/MATH.toml",
    "configs/core/shared/benchmarks/gpqa/dataset.toml",
    "configs/core/shared/benchmarks/gsm8k/test.toml",
    "configs/core/shared/benchmarks/hotpotqa/validation_distractor.toml",
    "configs/core/shared/benchmarks/math500/test.toml",
    "configs/core/shared/benchmarks/mmlu-pro/test.toml",
)
COUNT20_SPLITS = {
    "competition_math": "count20_seed0",
    "gpqa_diamond": "count20_seed42",
    "gsm8k": "count20_seed42",
    "hotpotqa": "count20_seed42",
    "math500": "count20_seed42",
    "mmlu_pro": "count20_seed42",
}
COUNT100_SPLITS = {
    "competition_math": "count100_total_seed0",
    "gpqa_diamond": "count100_seed42",
    "gsm8k": "count100_seed42",
    "hotpotqa": "count100_seed42",
    "math500": "count100_seed42",
    "mmlu_pro": "count100_seed42",
}
COUNT100_CONFIG_NAMES = {
    DEFAULT_PROMPT_VERSION: "baseline_ceiling_v1_selected_current_prompt",
    UNIFIED_CONTROL_PORT_PROMPT_VERSION: "baseline_ceiling_v1_selected_unified_control",
    ZERO_SHOT_COT_PROMPT_VERSION: "baseline_ceiling_v1_selected_zero_shot_cot",
}
COUNT100_CONFIG_DESCRIPTIONS = {
    DEFAULT_PROMPT_VERSION: "Count100 ceiling verification for selected fixed-budget baseline candidates using the current single-agent prompt family.",
    UNIFIED_CONTROL_PORT_PROMPT_VERSION: "Count100 ceiling verification for selected fixed-budget baseline candidates using the ported unified control prompt family.",
    ZERO_SHOT_COT_PROMPT_VERSION: "Count100 ceiling verification for selected fixed-budget baseline candidates using the zero-shot CoT prompt family.",
}
ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class RunContext:
    run_dir: Path
    family_name: str
    experiment_name: str
    phase_name: str
    prompt_version: str
    manifest: dict[str, Any]
    method_specs: dict[str, MethodConfig]


@dataclass(frozen=True)
class CandidateSummary:
    prompt_version: str
    method_name: str
    base_method: str
    temperature: float
    family: str
    budget_calls: int
    question_count: int
    correct_count: int
    accuracy: float
    mean_total_tokens: float
    run_dir: str


@dataclass(frozen=True)
class CandidateCeiling:
    prompt_version: str
    method_name: str
    base_method: str
    temperature: float
    family: str
    budget_calls: int
    question_count: int
    rerun_count: int
    mean_accuracy: float
    best_single_rerun_accuracy: float
    mean_total_tokens: float
    run_dir: str
    dataset_metrics: dict[str, dict[str, float]]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "reference-audit":
        payload = build_reference_audit(
            run_dirs=[Path(item) for item in args.run_dir],
            output_dir=Path(args.output_dir),
        )
        print(payload["markdown_path"])
        return 0
    if args.command == "select-screening":
        payload = select_screening_candidates(
            run_dirs=[Path(item) for item in args.run_dir],
            output_dir=Path(args.output_dir),
        )
        print(payload["selection_json"])
        return 0
    if args.command == "summarize-ceiling":
        payload = summarize_ceiling_results(
            reference_run_dirs=[Path(item) for item in args.reference_run_dir],
            optimized_run_dirs=[Path(item) for item in args.optimized_run_dir],
            output_dir=Path(args.output_dir),
            selection_json=Path(args.selection_json) if args.selection_json else None,
        )
        print(payload["markdown_path"])
        return 0
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
    parser = argparse.ArgumentParser(description="固定预算 baseline ceiling 审计工具。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    reference = subparsers.add_parser("reference-audit", help="汇总现有 official baseline 与 stronger control 证据。")
    reference.add_argument("--run-dir", action="append", required=True, help="输入 run 目录，可重复传入。")
    reference.add_argument("--output-dir", required=True, help="输出目录。")

    screen = subparsers.add_parser("select-screening", help="从 count20 screening runs 生成晋级到 count100 的配置。")
    screen.add_argument("--run-dir", action="append", required=True, help="count20 screening run 目录，可重复传入。")
    screen.add_argument("--output-dir", required=True, help="输出目录。")

    summary = subparsers.add_parser("summarize-ceiling", help="汇总 count100 ceiling 结论。")
    summary.add_argument("--reference-run-dir", action="append", required=True, help="reference run 目录，可重复传入。")
    summary.add_argument("--optimized-run-dir", action="append", required=True, help="optimized count100 run 目录，可重复传入。")
    summary.add_argument("--selection-json", help="`select-screening` 产出的 JSON。")
    summary.add_argument("--output-dir", required=True, help="输出目录。")

    rebaseline = subparsers.add_parser("rebaseline-conclusions", help="用 canonical simple baseline 复核主方法结论。")
    rebaseline.add_argument("--canonical-summary-json", required=True, help="`summarize-ceiling` 产出的 JSON。")
    rebaseline.add_argument("--run-dir", action="append", required=True, help="需要复核的主方法 count100 run 目录，可重复传入。")
    rebaseline.add_argument("--focus-dataset", action="append", help="重点展示的数据集，可重复传入。")
    rebaseline.add_argument("--output-dir", required=True, help="输出目录。")
    return parser


def build_reference_audit(*, run_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        context = load_run_context(run_dir)
        summary_rows = read_json(run_dir / "views" / "metrics.json").get("summary", [])
        split_lookup = build_split_lookup(context.manifest)
        for row in summary_rows:
            method_name = str(row.get("method_name") or "")
            if method_name not in context.method_specs:
                continue
            base_method = base_method_name(method_name)
            if base_method is None:
                continue
            spec = context.method_specs[method_name]
            rows.append(
                {
                    "dataset": str(row.get("dataset") or ""),
                    "base_method": base_method,
                    "candidate_method": method_name,
                    "prompt_version": context.prompt_version,
                    "temperature": spec.temperature,
                    "budget_calls": spec.budget_calls,
                    "split": split_lookup.get(str(row.get("dataset") or ""), ""),
                    "accuracy_mean": float(row.get("accuracy_mean") or 0.0),
                    "total_tokens_mean": float(row.get("total_tokens_mean") or 0.0),
                    "source_family": context.family_name,
                    "source_experiment": context.experiment_name,
                    "run_dir": run_dir.as_posix(),
                }
            )

    rows.sort(key=lambda item: (item["dataset"], BASE_METHOD_ORDER.index(item["base_method"]), item["source_experiment"]))
    payload = {"rows": rows, "run_dirs": [path.as_posix() for path in run_dirs]}
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = write_json(output_dir / "reference_audit.json", payload)
    markdown_path = write_markdown(output_dir / "reference_audit.md", render_reference_audit_markdown(rows, run_dirs))
    return {
        "json_path": json_path.as_posix(),
        "markdown_path": markdown_path.as_posix(),
    }


def select_screening_candidates(*, run_dirs: list[Path], output_dir: Path) -> dict[str, Any]:
    candidate_map: dict[tuple[str, str], CandidateSummary] = {}
    rankings_by_base_method: dict[str, list[CandidateSummary]] = {}

    for run_dir in run_dirs:
        context = load_run_context(run_dir)
        if context.family_name != "single_agent":
            raise ValueError(f"Screening run must be single_agent: {run_dir}")
        if context.phase_name != "count20":
            raise ValueError(f"Screening run must use phase count20: {run_dir}")
        if context.prompt_version not in REFERENCE_PROMPT_VERSIONS:
            raise ValueError(f"Unexpected prompt_version for screening run {run_dir}: {context.prompt_version}")
        predictions = read_jsonl(run_dir / "views" / "predictions.jsonl")
        rerun_indices = sorted({int(row.get("rerun_index", 0)) for row in predictions})
        if rerun_indices != [0]:
            raise ValueError(f"Screening run must have exactly one rerun (rerun0 only): {run_dir}")
        per_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in predictions:
            method_name = str(row.get("method_name") or "")
            if base_method_name(method_name) is None:
                continue
            per_method[method_name].append(row)
        for method_name, rows in per_method.items():
            spec = context.method_specs[method_name]
            summary = summarize_screen_candidate(context, spec, rows)
            candidate_map[(summary.prompt_version, summary.method_name)] = summary

    for base_method in BASE_METHOD_ORDER:
        candidates = [item for item in candidate_map.values() if item.base_method == base_method]
        rankings_by_base_method[base_method] = sorted(
            candidates,
            key=lambda item: (-item.correct_count, item.mean_total_tokens, item.prompt_version, item.method_name),
        )

    selected_candidates: list[CandidateSummary] = []
    for base_method in BASE_METHOD_ORDER:
        ranked = rankings_by_base_method[base_method]
        if not ranked:
            continue
        selected_lookup: dict[tuple[str, str], CandidateSummary] = {}
        for candidate in ranked[:2]:
            selected_lookup[(candidate.prompt_version, candidate.method_name)] = candidate
        best_correct = ranked[0].correct_count
        for candidate in ranked[2:]:
            if candidate.correct_count >= best_correct - 1:
                selected_lookup[(candidate.prompt_version, candidate.method_name)] = candidate
        selected_candidates.extend(selected_lookup.values())

    output_dir.mkdir(parents=True, exist_ok=True)
    methods_dir = output_dir / "methods"
    experiments_dir = output_dir / "experiments"
    generated_configs = write_selected_count100_configs(selected_candidates, methods_dir, experiments_dir)
    payload = {
        "selection_rule": "keep_top2_per_base_method_and_any_additional_candidate_within_one_correct_answer_of_the_best",
        "screen_run_dirs": [path.as_posix() for path in run_dirs],
        "rankings_by_base_method": {
            base_method: [candidate_summary_payload(item) for item in rankings]
            for base_method, rankings in rankings_by_base_method.items()
        },
        "selected_candidates": [candidate_summary_payload(item) for item in selected_candidates],
        "generated_count100_configs": generated_configs,
    }
    json_path = write_json(output_dir / "selected_candidates.json", payload)
    markdown_path = write_markdown(
        output_dir / "selected_candidates.md",
        render_selected_candidates_markdown(rankings_by_base_method, selected_candidates, generated_configs),
    )
    return {
        "selection_json": json_path.as_posix(),
        "selection_markdown": markdown_path.as_posix(),
    }


def summarize_ceiling_results(
    *,
    reference_run_dirs: list[Path],
    optimized_run_dirs: list[Path],
    output_dir: Path,
    selection_json: Path | None,
) -> dict[str, Any]:
    reference_rows = build_reference_baselines(reference_run_dirs)
    optimized_candidates = build_optimized_candidates(optimized_run_dirs)
    selection_payload = read_json(selection_json) if selection_json is not None else {}
    screen_winners = {
        base_method: (
            str(rankings[0]["prompt_version"]),
            str(rankings[0]["method_name"]),
        )
        for base_method, rankings in (selection_payload.get("rankings_by_base_method") or {}).items()
        if rankings
    }

    overall_rows: list[dict[str, Any]] = []
    per_dataset_rows: list[dict[str, Any]] = []
    winner_payload: dict[str, Any] = {}
    for base_method in BASE_METHOD_ORDER:
        candidates = sorted(
            optimized_candidates.get(base_method, []),
            key=lambda item: (-item.mean_accuracy, -item.best_single_rerun_accuracy, item.mean_total_tokens, item.prompt_version, item.method_name),
        )
        if not candidates:
            continue
        winner = candidates[0]
        reference_overall = reference_rows["overall"].get(base_method)
        screen_winner = screen_winners.get(base_method)
        winner_payload[base_method] = {
            "screen_winner": list(screen_winner) if screen_winner is not None else None,
            "count100_winner": [winner.prompt_version, winner.method_name],
            "count20_and_count100_winner_match": screen_winner == (winner.prompt_version, winner.method_name),
        }
        overall_rows.append(
            {
                "base_method": base_method,
                "reference_candidate": reference_overall["candidate_method"] if reference_overall else "",
                "optimized_candidate": winner.method_name,
                "optimized_prompt_version": winner.prompt_version,
                "reference_overall_accuracy": float(reference_overall["accuracy_mean"]) if reference_overall else 0.0,
                "optimized_mean_accuracy": winner.mean_accuracy,
                "optimized_best_single_rerun_accuracy": winner.best_single_rerun_accuracy,
                "delta_accuracy": winner.mean_accuracy - float(reference_overall["accuracy_mean"]) if reference_overall else winner.mean_accuracy,
                "reference_total_tokens_mean": float(reference_overall["total_tokens_mean"]) if reference_overall else 0.0,
                "optimized_total_tokens_mean": winner.mean_total_tokens,
                "count20_and_count100_winner_match": screen_winner == (winner.prompt_version, winner.method_name),
            }
        )
        for dataset, metrics in winner.dataset_metrics.items():
            reference_dataset = reference_rows["per_dataset"].get((base_method, dataset))
            per_dataset_rows.append(
                {
                    "base_method": base_method,
                    "dataset": dataset,
                    "reference_candidate": reference_dataset["candidate_method"] if reference_dataset else "",
                    "optimized_candidate": winner.method_name,
                    "reference_accuracy": float(reference_dataset["accuracy_mean"]) if reference_dataset else 0.0,
                    "optimized_mean_accuracy": metrics["mean_accuracy"],
                    "delta_accuracy": metrics["mean_accuracy"] - float(reference_dataset["accuracy_mean"]) if reference_dataset else metrics["mean_accuracy"],
                    "reference_total_tokens_mean": float(reference_dataset["total_tokens_mean"]) if reference_dataset else 0.0,
                    "optimized_total_tokens_mean": metrics["mean_total_tokens"],
                    "reference_source": reference_dataset["source_experiment"] if reference_dataset else "",
                }
            )

    payload = {
        "reference_run_dirs": [path.as_posix() for path in reference_run_dirs],
        "optimized_run_dirs": [path.as_posix() for path in optimized_run_dirs],
        "overall_rows": overall_rows,
        "per_dataset_rows": per_dataset_rows,
        "winner_consistency": winner_payload,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = write_json(output_dir / "ceiling_summary.json", payload)
    markdown_path = write_markdown(output_dir / "ceiling_summary.md", render_ceiling_summary_markdown(payload))
    return {
        "json_path": json_path.as_posix(),
        "markdown_path": markdown_path.as_posix(),
    }


def rebaseline_core_conclusions(
    *,
    canonical_summary_json: Path,
    run_dirs: list[Path],
    output_dir: Path,
    focus_datasets: tuple[str, ...] = DEFAULT_REBASELINE_FOCUS_DATASETS,
) -> dict[str, Any]:
    canonical_index = build_canonical_baseline_index(read_json(canonical_summary_json))
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
    focus_rows = [
        row
        for row in per_dataset_rows
        if str(row["dataset"]) in set(focus_datasets)
    ]
    payload = {
        "canonical_summary_json": canonical_summary_json.as_posix(),
        "run_dirs": [path.as_posix() for path in run_dirs],
        "focus_datasets": list(focus_datasets),
        "canonical_baselines": canonical_index["per_dataset"],
        "aggregate_rows": aggregate_rows,
        "per_dataset_rows": per_dataset_rows,
        "focus_rows": focus_rows,
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
    return {
        "per_dataset": dict(per_dataset),
    }


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
    return sorted(
        aggregate_rows,
        key=lambda row: (str(row["experiment_name"]), str(row["method_name"])),
    )


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
        "- Canonical simple baseline 固定为 `cot_1@temp=0.7 / mv_3@temp=0.7 / sc_5@temp=0.7`。",
        "- `old official cot_1` 来自 ceiling summary 的旧 official baseline；`run cot_1` 是被复核 run 自带的 control，二者可能不同。",
        "- `holds_vs_canonical_best` 表示方法超过同数据集上 `cot_1/mv_3/sc_5` 三者中最强的 canonical baseline。",
        "- `only_beats_old_official_cot1` 表示结论只相对旧 `cot_1` 成立，不能再包装成优于 strong simple baseline。",
        "- `borderline_*` 表示差值小于 `0.01`，在 count100 口径下应视作边际信号。",
        "- 本表只用于同上下文/full-context 口径；split-context 结论应继续用 split no-comm baseline 单独复核。",
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
    method_specs = method_specs_for_manifest(manifest)
    return RunContext(
        run_dir=run_dir,
        family_name=str(manifest.get("family_name") or ""),
        experiment_name=str(manifest.get("experiment_name") or manifest.get("experiment") or ""),
        phase_name=str(manifest.get("phase_name") or manifest.get("phase") or ""),
        prompt_version=str(manifest.get("prompt_version") or ""),
        manifest=manifest,
        method_specs=method_specs,
    )


def method_specs_for_manifest(manifest: dict[str, Any]) -> dict[str, MethodConfig]:
    family_name = str(manifest.get("family_name") or "")
    if family_name == "single_agent":
        methods_payload = manifest.get("methods")
        if isinstance(methods_payload, dict) and methods_payload:
            return {
                str(name): MethodConfig(name=str(name), **_strip_name_key(config))
                for name, config in methods_payload.items()
            }
        return infer_single_agent_method_specs(manifest)
    if family_name == "adaptive_sparse_mad":
        controls = manifest.get("controls") or {}
        return {
            str(name): MethodConfig(name=str(name), **_strip_name_key(config))
            for name, config in controls.items()
        }
    return {}


def infer_single_agent_method_specs(manifest: dict[str, Any]) -> dict[str, MethodConfig]:
    experiment_name = str(manifest.get("experiment_name") or manifest.get("experiment") or "")
    phase_name = str(manifest.get("phase_name") or manifest.get("phase") or "")
    config_path = find_single_agent_config_path(experiment_name)
    experiment = load_experiment_config(config_path)
    phase_payload = experiment.raw["phases"][phase_name]
    method_names = [str(item) for item in phase_payload.get("methods", [])]
    catalog = load_method_catalog(experiment.method_catalog)
    return {
        method_name: catalog[method_name]
        for method_name in method_names
        if method_name in catalog
    }


def find_single_agent_config_path(experiment_name: str) -> Path:
    experiments_root = ROOT / "configs" / "families" / "single_agent" / "experiments"
    for path in experiments_root.glob("*.toml"):
        payload = read_toml(path)
        if str(payload.get("name") or "") == experiment_name:
            return path
    raise FileNotFoundError(f"Unable to locate single_agent config for experiment {experiment_name!r}")


def summarize_screen_candidate(context: RunContext, spec: MethodConfig, rows: list[dict[str, Any]]) -> CandidateSummary:
    correct_count = sum(int(float(row.get("score") or 0.0)) for row in rows)
    question_count = len(rows)
    mean_total_tokens = sum(float(row.get("total_tokens_per_question") or 0.0) for row in rows) / question_count if question_count else 0.0
    return CandidateSummary(
        prompt_version=context.prompt_version,
        method_name=spec.name,
        base_method=base_method_name(spec.name) or spec.name,
        temperature=spec.temperature,
        family=spec.family,
        budget_calls=spec.budget_calls,
        question_count=question_count,
        correct_count=correct_count,
        accuracy=(correct_count / question_count) if question_count else 0.0,
        mean_total_tokens=mean_total_tokens,
        run_dir=context.run_dir.as_posix(),
    )


def build_reference_baselines(run_dirs: list[Path]) -> dict[str, dict[Any, dict[str, Any]]]:
    per_dataset: dict[tuple[str, str], dict[str, Any]] = {}
    for run_dir in run_dirs:
        context = load_run_context(run_dir)
        split_lookup = build_split_lookup(context.manifest)
        summary_rows = read_json(run_dir / "views" / "metrics.json").get("summary", [])
        for row in summary_rows:
            method_name = str(row.get("method_name") or "")
            if method_name not in context.method_specs:
                continue
            base_method = base_method_name(method_name)
            if base_method is None:
                continue
            payload = {
                "dataset": str(row.get("dataset") or ""),
                "base_method": base_method,
                "candidate_method": method_name,
                "accuracy_mean": float(row.get("accuracy_mean") or 0.0),
                "total_tokens_mean": float(row.get("total_tokens_mean") or 0.0),
                "question_count": int(row.get("questions_per_rerun") or row.get("question_count") or 0),
                "prompt_version": context.prompt_version,
                "source_family": context.family_name,
                "source_experiment": context.experiment_name,
                "split": split_lookup.get(str(row.get("dataset") or ""), ""),
            }
            key = (base_method, payload["dataset"])
            current = per_dataset.get(key)
            if current is None or reference_precedence(payload) < reference_precedence(current):
                per_dataset[key] = payload

    grouped_for_overall: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in per_dataset.values():
        grouped_for_overall[str(item["base_method"])].append(item)

    overall: dict[str, dict[str, Any]] = {}
    for base_method, rows in grouped_for_overall.items():
        unique_rows = {(row["dataset"], row["candidate_method"], row["source_experiment"]): row for row in rows}.values()
        total_questions = sum(int(row["question_count"]) for row in unique_rows)
        total_correct = sum(float(row["accuracy_mean"]) * int(row["question_count"]) for row in unique_rows)
        total_tokens = sum(float(row["total_tokens_mean"]) * int(row["question_count"]) for row in unique_rows)
        ordered = sorted(unique_rows, key=lambda row: (reference_precedence(row), row["dataset"]))
        overall[base_method] = {
            "base_method": base_method,
            "candidate_method": ",".join(sorted({str(row["candidate_method"]) for row in ordered})),
            "accuracy_mean": (total_correct / total_questions) if total_questions else 0.0,
            "total_tokens_mean": (total_tokens / total_questions) if total_questions else 0.0,
            "question_count": total_questions,
            "source_experiment": ",".join(sorted({str(row["source_experiment"]) for row in ordered})),
        }
    return {
        "overall": overall,
        "per_dataset": per_dataset,
    }


def build_optimized_candidates(run_dirs: list[Path]) -> dict[str, list[CandidateCeiling]]:
    grouped: dict[str, list[CandidateCeiling]] = defaultdict(list)
    for run_dir in run_dirs:
        context = load_run_context(run_dir)
        if context.family_name != "single_agent":
            raise ValueError(f"Optimized run must be single_agent: {run_dir}")
        if context.phase_name != "count100":
            raise ValueError(f"Optimized run must use phase count100: {run_dir}")
        predictions = read_jsonl(run_dir / "views" / "predictions.jsonl")
        rows_by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in predictions:
            method_name = str(row.get("method_name") or "")
            if base_method_name(method_name) is None:
                continue
            rows_by_method[method_name].append(row)
        for method_name, rows in rows_by_method.items():
            spec = context.method_specs[method_name]
            ceiling = summarize_count100_candidate(context, spec, rows)
            grouped[ceiling.base_method].append(ceiling)
    return grouped


def summarize_count100_candidate(context: RunContext, spec: MethodConfig, rows: list[dict[str, Any]]) -> CandidateCeiling:
    rerun_scores: dict[int, list[float]] = defaultdict(list)
    rerun_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    dataset_rerun_scores: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    dataset_token_sums: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        rerun_index = int(row.get("rerun_index", 0))
        score = float(row.get("score") or 0.0)
        rerun_scores[rerun_index].append(score)
        rerun_rows[rerun_index].append(row)
        dataset = str(row.get("dataset") or "")
        dataset_rerun_scores[dataset][rerun_index].append(score)
        dataset_token_sums[dataset].append(float(row.get("total_tokens_per_question") or 0.0))
    rerun_accuracies = [
        (sum(scores) / len(scores)) if scores else 0.0
        for _, scores in sorted(rerun_scores.items())
    ]
    dataset_metrics = {
        dataset: {
            "mean_accuracy": (
                sum((sum(scores) / len(scores)) if scores else 0.0 for _, scores in sorted(per_rerun.items())) / len(per_rerun)
                if per_rerun else 0.0
            ),
            "mean_total_tokens": (sum(dataset_token_sums[dataset]) / len(dataset_token_sums[dataset])) if dataset_token_sums[dataset] else 0.0,
        }
        for dataset, per_rerun in dataset_rerun_scores.items()
    }
    mean_total_tokens = sum(float(row.get("total_tokens_per_question") or 0.0) for row in rows) / len(rows) if rows else 0.0
    question_count = len(rows) // max(1, len(rerun_scores))
    return CandidateCeiling(
        prompt_version=context.prompt_version,
        method_name=spec.name,
        base_method=base_method_name(spec.name) or spec.name,
        temperature=spec.temperature,
        family=spec.family,
        budget_calls=spec.budget_calls,
        question_count=question_count,
        rerun_count=len(rerun_scores),
        mean_accuracy=sum(rerun_accuracies) / len(rerun_accuracies) if rerun_accuracies else 0.0,
        best_single_rerun_accuracy=max(rerun_accuracies) if rerun_accuracies else 0.0,
        mean_total_tokens=mean_total_tokens,
        run_dir=context.run_dir.as_posix(),
        dataset_metrics=dataset_metrics,
    )


def write_selected_count100_configs(
    selected_candidates: list[CandidateSummary],
    methods_dir: Path,
    experiments_dir: Path,
) -> list[dict[str, str]]:
    methods_dir.mkdir(parents=True, exist_ok=True)
    experiments_dir.mkdir(parents=True, exist_ok=True)
    selected_by_prompt: dict[str, list[CandidateSummary]] = defaultdict(list)
    for candidate in selected_candidates:
        selected_by_prompt[candidate.prompt_version].append(candidate)

    generated: list[dict[str, str]] = []
    for prompt_version, candidates in sorted(selected_by_prompt.items()):
        prompt_slug = prompt_version_slug(prompt_version)
        method_catalog_path = methods_dir / f"{prompt_slug}.toml"
        experiment_path = experiments_dir / f"{prompt_slug}.toml"
        method_catalog_path.write_text(render_method_catalog(candidates), encoding="utf-8")
        experiment_path.write_text(render_count100_experiment(prompt_version, method_catalog_path, candidates), encoding="utf-8")
        generated.append(
            {
                "prompt_version": prompt_version,
                "method_catalog": method_catalog_path.as_posix(),
                "experiment_config": experiment_path.as_posix(),
                "run_command": (
                    "uv run research_cli experiment --family single_agent run "
                    f"--experiment {experiment_path.as_posix()} --phase count100 --model xiaomimimo/mimo-v2.5"
                ),
            }
        )
    return generated


def render_method_catalog(candidates: list[CandidateSummary]) -> str:
    ordered = sorted(candidates, key=lambda item: (BASE_METHOD_ORDER.index(item.base_method), item.temperature, item.method_name))
    lines = [
        "# 自动生成：count100 ceiling 晋级候选",
        "# 由 `research_experiments.families.single_agent.ceiling_audit select-screening` 生成。",
        "",
    ]
    for candidate in ordered:
        lines.append(f"[methods.{candidate.method_name}]")
        lines.append(f'family = "{candidate.family}"')
        lines.append(f"budget_calls = {candidate.budget_calls}")
        lines.append(f"temperature = {candidate.temperature:.1f}")
        lines.append("top_p = 1.0")
        lines.append("max_output_tokens = 256")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_count100_experiment(prompt_version: str, method_catalog_path: Path, candidates: list[CandidateSummary]) -> str:
    method_names = sorted(
        (candidate.method_name for candidate in candidates),
        key=lambda name: (BASE_METHOD_ORDER.index(base_method_name(name) or name), name),
    )
    lines = [
        f'name = "{COUNT100_CONFIG_NAMES[prompt_version]}"',
        f'description = "{COUNT100_CONFIG_DESCRIPTIONS[prompt_version]}"',
        'experiment_note = """',
        "自动生成：只保留通过 count20 screen 的 fixed-budget baseline 候选。",
        "该配置只用于 authoritative count100 ceiling 验证。",
        '"""',
        f'method_catalog = "{method_catalog_path.as_posix()}"',
        "benchmark_configs = [",
    ]
    for benchmark_config in BENCHMARK_CONFIGS:
        lines.append(f'  "{benchmark_config}",')
    lines.extend(
        [
            "]",
            "global_seed = 42",
            "reruns_per_method = 3",
            "cot_uses_reruns = true",
            f'prompt_version = "{prompt_version}"',
            'primary_model_ref = "xiaomimimo/mimo-v2.5"',
            "",
            "[phases.count100]",
            "split_overrides = { competition_math = \"count100_total_seed0\", gpqa_diamond = \"count100_seed42\", gsm8k = \"count100_seed42\", hotpotqa = \"count100_seed42\", math500 = \"count100_seed42\", mmlu_pro = \"count100_seed42\" }",
            "methods = [",
        ]
    )
    for method_name in method_names:
        lines.append(f'  "{method_name}",')
    lines.extend(
        [
            "]",
            "reruns_override = 3",
            "",
        ]
    )
    return "\n".join(lines)


def render_reference_audit_markdown(rows: list[dict[str, Any]], run_dirs: list[Path]) -> str:
    lines = [
        "# Baseline Reference Audit",
        "",
        "## 摘要",
        "",
        "- 该表并排记录当前 `single_agent` official baseline 与 A-SMAD stronger control 证据。",
        "- 若同一 `dataset/base_method` 在 stronger control 口径下明显优于 official baseline，则当前 official baseline 不能视为 ceiling。",
        "",
        "## 输入 runs",
        "",
    ]
    for run_dir in run_dirs:
        lines.append(f"- `{run_dir.as_posix()}`")
    lines.extend(
        [
            "",
            "## 证据表",
            "",
            "| Dataset | Base Method | Candidate | Prompt Version | Temp | Budget | Split | Acc | Tokens | Source |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {dataset} | `{base_method}` | `{candidate_method}` | `{prompt_version}` | {temperature:.1f} | {budget_calls} | `{split}` | {accuracy_mean:.4f} | {total_tokens_mean:.2f} | `{source_experiment}` |".format(
                **row
            )
        )
    lines.append("")
    return "\n".join(lines)


def render_selected_candidates_markdown(
    rankings_by_base_method: dict[str, list[CandidateSummary]],
    selected_candidates: list[CandidateSummary],
    generated_configs: list[dict[str, str]],
) -> str:
    lines = [
        "# Count20 Screening Selection",
        "",
        "## 规则",
        "",
        "- 每个 base method 保留前 2 名候选。",
        "- 另外保留所有与第一名只差 `<=1` 个 overall 正确样本的候选。",
        "",
    ]
    for base_method in BASE_METHOD_ORDER:
        lines.extend(
            [
                f"## {base_method}",
                "",
                "| Rank | Prompt Version | Candidate | Temp | Correct | Total | Acc | Tokens |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for rank, candidate in enumerate(rankings_by_base_method.get(base_method, []), start=1):
            lines.append(
                f"| {rank} | `{candidate.prompt_version}` | `{candidate.method_name}` | {candidate.temperature:.1f} | {candidate.correct_count} | {candidate.question_count} | {candidate.accuracy:.4f} | {candidate.mean_total_tokens:.2f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## 晋级候选",
            "",
            "| Base Method | Prompt Version | Candidate | Temp | Correct | Acc |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for candidate in sorted(selected_candidates, key=lambda item: (BASE_METHOD_ORDER.index(item.base_method), item.prompt_version, item.method_name)):
        lines.append(
            f"| `{candidate.base_method}` | `{candidate.prompt_version}` | `{candidate.method_name}` | {candidate.temperature:.1f} | {candidate.correct_count} | {candidate.accuracy:.4f} |"
        )
    lines.extend(["", "## 生成的 count100 配置", ""])
    for item in generated_configs:
        lines.append(f"- `{item['prompt_version']}`")
        lines.append(f"  config: `{item['experiment_config']}`")
        lines.append(f"  methods: `{item['method_catalog']}`")
        lines.append(f"  command: `{item['run_command']}`")
    lines.append("")
    return "\n".join(lines)


def render_ceiling_summary_markdown(payload: dict[str, Any]) -> str:
    overall_rows = payload.get("overall_rows") or []
    per_dataset_rows = payload.get("per_dataset_rows") or []
    lines = [
        "# Count100 Ceiling Summary",
        "",
        "## 总体结果",
        "",
        "| Base Method | Reference | Optimized | Prompt Version | Ref Acc | Mean Acc | Best Single Rerun | Delta | Ref Tokens | New Tokens | Winner Stable |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in overall_rows:
        lines.append(
            f"| `{row['base_method']}` | `{row['reference_candidate']}` | `{row['optimized_candidate']}` | `{row['optimized_prompt_version']}` | {row['reference_overall_accuracy']:.4f} | {row['optimized_mean_accuracy']:.4f} | {row['optimized_best_single_rerun_accuracy']:.4f} | {row['delta_accuracy']:+.4f} | {row['reference_total_tokens_mean']:.2f} | {row['optimized_total_tokens_mean']:.2f} | `{str(bool(row['count20_and_count100_winner_match'])).lower()}` |"
        )
    lines.extend(["", "## 分数据集结果", ""])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_dataset_rows:
        grouped[str(row["base_method"])].append(row)
    for base_method in BASE_METHOD_ORDER:
        rows = sorted(grouped.get(base_method, []), key=lambda item: item["dataset"])
        if not rows:
            continue
        lines.extend(
            [
                f"### {base_method}",
                "",
                "| Dataset | Reference | Optimized | Ref Acc | Mean Acc | Delta | Ref Tokens | New Tokens | Source |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in rows:
            lines.append(
                f"| `{row['dataset']}` | `{row['reference_candidate']}` | `{row['optimized_candidate']}` | {row['reference_accuracy']:.4f} | {row['optimized_mean_accuracy']:.4f} | {row['delta_accuracy']:+.4f} | {row['reference_total_tokens_mean']:.2f} | {row['optimized_total_tokens_mean']:.2f} | `{row['reference_source']}` |"
            )
        lines.append("")
    return "\n".join(lines)


def build_split_lookup(manifest: dict[str, Any]) -> dict[str, str]:
    phase_payload = manifest.get("phase_metadata") or {}
    split_lookup: dict[str, str] = {}
    if isinstance(phase_payload.get("split_overrides"), dict):
        split_lookup.update({str(key): str(value) for key, value in phase_payload["split_overrides"].items()})
    split_suffix = phase_payload.get("split_suffix")
    if split_suffix:
        for benchmark in manifest.get("benchmarks") or []:
            split_lookup[str(benchmark.get("slug") or "")] = str(split_suffix)
    return split_lookup


def base_method_name(method_name: str) -> str | None:
    candidate = str(method_name or "")
    if candidate in BASE_METHOD_ORDER:
        return candidate
    match = re.match(r"^(cot_1|mv_3|sc_5)_temp_", candidate)
    if match:
        return match.group(1)
    return None


def prompt_version_slug(prompt_version: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", prompt_version.lower()).strip("_")


def _strip_name_key(payload: Any) -> dict[str, Any]:
    normalized = dict(payload or {})
    normalized.pop("name", None)
    return normalized


def reference_precedence(row: dict[str, Any]) -> int:
    family_name = str(row.get("source_family") or "")
    if family_name == "single_agent":
        return 0
    if family_name == "adaptive_sparse_mad":
        return 1
    return 9


def candidate_summary_payload(candidate: CandidateSummary) -> dict[str, Any]:
    return {
        "prompt_version": candidate.prompt_version,
        "method_name": candidate.method_name,
        "base_method": candidate.base_method,
        "temperature": candidate.temperature,
        "family": candidate.family,
        "budget_calls": candidate.budget_calls,
        "question_count": candidate.question_count,
        "correct_count": candidate.correct_count,
        "accuracy": candidate.accuracy,
        "mean_total_tokens": candidate.mean_total_tokens,
        "run_dir": candidate.run_dir,
    }


if __name__ == "__main__":
    raise SystemExit(main())
