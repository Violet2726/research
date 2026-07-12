"""RCTA 跨 backbone 等权 full 分析与预注册 SOTA 晋级门。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from research_experiments.core.data.datasets import load_split_ids
from research_experiments.families.risk_controlled_trace_mad.router import clopper_pearson_upper

PRIMARY_DATASETS = ("omni_math_2_filtered", "bbeh")
PRE_REGISTERED_COMPETITORS = (
    "cot_1",
    "sc_3",
    "sc_5",
    "sc_7",
    "sc_9",
    "adaptive_sc_9",
    "gsa_trace_1",
    "mad_5a_r1",
    "confidence_mad_5a_r1",
)


def analyze_runs(
    run_dirs: list[str | Path],
    *,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    if len(run_dirs) != 2:
        raise ValueError("Exactly two full_seed42 run directories are required.")
    rows_by_model: dict[str, list[dict[str, Any]]] = {}
    input_run_hashes: list[str] = []
    for raw_dir in run_dirs:
        root = Path(raw_dir)
        manifest_path = root / "manifest.json"
        predictions_path = root / "views" / "predictions.jsonl"
        validation_path = root / "run_validation.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if str(manifest.get("phase_name") or manifest.get("phase")) != "full_seed42":
            raise ValueError(f"Cross-backbone analysis only accepts full_seed42 runs: {root}")
        if validation.get("passed") is not True:
            raise ValueError(f"Cross-backbone analysis requires a validated run: {root}")
        model_name = str((manifest.get("resolved_model") or {}).get("name") or "")
        if not model_name or model_name in rows_by_model:
            raise ValueError("The two full runs must use distinct resolved backbone names.")
        rows = [
            json.loads(line)
            for line in predictions_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rows_by_model[model_name] = _exclude_count300(rows, manifest)
        input_run_hashes.append(_hash_files(manifest_path, predictions_path, validation_path))
    result = analyze_cross_backbone_rows(rows_by_model, bootstrap_samples=bootstrap_samples)
    result["input_run_hashes"] = sorted(input_run_hashes)
    result["analysis_sha256"] = _payload_hash(result)
    return result


def analyze_cross_backbone_rows(
    rows_by_model: dict[str, list[dict[str, Any]]],
    *,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    if len(rows_by_model) != 2:
        raise ValueError("Equal-backbone analysis requires exactly two backbone strata.")
    models = sorted(rows_by_model)
    comparisons: list[dict[str, Any]] = []
    for dataset in PRIMARY_DATASETS:
        for competitor in PRE_REGISTERED_COMPETITORS:
            model_pairs = {
                model: _paired_rows(rows_by_model[model], dataset, competitor)
                for model in models
            }
            if any(not pairs for pairs in model_pairs.values()):
                raise ValueError(f"Missing paired {dataset}/{competitor} rows in one or more backbones.")
            draws, point = _equal_backbone_bootstrap(
                model_pairs,
                dataset=dataset,
                samples=bootstrap_samples,
                seed=f"rcta-full:{dataset}:{competitor}",
            )
            low, high = np.quantile(draws, [0.025, 0.975])
            nonpositive = (int(np.count_nonzero(draws <= 0.0)) + 1) / (bootstrap_samples + 1)
            nonnegative = (int(np.count_nonzero(draws >= 0.0)) + 1) / (bootstrap_samples + 1)
            comparisons.append(
                {
                    "dataset": dataset,
                    "comparison_method": competitor,
                    "equal_backbone_accuracy_delta": point,
                    "bootstrap_ci_95": [float(low), float(high)],
                    "bootstrap_two_sided_p": min(1.0, 2.0 * min(nonpositive, nonnegative)),
                    "bootstrap_samples": bootstrap_samples,
                }
            )
    _holm(comparisons, p_key="bootstrap_two_sided_p")
    cell_gate = _four_cell_gate(rows_by_model)
    token_gate = _token_gate(rows_by_model)
    coverage_gate = _coverage_gate(rows_by_model)
    inference_gate = all(
        float(item["bootstrap_ci_95"][0]) > 0.0 and float(item["holm_adjusted_p"]) < 0.05
        for item in comparisons
    )
    sota_gate = bool(
        inference_gate
        and cell_gate["passed"]
        and token_gate["passed"]
        and coverage_gate["passed"]
    )
    return {
        "analysis_version": "rcta_cross_backbone_full_v1",
        "models": models,
        "primary_datasets": list(PRIMARY_DATASETS),
        "reference_method": "rcta_1",
        "competitors": list(PRE_REGISTERED_COMPETITORS),
        "weighting": "equal backbone; within-task stratified BBEH bootstrap",
        "comparisons": comparisons,
        "inference_gate_passed": inference_gate,
        "four_cell_gate": cell_gate,
        "token_gate": token_gate,
        "coverage_gate": coverage_gate,
        "sota_gate_passed": sota_gate,
        "claim_if_passed": "fixed backbones and at most ten logical calls",
    }


def _exclude_count300(rows: list[dict[str, Any]], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    namespaces = {
        str(item["slug"]): str(item.get("cache_namespace") or item["slug"])
        for item in manifest.get("benchmarks") or []
    }
    excluded = {
        dataset: set(load_split_ids(namespaces[dataset], "count300_seed42", random_seed=42))
        for dataset in PRIMARY_DATASETS
    }
    return [
        row
        for row in rows
        if row.get("dataset") not in excluded
        or str(row.get("sample_id")) not in excluded[str(row["dataset"])]
    ]


def _paired_rows(rows: list[dict[str, Any]], dataset: str, competitor: str) -> list[tuple[float, float, str]]:
    index = {
        (str(row["sample_id"]), str(row["method_name"])): row
        for row in rows
        if row.get("dataset") == dataset
    }
    sample_ids = sorted(
        {sample_id for sample_id, method in index if method == "rcta_1"}
        & {sample_id for sample_id, method in index if method == competitor}
    )
    return [
        (
            float(index[(sample_id, "rcta_1")].get("score") or 0.0),
            float(index[(sample_id, competitor)].get("score") or 0.0),
            str(index[(sample_id, "rcta_1")].get("task") or "unknown"),
        )
        for sample_id in sample_ids
    ]


def _equal_backbone_bootstrap(
    model_pairs: dict[str, list[tuple[float, float, str]]],
    *,
    dataset: str,
    samples: int,
    seed: str,
) -> tuple[np.ndarray, float]:
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    rng = np.random.default_rng(int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16))
    model_draws: list[np.ndarray] = []
    model_points: list[float] = []
    for model in sorted(model_pairs):
        grouped: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
        for pair in model_pairs[model]:
            grouped[pair[2] if dataset == "bbeh" else "all"].append(pair)
        left_tasks = [float(np.mean([left for left, _, _ in group])) for group in grouped.values()]
        right_tasks = [float(np.mean([right for _, right, _ in group])) for group in grouped.values()]
        model_points.append(_aggregate_tasks(left_tasks, dataset) - _aggregate_tasks(right_tasks, dataset))
        left_boot = np.empty((samples, len(grouped)), dtype=float)
        right_boot = np.empty((samples, len(grouped)), dtype=float)
        for task_index, group in enumerate(grouped.values()):
            left = np.asarray([item[0] for item in group], dtype=float)
            right = np.asarray([item[1] for item in group], dtype=float)
            for start in range(0, samples, 128):
                stop = min(samples, start + 128)
                indices = rng.integers(0, len(group), size=(stop - start, len(group)))
                left_boot[start:stop, task_index] = left[indices].mean(axis=1)
                right_boot[start:stop, task_index] = right[indices].mean(axis=1)
        model_draws.append(_aggregate_bootstrap_tasks(left_boot, dataset) - _aggregate_bootstrap_tasks(right_boot, dataset))
    return np.mean(np.vstack(model_draws), axis=0), float(np.mean(model_points))


def _aggregate_tasks(values: list[float], dataset: str) -> float:
    if dataset != "bbeh":
        return float(np.mean(values))
    if not values or any(value <= 0.0 for value in values):
        return 0.0
    return len(values) / sum(1.0 / value for value in values)


def _aggregate_bootstrap_tasks(values: np.ndarray, dataset: str) -> np.ndarray:
    if dataset != "bbeh":
        return values.mean(axis=1)
    valid = np.all(values > 0.0, axis=1)
    output = np.zeros(values.shape[0], dtype=float)
    output[valid] = values.shape[1] / np.sum(1.0 / values[valid], axis=1)
    return output


def _four_cell_gate(rows_by_model: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for model, rows in sorted(rows_by_model.items()):
        for dataset in PRIMARY_DATASETS:
            scores = {
                method: _method_accuracy(rows, dataset, method)
                for method in PRE_REGISTERED_COMPETITORS
            }
            strongest = sorted(scores, key=lambda method: (-scores[method], method))[0]
            rcta_score = _method_accuracy(rows, dataset, "rcta_1")
            cells.append(
                {
                    "dataset": dataset,
                    "model_name": model,
                    "strongest_competitor": strongest,
                    "rcta_accuracy": rcta_score,
                    "competitor_accuracy": scores[strongest],
                    "delta": rcta_score - scores[strongest],
                }
            )
    return {"passed": len(cells) == 4 and all(item["delta"] > 0.0 for item in cells), "cells": cells}


def _token_gate(rows_by_model: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    accuracy = {
        method: float(
            np.mean(
                [
                    _method_accuracy(rows, dataset, method)
                    for rows in rows_by_model.values()
                    for dataset in PRIMARY_DATASETS
                ]
            )
        )
        for method in PRE_REGISTERED_COMPETITORS
    }
    strongest = sorted(accuracy, key=lambda method: (-accuracy[method], method))[0]
    rcta_tokens = float(
        np.mean(
            [
                _method_tokens(rows, dataset, "rcta_1")
                for rows in rows_by_model.values()
                for dataset in PRIMARY_DATASETS
            ]
        )
    )
    competitor_tokens = float(
        np.mean(
            [
                _method_tokens(rows, dataset, strongest)
                for rows in rows_by_model.values()
                for dataset in PRIMARY_DATASETS
            ]
        )
    )
    return {
        "passed": rcta_tokens <= competitor_tokens,
        "rcta_equal_cell_mean_tokens": rcta_tokens,
        "strongest_accuracy_competitor": strongest,
        "strongest_competitor_equal_cell_accuracy": accuracy[strongest],
        "strongest_competitor_equal_cell_mean_tokens": competitor_tokens,
    }


def _coverage_gate(rows_by_model: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    rows = [
        row
        for model_rows in rows_by_model.values()
        for row in model_rows
        if row.get("dataset") in PRIMARY_DATASETS and row.get("method_name") == "rcta_1"
    ]
    corrected = sum(bool(row.get("corrected_by_debate")) for row in rows)
    harmed = sum(bool(row.get("harmed_by_debate")) for row in rows)
    decisive = corrected + harmed
    upper = clopper_pearson_upper(harmed, decisive, alpha=0.05)
    return {
        "passed": corrected > harmed and upper <= 1.0 / 3.0,
        "corrected": corrected,
        "harmed": harmed,
        "decisive": decisive,
        "harm_fraction_upper_95": upper,
    }


def _method_accuracy(rows: list[dict[str, Any]], dataset: str, method: str) -> float:
    items = [row for row in rows if row.get("dataset") == dataset and row.get("method_name") == method]
    if not items:
        raise ValueError(f"Missing {dataset}/{method} rows")
    if dataset != "bbeh":
        return float(np.mean([float(row.get("score") or 0.0) for row in items]))
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in items:
        grouped[str(row.get("task") or "unknown")].append(float(row.get("score") or 0.0))
    return _aggregate_tasks([float(np.mean(values)) for values in grouped.values()], dataset)


def _method_tokens(rows: list[dict[str, Any]], dataset: str, method: str) -> float:
    values = [
        float(row.get("total_tokens_per_question") or 0.0)
        for row in rows
        if row.get("dataset") == dataset and row.get("method_name") == method
    ]
    if not values:
        raise ValueError(f"Missing token rows for {dataset}/{method}")
    return float(np.mean(values))


def _holm(rows: list[dict[str, Any]], *, p_key: str) -> None:
    ordered = sorted(rows, key=lambda item: float(item[p_key]))
    running = 0.0
    total = len(ordered)
    for index, row in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * float(row[p_key])))
        row["holm_adjusted_p"] = running


def _hash_files(*paths: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _payload_hash(payload: dict[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key != "analysis_sha256"}
    return hashlib.sha256(
        json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze two validated RCTA full_seed42 runs.")
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = analyze_runs(args.run_dir, bootstrap_samples=args.bootstrap_samples)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
