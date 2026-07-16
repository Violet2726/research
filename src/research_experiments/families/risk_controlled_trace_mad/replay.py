"""v5 止损门槛的历史 BBEH-300 轨迹保守回放。"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from research_experiments.core.config import load_benchmark_config
from research_experiments.core.data.datasets import DatasetSample, load_samples
from research_experiments.core.data.evaluation import canonicalize_answer, score_prediction
from research_experiments.families.risk_controlled_trace_mad.algorithms import (
    class_majority,
    homogeneous_stage_decision,
)
from research_experiments.families.risk_controlled_trace_mad.run.metrics import build_metrics


def replay_historical_development(source_run: str | Path) -> dict[str, Any]:
    """Replay only recorded calls; no provider or cache request is permitted."""

    root = Path(source_run)
    turns_path = root / "turns" / "agent_turns.jsonl"
    predictions_path = root / "views" / "predictions.jsonl"
    if not turns_path.exists() or not predictions_path.exists():
        raise FileNotFoundError(f"Historical replay artifacts are incomplete under {root}")
    sample_map = _load_bbeh_sample_map()
    metadata: dict[str, dict[str, Any]] = {}
    with predictions_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("dataset") != "bbeh":
                continue
            metadata.setdefault(
                str(row["sample_id"]),
                {
                    "gold": row.get("gold", ""),
                    "task": row.get("task") or "unknown",
                    "sample": sample_map.get(str(row["sample_id"])),
                },
            )
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    with turns_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("dataset") != "bbeh":
                continue
            method = str(row.get("method_name") or "")
            if method not in {
                "brd_stage_a_shared",
                "conditional_resample_3",
                "gsa_shared_panel",
                "hsgsa_stage_a_shared",
                "hsgsa_resample_shared",
                "hsgsa_blind_reviewer_shared",
            }:
                continue
            answer = _recover_legacy_answer(row)
            sample = sample_map.get(str(row.get("sample_id") or ""))
            canonical = canonicalize_answer(sample, answer) if sample is not None and answer else None
            row["normalized_answer"] = canonical.key if canonical is not None and canonical.valid else ""
            row["answer_class_key"] = row["normalized_answer"]
            row["canonicalization_status"] = "valid" if canonical is not None and canonical.valid else "invalid"
            row["canonicalization_invalid_reason"] = canonical.invalid_reason if canonical is not None else "missing_sample"
            grouped[str(row["sample_id"])][method].append(row)

    replay_rows: list[dict[str, Any]] = []
    invalid_answer_count = 0
    malformed_samples: list[str] = []
    for sample_id in sorted(metadata):
        pools = grouped.get(sample_id, {})
        sample = metadata[sample_id].get("sample")
        if not isinstance(sample, DatasetSample):
            malformed_samples.append(sample_id)
            continue
        stage_rows = sorted(
            pools.get("hsgsa_stage_a_shared") or pools.get("brd_stage_a_shared", []),
            key=lambda row: int(row.get("agent_id") or 0),
        )
        if len(stage_rows) != 5:
            malformed_samples.append(sample_id)
            continue
        stage = homogeneous_stage_decision(stage_rows, dataset="bbeh", seed=42, sample_id=sample_id)
        resamples = sorted(
            pools.get("hsgsa_resample_shared") or pools.get("conditional_resample_3", []),
            key=lambda row: int(row.get("agent_id") or 0),
        )
        invalid_answer_count += sum(not row.get("answer_class_key") for row in [*stage_rows, *resamples])
        adaptive_key, adaptive_answer, _, _ = class_majority(
            [*stage_rows, *resamples],
            dataset="bbeh",
            seed=42,
            sample_id=sample_id,
            purpose="historical_replay_adaptive_sc8",
            fallback_key=stage.anchor_key,
            fallback_answer=stage.anchor_answer,
        )
        gold = str(metadata[sample_id]["gold"])
        task = str(metadata[sample_id]["task"])
        initial_score = score_prediction("bbeh", stage.anchor_answer, gold, sample=sample)
        adaptive_score = score_prediction("bbeh", adaptive_answer, gold, sample=sample)
        replay_rows.extend(
            [
                _replay_prediction(
                    sample_id, task, "sc_5", initial_score, stage_rows, False,
                    initial_score, initial_score
                ),
                _replay_prediction(
                    sample_id, task, "adaptive_sc_8", adaptive_score, stage_rows + resamples, False,
                    initial_score, adaptive_score
                ),
            ]
        )
    metrics = build_metrics(
        replay_rows,
        dataset_order=["bbeh"],
        method_order=["sc_5", "adaptive_sc_8"],
        bbeh_harmonic=True,
    )
    summaries = {
        row["method_name"]: row
        for row in metrics["summary"]
        if row.get("dataset") == "bbeh"
    }
    sc5_harmonic = float(summaries.get("sc_5", {}).get("accuracy_mean") or 0.0)
    adaptive_harmonic = float(summaries.get("adaptive_sc_8", {}).get("accuracy_mean") or 0.0)
    sc5_accuracy = float(summaries.get("sc_5", {}).get("micro_accuracy") or 0.0)
    adaptive_accuracy = float(summaries.get("adaptive_sc_8", {}).get("micro_accuracy") or 0.0)
    delta = adaptive_accuracy - sc5_accuracy
    replay_failures = ["malformed_historical_stage_pool"] if malformed_samples else []
    # The old H-SGSA board was formed before sample-aware canonicalization.  A
    # SC5/adaptive-SC8 diagnostic can be replayed, but the discarded board can
    # neither be re-scored nor used as evidence for the old confirmation gate.
    # Keep the historical positive route locked even when the diagnostic itself
    # is mechanically complete.
    failures = [*replay_failures, "hsgsa_positive_route_unconfirmable"]
    return {
        "audit_kind": "sample_aware_historical_sc5_adaptive_replay",
        "source_run": str(root),
        "dataset": "bbeh",
        "sample_count": len(replay_rows) // 2,
        "network_calls_made": 0,
        "passed": False,
        "diagnostic_replay_completed": not replay_failures,
        "failures": failures,
        "development_thresholds": {
            "historical_claim": "diagnostic_only",
            "hsgsa_replay": "prohibited_due_to_precanonicalized_candidate_board",
        },
        "results": {
            "sc5_micro_accuracy": sc5_accuracy,
            "adaptive_sc8_micro_accuracy": adaptive_accuracy,
            "sc5_task_harmonic_accuracy": sc5_harmonic,
            "adaptive_sc8_task_harmonic_accuracy": adaptive_harmonic,
            "adaptive_sc8_minus_sc5": delta,
            "invalid_answer_count": invalid_answer_count,
            "malformed_sample_count": len(malformed_samples),
        },
        "limitations": [
            "Only recorded Stage-A and resample calls are replayed; this function makes no provider or cache request.",
            "H-SGSA candidate boards were grouped before sample-aware canonicalization and are explicitly not replayed.",
            "The output is a normalization-impact audit, never confirmation evidence.",
        ],
    }


def write_replay_audit(source_run: str | Path, output_path: str | Path) -> dict[str, Any]:
    payload = replay_historical_development(source_run)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _recover_legacy_answer(row: dict[str, Any]) -> str:
    validated = row.get("validated_output") or {}
    answer = str(validated.get("final_answer") or "").strip() if isinstance(validated, dict) else ""
    if answer:
        return answer
    match = re.search(r"(?im)^\s*FINAL_ANSWER\s*:\s*(.*?)\s*$", str(row.get("assistant_text") or ""))
    return str(match.group(1)).strip() if match else str(row.get("prediction") or "").strip()


def _load_bbeh_sample_map() -> dict[str, DatasetSample]:
    benchmark = load_benchmark_config("configs/core/shared/benchmarks/bbeh/bbeh-main.toml")
    return {sample.sample_id: sample for sample in load_samples(benchmark)}


def _replay_prediction(sample_id, task, method, score, rows, override, initial_score, final_score):
    return {
        "dataset": "bbeh",
        "sample_id": sample_id,
        "task": task,
        "method_name": method,
        "score": score,
        "total_tokens_per_question": sum(float(row.get("total_tokens") or 0) for row in rows),
        "logical_calls_per_question": len(rows),
        "override_accepted": override,
        "vote_flipped": override,
        "corrected_by_debate": initial_score < 1 and final_score == 1,
        "harmed_by_debate": initial_score == 1 and final_score < 1,
    }
