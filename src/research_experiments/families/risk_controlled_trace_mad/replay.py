"""v5 止损门槛的历史 BBEH-300 轨迹保守回放。"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from research_experiments.core.data.evaluation import answer_class_key, normalize_prediction, score_prediction
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
    metadata: dict[str, dict[str, Any]] = {}
    with predictions_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("dataset") != "bbeh":
                continue
            metadata.setdefault(
                str(row["sample_id"]),
                {"gold": row.get("gold", ""), "task": row.get("task") or "unknown"},
            )
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    with turns_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("dataset") != "bbeh":
                continue
            method = str(row.get("method_name") or "")
            if method not in {"brd_stage_a_shared", "conditional_resample_3", "gsa_shared_panel"}:
                continue
            answer = _recover_legacy_answer(row)
            row["normalized_answer"] = answer
            row["answer_class_key"] = answer_class_key("bbeh", answer) if answer else ""
            grouped[str(row["sample_id"])][method].append(row)

    replay_rows: list[dict[str, Any]] = []
    reviewer_total = reviewer_valid = 0
    override_count = corrected = harmed = 0
    malformed_samples: list[str] = []
    for sample_id in sorted(metadata):
        pools = grouped.get(sample_id, {})
        stage_rows = sorted(pools.get("brd_stage_a_shared", []), key=lambda row: int(row.get("agent_id") or 0))
        if len(stage_rows) != 5:
            malformed_samples.append(sample_id)
            continue
        stage = homogeneous_stage_decision(stage_rows, dataset="bbeh", seed=42, sample_id=sample_id)
        resamples = sorted(pools.get("conditional_resample_3", []), key=lambda row: int(row.get("agent_id") or 0))
        reviewers = sorted(pools.get("gsa_shared_panel", []), key=lambda row: int(row.get("agent_id") or 0))
        reviewer_total += len(reviewers)
        valid_reviewer_keys = []
        for row in reviewers:
            key = str(row.get("answer_class_key") or "")
            if key:
                reviewer_valid += 1
                valid_reviewer_keys.append(key)
        adaptive_key, adaptive_answer, _, _ = class_majority(
            [*stage_rows, *resamples],
            dataset="bbeh",
            seed=42,
            sample_id=sample_id,
            purpose="historical_replay_adaptive_sc8",
            fallback_key=stage.anchor_key,
            fallback_answer=stage.anchor_answer,
        )
        hsgsa_key = stage.anchor_key
        if (
            len(valid_reviewer_keys) == 3
            and len(set(valid_reviewer_keys)) == 1
            and valid_reviewer_keys[0] in stage.vote_counts
            and valid_reviewer_keys[0] != stage.anchor_key
        ):
            hsgsa_key = valid_reviewer_keys[0]
        hsgsa_answer = stage.answer_by_key.get(hsgsa_key, stage.anchor_answer)
        gold = str(metadata[sample_id]["gold"])
        task = str(metadata[sample_id]["task"])
        initial_score = score_prediction("bbeh", stage.anchor_answer, gold)
        hsgsa_score = score_prediction("bbeh", hsgsa_answer, gold)
        adaptive_score = score_prediction("bbeh", adaptive_answer, gold)
        override = hsgsa_key != stage.anchor_key
        override_count += int(override)
        corrected += int(override and initial_score < 1 and hsgsa_score == 1)
        harmed += int(override and initial_score == 1 and hsgsa_score < 1)
        replay_rows.extend(
            [
                _replay_prediction(
                    sample_id, task, "hsgsa_unanimous_3", hsgsa_score, stage_rows + reviewers, override,
                    initial_score, hsgsa_score
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
        method_order=["hsgsa_unanimous_3", "adaptive_sc_8"],
        bbeh_harmonic=True,
    )
    summaries = {
        row["method_name"]: row
        for row in metrics["summary"]
        if row.get("dataset") == "bbeh"
    }
    hsgsa_harmonic = float(summaries.get("hsgsa_unanimous_3", {}).get("accuracy_mean") or 0.0)
    adaptive_harmonic = float(summaries.get("adaptive_sc_8", {}).get("accuracy_mean") or 0.0)
    hsgsa_accuracy = float(summaries.get("hsgsa_unanimous_3", {}).get("micro_accuracy") or 0.0)
    adaptive_accuracy = float(summaries.get("adaptive_sc_8", {}).get("micro_accuracy") or 0.0)
    delta = hsgsa_accuracy - adaptive_accuracy
    parse_rate = reviewer_valid / reviewer_total if reviewer_total else 0.0
    failures = []
    if parse_rate < 0.995:
        failures.append("historical_reviewer_parse_rate_below_99_5_percent")
    if delta < 0.01:
        failures.append("historical_replay_hsgsa_lead_below_1pp")
    if corrected <= harmed:
        failures.append("historical_replay_nonpositive_net_correction")
    if malformed_samples:
        failures.append("malformed_historical_stage_pool")
    return {
        "audit_kind": "label_free_historical_trajectory_replay",
        "source_run": str(root),
        "dataset": "bbeh",
        "sample_count": len(replay_rows) // 2,
        "network_calls_made": 0,
        "passed": not failures,
        "failures": failures,
        "development_thresholds": {
            "reviewer_parse_rate_min": 0.995,
            "hsgsa_minus_adaptive_sc8_min": 0.01,
            "net_correction_positive": True,
        },
        "results": {
            "hsgsa_micro_accuracy": hsgsa_accuracy,
            "adaptive_sc8_micro_accuracy": adaptive_accuracy,
            "hsgsa_task_harmonic_accuracy": hsgsa_harmonic,
            "adaptive_sc8_task_harmonic_accuracy": adaptive_harmonic,
            "hsgsa_minus_adaptive_sc8": delta,
            "reviewer_parse_rate": parse_rate,
            "reviewer_valid": reviewer_valid,
            "reviewer_total": reviewer_total,
            "override_count": override_count,
            "corrected": corrected,
            "harmed": harmed,
            "malformed_sample_count": len(malformed_samples),
        },
        "limitations": [
            "The v5 PICK protocol cannot be counterfactually reconstructed from v2 text; legacy FINAL_ANSWER is used as the conservative reviewer pick proxy.",
            "Passing this replay would be necessary but not sufficient evidence for the new prompt. Failing it activates the pre-registered stop rule.",
            "No OmniMath row is used in the positive gate.",
        ],
    }


def write_replay_audit(source_run: str | Path, output_path: str | Path) -> dict[str, Any]:
    payload = replay_historical_development(source_run)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _recover_legacy_answer(row: dict[str, Any]) -> str:
    answer = str(row.get("normalized_answer") or row.get("prediction") or "").strip()
    if answer:
        return normalize_prediction("bbeh", answer)
    match = re.search(r"(?im)^\s*FINAL_ANSWER\s*:\s*(.*?)\s*$", str(row.get("assistant_text") or ""))
    return normalize_prediction("bbeh", match.group(1)) if match else ""


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
