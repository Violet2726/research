"""历史 CATCH-v1/v2 轨迹的零网络规范化重放。"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from research_experiments.core.data.datasets import DatasetSample, select_samples
from research_experiments.core.data.evaluation import score_prediction
from research_experiments.families.contrastive_active_testing.algorithms import build_stage_decision
from research_experiments.family_runtime.free_text_protocol import parse_sample_answer_output


def replay_canonicalization(
    run_dir: str | Path | list[str | Path] | tuple[str | Path, ...],
    *,
    samples: list[DatasetSample],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Rejudge only archived Stage-A/resample text with current sample contracts."""

    roots = [Path(value) for value in run_dir] if isinstance(run_dir, (list, tuple)) else [Path(run_dir)]
    sample_by_id = {sample.sample_id: sample for sample in samples}
    source_rows: list[dict[str, Any]] = []
    source_artifact_hashes: dict[str, str] = {}
    for root in roots:
        found = False
        for filename in ("agent_turns.jsonl", "preflight_turns.jsonl"):
            turns_path = root / "turns" / filename
            if not turns_path.exists():
                continue
            found = True
            source_artifact_hashes[f"{root.as_posix()}::{filename}"] = hashlib.sha256(
                turns_path.read_bytes()
            ).hexdigest()
            source_rows.extend(
                {**json.loads(line), "_replay_source": root.as_posix()}
                for line in turns_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        if not found:
            raise FileNotFoundError(f"Historical turns are unavailable under {root / 'turns'}.")
    relevant_rows = [
        row
        for row in source_rows
        if row.get("role") in {"stage_a_solver", "independent_resample"}
        and row.get("sample_id") in sample_by_id
    ]
    relevant_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    duplicate_sources: list[dict[str, Any]] = []
    source_conflicts: list[dict[str, Any]] = []
    for row in relevant_rows:
        key = (str(row["sample_id"]), str(row["role"]), int(row.get("agent_id") or 0))
        previous = relevant_by_key.get(key)
        if previous is None:
            relevant_by_key[key] = row
            continue
        same = str(previous.get("assistant_text") or "") == str(row.get("assistant_text") or "")
        record = {
            "sample_id": key[0],
            "role": key[1],
            "agent_id": key[2],
            "first_source": previous.get("_replay_source"),
            "second_source": row.get("_replay_source"),
        }
        (duplicate_sources if same else source_conflicts).append(record)
    relevant = list(relevant_by_key.values())
    replayed: list[dict[str, Any]] = []
    changed: list[dict[str, Any]] = []
    invalid_counts: dict[str, int] = defaultdict(int)
    for row in relevant:
        sample = sample_by_id[str(row["sample_id"])]
        parsed = parse_sample_answer_output(sample, str(row.get("assistant_text") or ""))
        key = str(parsed.get("canonical_key") or "") if parsed.get("canonical_valid") else ""
        reason = str(parsed.get("canonical_invalid_reason") or "") if not key else None
        current = {
            **row,
            "answer_class_key": key,
            "normalized_answer": key,
            "prediction": key,
            "canonicalization_invalid_reason": reason,
            "validated_output": parsed,
        }
        replayed.append(current)
        old_key = str(row.get("answer_class_key") or row.get("normalized_answer") or "")
        old_reason = row.get("canonicalization_invalid_reason")
        if old_key != key or (not key and old_reason != reason):
            changed.append(
                {
                    "sample_id": sample.sample_id,
                    "task": sample.metadata.get("task"),
                    "role": row.get("role"),
                    "agent_id": row.get("agent_id"),
                    "old_key": old_key,
                    "new_key": key,
                    "old_invalid_reason": old_reason,
                    "new_invalid_reason": reason,
                    "assistant_text_sha256": _sha256(str(row.get("assistant_text") or "")),
                }
            )
        if reason:
            invalid_counts[reason] += 1

    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in replayed:
        by_sample[str(row["sample_id"])].append(row)
    per_sample: list[dict[str, Any]] = []
    for sequence_index, sample in enumerate(samples):
        rows = by_sample.get(sample.sample_id, [])
        stage_rows = [row for row in rows if row.get("role") == "stage_a_solver"]
        resample_rows = [row for row in rows if row.get("role") == "independent_resample"]
        stage = build_stage_decision(stage_rows, seed=42, sample_id=sample.sample_id)
        adaptive = build_stage_decision([*stage_rows, *resample_rows], seed=42, sample_id=sample.sample_id)
        sc_answer = stage.anchor_answer
        adaptive_answer = adaptive.anchor_answer or sc_answer
        candidate_oracle = any(_correct(sample, item.answer) for item in stage.candidates)
        target = stage.candidates[:3]
        target_oracle = any(_correct(sample, item.answer) for item in target)
        per_sample.append(
            {
                "sample_sequence_index": sequence_index,
                "sample_id": sample.sample_id,
                "task": sample.metadata.get("task"),
                "stage_valid_count": stage.valid_count,
                "stage_candidate_keys": [item.key for item in stage.candidates],
                "stage_vote_counts": stage.vote_counts,
                "triggered": stage.triggered,
                "sc5_prediction": sc_answer,
                "sc5_correct": _correct(sample, sc_answer),
                "adaptive_sc8_prediction": adaptive_answer,
                "adaptive_sc8_correct": _correct(sample, adaptive_answer),
                "candidate_oracle_correct": candidate_oracle,
                "target_keys": [item.key for item in target],
                "target_oracle_correct": target_oracle,
            }
        )

    count = len(per_sample)
    metrics = {
        "sample_count": count,
        "sc5_micro": _rate(sum(row["sc5_correct"] for row in per_sample), count),
        "adaptive_sc8_micro": _rate(sum(row["adaptive_sc8_correct"] for row in per_sample), count),
        "candidate_oracle_micro": _rate(sum(row["candidate_oracle_correct"] for row in per_sample), count),
        "target_oracle_micro": _rate(sum(row["target_oracle_correct"] for row in per_sample), count),
        "disagreement_count": sum(row["triggered"] for row in per_sample),
    }
    feasibility = {
        "candidate_oracle_minus_sc5_at_least_5pp": metrics["candidate_oracle_micro"] - metrics["sc5_micro"] >= 0.05,
        "target_oracle_minus_sc5_at_least_8pp": metrics["target_oracle_micro"] - metrics["sc5_micro"] >= 0.08,
        "target_oracle_minus_adaptive_at_least_5pp": metrics["target_oracle_micro"] - metrics["adaptive_sc8_micro"] >= 0.05,
    }
    old_prediction_hashes = {
        root.as_posix(): hashlib.sha256((root / "views" / "predictions.jsonl").read_bytes()).hexdigest()
        for root in roots
        if (root / "views" / "predictions.jsonl").exists()
    }
    new_canonical = json.dumps(per_sample, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    payload = {
        "replay_version": "catch_v3_canonicalization_replay_v1",
        "source_runs": [root.as_posix() for root in roots],
        "generated_at": datetime.now(UTC).isoformat(),
        "network_requests": 0,
        "read_only_source": True,
        "metrics": metrics,
        "feasibility_conditions": feasibility,
        "passed": all(feasibility.values()),
        "source_resolution_policy": (
            "primary_v1_trajectory_first; later v2 duplicates are audited but never replace a frozen logical turn"
        ),
        "duplicate_source_turn_count": len(duplicate_sources),
        "duplicate_source_turns": duplicate_sources,
        "source_conflict_count": len(source_conflicts),
        "source_conflicts": source_conflicts,
        "changed_turn_count": len(changed),
        "changed_turns": changed,
        "invalid_reason_counts": dict(sorted(invalid_counts.items())),
        "class_merges": _class_merges(changed),
        "per_sample": per_sample,
        "hashes": {
            "source_artifact_sha256": source_artifact_hashes,
            "old_predictions_sha256": old_prediction_hashes,
            "new_replay_sha256": _sha256(new_canonical),
        },
        "legacy_method_replayability": {
            "catch_v1": "faithful_as_rejudged_sc5_only_because_observed_override_count_was_zero",
            "direct_judge_3": "not_faithfully_replayable_after_candidate_class_changes",
            "catch_codebook": "not_faithfully_replayable_after_candidate_class_changes",
        },
    }
    if output_path is not None:
        target_path = Path(output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def replay_from_experiment(
    run_dir: str | Path | list[str | Path] | tuple[str | Path, ...],
    experiment,
    *,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    benchmark = next(item for item in experiment_benchmarks(experiment) if item.slug == "bbeh")
    split_name = str(experiment.raw["phases"]["development"]["split_overrides"]["bbeh"])
    samples = select_samples(benchmark, split_name)
    return replay_canonicalization(run_dir, samples=samples, output_path=output_path)


def experiment_benchmarks(experiment):
    # Local import avoids coupling the pure replay primitive to TOML loading.
    from research_experiments.families.contrastive_active_testing.config import load_phase_benchmarks

    return load_phase_benchmarks(experiment, "development")


def _correct(sample: DatasetSample, answer: str) -> bool:
    return bool(answer) and score_prediction(
        sample.dataset, answer, sample.reference_answer, sample=sample
    ) == 1.0


def _class_merges(changed: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"old_key": old, "new_key": new}
        for old, new in sorted(
            {
                (str(row["old_key"]), str(row["new_key"]))
                for row in changed
                if row.get("old_key") and row.get("new_key") and row["old_key"] != row["new_key"]
            }
        )
    ]


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
