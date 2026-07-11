"""从 BRD pilot 到 locked confirmation 的可审计晋级门。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from research_experiments.core.io import read_json, read_jsonl

PRIMARY_DATASETS = ("omni_math_2_filtered", "bbeh")


def evaluate_pilot_gate(
    *,
    prediction_rows: list[dict[str, Any]],
    turn_rows: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    """Evaluate the fixed promotion gate without peeking at locked results."""

    brd_rows = [row for row in prediction_rows if row.get("method_name") == "brd_quorum_3" and row.get("dataset") in PRIMARY_DATASETS]
    per_dataset = {row["dataset"]: row for row in diagnostics.get("summary_rows", []) if row.get("method_name") == "brd_quorum_3" and row.get("dataset") in PRIMARY_DATASETS}
    oracle = {
        dataset: float((per_dataset.get(dataset) or {}).get("candidate_oracle_gap_over_anchor") or 0.0)
        for dataset in PRIMARY_DATASETS
    }
    paired_sc5 = _paired_delta(prediction_rows, "brd_quorum_3", "sc_5")
    paired_resample = _paired_delta(prediction_rows, "brd_quorum_3", "conditional_resample_3")
    overrides = [row for row in brd_rows if row.get("override_accepted")]
    correct_overrides = [row for row in overrides if float(row.get("score") or 0.0) == 1.0]
    corrected = sum(bool(row.get("corrected_by_debate")) for row in brd_rows)
    harmed = sum(bool(row.get("harmed_by_debate")) for row in brd_rows)
    request_failures = sum(bool(row.get("request_error")) or row.get("request_status") == "request_fail" for row in turn_rows)
    protocol_failures = sum(row.get("protocol_parse_status") == "failed" for row in turn_rows)
    conditions = {
        "zero_request_failures": request_failures == 0,
        "zero_protocol_failures": protocol_failures == 0,
        "oracle_gap_at_least_3pp_on_both_primary_sets": all(oracle[dataset] >= 0.03 for dataset in PRIMARY_DATASETS),
        "brd_has_positive_net_correction": corrected - harmed > 0,
        "brd_accuracy_positive_vs_sc5": paired_sc5["net_score_delta"] > 0,
        "brd_accuracy_positive_vs_conditional_resample": paired_resample["net_score_delta"] > 0,
        "at_least_20_overrides": len(overrides) >= 20,
        "override_precision_at_least_two_thirds": len(correct_overrides) / len(overrides) >= 2 / 3 if overrides else False,
    }
    return {
        "gate_name": "brd_mad_pilot_v1",
        "model_name": model_name,
        "primary_datasets": list(PRIMARY_DATASETS),
        "conditions": conditions,
        "passed": all(conditions.values()),
        "evidence": {
            "candidate_oracle_gap_over_anchor": oracle,
            "corrected": corrected,
            "harmed": harmed,
            "net_corrected": corrected - harmed,
            "override_count": len(overrides),
            "correct_override_count": len(correct_overrides),
            "override_precision": len(correct_overrides) / len(overrides) if overrides else 0.0,
            "paired_vs_sc5": paired_sc5,
            "paired_vs_conditional_resample": paired_resample,
            "request_failures": request_failures,
            "protocol_failures": protocol_failures,
        },
    }


def find_passing_pilot_gate(*, family_run_root: str | Path, model_name: str, experiment_name: str = "brd_mad_pilot") -> Path | None:
    """Find the newest successful pilot gate for exactly the requested backbone."""

    root = Path(family_run_root) / experiment_name / "pilot"
    candidates = sorted(root.glob("*/diagnostics/pilot_gate.json"), key=lambda path: path.stat().st_mtime, reverse=True) if root.exists() else []
    for path in candidates:
        payload = read_json(path)
        if payload.get("passed") and payload.get("model_name") == model_name:
            return path
    return None


def load_pilot_gate(path: str | Path) -> dict[str, Any]:
    return read_json(path)


def _paired_delta(rows: list[dict[str, Any]], reference: str, comparator: str) -> dict[str, Any]:
    keyed: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        dataset = str(row.get("dataset") or "")
        if dataset not in PRIMARY_DATASETS:
            continue
        method = str(row.get("method_name") or "")
        if method in {reference, comparator}:
            keyed[(dataset, str(row.get("sample_id") or ""))][method] = float(row.get("score") or 0.0)
    deltas = [values[reference] - values[comparator] for values in keyed.values() if reference in values and comparator in values]
    return {
        "paired_question_count": len(deltas),
        "net_score_delta": sum(deltas),
        "mean_accuracy_delta": sum(deltas) / len(deltas) if deltas else 0.0,
    }


def evaluate_pilot_gate_from_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    manifest = read_json(root / "manifest.json")
    return evaluate_pilot_gate(
        prediction_rows=read_jsonl(root / "views" / "predictions.jsonl"),
        turn_rows=read_jsonl(root / "turns" / "agent_turns.jsonl"),
        diagnostics=read_json(root / "diagnostics" / "brd_diagnostics.json"),
        model_name=str((manifest.get("resolved_model") or {}).get("name") or ""),
    )
