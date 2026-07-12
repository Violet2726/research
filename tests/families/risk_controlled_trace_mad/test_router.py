from __future__ import annotations

import json

import pytest

from research_experiments.families.risk_controlled_trace_mad.algorithms import FEATURE_NAMES
from research_experiments.families.risk_controlled_trace_mad.router import (
    artifact_hash,
    build_router_artifact,
    clopper_pearson_upper,
    crossfit_scores,
    load_router,
)


def _vector(value: float) -> dict[str, float]:
    return {name: (value if index < 9 else float(name == "certificate_pass")) for index, name in enumerate(FEATURE_NAMES)}


def _records() -> list[dict]:
    rows = []
    for dataset in ("omni_math_2_filtered", "bbeh"):
        for model in ("qwen", "mimo"):
            for index in range(80):
                gain = index < 20
                harm = 20 <= index < 30
                anchor_score = float(not gain)
                synthesis_score = float(gain or not harm)
                rows.append(
                    {
                        "dataset": dataset,
                        "model_name": model,
                        "sample_id": f"{dataset}-{model}-{index}",
                        "feature_vector": _vector(0.9 if gain else 0.1),
                        "gain_label": int(gain),
                        "harm_label": int(harm),
                        "anchor_score": anchor_score,
                        "synthesis_score": synthesis_score,
                        "sc9_score": anchor_score,
                        "gsa_score": synthesis_score,
                        "rcta_tokens": 6.0,
                        "competitors": {
                            "sc_9": {"score": anchor_score, "tokens": 9.0},
                            "gsa_trace_1": {"score": synthesis_score, "tokens": 7.0},
                        },
                    }
                )
    return rows


def test_router_artifact_is_deterministic_global_and_hash_checked(tmp_path) -> None:
    first = build_router_artifact(_records(), input_run_hashes=["b", "a"], bootstrap_samples=500)
    second = build_router_artifact(_records(), input_run_hashes=["a", "b"], bootstrap_samples=500)
    assert first == second
    assert first["artifact_sha256"] == artifact_hash(first)
    assert first["global_threshold"] is not None
    assert first["development_gate_passed"] is True
    path = tmp_path / "router.json"
    path.write_text(json.dumps(first), encoding="utf-8")
    router = load_router(path)
    assert isinstance(router.score(_vector(0.9))["accept"], bool)
    payload = json.loads(path.read_text())
    payload["global_threshold"] += 0.1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        load_router(path)


def test_clopper_pearson_is_conservative() -> None:
    assert clopper_pearson_upper(0, 50, alpha=0.05 / 19) < 1 / 3
    assert clopper_pearson_upper(0, 0, alpha=0.05 / 19) == 1.0


def test_crossfit_keeps_same_dataset_sample_out_of_training_across_backbones() -> None:
    records = _records()
    for row in records:
        row["sample_id"] = row["sample_id"].replace(f"-{row['model_name']}-", "-")
    scored = crossfit_scores(records)
    folds_by_id: dict[tuple[str, str], set[int]] = {}
    for row in scored:
        folds_by_id.setdefault((row["dataset"], row["sample_id"]), set()).add(row["fold"])
    assert all(len(folds) == 1 for folds in folds_by_id.values())
