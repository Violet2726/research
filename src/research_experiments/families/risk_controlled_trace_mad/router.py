"""RCTA 的可复现双 Logistic 风险路由与有限样本阈值。"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import beta

from research_experiments.families.risk_controlled_trace_mad.algorithms import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    validate_feature_vector,
)

ROUTER_ARTIFACT_VERSION = "rcta_router_v1"


@dataclass(frozen=True)
class LogisticModel:
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float

    def probability(self, vector: dict[str, float]) -> float:
        validate_feature_vector(vector)
        values = np.asarray([float(vector[name]) for name in FEATURE_NAMES], dtype=float)
        normalized = (values - np.asarray(self.means)) / np.asarray(self.scales)
        logit = float(np.dot(normalized, np.asarray(self.coefficients)) + self.intercept)
        return _sigmoid(logit)


@dataclass(frozen=True)
class RiskRouter:
    gain_model: LogisticModel
    harm_model: LogisticModel
    threshold: float
    artifact_sha256: str

    def score(self, vector: dict[str, float], *, without_certificate: bool = False) -> dict[str, float | bool]:
        features = dict(vector)
        if without_certificate:
            for name in FEATURE_NAMES:
                if name.startswith("certificate_"):
                    features[name] = 0.0
            features["certificate_unsupported"] = 1.0
        gain = self.gain_model.probability(features)
        harm = self.harm_model.probability(features)
        risk_score = gain - harm
        return {"p_gain": gain, "p_harm": harm, "risk_score": risk_score, "accept": risk_score >= self.threshold}


def fit_logistic(vectors: list[dict[str, float]], labels: list[int], *, l2: float = 1.0) -> LogisticModel:
    if not vectors or len(vectors) != len(labels):
        raise ValueError("vectors and labels must be non-empty and aligned")
    for vector in vectors:
        validate_feature_vector(vector)
    matrix = np.asarray([[float(row[name]) for name in FEATURE_NAMES] for row in vectors], dtype=float)
    target = np.asarray(labels, dtype=float)
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales[scales < 1e-12] = 1.0
    normalized = (matrix - means) / scales
    if np.all(target == target[0]):
        prevalence = min(1.0 - 1e-6, max(1e-6, float(target[0])))
        return LogisticModel(tuple(means), tuple(scales), tuple(np.zeros(matrix.shape[1])), math.log(prevalence / (1.0 - prevalence)))

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        weights = parameters[:-1]
        intercept = parameters[-1]
        logits = np.clip(normalized @ weights + intercept, -40.0, 40.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        eps = 1e-12
        loss = -np.sum(target * np.log(probabilities + eps) + (1.0 - target) * np.log(1.0 - probabilities + eps))
        loss += 0.5 * l2 * float(weights @ weights)
        residual = probabilities - target
        gradient = np.concatenate([normalized.T @ residual + l2 * weights, [residual.sum()]])
        return float(loss), gradient

    result = minimize(lambda value: objective(value), np.zeros(matrix.shape[1] + 1), jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"Logistic fit failed: {result.message}")
    return LogisticModel(tuple(means), tuple(scales), tuple(result.x[:-1]), float(result.x[-1]))


def crossfit_scores(records: list[dict[str, Any]], *, folds: int = 5) -> list[dict[str, Any]]:
    if folds < 2:
        raise ValueError("folds must be at least 2")
    assignments = _stratified_folds(records, folds)
    scored: list[dict[str, Any]] = []
    for fold in range(folds):
        training = [record for index, record in enumerate(records) if assignments[index] != fold]
        testing = [record for index, record in enumerate(records) if assignments[index] == fold]
        if not training or not testing:
            raise ValueError("every crossfit fold must have train and test records")
        gain_model = fit_logistic([item["feature_vector"] for item in training], [int(item["gain_label"]) for item in training])
        harm_model = fit_logistic([item["feature_vector"] for item in training], [int(item["harm_label"]) for item in training])
        for item in testing:
            gain = gain_model.probability(item["feature_vector"])
            harm = harm_model.probability(item["feature_vector"])
            scored.append({**item, "fold": fold, "p_gain": gain, "p_harm": harm, "risk_score": gain - harm})
    return sorted(scored, key=lambda item: (str(item["dataset"]), str(item["model_name"]), str(item["sample_id"])))


def choose_global_threshold(scored: list[dict[str, Any]], *, risk_limit: float = 1.0 / 3.0, delta: float = 0.05) -> dict[str, Any]:
    if not scored:
        raise ValueError("scored records are required")
    scores = np.asarray([float(item["risk_score"]) for item in scored], dtype=float)
    quantiles = [round(0.50 + 0.025 * index, 3) for index in range(19)]
    thresholds = sorted({float(np.quantile(scores, quantile)) for quantile in quantiles})
    alpha = delta / 19.0
    candidates: list[dict[str, Any]] = []
    for threshold in thresholds:
        accepted = [item for item in scored if float(item["risk_score"]) >= threshold]
        corrected = sum(int(item["gain_label"]) for item in accepted)
        harmed = sum(int(item["harm_label"]) for item in accepted)
        decisive = corrected + harmed
        upper = clopper_pearson_upper(harmed, decisive, alpha=alpha)
        cell_net: dict[str, int] = defaultdict(int)
        for item in accepted:
            cell_net[f"{item['dataset']}::{item['model_name']}"] += int(item["gain_label"]) - int(item["harm_label"])
        feasible = decisive >= 50 and upper <= risk_limit and len(cell_net) == 4 and all(value > 0 for value in cell_net.values())
        candidates.append({
            "threshold": threshold,
            "accepted": len(accepted),
            "corrected": corrected,
            "harmed": harmed,
            "decisive": decisive,
            "net_gain": corrected - harmed,
            "harm_fraction_upper": upper,
            "cell_net_gain": dict(sorted(cell_net.items())),
            "feasible": feasible,
        })
    feasible = [item for item in candidates if item["feasible"]]
    selected = sorted(feasible, key=lambda item: (-int(item["net_gain"]), -int(item["accepted"]), float(item["threshold"])))[0] if feasible else None
    return {"selected": selected, "candidates": candidates, "risk_limit": risk_limit, "delta": delta, "bonferroni_alpha": alpha}


def clopper_pearson_upper(harmed: int, decisive: int, *, alpha: float) -> float:
    if decisive <= 0:
        return 1.0
    if harmed >= decisive:
        return 1.0
    return float(beta.ppf(1.0 - alpha, harmed + 1, decisive - harmed))


def build_router_artifact(
    records: list[dict[str, Any]],
    *,
    input_run_hashes: list[str],
    risk_limit: float = 1.0 / 3.0,
    delta: float = 0.05,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    scored = crossfit_scores(records)
    threshold_result = choose_global_threshold(scored, risk_limit=risk_limit, delta=delta)
    selected = threshold_result["selected"]
    development_gate = evaluate_development_gate(
        scored,
        selected,
        bootstrap_samples=bootstrap_samples,
    )
    gain_model = fit_logistic([item["feature_vector"] for item in records], [int(item["gain_label"]) for item in records])
    harm_model = fit_logistic([item["feature_vector"] for item in records], [int(item["harm_label"]) for item in records])
    artifact = {
        "artifact_version": ROUTER_ARTIFACT_VERSION,
        "feature_version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "gain_model": _model_payload(gain_model),
        "harm_model": _model_payload(harm_model),
        "global_threshold": float(selected["threshold"]) if selected else None,
        "development_gate_passed": development_gate["passed"],
        "development_gate": development_gate,
        "threshold_diagnostics": threshold_result,
        "training_id_hash": _training_id_hash(records),
        "input_run_hashes": sorted(input_run_hashes),
        "record_count": len(records),
        "fold_count": 5,
        "leave_one_backbone_out": leave_one_backbone_out(records),
    }
    artifact["artifact_sha256"] = artifact_hash(artifact)
    return artifact


def evaluate_development_gate(
    scored: list[dict[str, Any]],
    selected: dict[str, Any] | None,
    *,
    bootstrap_samples: int = 10_000,
) -> dict[str, Any]:
    """Evaluate every preregistered count300 condition that is available at fit time."""
    if selected is None:
        return {
            "passed": False,
            "reason": "no_threshold_satisfies_preregistered_risk_constraints",
            "threshold_gate_passed": False,
        }

    threshold = float(selected["threshold"])
    evaluated = [
        {
            **item,
            "rcta_score": (
                float(item["synthesis_score"])
                if float(item["risk_score"]) >= threshold
                else float(item["anchor_score"])
            ),
        }
        for item in scored
    ]
    required_competitors = ("sc_9", "gsa_trace_1")
    cell_deltas: dict[str, dict[str, float]] = {}
    cells = sorted({(str(item["dataset"]), str(item["model_name"])) for item in evaluated})
    for dataset, model_name in cells:
        cell_rows = [
            item
            for item in evaluated
            if str(item["dataset"]) == dataset and str(item["model_name"]) == model_name
        ]
        cell_deltas[f"{dataset}::{model_name}"] = {
            competitor: float(
                np.mean(
                    [
                        float(item["rcta_score"]) - float(item[f"{_competitor_prefix(competitor)}_score"])
                        for item in cell_rows
                    ]
                )
            )
            for competitor in required_competitors
        }
    four_cell_positive = len(cell_deltas) == 4 and all(
        delta_value > 0.0
        for comparisons in cell_deltas.values()
        for delta_value in comparisons.values()
    )

    cross_backbone: list[dict[str, Any]] = []
    for dataset in sorted({str(item["dataset"]) for item in evaluated}):
        dataset_rows = [item for item in evaluated if str(item["dataset"]) == dataset]
        for competitor in required_competitors:
            comparison_key = f"{_competitor_prefix(competitor)}_score"
            low, high, point = _equal_backbone_bootstrap(
                dataset_rows,
                comparison_key=comparison_key,
                samples=bootstrap_samples,
                seed=f"rcta_count300:{dataset}:{competitor}",
            )
            cross_backbone.append(
                {
                    "dataset": dataset,
                    "comparison_method": competitor,
                    "mean_accuracy_delta": point,
                    "bootstrap_ci_95": [low, high],
                    "passed": low > 0.0,
                }
            )
    cross_backbone_passed = len(cross_backbone) == 4 and all(item["passed"] for item in cross_backbone)
    token_check = _token_gate(evaluated)
    passed = bool(
        selected["feasible"]
        and four_cell_positive
        and cross_backbone_passed
        and token_check["passed"]
    )
    return {
        "passed": passed,
        "threshold_gate_passed": bool(selected["feasible"]),
        "accepted_decisive_flips": int(selected["decisive"]),
        "risk_upper_bound": float(selected["harm_fraction_upper"]),
        "four_cell_comparisons": cell_deltas,
        "four_cell_positive": four_cell_positive,
        "cross_backbone_bootstrap_samples": bootstrap_samples,
        "cross_backbone_comparisons": cross_backbone,
        "cross_backbone_ci_passed": cross_backbone_passed,
        "token_gate": token_check,
    }


def load_router(path: str | Path, *, require_passing_gate: bool = True) -> RiskRouter:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = str(payload.get("artifact_sha256") or "")
    if expected != artifact_hash(payload):
        raise ValueError("RCTA router artifact hash mismatch")
    if payload.get("feature_version") != FEATURE_VERSION or tuple(payload.get("feature_names") or []) != FEATURE_NAMES:
        raise ValueError("RCTA router feature contract mismatch")
    if require_passing_gate and payload.get("development_gate_passed") is not True:
        raise RuntimeError("RCTA full run blocked: the frozen count300 development gate did not pass.")
    threshold = payload.get("global_threshold")
    if threshold is None:
        raise RuntimeError("RCTA router has no feasible global threshold.")
    return RiskRouter(_model_from_payload(payload["gain_model"]), _model_from_payload(payload["harm_model"]), float(threshold), expected)


def leave_one_backbone_out(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    models = sorted({str(item["model_name"]) for item in records})
    for held_out in models:
        train = [item for item in records if str(item["model_name"]) != held_out]
        test = [item for item in records if str(item["model_name"]) == held_out]
        if not train or not test:
            continue
        gain = fit_logistic([item["feature_vector"] for item in train], [int(item["gain_label"]) for item in train])
        harm = fit_logistic([item["feature_vector"] for item in train], [int(item["harm_label"]) for item in train])
        deltas = [gain.probability(item["feature_vector"]) - harm.probability(item["feature_vector"]) for item in test]
        output.append({"held_out_model": held_out, "record_count": len(test), "mean_risk_score": float(np.mean(deltas))})
    return output


def _competitor_prefix(method_name: str) -> str:
    mapping = {"sc_9": "sc9", "gsa_trace_1": "gsa"}
    try:
        return mapping[method_name]
    except KeyError as exc:
        raise ValueError(f"Unsupported preregistered gate competitor: {method_name}") from exc


def _equal_backbone_bootstrap(
    rows: list[dict[str, Any]],
    *,
    comparison_key: str,
    samples: int,
    seed: str,
) -> tuple[float, float, float]:
    if samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    by_model: dict[str, list[float]] = defaultdict(list)
    for item in rows:
        by_model[str(item["model_name"])].append(
            float(item["rcta_score"]) - float(item[comparison_key])
        )
    if len(by_model) != 2 or any(not values for values in by_model.values()):
        raise ValueError("Cross-backbone gate requires two non-empty backbone strata")
    rng = np.random.default_rng(int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16))
    draws = np.empty(samples, dtype=float)
    ordered = [np.asarray(by_model[name], dtype=float) for name in sorted(by_model)]
    for index in range(samples):
        stratum_means = [float(np.mean(values[rng.integers(0, len(values), len(values))])) for values in ordered]
        draws[index] = float(np.mean(stratum_means))
    low, high = np.quantile(draws, [0.025, 0.975])
    point = float(np.mean([float(np.mean(values)) for values in ordered]))
    return float(low), float(high), point


def _token_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    competitor_names = sorted(
        {
            name
            for item in rows
            for name in (item.get("competitors") or {})
        }
    )
    if not competitor_names:
        return {"passed": False, "reason": "missing_competitor_token_records"}

    cells = sorted({(str(item["dataset"]), str(item["model_name"])) for item in rows})
    rcta_tokens_by_cell: list[float] = []
    competitor_summaries: list[dict[str, Any]] = []
    for dataset, model_name in cells:
        cell_rows = [
            item
            for item in rows
            if str(item["dataset"]) == dataset and str(item["model_name"]) == model_name
        ]
        rcta_tokens_by_cell.append(float(np.mean([float(item["rcta_tokens"]) for item in cell_rows])))
    for method_name in competitor_names:
        accuracy_by_cell: list[float] = []
        tokens_by_cell: list[float] = []
        complete = True
        for dataset, model_name in cells:
            cell_rows = [
                item
                for item in rows
                if str(item["dataset"]) == dataset and str(item["model_name"]) == model_name
            ]
            method_rows = [(item.get("competitors") or {}).get(method_name) for item in cell_rows]
            if any(not isinstance(value, dict) for value in method_rows):
                complete = False
                break
            accuracy_by_cell.append(float(np.mean([float(value["score"]) for value in method_rows])))
            tokens_by_cell.append(float(np.mean([float(value["tokens"]) for value in method_rows])))
        if complete:
            competitor_summaries.append(
                {
                    "method_name": method_name,
                    "equal_cell_accuracy": float(np.mean(accuracy_by_cell)),
                    "equal_cell_mean_tokens": float(np.mean(tokens_by_cell)),
                }
            )
    if not competitor_summaries:
        return {"passed": False, "reason": "no_complete_competitor_token_records"}
    strongest = sorted(
        competitor_summaries,
        key=lambda item: (-float(item["equal_cell_accuracy"]), str(item["method_name"])),
    )[0]
    rcta_mean_tokens = float(np.mean(rcta_tokens_by_cell))
    return {
        "passed": rcta_mean_tokens <= float(strongest["equal_cell_mean_tokens"]),
        "rcta_equal_cell_mean_tokens": rcta_mean_tokens,
        "strongest_accuracy_competitor": strongest,
        "competitors": competitor_summaries,
        "scope": "disagreement-triggered records; conservative for RCTA token use",
    }


def artifact_hash(payload: dict[str, Any]) -> str:
    clean = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    encoded = json.dumps(clean, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _stratified_folds(records: list[dict[str, Any]], folds: int) -> list[int]:
    assignments = [-1] * len(records)
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, item in enumerate(records):
        groups[(str(item["dataset"]), str(item["sample_id"]))].append(index)
    by_dataset: dict[str, list[tuple[str, list[int]]]] = defaultdict(list)
    for (dataset, sample_id), indices in groups.items():
        by_dataset[dataset].append((sample_id, indices))
    for dataset, sample_groups in sorted(by_dataset.items()):
        ordered = sorted(
            sample_groups,
            key=lambda item: hashlib.sha256(f"{dataset}:{item[0]}".encode()).hexdigest(),
        )
        for offset, (_, indices) in enumerate(ordered):
            fold = offset % folds
            for index in indices:
                assignments[index] = fold
    if any(value < 0 for value in assignments):
        raise RuntimeError("failed to assign every router record to a crossfit fold")
    return assignments


def _training_id_hash(records: list[dict[str, Any]]) -> str:
    values = sorted(f"{item['dataset']}::{item['model_name']}::{item['sample_id']}" for item in records)
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def _model_payload(model: LogisticModel) -> dict[str, Any]:
    return {"means": list(model.means), "scales": list(model.scales), "coefficients": list(model.coefficients), "intercept": model.intercept, "l2": 1.0}


def _model_from_payload(payload: dict[str, Any]) -> LogisticModel:
    expected = len(FEATURE_NAMES)
    values = [payload.get("means"), payload.get("scales"), payload.get("coefficients")]
    if any(not isinstance(value, list) or len(value) != expected for value in values):
        raise ValueError("Malformed RCTA logistic model")
    return LogisticModel(tuple(map(float, values[0])), tuple(map(float, values[1])), tuple(map(float, values[2])), float(payload["intercept"]))


def _sigmoid(value: float) -> float:
    if value >= 0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)
