"""跨 family 参考 run 的正式读取接口。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Any

from research_experiments.core.foundation.config import ResolvedModelConfig
from research_experiments.core.foundation.workspace import default_runs_root


@dataclass(frozen=True)
class TriggerReferenceConfig:
    """声明一个供其他 family 消费的 trigger 参考来源。"""

    source_family: str
    source_experiment: str
    source_phase: str
    default_policy: str
    fallback_policy: str
    drop_questions_threshold: float


@dataclass(frozen=True)
class TriggerReferenceDecision:
    """记录一次 trigger 参考选择的完整结果。"""

    selected_policy: str
    reason: str
    reference_run_dir: str | None
    reference_run_id: str | None
    target_model_name: str
    reference_model_name: str | None
    default_policy: str
    fallback_policy: str
    always_accuracy: float | None
    default_policy_accuracy: float | None
    drop_questions: float | None
    threshold_questions: float


def write_policy_reference_summary(
    run_dir: str | Path,
    *,
    manifest: dict[str, Any],
    metrics_payload: dict[str, Any],
) -> dict[str, Any]:
    """为选择性通信 run 生成稳定的机器可读参考摘要。"""

    root = Path(run_dir)
    resolved_model = manifest.get("resolved_model") or {}
    overall_rows = [
        row
        for row in metrics_payload.get("summary", [])
        if row.get("dataset") == "overall" and row.get("method_kind") == "policy"
    ]
    payload = {
        "kind": "policy_reference_summary",
        "family_name": str(manifest.get("family_name") or "selective_comm"),
        "experiment_name": str(manifest.get("experiment_name") or ""),
        "phase_name": str(manifest.get("phase_name") or ""),
        "run_id": str(manifest.get("run_id") or root.name),
        "run_dir": root.as_posix(),
        "model_name": str(resolved_model.get("name") or ""),
        "provider": str(resolved_model.get("provider") or ""),
        "model_id": str(resolved_model.get("model_id") or ""),
        "primary_model_ref": str(manifest.get("primary_model_ref") or ""),
        "overall_policies": {
            str(row["method_name"]): {
                "method_name": row["method_name"],
                "display_name": row.get("display_name"),
                "question_count": row.get("question_count"),
                "accuracy_mean": row.get("accuracy_mean"),
                "trigger_rate": row.get("trigger_rate"),
                "early_exit_rate": row.get("early_exit_rate"),
                "total_tokens_mean": row.get("total_tokens_mean"),
                "communication_tokens_mean": row.get("communication_tokens_mean"),
            }
            for row in overall_rows
        },
    }
    output_path = root / "policy_reference_summary.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def resolve_trigger_reference_selection(
    *,
    backbone: ResolvedModelConfig,
    reference: TriggerReferenceConfig,
) -> dict[str, Any]:
    """按显式 reference 配置选择一个可复用 trigger 策略。"""

    candidates = _collect_reference_summaries(reference)
    exact_match = _pick_best_summary(candidates, lambda row: row["model_name"] == backbone.name)
    prefix = backbone.model_id.split("-", 1)[0]
    family_match = _pick_best_summary(
        candidates,
        lambda row: row["provider"] == backbone.provider and str(row["model_id"]).split("-", 1)[0] == prefix,
    )
    selected = exact_match or family_match
    if selected is None:
        return asdict(
            TriggerReferenceDecision(
                selected_policy=reference.default_policy,
                reason="reference_missing",
                reference_run_dir=None,
                reference_run_id=None,
                target_model_name=backbone.name,
                reference_model_name=None,
                default_policy=reference.default_policy,
                fallback_policy=reference.fallback_policy,
                always_accuracy=None,
                default_policy_accuracy=None,
                drop_questions=None,
                threshold_questions=reference.drop_questions_threshold,
            )
        )

    overall_policies = selected.get("overall_policies", {})
    always_row = overall_policies.get("always_communicate")
    default_row = overall_policies.get(reference.default_policy)
    if always_row is None or default_row is None:
        return asdict(
            TriggerReferenceDecision(
                selected_policy=reference.fallback_policy,
                reason="reference_incomplete",
                reference_run_dir=str(selected.get("run_dir") or ""),
                reference_run_id=str(selected.get("run_id") or ""),
                target_model_name=backbone.name,
                reference_model_name=str(selected.get("model_name") or ""),
                default_policy=reference.default_policy,
                fallback_policy=reference.fallback_policy,
                always_accuracy=_optional_float(always_row, "accuracy_mean"),
                default_policy_accuracy=_optional_float(default_row, "accuracy_mean"),
                drop_questions=None,
                threshold_questions=reference.drop_questions_threshold,
            )
        )

    question_count = int(always_row.get("question_count") or 0)
    always_accuracy = float(always_row.get("accuracy_mean") or 0.0)
    default_accuracy = float(default_row.get("accuracy_mean") or 0.0)
    drop_questions = max(0.0, (always_accuracy - default_accuracy) * question_count)
    selected_policy = reference.default_policy
    reason = "default_policy_kept"
    if drop_questions > reference.drop_questions_threshold:
        selected_policy = reference.fallback_policy
        reason = "default_policy_drops_more_than_threshold"
    return asdict(
        TriggerReferenceDecision(
            selected_policy=selected_policy,
            reason=reason,
            reference_run_dir=str(selected.get("run_dir") or ""),
            reference_run_id=str(selected.get("run_id") or ""),
            target_model_name=backbone.name,
            reference_model_name=str(selected.get("model_name") or ""),
            default_policy=reference.default_policy,
            fallback_policy=reference.fallback_policy,
            always_accuracy=always_accuracy,
            default_policy_accuracy=default_accuracy,
            drop_questions=round(drop_questions, 6),
            threshold_questions=reference.drop_questions_threshold,
        )
    )


def _collect_reference_summaries(reference: TriggerReferenceConfig) -> list[dict[str, Any]]:
    root = Path(default_runs_root(reference.source_family)) / reference.source_experiment / reference.source_phase
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for path in root.rglob("policy_reference_summary.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        rows.append(payload)
    return rows


def _pick_best_summary(candidates: list[dict[str, Any]], predicate) -> dict[str, Any] | None:
    matched = [row for row in candidates if predicate(row)]
    if not matched:
        return None
    return sorted(matched, key=lambda row: str(row.get("run_id") or ""), reverse=True)[0]


def _optional_float(row: dict[str, Any] | None, key: str) -> float | None:
    if row is None:
        return None
    value = row.get(key)
    return float(value) if value is not None else None

