"""共享的选择性通信信号与 trigger 规则。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ConfidenceSummary:
    """一组 agent confidence 的聚合统计。"""

    valid_confidences: list[float]
    invalid_agent_ids: list[int]
    any_invalid_confidence: bool
    mean_confidence: float | None
    confidence_spread: float | None


@dataclass(frozen=True)
class TriggerDecision:
    """一次 trigger 规则判断结果。"""

    triggered: bool
    decision_reason: str
    fail_open_applied: bool


def confidence_display(value: object) -> object:
    """保留 confidence 原始展示形式。"""
    if value is None:
        return ""
    return value


def normalize_confidence(raw_value: object) -> tuple[float | None, bool, str]:
    """把 confidence_raw 归一化到可比较的 ``[0, 1]``。"""
    if raw_value is None:
        return None, False, "missing"
    if isinstance(raw_value, bool):
        return None, False, "invalid_bool"
    if isinstance(raw_value, (int, float)):
        numeric_value = float(raw_value)
    else:
        text = str(raw_value).strip().rstrip("%")
        try:
            numeric_value = float(text)
        except ValueError:
            return None, False, "non_numeric"
    if 0.0 <= numeric_value <= 1.0:
        return round(numeric_value, 6), True, "unit_interval"
    if 1.0 < numeric_value <= 100.0:
        return round(numeric_value / 100.0, 6), True, "percent_scaled"
    return None, False, "out_of_range"


def summarize_confidence_rows(
    rows: list[dict[str, Any]],
    *,
    confidence_value_key: str = "confidence_value",
    confidence_valid_key: str = "confidence_valid",
    agent_id_key: str = "agent_id",
) -> ConfidenceSummary:
    """从 turn rows 汇总 mean / spread 与非法 confidence agent。"""
    valid_confidences = [
        float(row[confidence_value_key])
        for row in rows
        if row.get(confidence_valid_key) and row.get(confidence_value_key) is not None
    ]
    invalid_agent_ids = [
        int(row[agent_id_key])
        for row in rows
        if not row.get(confidence_valid_key)
    ]
    mean_confidence = round(sum(valid_confidences) / len(valid_confidences), 6) if valid_confidences else None
    if len(valid_confidences) >= 2:
        confidence_spread = round(max(valid_confidences) - min(valid_confidences), 6)
    elif len(valid_confidences) == 1:
        confidence_spread = 0.0
    else:
        confidence_spread = None
    return ConfidenceSummary(
        valid_confidences=valid_confidences,
        invalid_agent_ids=invalid_agent_ids,
        any_invalid_confidence=bool(invalid_agent_ids),
        mean_confidence=mean_confidence,
        confidence_spread=confidence_spread,
    )


def decide_trigger(
    *,
    trigger_type: str,
    initial_disagreement: bool,
    mean_confidence: float | None,
    confidence_spread: float | None,
    any_invalid_confidence: bool,
    mean_conf_threshold: float | None = None,
    conf_spread_threshold: float | None = None,
    fail_open_to_always: bool = True,
) -> TriggerDecision:
    """按统一规则判断是否触发通信。"""
    if trigger_type == "always_communicate":
        return TriggerDecision(True, "always_on", False)
    if trigger_type == "disagreement_triggered":
        return TriggerDecision(
            initial_disagreement,
            "answer_disagreement" if initial_disagreement else "early_exit_by_agreement",
            False,
        )
    if trigger_type == "confidence_triggered":
        if any_invalid_confidence and fail_open_to_always:
            return TriggerDecision(True, "invalid_confidence_fail_open", True)
        threshold = mean_conf_threshold if mean_conf_threshold is not None else 0.75
        if mean_confidence is None:
            return TriggerDecision(False, "missing_confidence_without_fail_open", False)
        triggered = mean_confidence < threshold
        return TriggerDecision(
            triggered,
            "low_mean_confidence" if triggered else "early_exit_by_high_confidence",
            False,
        )
    if trigger_type == "hybrid_trigger":
        if initial_disagreement:
            return TriggerDecision(True, "answer_disagreement", False)
        if any_invalid_confidence and fail_open_to_always:
            return TriggerDecision(True, "invalid_confidence_fail_open", True)
        threshold = mean_conf_threshold if mean_conf_threshold is not None else 0.75
        spread_threshold = conf_spread_threshold if conf_spread_threshold is not None else 0.20
        if mean_confidence is None:
            return TriggerDecision(False, "missing_confidence_without_fail_open", False)
        if mean_confidence < threshold:
            return TriggerDecision(True, "low_mean_confidence", False)
        if confidence_spread is not None and confidence_spread > spread_threshold:
            return TriggerDecision(True, "high_confidence_spread", False)
        return TriggerDecision(False, "early_exit_by_hybrid_rule", False)
    raise ValueError(f"Unsupported trigger type: {trigger_type}")


def decide_trigger_from_policy(
    policy: Any,
    *,
    initial_disagreement: bool,
    mean_confidence: float | None,
    confidence_spread: float | None,
    any_invalid_confidence: bool,
) -> TriggerDecision:
    """从策略对象读取参数后复用统一 trigger 规则。"""
    return decide_trigger(
        trigger_type=str(getattr(policy, "trigger_type")),
        initial_disagreement=initial_disagreement,
        mean_confidence=mean_confidence,
        confidence_spread=confidence_spread,
        any_invalid_confidence=any_invalid_confidence,
        mean_conf_threshold=getattr(policy, "mean_conf_threshold", None),
        conf_spread_threshold=getattr(policy, "conf_spread_threshold", None),
        fail_open_to_always=bool(getattr(policy, "fail_open_to_always", True)),
    )
