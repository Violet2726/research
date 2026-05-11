"""family 级公共辅助原语。

这里只承接稳定、低层、无业务语义的重复逻辑，避免各实验线继续复制
phase split 解析、trace 哈希、问题预览和基础统计工具。
"""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Iterable

from research_experiments.core.foundation.config_helpers import SupportsRawPhases, phase_metadata


def resolve_phase_split_name(experiment: SupportsRawPhases, phase_name: str, benchmark_slug: str) -> str:
    """解析某个 benchmark 在指定 phase 下使用的冻结 split 名称。"""
    phase = phase_metadata(experiment, phase_name)
    if "split_overrides" in phase:
        return str(phase["split_overrides"][benchmark_slug])
    return str(phase["split_suffix"])


def build_question_preview(question: str, max_chars: int = 120) -> str:
    """生成稳定且适合日志展示的问题预览。"""
    cleaned = " ".join(question.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3] + "..."


def stable_trace_hash(rows: list[dict[str, Any]], keys: list[str]) -> str:
    """按给定字段集生成稳定 trace 哈希。"""
    payload = [{key: row.get(key) for key in keys} for row in rows]
    return sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def summarize_row_cost(rows: list[dict[str, Any]]) -> dict[str, float]:
    """汇总一组行记录中的 token 与时延成本。"""
    return {
        "prompt_tokens": round(sum_metric(rows, "prompt_tokens"), 6),
        "completion_tokens": round(sum_metric(rows, "completion_tokens"), 6),
        "total_tokens": round(sum_metric(rows, "total_tokens"), 6),
        "latency_ms": round(sum_metric(rows, "latency_ms"), 6),
    }


def sum_metric(rows: list[dict[str, Any]], key: str) -> float:
    """对一组行记录中的数值字段求和。"""
    return sum(float(row.get(key) or 0.0) for row in rows)


def safe_mean(values: Iterable[float]) -> float:
    """安全计算均值。"""
    materialized = list(values)
    if not materialized:
        return 0.0
    return round(sum(materialized) / len(materialized), 6)


def safe_ratio(numerator: int | float, denominator: int | float) -> float:
    """安全计算比例。"""
    if not denominator:
        return 0.0
    return round(float(numerator) / float(denominator), 6)
