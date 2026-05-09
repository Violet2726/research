"""报告层使用的显式行视图。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class SummaryRowView:
    """对 `metrics.summary` 行提供稳定的字段访问接口。"""

    raw: Mapping[str, Any]
    dataset: str
    method_name: str
    display_name: str

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> SummaryRowView:
        dataset = str(row.get("dataset") or "")
        method_name = str(row.get("method_name") or row.get("policy_name") or "unknown")
        display_name = str(row.get("display_name") or method_name)
        return cls(raw=row, dataset=dataset, method_name=method_name, display_name=display_name)

    def text(self, field: str, default: str = "") -> str:
        value = self.raw.get(field, default)
        if value is None:
            return default
        return str(value)

    def number(self, field: str) -> float | None:
        value = self.raw.get(field)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def label(self, preferred_field: str = "display_name") -> str:
        candidate = self.raw.get(preferred_field)
        if candidate not in {None, ""}:
            return str(candidate)
        return self.display_name

    def short_label(self, preferred_field: str = "display_name", limit: int = 24) -> str:
        label = self.label(preferred_field)
        short = label.replace("_", " ").replace("/", " / ")[:limit].strip()
        return short or label[:limit]


@dataclass(frozen=True)
class DiagnosticRowView:
    """对诊断类行提供统一的名字与数值访问。"""

    raw: Mapping[str, Any]
    dataset: str
    name: str

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> DiagnosticRowView:
        dataset = str(row.get("dataset") or "")
        name = str(
            row.get("display_name")
            or row.get("method_name")
            or row.get("policy_name")
            or "unknown"
        )
        return cls(raw=row, dataset=dataset, name=name)

    def text(self, field: str, default: str = "") -> str:
        value = self.raw.get(field, default)
        if value is None:
            return default
        return str(value)

    def number(self, field: str) -> float | None:
        value = self.raw.get(field)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


def coerce_summary_rows(rows: Iterable[Mapping[str, Any]]) -> list[SummaryRowView]:
    """把原始 summary 行转成稳定视图。"""

    return [SummaryRowView.from_row(row) for row in rows]


def coerce_diagnostic_rows(rows: Iterable[Mapping[str, Any]]) -> list[DiagnosticRowView]:
    """把原始诊断行转成稳定视图。"""

    return [DiagnosticRowView.from_row(row) for row in rows]
