"""报告层使用的显式数据合同与读取入口。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Callable, Iterable, Mapping, Sequence


def load_json_payload(path: str | Path) -> dict[str, Any]:
    """读取 JSON 文件，缺失时返回空字典。"""

    target = Path(path)
    if not target.exists():
        return {}
    return json.loads(target.read_text(encoding="utf-8"))


def load_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件，缺失时返回空列表。"""

    target = Path(path)
    if not target.exists():
        return []
    with target.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


@dataclass(frozen=True)
class _BaseRowView:
    """报告层行视图的公共访问器。"""

    raw: Mapping[str, Any]

    def text(self, field: str, default: str = "") -> str:
        value = self.raw.get(field, default)
        if value is None:
            return default
        return str(value)

    def text_alias(self, *fields: str, default: str = "") -> str:
        for field in fields:
            value = self.raw.get(field)
            if value not in {None, ""}:
                return str(value)
        return default

    def number(self, field: str) -> float | None:
        value = self.raw.get(field)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def number_alias(self, *fields: str) -> float | None:
        for field in fields:
            value = self.number(field)
            if value is not None:
                return value
        return None

    def integer(self, field: str) -> int | None:
        value = self.raw.get(field)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None


@dataclass(frozen=True)
class SummaryRowView(_BaseRowView):
    """对 `metrics.summary` 行提供稳定字段合同。"""

    dataset: str
    method_name: str
    display_name: str

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> SummaryRowView:
        dataset = str(row.get("dataset") or "")
        method_name = str(row.get("method_name") or row.get("policy_name") or "unknown")
        display_name = str(row.get("display_name") or method_name)
        return cls(raw=row, dataset=dataset, method_name=method_name, display_name=display_name)

    @property
    def accuracy_mean(self) -> float | None:
        return self.number("accuracy_mean")

    @property
    def accuracy_std(self) -> float | None:
        return self.number("accuracy_std")

    @property
    def answer_em_mean(self) -> float | None:
        return self.number("answer_em_mean")

    @property
    def answer_f1_mean(self) -> float | None:
        return self.number("answer_f1_mean")

    @property
    def support_f1_mean(self) -> float | None:
        return self.number_alias("support_f1_mean", "supporting_f1_mean")

    @property
    def joint_f1_mean(self) -> float | None:
        return self.number("joint_f1_mean")

    @property
    def communication_tokens_mean(self) -> float | None:
        return self.number("communication_tokens_mean")

    @property
    def audit_tokens_mean(self) -> float | None:
        return self.number("audit_tokens_mean")

    @property
    def total_tokens_mean(self) -> float | None:
        return self.number("total_tokens_mean")

    @property
    def prompt_tokens_mean(self) -> float | None:
        return self.number("prompt_tokens_mean")

    @property
    def completion_tokens_mean(self) -> float | None:
        return self.number("completion_tokens_mean")

    @property
    def calls_per_question_mean(self) -> float | None:
        return self.number("calls_per_question_mean")

    @property
    def acc_per_1k_tokens(self) -> float | None:
        return self.number_alias("acc_per_1k_tokens", "accuracy_per_1k_tokens")

    @property
    def compression_ratio_vs_full_cot(self) -> float | None:
        return self.number("compression_ratio_vs_full_cot")

    @property
    def compression_ratio_mean(self) -> float | None:
        return self.number("compression_ratio_mean")

    @property
    def trigger_rate(self) -> float | None:
        return self.number("trigger_rate")

    @property
    def early_exit_rate(self) -> float | None:
        return self.number("early_exit_rate")

    @property
    def budget_utilization_mean(self) -> float | None:
        return self.number("budget_utilization_mean")

    @property
    def full_ratio_mean(self) -> float | None:
        return self.number("full_ratio_mean")

    @property
    def summary_ratio_mean(self) -> float | None:
        return self.number("summary_ratio_mean")

    @property
    def keywords_ratio_mean(self) -> float | None:
        return self.number("keywords_ratio_mean")

    @property
    def silence_ratio_mean(self) -> float | None:
        return self.number("silence_ratio_mean")

    @property
    def changed_answer_rate(self) -> float | None:
        return self.number("changed_answer_rate")

    @property
    def judge_fallback_rate(self) -> float | None:
        return self.number("judge_fallback_rate")

    @property
    def corrected_count(self) -> int:
        return self.integer("corrected_count") or 0

    @property
    def harmed_count(self) -> int:
        return self.integer("harmed_count") or 0

    @property
    def question_count(self) -> int:
        return self.integer("question_count") or 0

    @property
    def resolve_rate(self) -> float | None:
        return self.number("resolve_rate")

    @property
    def abstain_rate(self) -> float | None:
        return self.number("abstain_rate")

    @property
    def wrong_overrule_rate(self) -> float | None:
        return self.number("wrong_overrule_rate")

    @property
    def minority_rescue_count(self) -> int:
        return self.integer("minority_rescue_count") or 0

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
class DiagnosticRowView(_BaseRowView):
    """对诊断类行提供统一的名字与字段访问。"""

    dataset: str
    name: str

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> DiagnosticRowView:
        dataset = str(row.get("dataset") or "")
        name = str(
            row.get("display_name")
            or row.get("comparison")
            or row.get("method_name")
            or row.get("policy_name")
            or "unknown"
        )
        return cls(raw=row, dataset=dataset, name=name)

    @property
    def method_name(self) -> str:
        return self.text_alias("method_name", "policy_name", default=self.name)

    @property
    def comparison(self) -> str:
        return self.text("comparison", self.name)

    @property
    def accuracy_mean(self) -> float | None:
        return self.number("accuracy_mean")

    @property
    def trigger_rate(self) -> float | None:
        return self.number("trigger_rate")

    @property
    def early_exit_rate(self) -> float | None:
        return self.number("early_exit_rate")

    @property
    def precision(self) -> float | None:
        return self.number("precision")

    @property
    def recall(self) -> float | None:
        return self.number("recall")

    @property
    def false_trigger_rate(self) -> float | None:
        return self.number("false_trigger_rate")

    @property
    def missed_beneficial_comm_rate(self) -> float | None:
        return self.number("missed_beneficial_comm_rate")

    @property
    def communication_tokens_mean(self) -> float | None:
        return self.number("communication_tokens_mean")

    @property
    def shared_actual_tokens(self) -> float | None:
        return self.number("shared_actual_tokens")

    @property
    def naive_independent_tokens(self) -> float | None:
        return self.number("naive_independent_tokens")

    @property
    def shared_prefix_savings_ratio(self) -> float | None:
        return self.number("shared_prefix_savings_ratio")

    @property
    def joint_f1_delta(self) -> float | None:
        return self.number("joint_f1_delta")

    @property
    def answer_em_delta(self) -> float | None:
        return self.number("answer_em_delta")

    @property
    def supporting_f1_delta(self) -> float | None:
        return self.number("supporting_f1_delta")

    @property
    def communication_tokens_delta(self) -> float | None:
        return self.number("communication_tokens_delta")


@dataclass(frozen=True)
class SummaryTableView:
    """`metrics.summary` 的集合视图。"""

    rows: tuple[SummaryRowView, ...]

    @classmethod
    def from_rows(cls, rows: Iterable[Mapping[str, Any]]) -> SummaryTableView:
        return cls(tuple(SummaryRowView.from_row(row) for row in rows))

    @classmethod
    def from_metrics_payload(cls, payload: Mapping[str, Any]) -> SummaryTableView:
        return cls.from_rows(payload.get("summary", []))

    def grouped_by_dataset(self) -> dict[str, list[SummaryRowView]]:
        grouped: dict[str, list[SummaryRowView]] = {}
        for row in self.rows:
            grouped.setdefault(row.dataset, []).append(row)
        return grouped

    def dataset_names(self, *, include_overall: bool = False) -> list[str]:
        datasets = sorted({row.dataset for row in self.rows if include_overall or row.dataset != "overall"})
        return datasets

    def overall_rows(self) -> list[SummaryRowView]:
        return [row for row in self.rows if row.dataset == "overall"]

    def non_overall_rows(self) -> list[SummaryRowView]:
        return [row for row in self.rows if row.dataset != "overall"]

    def dataset_rows(self, dataset: str) -> list[SummaryRowView]:
        return [row for row in self.rows if row.dataset == dataset]

    def ordered(
        self,
        rows: Sequence[SummaryRowView] | None = None,
        *,
        method_order: Sequence[str] | None = None,
    ) -> list[SummaryRowView]:
        items = list(self.rows if rows is None else rows)
        if not method_order:
            return sorted(items, key=lambda row: (row.method_name, row.display_name))
        order_lookup = {name: index for index, name in enumerate(method_order)}
        return sorted(items, key=lambda row: (order_lookup.get(row.method_name, 999), row.method_name))

    def best_by(
        self,
        field: str,
        *,
        rows: Sequence[SummaryRowView] | None = None,
        predicate: Callable[[SummaryRowView], bool] | None = None,
    ) -> SummaryRowView | None:
        candidates = list(self.rows if rows is None else rows)
        if predicate is not None:
            candidates = [row for row in candidates if predicate(row)]
        scored = [(row, getattr(row, field, None)) for row in candidates]
        scored = [(row, value) for row, value in scored if value is not None]
        if not scored:
            return None
        return max(scored, key=lambda item: float(item[1]))[0]


@dataclass(frozen=True)
class DiagnosticTableView:
    """诊断类行的集合视图。"""

    rows: tuple[DiagnosticRowView, ...]

    @classmethod
    def from_rows(cls, rows: Iterable[Mapping[str, Any]]) -> DiagnosticTableView:
        return cls(tuple(DiagnosticRowView.from_row(row) for row in rows))

    def overall_rows(self) -> list[DiagnosticRowView]:
        return [row for row in self.rows if row.dataset == "overall"]

    def dataset_rows(self, dataset: str) -> list[DiagnosticRowView]:
        return [row for row in self.rows if row.dataset == dataset]

    def ordered(
        self,
        rows: Sequence[DiagnosticRowView] | None = None,
        *,
        name_order: Sequence[str] | None = None,
    ) -> list[DiagnosticRowView]:
        items = list(self.rows if rows is None else rows)
        if not name_order:
            return sorted(items, key=lambda row: (row.method_name, row.name))
        order_lookup = {name: index for index, name in enumerate(name_order)}
        return sorted(items, key=lambda row: (order_lookup.get(row.method_name, 999), row.method_name))

    def best_by(
        self,
        field: str,
        *,
        rows: Sequence[DiagnosticRowView] | None = None,
    ) -> DiagnosticRowView | None:
        candidates = list(self.rows if rows is None else rows)
        scored = [(row, getattr(row, field, None)) for row in candidates]
        scored = [(row, value) for row, value in scored if value is not None]
        if not scored:
            return None
        return max(scored, key=lambda item: float(item[1]))[0]


def coerce_summary_rows(rows: Iterable[Mapping[str, Any]]) -> list[SummaryRowView]:
    """把原始 summary 行转成稳定视图。"""

    return list(SummaryTableView.from_rows(rows).rows)


def coerce_diagnostic_rows(rows: Iterable[Mapping[str, Any]]) -> list[DiagnosticRowView]:
    """把原始诊断行转成稳定视图。"""

    return list(DiagnosticTableView.from_rows(rows).rows)


@dataclass(frozen=True)
class MatrixStateEntryView(_BaseRowView):
    """faithful matrix state 中单个 completed entry 的显式视图。"""

    family: str
    config_path: str
    experiment_name: str
    run_dir: str
    status: str

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> MatrixStateEntryView:
        return cls(
            raw=row,
            family=str(row.get("family") or ""),
            config_path=str(row.get("config_path") or ""),
            experiment_name=str(row.get("experiment_name") or ""),
            run_dir=str(row.get("run_dir") or ""),
            status=str(row.get("status") or ""),
        )


@dataclass(frozen=True)
class MatrixAnalysisRowView(_BaseRowView):
    """faithful_analysis 行的显式合同。"""

    family: str
    experiment_name: str
    evaluation_track: str
    evidence_tier: str
    dataset: str
    primary_method_name: str

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> MatrixAnalysisRowView:
        return cls(
            raw=row,
            family=str(row.get("family") or ""),
            experiment_name=str(row.get("experiment_name") or ""),
            evaluation_track=str(row.get("evaluation_track") or ""),
            evidence_tier=str(row.get("evidence_tier") or ""),
            dataset=str(row.get("dataset") or ""),
            primary_method_name=str(row.get("primary_method_name") or ""),
        )

    @property
    def faithful_score(self) -> float | None:
        return self.number("faithful_score")

    @property
    def best_no_comm_control(self) -> str:
        return self.text("best_no_comm_control")

    @property
    def best_no_comm_score(self) -> float | None:
        return self.number("best_no_comm_score")

    @property
    def delta_vs_best_no_comm(self) -> float | None:
        return self.number("delta_vs_best_no_comm")

    @property
    def full_comm_reference(self) -> str:
        return self.text("full_comm_reference")

    @property
    def delta_vs_full_comm(self) -> float | None:
        return self.number("delta_vs_full_comm")

    @property
    def full_context_reference(self) -> str:
        return self.text("full_context_reference")

    @property
    def delta_vs_full_context(self) -> float | None:
        return self.number("delta_vs_full_context")

    @property
    def family_envelope(self) -> str:
        return self.text("family_envelope")

    @property
    def delta_vs_family_envelope(self) -> float | None:
        return self.number("delta_vs_family_envelope")

    @property
    def stage_ceiling(self) -> float | None:
        return self.number("stage_ceiling")

    @property
    def stage_ceiling_gap(self) -> float | None:
        return self.number("stage_ceiling_gap")

    @property
    def token_ratio_vs_best_no_comm(self) -> float | None:
        return self.number("token_ratio_vs_best_no_comm")

    @property
    def token_ratio_vs_full_comm(self) -> float | None:
        return self.number("token_ratio_vs_full_comm")

    @property
    def communication_token_ratio_vs_full_comm(self) -> float | None:
        return self.number("communication_token_ratio_vs_full_comm")

    @property
    def engineering_noise_gap(self) -> float | None:
        return self.number("engineering_noise_gap")

    @property
    def calls_per_question_mean(self) -> float | None:
        return self.number("calls_per_question_mean")

    @property
    def total_tokens_mean(self) -> float | None:
        return self.number("total_tokens_mean")

    @property
    def communication_tokens_mean(self) -> float | None:
        return self.number("communication_tokens_mean")

    @property
    def run_dir(self) -> str:
        return self.text("run_dir")

    def to_dict(self) -> dict[str, Any]:
        return dict(self.raw)


@dataclass(frozen=True)
class MatrixAnalysisTableView:
    """faithful_analysis 行集合视图。"""

    rows: tuple[MatrixAnalysisRowView, ...]

    @classmethod
    def from_rows(cls, rows: Iterable[Mapping[str, Any]]) -> MatrixAnalysisTableView:
        return cls(tuple(MatrixAnalysisRowView.from_row(row) for row in rows))

    @classmethod
    def from_analysis_payload(cls, payload: Mapping[str, Any]) -> MatrixAnalysisTableView:
        return cls.from_rows(payload.get("combined_overall", []))

    def by_tier(self, evidence_tier: str, *, track: str | None = None) -> list[MatrixAnalysisRowView]:
        selected = [row for row in self.rows if row.evidence_tier == evidence_tier]
        if track is not None:
            selected = [row for row in selected if row.evaluation_track == track]
        return self.sorted(selected)

    def overall_rows(self) -> list[MatrixAnalysisRowView]:
        return [row for row in self.rows if row.dataset == "overall"]

    def sorted(self, rows: Sequence[MatrixAnalysisRowView] | None = None) -> list[MatrixAnalysisRowView]:
        items = list(self.rows if rows is None else rows)
        return sorted(items, key=lambda row: (row.family, row.experiment_name, row.dataset))


@dataclass(frozen=True)
class HelpfulHarmfulRowView(_BaseRowView):
    """有益/有害通信聚合行视图。"""

    experiment_name: str
    method_name: str

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> HelpfulHarmfulRowView:
        return cls(
            raw=row,
            experiment_name=str(row.get("experiment_name") or ""),
            method_name=str(row.get("method_name") or ""),
        )

    @property
    def helpful_rate(self) -> float | None:
        return self.number("helpful_rate")

    @property
    def harmful_rate(self) -> float | None:
        return self.number("harmful_rate")

    @property
    def sample_method_rows(self) -> int:
        return self.integer("sample_method_rows") or 0


@dataclass(frozen=True)
class HelpfulHarmfulTableView:
    rows: tuple[HelpfulHarmfulRowView, ...]

    @classmethod
    def from_rows(cls, rows: Iterable[Mapping[str, Any]]) -> HelpfulHarmfulTableView:
        return cls(tuple(HelpfulHarmfulRowView.from_row(row) for row in rows))


@dataclass(frozen=True)
class StatisticComparisonView(_BaseRowView):
    """paper_statistics comparison 行视图。"""

    comparison_id: str
    status: str

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> StatisticComparisonView:
        return cls(
            raw=row,
            comparison_id=str(row.get("comparison_id") or ""),
            status=str(row.get("status") or ""),
        )

    @property
    def paired_n(self) -> int | None:
        return self.integer("paired_n")

    @property
    def mean_delta(self) -> float | None:
        return self.number("mean_delta")


@dataclass(frozen=True)
class StatisticComparisonTableView:
    rows: tuple[StatisticComparisonView, ...]

    @classmethod
    def from_rows(cls, rows: Iterable[Mapping[str, Any]]) -> StatisticComparisonTableView:
        return cls(tuple(StatisticComparisonView.from_row(row) for row in rows))
