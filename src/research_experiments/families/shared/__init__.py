"""family 层共享脚手架与公共工具入口。"""

from __future__ import annotations

from research_experiments.families.shared.cli import (
    build_standard_family_parser,
    dispatch_standard_family_cli,
)
from research_experiments.families.shared.common import (
    build_question_preview,
    resolve_phase_split_name,
    safe_mean,
    safe_ratio,
    stable_trace_hash,
    sum_metric,
    summarize_row_cost,
)
from research_experiments.families.shared.config_loading import (
    SupportsBenchmarkConfigs,
    SupportsRawPhases,
    first_str,
    load_benchmarks,
    load_toml,
    optional_float,
    optional_int,
    optional_str,
    phase_metadata,
    resolve_model,
)
from research_experiments.families.shared.method_catalog import MethodConfig, load_method_catalog
from research_experiments.families.shared.report_common import (
    render_family_report_bundle,
    render_family_scientific_report,
)
from research_experiments.families.shared.reference_runs import (
    TriggerReferenceConfig,
    TriggerReferenceDecision,
    resolve_trigger_reference_selection,
    write_policy_reference_summary,
)
from research_experiments.families.shared.validate_common import (
    load_json,
    load_jsonl,
    validate_shared_contracts,
)

__all__ = [
    "build_standard_family_parser",
    "dispatch_standard_family_cli",
    "build_question_preview",
    "resolve_phase_split_name",
    "safe_mean",
    "safe_ratio",
    "stable_trace_hash",
    "sum_metric",
    "summarize_row_cost",
    "SupportsRawPhases",
    "SupportsBenchmarkConfigs",
    "load_toml",
    "optional_int",
    "optional_float",
    "optional_str",
    "first_str",
    "phase_metadata",
    "load_benchmarks",
    "resolve_model",
    "MethodConfig",
    "load_method_catalog",
    "render_family_report_bundle",
    "render_family_scientific_report",
    "TriggerReferenceConfig",
    "TriggerReferenceDecision",
    "write_policy_reference_summary",
    "resolve_trigger_reference_selection",
    "load_json",
    "load_jsonl",
    "validate_shared_contracts",
]
