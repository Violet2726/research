"""统一编排 faithful 实验矩阵运行、分析与验收。"""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import tomllib
from typing import Any

from budget_comm.config import load_experiment_config as load_budget_experiment_config
from budget_comm.config import resolve_model as resolve_budget_model
from budget_comm.runner import run_experiment as run_budget_experiment
from budget_comm.validation import validate_run as validate_budget_run
from comm_necessary.config import load_experiment_config as load_comm_necessary_experiment_config
from comm_necessary.config import resolve_model as resolve_comm_necessary_model
from comm_necessary.runner import run_experiment as run_comm_necessary_experiment
from comm_necessary.validation import validate_run as validate_comm_necessary_run
from cue.config import load_experiment_config as load_cue_experiment_config
from cue.config import resolve_model as resolve_cue_model
from cue.runner import run_experiment as run_cue_experiment
from cue.validation import validate_run as validate_cue_run
from experiment_core.foundation.config import load_benchmark_config, resolve_model_ref
from experiment_core.matrix.faithful_acceptance import render_acceptance_summary
from experiment_core.matrix.faithful_analysis import render_faithful_analysis
from experiment_core.matrix.matrix_specs import get_experiment_matrix_spec
from experiment_core.reporting.paper_package import render_paper_package
from experiment_core.reporting.paper_statistics import render_paper_statistics
from experiment_core.foundation.workspace import default_runs_root, workspace_defaults
from free_mad_lite.config import load_experiment_config as load_free_mad_experiment_config
from free_mad_lite.config import resolve_model as resolve_free_mad_model
from free_mad_lite.runner import run_experiment as run_free_mad_experiment
from free_mad_lite.validation import validate_run as validate_free_mad_run
from multi_agent.config import load_experiment_config as load_multi_agent_experiment_config
from multi_agent.config import resolve_model as resolve_multi_agent_model
from multi_agent.runner import run_experiment as run_multi_agent_experiment
from multi_agent.validation import validate_run as validate_multi_agent_run
from selective_comm.config import load_experiment_config as load_selective_experiment_config
from selective_comm.config import resolve_model as resolve_selective_model
from selective_comm.runner import run_experiment as run_selective_experiment
from selective_comm.validation import validate_run as validate_selective_run
from sid_lite.config import load_experiment_config as load_sid_experiment_config
from sid_lite.config import resolve_model as resolve_sid_model
from sid_lite.runner import run_experiment as run_sid_experiment
from sid_lite.validation import validate_run as validate_sid_run
from single_agent.config import ExperimentConfig as SingleAgentExperimentConfig
from single_agent.config import load_experiment_config as load_single_agent_experiment_config
from single_agent.runner import run_experiment as run_single_agent_experiment
from single_agent.validation import validate_run as validate_single_agent_run
from sparc.config import load_experiment_config as load_sparc_experiment_config
from sparc.config import resolve_model as resolve_sparc_model
from sparc.runner import run_experiment as run_sparc_experiment
from sparc.validation import validate_run as validate_sparc_run


DEFAULT_PHASE = "smoke20"
DEFAULT_MODEL_REF = "xiaomimimo/mimo-v2.5"
DEFAULT_MAX_CONCURRENT_REQUESTS = 90
DEFAULT_REQUESTS_PER_MINUTE = 95
DEFAULT_TOKENS_PER_MINUTE = 9000000
MATRIX_EXPERIMENT_KIND = "faithful_matrix"

EXCLUDED_CONFIGS: dict[str, str] = {}

RUN_ORDER = [
    "configs/single_agent/experiments/same_context_core_benchmarks.toml",
    "configs/multi_agent/experiments/same_context_controlled_debate.toml",
    "configs/free_mad_lite/experiments/free_mad_lite_mechanism_validation.toml",
    "configs/budget_comm/experiments/dala_lite_same_context_main.toml",
    "configs/comm_necessary/experiments/hotpotqa_split_context_communication_necessity.toml",
    "configs/sid_lite/experiments/sid_lite_mechanism_validation.toml",
    "configs/selective_comm/experiments/trigger_early_exit_main.toml",
    "configs/sparc/experiments/end_to_end_main.toml",
    "configs/single_agent/experiments/same_context_main_table.toml",
    "configs/single_agent/experiments/cross_provider_robustness.toml",
    "configs/budget_comm/experiments/dala_lite_split_context_main.toml",
    "configs/sparc/experiments/content_ablation.toml",
    "configs/sparc/experiments/local_auditing_ablation.toml",
    "configs/cue/experiments/cue_black_box_utility_main.toml",
    "configs/selective_comm/experiments/voc_trigger_main.toml",
]


@dataclass(frozen=True)
class RuntimeOverrides:
    """矩阵批跑时可统一覆盖的运行时参数。"""

    phase_name: str = DEFAULT_PHASE
    model_ref: str = DEFAULT_MODEL_REF
    max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT_REQUESTS
    requests_per_minute_limit: int = DEFAULT_REQUESTS_PER_MINUTE
    tokens_per_minute_limit: int = DEFAULT_TOKENS_PER_MINUTE


@dataclass(frozen=True)
class DiscoveredConfig:
    """从 `configs/*/experiments` 中发现的实验入口摘要。"""

    family: str
    config_path: str
    experiment_name: str
    description: str


@dataclass
class MatrixEntry:
    """矩阵中的单个可执行或已排除实验条目。"""

    family: str
    config_path: str
    experiment_name: str
    description: str
    phase_name: str
    evaluation_track: str
    evidence_tier: str
    primary_method_name: str
    best_no_comm_candidates: list[str]
    full_comm_reference: str | None
    full_context_reference: str | None
    status: str
    excluded_reason: str | None = None
    run_dir: str | None = None
    validation_passed: bool | None = None
    review_passed: bool | None = None
    review_notes: str = ""


@dataclass(frozen=True)
class MatrixBuild:
    """一次矩阵构建的完整结果。"""

    overrides: RuntimeOverrides
    entries: list[MatrixEntry]
    semantic_entries: list[MatrixEntry]
    counts: dict[str, int]


@dataclass(frozen=True)
class ReviewResult:
    """运行后健康检查的结论。"""

    passed: bool
    notes: str


@dataclass(frozen=True)
class OrchestratorPaths:
    """矩阵 orchestrator 维护的状态与报告路径。"""

    root: Path
    matrix: Path
    state: Path
    report: Path
    published_summary: Path


def discover_phase_configs(phase_name: str, config_root: str | Path = "configs") -> list[DiscoveredConfig]:
    """发现声明了给定 phase 的实验配置入口。"""
    discovered: list[DiscoveredConfig] = []
    for path in sorted(Path(config_root).rglob("*.toml")):
        if "experiments" not in path.parts:
            continue
        payload = _load_toml(path)
        phases = payload.get("phases", {})
        if phase_name not in phases:
            continue
        discovered.append(
            DiscoveredConfig(
                family=path.parts[1],
                config_path=path.as_posix(),
                experiment_name=str(payload["name"]),
                description=str(payload.get("description", "")),
            )
        )
    return discovered


def build_run_matrix(
    overrides: RuntimeOverrides,
) -> MatrixBuild:
    """根据覆盖参数生成可执行矩阵与语义去重后的目标集合。"""
    discovered = discover_phase_configs(overrides.phase_name)

    entries: list[MatrixEntry] = []
    semantic_entries: list[MatrixEntry] = []
    for item in discovered:
        if item.config_path in EXCLUDED_CONFIGS:
            spec = get_experiment_matrix_spec(item.config_path)
            entries.append(
                MatrixEntry(
                    family=item.family,
                    config_path=item.config_path,
                    experiment_name=item.experiment_name,
                    description=item.description,
                    phase_name=overrides.phase_name,
                    evaluation_track=spec.evaluation_track,
                    evidence_tier=spec.evidence_tier,
                    primary_method_name=spec.primary_method_name,
                    best_no_comm_candidates=list(spec.best_no_comm_candidates),
                    full_comm_reference=spec.full_comm_reference,
                    full_context_reference=spec.full_context_reference,
                    status="excluded",
                    excluded_reason=EXCLUDED_CONFIGS[item.config_path],
                )
            )
            continue
        spec = get_experiment_matrix_spec(item.config_path)
        entry = MatrixEntry(
            family=item.family,
            config_path=item.config_path,
            experiment_name=item.experiment_name,
            description=item.description,
            phase_name=overrides.phase_name,
            evaluation_track=spec.evaluation_track,
            evidence_tier=spec.evidence_tier,
            primary_method_name=spec.primary_method_name,
            best_no_comm_candidates=list(spec.best_no_comm_candidates),
            full_comm_reference=spec.full_comm_reference,
            full_context_reference=spec.full_context_reference,
            status="pending",
        )
        entries.append(entry)
        semantic_entries.append(entry)

    counts = Counter(entry.status for entry in semantic_entries)
    counts.update(
        {
            "excluded": sum(1 for entry in entries if entry.status == "excluded"),
            "semantic_unique_targets": len(semantic_entries),
        }
    )

    expected_order = {path: index for index, path in enumerate(RUN_ORDER)}
    semantic_entries.sort(key=lambda entry: expected_order.get(entry.config_path, 10_000))
    entries.sort(key=lambda entry: expected_order.get(entry.config_path, 10_000))
    return MatrixBuild(overrides=overrides, entries=entries, semantic_entries=semantic_entries, counts=dict(counts))


def review_run_health(run_dir: str | Path, family: str) -> ReviewResult:
    """复核单个 run 的进度、验证结果与核心汇总文件。"""
    root = Path(run_dir)
    progress = _safe_load_json(root / "progress.json") or {}
    if str(progress.get("status") or "") != "completed":
        return ReviewResult(False, "progress_not_completed")

    validation = _safe_load_json(root / "run_validation.json") or {}
    if not bool(validation.get("passed")):
        return ReviewResult(False, "validation_not_passed")

    metrics_name = "policy_metrics.json" if family in {"cue", "selective_comm"} else "metrics.json"
    metrics_payload = _safe_load_json(root / metrics_name) or {}
    summary_rows = metrics_payload.get("summary", []) if isinstance(metrics_payload, dict) else []
    if not summary_rows:
        return ReviewResult(False, f"missing_summary_rows:{metrics_name}")

    question_counts = [
        int(
            row.get("question_count")
            or row.get("questions_per_rerun")
            or row.get("prediction_rows")
            or 0
        )
        for row in summary_rows
        if isinstance(row, dict)
    ]
    if not question_counts or max(question_counts) <= 0:
        return ReviewResult(False, "empty_question_counts")

    token_totals = [float(row.get("total_tokens_mean") or 0.0) for row in summary_rows if isinstance(row, dict)]
    if token_totals and max(token_totals) <= 0.0:
        return ReviewResult(False, "zero_total_tokens")

    return ReviewResult(True, "validation_passed_and_metrics_nonempty")


def run_faithful_matrix(
    overrides: RuntimeOverrides,
    *,
    state_root: str | Path | None = None,
    reference_state_path_or_root: str | Path | None = None,
) -> Path:
    """执行 faithful 矩阵批跑、写入状态文件，并产出分析报告。"""
    matrix = build_run_matrix(overrides)
    paths = _prepare_orchestrator_paths(state_root, overrides)
    family_blocked: set[str] = set()
    _write_matrix_state(paths, matrix)

    entries_by_config = {entry.config_path: entry for entry in matrix.semantic_entries}
    for config_path in RUN_ORDER:
        entry = entries_by_config.get(config_path)
        if entry is None or entry.status != "pending":
            continue
        if entry.family in family_blocked:
            entry.review_notes = "family_blocked_after_previous_failure"
            continue

        entry.status = "running"
        _write_matrix_state(paths, matrix)
        try:
            run_dir = _execute_entry(entry, overrides)
            validation = _validate_entry(entry.family, run_dir)
            review = review_run_health(run_dir, entry.family)
            entry.run_dir = run_dir.as_posix()
            entry.validation_passed = bool(validation.get("passed"))
            entry.review_passed = review.passed
            entry.review_notes = review.notes
            entry.status = "completed" if entry.validation_passed and entry.review_passed else "rerun-needed"
            if entry.status != "completed":
                family_blocked.add(entry.family)
        except Exception as exc:
            entry.status = "failed"
            entry.review_notes = f"runner_error:{exc}"
            family_blocked.add(entry.family)
        _write_matrix_state(paths, matrix)

    _write_matrix_state(paths, matrix)
    render_faithful_analysis(
        paths.root,
        reference_state_path_or_root=reference_state_path_or_root,
    )
    render_acceptance_summary(paths.root)
    render_paper_statistics(paths.root)
    render_paper_package(paths.root)
    return paths.root


def resume_faithful_matrix(
    state_path_or_root: str | Path,
    *,
    reference_state_path_or_root: str | Path | None = None,
) -> Path:
    """Resume a previously interrupted faithful matrix run in-place."""
    matrix, paths = _load_existing_matrix_state(state_path_or_root)

    for entry in matrix.semantic_entries:
        if entry.status == "running":
            entry.status = "rerun-needed"
            if not entry.review_notes:
                entry.review_notes = "interrupted_previous_run"

    family_blocked: set[str] = set()
    _write_matrix_state(paths, matrix)

    entries_by_config = {entry.config_path: entry for entry in matrix.semantic_entries}
    resumable_statuses = {"pending", "rerun-needed", "failed", "running"}
    for config_path in RUN_ORDER:
        entry = entries_by_config.get(config_path)
        if entry is None or entry.status not in resumable_statuses:
            continue
        if entry.family in family_blocked:
            entry.review_notes = "family_blocked_after_previous_failure"
            continue

        entry.status = "running"
        _write_matrix_state(paths, matrix)
        try:
            run_dir = _execute_entry(entry, matrix.overrides)
            validation = _validate_entry(entry.family, run_dir)
            review = review_run_health(run_dir, entry.family)
            entry.run_dir = run_dir.as_posix()
            entry.validation_passed = bool(validation.get("passed"))
            entry.review_passed = review.passed
            entry.review_notes = review.notes
            entry.status = "completed" if entry.validation_passed and entry.review_passed else "rerun-needed"
            if entry.status != "completed":
                family_blocked.add(entry.family)
        except Exception as exc:
            entry.status = "failed"
            entry.review_notes = f"runner_error:{exc}"
            family_blocked.add(entry.family)
        _write_matrix_state(paths, matrix)

    _write_matrix_state(paths, matrix)
    render_faithful_analysis(
        paths.root,
        reference_state_path_or_root=reference_state_path_or_root,
    )
    render_acceptance_summary(paths.root)
    render_paper_statistics(paths.root)
    render_paper_package(paths.root)
    return paths.root


def _execute_entry(entry: MatrixEntry, overrides: RuntimeOverrides) -> Path:
    family = entry.family
    config_path = entry.config_path
    if family == "single_agent":
        experiment = load_single_agent_experiment_config(config_path)
        overridden = apply_runtime_overrides(family, experiment, overrides)
        model = resolve_model_ref(overrides.model_ref)
        benchmarks = [load_benchmark_config(path) for path in overridden.benchmark_configs]
        return run_single_agent_experiment(
            experiment=overridden,
            phase_name=overrides.phase_name,
            models=[model],
            benchmarks=benchmarks,
        )

    loader_map = {
        "budget_comm": load_budget_experiment_config,
        "comm_necessary": load_comm_necessary_experiment_config,
        "cue": load_cue_experiment_config,
        "free_mad_lite": load_free_mad_experiment_config,
        "multi_agent": load_multi_agent_experiment_config,
        "selective_comm": load_selective_experiment_config,
        "sid_lite": load_sid_experiment_config,
        "sparc": load_sparc_experiment_config,
    }
    resolver_map = {
        "budget_comm": resolve_budget_model,
        "comm_necessary": resolve_comm_necessary_model,
        "cue": resolve_cue_model,
        "free_mad_lite": resolve_free_mad_model,
        "multi_agent": resolve_multi_agent_model,
        "selective_comm": resolve_selective_model,
        "sid_lite": resolve_sid_model,
        "sparc": resolve_sparc_model,
    }
    runner_map = {
        "budget_comm": run_budget_experiment,
        "comm_necessary": run_comm_necessary_experiment,
        "cue": run_cue_experiment,
        "free_mad_lite": run_free_mad_experiment,
        "multi_agent": run_multi_agent_experiment,
        "selective_comm": run_selective_experiment,
        "sid_lite": run_sid_experiment,
        "sparc": run_sparc_experiment,
    }
    experiment = loader_map[family](config_path)
    overridden = apply_runtime_overrides(family, experiment, overrides)
    backbone = resolver_map[family](overrides.model_ref)
    return runner_map[family](
        experiment=overridden,
        phase_name=overrides.phase_name,
        backbone=backbone,
    )


def apply_runtime_overrides(family: str, experiment: Any, overrides: RuntimeOverrides) -> Any:
    """把命令行层面的并发与限流覆盖写回实验配置对象。"""
    raw = copy.deepcopy(experiment.raw)
    raw["max_concurrent_requests"] = overrides.max_concurrent_requests
    raw["requests_per_minute_limit"] = overrides.requests_per_minute_limit
    raw["tokens_per_minute_limit"] = overrides.tokens_per_minute_limit

    replace_kwargs: dict[str, Any] = {
        "max_concurrent_requests": overrides.max_concurrent_requests,
        "requests_per_minute_limit": overrides.requests_per_minute_limit,
        "tokens_per_minute_limit": overrides.tokens_per_minute_limit,
        "raw": raw,
    }

    if hasattr(experiment, "primary_model_ref"):
        raw["primary_model_ref"] = overrides.model_ref
        replace_kwargs["primary_model_ref"] = overrides.model_ref

    if family == "single_agent":
        raw.setdefault("phases", {}).setdefault(overrides.phase_name, {})
        raw["phases"][overrides.phase_name]["required_model_tags"] = []
        raw["phases"][overrides.phase_name]["benchmark_required_tags"] = {}
        replace_kwargs["required_model_tags"] = []
        replace_kwargs["benchmark_required_tags"] = {}

    return replace(experiment, **replace_kwargs)


def _validate_entry(family: str, run_dir: Path) -> dict[str, Any]:
    validators = {
        "budget_comm": validate_budget_run,
        "comm_necessary": validate_comm_necessary_run,
        "cue": validate_cue_run,
        "free_mad_lite": validate_free_mad_run,
        "multi_agent": validate_multi_agent_run,
        "selective_comm": validate_selective_run,
        "sid_lite": validate_sid_run,
        "single_agent": validate_single_agent_run,
        "sparc": validate_sparc_run,
    }
    payload = validators[family](run_dir)
    (Path(run_dir) / "run_validation.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _prepare_orchestrator_paths(state_root: str | Path | None, overrides: RuntimeOverrides) -> OrchestratorPaths:
    root_base = Path(state_root or default_runs_root(MATRIX_EXPERIMENT_KIND))
    model_slug = overrides.model_ref.replace("/", "-")
    run_id = datetime.now(timezone.utc).strftime(f"%Y%m%dT%H%M%SZ-{overrides.phase_name}-{model_slug}")
    root = root_base / run_id
    root.mkdir(parents=True, exist_ok=True)
    return OrchestratorPaths(
        root=root,
        matrix=root / "matrix.json",
        state=root / "state.json",
        report=root / "matrix_status.md",
        published_summary=Path("reports") / "summary" / f"{run_id}.md",
    )


def _existing_orchestrator_paths(root: Path) -> OrchestratorPaths:
    return OrchestratorPaths(
        root=root,
        matrix=root / "matrix.json",
        state=root / "state.json",
        report=root / "matrix_status.md",
        published_summary=Path("reports") / "summary" / f"{root.name}.md",
    )


def _load_existing_matrix_state(state_path_or_root: str | Path) -> tuple[MatrixBuild, OrchestratorPaths]:
    state_path = Path(state_path_or_root)
    if state_path.is_dir():
        state_path = state_path / "state.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    overrides = RuntimeOverrides(**payload["overrides"])
    entries = [MatrixEntry(**_normalize_entry_payload(entry)) for entry in payload.get("entries", [])]
    semantic_entries = [MatrixEntry(**_normalize_entry_payload(entry)) for entry in payload.get("semantic_entries", [])]
    matrix = MatrixBuild(
        overrides=overrides,
        entries=entries,
        semantic_entries=semantic_entries,
        counts=dict(payload.get("counts", {})),
    )
    return matrix, _existing_orchestrator_paths(state_path.parent)


def _normalize_entry_payload(entry: dict[str, Any]) -> dict[str, Any]:
    payload = dict(entry)
    if "evidence_tier" not in payload:
        payload["evidence_tier"] = get_experiment_matrix_spec(str(payload["config_path"])).evidence_tier
    return payload


def _write_matrix_state(paths: OrchestratorPaths, matrix: MatrixBuild) -> None:
    counts = Counter(entry.status for entry in matrix.semantic_entries)
    counts.update(
        {
            "excluded": sum(1 for entry in matrix.entries if entry.status == "excluded"),
            "semantic_unique_targets": len(matrix.semantic_entries),
        }
    )
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overrides": asdict(matrix.overrides),
        "counts": dict(counts),
        "entries": [asdict(entry) for entry in matrix.entries],
        "semantic_entries": [asdict(entry) for entry in matrix.semantic_entries],
    }
    paths.matrix.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    paths.state.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_text = _render_matrix_report(matrix, dict(counts))
    paths.report.write_text(report_text, encoding="utf-8")
    paths.published_summary.parent.mkdir(parents=True, exist_ok=True)
    paths.published_summary.write_text(report_text, encoding="utf-8")


def _render_matrix_report(matrix: MatrixBuild, counts: dict[str, int]) -> str:
    lines = [
        f"# {matrix.overrides.phase_name} faithful matrix",
        "",
        f"- generated_at: `{datetime.now(timezone.utc).isoformat()}`",
        f"- model: `{matrix.overrides.model_ref}`",
        f"- rate_limits: `{matrix.overrides.max_concurrent_requests}` / `{matrix.overrides.requests_per_minute_limit}` / `{matrix.overrides.tokens_per_minute_limit}`",
        f"- counts: `{json.dumps(counts, ensure_ascii=False)}`",
        "",
        "| family | config | evidence_tier | status | run_dir | notes |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for entry in matrix.entries:
        lines.append(
            "| "
            + " | ".join(
                [
                    entry.family,
                    Path(entry.config_path).name,
                    entry.evidence_tier,
                    entry.status,
                    entry.run_dir or "",
                    entry.review_notes or entry.excluded_reason or "",
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def _load_toml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def _safe_load_json(path: str | Path) -> dict[str, Any] | None:
    target = Path(path)
    if not target.exists():
        return None
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except Exception:
        return None


def build_parser() -> argparse.ArgumentParser:
    """构建 faithful 矩阵命令行解析器。"""
    parser = argparse.ArgumentParser(description="Run the unified faithful experiment matrix.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_cmd = subparsers.add_parser("inspect-matrix", help="Print the resolved faithful matrix.")
    run_cmd = subparsers.add_parser("run", help="Run the pending faithful matrix entries sequentially.")
    resume_cmd = subparsers.add_parser("resume", help="Resume a faithful-matrix run with pending or rerun-needed entries.")
    analyze_cmd = subparsers.add_parser("analyze-faithful", help="Render faithful analysis for an existing faithful-matrix run.")
    acceptance_cmd = subparsers.add_parser("evaluate-acceptance", help="Render acceptance summary for an existing faithful-matrix run.")
    statistics_cmd = subparsers.add_parser("render-statistics", help="Render paper-grade statistical artifacts for an existing run.")
    paper_cmd = subparsers.add_parser("render-paper-package", help="Render paper-facing tables and figures for an existing run.")

    for command in (inspect_cmd, run_cmd):
        command.add_argument("--phase", default=DEFAULT_PHASE)
        command.add_argument("--model", default=DEFAULT_MODEL_REF)
        command.add_argument("--max-concurrent-requests", type=int, default=DEFAULT_MAX_CONCURRENT_REQUESTS)
        command.add_argument("--requests-per-minute-limit", type=int, default=DEFAULT_REQUESTS_PER_MINUTE)
        command.add_argument("--tokens-per-minute-limit", type=int, default=DEFAULT_TOKENS_PER_MINUTE)

    run_cmd.add_argument("--state-root", default=default_runs_root(MATRIX_EXPERIMENT_KIND))
    run_cmd.add_argument("--reference-state-path")
    resume_cmd.add_argument("--state-path", required=True)
    resume_cmd.add_argument("--reference-state-path")
    analyze_cmd.add_argument("--state-path", required=True)
    analyze_cmd.add_argument("--reference-state-path")
    acceptance_cmd.add_argument("--analysis-path", required=True)
    statistics_cmd.add_argument("--state-path", required=True)
    paper_cmd.add_argument("--state-path", required=True)
    return parser


def main() -> None:
    """命令行入口。"""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "inspect-matrix":
        overrides = RuntimeOverrides(
            phase_name=args.phase,
            model_ref=args.model,
            max_concurrent_requests=args.max_concurrent_requests,
            requests_per_minute_limit=args.requests_per_minute_limit,
            tokens_per_minute_limit=args.tokens_per_minute_limit,
        )
        matrix = build_run_matrix(overrides)
        payload = {
            "overrides": asdict(overrides),
            "counts": matrix.counts,
            "workspace_defaults": workspace_defaults(MATRIX_EXPERIMENT_KIND),
            "entries": [asdict(entry) for entry in matrix.entries],
            "semantic_entries": [asdict(entry) for entry in matrix.semantic_entries],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "run":
        overrides = RuntimeOverrides(
            phase_name=args.phase,
            model_ref=args.model,
            max_concurrent_requests=args.max_concurrent_requests,
            requests_per_minute_limit=args.requests_per_minute_limit,
            tokens_per_minute_limit=args.tokens_per_minute_limit,
        )
        run_dir = run_faithful_matrix(
            overrides,
            state_root=args.state_root,
            reference_state_path_or_root=args.reference_state_path,
        )
        print(run_dir.as_posix())
        return

    if args.command == "resume":
        run_dir = resume_faithful_matrix(
            args.state_path,
            reference_state_path_or_root=args.reference_state_path,
        )
        print(run_dir.as_posix())
        return

    if args.command == "analyze-faithful":
        paths = render_faithful_analysis(
            args.state_path,
            reference_state_path_or_root=args.reference_state_path,
        )
        print(json.dumps(paths, ensure_ascii=False, indent=2))
        return

    if args.command == "evaluate-acceptance":
        paths = render_acceptance_summary(args.analysis_path)
        print(json.dumps(paths, ensure_ascii=False, indent=2))
        return

    if args.command == "render-statistics":
        paths = render_paper_statistics(args.state_path)
        print(json.dumps(paths, ensure_ascii=False, indent=2))
        return

    if args.command == "render-paper-package":
        paths = render_paper_package(args.state_path)
        print(json.dumps(paths, ensure_ascii=False, indent=2))
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()

