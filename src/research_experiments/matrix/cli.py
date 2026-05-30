"""矩阵命令行入口。"""

from __future__ import annotations

import argparse
from dataclasses import asdict

from research_experiments.cli_support.output import configure_utf8_stdio, emit_json
from research_experiments.matrix.execution import RuntimeOverrides, resume_matrix, run_matrix
from research_experiments.matrix.faithful_acceptance import render_acceptance_summary
from research_experiments.matrix.faithful_analysis import render_faithful_analysis
from research_experiments.matrix.matrix_specs import (
    DEFAULT_MATRIX_ID,
    MATRIX_ID_FAITHFUL,
    all_matrix_ids,
    get_matrix_profile,
)
from research_experiments.matrix.orchestrator import (
    DEFAULT_MAX_CONCURRENT_REQUESTS,
    DEFAULT_MODEL_REF,
    DEFAULT_PHASE,
    DEFAULT_REQUESTS_PER_MINUTE,
    DEFAULT_TOKENS_PER_MINUTE,
    _resolve_matrix_id_from_state,
)
from research_experiments.matrix.registry import build_run_matrix
from research_experiments.matrix.reproduction_analysis import render_reproduction_analysis
from research_experiments.reporting.family_landscape import render_family_landscape
from research_experiments.reporting.paper_package import render_paper_package
from research_experiments.reporting.paper_statistics import render_paper_statistics
from research_experiments.reporting.reproduction_landscape import render_reproduction_landscape
from research_experiments.reporting.reproduction_package import render_reproduction_package
from research_experiments.workspace.layout import workspace_defaults


def build_parser() -> argparse.ArgumentParser:
    """构建矩阵命令行解析器。"""

    parser = argparse.ArgumentParser(description="运行矩阵编排、恢复与分析命令。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_cmd = subparsers.add_parser("inspect-matrix", help="输出当前矩阵的解析结果。")
    run_cmd = subparsers.add_parser("run", help="顺序执行矩阵中的待运行条目。")
    resume_cmd = subparsers.add_parser("resume", help="恢复一个已有矩阵运行。")
    analyze_cmd = subparsers.add_parser("analyze-faithful", help="渲染 faithful 矩阵分析结果。")
    analyze_matrix_cmd = subparsers.add_parser("analyze-matrix", help="按矩阵 profile 渲染分析结果。")
    acceptance_cmd = subparsers.add_parser("evaluate-acceptance", help="渲染 faithful 矩阵验收摘要。")
    statistics_cmd = subparsers.add_parser("render-statistics", help="渲染统计产物。")
    paper_cmd = subparsers.add_parser("render-paper-package", help="渲染论文包产物。")
    landscape_cmd = subparsers.add_parser("render-family-landscape", help="渲染 family landscape。")
    package_matrix_cmd = subparsers.add_parser("render-matrix-package", help="按矩阵类型渲染 package。")
    landscape_matrix_cmd = subparsers.add_parser("render-matrix-landscape", help="按矩阵类型渲染 landscape。")

    for command in (inspect_cmd, run_cmd):
        command.add_argument("--matrix", default=DEFAULT_MATRIX_ID, choices=list(all_matrix_ids()))
        command.add_argument("--phase", default=DEFAULT_PHASE)
        command.add_argument("--model", default=DEFAULT_MODEL_REF)
        command.add_argument("--max-concurrent-requests", type=int, default=DEFAULT_MAX_CONCURRENT_REQUESTS)
        command.add_argument("--requests-per-minute-limit", type=int, default=DEFAULT_REQUESTS_PER_MINUTE)
        command.add_argument("--tokens-per-minute-limit", type=int, default=DEFAULT_TOKENS_PER_MINUTE)

    run_cmd.add_argument("--state-root", default=None)
    run_cmd.add_argument("--reference-state-path")
    resume_cmd.add_argument("--state-path", required=True)
    resume_cmd.add_argument("--reference-state-path")
    analyze_cmd.add_argument("--state-path", required=True)
    analyze_cmd.add_argument("--reference-state-path")
    analyze_matrix_cmd.add_argument("--state-path", required=True)
    analyze_matrix_cmd.add_argument("--reference-state-path")
    acceptance_cmd.add_argument("--analysis-path", required=True)
    statistics_cmd.add_argument("--state-path", required=True)
    paper_cmd.add_argument("--state-path", required=True)
    landscape_cmd.add_argument("--state-path", required=True)
    package_matrix_cmd.add_argument("--state-path", required=True)
    landscape_matrix_cmd.add_argument("--state-path", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    """命令行入口。"""

    configure_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "inspect-matrix":
        overrides = RuntimeOverrides(
            phase_name=args.phase,
            model_ref=args.model,
            max_concurrent_requests=args.max_concurrent_requests,
            requests_per_minute_limit=args.requests_per_minute_limit,
            tokens_per_minute_limit=args.tokens_per_minute_limit,
        )
        matrix = build_run_matrix(overrides, matrix_id=args.matrix)
        profile = get_matrix_profile(args.matrix)
        emit_json(
            {
                "matrix_id": profile.matrix_id,
                "matrix_kind": profile.matrix_kind,
                "overrides": asdict(overrides),
                "counts": matrix.counts,
                "workspace_defaults": workspace_defaults(profile.matrix_kind),
                "entries": [asdict(entry) for entry in matrix.entries],
                "semantic_entries": [asdict(entry) for entry in matrix.semantic_entries],
            }
        )
        return

    if args.command == "run":
        overrides = RuntimeOverrides(
            phase_name=args.phase,
            model_ref=args.model,
            max_concurrent_requests=args.max_concurrent_requests,
            requests_per_minute_limit=args.requests_per_minute_limit,
            tokens_per_minute_limit=args.tokens_per_minute_limit,
        )
        run_dir = run_matrix(
            args.matrix,
            overrides,
            state_root=args.state_root,
            reference_state_path_or_root=args.reference_state_path,
        )
        print(run_dir.as_posix())
        return

    if args.command == "resume":
        run_dir = resume_matrix(args.state_path, reference_state_path_or_root=args.reference_state_path)
        print(run_dir.as_posix())
        return

    if args.command == "analyze-faithful":
        emit_json(render_faithful_analysis(args.state_path, reference_state_path_or_root=args.reference_state_path))
        return

    if args.command == "analyze-matrix":
        matrix_id = _resolve_matrix_id_from_state(args.state_path)
        if matrix_id == MATRIX_ID_FAITHFUL:
            payload = render_faithful_analysis(args.state_path, reference_state_path_or_root=args.reference_state_path)
        else:
            payload = render_reproduction_analysis(args.state_path)
        emit_json(payload)
        return

    if args.command == "evaluate-acceptance":
        emit_json(render_acceptance_summary(args.analysis_path))
        return

    if args.command == "render-statistics":
        emit_json(render_paper_statistics(args.state_path))
        return

    if args.command == "render-paper-package":
        emit_json(render_paper_package(args.state_path))
        return

    if args.command == "render-family-landscape":
        emit_json(render_family_landscape(args.state_path))
        return

    if args.command == "render-matrix-package":
        matrix_id = _resolve_matrix_id_from_state(args.state_path)
        if matrix_id == MATRIX_ID_FAITHFUL:
            payload = render_paper_package(args.state_path)
        else:
            payload = render_reproduction_package(args.state_path)
        emit_json(payload)
        return

    if args.command == "render-matrix-landscape":
        matrix_id = _resolve_matrix_id_from_state(args.state_path)
        if matrix_id == MATRIX_ID_FAITHFUL:
            payload = render_family_landscape(args.state_path)
        else:
            payload = render_reproduction_landscape(args.state_path)
        emit_json(payload)
        return

    parser.error(f"Unsupported command: {args.command}")

