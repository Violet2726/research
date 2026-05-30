"""无效产物清理命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path

from research_experiments.cli_support.output import configure_utf8_stdio, emit_json
from research_experiments.workspace.artifact_cleanup import cleanup_invalid_artifacts, summary_to_dict
from research_experiments.workspace.layout import workspace_layout


def build_parser() -> argparse.ArgumentParser:
    """构造无效产物清理命令行参数。"""

    layout = workspace_layout()
    parser = argparse.ArgumentParser(description="删除无效运行记录与无效报告。")
    parser.add_argument("--workspace-root", default=".", help="工作区根目录，默认是当前目录。")
    parser.add_argument("--runs-root", default=layout.runs_root.as_posix(), help="相对工作区根目录的 runs 根路径。")
    parser.add_argument("--reports-root", default=layout.reports_root.as_posix(), help="相对工作区根目录的 reports 根路径。")
    parser.add_argument(
        "--revalidate-runs",
        action="store_true",
        help="使用当前 validator 重新校验 run；默认优先信任已有 run_validation.json。",
    )
    parser.add_argument("--dry-run", action="store_true", help="只输出候选项，不执行删除。")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出清理汇总。")
    return parser


def main(argv: list[str] | None = None) -> None:
    """命令行入口。"""

    configure_utf8_stdio()
    args = build_parser().parse_args(argv)
    workspace_root = Path(args.workspace_root).resolve()
    runs_root = workspace_root / args.runs_root
    reports_root = workspace_root / args.reports_root
    summary = cleanup_invalid_artifacts(
        workspace_root=workspace_root,
        runs_root=runs_root,
        reports_root=reports_root,
        dry_run=args.dry_run,
        revalidate_runs=args.revalidate_runs,
    )
    payload = summary_to_dict(summary)
    emit_json(payload)

