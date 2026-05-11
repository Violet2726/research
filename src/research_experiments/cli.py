"""统一命令入口。"""

from __future__ import annotations

import argparse
from typing import Callable

from research_experiments.core.foundation.cli_output import configure_utf8_stdio
from research_experiments.families.registry import get_family_spec, registered_family_names
from research_experiments.matrix.faithful_matrix import main as matrix_main
from research_experiments.tools.archive_runs import main as archive_runs_main
from research_experiments.tools.artifact_cleanup import main as artifact_cleanup_main
from research_experiments.tools.cache_archive import main as cache_archive_main
from research_experiments.tools.cache_inspector import main as cache_inspector_main
from research_experiments.tools.dataset_assets import main as dataset_assets_main
from research_experiments.tools.hf_sync import main as hf_sync_main


ToolMain = Callable[[list[str] | None], None]

TOOL_MAINS: dict[str, ToolMain] = {
    "archive-runs": archive_runs_main,
    "artifact-cleanup": artifact_cleanup_main,
    "cache-archive": cache_archive_main,
    "cache-inspector": cache_inspector_main,
    "dataset-assets": dataset_assets_main,
    "hf-sync": hf_sync_main,
}


def build_parser() -> argparse.ArgumentParser:
    """构建统一命令入口解析器。"""

    parser = argparse.ArgumentParser(description="Unified CLI for research experiments.")
    subparsers = parser.add_subparsers(dest="group", required=True)

    family = subparsers.add_parser("family", help="Run one experiment family command.")
    family.add_argument("family_name", choices=registered_family_names())
    family.add_argument("family_args", nargs=argparse.REMAINDER)

    matrix = subparsers.add_parser("matrix", help="Run faithful matrix commands.")
    matrix.add_argument("matrix_args", nargs=argparse.REMAINDER)

    tools = subparsers.add_parser("tools", help="Run workspace and archive tools.")
    tools.add_argument("tool_name", choices=tuple(sorted(TOOL_MAINS)))
    tools.add_argument("tool_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> None:
    """统一入口。"""

    configure_utf8_stdio()
    args = build_parser().parse_args(argv)

    if args.group == "family":
        get_family_spec(args.family_name).cli_main(args.family_args)
        return

    if args.group == "matrix":
        matrix_main(args.matrix_args)
        return

    if args.group == "tools":
        TOOL_MAINS[args.tool_name](args.tool_args)
        return

    raise RuntimeError(f"Unsupported command group: {args.group}")
