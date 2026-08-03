"""统一命令入口。"""

from __future__ import annotations

import argparse
from collections.abc import Callable

from research_experiments.cli.family import dispatch_family_cli
from research_experiments.cli.tools.artifact_cleanup import main as artifact_cleanup_main
from research_experiments.cli.tools.cache_rekey import main as cache_rekey_main
from research_experiments.cli.tools.dataset_assets import main as dataset_assets_main
from research_experiments.cli.tools.hf import main as hf_main
from research_experiments.cli_support.output import configure_utf8_stdio
from research_experiments.families.registry import get_family_registration, registered_family_names
from research_experiments.matrix.cli import main as matrix_main

ToolMain = Callable[[list[str] | None], None]

TOOL_MAINS: dict[str, ToolMain] = {
    "artifact-cleanup": artifact_cleanup_main,
    "cache-rekey": cache_rekey_main,
    "dataset-assets": dataset_assets_main,
    "hf": hf_main,
}


def build_parser() -> argparse.ArgumentParser:
    """构建统一命令入口解析器。"""

    parser = argparse.ArgumentParser(description="研究实验平台统一命令入口。")
    subparsers = parser.add_subparsers(dest="group", required=True)

    experiment = subparsers.add_parser("experiment", help="运行某个实验家族的命令。")
    experiment.add_argument("--family", required=True, choices=registered_family_names())
    experiment.add_argument("experiment_args", nargs=argparse.REMAINDER)

    matrix = subparsers.add_parser("matrix", help="运行矩阵编排与分析命令。")
    matrix.add_argument("matrix_args", nargs=argparse.REMAINDER)

    tools = subparsers.add_parser("tools", help="运行工作区、归档与缓存工具。")
    tools.add_argument("tool_name", choices=tuple(sorted(TOOL_MAINS)))
    tools.add_argument("tool_args", nargs=argparse.REMAINDER)
    return parser


def main(argv: list[str] | None = None) -> None:
    """统一入口。"""

    configure_utf8_stdio()
    args = build_parser().parse_args(argv)

    if args.group == "experiment":
        dispatch_family_cli(get_family_registration(args.family), args.experiment_args)
        return

    if args.group == "matrix":
        matrix_main(args.matrix_args)
        return

    if args.group == "tools":
        TOOL_MAINS[args.tool_name](args.tool_args)
        return

    raise RuntimeError(f"Unsupported command group: {args.group}")

