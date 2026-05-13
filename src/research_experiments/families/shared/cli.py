"""family 子命令共用的 CLI 构造与分发脚手架。"""

from __future__ import annotations

import argparse
from typing import Any, Callable

from dotenv import load_dotenv

from research_experiments.cli_support.output import configure_utf8_stdio, emit_json
from research_experiments.workspace.layout import (
    default_cache_root,
    default_reports_root,
    default_runs_root,
)


InspectPayloadBuilder = Callable[[str, str | None], dict[str, Any]]
RunCommand = Callable[[argparse.Namespace], Any]
ExtraDispatch = Callable[[argparse.Namespace], bool]
ValidateCommand = Callable[[argparse.Namespace], dict[str, Any]]
RenderCommand = Callable[[argparse.Namespace], dict[str, Any]]


def build_standard_family_parser(
    *,
    family_name: str,
    description: str,
    inspect_help: str,
    run_help: str,
    summarize_help: str,
    validate_help: str,
    report_help: str,
    include_resume_run_dir: bool = False,
) -> argparse.ArgumentParser:
    """构建统一 family 命令面。"""

    load_dotenv(".env.local", override=False)
    parser = argparse.ArgumentParser(description=description)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect-experiment", help=inspect_help)
    inspect.add_argument("--experiment", required=True)
    inspect.add_argument("--model", default=None)

    run = subparsers.add_parser("run", help=run_help)
    run.add_argument("--experiment", required=True)
    run.add_argument("--phase", required=True)
    run.add_argument("--model", default=None)
    run.add_argument("--runs-root", default=default_runs_root(family_name))
    run.add_argument("--cache-root", default=default_cache_root())
    if include_resume_run_dir:
        run.add_argument("--resume-run-dir", default=None)

    summarize = subparsers.add_parser("summarize-run", help=summarize_help)
    summarize.add_argument("--run-dir", required=True)

    validate = subparsers.add_parser("validate-run", help=validate_help)
    validate.add_argument("--run-dir", required=True)

    report = subparsers.add_parser("render-report", help=report_help)
    report.add_argument("--run-dir", required=True)
    report.add_argument("--publish-dir", default=default_reports_root(family_name))

    return parser


def dispatch_standard_family_cli(
    *,
    parser: argparse.ArgumentParser,
    inspect_payload_builder: InspectPayloadBuilder,
    run_command: RunCommand,
    summarize_run: Callable[[str], dict[str, Any]],
    validate_run: Callable[[str], dict[str, Any]],
    render_report: Callable[[str, str | None], dict[str, Any]],
    extra_dispatch: ExtraDispatch | None = None,
    validate_command: ValidateCommand | None = None,
    render_command: RenderCommand | None = None,
    argv: list[str] | None = None,
) -> None:
    """分发统一 family 命令面。"""

    configure_utf8_stdio()
    args = parser.parse_args(argv)

    if extra_dispatch is not None and extra_dispatch(args):
        return

    if args.command == "inspect-experiment":
        emit_json(inspect_payload_builder(args.experiment, args.model))
        return

    if args.command == "run":
        print(run_command(args).as_posix())
        return

    if args.command == "summarize-run":
        emit_json(summarize_run(args.run_dir))
        return

    if args.command == "validate-run":
        emit_json(validate_command(args) if validate_command is not None else validate_run(args.run_dir))
        return

    if args.command == "render-report":
        emit_json(render_command(args) if render_command is not None else render_report(args.run_dir, publish_dir=args.publish_dir))
        return

    parser.error(f"Unsupported command: {args.command}")


