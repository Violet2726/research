"""family 注册驱动的统一 CLI 运行时。"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from research_experiments.cli_support.output import configure_utf8_stdio, emit_json
from research_experiments.core.contracts import FamilyRegistration, FamilyRunRequest
from research_experiments.workspace.layout import (
    default_cache_root,
    default_reports_root,
    default_runs_root,
)


def build_family_parser(registration: FamilyRegistration) -> argparse.ArgumentParser:
    """按注册信息构建统一 family CLI。"""

    load_dotenv(".env.local", override=False)
    parser = argparse.ArgumentParser(description=registration.cli_help.description)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect-experiment", help=registration.cli_help.inspect_help)
    inspect.add_argument("--experiment", required=True)
    inspect.add_argument("--model", default=None)

    run = subparsers.add_parser("run", help=registration.cli_help.run_help)
    run.add_argument("--experiment", required=True)
    run.add_argument("--phase", required=True)
    run.add_argument("--model", default=None)
    run.add_argument("--runs-root", default=default_runs_root(registration.family_name))
    run.add_argument("--cache-root", default=default_cache_root())
    if registration.cli_help.include_resume_run_dir:
        run.add_argument("--resume-run-dir", default=None)

    summarize = subparsers.add_parser("summarize-run", help=registration.cli_help.summarize_help)
    summarize.add_argument("--run-dir", required=True)

    validate = subparsers.add_parser("validate-run", help=registration.cli_help.validate_help)
    validate.add_argument("--run-dir", required=True)

    report = subparsers.add_parser("render-report", help=registration.cli_help.report_help)
    report.add_argument("--run-dir", required=True)
    report.add_argument("--publish-dir", default=default_reports_root(registration.family_name))

    if registration.configure_parser is not None:
        registration.configure_parser(parser)
    return parser


def dispatch_family_cli(registration: FamilyRegistration, argv: list[str] | None = None) -> None:
    """分发某个 family 的统一 CLI。"""

    configure_utf8_stdio()
    parser = build_family_parser(registration)
    args = parser.parse_args(argv)

    if registration.dispatch_extra_command is not None and registration.dispatch_extra_command(args):
        return

    if args.command == "inspect-experiment":
        emit_json(registration.inspect_experiment(args.experiment, args.model))
        return

    if args.command == "run":
        run_dir = registration.run_from_cli(
            FamilyRunRequest(
                experiment_path=args.experiment,
                phase_name=args.phase,
                model_ref=args.model,
                runs_root=args.runs_root,
                cache_root=args.cache_root,
                resume_run_dir=getattr(args, "resume_run_dir", None),
            )
        )
        print(run_dir.as_posix())
        return

    if args.command == "summarize-run":
        emit_json(registration.summarize_run(args.run_dir))
        return

    if args.command == "validate-run":
        payload = (
            registration.validate_from_cli(args)
            if registration.validate_from_cli is not None
            else registration.validate_run(args.run_dir)
        )
        emit_json(payload)
        return

    if args.command == "render-report":
        payload = (
            registration.render_from_cli(args)
            if registration.render_from_cli is not None
            else registration.render_report(args.run_dir, publish_dir=args.publish_dir)
        )
        emit_json(payload)
        return

    parser.error(f"Unsupported command: {args.command}")
