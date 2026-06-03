"""统一的 Hugging Face 同步命令入口。"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from research_experiments.cli_support.output import configure_utf8_stdio, emit_json
from research_experiments.workspace.hf import (
    pull_cache_from_hub,
    pull_runs_from_hub,
    push_cache_to_hub,
    push_runs_to_hub,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the unified Hugging Face parser."""

    parser = argparse.ArgumentParser(description="统一管理 runs 与 cache 的 Hugging Face 同步。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    push_cache = subparsers.add_parser("push-cache", help="推送整个本地 cache 工作区。")
    push_cache.add_argument("--json", action="store_true")

    pull_cache = subparsers.add_parser("pull-cache", help="拉取整个远端 cache 工作区。")
    pull_cache.add_argument("--json", action="store_true")

    push_runs = subparsers.add_parser("push-runs", help="推送本地 runs。")
    push_runs.add_argument("--source", action="append", default=[], help="runs 根目录内的完整 run 或其父级目录，可重复传入。")
    push_runs.add_argument("--skip-validation", action="store_true", help="不检查 run_validation.json 或矩阵完整性，直接推送。")
    push_runs.add_argument("--json", action="store_true")

    pull_runs = subparsers.add_parser("pull-runs", help="拉取远端 runs。")
    pull_runs.add_argument("--prefix", action="append", default=[], help="远端 runs repo 内的前缀，可重复传入。")
    pull_runs.add_argument("--recent-hours", type=float, help="仅拉取最近 N 小时内发布的 runs。")
    pull_runs.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""

    load_dotenv(".env.local", override=False)
    configure_utf8_stdio()
    args = build_parser().parse_args(argv)

    if args.command == "push-cache":
        payload = push_cache_to_hub()
    elif args.command == "pull-cache":
        payload = pull_cache_from_hub()
    elif args.command == "push-runs":
        payload = push_runs_to_hub(
            sources=args.source,
            skip_validation=args.skip_validation,
        )
    elif args.command == "pull-runs":
        payload = pull_runs_from_hub(
            prefixes=args.prefix,
            recent_hours=args.recent_hours,
        )
    else:  # pragma: no cover
        raise RuntimeError(f"Unsupported command: {args.command}")

    emit_json(payload)


if __name__ == "__main__":
    main()
