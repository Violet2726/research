"""Hugging Face 工作区同步总控命令。"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from research_experiments.cli_support.output import configure_utf8_stdio, emit_json
from research_experiments.workspace.hf_sync import collect_hf_sync_status, pull_workspace_from_hub, push_workspace_to_hub
from research_experiments.workspace.layout import default_cache_hf_repo, default_cache_root, default_runs_hf_repo, workspace_layout


def build_parser() -> argparse.ArgumentParser:
    """构造 Hugging Face 同步命令行参数。"""

    parser = argparse.ArgumentParser(description="批量同步本地 runs 与 cache 到 Hugging Face。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    push_workspace = subparsers.add_parser("push-workspace", help="批量推送本地 runs 与 cache。")
    push_workspace.add_argument("--runs-root", default=workspace_layout().runs_root.as_posix())
    push_workspace.add_argument("--cache-root", default=default_cache_root())
    push_workspace.add_argument("--run-dir", action="append", default=[], help="仅推送某个或某些本地 run 目录，可重复传入。")
    push_workspace.add_argument("--cache-shard", action="append", default=[], help="仅推送某个或某些 cache 分库目录，可重复传入。")
    push_workspace.add_argument("--runs-repo")
    push_workspace.add_argument("--cache-repo")
    push_workspace.add_argument("--token")
    push_workspace.add_argument("--skip-runs", action="store_true")
    push_workspace.add_argument("--skip-cache", action="store_true")
    push_workspace.add_argument("--no-matrix", action="store_true")
    push_workspace.add_argument("--force-runs", action="store_true")
    push_workspace.add_argument("--public-cache", action="store_true", help="默认按私有 cache repo 创建；显式指定后改为公开。")
    push_workspace.add_argument("--stop-on-error", action="store_true")
    push_workspace.add_argument("--json", action="store_true")

    pull_workspace = subparsers.add_parser("pull-workspace", help="批量回拉远端 runs 与 cache。")
    pull_workspace.add_argument("--runs-root", default=workspace_layout().runs_root.as_posix())
    pull_workspace.add_argument("--cache-root", default=default_cache_root())
    pull_workspace.add_argument("--run-id", action="append", default=[], help="仅回拉某个或某些 run_id，可重复传入。")
    pull_workspace.add_argument("--run-prefix", action="append", default=[], help="仅回拉某个或某些远端 run 目录前缀，可重复传入。")
    pull_workspace.add_argument("--cache-shard", action="append", default=[], help="仅回拉某个或某些 cache 分库目录，可重复传入。")
    pull_workspace.add_argument("--runs-repo")
    pull_workspace.add_argument("--cache-repo")
    pull_workspace.add_argument("--token")
    pull_workspace.add_argument("--skip-runs", action="store_true")
    pull_workspace.add_argument("--skip-cache", action="store_true")
    pull_workspace.add_argument("--keep-existing-runs", action="store_true")
    pull_workspace.add_argument("--json", action="store_true")

    status = subparsers.add_parser("status", help="查看本地与远端同步状态。")
    status.add_argument("--runs-root", default=workspace_layout().runs_root.as_posix())
    status.add_argument("--cache-root", default=default_cache_root())
    status.add_argument("--runs-repo")
    status.add_argument("--cache-repo")
    status.add_argument("--token")
    status.add_argument("--local-only", action="store_true")
    status.add_argument("--no-matrix", action="store_true")
    status.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    """命令行入口。"""

    load_dotenv(".env.local", override=False)
    configure_utf8_stdio()
    args = build_parser().parse_args(argv)

    if args.command == "push-workspace":
        payload = push_workspace_to_hub(
            runs_root=args.runs_root,
            cache_root=args.cache_root,
            runs_repo_id=args.runs_repo or default_runs_hf_repo(),
            cache_repo_id=args.cache_repo or default_cache_hf_repo(),
            token=args.token,
            publish_runs=not args.skip_runs,
            push_cache=not args.skip_cache,
            include_matrix=not args.no_matrix,
            force_runs=args.force_runs,
            selected_run_dirs=args.run_dir,
            cache_shard_filters=args.cache_shard,
            private_cache_repo=not args.public_cache,
            continue_on_error=not args.stop_on_error,
        )
    elif args.command == "pull-workspace":
        payload = pull_workspace_from_hub(
            runs_root=args.runs_root,
            cache_root=args.cache_root,
            runs_repo_id=args.runs_repo or default_runs_hf_repo(),
            cache_repo_id=args.cache_repo or default_cache_hf_repo(),
            token=args.token,
            fetch_runs=not args.skip_runs,
            pull_cache=not args.skip_cache,
            overwrite_runs=not args.keep_existing_runs,
            selected_run_ids=args.run_id,
            selected_run_prefixes=args.run_prefix,
            cache_shard_filters=args.cache_shard,
        )
    elif args.command == "status":
        payload = collect_hf_sync_status(
            runs_root=args.runs_root,
            cache_root=args.cache_root,
            runs_repo_id=args.runs_repo or default_runs_hf_repo(),
            cache_repo_id=args.cache_repo or default_cache_hf_repo(),
            token=args.token,
            include_remote=not args.local_only,
            include_matrix=not args.no_matrix,
        )
    else:  # pragma: no cover - argparse 已保证不会走到这里
        raise RuntimeError(f"Unsupported command: {args.command}")

    if args.json:
        emit_json(payload)
        return
    emit_json(payload)


if __name__ == "__main__":
    main()

