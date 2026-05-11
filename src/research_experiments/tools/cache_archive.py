"""cache 最新快照同步命令。"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from research_experiments.core.foundation.cache_snapshots import pull_latest_cache_snapshot, push_latest_cache_snapshot
from research_experiments.core.foundation.cli_output import configure_utf8_stdio, emit_json
from research_experiments.core.foundation.workspace import default_cache_hf_repo, default_cache_root


def build_parser() -> argparse.ArgumentParser:
    """构造 cache 快照命令行参数。"""

    parser = argparse.ArgumentParser(description="同步 cache 的 latest-only Hugging Face 快照。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    push_latest = subparsers.add_parser("push-latest", help="压缩并推送当前 cache 最新快照。")
    push_latest.add_argument("--cache-root", default=default_cache_root())
    push_latest.add_argument("--cache-shard", action="append", default=[], help="仅同步某个或某些分库目录，可重复传入。")
    push_latest.add_argument("--repo")
    push_latest.add_argument("--token")
    push_latest.add_argument("--public", action="store_true", help="默认按私有 dataset repo 创建；显式指定后改为公开。")
    push_latest.add_argument("--no-create-repo", action="store_true")
    push_latest.add_argument("--json", action="store_true")

    pull_latest = subparsers.add_parser("pull-latest", help="从远端 latest-only 快照恢复 cache。")
    pull_latest.add_argument("--target", required=True)
    pull_latest.add_argument("--cache-shard", action="append", default=[], help="仅恢复某个或某些分库目录，可重复传入。")
    pull_latest.add_argument("--repo")
    pull_latest.add_argument("--token")
    pull_latest.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    """命令行入口。"""

    load_dotenv(".env.local", override=False)
    configure_utf8_stdio()
    args = build_parser().parse_args(argv)
    if args.command == "push-latest":
        payload = push_latest_cache_snapshot(
            args.cache_root,
            repo_id=_require_repo(args.repo),
            token=args.token,
            create_repo=not args.no_create_repo,
            private=not args.public,
            shard_filters=args.cache_shard,
        )
    elif args.command == "pull-latest":
        payload = pull_latest_cache_snapshot(
            args.target,
            repo_id=_require_repo(args.repo),
            token=args.token,
            shard_filters=args.cache_shard,
        )
    else:  # pragma: no cover - argparse 已保证不会走到这里
        raise RuntimeError(f"Unsupported command: {args.command}")

    if args.json:
        emit_json(payload)
        return
    emit_json(payload)


def _require_repo(explicit_repo: str | None) -> str:
    repo_id = (explicit_repo or default_cache_hf_repo() or "").strip()
    if not repo_id:
        raise RuntimeError("缺少 cache Hugging Face repo；请传 `--repo` 或配置 `RESEARCH_CACHE_HF_REPO`。")
    return repo_id


if __name__ == "__main__":
    main()
