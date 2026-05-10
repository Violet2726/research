"""run 归档与 Hugging Face 发布命令。"""

from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from experiment_core.foundation.run_archives import fetch_run_from_hub, pack_run_artifacts, publish_run_to_hub
from experiment_core.foundation.workspace import workspace_layout


def build_parser() -> argparse.ArgumentParser:
    """构造 run 归档命令行参数。"""
    parser = argparse.ArgumentParser(description="打包、发布与回取 run 级科研归档。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pack_run = subparsers.add_parser("pack-run", help="为单个 run 生成 archive_manifest 与 tar.zst 归档包。")
    pack_run.add_argument("--run-root", required=True)
    pack_run.add_argument("--json", action="store_true")

    publish_run = subparsers.add_parser("publish-run", help="把单个 run 发布到 Hugging Face dataset repo。")
    publish_run.add_argument("--run-root", required=True)
    publish_run.add_argument("--repo", required=True)
    publish_run.add_argument("--token")
    publish_run.add_argument("--no-create-repo", action="store_true")
    publish_run.add_argument("--json", action="store_true")

    fetch_run = subparsers.add_parser("fetch-run", help="从 Hugging Face dataset repo 回取单个 run。")
    fetch_run.add_argument("--run-id", required=True)
    fetch_run.add_argument("--repo", required=True)
    fetch_run.add_argument("--include", choices=["traces", "predictions", "all"], default="all")
    fetch_run.add_argument("--target-root", default=workspace_layout().runs_root.as_posix())
    fetch_run.add_argument("--token")
    fetch_run.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    """命令行入口。"""
    load_dotenv(".env.local", override=False)
    args = build_parser().parse_args()
    if args.command == "pack-run":
        payload = pack_run_artifacts(args.run_root)
    elif args.command == "publish-run":
        payload = publish_run_to_hub(
            args.run_root,
            repo_id=args.repo,
            token=args.token,
            create_repo=not args.no_create_repo,
        )
    elif args.command == "fetch-run":
        payload = fetch_run_from_hub(
            args.run_id,
            repo_id=args.repo,
            include=args.include,
            token=args.token,
            target_root=args.target_root,
        )
    else:  # pragma: no cover - argparse 已保证不会走到这里
        raise RuntimeError(f"Unsupported command: {args.command}")

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    print(json.dumps(payload, ensure_ascii=False, indent=2))
