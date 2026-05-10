"""缓存分库查看工具。"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiment_core.foundation.cache import (
    CacheRootSummary,
    CacheShardSummary,
    format_bytes,
    inspect_cache_shard,
    resolve_cache_shard_path,
    summarize_cache_root,
)
from experiment_core.foundation.cli_output import configure_utf8_stdio, emit_json
from experiment_core.foundation.workspace import default_cache_root


def build_parser() -> argparse.ArgumentParser:
    """构造缓存查看命令行参数。"""
    parser = argparse.ArgumentParser(description="查看按供应商、请求模型和数据集分层的缓存分库。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    summarize = subparsers.add_parser("summarize", help="汇总缓存分库数量、大小与请求条数。")
    summarize.add_argument("--cache-root", default=default_cache_root())
    summarize.add_argument("--top-shards", type=int, default=10)
    summarize.add_argument("--json", action="store_true", help="输出 JSON，而不是文本摘要。")

    target = subparsers.add_parser("inspect-target", help="查看某个供应商/请求模型/数据集对应的缓存分库。")
    target.add_argument("--cache-root", default=default_cache_root())
    target.add_argument("--provider", required=True)
    target.add_argument("--request-model", required=True)
    target.add_argument("--dataset", required=True)
    target.add_argument("--json", action="store_true", help="输出 JSON，而不是文本摘要。")

    return parser


def main() -> None:
    """命令行入口。"""
    configure_utf8_stdio()
    args = build_parser().parse_args()

    if args.command == "summarize":
        summary = summarize_cache_root(args.cache_root)
        if args.json:
            emit_json(_root_summary_to_dict(summary, top_shards=max(int(args.top_shards), 0)))
            return
        _print_root_summary(summary, top_shards=max(int(args.top_shards), 0))
        return

    if args.command == "inspect-target":
        shard_path = resolve_cache_shard_path(
            args.cache_root,
            provider=args.provider,
            request_model=args.request_model,
            dataset=args.dataset,
        )
        shard = inspect_cache_shard(shard_path, args.cache_root)
        if args.json:
            emit_json(_shard_summary_to_dict(shard, Path(args.cache_root)))
            return
        _print_shard_summary(shard, Path(args.cache_root))
        return

    raise RuntimeError(f"Unsupported command: {args.command}")


def _root_summary_to_dict(summary: CacheRootSummary, top_shards: int) -> dict[str, object]:
    """把缓存根目录汇总结果转换成 JSON 友好结构。"""
    cache_root = summary.cache_root
    return {
        "cache_root": cache_root.as_posix(),
        "shard_count": summary.shard_count,
        "provider_count": summary.provider_count,
        "total_request_count": summary.total_request_count,
        "total_size_bytes": summary.total_size_bytes,
        "total_size_human": format_bytes(summary.total_size_bytes),
        "providers": [
            {
                "provider": item.provider,
                "model_count": item.model_count,
                "dataset_count": item.dataset_count,
                "shard_count": item.shard_count,
                "total_request_count": item.total_request_count,
                "total_size_bytes": item.total_size_bytes,
                "total_size_human": format_bytes(item.total_size_bytes),
            }
            for item in summary.providers
        ],
        "top_shards": [
            _shard_summary_to_dict(item, cache_root)
            for item in summary.shards[:top_shards]
        ],
    }


def _shard_summary_to_dict(shard: CacheShardSummary, cache_root: Path) -> dict[str, object]:
    """把单个缓存分库结果转换成 JSON 友好结构。"""
    try:
        relative_path = shard.shard_path.relative_to(cache_root).as_posix()
    except ValueError:
        relative_path = shard.shard_path.as_posix()
    return {
        "shard_path": shard.shard_path.as_posix(),
        "relative_path": relative_path,
        "provider": shard.provider,
        "request_model": shard.request_model,
        "dataset": shard.dataset,
        "exists": shard.exists,
        "file_size_bytes": shard.file_size_bytes,
        "file_size_human": format_bytes(shard.file_size_bytes),
        "request_count": shard.request_count,
        "error": shard.error,
    }


def _print_root_summary(summary: CacheRootSummary, top_shards: int) -> None:
    """输出缓存根目录的人类可读汇总。"""
    print(f"缓存根目录: {summary.cache_root.as_posix()}")
    print(f"分库数量: {summary.shard_count}")
    print(f"Provider 数量: {summary.provider_count}")
    print(f"请求总数: {summary.total_request_count}")
    print(f"磁盘占用: {format_bytes(summary.total_size_bytes)}")

    print("Provider 分布:")
    if not summary.providers:
        print("  - 无缓存分库")
    for item in summary.providers:
        print(
            "  - "
            f"{item.provider}: 模型 {item.model_count} 个, "
            f"数据集 {item.dataset_count} 个, "
            f"分库 {item.shard_count} 个, "
            f"请求 {item.total_request_count} 条, "
            f"大小 {format_bytes(item.total_size_bytes)}"
        )

    if top_shards <= 0:
        return

    print(f"最大分库 Top {min(top_shards, len(summary.shards))}:")
    if not summary.shards:
        print("  - 无缓存分库")
    for shard in summary.shards[:top_shards]:
        _print_shard_summary(shard, summary.cache_root, indent="  - ")


def _print_shard_summary(shard: CacheShardSummary, cache_root: Path, indent: str = "") -> None:
    """输出单个缓存分库的人类可读信息。"""
    try:
        relative_path = shard.shard_path.relative_to(cache_root).as_posix()
    except ValueError:
        relative_path = shard.shard_path.as_posix()

    request_count = shard.request_count if shard.request_count is not None else "未知"
    line = (
        f"{relative_path} | "
        f"provider {shard.provider} | "
        f"model {shard.request_model} | "
        f"dataset {shard.dataset} | "
        f"大小 {format_bytes(shard.file_size_bytes)} | "
        f"请求 {request_count}"
    )
    if not shard.exists:
        line += " | 未创建"
    if shard.error:
        line += f" | 错误 {shard.error}"
    print(f"{indent}{line}")

