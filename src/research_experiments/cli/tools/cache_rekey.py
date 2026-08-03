"""用于一次性、与完成上限无关的缓存迁移的命令行工具。"""

from __future__ import annotations

import argparse
from pathlib import Path

from research_experiments.cli_support.output import configure_utf8_stdio, emit_json
from research_experiments.workspace.cache_rekey import apply_cache_rekey, inspect_cache_rekey


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect or atomically rebuild cache entries under the cap-independent key policy."
    )
    parser.add_argument("--cache-root", default="local/cache")
    parser.add_argument("--apply", action="store_true", help="Build a temporary cache root and atomically replace cache-root.")
    parser.add_argument(
        "--temporary-root",
        default=None,
        help="Optional empty sibling directory used during --apply.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    configure_utf8_stdio()
    args = build_parser().parse_args(argv)
    root = Path(args.cache_root)
    if args.apply:
        emit_json(
            apply_cache_rekey(
                root,
                temporary_root=Path(args.temporary_root) if args.temporary_root else None,
            )
        )
        return
    emit_json(inspect_cache_rekey(root))
