"""项目已用数据集的下载、盘点与 split 重建 CLI。"""

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from research_experiments.cli_support.output import configure_utf8_stdio, emit_json
from research_experiments.workspace.dataset_assets import (
    build_primary_dataset_specs,
    build_supplementary_dataset_specs,
    download_primary_dataset_sources,
    download_supplementary_dataset_sources,
    load_used_benchmark_configs,
    prepare_all_dataset_sources,
    prepare_used_datasets,
    regenerate_used_dataset_splits,
    write_dataset_inventory_files,
)


def build_parser() -> argparse.ArgumentParser:
    """构建数据集资产命令行。"""

    load_dotenv(".env.local", override=False)
    parser = argparse.ArgumentParser(description="下载并准备项目 benchmark 数据集。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_used = subparsers.add_parser("list-used", help="列出项目当前使用的 benchmark 与数据集资产。")
    list_used.add_argument("--configs-root", default="configs")

    download_used = subparsers.add_parser("download-used", help="下载主评测源文件。")
    download_used.add_argument("--configs-root", default="configs")
    download_used.add_argument("--force", action="store_true")

    download_training = subparsers.add_parser("download-training", help="下载训练集与可用的非测试补充源。")
    download_training.add_argument("--configs-root", default="configs")
    download_training.add_argument("--force", action="store_true")

    generate = subparsers.add_parser("generate-splits", help="为当前项目使用的数据集重建冻结 split。")
    generate.add_argument("--configs-root", default="configs")
    generate.add_argument("--splits-root", default="configs/core/shared/benchmarks/splits")

    prepare = subparsers.add_parser("prepare-used", help="下载主评测源、重建 split，并刷新说明文档。")
    prepare.add_argument("--configs-root", default="configs")
    prepare.add_argument("--splits-root", default="configs/core/shared/benchmarks/splits")
    prepare.add_argument("--force", action="store_true")

    prepare_all = subparsers.add_parser("prepare-all-sources", help="下载主评测源与训练补充源，并重建 split。")
    prepare_all.add_argument("--configs-root", default="configs")
    prepare_all.add_argument("--splits-root", default="configs/core/shared/benchmarks/splits")
    prepare_all.add_argument("--force", action="store_true")

    refresh = subparsers.add_parser("refresh-readme", help="刷新 datasets/README.md 与 local/datasets/manifest.json。")
    refresh.add_argument("--configs-root", default="configs")
    refresh.add_argument("--splits-root", default="configs/core/shared/benchmarks/splits")

    return parser


def main(argv: list[str] | None = None) -> None:
    """解析并执行数据集资产命令。"""

    configure_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list-used":
        benchmarks = load_used_benchmark_configs(args.configs_root)
        payload = {
            "benchmark_count": len(benchmarks),
            "benchmarks": [
                {
                    "slug": benchmark.slug,
                    "name": benchmark.name,
                    "source_path": benchmark.source_path,
                    "source_split": benchmark.source_split,
                    "loader": benchmark.loader,
                }
                for benchmark in benchmarks
            ],
            "primary_assets": [_serialize_spec(spec) for spec in build_primary_dataset_specs(benchmarks)],
            "supplementary_assets": [_serialize_spec(spec) for spec in build_supplementary_dataset_specs(benchmarks)],
        }
        emit_json(payload)
        return

    if args.command == "download-used":
        benchmarks = load_used_benchmark_configs(args.configs_root)
        results = download_primary_dataset_sources(benchmarks, force=args.force)
        emit_json({"download_count": len(results), "downloads": [_serialize_result(item) for item in results]})
        return

    if args.command == "download-training":
        benchmarks = load_used_benchmark_configs(args.configs_root)
        results = download_supplementary_dataset_sources(benchmarks, force=args.force)
        emit_json({"download_count": len(results), "downloads": [_serialize_result(item) for item in results]})
        return

    if args.command == "generate-splits":
        benchmarks = load_used_benchmark_configs(args.configs_root)
        created = regenerate_used_dataset_splits(benchmarks, output_dir=args.splits_root)
        emit_json({"split_count": len(created), "splits": [path.as_posix() for path in created]})
        return

    if args.command == "prepare-used":
        emit_json(prepare_used_datasets(configs_root=args.configs_root, splits_root=args.splits_root, force=args.force))
        return

    if args.command == "prepare-all-sources":
        emit_json(prepare_all_dataset_sources(configs_root=args.configs_root, splits_root=args.splits_root, force=args.force))
        return

    if args.command == "refresh-readme":
        benchmarks = load_used_benchmark_configs(args.configs_root)
        paths = write_dataset_inventory_files(benchmarks, docs_root="datasets", splits_root=args.splits_root)
        emit_json({key: value.as_posix() for key, value in paths.items()})
        return

    parser.error(f"Unsupported command: {args.command}")


def _serialize_spec(spec) -> dict[str, object]:
    return {
        "slug": spec.slug,
        "dataset_name": spec.dataset_name,
        "asset_id": spec.asset_id,
        "purpose": spec.purpose,
        "relative_path": spec.relative_path.as_posix(),
        "source_label": spec.source_label,
        "source_url": spec.source_url,
        "source_split": spec.source_split,
        "source_kind": spec.source_kind,
    }


def _serialize_result(result) -> dict[str, object]:
    return {
        "slug": result.slug,
        "asset_id": result.asset_id,
        "purpose": result.purpose,
        "local_path": result.local_path.as_posix(),
        "source_label": result.source_label,
        "source_url": result.source_url,
        "status": result.status,
        "size_bytes": result.size_bytes,
    }


if __name__ == "__main__":
    main()

