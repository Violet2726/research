"""工作区布局、归档与资产工具入口。"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS: dict[str, tuple[str, str]] = {
    "ArchiveGroupPlan": ("research_experiments.workspace.run_archives", "ArchiveGroupPlan"),
    "WorkspaceLayout": ("research_experiments.workspace.layout", "WorkspaceLayout"),
    "auto_publish_runs_enabled": ("research_experiments.workspace.layout", "auto_publish_runs_enabled"),
    "auto_push_cache_snapshot_enabled": ("research_experiments.workspace.layout", "auto_push_cache_snapshot_enabled"),
    "build_cache_snapshot": ("research_experiments.workspace.cache_snapshots", "build_cache_snapshot"),
    "build_primary_dataset_specs": ("research_experiments.workspace.dataset_assets", "build_primary_dataset_specs"),
    "build_runtime_support_dataset_specs": ("research_experiments.workspace.dataset_assets", "build_runtime_support_dataset_specs"),
    "build_supplementary_dataset_specs": ("research_experiments.workspace.dataset_assets", "build_supplementary_dataset_specs"),
    "collect_dataset_inventory": ("research_experiments.workspace.dataset_assets", "collect_dataset_inventory"),
    "collect_hf_sync_status": ("research_experiments.workspace.hf_sync", "collect_hf_sync_status"),
    "default_cache_hf_repo": ("research_experiments.workspace.layout", "default_cache_hf_repo"),
    "default_cache_root": ("research_experiments.workspace.layout", "default_cache_root"),
    "default_datasets_root": ("research_experiments.workspace.layout", "default_datasets_root"),
    "default_files_root": ("research_experiments.workspace.layout", "default_files_root"),
    "default_reports_root": ("research_experiments.workspace.layout", "default_reports_root"),
    "default_runs_hf_repo": ("research_experiments.workspace.layout", "default_runs_hf_repo"),
    "default_runs_root": ("research_experiments.workspace.layout", "default_runs_root"),
    "discover_publishable_runs": ("research_experiments.workspace.hf_sync", "discover_publishable_runs"),
    "discover_used_benchmark_config_paths": ("research_experiments.workspace.dataset_assets", "discover_used_benchmark_config_paths"),
    "download_primary_dataset_sources": ("research_experiments.workspace.dataset_assets", "download_primary_dataset_sources"),
    "download_runtime_support_dataset_sources": ("research_experiments.workspace.dataset_assets", "download_runtime_support_dataset_sources"),
    "download_supplementary_dataset_sources": ("research_experiments.workspace.dataset_assets", "download_supplementary_dataset_sources"),
    "extract_run_archives": ("research_experiments.workspace.run_archives", "extract_run_archives"),
    "fetch_run_from_hub": ("research_experiments.workspace.run_archives", "fetch_run_from_hub"),
    "load_used_benchmark_configs": ("research_experiments.workspace.dataset_assets", "load_used_benchmark_configs"),
    "pack_run_artifacts": ("research_experiments.workspace.run_archives", "pack_run_artifacts"),
    "prepare_all_dataset_sources": ("research_experiments.workspace.dataset_assets", "prepare_all_dataset_sources"),
    "prepare_used_datasets": ("research_experiments.workspace.dataset_assets", "prepare_used_datasets"),
    "publish_run_if_configured": ("research_experiments.workspace.run_archives", "publish_run_if_configured"),
    "publish_run_to_hub": ("research_experiments.workspace.run_archives", "publish_run_to_hub"),
    "pull_latest_cache_snapshot": ("research_experiments.workspace.cache_snapshots", "pull_latest_cache_snapshot"),
    "pull_workspace_from_hub": ("research_experiments.workspace.hf_sync", "pull_workspace_from_hub"),
    "push_cache_snapshot_if_configured": ("research_experiments.workspace.cache_snapshots", "push_cache_snapshot_if_configured"),
    "push_latest_cache_snapshot": ("research_experiments.workspace.cache_snapshots", "push_latest_cache_snapshot"),
    "push_workspace_to_hub": ("research_experiments.workspace.hf_sync", "push_workspace_to_hub"),
    "regenerate_used_dataset_splits": ("research_experiments.workspace.dataset_assets", "regenerate_used_dataset_splits"),
    "restore_cache_snapshot": ("research_experiments.workspace.cache_snapshots", "restore_cache_snapshot"),
    "validate_archive_contract": ("research_experiments.workspace.run_archives", "validate_archive_contract"),
    "workspace_defaults": ("research_experiments.workspace.layout", "workspace_defaults"),
    "workspace_layout": ("research_experiments.workspace.layout", "workspace_layout"),
    "write_dataset_inventory_files": ("research_experiments.workspace.dataset_assets", "write_dataset_inventory_files"),
}


def __getattr__(name: str) -> Any:
    """按需加载工作区子模块中的公开符号，避免包初始化阶段循环依赖。"""

    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_path), attr_name)
    globals()[name] = value
    return value


__all__ = sorted(_EXPORTS)
