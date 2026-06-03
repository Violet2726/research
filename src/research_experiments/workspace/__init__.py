"""工作区布局、归档与资产工具入口。"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "WorkspaceLayout": ("research_experiments.workspace.layout", "WorkspaceLayout"),
    "auto_publish_runs_enabled": ("research_experiments.workspace.layout", "auto_publish_runs_enabled"),
    "auto_push_cache_snapshot_enabled": ("research_experiments.workspace.layout", "auto_push_cache_snapshot_enabled"),
    "build_primary_dataset_specs": ("research_experiments.workspace.datasets", "build_primary_dataset_specs"),
    "build_runtime_support_dataset_specs": ("research_experiments.workspace.datasets", "build_runtime_support_dataset_specs"),
    "build_supplementary_dataset_specs": ("research_experiments.workspace.datasets", "build_supplementary_dataset_specs"),
    "compute_run_bundle_sha256": ("research_experiments.workspace.hf", "compute_run_bundle_sha256"),
    "collect_dataset_inventory": ("research_experiments.workspace.datasets", "collect_dataset_inventory"),
    "pull_cache_from_hub": ("research_experiments.workspace.hf", "pull_cache_from_hub"),
    "push_cache_if_configured": ("research_experiments.workspace.hf", "push_cache_if_configured"),
    "push_cache_to_hub": ("research_experiments.workspace.hf", "push_cache_to_hub"),
    "default_cache_hf_repo": ("research_experiments.workspace.layout", "default_cache_hf_repo"),
    "default_cache_root": ("research_experiments.workspace.layout", "default_cache_root"),
    "default_datasets_root": ("research_experiments.workspace.layout", "default_datasets_root"),
    "default_files_root": ("research_experiments.workspace.layout", "default_files_root"),
    "default_reports_root": ("research_experiments.workspace.layout", "default_reports_root"),
    "default_runs_hf_repo": ("research_experiments.workspace.layout", "default_runs_hf_repo"),
    "default_runs_root": ("research_experiments.workspace.layout", "default_runs_root"),
    "discover_publishable_runs": ("research_experiments.workspace.hf", "discover_publishable_runs"),
    "discover_used_benchmark_config_paths": ("research_experiments.workspace.datasets", "discover_used_benchmark_config_paths"),
    "download_primary_dataset_sources": ("research_experiments.workspace.datasets", "download_primary_dataset_sources"),
    "download_runtime_support_dataset_sources": ("research_experiments.workspace.datasets", "download_runtime_support_dataset_sources"),
    "download_supplementary_dataset_sources": ("research_experiments.workspace.datasets", "download_supplementary_dataset_sources"),
    "load_used_benchmark_configs": ("research_experiments.workspace.datasets", "load_used_benchmark_configs"),
    "pack_run_artifacts": ("research_experiments.workspace.hf", "pack_run_artifacts"),
    "prepare_all_dataset_sources": ("research_experiments.workspace.datasets", "prepare_all_dataset_sources"),
    "prepare_used_datasets": ("research_experiments.workspace.datasets", "prepare_used_datasets"),
    "publish_run_if_configured": ("research_experiments.workspace.hf", "publish_run_if_configured"),
    "pull_runs_from_hub": ("research_experiments.workspace.hf", "pull_runs_from_hub"),
    "push_runs_to_hub": ("research_experiments.workspace.hf", "push_runs_to_hub"),
    "resolve_local_cache_hash": ("research_experiments.workspace.hf", "resolve_local_cache_hash"),
    "regenerate_used_dataset_splits": ("research_experiments.workspace.datasets", "regenerate_used_dataset_splits"),
    "validate_archive_contract": ("research_experiments.workspace.hf", "validate_archive_contract"),
    "workspace_defaults": ("research_experiments.workspace.layout", "workspace_defaults"),
    "workspace_layout": ("research_experiments.workspace.layout", "workspace_layout"),
    "write_dataset_inventory_files": ("research_experiments.workspace.datasets", "write_dataset_inventory_files"),
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
