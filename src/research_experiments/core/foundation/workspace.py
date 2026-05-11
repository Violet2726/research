"""统一管理仓库中的工作目录、远程归档配置与环境变量覆盖。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os


RUNS_ROOT_ENV = "RESEARCH_RUNS_ROOT"
REPORTS_ROOT_ENV = "RESEARCH_REPORTS_ROOT"
CACHE_ROOT_ENV = "RESEARCH_CACHE_ROOT"
DATASETS_ROOT_ENV = "RESEARCH_DATASETS_ROOT"
FILES_ROOT_ENV = "RESEARCH_FILES_ROOT"
RUNS_HF_REPO_ENV = "RESEARCH_RUNS_HF_REPO"
CACHE_HF_REPO_ENV = "RESEARCH_CACHE_HF_REPO"
AUTO_PUBLISH_RUNS_ENV = "RESEARCH_AUTO_PUBLISH_RUNS"
AUTO_PUSH_CACHE_SNAPSHOT_ENV = "RESEARCH_AUTO_PUSH_CACHE_SNAPSHOT"


@dataclass(frozen=True)
class WorkspaceLayout:
    """表示当前进程生效的工作目录布局。"""

    runs_root: Path
    reports_root: Path
    cache_root: Path
    datasets_root: Path
    files_root: Path


def workspace_layout() -> WorkspaceLayout:
    """读取当前环境下生效的工作目录布局。"""
    return WorkspaceLayout(
        runs_root=Path(os.getenv(RUNS_ROOT_ENV, "local/runs")),
        reports_root=Path(os.getenv(REPORTS_ROOT_ENV, "local/reports")),
        cache_root=Path(os.getenv(CACHE_ROOT_ENV, "local/cache")),
        datasets_root=Path(os.getenv(DATASETS_ROOT_ENV, "local/datasets")),
        files_root=Path(os.getenv(FILES_ROOT_ENV, "files")),
    )


def default_runs_root(family_name: str) -> str:
    """返回某个 family 默认的运行产物根目录。"""
    return _to_posix(workspace_layout().runs_root / family_name)


def default_reports_root(family_name: str) -> str:
    """返回某个 family 默认的报告发布根目录。"""
    return _to_posix(workspace_layout().reports_root / family_name)


def default_cache_root() -> str:
    """返回共享缓存根目录。"""
    return _to_posix(workspace_layout().cache_root)


def default_datasets_root() -> str:
    """返回本地数据集资产根目录。"""
    return _to_posix(workspace_layout().datasets_root)


def default_files_root() -> str:
    """返回仓库通用资料目录的根路径。"""
    return _to_posix(workspace_layout().files_root)


def default_runs_hf_repo() -> str | None:
    """返回 runs 的 Hugging Face dataset repo。"""
    value = os.getenv(RUNS_HF_REPO_ENV, "").strip()
    return value or None


def default_cache_hf_repo() -> str | None:
    """返回 cache 的 Hugging Face dataset repo。"""
    value = os.getenv(CACHE_HF_REPO_ENV, "").strip()
    return value or None


def auto_publish_runs_enabled() -> bool:
    """判断是否为正式 run 开启自动发布。"""
    return _env_flag(AUTO_PUBLISH_RUNS_ENV)


def auto_push_cache_snapshot_enabled() -> bool:
    """判断是否在批量运行后自动推送 cache 最新快照。"""
    return _env_flag(AUTO_PUSH_CACHE_SNAPSHOT_ENV)


def workspace_defaults(family_name: str | None = None) -> dict[str, Any]:
    """导出当前生效的工作目录配置，供 CLI 和文档展示。"""
    layout = workspace_layout()
    payload: dict[str, Any] = {
        "runs_root": _to_posix(layout.runs_root),
        "reports_root": _to_posix(layout.reports_root),
        "cache_root": _to_posix(layout.cache_root),
        "datasets_root": _to_posix(layout.datasets_root),
        "files_root": _to_posix(layout.files_root),
        "runs_hf_repo": default_runs_hf_repo(),
        "cache_hf_repo": default_cache_hf_repo(),
        "auto_publish_runs": auto_publish_runs_enabled(),
        "auto_push_cache_snapshot": auto_push_cache_snapshot_enabled(),
        "env_overrides": {
            "runs_root": RUNS_ROOT_ENV,
            "reports_root": REPORTS_ROOT_ENV,
            "cache_root": CACHE_ROOT_ENV,
            "datasets_root": DATASETS_ROOT_ENV,
            "files_root": FILES_ROOT_ENV,
            "runs_hf_repo": RUNS_HF_REPO_ENV,
            "cache_hf_repo": CACHE_HF_REPO_ENV,
            "auto_publish_runs": AUTO_PUBLISH_RUNS_ENV,
            "auto_push_cache_snapshot": AUTO_PUSH_CACHE_SNAPSHOT_ENV,
        },
    }
    if family_name is not None:
        payload["family_name"] = family_name
        payload["family_runs_root"] = default_runs_root(family_name)
        payload["family_reports_root"] = default_reports_root(family_name)
        payload["family_cache_root"] = default_cache_root()
    return payload


def _to_posix(path: Path) -> str:
    return path.as_posix()


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}
