"""统一管理仓库中的工作目录与环境变量覆盖。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os


RUNS_ROOT_ENV = "RESEARCH_RUNS_ROOT"
REPORTS_ROOT_ENV = "RESEARCH_REPORTS_ROOT"
CACHE_ROOT_ENV = "RESEARCH_CACHE_ROOT"
FILES_ROOT_ENV = "RESEARCH_FILES_ROOT"


@dataclass(frozen=True)
class WorkspaceLayout:
    """仓库工作目录布局。"""

    runs_root: Path
    reports_root: Path
    cache_root: Path
    files_root: Path


def workspace_layout() -> WorkspaceLayout:
    """读取当前环境下生效的工作目录布局。"""
    return WorkspaceLayout(
        runs_root=Path(os.getenv(RUNS_ROOT_ENV, "runs")),
        reports_root=Path(os.getenv(REPORTS_ROOT_ENV, "reports")),
        cache_root=Path(os.getenv(CACHE_ROOT_ENV, "cache")),
        files_root=Path(os.getenv(FILES_ROOT_ENV, "files")),
    )


def default_runs_root(experiment_kind: str) -> str:
    """返回某类实验默认的运行产物根目录。"""
    return _to_posix(workspace_layout().runs_root / experiment_kind)


def default_reports_root(experiment_kind: str) -> str:
    """返回某类实验默认的报告发布根目录。"""
    return _to_posix(workspace_layout().reports_root / experiment_kind)


def default_cache_root() -> str:
    """返回共享缓存根目录。"""
    return _to_posix(workspace_layout().cache_root)


def default_files_root() -> str:
    """返回仓库通用资料目录的根路径。"""
    return _to_posix(workspace_layout().files_root)


def workspace_defaults(experiment_kind: str | None = None) -> dict[str, Any]:
    """导出当前生效的工作目录配置，供 CLI 和文档展示。"""
    layout = workspace_layout()
    payload: dict[str, Any] = {
        "runs_root": _to_posix(layout.runs_root),
        "reports_root": _to_posix(layout.reports_root),
        "cache_root": _to_posix(layout.cache_root),
        "files_root": _to_posix(layout.files_root),
        "env_overrides": {
            "runs_root": RUNS_ROOT_ENV,
            "reports_root": REPORTS_ROOT_ENV,
            "cache_root": CACHE_ROOT_ENV,
            "files_root": FILES_ROOT_ENV,
        },
    }
    if experiment_kind is not None:
        payload["experiment_kind"] = experiment_kind
        payload["experiment_runs_root"] = default_runs_root(experiment_kind)
        payload["experiment_reports_root"] = default_reports_root(experiment_kind)
        payload["experiment_cache_root"] = default_cache_root()
    return payload


def _to_posix(path: Path) -> str:
    return path.as_posix()
