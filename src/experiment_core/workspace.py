"""统一管理仓库内的工作区目录与默认输出路径。

本模块只负责回答两类问题：
1. 运行产物默认应该写到哪里；
2. 用户如果想把输出重定向到别处，应该通过哪些环境变量覆盖。

这样各实验包就不需要在 CLI、runner、reporting 里分别硬编码
`local/runs/...`、`local/reports/...`、`cache/...sqlite` 这些路径。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os


LOCAL_ROOT_ENV = "RESEARCH_LOCAL_ROOT"
CACHE_ROOT_ENV = "RESEARCH_CACHE_ROOT"
FILES_ROOT_ENV = "RESEARCH_FILES_ROOT"


@dataclass(frozen=True)
class WorkspaceLayout:
    """仓库内工作目录的顶层布局。"""

    local_root: Path
    cache_root: Path
    files_root: Path

    @property
    def runs_root(self) -> Path:
        return self.local_root / "runs"

    @property
    def reports_root(self) -> Path:
        return self.local_root / "reports"


def workspace_layout() -> WorkspaceLayout:
    """读取当前环境下生效的工作区目录布局。"""
    return WorkspaceLayout(
        local_root=Path(os.getenv(LOCAL_ROOT_ENV, "local")),
        cache_root=Path(os.getenv(CACHE_ROOT_ENV, "cache")),
        files_root=Path(os.getenv(FILES_ROOT_ENV, "files")),
    )


def default_runs_root(experiment_kind: str) -> str:
    """返回某条实验线默认的运行产物根目录。"""
    return _to_posix(workspace_layout().runs_root / experiment_kind)


def default_reports_root(experiment_kind: str) -> str:
    """返回某条实验线默认的报告发布目录。"""
    return _to_posix(workspace_layout().reports_root / experiment_kind)


def default_cache_path(experiment_kind: str) -> str:
    """返回某条实验线默认的请求缓存 SQLite 路径。"""
    return _to_posix(workspace_layout().cache_root / f"{experiment_kind}_requests.sqlite")


def default_files_root() -> str:
    """返回仓库内通用资料目录的根路径。"""
    return _to_posix(workspace_layout().files_root)


def workspace_defaults(experiment_kind: str | None = None) -> dict[str, Any]:
    """导出当前生效的目录配置，便于 CLI/文档直接展示。"""
    layout = workspace_layout()
    payload: dict[str, Any] = {
        "local_root": _to_posix(layout.local_root),
        "runs_root": _to_posix(layout.runs_root),
        "reports_root": _to_posix(layout.reports_root),
        "cache_root": _to_posix(layout.cache_root),
        "files_root": _to_posix(layout.files_root),
        "env_overrides": {
            "local_root": LOCAL_ROOT_ENV,
            "cache_root": CACHE_ROOT_ENV,
            "files_root": FILES_ROOT_ENV,
        },
    }
    if experiment_kind is not None:
        payload["experiment_kind"] = experiment_kind
        payload["experiment_runs_root"] = default_runs_root(experiment_kind)
        payload["experiment_reports_root"] = default_reports_root(experiment_kind)
        payload["experiment_cache_path"] = default_cache_path(experiment_kind)
    return payload


def _to_posix(path: Path) -> str:
    return path.as_posix()
