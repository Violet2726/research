"""Shared helpers for the Hugging Face sync stack."""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

import requests
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub import configure_http_backend as hf_configure_http_backend

from research_experiments.core.io import read_json, write_json
from research_experiments.workspace.layout import default_cache_hf_repo, default_runs_hf_repo

CACHE_MANIFEST_FILENAME = "cache_manifest.json"
RUNS_MANIFEST_FILENAME = "runs_manifest.json"
CACHE_HASH_FILENAME = "requests.sqlite.hfhash.json"
HF_RUN_STATE_FILENAME = "hf_run.json"
IGNORED_PUBLISH_STATUS_FILENAME = "hf_publish.json"
HF_SYNC_SCHEMA_VERSION = 1
_HF_HTTP_BACKEND_CONFIGURED = False
_HF_HTTP_TRUST_ENV = True


def utcnow() -> datetime:
    """Return the current UTC time."""

    return datetime.now(UTC)


def iso_utc_now() -> str:
    """Return the current UTC time formatted as ISO 8601."""

    return utcnow().isoformat()


def resolve_runs_repo_id(explicit_repo: str | None = None) -> str:
    """Resolve the runs repo id or raise when unavailable."""

    repo_id = (explicit_repo or default_runs_hf_repo() or "").strip()
    if not repo_id:
        raise RuntimeError("缺少 runs Hugging Face repo；请配置 `RESEARCH_RUNS_HF_REPO`。")
    return repo_id


def resolve_cache_repo_id(explicit_repo: str | None = None) -> str:
    """Resolve the cache repo id or raise when unavailable."""

    repo_id = (explicit_repo or default_cache_hf_repo() or "").strip()
    if not repo_id:
        raise RuntimeError("缺少 cache Hugging Face repo；请配置 `RESEARCH_CACHE_HF_REPO`。")
    return repo_id


def load_json_if_exists(path: str | Path) -> dict[str, Any]:
    """Read a JSON file when present, otherwise return an empty payload."""

    target = Path(path)
    if not target.exists():
        return {}
    return read_json(target)


def run_hf_request[T](operation: Callable[[], T]) -> T:
    """Run a Hugging Face request with one-shot proxy fallback."""

    _configure_hf_http_backend(trust_env=_HF_HTTP_TRUST_ENV)
    try:
        return operation()
    except requests.exceptions.ProxyError:
        if not _HF_HTTP_TRUST_ENV:
            raise
        _configure_hf_http_backend(trust_env=False)
        return operation()


def download_repo_manifest(
    *,
    repo_id: str,
    filename: str,
    token: str | None = None,
    missing_ok: bool = True,
) -> dict[str, Any]:
    """Download a manifest file from a dataset repo."""

    try:
        manifest_path = Path(
            run_hf_request(
                lambda: hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=filename,
                    token=token,
                )
            )
        )
    except Exception as exc:
        if missing_ok:
            return {}
        raise RuntimeError(f"无法读取远端 manifest：{repo_id}/{filename}") from exc
    return load_json_if_exists(manifest_path)


def upload_manifest(
    api: HfApi,
    *,
    repo_id: str,
    filename: str,
    payload: dict[str, Any],
    token: str | None = None,
    commit_message: str,
) -> None:
    """Upload a JSON manifest file to a dataset repo."""

    with tempfile.TemporaryDirectory(prefix="research-hf-manifest-") as temp_dir:
        manifest_path = Path(temp_dir) / filename
        write_json(manifest_path, payload)
        run_hf_request(
            lambda: api.upload_file(
                path_or_fileobj=manifest_path,
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
                commit_message=commit_message,
            )
        )


def normalize_repo_prefix(value: str) -> str:
    """Normalize a repo-relative prefix."""

    normalized = PurePosixPath(str(value).replace("\\", "/")).as_posix().strip("/")
    if not normalized:
        raise ValueError("repo prefix must not be empty")
    if ".." in PurePosixPath(normalized).parts:
        raise ValueError(f"repo prefix must stay within repo root: {value}")
    return normalized


def is_within_root(root: Path, candidate: Path) -> bool:
    """Return whether a path resolves within a root."""

    try:
        candidate.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def published_after_cutoff(recent_hours: float | None) -> datetime | None:
    """Resolve the UTC cutoff for a recent-hours filter."""

    if recent_hours is None:
        return None
    if recent_hours <= 0:
        raise ValueError("`recent_hours` 必须大于 0。")
    return utcnow() - timedelta(hours=recent_hours)


def parse_utc_timestamp(value: Any) -> datetime | None:
    """Parse a stored UTC timestamp."""

    if not value:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    return datetime.fromisoformat(str(value)).astimezone(UTC)


def _build_requests_session(*, trust_env: bool) -> requests.Session:
    session = requests.Session()
    session.trust_env = trust_env
    return session


def _configure_hf_http_backend(*, trust_env: bool) -> None:
    global _HF_HTTP_BACKEND_CONFIGURED, _HF_HTTP_TRUST_ENV

    if _HF_HTTP_BACKEND_CONFIGURED and trust_env == _HF_HTTP_TRUST_ENV:
        return
    hf_configure_http_backend(lambda: _build_requests_session(trust_env=trust_env))
    _HF_HTTP_BACKEND_CONFIGURED = True
    _HF_HTTP_TRUST_ENV = trust_env
