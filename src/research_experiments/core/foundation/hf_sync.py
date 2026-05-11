"""统一管理本地工作区与 Hugging Face 之间的批量同步。"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
import json
import shutil
from typing import Any

from huggingface_hub import HfApi, snapshot_download

from research_experiments.core.foundation.cache_snapshots import pull_latest_cache_snapshot, push_latest_cache_snapshot
from research_experiments.core.foundation.run_archives import (
    ARCHIVE_MANIFEST_FILENAME,
    HF_PUBLISH_STATUS_FILENAME,
    extract_run_archives,
    publish_run_to_hub,
)
from research_experiments.core.foundation.workspace import default_cache_hf_repo, default_runs_hf_repo, workspace_layout


def discover_publishable_runs(
    runs_root: str | Path | None = None,
    *,
    include_matrix: bool = True,
) -> tuple[dict[str, Any], ...]:
    """扫描本地工作区中可发布的 run 目录。"""
    root = Path(runs_root) if runs_root is not None else workspace_layout().runs_root
    discovered: dict[str, dict[str, Any]] = {}

    for manifest_path in sorted(root.rglob("manifest.json")):
        run_root = manifest_path.parent
        if not _looks_like_standard_run(run_root):
            continue
        discovered[run_root.as_posix()] = _describe_standard_run(run_root)

    if include_matrix:
        for state_path in sorted(root.rglob("state.json")):
            run_root = state_path.parent
            if not _looks_like_matrix_run(run_root):
                continue
            discovered.setdefault(run_root.as_posix(), _describe_matrix_run(run_root))

    return tuple(discovered[key] for key in sorted(discovered))


def push_workspace_to_hub(
    *,
    runs_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    runs_repo_id: str | None = None,
    cache_repo_id: str | None = None,
    token: str | None = None,
    publish_runs: bool = True,
    push_cache: bool = True,
    include_matrix: bool = True,
    force_runs: bool = False,
    selected_run_dirs: list[str] | None = None,
    cache_shard_filters: list[str] | None = None,
    create_runs_repo: bool = True,
    create_cache_repo: bool = True,
    private_cache_repo: bool = True,
    continue_on_error: bool = True,
) -> dict[str, Any]:
    """批量推送本地 runs 与 cache 到 Hugging Face。"""
    resolved_runs_root = Path(runs_root) if runs_root is not None else workspace_layout().runs_root
    resolved_cache_root = Path(cache_root) if cache_root is not None else workspace_layout().cache_root
    resolved_runs_repo = (runs_repo_id or default_runs_hf_repo() or "").strip() or None
    resolved_cache_repo = (cache_repo_id or default_cache_hf_repo() or "").strip() or None

    if publish_runs and not resolved_runs_repo:
        raise RuntimeError("缺少 runs Hugging Face repo；请传入 `--runs-repo` 或配置 `RESEARCH_RUNS_HF_REPO`。")
    if push_cache and not resolved_cache_repo:
        raise RuntimeError("缺少 cache Hugging Face repo；请传入 `--cache-repo` 或配置 `RESEARCH_CACHE_HF_REPO`。")

    run_candidates = (
        _select_run_candidates(
            resolved_runs_root,
            include_matrix=include_matrix,
            selected_run_dirs=selected_run_dirs,
        )
        if publish_runs
        else ()
    )
    run_results: list[dict[str, Any]] = []
    published_runs: list[dict[str, Any]] = []
    skipped_runs: list[dict[str, Any]] = []
    run_errors: list[dict[str, Any]] = []

    for candidate in run_candidates:
        run_root = Path(candidate["run_root"])
        if not candidate["publishable"]:
            skipped_runs.append(
                {
                    "run_root": run_root.as_posix(),
                    "run_kind": candidate["run_kind"],
                    "reason": candidate["reason"],
                }
            )
            continue
        if not force_runs and _has_matching_publish_status(run_root, resolved_runs_repo):
            skipped_runs.append(
                {
                    "run_root": run_root.as_posix(),
                    "run_kind": candidate["run_kind"],
                    "reason": "already_published",
                }
            )
            continue
        try:
            payload = publish_run_to_hub(
                run_root,
                repo_id=resolved_runs_repo,
                token=token,
                runs_root=resolved_runs_root,
                create_repo=create_runs_repo,
            )
            (run_root / HF_PUBLISH_STATUS_FILENAME).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            published_runs.append(
                {
                    "run_root": run_root.as_posix(),
                    "run_kind": candidate["run_kind"],
                    "remote_prefix": payload.get("remote_prefix"),
                }
            )
            run_results.append(payload)
        except Exception as exc:
            error_row = {
                "run_root": run_root.as_posix(),
                "run_kind": candidate["run_kind"],
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            run_errors.append(error_row)
            if not continue_on_error:
                raise

    cache_payload = None
    if push_cache:
        cache_payload = push_latest_cache_snapshot(
            resolved_cache_root,
            repo_id=resolved_cache_repo,
            token=token,
            create_repo=create_cache_repo,
            private=private_cache_repo,
            shard_filters=cache_shard_filters,
        )

    return {
        "runs_root": resolved_runs_root.as_posix(),
        "cache_root": resolved_cache_root.as_posix(),
        "runs_repo": resolved_runs_repo,
        "cache_repo": resolved_cache_repo,
        "candidate_run_count": len(run_candidates),
        "published_run_count": len(published_runs),
        "skipped_run_count": len(skipped_runs),
        "run_error_count": len(run_errors),
        "published_runs": published_runs,
        "skipped_runs": skipped_runs,
        "run_errors": run_errors,
        "cache_pushed": cache_payload is not None,
        "cache_snapshot": cache_payload,
    }


def pull_workspace_from_hub(
    *,
    runs_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    runs_repo_id: str | None = None,
    cache_repo_id: str | None = None,
    token: str | None = None,
    fetch_runs: bool = True,
    pull_cache: bool = True,
    overwrite_runs: bool = True,
    selected_run_ids: list[str] | None = None,
    selected_run_prefixes: list[str] | None = None,
    cache_shard_filters: list[str] | None = None,
) -> dict[str, Any]:
    """批量回拉 Hugging Face 上的 runs 与 cache。"""
    resolved_runs_root = Path(runs_root) if runs_root is not None else workspace_layout().runs_root
    resolved_cache_root = Path(cache_root) if cache_root is not None else workspace_layout().cache_root
    resolved_runs_repo = (runs_repo_id or default_runs_hf_repo() or "").strip() or None
    resolved_cache_repo = (cache_repo_id or default_cache_hf_repo() or "").strip() or None

    if fetch_runs and not resolved_runs_repo:
        raise RuntimeError("缺少 runs Hugging Face repo；请传入 `--runs-repo` 或配置 `RESEARCH_RUNS_HF_REPO`。")
    if pull_cache and not resolved_cache_repo:
        raise RuntimeError("缺少 cache Hugging Face repo；请传入 `--cache-repo` 或配置 `RESEARCH_CACHE_HF_REPO`。")

    fetched_runs: list[dict[str, Any]] = []
    if fetch_runs:
        api = HfApi(token=token)
        remote_prefixes = _filter_remote_prefixes(
            list_remote_run_prefixes(api, repo_id=resolved_runs_repo),
            selected_run_ids=selected_run_ids,
            selected_run_prefixes=selected_run_prefixes,
        )
        for remote_prefix in remote_prefixes:
            target_run_root = resolved_runs_root / PurePosixPath(remote_prefix)
            if overwrite_runs and target_run_root.exists():
                shutil.rmtree(target_run_root)
            target_run_root.parent.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=resolved_runs_repo,
                repo_type="dataset",
                allow_patterns=[f"{remote_prefix}/**"],
                local_dir=resolved_runs_root,
                token=token,
            )
            extracted_members = extract_run_archives(target_run_root)
            fetched_runs.append(
                {
                    "remote_prefix": remote_prefix,
                    "target_run_root": target_run_root.as_posix(),
                    "extracted_member_count": len(extracted_members),
                }
            )

    cache_payload = None
    if pull_cache:
        cache_payload = pull_latest_cache_snapshot(
            resolved_cache_root,
            repo_id=resolved_cache_repo,
            token=token,
            shard_filters=cache_shard_filters,
        )

    return {
        "runs_root": resolved_runs_root.as_posix(),
        "cache_root": resolved_cache_root.as_posix(),
        "runs_repo": resolved_runs_repo,
        "cache_repo": resolved_cache_repo,
        "fetched_run_count": len(fetched_runs),
        "fetched_runs": fetched_runs,
        "cache_pulled": cache_payload is not None,
        "cache_snapshot": cache_payload,
    }


def collect_hf_sync_status(
    *,
    runs_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    runs_repo_id: str | None = None,
    cache_repo_id: str | None = None,
    token: str | None = None,
    include_remote: bool = True,
    include_matrix: bool = True,
) -> dict[str, Any]:
    """汇总本地与远端的同步状态。"""
    resolved_runs_root = Path(runs_root) if runs_root is not None else workspace_layout().runs_root
    resolved_cache_root = Path(cache_root) if cache_root is not None else workspace_layout().cache_root
    resolved_runs_repo = (runs_repo_id or default_runs_hf_repo() or "").strip() or None
    resolved_cache_repo = (cache_repo_id or default_cache_hf_repo() or "").strip() or None
    candidates = discover_publishable_runs(resolved_runs_root, include_matrix=include_matrix)

    published_locally = 0
    for candidate in candidates:
        if _has_any_publish_status(Path(candidate["run_root"])):
            published_locally += 1

    payload: dict[str, Any] = {
        "runs_root": resolved_runs_root.as_posix(),
        "cache_root": resolved_cache_root.as_posix(),
        "runs_repo": resolved_runs_repo,
        "cache_repo": resolved_cache_repo,
        "local_run_count": len(candidates),
        "local_publishable_run_count": sum(1 for row in candidates if row["publishable"]),
        "local_published_run_count": published_locally,
        "local_cache_sqlite_count": len(tuple(resolved_cache_root.rglob("requests.sqlite"))),
    }

    if include_remote:
        api = HfApi(token=token)
        if resolved_runs_repo:
            remote_run_prefixes = list_remote_run_prefixes(api, repo_id=resolved_runs_repo)
            payload["remote_run_count"] = len(remote_run_prefixes)
            payload["remote_run_prefixes_preview"] = remote_run_prefixes[:10]
        else:
            payload["remote_run_count"] = None
        if resolved_cache_repo:
            cache_files = api.list_repo_files(repo_id=resolved_cache_repo, repo_type="dataset")
            payload["remote_cache_snapshot_present"] = "snapshot_manifest.json" in cache_files
            payload["remote_cache_file_count"] = len(cache_files)
        else:
            payload["remote_cache_snapshot_present"] = None
            payload["remote_cache_file_count"] = None
    return payload


def list_remote_run_prefixes(api: HfApi, *, repo_id: str) -> list[str]:
    """列出远端 runs repo 中所有已发布 run 的相对前缀。"""
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    prefixes = [
        PurePosixPath(path).parent.as_posix()
        for path in repo_files
        if path.endswith(f"/{ARCHIVE_MANIFEST_FILENAME}")
    ]
    return sorted(dict.fromkeys(prefixes))


def _select_run_candidates(
    runs_root: Path,
    *,
    include_matrix: bool,
    selected_run_dirs: list[str] | None,
) -> tuple[dict[str, Any], ...]:
    if not selected_run_dirs:
        return discover_publishable_runs(runs_root, include_matrix=include_matrix)

    discovered: dict[str, dict[str, Any]] = {}
    for item in selected_run_dirs:
        run_root = _resolve_requested_run_root(item, runs_root)
        key = run_root.as_posix()
        if key in discovered:
            continue
        if _looks_like_standard_run(run_root):
            discovered[key] = _describe_standard_run(run_root)
            continue
        if include_matrix and _looks_like_matrix_run(run_root):
            discovered[key] = _describe_matrix_run(run_root)
            continue
        raise RuntimeError(f"指定目录不是可发布的 run：{run_root.as_posix()}")
    return tuple(discovered[key] for key in sorted(discovered))


def _filter_remote_prefixes(
    remote_prefixes: list[str],
    *,
    selected_run_ids: list[str] | None,
    selected_run_prefixes: list[str] | None,
) -> list[str]:
    normalized_prefix_filters = {
        PurePosixPath(item.replace("\\", "/")).as_posix().strip("/")
        for item in (selected_run_prefixes or [])
        if str(item).strip()
    }
    normalized_run_ids = {str(item).strip() for item in (selected_run_ids or []) if str(item).strip()}
    if not normalized_prefix_filters and not normalized_run_ids:
        return remote_prefixes

    filtered = [
        prefix
        for prefix in remote_prefixes
        if prefix in normalized_prefix_filters or PurePosixPath(prefix).name in normalized_run_ids
    ]
    matched_run_ids = {PurePosixPath(prefix).name for prefix in filtered}
    missing_prefixes = normalized_prefix_filters.difference(filtered)
    missing_run_ids = normalized_run_ids.difference(matched_run_ids)
    if missing_prefixes or missing_run_ids:
        messages: list[str] = []
        if missing_prefixes:
            messages.append(f"未找到 run_prefix: {sorted(missing_prefixes)}")
        if missing_run_ids:
            messages.append(f"未找到 run_id: {sorted(missing_run_ids)}")
        raise FileNotFoundError("；".join(messages))
    return filtered


def _resolve_requested_run_root(requested: str, runs_root: Path) -> Path:
    candidate = Path(requested)
    candidates: list[Path] = []
    if candidate.is_absolute():
        candidates.append(candidate)
    else:
        candidates.append(runs_root / candidate)
        candidates.append(candidate)
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"未找到指定 run 目录：{requested}")


def _looks_like_standard_run(run_root: Path) -> bool:
    return (run_root / "manifest.json").exists() and (run_root / "report.md").exists()


def _looks_like_matrix_run(run_root: Path) -> bool:
    return (run_root / "state.json").exists() and (run_root / "paper_package.json").exists()


def _describe_standard_run(run_root: Path) -> dict[str, Any]:
    validation = _load_json(run_root / "run_validation.json")
    publishable = bool(validation.get("passed"))
    reason = "ready" if publishable else "validation_not_passed"
    manifest = _load_json(run_root / "manifest.json")
    return {
        "run_root": run_root.as_posix(),
        "run_kind": "standard",
        "run_id": str(manifest.get("run_id") or run_root.name),
        "publishable": publishable,
        "reason": reason,
    }


def _describe_matrix_run(run_root: Path) -> dict[str, Any]:
    state = _load_json(run_root / "state.json")
    entries = state.get("entries", []) if isinstance(state, dict) else []
    blocking = [
        row
        for row in entries
        if isinstance(row, dict) and str(row.get("status") or "") not in {"completed", "excluded"}
    ]
    counts = state.get("counts", {}) if isinstance(state, dict) else {}
    completed = int(counts.get("completed") or 0) if isinstance(counts, dict) else 0
    expected = int(counts.get("semantic_unique_targets") or 0) if isinstance(counts, dict) else 0
    publishable = not blocking and completed >= expected and expected > 0
    reason = "ready" if publishable else "matrix_not_completed"
    return {
        "run_root": run_root.as_posix(),
        "run_kind": "matrix",
        "run_id": run_root.name,
        "publishable": publishable,
        "reason": reason,
    }


def _has_matching_publish_status(run_root: Path, repo_id: str | None) -> bool:
    if not repo_id:
        return False
    payload = _load_json(run_root / HF_PUBLISH_STATUS_FILENAME)
    return bool(payload.get("published")) and str(payload.get("remote_repo") or "") == repo_id


def _has_any_publish_status(run_root: Path) -> bool:
    payload = _load_json(run_root / HF_PUBLISH_STATUS_FILENAME)
    return bool(payload.get("published"))


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
