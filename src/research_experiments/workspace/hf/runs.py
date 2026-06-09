"""Hugging Face runs 同步服务。"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, snapshot_download

from research_experiments.core.io import write_json
from research_experiments.workspace.archive_utils import copy_relative_files, sha256_file
from research_experiments.workspace.hf.common import (
    HF_RUN_STATE_FILENAME,
    HF_SYNC_SCHEMA_VERSION,
    IGNORED_PUBLISH_STATUS_FILENAME,
    RUNS_MANIFEST_FILENAME,
    download_repo_manifest,
    is_within_root,
    iso_utc_now,
    load_json_if_exists,
    normalize_repo_prefix,
    parse_utc_timestamp,
    published_after_cutoff,
    resolve_runs_repo_id,
    run_hf_request,
    upload_manifest,
)
from research_experiments.workspace.layout import auto_publish_runs_enabled, workspace_layout
from research_experiments.workspace.run_archives import (
    ARCHIVE_MANIFEST_FILENAME,
    KNOWN_ARCHIVE_NAMES,
    extract_run_archives,
    pack_run_artifacts,
)
from research_experiments.workspace.run_archives import (
    validate_archive_contract as _validate_archive_contract,
)

validate_archive_contract = _validate_archive_contract


def push_runs_to_hub(
    runs_root: str | Path | None = None,
    *,
    sources: list[str] | None = None,
    repo_id: str | None = None,
    token: str | None = None,
    skip_validation: bool = False,
    create_repo: bool = True,
) -> dict[str, Any]:
    """Push selected local runs to Hugging Face."""

    resolved_runs_root = Path(runs_root) if runs_root is not None else workspace_layout().runs_root
    resolved_repo_id = resolve_runs_repo_id(repo_id)
    api = HfApi(token=token)
    remote_manifest = download_repo_manifest(
        repo_id=resolved_repo_id,
        filename=RUNS_MANIFEST_FILENAME,
        token=token,
        missing_ok=True,
    )
    remote_rows = _index_run_records(remote_manifest)
    manifest_rows = dict(remote_rows)
    candidates = discover_publishable_runs(resolved_runs_root, sources=sources)

    published_runs: list[dict[str, Any]] = []
    skipped_runs: list[dict[str, Any]] = []
    now_iso = iso_utc_now()
    repo_created = False

    for candidate in candidates:
        run_root = Path(candidate["run_root"])
        remote_prefix = _infer_remote_prefix(run_root, resolved_runs_root)
        if not candidate["publishable"] and not skip_validation:
            skipped_runs.append(
                {
                    "run_root": run_root.as_posix(),
                    "remote_prefix": remote_prefix,
                    "run_kind": candidate["run_kind"],
                    "reason": candidate["reason"],
                }
            )
            continue

        pack_run_artifacts(run_root, runs_root=resolved_runs_root)
        bundle_sha256 = compute_run_bundle_sha256(run_root)
        remote_row = remote_rows.get(remote_prefix)
        if remote_row and str(remote_row.get("bundle_sha256") or "") == bundle_sha256:
            _write_run_state(
                run_root,
                remote_prefix=remote_prefix,
                bundle_sha256=bundle_sha256,
                last_published_at=str(remote_row.get("published_at") or ""),
            )
            skipped_runs.append(
                {
                    "run_root": run_root.as_posix(),
                    "remote_prefix": remote_prefix,
                    "run_kind": candidate["run_kind"],
                    "reason": "already_published",
                }
            )
            continue

        if create_repo and not repo_created:
            run_hf_request(
                lambda: api.create_repo(repo_id=resolved_repo_id, repo_type="dataset", private=False, exist_ok=True)
            )
            repo_created = True
        published_at = now_iso
        record = {
            "remote_prefix": remote_prefix,
            "run_kind": candidate["run_kind"],
            "run_id": candidate["run_id"],
            "bundle_sha256": bundle_sha256,
            "published_at": published_at,
        }
        next_manifest_rows = dict(manifest_rows)
        next_manifest_rows[remote_prefix] = record
        next_manifest_payload = {
            "schema_version": HF_SYNC_SCHEMA_VERSION,
            "generated_at": now_iso,
            "runs": sorted(next_manifest_rows.values(), key=lambda row: str(row["remote_prefix"])),
        }
        _commit_run_bundle(
            api,
            repo_id=resolved_repo_id,
            token=token,
            run_root=run_root,
            remote_prefix=remote_prefix,
            manifest_payload=next_manifest_payload,
            replace_existing=remote_row is not None,
        )

        manifest_rows = next_manifest_rows
        remote_rows[remote_prefix] = record
        _write_run_state(
            run_root,
            remote_prefix=remote_prefix,
            bundle_sha256=bundle_sha256,
            last_published_at=published_at,
        )
        published_runs.append(
            {
                "run_root": run_root.as_posix(),
                "remote_prefix": remote_prefix,
                "run_kind": candidate["run_kind"],
                "bundle_sha256": bundle_sha256,
                "published_at": published_at,
            }
        )

    manifest_payload = {
        "schema_version": HF_SYNC_SCHEMA_VERSION,
        "generated_at": now_iso,
        "runs": sorted(manifest_rows.values(), key=lambda row: str(row["remote_prefix"])),
    }
    manifest_changed = _runs_manifest_signature(remote_manifest) != _runs_manifest_signature(manifest_payload)
    if not published_runs and manifest_changed:
        if create_repo and not repo_created:
            run_hf_request(
                lambda: api.create_repo(repo_id=resolved_repo_id, repo_type="dataset", private=False, exist_ok=True)
            )
        upload_manifest(
            api,
            repo_id=resolved_repo_id,
            filename=RUNS_MANIFEST_FILENAME,
            payload=manifest_payload,
            token=token,
            commit_message="Update runs manifest",
        )
        published = True
    else:
        published = bool(published_runs)

    return {
        "runs_root": resolved_runs_root.as_posix(),
        "remote_repo": resolved_repo_id,
        "published": published,
        "candidate_run_count": len(candidates),
        "published_run_count": len(published_runs),
        "published_runs": published_runs,
        "skipped_run_count": len(skipped_runs),
        "skipped_runs": skipped_runs,
        "skip_validation": skip_validation,
    }


def pull_runs_from_hub(
    runs_root: str | Path | None = None,
    *,
    prefixes: list[str] | None = None,
    recent_hours: float | None = None,
    repo_id: str | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Pull selected remote runs from Hugging Face."""

    resolved_runs_root = Path(runs_root) if runs_root is not None else workspace_layout().runs_root
    resolved_repo_id = resolve_runs_repo_id(repo_id)
    manifest = download_repo_manifest(
        repo_id=resolved_repo_id,
        filename=RUNS_MANIFEST_FILENAME,
        token=token,
        missing_ok=False,
    )
    selected_rows = _select_remote_run_records(
        manifest,
        prefixes=prefixes,
        recent_hours=recent_hours,
    )
    fetched_runs: list[dict[str, Any]] = []
    skipped_runs: list[dict[str, Any]] = []
    conflict_runs: list[dict[str, Any]] = []

    for row in selected_rows:
        remote_prefix = str(row["remote_prefix"])
        target_run_root = resolved_runs_root / PurePosixPath(remote_prefix)
        expected_bundle_sha256 = str(row.get("bundle_sha256") or "")
        if target_run_root.exists():
            local_bundle_sha256 = _resolve_existing_local_bundle_sha256(target_run_root)
            if local_bundle_sha256 == expected_bundle_sha256:
                _write_run_state(
                    target_run_root,
                    remote_prefix=remote_prefix,
                    bundle_sha256=expected_bundle_sha256,
                    last_published_at=str(row.get("published_at") or ""),
                )
                skipped_runs.append(
                    {
                        "remote_prefix": remote_prefix,
                        "target_run_root": target_run_root.as_posix(),
                        "reason": "already_present",
                    }
                )
                continue
            conflict_runs.append(
                {
                    "remote_prefix": remote_prefix,
                    "target_run_root": target_run_root.as_posix(),
                    "reason": "bundle_conflict",
                    "local_bundle_sha256": local_bundle_sha256,
                    "remote_bundle_sha256": expected_bundle_sha256,
                }
            )
            continue

        target_run_root.parent.mkdir(parents=True, exist_ok=True)
        run_hf_request(
            lambda remote_prefix=remote_prefix: snapshot_download(
                repo_id=resolved_repo_id,
                repo_type="dataset",
                allow_patterns=[f"{remote_prefix}/**"],
                local_dir=resolved_runs_root,
                token=token,
            )
        )
        extract_run_archives(target_run_root)
        local_bundle_sha256 = compute_run_bundle_sha256(target_run_root)
        if local_bundle_sha256 != expected_bundle_sha256:
            conflict_runs.append(
                {
                    "remote_prefix": remote_prefix,
                    "target_run_root": target_run_root.as_posix(),
                    "reason": "bundle_mismatch_after_pull",
                    "local_bundle_sha256": local_bundle_sha256,
                    "remote_bundle_sha256": expected_bundle_sha256,
                }
            )
            continue
        _write_run_state(
            target_run_root,
            remote_prefix=remote_prefix,
            bundle_sha256=expected_bundle_sha256,
            last_published_at=str(row.get("published_at") or ""),
        )
        fetched_runs.append(
            {
                "remote_prefix": remote_prefix,
                "target_run_root": target_run_root.as_posix(),
                "published_at": str(row.get("published_at") or ""),
            }
        )

    return {
        "runs_root": resolved_runs_root.as_posix(),
        "remote_repo": resolved_repo_id,
        "recent_hours": recent_hours,
        "selected_prefixes": [normalize_repo_prefix(item) for item in (prefixes or []) if str(item).strip()],
        "fetched_run_count": len(fetched_runs),
        "fetched_runs": fetched_runs,
        "skipped_run_count": len(skipped_runs),
        "skipped_runs": skipped_runs,
        "conflict_run_count": len(conflict_runs),
        "conflict_runs": conflict_runs,
        "passed": not conflict_runs,
    }


def publish_run_if_configured(
    run_dir: str | Path,
    *,
    repo_id: str | None = None,
    token: str | None = None,
    runs_root: str | Path | None = None,
    create_repo: bool = True,
    validation: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Auto-publish a completed run when configured."""

    if not auto_publish_runs_enabled():
        return None
    if validation is not None and not bool(validation.get("passed")):
        return None
    payload = push_runs_to_hub(
        runs_root=runs_root,
        sources=[str(run_dir)],
        repo_id=repo_id,
        token=token,
        skip_validation=False,
        create_repo=create_repo,
    )
    run_root = Path(run_dir)
    remote_prefix = _infer_remote_prefix(run_root, Path(runs_root) if runs_root is not None else workspace_layout().runs_root)
    for row in payload["published_runs"]:
        if row["remote_prefix"] == remote_prefix:
            return {
                "published": True,
                "remote_repo": payload["remote_repo"],
                **row,
            }
    for row in payload["skipped_runs"]:
        if row["remote_prefix"] == remote_prefix:
            return {
                "published": False,
                "remote_repo": payload["remote_repo"],
                **row,
            }
    return None


def discover_publishable_runs(
    runs_root: str | Path | None = None,
    *,
    sources: list[str] | None = None,
) -> tuple[dict[str, Any], ...]:
    """Discover complete standard and matrix runs beneath runs_root or sources."""

    resolved_runs_root = Path(runs_root) if runs_root is not None else workspace_layout().runs_root
    roots: dict[str, dict[str, Any]] = {}
    if sources:
        for source in sources:
            resolved_source = _resolve_source_path(resolved_runs_root, source)
            if _looks_like_standard_run(resolved_source, resolved_runs_root):
                roots[resolved_source.as_posix()] = _describe_standard_run(resolved_source)
                continue
            if _looks_like_matrix_run(resolved_source, resolved_runs_root):
                roots[resolved_source.as_posix()] = _describe_matrix_run(resolved_source)
                continue
            for run_root in _discover_runs_beneath(resolved_source, resolved_runs_root):
                roots.setdefault(run_root.as_posix(), _describe_run_root(run_root))
    else:
        for run_root in _discover_runs_beneath(resolved_runs_root, resolved_runs_root):
            roots.setdefault(run_root.as_posix(), _describe_run_root(run_root))
    return tuple(roots[key] for key in sorted(roots))


def compute_run_bundle_sha256(run_dir: str | Path) -> str:
    """Compute the logical bundle hash for a local run."""

    root = Path(run_dir)
    logical_files = [
        path.relative_to(root).as_posix()
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.name not in {ARCHIVE_MANIFEST_FILENAME, *KNOWN_ARCHIVE_NAMES, HF_RUN_STATE_FILENAME, IGNORED_PUBLISH_STATUS_FILENAME}
    ]
    digest = hashlib.sha256()
    for relative_path in logical_files:
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_logical_file_content_digest(root, relative_path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _discover_runs_beneath(root: Path, runs_root: Path) -> tuple[Path, ...]:
    discovered: dict[str, Path] = {}
    for manifest_path in sorted(root.rglob("manifest.json")):
        run_root = manifest_path.parent
        if _looks_like_standard_run(run_root, runs_root):
            discovered[run_root.as_posix()] = run_root
    for state_path in sorted(root.rglob("state.json")):
        run_root = state_path.parent
        if _looks_like_matrix_run(run_root, runs_root):
            discovered.setdefault(run_root.as_posix(), run_root)
    return tuple(discovered[key] for key in sorted(discovered))


def _describe_run_root(run_root: Path) -> dict[str, Any]:
    if (run_root / "state.json").exists():
        return _describe_matrix_run(run_root)
    if (run_root / "manifest.json").exists():
        return _describe_standard_run(run_root)
    raise RuntimeError(f"Unrecognized run root: {run_root}")


def _describe_standard_run(run_root: Path) -> dict[str, Any]:
    validation = load_json_if_exists(run_root / "run_validation.json")
    manifest = load_json_if_exists(run_root / "manifest.json")
    publishable = bool(validation.get("passed"))
    return {
        "run_root": run_root.as_posix(),
        "run_kind": "standard",
        "run_id": str(manifest.get("run_id") or run_root.name),
        "publishable": publishable,
        "reason": "ready" if publishable else "validation_not_passed",
    }


def _describe_matrix_run(run_root: Path) -> dict[str, Any]:
    state = load_json_if_exists(run_root / "state.json")
    counts = state.get("counts", {}) if isinstance(state, dict) else {}
    entries = state.get("entries", []) if isinstance(state, dict) else []
    completed = int(counts.get("completed") or 0) if isinstance(counts, dict) else 0
    expected = int(counts.get("semantic_unique_targets") or 0) if isinstance(counts, dict) else 0
    blocking = [
        row
        for row in entries
        if isinstance(row, dict) and str(row.get("status") or "") not in {"completed", "excluded"}
    ]
    publishable = expected > 0 and completed >= expected and not blocking
    return {
        "run_root": run_root.as_posix(),
        "run_kind": "matrix",
        "run_id": run_root.name,
        "publishable": publishable,
        "reason": "ready" if publishable else "matrix_not_completed",
    }


def _looks_like_standard_run(run_root: Path, runs_root: Path) -> bool:
    if _is_ignored_run_path(run_root, runs_root) or _looks_like_matrix_run(run_root, runs_root):
        return False
    relative_parts = _relative_run_parts(run_root, runs_root)
    if len(relative_parts) < 4:
        return False
    return any(
        (run_root / filename).exists()
        for filename in ("manifest.json", "run_validation.json", ARCHIVE_MANIFEST_FILENAME, "report.md")
    )


def _looks_like_matrix_run(run_root: Path, runs_root: Path) -> bool:
    if _is_ignored_run_path(run_root, runs_root):
        return False
    relative_parts = _relative_run_parts(run_root, runs_root)
    if len(relative_parts) != 2:
        return False
    if not relative_parts[0].endswith("_matrix"):
        return False
    return any(
        (run_root / filename).exists()
        for filename in ("state.json", "manifest.json", ARCHIVE_MANIFEST_FILENAME, "paper_package.json", "reproduction_package.json")
    )


def _resolve_source_path(runs_root: Path, source: str) -> Path:
    candidate = Path(source)
    if not candidate.is_absolute():
        candidate = (Path.cwd() / candidate).resolve() if not (runs_root / candidate).exists() else (runs_root / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"未找到指定 source：{source}")
    if not is_within_root(runs_root, candidate):
        raise RuntimeError(f"source 必须位于 runs 根目录内：{source}")
    return candidate


def _relative_run_parts(run_root: Path, runs_root: Path) -> tuple[str, ...]:
    try:
        return run_root.resolve().relative_to(runs_root.resolve()).parts
    except ValueError:
        return ()


def _is_ignored_run_path(run_root: Path, runs_root: Path) -> bool:
    relative_parts = _relative_run_parts(run_root, runs_root)
    return bool(relative_parts) and relative_parts[0] == ".cache"


def _infer_remote_prefix(run_root: Path, runs_root: Path) -> str:
    return run_root.resolve().relative_to(runs_root.resolve()).as_posix()


def _stage_run_for_sync(run_root: Path, stage_root: Path) -> None:
    archive_manifest = load_json_if_exists(run_root / ARCHIVE_MANIFEST_FILENAME)
    visible_files = [
        str(item)
        for item in archive_manifest.get("visible_files", [])
        if isinstance(item, str) and Path(item).name not in {HF_RUN_STATE_FILENAME, IGNORED_PUBLISH_STATUS_FILENAME}
    ]
    archive_files = [
        str(row.get("archive_path"))
        for row in archive_manifest.get("archives", [])
        if isinstance(row, dict) and row.get("archive_path")
    ]
    members = visible_files + [ARCHIVE_MANIFEST_FILENAME] + archive_files
    copy_relative_files(source_root=run_root, target_root=stage_root, members=members)


def _iter_staged_run_files(stage_root: Path) -> tuple[Path, ...]:
    return tuple(sorted(path for path in stage_root.rglob("*") if path.is_file()))


def _logical_file_content_digest(run_root: Path, relative_path: str) -> str:
    target = run_root / relative_path
    if target.name == "manifest.json":
        payload = load_json_if_exists(target)
        payload.pop("remote_repo", None)
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    if target.name == "run_validation.json":
        payload = load_json_if_exists(target)
        payload.pop("hf_publish", None)
        return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return sha256_file(target)


def _write_run_state(
    run_root: Path,
    *,
    remote_prefix: str,
    bundle_sha256: str,
    last_published_at: str,
) -> None:
    payload = {
        "schema_version": HF_SYNC_SCHEMA_VERSION,
        "remote_prefix": remote_prefix,
        "bundle_sha256": bundle_sha256,
        "last_published_at": last_published_at,
        "last_checked_at": iso_utc_now(),
    }
    write_json(run_root / HF_RUN_STATE_FILENAME, payload)


def _resolve_existing_local_bundle_sha256(run_root: Path) -> str:
    payload = load_json_if_exists(run_root / HF_RUN_STATE_FILENAME)
    stored_prefix = str(payload.get("remote_prefix") or "")
    stored_hash = str(payload.get("bundle_sha256") or "")
    if stored_prefix and stored_hash:
        return stored_hash
    pack_run_artifacts(run_root)
    return compute_run_bundle_sha256(run_root)


def _index_run_records(manifest: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in (manifest or {}).get("runs", []):
        if not isinstance(row, dict):
            continue
        remote_prefix = str(row.get("remote_prefix") or "").strip()
        if remote_prefix:
            rows[remote_prefix] = row
    return rows


def _select_remote_run_records(
    manifest: dict[str, Any],
    *,
    prefixes: list[str] | None,
    recent_hours: float | None,
) -> list[dict[str, Any]]:
    rows = [row for row in manifest.get("runs", []) if isinstance(row, dict) and str(row.get("remote_prefix") or "").strip()]
    normalized_prefixes = [normalize_repo_prefix(item) for item in (prefixes or []) if str(item).strip()]
    if normalized_prefixes:
        rows = [
            row
            for row in rows
            if any(
                str(row["remote_prefix"]) == prefix or str(row["remote_prefix"]).startswith(f"{prefix}/")
                for prefix in normalized_prefixes
            )
        ]
    cutoff = published_after_cutoff(recent_hours)
    if cutoff is not None:
        rows = [
            row
            for row in rows
            if (published_at := parse_utc_timestamp(row.get("published_at"))) is not None and published_at >= cutoff
        ]
    return sorted(rows, key=lambda row: str(row["remote_prefix"]))


def _runs_manifest_signature(manifest: dict[str, Any] | None) -> tuple[tuple[str, str, str, str, str], ...]:
    rows: list[tuple[str, str, str, str, str]] = []
    for row in (manifest or {}).get("runs", []):
        if not isinstance(row, dict):
            continue
        rows.append(
            (
                str(row.get("remote_prefix") or ""),
                str(row.get("run_kind") or ""),
                str(row.get("run_id") or ""),
                str(row.get("bundle_sha256") or ""),
                str(row.get("published_at") or ""),
            )
        )
    return tuple(sorted(rows))


def _commit_run_bundle(
    api: HfApi,
    *,
    repo_id: str,
    token: str | None,
    run_root: Path,
    remote_prefix: str,
    manifest_payload: dict[str, Any],
    replace_existing: bool,
) -> None:
    with tempfile.TemporaryDirectory(prefix="research-hf-run-push-") as temp_dir:
        temp_root = Path(temp_dir)
        stage_root = temp_root / "run"
        stage_root.mkdir(parents=True, exist_ok=True)
        _stage_run_for_sync(run_root, stage_root)

        manifest_path = temp_root / RUNS_MANIFEST_FILENAME
        write_json(manifest_path, manifest_payload)

        operations: list[Any] = []
        if replace_existing:
            operations.append(CommitOperationDelete(path_in_repo=remote_prefix, is_folder=True))
        for staged_file in _iter_staged_run_files(stage_root):
            relative_path = staged_file.relative_to(stage_root).as_posix()
            operations.append(
                CommitOperationAdd(
                    path_in_repo=f"{remote_prefix}/{relative_path}",
                    path_or_fileobj=staged_file,
                )
            )
        operations.append(
            CommitOperationAdd(
                path_in_repo=RUNS_MANIFEST_FILENAME,
                path_or_fileobj=manifest_path,
            )
        )
        run_hf_request(
            lambda operations=tuple(operations), remote_prefix=remote_prefix: api.create_commit(
                repo_id=repo_id,
                repo_type="dataset",
                operations=operations,
                token=token,
                commit_message=f"Publish run {remote_prefix}",
            )
        )
