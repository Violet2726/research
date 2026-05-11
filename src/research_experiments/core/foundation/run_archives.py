"""run 级归档与 Hugging Face 发布能力。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import json
import shutil
import tempfile
from typing import Any

from huggingface_hub import HfApi, snapshot_download

from research_experiments.core.foundation.archive_common import copy_relative_files, extract_tar_zst, pack_tar_zst
from research_experiments.core.foundation.workspace import auto_publish_runs_enabled, default_runs_hf_repo, workspace_layout


ARCHIVE_MANIFEST_FILENAME = "archive_manifest.json"
HF_PUBLISH_STATUS_FILENAME = "hf_publish.json"
ARCHIVE_SCHEMA_VERSION = 1
KNOWN_ARCHIVE_NAMES = (
    "traces.tar.zst",
    "predictions.tar.zst",
    "artifacts.tar.zst",
)
TRACE_FILE_NAMES = {
    "raw_responses.jsonl",
    "sample_views.jsonl",
    "message_packets.jsonl",
    "debate_messages.jsonl",
    "belief_updates.jsonl",
    "candidate_packets.jsonl",
    "auction_decisions.jsonl",
    "trigger_decisions.jsonl",
    "control_turns.jsonl",
    "trajectory_scores.jsonl",
    "agent_turns.jsonl",
    "stage_a_turns.jsonl",
    "stage_b_turns.jsonl",
    "audit_turns.jsonl",
}
PREDICTION_FILE_NAMES = {
    "predictions.jsonl",
    "final_predictions.jsonl",
    "policy_predictions.jsonl",
}


@dataclass(frozen=True)
class ArchiveGroupPlan:
    """描述某一类重型产物应该如何归档。"""

    group_kind: str
    archive_name: str
    members: tuple[str, ...]


def pack_run_artifacts(
    run_dir: str | Path,
    *,
    runs_root: str | Path | None = None,
) -> dict[str, Any]:
    """为单个 run 生成 `archive_manifest.json` 与 `tar.zst` 归档包。"""
    root = Path(run_dir)
    manifest_path = root / "manifest.json"
    manifest_payload = _load_json(manifest_path)
    remote_prefix = str(manifest_payload.get("remote_prefix") or _infer_remote_prefix(root, runs_root))
    remote_repo = manifest_payload.get("remote_repo")

    relative_files = _list_run_files(root)
    plans = _build_group_plans(relative_files)
    grouped_members = {member for plan in plans for member in plan.members}
    visible_files = sorted(path for path in relative_files if path not in grouped_members)

    packed_rows: list[dict[str, Any]] = []
    used_archive_names = {plan.archive_name for plan in plans if plan.members}
    for archive_name in KNOWN_ARCHIVE_NAMES:
        archive_path = root / archive_name
        if archive_name not in used_archive_names and archive_path.exists():
            archive_path.unlink()

    for plan in plans:
        if not plan.members:
            continue
        packed = pack_tar_zst(
            source_root=root,
            members=list(plan.members),
            archive_path=root / plan.archive_name,
        )
        packed_rows.append(
            {
                "group_kind": plan.group_kind,
                "archive_name": packed.archive_name,
                "archive_path": packed.archive_name,
                "member_count": len(packed.members),
                "members": list(packed.members),
                "original_size_bytes": packed.original_size_bytes,
                "archive_size_bytes": packed.archive_size_bytes,
                "sha256": packed.sha256_hex,
            }
        )

    archive_manifest = {
        "schema_version": ARCHIVE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_id": manifest_payload.get("run_id") or root.name,
        "remote_repo": remote_repo,
        "remote_prefix": remote_prefix,
        "artifacts_packaged": bool(packed_rows),
        "visible_files": visible_files,
        "archives": packed_rows,
    }
    archive_manifest_path = root / ARCHIVE_MANIFEST_FILENAME
    archive_manifest_path.write_text(json.dumps(archive_manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_payload["remote_repo"] = remote_repo
    manifest_payload["remote_prefix"] = remote_prefix
    manifest_payload["archive_manifest_path"] = ARCHIVE_MANIFEST_FILENAME
    manifest_payload["artifacts_packaged"] = bool(packed_rows)
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "run_dir": root.as_posix(),
        "archive_manifest": archive_manifest_path.as_posix(),
        "remote_repo": remote_repo,
        "remote_prefix": remote_prefix,
        "artifacts_packaged": bool(packed_rows),
        "visible_file_count": len(visible_files),
        "archive_count": len(packed_rows),
        "archives": packed_rows,
    }


def publish_run_to_hub(
    run_dir: str | Path,
    *,
    repo_id: str,
    token: str | None = None,
    runs_root: str | Path | None = None,
    create_repo: bool = True,
) -> dict[str, Any]:
    """把单个 run 的可浏览外壳与归档包发布到 HF dataset repo。"""
    root = Path(run_dir)
    summary = pack_run_artifacts(root, runs_root=runs_root)
    manifest_path = root / "manifest.json"
    manifest_payload = _load_json(manifest_path)
    manifest_payload["remote_repo"] = repo_id
    manifest_payload["remote_prefix"] = summary["remote_prefix"]
    manifest_payload["archive_manifest_path"] = ARCHIVE_MANIFEST_FILENAME
    manifest_payload["artifacts_packaged"] = bool(summary["archives"])
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = pack_run_artifacts(root, runs_root=runs_root)

    api = HfApi(token=token)
    if create_repo:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="research-run-publish-") as temp_dir:
        staging_root = Path(temp_dir)
        _stage_run_for_publish(root, staging_root)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=staging_root.as_posix(),
            path_in_repo=str(summary["remote_prefix"]),
            commit_message=_build_run_publish_commit_message(str(summary["remote_prefix"])),
        )

    return {
        **summary,
        "remote_repo": repo_id,
        "published": True,
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
    """按环境约定自动发布单个正式 run；未启用时返回 `None`。"""
    if not auto_publish_runs_enabled():
        return None
    resolved_repo = repo_id or default_runs_hf_repo()
    if not resolved_repo:
        return None
    if validation is not None and not bool(validation.get("passed")):
        return None

    payload = publish_run_to_hub(
        run_dir,
        repo_id=resolved_repo,
        token=token,
        runs_root=runs_root,
        create_repo=create_repo,
    )
    status_path = Path(run_dir) / HF_PUBLISH_STATUS_FILENAME
    status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def fetch_run_from_hub(
    run_id: str | None = None,
    *,
    repo_id: str,
    remote_prefix: str | None = None,
    token: str | None = None,
    target_root: str | Path | None = None,
) -> dict[str, Any]:
    """从 HF dataset repo 拉取单个 run，并直接完成归档解压。"""
    if bool(run_id) == bool(remote_prefix):
        raise ValueError("Please provide exactly one of run_id or remote_prefix.")

    api = HfApi(token=token)
    resolved_remote_prefix = (
        _normalize_remote_prefix(remote_prefix)
        if remote_prefix is not None
        else _discover_remote_prefix(api, repo_id=repo_id, run_id=str(run_id))
    )
    target_base = Path(target_root) if target_root is not None else workspace_layout().runs_root
    target_run_root = target_base / resolved_remote_prefix
    if target_run_root.exists():
        shutil.rmtree(target_run_root)
    target_run_root.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns=[f"{resolved_remote_prefix}/**"],
        local_dir=target_base,
        token=token,
    )

    extracted_members = extract_run_archives(target_run_root)
    return {
        "run_id": run_id or PurePosixPath(resolved_remote_prefix).name,
        "remote_repo": repo_id,
        "remote_prefix": resolved_remote_prefix,
        "target_run_root": target_run_root.as_posix(),
        "extracted_member_count": len(extracted_members),
        "extracted_members": list(extracted_members),
    }


def extract_run_archives(run_dir: str | Path) -> tuple[str, ...]:
    """在本地 run 目录中解压全部归档包。"""
    root = Path(run_dir)
    payload = _load_json(root / ARCHIVE_MANIFEST_FILENAME)
    archives = payload.get("archives", []) if isinstance(payload, dict) else []

    extracted: list[str] = []
    for row in archives:
        if not isinstance(row, dict):
            continue
        archive_path = root / str(row.get("archive_path") or "")
        if not archive_path.exists():
            continue
        extracted.extend(extract_tar_zst(archive_path=archive_path, target_root=root))
    return tuple(extracted)


def validate_archive_contract(run_dir: str | Path, *, verify_sha256: bool = False) -> dict[str, Any]:
    """校验 run 级归档合同是否完整。"""
    root = Path(run_dir)
    manifest_path = root / ARCHIVE_MANIFEST_FILENAME
    payload = _load_json(manifest_path)
    visible_files = payload.get("visible_files", []) if isinstance(payload, dict) else []
    archives = payload.get("archives", []) if isinstance(payload, dict) else []

    missing_visible_files = [
        str(item)
        for item in visible_files
        if isinstance(item, str) and not (root / item).exists()
    ]
    missing_archives = []
    invalid_archive_extensions = []
    hash_mismatches = []
    for row in archives:
        if not isinstance(row, dict):
            continue
        archive_path = str(row.get("archive_path") or "")
        archive_file = root / archive_path
        if not archive_file.exists():
            missing_archives.append(archive_path)
            continue
        if not archive_path.endswith(".tar.zst"):
            invalid_archive_extensions.append(archive_path)
        if verify_sha256:
            expected_hash = str(row.get("sha256") or "")
            if expected_hash and expected_hash != _sha256_of(root / archive_path):
                hash_mismatches.append(archive_path)

    overlapping_paths = [
        path
        for path in visible_files
        if isinstance(path, str) and any(path in row.get("members", []) for row in archives if isinstance(row, dict))
    ]
    return {
        "passed": manifest_path.exists()
        and not missing_visible_files
        and not missing_archives
        and not invalid_archive_extensions
        and not hash_mismatches
        and not overlapping_paths,
        "archive_manifest_present": manifest_path.exists(),
        "archive_count": len([row for row in archives if isinstance(row, dict)]),
        "visible_file_count": len([item for item in visible_files if isinstance(item, str)]),
        "missing_visible_files": missing_visible_files,
        "missing_archives": missing_archives,
        "invalid_archive_extensions": invalid_archive_extensions,
        "hash_mismatches": hash_mismatches,
        "overlapping_paths": overlapping_paths,
    }


def _stage_run_for_publish(run_root: Path, staging_root: Path) -> None:
    payload = _load_json(run_root / ARCHIVE_MANIFEST_FILENAME)
    visible_files = [str(item) for item in payload.get("visible_files", []) if isinstance(item, str)]
    archive_files = [
        str(row.get("archive_path"))
        for row in payload.get("archives", [])
        if isinstance(row, dict) and row.get("archive_path")
    ]
    files_to_copy = visible_files + [ARCHIVE_MANIFEST_FILENAME] + archive_files
    copy_relative_files(source_root=run_root, target_root=staging_root, members=files_to_copy)


def _build_group_plans(relative_files: list[str]) -> list[ArchiveGroupPlan]:
    groups: dict[str, list[str]] = {
        "traces": [],
        "predictions": [],
        "artifacts": [],
    }
    for relative_path in relative_files:
        group_kind = _classify_path(relative_path)
        if group_kind is None:
            continue
        groups[group_kind].append(relative_path)
    return [
        ArchiveGroupPlan("traces", "traces.tar.zst", tuple(sorted(groups["traces"]))),
        ArchiveGroupPlan("predictions", "predictions.tar.zst", tuple(sorted(groups["predictions"]))),
        ArchiveGroupPlan("artifacts", "artifacts.tar.zst", tuple(sorted(groups["artifacts"]))),
    ]


def _classify_path(relative_path: str) -> str | None:
    path = PurePosixPath(relative_path)
    parts = path.parts
    basename = path.name

    if basename in {ARCHIVE_MANIFEST_FILENAME, *KNOWN_ARCHIVE_NAMES}:
        return None
    if parts and parts[0] == "figures":
        return None
    if any(part.endswith("_predictions") or part == "hotpot_predictions" for part in parts[:-1]):
        return "predictions"
    if basename in PREDICTION_FILE_NAMES:
        return "predictions"
    if basename in TRACE_FILE_NAMES or basename.endswith("_turns.jsonl"):
        return "traces"
    if basename.endswith(".jsonl"):
        return "artifacts"
    return None


def _discover_remote_prefix(api: HfApi, *, repo_id: str, run_id: str) -> str:
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    candidates = [
        PurePosixPath(path).parent.as_posix()
        for path in repo_files
        if path.endswith(f"/{ARCHIVE_MANIFEST_FILENAME}") and f"/{run_id}/" in path
    ]
    unique = sorted(dict.fromkeys(candidates))
    if not unique:
        raise FileNotFoundError(f"Remote repo {repo_id} does not contain run_id {run_id}.")
    if len(unique) > 1:
        raise RuntimeError(f"Remote repo {repo_id} contains multiple prefixes for run_id {run_id}: {unique}")
    return unique[0]


def _normalize_remote_prefix(value: str) -> str:
    normalized = PurePosixPath(value.replace("\\", "/")).as_posix().strip("/")
    if not normalized:
        raise ValueError("remote_prefix must not be empty.")
    return normalized


def _list_run_files(root: Path) -> list[str]:
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in {ARCHIVE_MANIFEST_FILENAME, *KNOWN_ARCHIVE_NAMES}
    )


def _infer_remote_prefix(run_root: Path, runs_root: str | Path | None) -> str:
    if runs_root is not None:
        candidate_roots = [Path(runs_root)]
    else:
        layout_root = workspace_layout().runs_root
        candidate_roots = [layout_root]
        parts = list(run_root.resolve().parts)
        if "runs" in parts:
            runs_index = parts.index("runs")
            candidate_roots.append(Path(*parts[: runs_index + 1]))

    for candidate_root in candidate_roots:
        try:
            return run_root.resolve().relative_to(candidate_root.resolve()).as_posix()
        except ValueError:
            continue
    return run_root.name


def _build_run_publish_commit_message(remote_prefix: str) -> str:
    normalized_parts = [part for part in PurePosixPath(remote_prefix).parts if part]
    if len(normalized_parts) >= 4:
        family, experiment, phase, run_id = normalized_parts[:4]
        return f"发布 run [{family}] {experiment} | {phase} | {run_id}"
    if len(normalized_parts) == 2:
        family, run_id = normalized_parts
        return f"发布 run [{family}] {run_id}"
    if normalized_parts:
        return f"发布 run {' / '.join(normalized_parts)}"
    return "发布 run"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_of(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
