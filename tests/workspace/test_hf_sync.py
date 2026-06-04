from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete
from testsupport.filesystem import write_json

from research_experiments.workspace.hf.runs import (
    _stage_run_for_sync,
    compute_run_bundle_sha256,
    discover_publishable_runs,
    pull_runs_from_hub,
    push_runs_to_hub,
)
from research_experiments.workspace.run_archives import pack_run_artifacts


def test_discover_publishable_runs_recurses_from_parent_sources(tmp_path: Path) -> None:
    standard_root = tmp_path / "runs" / "single_agent" / "demo" / "count20" / "20260510T000000Z-model"
    _seed_standard_run(standard_root, passed=True)

    invalid_root = tmp_path / "runs" / "single_agent" / "other" / "count20" / "20260510T000001Z-model"
    _seed_standard_run(invalid_root, passed=False)

    matrix_root = tmp_path / "runs" / "faithful_matrix" / "20260510T000100Z-count20-model"
    _seed_matrix_run(matrix_root, completed=2, expected=2, status="completed")

    rows = discover_publishable_runs(tmp_path / "runs", sources=["single_agent"])
    by_root = {row["run_root"]: row for row in rows}

    assert standard_root.as_posix() in by_root
    assert invalid_root.as_posix() in by_root
    assert matrix_root.as_posix() not in by_root
    assert by_root[invalid_root.as_posix()]["reason"] == "validation_not_passed"


def test_discover_publishable_runs_includes_old_incomplete_run_shapes_and_ignores_hf_cache(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"

    standard_root = runs_root / "single_agent" / "legacy_demo" / "count20" / "20260510T000000Z-model"
    write_json(standard_root / "manifest.json", {"run_id": standard_root.name})

    matrix_root = runs_root / "faithful_matrix" / "20260510T000100Z-count20-model"
    write_json(
        matrix_root / "state.json",
        {
            "counts": {"completed": 0, "semantic_unique_targets": 1},
            "entries": [{"status": "running"}],
        },
    )

    cached_root = runs_root / ".cache" / "huggingface" / "download" / "single_agent" / "cached" / "count20" / "20260510T000200Z-model"
    write_json(cached_root / "manifest.json", {"run_id": cached_root.name})
    (cached_root / "report.md").write_text("# cached\n", encoding="utf-8")

    rows = discover_publishable_runs(runs_root)
    by_root = {row["run_root"]: row for row in rows}

    assert standard_root.as_posix() in by_root
    assert by_root[standard_root.as_posix()]["reason"] == "validation_not_passed"
    assert matrix_root.as_posix() in by_root
    assert by_root[matrix_root.as_posix()]["reason"] == "matrix_not_completed"
    assert cached_root.as_posix() not in by_root


def test_push_runs_to_hub_filters_validation_and_skips_matching_remote_bundle(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    publish_root = runs_root / "single_agent" / "demo" / "count20" / "20260510T000000Z-model"
    _seed_standard_run(publish_root, passed=True)
    pack_run_artifacts(publish_root, runs_root=runs_root)

    already_root = runs_root / "dmad" / "demo" / "count20" / "20260510T000001Z-model"
    _seed_standard_run(already_root, passed=True)
    pack_run_artifacts(already_root, runs_root=runs_root)
    already_hash = compute_run_bundle_sha256(already_root)

    invalid_root = runs_root / "budget_comm" / "demo" / "count20" / "20260510T000002Z-model"
    _seed_standard_run(invalid_root, passed=False)

    commit_calls: list[dict[str, object]] = []

    class FakeApi:
        def __init__(self, token=None) -> None:
            self.token = token

        def create_repo(self, **kwargs) -> None:
            return None

        def create_commit(self, **kwargs) -> None:
            commit_calls.append(kwargs)

    monkeypatch.setattr("research_experiments.workspace.hf.runs.download_repo_manifest", lambda **kwargs: {
        "runs": [
            {
                "remote_prefix": "dmad/demo/count20/20260510T000001Z-model",
                "run_kind": "standard",
                "run_id": "20260510T000001Z-model",
                "bundle_sha256": already_hash,
                "published_at": "2026-06-03T10:00:00+00:00",
            }
        ]
    })
    monkeypatch.setattr("research_experiments.workspace.hf.runs.HfApi", FakeApi)

    payload = push_runs_to_hub(runs_root, repo_id="owner/research-runs")

    assert len(commit_calls) == 1
    operations = commit_calls[0]["operations"]
    add_paths = [op.path_in_repo for op in operations if isinstance(op, CommitOperationAdd)]
    delete_paths = [op.path_in_repo for op in operations if isinstance(op, CommitOperationDelete)]
    assert delete_paths == []
    assert "runs_manifest.json" in add_paths
    run_paths = [path for path in add_paths if path != "runs_manifest.json"]
    assert run_paths
    assert all(path.startswith("single_agent/demo/count20/20260510T000000Z-model/") for path in run_paths)
    assert payload["published_run_count"] == 1
    assert payload["skipped_runs"] == [
        {
            "run_root": invalid_root.as_posix(),
            "remote_prefix": "budget_comm/demo/count20/20260510T000002Z-model",
            "run_kind": "standard",
            "reason": "validation_not_passed",
        },
        {
            "run_root": already_root.as_posix(),
            "remote_prefix": "dmad/demo/count20/20260510T000001Z-model",
            "run_kind": "standard",
            "reason": "already_published",
        },
    ]


def test_push_runs_to_hub_can_skip_validation_for_invalid_and_incomplete_matrix(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    invalid_root = runs_root / "budget_comm" / "demo" / "count20" / "20260510T000002Z-model"
    _seed_standard_run(invalid_root, passed=False)

    matrix_root = runs_root / "faithful_matrix" / "20260510T000100Z-count20-model"
    _seed_matrix_run(matrix_root, completed=0, expected=1, status="running")

    commit_calls: list[dict[str, object]] = []

    class FakeApi:
        def __init__(self, token=None) -> None:
            self.token = token

        def create_repo(self, **kwargs) -> None:
            return None

        def create_commit(self, **kwargs) -> None:
            commit_calls.append(kwargs)

    monkeypatch.setattr("research_experiments.workspace.hf.runs.download_repo_manifest", lambda **kwargs: {})
    monkeypatch.setattr("research_experiments.workspace.hf.runs.HfApi", FakeApi)

    payload = push_runs_to_hub(
        runs_root,
        repo_id="owner/research-runs",
        skip_validation=True,
    )

    assert payload["skip_validation"] is True
    assert len(commit_calls) == 2
    add_path_groups = [
        [op.path_in_repo for op in call["operations"] if isinstance(op, CommitOperationAdd)]
        for call in commit_calls
    ]
    assert any(
        any(path.startswith("budget_comm/demo/count20/20260510T000002Z-model/") for path in paths)
        for paths in add_path_groups
    )
    assert any(
        any(path.startswith("faithful_matrix/20260510T000100Z-count20-model/") for path in paths)
        for paths in add_path_groups
    )
    assert all("runs_manifest.json" in paths for paths in add_path_groups)
    assert payload["published_run_count"] == 2


def test_push_runs_to_hub_can_skip_validation_for_legacy_run_shapes(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    standard_root = runs_root / "single_agent" / "legacy_demo" / "count20" / "20260510T000000Z-model"
    write_json(standard_root / "manifest.json", {"run_id": standard_root.name})
    (standard_root / "metrics.json").write_text("{}", encoding="utf-8")

    matrix_root = runs_root / "faithful_matrix" / "20260510T000100Z-count20-model"
    write_json(
        matrix_root / "state.json",
        {
            "counts": {"completed": 0, "semantic_unique_targets": 1},
            "entries": [{"status": "running"}],
        },
    )

    commit_calls: list[dict[str, object]] = []

    class FakeApi:
        def __init__(self, token=None) -> None:
            self.token = token

        def create_repo(self, **kwargs) -> None:
            return None

        def create_commit(self, **kwargs) -> None:
            commit_calls.append(kwargs)

    monkeypatch.setattr("research_experiments.workspace.hf.runs.download_repo_manifest", lambda **kwargs: {})
    monkeypatch.setattr("research_experiments.workspace.hf.runs.HfApi", FakeApi)

    payload = push_runs_to_hub(
        runs_root,
        repo_id="owner/research-runs",
        skip_validation=True,
    )

    assert payload["published_run_count"] == 2
    add_path_groups = [
        [op.path_in_repo for op in call["operations"] if isinstance(op, CommitOperationAdd)]
        for call in commit_calls
    ]
    assert any(
        any(path.startswith("single_agent/legacy_demo/count20/20260510T000000Z-model/") for path in paths)
        for paths in add_path_groups
    )
    assert any(
        any(path.startswith("faithful_matrix/20260510T000100Z-count20-model/") for path in paths)
        for paths in add_path_groups
    )


def test_push_runs_to_hub_replaces_existing_run_in_single_commit(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    publish_root = runs_root / "single_agent" / "demo" / "count20" / "20260510T000000Z-model"
    _seed_standard_run(publish_root, passed=True)

    commit_calls: list[dict[str, object]] = []

    class FakeApi:
        def __init__(self, token=None) -> None:
            self.token = token

        def create_repo(self, **kwargs) -> None:
            return None

        def create_commit(self, **kwargs) -> None:
            commit_calls.append(kwargs)

    monkeypatch.setattr("research_experiments.workspace.hf.runs.download_repo_manifest", lambda **kwargs: {
        "runs": [
            {
                "remote_prefix": "single_agent/demo/count20/20260510T000000Z-model",
                "run_kind": "standard",
                "run_id": "20260510T000000Z-model",
                "bundle_sha256": "old-hash",
                "published_at": "2026-06-03T10:00:00+00:00",
            }
        ]
    })
    monkeypatch.setattr("research_experiments.workspace.hf.runs.HfApi", FakeApi)

    payload = push_runs_to_hub(runs_root, repo_id="owner/research-runs")

    assert payload["published_run_count"] == 1
    assert len(commit_calls) == 1
    operations = commit_calls[0]["operations"]
    delete_ops = [op for op in operations if isinstance(op, CommitOperationDelete)]
    add_ops = [op for op in operations if isinstance(op, CommitOperationAdd)]
    assert len(delete_ops) == 1
    assert delete_ops[0].path_in_repo == "single_agent/demo/count20/20260510T000000Z-model"
    assert delete_ops[0].is_folder is True
    assert any(op.path_in_repo == "runs_manifest.json" for op in add_ops)
    assert any(
        op.path_in_repo.startswith("single_agent/demo/count20/20260510T000000Z-model/")
        for op in add_ops
    )


def test_pull_runs_from_hub_skips_matching_local_runs_and_records_conflicts(monkeypatch, tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    matching_root = runs_root / "dmad" / "demo" / "count20" / "20260510T000000Z-model"
    _seed_standard_run(matching_root, passed=True)
    pack_run_artifacts(matching_root, runs_root=runs_root)
    matching_hash = compute_run_bundle_sha256(matching_root)

    conflict_root = runs_root / "single_agent" / "demo" / "count20" / "20260510T000001Z-model"
    _seed_standard_run(conflict_root, passed=True)
    pack_run_artifacts(conflict_root, runs_root=runs_root)
    conflict_hash = compute_run_bundle_sha256(conflict_root)

    remote_manifest = {
        "runs": [
            {
                "remote_prefix": "dmad/demo/count20/20260510T000000Z-model",
                "run_kind": "standard",
                "run_id": "20260510T000000Z-model",
                "bundle_sha256": matching_hash,
                "published_at": "2026-06-03T10:00:00+00:00",
            },
            {
                "remote_prefix": "single_agent/demo/count20/20260510T000001Z-model",
                "run_kind": "standard",
                "run_id": "20260510T000001Z-model",
                "bundle_sha256": "remote-conflict-hash",
                "published_at": "2026-06-03T10:00:00+00:00",
            },
        ]
    }
    download_calls: list[str] = []

    monkeypatch.setattr("research_experiments.workspace.hf.runs.download_repo_manifest", lambda **kwargs: remote_manifest)
    monkeypatch.setattr(
        "research_experiments.workspace.hf.runs.snapshot_download",
        lambda **kwargs: download_calls.append(kwargs["allow_patterns"][0]),
    )

    payload = pull_runs_from_hub(runs_root, repo_id="owner/research-runs")

    assert download_calls == []
    assert payload["skipped_run_count"] == 1
    assert payload["conflict_run_count"] == 1
    assert payload["conflict_runs"][0]["target_run_root"] == conflict_root.as_posix()
    assert payload["conflict_runs"][0]["local_bundle_sha256"] == conflict_hash
    assert payload["passed"] is False


def test_pull_runs_from_hub_fetches_missing_runs_by_recent_hours_and_prefix(monkeypatch, tmp_path: Path) -> None:
    remote_source_root = tmp_path / "remote-source"
    recent_root = remote_source_root / "dmad" / "demo" / "count20" / "20260510T000000Z-model"
    _seed_standard_run(recent_root, passed=True)
    pack_run_artifacts(recent_root, runs_root=remote_source_root)
    recent_hash = compute_run_bundle_sha256(recent_root)
    recent_stage = tmp_path / "remote-stage" / "dmad" / "demo" / "count20" / "20260510T000000Z-model"
    _stage_run_for_sync(recent_root, recent_stage)

    old_root = remote_source_root / "single_agent" / "demo" / "count20" / "20260510T000001Z-model"
    _seed_standard_run(old_root, passed=True)
    pack_run_artifacts(old_root, runs_root=remote_source_root)
    old_hash = compute_run_bundle_sha256(old_root)
    old_stage = tmp_path / "remote-stage" / "single_agent" / "demo" / "count20" / "20260510T000001Z-model"
    _stage_run_for_sync(old_root, old_stage)

    remote_manifest = {
        "runs": [
            {
                "remote_prefix": "dmad/demo/count20/20260510T000000Z-model",
                "run_kind": "standard",
                "run_id": "20260510T000000Z-model",
                "bundle_sha256": recent_hash,
                "published_at": "2026-06-03T10:30:00+00:00",
            },
            {
                "remote_prefix": "single_agent/demo/count20/20260510T000001Z-model",
                "run_kind": "standard",
                "run_id": "20260510T000001Z-model",
                "bundle_sha256": old_hash,
                "published_at": "2026-06-03T08:30:00+00:00",
            },
        ]
    }

    def _snapshot_download(repo_id, repo_type, allow_patterns, local_dir, token):
        del repo_id, repo_type, token
        remote_prefix = allow_patterns[0][:-3]
        source_dir = tmp_path / "remote-stage" / Path(remote_prefix)
        target_dir = Path(local_dir) / remote_prefix
        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
        return target_dir.as_posix()

    monkeypatch.setattr("research_experiments.workspace.hf.common.utcnow", lambda: datetime(2026, 6, 3, 11, 0, tzinfo=UTC))
    monkeypatch.setattr("research_experiments.workspace.hf.runs.download_repo_manifest", lambda **kwargs: remote_manifest)
    monkeypatch.setattr("research_experiments.workspace.hf.runs.snapshot_download", _snapshot_download)

    payload = pull_runs_from_hub(
        tmp_path / "pulled-runs",
        repo_id="owner/research-runs",
        prefixes=["dmad"],
        recent_hours=1,
    )

    fetched_root = tmp_path / "pulled-runs" / "dmad" / "demo" / "count20" / "20260510T000000Z-model"
    assert payload["fetched_run_count"] == 1
    assert payload["fetched_runs"][0]["remote_prefix"] == "dmad/demo/count20/20260510T000000Z-model"
    assert fetched_root.exists()
    assert (fetched_root / "hf_run.json").exists()


def _seed_standard_run(root: Path, *, passed: bool) -> None:
    write_json(root / "manifest.json", {"run_id": root.name})
    (root / "report.md").write_text("# report\n", encoding="utf-8")
    write_json(root / "run_validation.json", {"passed": passed})
    write_json(root / "metrics.json", {"summary": []})
    (root / "raw_responses.jsonl").write_text("{}\n", encoding="utf-8")


def _seed_matrix_run(root: Path, *, completed: int, expected: int, status: str) -> None:
    write_json(
        root / "state.json",
        {
            "counts": {"completed": completed, "semantic_unique_targets": expected},
            "entries": [{"status": status}],
        },
    )
    write_json(root / "paper_package.json", {"ok": True})
