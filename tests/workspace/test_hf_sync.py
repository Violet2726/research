from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from testsupport.filesystem import write_json

from research_experiments.workspace.hf_sync import (
    discover_publishable_runs,
    list_remote_runs,
    pull_workspace_from_hub,
    push_workspace_to_hub,
)


def test_discover_publishable_runs_includes_standard_and_matrix(tmp_path: Path) -> None:
    standard_root = tmp_path / "single_agent" / "demo" / "count20" / "20260510T000000Z-model"
    write_json(standard_root / "manifest.json", {"run_id": "20260510T000000Z-model"})
    (standard_root / "report.md").write_text("# report\n", encoding="utf-8")
    write_json(standard_root / "run_validation.json", {"passed": True})

    invalid_root = tmp_path / "budget_comm" / "demo" / "count20" / "20260510T000001Z-model"
    write_json(invalid_root / "manifest.json", {"run_id": "20260510T000001Z-model"})
    (invalid_root / "report.md").write_text("# report\n", encoding="utf-8")
    write_json(invalid_root / "run_validation.json", {"passed": False})

    matrix_root = tmp_path / "faithful_matrix" / "20260510T000100Z-count20-model"
    write_json(
        matrix_root / "state.json",
        {
            "counts": {"completed": 2, "semantic_unique_targets": 2},
            "entries": [{"status": "completed"}, {"status": "completed"}],
        },
    )
    write_json(matrix_root / "paper_package.json", {"ok": True})

    rows = discover_publishable_runs(tmp_path)
    by_root = {row["run_root"]: row for row in rows}

    assert by_root[standard_root.as_posix()]["publishable"] is True
    assert by_root[invalid_root.as_posix()]["publishable"] is False
    assert by_root[matrix_root.as_posix()]["run_kind"] == "matrix"
    assert by_root[matrix_root.as_posix()]["publishable"] is True


def test_discover_publishable_runs_accepts_reproduction_matrix(tmp_path: Path) -> None:
    matrix_root = tmp_path / "reproduction_matrix" / "20260516T000100Z-count20-model"
    write_json(
        matrix_root / "state.json",
        {
            "matrix_id": "reproduction",
            "counts": {"completed": 2, "semantic_unique_targets": 2},
            "entries": [{"status": "completed"}, {"status": "completed"}],
        },
    )
    write_json(matrix_root / "reproduction_package.json", {"ok": True})

    rows = discover_publishable_runs(tmp_path)
    by_root = {row["run_root"]: row for row in rows}

    assert by_root[matrix_root.as_posix()]["run_kind"] == "matrix"
    assert by_root[matrix_root.as_posix()]["publishable"] is True


def test_push_workspace_to_hub_batches_runs_and_cache(monkeypatch, tmp_path: Path) -> None:
    standard_root = tmp_path / "runs" / "single_agent" / "demo" / "count20" / "20260510T000000Z-model"
    write_json(standard_root / "manifest.json", {"run_id": "20260510T000000Z-model"})
    (standard_root / "report.md").write_text("# report\n", encoding="utf-8")
    write_json(standard_root / "run_validation.json", {"passed": True})

    invalid_root = tmp_path / "runs" / "budget_comm" / "demo" / "count20" / "20260510T000001Z-model"
    write_json(invalid_root / "manifest.json", {"run_id": "20260510T000001Z-model"})
    (invalid_root / "report.md").write_text("# report\n", encoding="utf-8")
    write_json(invalid_root / "run_validation.json", {"passed": False})

    matrix_root = tmp_path / "runs" / "faithful_matrix" / "20260510T000100Z-count20-model"
    write_json(
        matrix_root / "state.json",
        {
            "counts": {"completed": 1, "semantic_unique_targets": 1},
            "entries": [{"status": "completed"}],
        },
    )
    write_json(matrix_root / "paper_package.json", {"ok": True})
    write_json(matrix_root / "hf_publish.json", {"published": True, "remote_repo": "owner/research-runs"})

    published_roots: list[str] = []

    def _publish_run(run_dir, repo_id, token, runs_root, create_repo):
        del token, runs_root, create_repo
        published_roots.append(str(run_dir))
        return {
            "run_dir": str(run_dir),
            "remote_repo": repo_id,
            "remote_prefix": Path(run_dir).name,
            "published": True,
        }

    monkeypatch.setattr(
        "research_experiments.workspace.hf_sync.publish_run_to_hub",
        _publish_run,
    )
    monkeypatch.setattr(
        "research_experiments.workspace.hf_sync.push_latest_cache_snapshot",
        lambda cache_root, repo_id, token, create_repo, private, shard_filters=None: {
            "cache_root": str(cache_root),
            "remote_repo": repo_id,
            "published": True,
            "private_repo": private,
            "shard_filters": shard_filters or [],
        },
    )

    payload = push_workspace_to_hub(
        runs_root=tmp_path / "runs",
        cache_root=tmp_path / "cache",
        runs_repo_id="owner/research-runs",
        cache_repo_id="owner/research-cache",
    )

    assert published_roots == [str(standard_root)]
    assert payload["candidate_run_count"] == 3
    assert payload["published_run_count"] == 1
    assert payload["skipped_run_count"] == 2
    assert payload["cache_pushed"] is True
    assert (standard_root / "hf_publish.json").exists()


def test_list_remote_runs_can_include_publish_timestamps() -> None:
    published_at = datetime(2026, 6, 3, 10, 0, tzinfo=UTC)

    class _FakeApi:
        def list_repo_files(self, repo_id, repo_type):
            assert repo_id == "owner/research-runs"
            assert repo_type == "dataset"
            return [
                "single_agent/demo/count20/20260510T000000Z-model/archive_manifest.json",
                "single_agent/demo/count20/20260510T000000Z-model/report.md",
                "faithful_matrix/20260510T000100Z-count20-model/archive_manifest.json",
            ]

        def get_paths_info(self, repo_id, paths, repo_type, expand):
            assert repo_id == "owner/research-runs"
            assert repo_type == "dataset"
            assert expand is True
            return [
                SimpleNamespace(
                    path=path,
                    last_commit=SimpleNamespace(date=published_at),
                )
                for path in paths
            ]

    rows = list_remote_runs(_FakeApi(), repo_id="owner/research-runs", include_commit_timestamps=True)

    assert rows == (
        {
            "remote_prefix": "faithful_matrix/20260510T000100Z-count20-model",
            "published_at": "2026-06-03T10:00:00+00:00",
        },
        {
            "remote_prefix": "single_agent/demo/count20/20260510T000000Z-model",
            "published_at": "2026-06-03T10:00:00+00:00",
        },
    )


def test_pull_workspace_from_hub_batches_runs_and_cache(monkeypatch, tmp_path: Path) -> None:
    downloaded_prefixes: list[str] = []

    def _snapshot_download(repo_id, repo_type, allow_patterns, local_dir, token):
        del repo_id, repo_type, token
        prefix = allow_patterns[0][:-3]
        downloaded_prefixes.append(prefix)
        return (Path(local_dir) / prefix).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "research_experiments.workspace.hf_sync.list_remote_runs",
        lambda api, repo_id, include_commit_timestamps=False: [
            {
                "remote_prefix": "single_agent/demo/count20/20260510T000000Z-model",
                "published_at": None,
            },
            {
                "remote_prefix": "faithful_matrix/20260510T000100Z-count20-model",
                "published_at": None,
            },
        ],
    )
    monkeypatch.setattr(
        "research_experiments.workspace.hf_sync.snapshot_download",
        _snapshot_download,
    )
    monkeypatch.setattr(
        "research_experiments.workspace.hf_sync.extract_run_archives",
        lambda run_dir: ("a.jsonl", "b.jsonl"),
    )
    monkeypatch.setattr(
        "research_experiments.workspace.hf_sync.pull_latest_cache_snapshot",
        lambda target_root, repo_id, token, shard_filters=None: {
            "target_root": str(target_root),
            "remote_repo": repo_id,
            "restored_shard_count": 6,
            "shard_filters": shard_filters or [],
        },
    )

    payload = pull_workspace_from_hub(
        runs_root=tmp_path / "runs",
        cache_root=tmp_path / "cache",
        runs_repo_id="owner/research-runs",
        cache_repo_id="owner/research-cache",
    )

    assert downloaded_prefixes == [
        "single_agent/demo/count20/20260510T000000Z-model",
        "faithful_matrix/20260510T000100Z-count20-model",
    ]
    assert payload["fetched_run_count"] == 2
    assert payload["cache_pulled"] is True


def test_pull_workspace_from_hub_can_filter_runs_published_within_hours(monkeypatch, tmp_path: Path) -> None:
    downloaded_prefixes: list[str] = []

    def _snapshot_download(repo_id, repo_type, allow_patterns, local_dir, token):
        del repo_id, repo_type, token
        prefix = allow_patterns[0][:-3]
        downloaded_prefixes.append(prefix)
        return (Path(local_dir) / prefix).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "research_experiments.workspace.hf_sync._utcnow",
        lambda: datetime(2026, 6, 3, 11, 0, tzinfo=UTC),
    )
    monkeypatch.setattr(
        "research_experiments.workspace.hf_sync.list_remote_runs",
        lambda api, repo_id, include_commit_timestamps=False: [
            {
                "remote_prefix": "single_agent/demo/count20/20260510T000000Z-model",
                "published_at": "2026-06-03T10:30:00+00:00",
            },
            {
                "remote_prefix": "budget_comm/demo/count20/20260510T000001Z-model",
                "published_at": "2026-06-03T09:30:00+00:00",
            },
        ],
    )
    monkeypatch.setattr(
        "research_experiments.workspace.hf_sync.snapshot_download",
        _snapshot_download,
    )
    monkeypatch.setattr(
        "research_experiments.workspace.hf_sync.extract_run_archives",
        lambda run_dir: (),
    )

    payload = pull_workspace_from_hub(
        runs_root=tmp_path / "runs",
        cache_root=tmp_path / "cache",
        runs_repo_id="owner/research-runs",
        fetch_runs=True,
        pull_cache=False,
        published_within_hours=1,
    )

    assert downloaded_prefixes == ["single_agent/demo/count20/20260510T000000Z-model"]
    assert payload["published_within_hours"] == 1
    assert payload["published_after"] == "2026-06-03T10:00:00+00:00"
    assert payload["fetched_run_count"] == 1
    assert payload["fetched_runs"][0]["published_at"] == "2026-06-03T10:30:00+00:00"


def test_push_workspace_to_hub_can_target_selected_run_dirs(monkeypatch, tmp_path: Path) -> None:
    standard_root = tmp_path / "runs" / "single_agent" / "demo" / "count20" / "20260510T000000Z-model"
    write_json(standard_root / "manifest.json", {"run_id": "20260510T000000Z-model"})
    (standard_root / "report.md").write_text("# report\n", encoding="utf-8")
    write_json(standard_root / "run_validation.json", {"passed": True})

    another_root = tmp_path / "runs" / "budget_comm" / "demo" / "count20" / "20260510T000001Z-model"
    write_json(another_root / "manifest.json", {"run_id": "20260510T000001Z-model"})
    (another_root / "report.md").write_text("# report\n", encoding="utf-8")
    write_json(another_root / "run_validation.json", {"passed": True})

    published_roots: list[str] = []

    def _publish_run(run_dir, repo_id, token, runs_root, create_repo):
        del token, runs_root, create_repo
        published_roots.append(str(run_dir))
        return {
            "run_dir": str(run_dir),
            "remote_repo": repo_id,
            "remote_prefix": Path(run_dir).name,
            "published": True,
        }

    monkeypatch.setattr(
        "research_experiments.workspace.hf_sync.publish_run_to_hub",
        _publish_run,
    )
    monkeypatch.setattr(
        "research_experiments.workspace.hf_sync.push_latest_cache_snapshot",
        lambda cache_root, repo_id, token, create_repo, private, shard_filters=None: {
            "cache_root": str(cache_root),
            "remote_repo": repo_id,
            "published": True,
        },
    )

    payload = push_workspace_to_hub(
        runs_root=tmp_path / "runs",
        cache_root=tmp_path / "cache",
        runs_repo_id="owner/research-runs",
        cache_repo_id="owner/research-cache",
        selected_run_dirs=[standard_root.as_posix()],
    )

    assert published_roots == [str(standard_root)]
    assert payload["candidate_run_count"] == 1
    assert payload["published_run_count"] == 1


def test_pull_workspace_from_hub_can_target_selected_run_ids_and_prefixes(monkeypatch, tmp_path: Path) -> None:
    downloaded_prefixes: list[str] = []

    def _snapshot_download(repo_id, repo_type, allow_patterns, local_dir, token):
        del repo_id, repo_type, token
        prefix = allow_patterns[0][:-3]
        downloaded_prefixes.append(prefix)
        return (Path(local_dir) / prefix).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        "research_experiments.workspace.hf_sync.list_remote_runs",
        lambda api, repo_id, include_commit_timestamps=False: [
            {
                "remote_prefix": "single_agent/demo/count20/20260510T000000Z-model",
                "published_at": None,
            },
            {
                "remote_prefix": "budget_comm/demo/count20/20260510T000001Z-model",
                "published_at": None,
            },
        ],
    )
    monkeypatch.setattr(
        "research_experiments.workspace.hf_sync.snapshot_download",
        _snapshot_download,
    )
    monkeypatch.setattr(
        "research_experiments.workspace.hf_sync.extract_run_archives",
        lambda run_dir: (),
    )
    monkeypatch.setattr(
        "research_experiments.workspace.hf_sync.pull_latest_cache_snapshot",
        lambda target_root, repo_id, token, shard_filters=None: {
            "target_root": str(target_root),
            "remote_repo": repo_id,
            "restored_shard_count": 0,
        },
    )

    payload = pull_workspace_from_hub(
        runs_root=tmp_path / "runs",
        cache_root=tmp_path / "cache",
        runs_repo_id="owner/research-runs",
        cache_repo_id="owner/research-cache",
        selected_run_ids=["20260510T000000Z-model"],
        selected_run_prefixes=["budget_comm/demo/count20/20260510T000001Z-model"],
    )

    assert downloaded_prefixes == [
        "single_agent/demo/count20/20260510T000000Z-model",
        "budget_comm/demo/count20/20260510T000001Z-model",
    ]
    assert payload["fetched_run_count"] == 2

