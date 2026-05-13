from __future__ import annotations

import json
from pathlib import Path

from research_experiments.workspace.hf_sync import discover_publishable_runs, pull_workspace_from_hub, push_workspace_to_hub


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_discover_publishable_runs_includes_standard_and_matrix(tmp_path: Path) -> None:
    standard_root = tmp_path / "single_agent" / "demo" / "count20" / "20260510T000000Z-model"
    _write_json(standard_root / "manifest.json", {"run_id": "20260510T000000Z-model"})
    (standard_root / "report.md").write_text("# report\n", encoding="utf-8")
    _write_json(standard_root / "run_validation.json", {"passed": True})

    invalid_root = tmp_path / "budget_comm" / "demo" / "count20" / "20260510T000001Z-model"
    _write_json(invalid_root / "manifest.json", {"run_id": "20260510T000001Z-model"})
    (invalid_root / "report.md").write_text("# report\n", encoding="utf-8")
    _write_json(invalid_root / "run_validation.json", {"passed": False})

    matrix_root = tmp_path / "faithful_matrix" / "20260510T000100Z-count20-model"
    _write_json(
        matrix_root / "state.json",
        {
            "counts": {"completed": 2, "semantic_unique_targets": 2},
            "entries": [{"status": "completed"}, {"status": "completed"}],
        },
    )
    _write_json(matrix_root / "paper_package.json", {"ok": True})

    rows = discover_publishable_runs(tmp_path)
    by_root = {row["run_root"]: row for row in rows}

    assert by_root[standard_root.as_posix()]["publishable"] is True
    assert by_root[invalid_root.as_posix()]["publishable"] is False
    assert by_root[matrix_root.as_posix()]["run_kind"] == "matrix"
    assert by_root[matrix_root.as_posix()]["publishable"] is True


def test_push_workspace_to_hub_batches_runs_and_cache(monkeypatch, tmp_path: Path) -> None:
    standard_root = tmp_path / "runs" / "single_agent" / "demo" / "count20" / "20260510T000000Z-model"
    _write_json(standard_root / "manifest.json", {"run_id": "20260510T000000Z-model"})
    (standard_root / "report.md").write_text("# report\n", encoding="utf-8")
    _write_json(standard_root / "run_validation.json", {"passed": True})

    invalid_root = tmp_path / "runs" / "budget_comm" / "demo" / "count20" / "20260510T000001Z-model"
    _write_json(invalid_root / "manifest.json", {"run_id": "20260510T000001Z-model"})
    (invalid_root / "report.md").write_text("# report\n", encoding="utf-8")
    _write_json(invalid_root / "run_validation.json", {"passed": False})

    matrix_root = tmp_path / "runs" / "faithful_matrix" / "20260510T000100Z-count20-model"
    _write_json(
        matrix_root / "state.json",
        {
            "counts": {"completed": 1, "semantic_unique_targets": 1},
            "entries": [{"status": "completed"}],
        },
    )
    _write_json(matrix_root / "paper_package.json", {"ok": True})
    _write_json(matrix_root / "hf_publish.json", {"published": True, "remote_repo": "owner/research-runs"})

    published_roots: list[str] = []

    monkeypatch.setattr(
        "research_experiments.workspace.hf_sync.publish_run_to_hub",
        lambda run_dir, repo_id, token, runs_root, create_repo: published_roots.append(str(run_dir))
        or {
            "run_dir": str(run_dir),
            "remote_repo": repo_id,
            "remote_prefix": Path(run_dir).name,
            "published": True,
        },
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


def test_pull_workspace_from_hub_batches_runs_and_cache(monkeypatch, tmp_path: Path) -> None:
    downloaded_prefixes: list[str] = []

    monkeypatch.setattr(
        "research_experiments.workspace.hf_sync.list_remote_run_prefixes",
        lambda api, repo_id: [
            "single_agent/demo/count20/20260510T000000Z-model",
            "faithful_matrix/20260510T000100Z-count20-model",
        ],
    )
    monkeypatch.setattr(
        "research_experiments.workspace.hf_sync.snapshot_download",
        lambda repo_id, repo_type, allow_patterns, local_dir, token: downloaded_prefixes.append(allow_patterns[0][:-3])
        or (Path(local_dir) / allow_patterns[0][:-3]).mkdir(parents=True, exist_ok=True),
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


def test_push_workspace_to_hub_can_target_selected_run_dirs(monkeypatch, tmp_path: Path) -> None:
    standard_root = tmp_path / "runs" / "single_agent" / "demo" / "count20" / "20260510T000000Z-model"
    _write_json(standard_root / "manifest.json", {"run_id": "20260510T000000Z-model"})
    (standard_root / "report.md").write_text("# report\n", encoding="utf-8")
    _write_json(standard_root / "run_validation.json", {"passed": True})

    another_root = tmp_path / "runs" / "budget_comm" / "demo" / "count20" / "20260510T000001Z-model"
    _write_json(another_root / "manifest.json", {"run_id": "20260510T000001Z-model"})
    (another_root / "report.md").write_text("# report\n", encoding="utf-8")
    _write_json(another_root / "run_validation.json", {"passed": True})

    published_roots: list[str] = []
    monkeypatch.setattr(
        "research_experiments.workspace.hf_sync.publish_run_to_hub",
        lambda run_dir, repo_id, token, runs_root, create_repo: published_roots.append(str(run_dir))
        or {
            "run_dir": str(run_dir),
            "remote_repo": repo_id,
            "remote_prefix": Path(run_dir).name,
            "published": True,
        },
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
    monkeypatch.setattr(
        "research_experiments.workspace.hf_sync.list_remote_run_prefixes",
        lambda api, repo_id: [
            "single_agent/demo/count20/20260510T000000Z-model",
            "budget_comm/demo/count20/20260510T000001Z-model",
        ],
    )
    monkeypatch.setattr(
        "research_experiments.workspace.hf_sync.snapshot_download",
        lambda repo_id, repo_type, allow_patterns, local_dir, token: downloaded_prefixes.append(allow_patterns[0][:-3])
        or (Path(local_dir) / allow_patterns[0][:-3]).mkdir(parents=True, exist_ok=True),
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

