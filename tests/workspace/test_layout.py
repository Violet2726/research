
"""覆盖工作区默认路径与环境变量覆盖行为。"""

from __future__ import annotations

import pytest

from research_experiments.workspace.layout import (
    auto_publish_runs_enabled,
    auto_push_cache_snapshot_enabled,
    default_cache_hf_repo,
    default_cache_root,
    default_datasets_root,
    default_files_root,
    default_reports_root,
    default_runs_hf_repo,
    default_runs_root,
    workspace_defaults,
)


def test_workspace_defaults_follow_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_RUNS_ROOT", "experiment-runs")
    monkeypatch.setenv("RESEARCH_REPORTS_ROOT", "published-reports")
    monkeypatch.setenv("RESEARCH_CACHE_ROOT", "tmp/cache")
    monkeypatch.setenv("RESEARCH_DATASETS_ROOT", "tmp/datasets")
    monkeypatch.setenv("RESEARCH_FILES_ROOT", "notes")
    monkeypatch.setenv("RESEARCH_RUNS_HF_REPO", "owner/research-runs")
    monkeypatch.setenv("RESEARCH_CACHE_HF_REPO", "owner/research-cache")
    monkeypatch.setenv("RESEARCH_AUTO_PUBLISH_RUNS", "1")
    monkeypatch.setenv("RESEARCH_AUTO_PUSH_CACHE_SNAPSHOT", "true")

    assert default_runs_root("selective_comm") == "experiment-runs/selective_comm"
    assert default_reports_root("selective_comm") == "published-reports/selective_comm"
    assert default_cache_root() == "tmp/cache"
    assert default_datasets_root() == "tmp/datasets"
    assert default_files_root() == "notes"
    assert default_runs_hf_repo() == "owner/research-runs"
    assert default_cache_hf_repo() == "owner/research-cache"
    assert auto_publish_runs_enabled() is True
    assert auto_push_cache_snapshot_enabled() is True

    payload = workspace_defaults("selective_comm")
    assert payload["runs_root"] == "experiment-runs"
    assert payload["reports_root"] == "published-reports"
    assert payload["family_runs_root"] == "experiment-runs/selective_comm"
    assert payload["family_reports_root"] == "published-reports/selective_comm"
    assert payload["family_cache_root"] == "tmp/cache"
    assert payload["datasets_root"] == "tmp/datasets"
    assert payload["runs_hf_repo"] == "owner/research-runs"
    assert payload["cache_hf_repo"] == "owner/research-cache"
    assert payload["auto_publish_runs"] is True
    assert payload["auto_push_cache_snapshot"] is True

def test_workspace_defaults_use_local_roots_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [
        "RESEARCH_RUNS_ROOT",
        "RESEARCH_REPORTS_ROOT",
        "RESEARCH_CACHE_ROOT",
        "RESEARCH_DATASETS_ROOT",
        "RESEARCH_FILES_ROOT",
        "RESEARCH_RUNS_HF_REPO",
        "RESEARCH_CACHE_HF_REPO",
        "RESEARCH_AUTO_PUBLISH_RUNS",
        "RESEARCH_AUTO_PUSH_CACHE_SNAPSHOT",
    ]:
        monkeypatch.delenv(name, raising=False)

    assert default_runs_root("single_agent") == "local/runs/single_agent"
    assert default_reports_root("single_agent") == "local/reports/single_agent"
    assert default_cache_root() == "local/cache"
    assert default_datasets_root() == "local/datasets"
    assert default_files_root() == "files"
    assert default_runs_hf_repo() is None
    assert default_cache_hf_repo() is None
    assert auto_publish_runs_enabled() is False
    assert auto_push_cache_snapshot_enabled() is False

