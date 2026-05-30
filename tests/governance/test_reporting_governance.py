"""报告接口、文档命令与目录治理约束测试。"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from research_experiments.cli import build_parser as build_root_parser
from research_experiments.cli.family import build_family_parser
from research_experiments.families.registry import (
    get_family_registration,
    registered_family_names,
)

ROOT = Path(__file__).resolve().parents[2]
LEGACY_REPORT_COMMANDS = (
    "report-run",
    "report-trigger",
    "report-cue",
    "report-debate-vs-vote",
)
MARKDOWN_DOCS = [
    ROOT / "README.md",
    ROOT / "docs" / "huggingface_archive_workflow.md",
    ROOT / "docs" / "run_report_pipeline.md",
    ROOT / "src" / "research_experiments" / "families" / "budget_comm" / "README.md",
    ROOT / "src" / "research_experiments" / "families" / "colmad" / "README.md",
    ROOT / "src" / "research_experiments" / "families" / "comm_necessary" / "README.md",
    ROOT / "src" / "research_experiments" / "families" / "consensagent" / "README.md",
    ROOT / "src" / "research_experiments" / "families" / "cue" / "README.md",
    ROOT / "src" / "research_experiments" / "families" / "dmad" / "README.md",
    ROOT / "src" / "research_experiments" / "families" / "econ" / "README.md",
    ROOT / "src" / "research_experiments" / "families" / "free_mad_lite" / "README.md",
    ROOT / "src" / "research_experiments" / "families" / "imad" / "README.md",
    ROOT / "src" / "research_experiments" / "families" / "macnet" / "README.md",
    ROOT / "src" / "research_experiments" / "families" / "madjudge" / "README.md",
    ROOT / "src" / "research_experiments" / "families" / "multi_agent" / "README.md",
    ROOT / "src" / "research_experiments" / "families" / "selective_comm" / "README.md",
    ROOT / "src" / "research_experiments" / "families" / "sid_lite" / "README.md",
    ROOT / "src" / "research_experiments" / "families" / "single_agent" / "README.md",
]
ACTIVE_PROJECT_DOCS = [
    ROOT / "README.md",
    ROOT / "src" / "README.md",
    ROOT / "docs" / "README.md",
    ROOT / "docs" / "project_structure.md",
]


def test_all_family_clis_expose_render_report() -> None:
    root_parser = build_root_parser()
    assert _subcommands(root_parser) == {"experiment", "matrix", "tools"}

    parsers = [build_family_parser(get_family_registration(name)) for name in registered_family_names()]
    for parser in parsers:
        assert "render-report" in _subcommands(parser)


def test_markdown_docs_do_not_reference_legacy_report_commands() -> None:
    for path in MARKDOWN_DOCS:
        text = path.read_text(encoding="utf-8")
        for legacy_command in LEGACY_REPORT_COMMANDS:
            assert legacy_command not in text, f"{path} still references legacy command {legacy_command}"


def test_markdown_docs_reference_existing_config_paths() -> None:
    pattern = re.compile(r"configs/[A-Za-z0-9_./-]+\.toml")
    for path in MARKDOWN_DOCS:
        text = path.read_text(encoding="utf-8")
        for match in pattern.findall(text):
            assert (ROOT / match).exists(), f"{path} references missing config {match}"


def test_family_readmes_use_render_report_example() -> None:
    family_readmes = [path for path in MARKDOWN_DOCS if path.parent.name not in {"docs", "research"} and path.name == "README.md"]
    for path in family_readmes:
        text = path.read_text(encoding="utf-8")
        assert "render-report --run-dir" in text, f"{path} is missing unified render-report example"
        assert "research_cli experiment --family " in text, f"{path} 未使用新的 experiment CLI 入口"


def test_active_project_docs_do_not_reference_removed_families() -> None:
    removed_markers = ("`sparc`", "sparc_v1", "families/shared", "core/families", "research_experiments/tools", "run/io.py")
    for path in ACTIVE_PROJECT_DOCS:
        text = path.read_text(encoding="utf-8")
        for marker in removed_markers:
            assert marker not in text, f"{path} still references removed marker {marker}"


def _subcommands(parser: argparse.ArgumentParser) -> set[str]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    return set()
