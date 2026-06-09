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
FAMILIES_SRC = ROOT / "src" / "research_experiments" / "families"
LEGACY_REPORT_COMMANDS = (
    "report-run",
    "report-trigger",
    "report-debate-vs-vote",
)
FAMILY_READMES = sorted(FAMILIES_SRC.glob("*/README.md"))
MARKDOWN_DOCS = [
    ROOT / "README.md",
    ROOT / "docs" / "huggingface_archive_workflow.md",
    ROOT / "docs" / "run_report_pipeline.md",
    *FAMILY_READMES,
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
    for path in FAMILY_READMES:
        text = path.read_text(encoding="utf-8")
        assert "render-report --run-dir" in text, f"{path} is missing unified render-report example"
        assert "research_cli experiment --family " in text, f"{path} 未使用新的 experiment CLI 入口"


def test_family_readme_run_dir_examples_match_experiment_configs() -> None:
    for path in FAMILY_READMES:
        text = path.read_text(encoding="utf-8")
        family_name = path.parent.name
        experiment_names = _experiment_names_in_doc(text)
        for run_family, run_experiment in _run_dir_examples_in_doc(text):
            if run_family != family_name or run_experiment == "<experiment>":
                continue
            assert run_experiment in experiment_names, (
                f"{path} 的 run-dir 示例实验名 {run_experiment!r} 未出现在 --experiment 配置示例中"
            )


def test_active_project_docs_do_not_reference_removed_families() -> None:
    removed_markers = ("`sparc`", "sparc_v1", "families/shared", "core/families", "research_experiments/tools", "run/io.py")
    for path in ACTIVE_PROJECT_DOCS:
        text = path.read_text(encoding="utf-8")
        for marker in removed_markers:
            assert marker not in text, f"{path} still references removed marker {marker}"


def test_phase_runner_docs_match_default_phase_sequence() -> None:
    """约束阶段脚本文档与实际默认序列一致，避免把轻量默认误写成完整确认序列。"""
    ps1_text = (ROOT / "run_all_phases.ps1").read_text(encoding="utf-8")
    sh_text = (ROOT / "run_all_phases.sh").read_text(encoding="utf-8")
    assert '[string[]]$Phases = @("count20", "count100")' in ps1_text
    assert "phases=(count20 count100)" in sh_text

    expected_doc_text = "默认运行 `count20 -> count100`"
    docs = [
        ROOT / "README.md",
        ROOT / "docs" / "run_report_pipeline.md",
    ]
    for path in docs:
        text = path.read_text(encoding="utf-8")
        assert expected_doc_text in text, f"{path} 未同步默认阶段序列"
        assert "四阶段结束后" not in text, f"{path} 仍保留过期的四阶段默认描述"


def _subcommands(parser: argparse.ArgumentParser) -> set[str]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    return set()


def _experiment_names_in_doc(text: str) -> set[str]:
    pattern = re.compile(r"--experiment configs/families/[A-Za-z0-9_-]+/experiments/([A-Za-z0-9_./-]+)\.toml")
    return {Path(match).stem for match in pattern.findall(text)}


def _run_dir_examples_in_doc(text: str) -> list[tuple[str, str]]:
    pattern = re.compile(r"local/runs/([^/\s]+)/([^/\s]+)/")
    return pattern.findall(text)
