"""报告接口、文档命令与目录治理约束测试。"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from budget_comm.cli import build_parser as build_budget_parser
from comm_necessary.cli import build_parser as build_comm_necessary_parser
from cue.cli import build_parser as build_cue_parser
from free_mad_lite.cli import build_parser as build_free_mad_parser
from multi_agent.cli import build_parser as build_multi_agent_parser
from selective_comm.cli import build_parser as build_selective_parser
from sid_lite.cli import build_parser as build_sid_parser
from single_agent.cli import build_parser as build_single_agent_parser
from sparc.cli import build_parser as build_sparc_parser


ROOT = Path(__file__).resolve().parents[2]
LEGACY_REPORT_COMMANDS = (
    "report-run",
    "report-trigger",
    "report-cue",
    "report-debate-vs-vote",
)
MARKDOWN_DOCS = [
    ROOT / "README.md",
    ROOT / "docs" / "cue_framework_guide.md",
    ROOT / "src" / "budget_comm" / "README.md",
    ROOT / "src" / "comm_necessary" / "README.md",
    ROOT / "src" / "cue" / "README.md",
    ROOT / "src" / "free_mad_lite" / "README.md",
    ROOT / "src" / "multi_agent" / "README.md",
    ROOT / "src" / "selective_comm" / "README.md",
    ROOT / "src" / "sid_lite" / "README.md",
    ROOT / "src" / "single_agent" / "README.md",
    ROOT / "src" / "sparc" / "README.md",
]


def test_all_family_clis_expose_render_report() -> None:
    parsers = [
        build_single_agent_parser(),
        build_multi_agent_parser(),
        build_selective_parser(),
        build_sparc_parser(),
        build_budget_parser(),
        build_sid_parser(),
        build_free_mad_parser(),
        build_comm_necessary_parser(),
        build_cue_parser(),
    ]
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


def _subcommands(parser: argparse.ArgumentParser) -> set[str]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return set(action.choices)
    return set()
